"""Corpus-grounded candidate-family generation.

This is a source adapter around the existing Referee. It preregisters the complete emitted
population and runs discovery with the production holdout physically read-only.
"""
from __future__ import annotations

import ast
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fcntl

from . import config, llm
from .brain import Claim
from .corpus import build_files
from .dream import (_atomic_json, _read_jsonl, _snapshot_hash, _validate_run_id,
                    _write_immutable, capability_manifest, record_candidates,
                    register_search)
from .pipeline import formulaic_signal

_SYSTEM = """You synthesize candidate quantitative-research families from a leveled corpus.
You propose; you never certify. Ground every component in supplied node ids and boundaries.
Return the complete requested population as strict JSON. Never use or request confirmation data.
Each candidate must keep the prose fields and also include a structured, buildable spec object:
- signal: a formulaic_signal DSL formula using ONLY the operators and capability series names below.
  Do not invent operators and do not use generic names such as close/open/price unless those exact
  names appear in the supplied capability series list. Window arguments are positive integer literals.
- trade_series: the exact capability series name whose returns are traded.
- position_map: either "sign" or "zscore_clip".
- params: named parameter values.
- param_grid: a prior parameter grid with reasonable ranges for every tunable param, not a single
  tuned point, e.g. {"lookback":[10,20,60]}.
- conditioning: an explicit regime/condition rule, or null.
- entry_exit: explicit entry and exit rules.
- horizon: holding horizon."""

_USER = """Create exactly {n} distinct candidate strategy families from these family principles
and cross-family mechanisms:
{corpus}

Capabilities:
{capabilities}

Allowed capability series names:
{capability_series}

Use ONLY the allowed capability series names as identifiers in spec.signal and spec.trade_series.
Allowed formulaic_signal DSL operators:
- returns(series,n): log return over n bars
- price(series): identity for a price series
- funding(series): identity for a funding or carry series
- rolling_sum(series,n), rolling_mean(series,n), rolling_std(series,n)
- sign(x)
- lag(series,n), delta(series,n), zscore(series,n)
- binary operators + - * / and unary -

Worked DSL example:
{{
  "signal": "zscore(funding_btc, 20) - rolling_mean(funding_btc, 60)",
  "trade_series": "btc_perp_close_daily_5y",
  "position_map": "zscore_clip",
  "params": {{"lookback": 20}},
  "param_grid": {{"lookback": [10, 20, 60]}},
  "conditioning": null,
  "entry_exit": "Hold one-bar-lagged clipped signal exposure to trade_series returns.",
  "horizon": "daily"
}}

Return {{"candidates":[{{"statement":"falsifiable claim","mechanism":"tentative mechanism",
"scope":"market/universe","horizon":"measurable horizon","strategy_class":"class",
"candidate_class":"testable_now|testable_with_new_module|requires_data_acquisition|conceptual_only",
"required_series":[],"inspired_by":["node ids"],"boundaries":[],"falsifier":"specific rejection",
"spec":{{"signal":"formulaic_signal DSL over allowed series","trade_series":"allowed_series_name",
"position_map":"sign|zscore_clip",
"params":{{"param_name":20}},"param_grid":{{"param_name":[10,20,60]}},
"conditioning":null,"entry_exit":"explicit entry and exit rules","horizon":"holding horizon"}}}}]}}."""

_REQUIRED_SPEC_FIELDS = (
    "signal", "trade_series", "position_map", "params", "param_grid", "entry_exit", "horizon"
)
_LEGACY_REQUIRED_SPEC_FIELDS = ("signal", "params", "param_grid", "entry_exit", "horizon")
_LEGACY_FUNC_ALIASES = {
    "ma": "rolling_mean",
    "sma": "rolling_mean",
    "sum": "rolling_sum",
    "std": "rolling_std",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inputs(graph: dict) -> dict:
    nodes = [n for n in graph.get("nodes", [])
             if n.get("level") in {"family_principle", "cross_family_mechanism"}]
    return {"nodes": nodes, "snapshot_hash": _snapshot_hash(nodes)}


def _capability_series_names(capabilities: object) -> set[str]:
    if not isinstance(capabilities, dict):
        return set()
    raw = capabilities.get("series")
    if isinstance(raw, dict):
        return {str(k) for k in raw if str(k).strip()}
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw if str(x).strip()}
    return set()


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _is_legacy_spec(spec: dict) -> bool:
    if str(spec.get("trade_series") or "").strip():
        return False
    series = spec.get("series")
    return isinstance(series, list) and bool(series)


def _legacy_signal_to_dsl(signal: str, params: dict) -> str:
    numeric = {
        str(k): v for k, v in (params or {}).items()
        if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    class _LegacyRewriter(ast.NodeTransformer):
        def visit_Call(self, node):  # noqa: N802 - ast API
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in _LEGACY_FUNC_ALIASES:
                node.func.id = _LEGACY_FUNC_ALIASES[node.func.id]
            return node

        def visit_Name(self, node):  # noqa: N802 - ast API
            if node.id in numeric:
                return ast.copy_location(ast.Constant(value=numeric[node.id]), node)
            return node

    try:
        tree = ast.parse(str(signal or "").strip(), mode="eval")
        tree = _LegacyRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return signal


def _dsl_values(spec: dict, capabilities: object | None) -> tuple[str, str, str]:
    signal = str(spec.get("signal") or "").strip()
    trade_series = str(spec.get("trade_series") or "").strip()
    position_map = str(spec.get("position_map") or "").strip().lower()

    # Backward-compatible migration for pre-DSL fixtures:
    # convert `series[0]` into trade_series and inline numeric param names into DSL literals.
    if _is_legacy_spec(spec):
        series = spec.get("series")
        if isinstance(series, list) and series and isinstance(series[0], str):
            trade_series = series[0].strip()
        if not position_map:
            position_map = "sign"
        signal = _legacy_signal_to_dsl(signal, spec.get("params") if isinstance(spec, dict) else {})
    return signal, trade_series, position_map


def _formulaic_spec_from_candidate(
    item: dict, claim_id: str, run_id: str, capabilities: object | None = None
) -> dict:
    spec = item.get("spec") if isinstance(item.get("spec"), dict) else {}
    signal, trade_series, position_map = _dsl_values(spec, capabilities)
    names = formulaic_signal.referenced_names(signal)
    inputs = sorted(names | {trade_series})
    param_grid = dict(spec.get("param_grid") or {})
    return {
        "module_id": f"auto_{claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": str(item.get("strategy_class") or "synthesized-family"),
        "claim_type": "formulaic_signal",
        "source": run_id,
        "claim_statement": str(item.get("statement") or ""),
        "claim_translation": (
            "Synthesized formulaic_signal candidate: parse the declared DSL signal, map it "
            "to exposure with position_map, and apply one-bar-lagged exposure to trade_series returns."
        ),
        "signal": signal,
        "trade_series": trade_series,
        "position_map": position_map,
        "inputs": inputs,
        "grid": param_grid,
        "param_grid": param_grid,
        "params": dict(spec.get("params") or {}),
        "binding_provenance": {
            "synthesis_run_id": run_id,
            "signal_names": sorted(names),
            "trade_series": {"series": trade_series, "confirmed": True},
        },
        "unknowns": [],
        "implementation_notes": (
            "Deterministic trusted formulaic_signal executor. No generated Python module; "
            "executor applies structural one-bar lag and production verdict gates."
        ),
        "_llm_mode": "synthesized-formulaic-signal",
    }


def _formulaic_source_span(statement: str, spec: dict | None) -> str:
    if not isinstance(spec, dict):
        return statement
    return "\n".join([
        statement,
        f"signal: {str(spec.get('signal') or '').strip()}",
        f"trade_series: {str(spec.get('trade_series') or '').strip()}",
        f"position_map: {str(spec.get('position_map') or '').strip().lower()}",
    ]).strip()


def _reconstructability(item: dict, capabilities: object | None) -> tuple[bool, str]:
    spec = item.get("spec")
    if not isinstance(spec, dict):
        return False, "missing spec"

    legacy_spec = _is_legacy_spec(spec)
    required_fields = _LEGACY_REQUIRED_SPEC_FIELDS if legacy_spec else _REQUIRED_SPEC_FIELDS
    for field in required_fields:
        if not _present(spec.get(field)):
            return False, f"missing spec.{field}"

    signal, trade_series, position_map = _dsl_values(spec, capabilities)
    if not signal:
        return False, "spec.signal must be a non-empty string"

    if not trade_series:
        return False, "spec.trade_series must be a non-empty capability series name"

    if position_map not in {"sign", "zscore_clip"}:
        return False, "spec.position_map must be sign or zscore_clip"

    params = spec.get("params")
    # N-1(c): param VALUES must be numeric, not just the keys — a non-numeric/None value breaks code-gen.
    if not isinstance(params, dict) or not params or not all(
        isinstance(k, str) and k.strip()
        and isinstance(v, (int, float)) and not isinstance(v, bool)
        for k, v in params.items()
    ):
        return False, "spec.params must map named params to numeric values"

    param_grid = spec.get("param_grid")
    if not isinstance(param_grid, dict) or not all(
        isinstance(k, str) and k.strip()
        and isinstance(v, (list, tuple, set)) and _present(v)
        for k, v in param_grid.items()
    ):
        return False, "spec.param_grid must be an object with non-empty ranges"
    # N-1(d): the grid may only vary DECLARED params — gridding an undeclared param while pinning the
    # real one to a single tuned point defeats the from-priors-grid promise (and feeds #58 a bad grid).
    if not set(param_grid) <= set(params):
        return False, "spec.param_grid grids a param not declared in spec.params"

    try:
        tree, signal_names = formulaic_signal.parse_formula(signal)
        formulaic_signal.validate_signal(signal)
    except formulaic_signal.FormulaicSignalError as exc:
        return False, f"signal not valid DSL: {exc}"
    if isinstance(tree.body, (ast.Name, ast.Constant)):
        return False, "spec.signal is a bare identifier/constant, not an operation on the series"
    # S72-1: a constant-only signal (e.g. zscore(1,20), sign(5)) parses + passes arity but references NO
    # series, so it is not executable (the DSL operators require a series argument). Require >=1 series.
    if not signal_names:
        return False, "spec.signal must reference at least one capability series (constant-only is not executable)"

    if legacy_spec:
        series = spec.get("series")
        declared_series = {str(x).strip() for x in series if isinstance(x, str) and x.strip()}
        undeclared = sorted(signal_names - declared_series)
        if undeclared:
            return False, f"undeclared signal identifier: {undeclared[0]}"

    # S72-3 (accepted, not "fixed"): the series-name check fails OPEN when capabilities are unavailable — this
    # is INTENTIONAL (capabilities are an optimization, not a guarantee; a made-up series is definitively
    # caught at EXECUTION as `data_unavailable`, an honest routing state, never a fabricated verdict). See
    # tests/test_synthesize_reconstructable.py::test_missing_capabilities_fails_open_for_series_resolution_only.
    allowed_series = _capability_series_names(capabilities)
    if allowed_series:
        unknown = sorted((signal_names | {trade_series}) - allowed_series)
        if unknown:
            return False, f"unknown series: {', '.join(unknown[:5])}"

    return True, ""


def _upsert_synthesis_summary(payload: dict) -> None:
    """Keep one durable lifecycle row per synthesis run, including retries."""
    path = config.SYNTHESIS_RUNS
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(path)
        by_id = {r.get("synthesis_run_id"): r for r in rows if r.get("synthesis_run_id")}
        by_id[payload["synthesis_run_id"]] = payload
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in by_id.values()))
        tmp.replace(path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def generate(n: int, corpus: dict, capabilities: dict) -> tuple[list[dict], dict]:
    capability_series = sorted(_capability_series_names(capabilities))
    user = _USER.format(n=n, corpus=json.dumps(corpus, indent=2)[:14000],
                        capabilities=json.dumps(capabilities, indent=2)[:6000],
                        capability_series=json.dumps(capability_series, indent=2)[:6000])
    parsed, resp = llm.call_json(
        "synthesizer", [{"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user}],
        temperature=0.5, timeout=300)
    rows = parsed.get("candidates", []) if isinstance(parsed, dict) else []
    rows = [x for x in rows if isinstance(x, dict) and x.get("statement")]
    return rows, {"model": resp.model, "prompt_hash": _snapshot_hash(user),
                  "in_tokens": resp.in_tokens, "out_tokens": resp.out_tokens,
                  "cost_usd": round(resp.cost_usd, 5)}


def normalize(run_id: str, raw: list[dict], graph: dict,
              capabilities: object | None = None) -> tuple[list[Claim], list[dict]]:
    node_map = {n.get("node_id"): n for n in graph.get("nodes", [])}
    valid_nodes = set(node_map)
    claims, normalized, seen = [], [], set()
    for i, item in enumerate(raw, 1):
        statement = " ".join(str(item.get("statement", "")).split())
        inspirations = [x for x in item.get("inspired_by", []) if x in valid_nodes]
        domains, datasets, periods = set(), set(), []
        for node_id in inspirations:
            p = node_map[node_id].get("data_provenance") or {}
            domains.update(map(str, p.get("data_domains") or []))
            datasets.update(map(str, p.get("datasets") or []))
            periods.extend(p.get("periods") or [])
        lineage = {"corpus_nodes": inspirations, "data_domains": sorted(domains),
                   "datasets": sorted(datasets), "periods": periods}
        klass = str(item.get("candidate_class", "conceptual_only"))
        grounded = bool(inspirations)
        duplicate = statement.lower() in seen
        seen.add(statement.lower())
        reconstructable, reconstructable_reason = _reconstructability(item, capabilities)
        claim_id = f"{run_id}-c{i:03d}"
        pipeline_spec = (
            _formulaic_spec_from_candidate(item, claim_id, run_id, capabilities)
            if reconstructable else None
        )
        if pipeline_spec is not None:
            lineage["formulaic_signal_spec"] = pipeline_spec
        source_span = _formulaic_source_span(statement, item.get("spec"))
        claim = Claim(
            claim_id=claim_id, statement=statement,
            mechanism=str(item.get("mechanism", ""))[:1000],
            scope=str(item.get("scope", ""))[:300], horizon=str(item.get("horizon", ""))[:120],
            source_id=run_id, source_span=source_span, claimed_metric_quote="",
            applicable_strategy_class="formulaic_signal",
            source_type="synthesized_hypothesis", search_cohort_id=run_id,
            raw_hypothesis_id=f"{run_id}-raw-{i:03d}",
            data_provenance=lineage)
        admitted = (grounded and not duplicate and reconstructable
                    and klass in {"testable_now", "testable_with_new_module"})
        claims.append(claim)
        normalized.append({"raw_hypothesis_id": claim.raw_hypothesis_id, "claim_id": claim_id,
                           "raw": item, "candidate_class": klass, "grounded": grounded,
                           "duplicate_in_run": duplicate, "reconstructable": reconstructable,
                           "reconstructable_reason": reconstructable_reason,
                           "admitted": admitted,
                           "pipeline_spec": pipeline_spec,
                           "data_provenance": lineage})
    return claims, normalized


def _source(root: Path, run_id: str, normalized: list[dict]) -> Path:
    text = [f"# Penrose synthesis {run_id}", "",
            "Untrusted candidate hypotheses. The Synthesizer proposes; the Referee judges.", ""]
    for row in normalized:
        spec = row.get("pipeline_spec") or {}
        text += [f"## {row['claim_id']}", "", row["raw"].get("statement", ""), "",
                 f"- signal: {spec.get('signal', '')}",
                 f"- trade_series: {spec.get('trade_series', '')}",
                 f"- position_map: {spec.get('position_map', '')}",
                 f"- inputs: {spec.get('inputs', [])}",
                 f"- grid: {spec.get('grid', {})}",
                 f"- inspired_by: {row['raw'].get('inspired_by', [])}",
                 f"- falsifier: {row['raw'].get('falsifier', '')}", ""]
    path = root / "source.md"
    _write_immutable(path, "\n".join(text))
    return path


def run_synthesis(*, n: int = 10, generate_only: bool = False,
                  run_id: str | None = None) -> dict:
    if not config.GENERATIVE_LAYER_ENABLED:
        raise RuntimeError("the generative layer is frozen (PEN-17): the verdict corpus is being "
                           "recalibrated; set PENROSE_GENERATIVE_LAYER=1 to override")
    if not 1 <= n <= 100:
        raise ValueError("n must be between 1 and 100")
    run_id = _validate_run_id(run_id or
        datetime.now(timezone.utc).strftime("synth-%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex[:8]}")
    root = config.SYNTHESIS_ARCHIVES / run_id
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("status") in {"complete", "generated_only", "no_admitted_candidates"}:
            _upsert_synthesis_summary(manifest)
            return manifest
    else:
        root.mkdir(parents=True)
        graph = build_files()
        corpus = _inputs(graph)
        manifest = {"schema_version": 1, "dream_run_id": run_id, "synthesis_run_id": run_id,
                    "source_type": "synthesized_hypothesis", "status": "registered",
                    "registered_at": _now(), "generation_budget": n,
                    "candidates_generated": 0, "candidates_admitted": 0,
                    "model": config.LLM_ROLES["synthesizer"]["model"],
                    "corpus_snapshot_hash": corpus["snapshot_hash"],
                    "artifact_dir": str(root)}
        _atomic_json(root / "manifest.json", manifest)
        _write_immutable(root / "corpus.json", json.dumps(corpus, indent=2))
        _write_immutable(root / "capabilities.json", json.dumps(capability_manifest(), indent=2))
    try:
        corpus = json.loads((root / "corpus.json").read_text())
        caps = json.loads((root / "capabilities.json").read_text())
        raw_path = root / "candidates.raw.jsonl"
        if raw_path.exists():
            raw = _read_jsonl(raw_path)
        else:
            raw, prov = generate(n, corpus, caps)
            manifest["generation_provenance"] = prov
            manifest = record_candidates(manifest, raw)
        graph = {"nodes": corpus.get("nodes", [])}
        claims, normalized = normalize(run_id, raw, graph, caps)
        manifest = register_search(manifest, claims, normalized)
        manifest["source_type"] = "synthesized_hypothesis"
        source = _source(root, run_id, normalized)
        admitted_ids = {r["claim_id"] for r in normalized if r["admitted"]}
        admitted = [c for c in claims if c.claim_id in admitted_ids]
        if generate_only or not admitted:
            manifest["status"] = "generated_only" if generate_only else "no_admitted_candidates"
        else:
            old = os.environ.get("PENROSE_HOLDOUT_MODE")
            os.environ["PENROSE_HOLDOUT_MODE"] = "readonly"
            try:
                from .pipeline.run import run_source
                result = run_source(source, use_llm=True, claims_override=admitted,
                                    source_type="synthesized_hypothesis")
            finally:
                if old is None: os.environ.pop("PENROSE_HOLDOUT_MODE", None)
                else: os.environ["PENROSE_HOLDOUT_MODE"] = old
            manifest["pipeline_run"] = {"decisions": result.get("decisions", []),
                                        "report": result.get("report")}
            manifest["status"] = "complete"
        manifest["completed_at"] = _now()
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"[:500]
        manifest["completed_at"] = _now()
        _atomic_json(root / "manifest.json", manifest)
        _upsert_synthesis_summary(manifest)
        raise
    _atomic_json(root / "manifest.json", manifest)
    _upsert_synthesis_summary(manifest)
    return manifest
