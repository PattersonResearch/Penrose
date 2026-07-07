"""Corpus-grounded candidate-family generation.

This is a source adapter around the existing Referee. It preregisters the complete emitted
population and runs discovery with the production holdout physically read-only.
"""
from __future__ import annotations

import ast
import json
import os
import re
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

_SYSTEM = """You synthesize candidate quantitative-research families from a leveled corpus.
You propose; you never certify. Ground every component in supplied node ids and boundaries.
Return the complete requested population as strict JSON. Never use or request confirmation data.
Each candidate must keep the prose fields and also include a structured, buildable spec object:
- signal: an explicit expression using only named catalog series from the supplied capabilities and
  named params, e.g. "zscore(funding_btc, window) - ma(funding_btc, window2)".
- series: exact catalog series names used by signal; every name must come from capabilities.
- params: named parameter values.
- param_grid: a prior parameter grid with reasonable ranges for every tunable param, not a single
  tuned point, e.g. {"window":[10,20,60]}.
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

Use ONLY the allowed capability series names in spec.series and spec.signal.

Return {{"candidates":[{{"statement":"falsifiable claim","mechanism":"tentative mechanism",
"scope":"market/universe","horizon":"measurable horizon","strategy_class":"class",
"candidate_class":"testable_now|testable_with_new_module|requires_data_acquisition|conceptual_only",
"required_series":[],"inspired_by":["node ids"],"boundaries":[],"falsifier":"specific rejection",
"spec":{{"signal":"expression using allowed series and params","series":["allowed_series_name"],
"params":{{"param_name":20}},"param_grid":{{"param_name":[10,20,60]}},
"conditioning":null,"entry_exit":"explicit entry and exit rules","horizon":"holding horizon"}}}}]}}."""

_REQUIRED_SPEC_FIELDS = ("signal", "series", "params", "param_grid", "entry_exit", "horizon")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SIGNAL_FUNCTIONS = frozenset({
    "abs", "clip", "corr", "cov", "diff", "ema", "ewm", "lag", "log", "ma", "max", "mean",
    "min", "pct_change", "rank", "ratio", "sign", "sma", "std", "sum", "vol", "zscore",
    # N-2: the reconstructor (LLM code-gen) can implement any pandas/numpy function, so the gate must
    # not be narrower than it. Add common, unambiguously-implementable ones so legitimate candidate
    # families are not silently dropped.
    "median", "var", "quantile", "exp", "sqrt", "cumsum", "cumprod", "skew", "kurt",
    "shift", "winsorize", "demean", "tanh", "roll", "cbrt",
})
# N-1: functions that require a series AND a second arg (window/period). A call with too few args is
# a malformed signal that would break reconstruction; reject it at admission. Functions not listed
# (abs/sign/log/exp/sqrt/clip/cumsum/...) accept any arity.
_FUNCTION_MIN_ARGS = {
    "zscore": 2, "ma": 2, "sma": 2, "ema": 2, "ewm": 2, "std": 2, "var": 2, "vol": 2, "lag": 2,
    "shift": 2, "diff": 2, "pct_change": 2, "rank": 2, "sum": 2, "corr": 2, "cov": 2, "ratio": 2,
    "median": 2, "quantile": 2, "roll": 2,
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


def _reconstructability(item: dict, capabilities: object | None) -> tuple[bool, str]:
    spec = item.get("spec")
    if not isinstance(spec, dict):
        return False, "missing spec"

    for field in _REQUIRED_SPEC_FIELDS:
        if not _present(spec.get(field)):
            return False, f"missing spec.{field}"

    signal = spec.get("signal")
    if not isinstance(signal, str) or not signal.strip():
        return False, "spec.signal must be a non-empty string"

    series = spec.get("series")
    if not isinstance(series, list) or not all(isinstance(x, str) and x.strip() for x in series):
        return False, "spec.series must be a non-empty list of names"
    series_names = {x.strip() for x in series}

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

    allowed_series = _capability_series_names(capabilities)
    if allowed_series:
        unknown = sorted(series_names - allowed_series)
        if unknown:
            return False, f"unknown series: {', '.join(unknown[:5])}"

    # N-1(a/b/e): the signal must be a well-formed EXPRESSION (not a bare identifier or malformed
    # string), every identifier declared, and known window-functions must meet their required arity —
    # so a syntactically-valid-but-vacuous signal can't be admitted and then pending_module downstream.
    try:
        tree = ast.parse(signal.strip(), mode="eval")
    except SyntaxError:
        return False, "spec.signal is not a valid expression"
    if isinstance(tree.body, (ast.Name, ast.Constant)):
        return False, "spec.signal is a bare identifier/constant, not an operation on the series"
    allowed_signal_names = series_names | {k.strip() for k in params}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id not in allowed_signal_names
                and node.id not in _SIGNAL_FUNCTIONS):
            return False, f"undeclared signal identifier: {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            need = _FUNCTION_MIN_ARGS.get(node.func.id)
            if need is not None and len(node.args) < need:
                return False, f"signal function {node.func.id}() needs >= {need} args"

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
        claim = Claim(
            claim_id=claim_id, statement=statement,
            mechanism=str(item.get("mechanism", ""))[:1000],
            scope=str(item.get("scope", ""))[:300], horizon=str(item.get("horizon", ""))[:120],
            source_id=run_id, source_span=statement, claimed_metric_quote="",
            applicable_strategy_class=str(item.get("strategy_class", "synthesized-family")),
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
                           "data_provenance": lineage})
    return claims, normalized


def _source(root: Path, run_id: str, normalized: list[dict]) -> Path:
    text = [f"# Penrose synthesis {run_id}", "",
            "Untrusted candidate hypotheses. The Synthesizer proposes; the Referee judges.", ""]
    for row in normalized:
        text += [f"## {row['claim_id']}", "", row["raw"].get("statement", ""), "",
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
