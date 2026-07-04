"""Corpus-grounded candidate-family generation.

This is a source adapter around the existing Referee. It preregisters the complete emitted
population and runs discovery with the production holdout physically read-only.
"""
from __future__ import annotations

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

_SYSTEM = """You synthesize candidate quantitative-research families from a leveled corpus.
You propose; you never certify. Ground every component in supplied node ids and boundaries.
Return the complete requested population as strict JSON. Never use or request confirmation data."""

_USER = """Create exactly {n} distinct candidate strategy families from these family principles
and cross-family mechanisms:
{corpus}

Capabilities:
{capabilities}

Return {{"candidates":[{{"statement":"falsifiable claim","mechanism":"tentative mechanism",
"scope":"market/universe","horizon":"measurable horizon","strategy_class":"class",
"candidate_class":"testable_now|testable_with_new_module|requires_data_acquisition|conceptual_only",
"required_series":[],"inspired_by":["node ids"],"boundaries":[],"falsifier":"specific rejection"}}]}}."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inputs(graph: dict) -> dict:
    nodes = [n for n in graph.get("nodes", [])
             if n.get("level") in {"family_principle", "cross_family_mechanism"}]
    return {"nodes": nodes, "snapshot_hash": _snapshot_hash(nodes)}


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
    user = _USER.format(n=n, corpus=json.dumps(corpus, indent=2)[:14000],
                        capabilities=json.dumps(capabilities, indent=2)[:6000])
    parsed, resp = llm.call_json(
        "synthesizer", [{"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user}],
        temperature=0.5, timeout=300)
    rows = parsed.get("candidates", []) if isinstance(parsed, dict) else []
    rows = [x for x in rows if isinstance(x, dict) and x.get("statement")]
    return rows, {"model": resp.model, "prompt_hash": _snapshot_hash(user),
                  "in_tokens": resp.in_tokens, "out_tokens": resp.out_tokens,
                  "cost_usd": round(resp.cost_usd, 5)}


def normalize(run_id: str, raw: list[dict], graph: dict) -> tuple[list[Claim], list[dict]]:
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
        admitted = grounded and not duplicate and klass in {"testable_now", "testable_with_new_module"}
        claims.append(claim)
        normalized.append({"raw_hypothesis_id": claim.raw_hypothesis_id, "claim_id": claim_id,
                           "raw": item, "candidate_class": klass, "grounded": grounded,
                           "duplicate_in_run": duplicate, "admitted": admitted,
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
        claims, normalized = normalize(run_id, raw, graph)
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
