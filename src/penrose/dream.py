"""Registered hypothesis generation for penrose.

Dreaming is a SOURCE ADAPTER, not a second research pipeline:

  observe flat-file evidence -> generate immutable raw hypotheses -> register the COMPLETE search
  -> normalize to canonical Claim objects -> send eligible claims through the existing P3-P9 path

The generator never writes to the knowledge store. It can run without it because its evidence packet
comes from penrose's own reports and advisory connection output. During dream triage the production
holdout is physically read-only; generated hypotheses are capped below `research-supported` until
an independent forward/external confirmation round exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fcntl

from . import config, llm
from .brain import Claim


_DREAM_SYSTEM = """You generate falsifiable quantitative-research hypotheses for penrose.
You are a generator, not a judge. Prior kills and principles INFORM your search but never forbid a
candidate. Return exactly the requested number of candidates; do not privately select a smaller
"best" subset. Every candidate must be distinct and must honestly declare whether it is testable
with the supplied data capabilities. Do not claim that an idea works. Output strict JSON only."""

_DREAM_USER = """Create exactly {n} candidate hypotheses.

AVAILABLE DATA / FEATURES:
{capabilities}

EVIDENCE PACKET (conditional observations, not universal truths):
{evidence}

Output:
{{"candidates": [{{
  "statement": "directional falsifiable claim",
  "mechanism": "why it might occur",
  "scope": "assets/market/universe",
  "horizon": "measurable horizon",
  "strategy_class": "stable concise class",
  "candidate_class": "testable_now|testable_with_new_module|requires_data_acquisition|conceptual_only",
  "required_series": ["exact available keys when possible"],
  "inspired_by": ["claim/principle/cluster ids"],
  "falsifier": "specific observation that would reject it"
}}]}}

Do not omit weak-looking ideas after considering them: the complete emitted set is registered as the
search denominator and penrose, not you, decides what survives."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _upsert_run_summary(payload: dict) -> None:
    """Keep one lifecycle summary per dream_run_id, including across retries."""
    path = config.DREAM_RUNS
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(path)
        by_id = {r.get("dream_run_id"): r for r in rows if r.get("dream_run_id")}
        by_id[payload["dream_run_id"]] = payload
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in by_id.values()))
        tmp.replace(path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:48] or "hypothesis"


def _validate_run_id(run_id: str) -> str:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id or "")
            or run_id in {".", ".."} or ".." in run_id):
        raise ValueError("run_id must be 1-96 safe filename characters without path traversal")
    return run_id


def _write_immutable(path: Path, content: str) -> None:
    """Create an artifact once; a retry may only confirm identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text() != content:
            raise RuntimeError(f"immutable dream artifact already exists with different content: {path}")
        return
    with os.fdopen(fd, "w") as f:
        f.write(content)


def _snapshot_hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def create_manifest(*, run_id: str, generation_budget: int, model: str,
                    corpus_snapshot_hash: str, root: Path | None = None) -> dict:
    """Create the pre-generation manifest. The budget exists before model output."""
    run_id = _validate_run_id(run_id)
    root = Path(root or (config.DREAM_ARCHIVES / run_id))
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dream_run_id": run_id,
        "source_type": "generated_hypothesis",
        "status": "registered",
        "registered_at": _now(),
        "generation_budget": int(generation_budget),
        "candidates_generated": 0,
        "candidates_admitted": 0,
        "model": model,
        "corpus_snapshot_hash": corpus_snapshot_hash,
        "prompt_hash": None,
        "family_denominators": {},
        "artifact_dir": str(root),
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def record_candidates(manifest: dict, candidates: list[dict]) -> dict:
    """Persist the complete raw emitted set before normalization or selection."""
    root = Path(manifest["artifact_dir"])
    raw_path = root / "candidates.raw.jsonl"
    _write_immutable(
        raw_path, "".join(json.dumps(c, default=str) + "\n" for c in candidates))
    manifest = dict(manifest)
    manifest["candidates_generated"] = len(candidates)
    manifest["status"] = "generated"
    manifest["generated_at"] = _now()
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def build_evidence_packet(limit: int = 16) -> dict:
    """Build a flat-file, eligibility-filtered reflection packet. The knowledge store is optional."""
    rows = _read_jsonl(config.ANALYSIS_INDEX)
    latest = list({r.get("claim_id"): r for r in rows if r.get("claim_id")}.values())
    eligible = []
    for r in latest:
        verdict = r.get("verdict")
        fidelity = r.get("fidelity") or {}
        if verdict in {"cannot_replicate", "pending_module"}:
            continue
        if r.get("synthetic"):
            continue
        if fidelity and fidelity.get("faithful") is False:
            continue
        eligible.append({
            "claim_id": r.get("claim_id"),
            "statement": r.get("statement"),
            "verdict": verdict,
            "kill_reason": r.get("kill_reason"),
            "source_title": r.get("source_title"),
        })
    eligible = eligible[-limit:]
    connections = {}
    cpath = config.ROOT / "dashboard" / "connections.json"
    if cpath.exists():
        try:
            raw = json.loads(cpath.read_text())
            connections = {
                "failure_clusters": (raw.get("failure_clusters") or [])[:6],
                "cross_domain": (raw.get("cross_domain") or [])[:4],
                "principles": (raw.get("principles") or [])[:6],
            }
        except (json.JSONDecodeError, OSError):
            pass
    packet = {"eligible_verdicts": eligible, "connections": connections}
    packet["snapshot_hash"] = _snapshot_hash(packet)
    return packet


def capability_manifest() -> dict:
    """Machine-readable test envelope shown to the generator."""
    from .pipeline.impl_gen import BUNDLE_KEYS
    return {
        "series": BUNDLE_KEYS,
        "trusted_strategy_classes": _trusted_classes(),
        "cost_provenance": config.COST_PROVENANCE,
        "minimum_oos_trades": config.DSR_DECISION["min_oos_bars"],
        "note": ("Claims outside this envelope may be retained as requires_data_acquisition, "
                 "but must not masquerade as testable_now."),
    }


def _trusted_classes() -> list[str]:
    try:
        from .pipeline import run as runmod
        runmod.REGISTRY.clear()
        runmod._register_known_modules()
        return sorted(runmod._known_classes().keys())
    except Exception:  # noqa: BLE001
        return []


def generate_candidates(n: int, evidence: dict, capabilities: dict) -> tuple[list[dict], dict]:
    user = _DREAM_USER.format(
        n=n,
        capabilities=json.dumps(capabilities, indent=2, default=str)[:10000],
        evidence=json.dumps(evidence, indent=2, default=str)[:10000],
    )
    parsed, resp = llm.call_json(
        "dreamer",
        [{"role": "system", "content": _DREAM_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.7,
        timeout=300,
    )
    raw = parsed.get("candidates", []) if isinstance(parsed, dict) else []
    candidates = [c for c in raw if isinstance(c, dict) and str(c.get("statement", "")).strip()]
    provenance = {
        "model": resp.model, "in_tokens": resp.in_tokens, "out_tokens": resp.out_tokens,
        "cost_usd": round(resp.cost_usd, 5), "cached": resp.cached,
        "prompt_hash": _snapshot_hash({"system": _DREAM_SYSTEM, "user": user}),
    }
    return candidates, provenance


def normalize_candidates(run_id: str, candidates: list[dict]) -> tuple[list[Claim], list[dict]]:
    """Create canonical Claims while preserving every raw candidate and duplicate."""
    claims = []
    normalized = []
    seen = set()
    allowed = {
        "testable_now", "testable_with_new_module",
        "requires_data_acquisition", "conceptual_only",
    }
    for i, raw in enumerate(candidates, 1):
        statement = " ".join(str(raw.get("statement", "")).split())
        raw_id = f"{run_id}-raw-{i:03d}"
        klass = str(raw.get("candidate_class", "conceptual_only"))
        if klass not in allowed:
            klass = "conceptual_only"
        key = statement.lower()
        duplicate = key in seen
        seen.add(key)
        claim_id = f"{run_id}-c{i:03d}"
        claim = Claim(
            claim_id=claim_id,
            statement=statement,
            mechanism=str(raw.get("mechanism", ""))[:1000],
            scope=str(raw.get("scope", ""))[:300],
            horizon=str(raw.get("horizon", ""))[:120],
            source_id=run_id,
            source_span=statement,  # exact immutable generated source
            claimed_metric_quote="",
            applicable_strategy_class=str(raw.get("strategy_class") or _slug(statement)),
            source_type="generated_hypothesis",
            search_cohort_id=run_id,
            raw_hypothesis_id=raw_id,
        )
        normalized.append({
            "raw_hypothesis_id": raw_id,
            "claim_id": claim_id,
            "raw": raw,
            "candidate_class": klass,
            "duplicate_in_run": duplicate,
            "admitted": (klass in {"testable_now", "testable_with_new_module"} and not duplicate),
        })
        claims.append(claim)
    return claims, normalized


def _family_for(claim: Claim) -> str:
    from .pipeline.run import _data_domain
    # Generated searches use a stable, conservative domain family. The model may invent or rename
    # strategy classes between nights; allowing that vocabulary to define the family would reset
    # the accumulated multiple-testing denominator.
    return f"generated::{_data_domain(claim)}"


def register_search(manifest: dict, claims: list[Claim], normalized: list[dict]) -> dict:
    """Register the full emitted search before any admitted claim enters P3."""
    from .pipeline import p7_backtest as p7
    by_id = {c.claim_id: c for c in claims}
    family_counts = Counter(_family_for(c) for c in claims)
    # Every emitted candidate competed inside the same generator search. The conservative,
    # auditable denominator is the PRE-REGISTERED cohort budget for every family the run touched,
    # not merely the candidates that happened to share that family or survive normalization.
    cohort_denominator = max(
        int(manifest["generation_budget"]), int(manifest.get("candidates_generated", 0)))
    rows = []
    for item in normalized:
        claim = by_id[item["claim_id"]]
        family = _family_for(claim)
        claim.search_denominator = cohort_denominator
        rows.append({
            "strategy": claim.claim_id,
            "family": family,
            "generation_source": "dream",
            "search_cohort_id": manifest["dream_run_id"],
            "search_denominator": cohort_denominator,
        })
    p7.register_trials(rows)
    manifest = dict(manifest)
    manifest["family_emitted_counts"] = dict(family_counts)
    manifest["family_denominators"] = {
        family: cohort_denominator for family in family_counts
    }
    manifest["effective_search_denominator"] = cohort_denominator
    manifest["candidates_admitted"] = sum(bool(x["admitted"]) for x in normalized)
    manifest["status"] = "search_registered"
    manifest["search_registered_at"] = _now()
    root = Path(manifest["artifact_dir"])
    _write_immutable(
        root / "candidates.normalized.jsonl",
        "".join(json.dumps(x, default=str) + "\n" for x in normalized))
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def _write_source(root: Path, run_id: str, normalized: list[dict]) -> Path:
    lines = [
        f"# Penrose Dream Run {run_id}",
        "",
        "Generated hypotheses below are untrusted source material. Their complete search was "
        "registered before any candidate entered the falsification pipeline.",
        "",
    ]
    for item in normalized:
        raw = item["raw"]
        lines += [
            f"## {item['claim_id']}",
            "",
            str(raw.get("statement", "")),
            "",
            f"- candidate_class: {item['candidate_class']}",
            f"- mechanism: {raw.get('mechanism', '')}",
            f"- falsifier: {raw.get('falsifier', '')}",
            "",
        ]
    path = root / "source.md"
    _write_immutable(path, "\n".join(lines))
    return path


def run_dream(*, n: int = 10, generate_only: bool = False,
              run_id: str | None = None) -> dict:
    """Generate, register, and optionally falsify one dream search."""
    if not config.GENERATIVE_LAYER_ENABLED:
        raise RuntimeError("the generative layer is frozen (PEN-17): the verdict corpus is being "
                           "recalibrated; set PENROSE_GENERATIVE_LAYER=1 to override")
    if n < 1 or n > 100:
        raise ValueError("n must be between 1 and 100")
    run_id = _validate_run_id(
        run_id or datetime.now(timezone.utc).strftime("dream-%Y%m%dT%H%M%S.%fZ")
        + f"-{uuid.uuid4().hex[:8]}")
    root = config.DREAM_ARCHIVES / run_id
    manifest_path = root / "manifest.json"
    terminal = {"complete", "generated_only", "no_admitted_candidates"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") in terminal:
            return manifest
        if int(manifest.get("generation_budget", n)) != n:
            raise ValueError("existing run_id has a different generation budget")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError:
            if not manifest_path.exists():
                raise RuntimeError(f"dream run directory exists without a manifest: {root}")
            manifest = json.loads(manifest_path.read_text())
        else:
            evidence = build_evidence_packet()
            capabilities = capability_manifest()
            manifest = create_manifest(
                run_id=run_id, generation_budget=n,
                model=config.LLM_ROLES["dreamer"]["model"],
                corpus_snapshot_hash=evidence["snapshot_hash"], root=root)
            _write_immutable(root / "evidence.json", json.dumps(evidence, indent=2, default=str))
            _write_immutable(
                root / "capabilities.json", json.dumps(capabilities, indent=2, default=str))

    if not (root / "evidence.json").exists():
        evidence = build_evidence_packet()
        _write_immutable(root / "evidence.json", json.dumps(evidence, indent=2, default=str))
    else:
        evidence = json.loads((root / "evidence.json").read_text())
    if not (root / "capabilities.json").exists():
        capabilities = capability_manifest()
        _write_immutable(
            root / "capabilities.json", json.dumps(capabilities, indent=2, default=str))
    else:
        capabilities = json.loads((root / "capabilities.json").read_text())
    manifest.pop("error", None)
    manifest.pop("completed_at", None)

    try:
        raw_path = root / "candidates.raw.jsonl"
        if raw_path.exists():
            candidates = _read_jsonl(raw_path)
        else:
            candidates, prov = generate_candidates(n, evidence, capabilities)
            manifest["prompt_hash"] = prov["prompt_hash"]
            manifest["generation_provenance"] = prov
            manifest = record_candidates(manifest, candidates)
        claims, normalized = normalize_candidates(run_id, candidates)
        manifest = register_search(manifest, claims, normalized)
        source_path = _write_source(root, run_id, normalized)

        admitted_ids = {x["claim_id"] for x in normalized if x["admitted"]}
        admitted = [c for c in claims if c.claim_id in admitted_ids]
        if generate_only or not admitted:
            manifest["status"] = "generated_only" if generate_only else "no_admitted_candidates"
        else:
            # Structural holdout isolation for the entire dream triage call. Restore the caller's
            # environment even when the pipeline raises.
            old_mode = os.environ.get("PENROSE_HOLDOUT_MODE")
            os.environ["PENROSE_HOLDOUT_MODE"] = "readonly"
            try:
                from .pipeline.run import run_source
                result = run_source(
                    source_path, use_llm=True, claims_override=admitted,
                    source_type="generated_hypothesis")
            finally:
                if old_mode is None:
                    os.environ.pop("PENROSE_HOLDOUT_MODE", None)
                else:
                    os.environ["PENROSE_HOLDOUT_MODE"] = old_mode
            manifest["pipeline_run"] = {
                "source_id": result.get("source_id"),
                "report": result.get("report"),
                "decisions": result.get("decisions", []),
            }
            manifest["status"] = "complete"
        manifest["completed_at"] = _now()
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(e).__name__}: {e}"[:500]
        manifest["completed_at"] = _now()
        _atomic_json(root / "manifest.json", manifest)
        _upsert_run_summary(manifest)
        raise

    _atomic_json(root / "manifest.json", manifest)
    _upsert_run_summary(manifest)
    return manifest
