"""Grounded Level-4 concept extraction.

Concepts are advisory memory. They never alter a Referee verdict and extraction failure is
deliberately represented as no concept, never as a pipeline failure.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fcntl

from . import config, llm


@dataclass
class Concept:
    concept_id: str
    source_claim_id: str
    statement: str
    mechanism: str = ""
    surviving_explanation: str = ""
    rejected_explanations: list[str] = field(default_factory=list)
    boundary: dict = field(default_factory=dict)
    reusable_principle: str = ""
    implementation_consequence: str = ""
    evidence_strength: dict = field(default_factory=dict)
    data_provenance: dict = field(default_factory=dict)
    source_type: str = "external_source"
    abstraction_level: str = "observation"
    created_at: str = ""
    seed: int = 0
    grounding_flags: list[str] = field(default_factory=list)
    source_verdict: str = ""
    evidence_direction: str = "unknown"

    def to_json(self) -> dict:
        return asdict(self)


_SYSTEM = """Extract one conservative research concept from the supplied experiment record.
You are not a certifier. Do not say validated, proven, alpha, causal, or supported when the
headline verdict is kill/underpowered/cannot_replicate. Use only facts present in the record.
Return strict JSON with: statement, mechanism, surviving_explanation, rejected_explanations,
boundary, reusable_principle, implementation_consequence."""

_GROUNDING_SYSTEM = """You are an adversarial grounding refuter. Compare a drafted concept
field-by-field against the experiment record. A field is supported only when the record directly
contains evidence for that exact claim and strength. Reject causal, universal, reliable, robust,
alpha, edge, validated, proven, predictive, or effectiveness language unless the record itself
states it at that strength. Return strict JSON:
{"supported_fields":["statement",...], "unsupported":{"field":"reason",...}}."""


def stable_seed(record: dict) -> int:
    blob = json.dumps(record, sort_keys=True, default=str).encode()
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def _strength(record: dict) -> dict:
    metrics = record.get("metrics") or {}
    return {k: metrics.get(k) for k in ("dsr", "psr", "n_oos", "n_trades", "power_sufficient")
            if metrics.get(k) is not None}


def _evidence_direction(record: dict) -> str:
    verdict = str(record.get("verdict", ""))
    if verdict in {"watch", "research-supported"}:
        return "positive"
    if verdict == "kill":
        return "negative"
    return "unknown"


def _fallback(record: dict) -> dict:
    verdict = str(record.get("verdict", ""))
    claim = str(record.get("statement", "")).strip()
    explanations = record.get("competing_explanations") or []
    survivors = [x.get("explanation", "") for x in explanations if x.get("verdict") == "survives"]
    rejected = [x.get("explanation", "") for x in explanations if x.get("verdict") == "rejected"]
    return {
        "statement": f"Experiment on: {claim}" if claim else "Experiment observation",
        "mechanism": str(record.get("mechanism", "")),
        "surviving_explanation": survivors[0] if survivors and verdict in {"watch", "research-supported"} else "",
        "rejected_explanations": rejected,
        "boundary": {"regime": (record.get("metrics") or {}).get("regime"),
                     "kill_reason": record.get("kill_reason")},
        "reusable_principle": "",
        "implementation_consequence": "",
    }


def ground_draft(draft: dict, record: dict,
                 supported_fields: set[str] | None = None) -> tuple[dict, list[str]]:
    """Field-level refuter: unsupported/over-strong fields are dropped, never repaired upward."""
    out = dict(draft or {})
    flags: list[str] = []
    verdict = str(record.get("verdict", ""))
    core_forbidden = (
        "validated", "proven", "confirmed alpha", "caused by", "research-supported",
        "guarantees",
    )
    strong_forbidden = core_forbidden + (
        "alpha", "edge", "reliable", "robust", "works", "effective",
        "predicts", "outperforms",
    )
    for key in ("statement", "mechanism", "surviving_explanation",
                "reusable_principle", "implementation_consequence"):
        value = str(out.get(key, "") or "").strip()
        forbidden = core_forbidden if key == "statement" else strong_forbidden
        if any(term in value.lower() for term in forbidden):
            out[key] = ""
            flags.append(f"{key}:overclaim")
        elif supported_fields is not None and key not in supported_fields:
            out[key] = ""
            flags.append(f"{key}:grounding_refuter_unsupported")
    if verdict not in {"watch", "research-supported"} and out.get("surviving_explanation"):
        out["surviving_explanation"] = ""
        flags.append("surviving_explanation:verdict_too_weak")
    tested = {str(x.get("explanation", "")) for x in record.get("competing_explanations", [])
              if x.get("verdict") == "rejected"}
    requested = [str(x) for x in out.get("rejected_explanations", []) if str(x) in tested]
    if len(requested) != len(out.get("rejected_explanations", []) or []):
        flags.append("rejected_explanations:unsupported_dropped")
    out["rejected_explanations"] = requested
    if not isinstance(out.get("boundary"), dict):
        out["boundary"] = {}
        flags.append("boundary:invalid")
    return out, flags


def _refute_grounding(draft: dict, record: dict, seed: int) -> tuple[set[str] | None, list[str]]:
    try:
        parsed, _ = llm.call_json(
            "concept_grounding_refuter",
            [{"role": "system", "content": _GROUNDING_SYSTEM},
             {"role": "user", "content": json.dumps(
                 {"seed": seed, "record": record, "draft": draft}, default=str)[:18000]}],
            temperature=0.0,
        )
        if not isinstance(parsed, dict) or not isinstance(parsed.get("supported_fields"), list):
            return set(), ["grounding_refuter:inconclusive"]
        supported = {str(x) for x in parsed["supported_fields"]}
        unsupported = parsed.get("unsupported") or {}
        flags = [f"{k}:grounding_refuter:{str(v)[:120]}" for k, v in unsupported.items()]
        return supported, flags
    except Exception as e:  # fail closed on the LLM draft; deterministic fallback remains available
        return set(), [f"grounding_refuter:error:{type(e).__name__}"]


def extract(record: dict, *, use_llm: bool = True) -> Concept | None:
    try:
        seed = stable_seed(record)
        draft = _fallback(record)
        refuter_flags: list[str] = []
        supported_fields = None
        if use_llm:
            parsed, _ = llm.call_json(
                "concept_extractor",
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": json.dumps(
                     {"seed": seed, "record": record}, default=str)[:14000]}],
                temperature=0.0,
            )
            if isinstance(parsed, dict):
                llm_draft = dict(draft)
                llm_draft.update(parsed)
                supported_fields, refuter_flags = _refute_grounding(llm_draft, record, seed)
                # The refuter authorizes LLM fields. If it fails/inconclusive, fall back to the
                # deterministic record-derived draft rather than trusting unsupported prose.
                if supported_fields:
                    draft = llm_draft
                else:
                    supported_fields = None
        grounded, flags = ground_draft(draft, record, supported_fields)
        flags.extend(refuter_flags)
        created_at = record.get("run_at") or record.get("created_at")
        if not created_at:
            created_at = "1970-01-01T00:00:00+00:00"
            flags.append("created_at:missing_source_timestamp")
        claim_id = str(record.get("claim_id", "")).strip()
        if not claim_id or not grounded.get("statement"):
            return None
        return Concept(
            concept_id=f"concept-{claim_id}-{seed:08x}",
            source_claim_id=claim_id,
            statement=str(grounded.get("statement", ""))[:1000],
            mechanism=str(grounded.get("mechanism", ""))[:1000],
            surviving_explanation=str(grounded.get("surviving_explanation", ""))[:1000],
            rejected_explanations=list(grounded.get("rejected_explanations", []))[:10],
            boundary=grounded.get("boundary") or {},
            reusable_principle=str(grounded.get("reusable_principle", ""))[:1000],
            implementation_consequence=str(grounded.get("implementation_consequence", ""))[:1000],
            evidence_strength=_strength(record),
            data_provenance=record.get("data_provenance") or {},
            source_type=str(record.get("source_type") or "external_source"),
            created_at=str(created_at),
            seed=seed, grounding_flags=flags,
            source_verdict=str(record.get("verdict", "")),
            evidence_direction=_evidence_direction(record),
        )
    except Exception:
        return None


def append(concept: Concept, path: Path | None = None) -> None:
    path = Path(path or config.CONCEPTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    existing.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
        by_id = {x.get("concept_id"): x for x in existing}
        by_id[concept.concept_id] = concept.to_json()
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("".join(json.dumps(x, default=str) + "\n" for x in by_id.values()))
        tmp.replace(path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def extract_and_append(record: dict, *, use_llm: bool = True) -> dict | None:
    concept = extract(record, use_llm=use_llm)
    if concept is None:
        return None
    append(concept)
    return concept.to_json()
