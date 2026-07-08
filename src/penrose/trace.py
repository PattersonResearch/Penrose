"""Structured per-claim traces and triage summaries.

This module is read-mostly by design. Trace construction must never feed back
into verdict logic; run.py catches all trace emission errors and fails open.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TRACE_FIELDS = (
    "run_id",
    "source_id",
    "claim_id",
    "claim_type",
    "strategy_class",
    "inputs_requested",
    "data_missing",
    "stages_reached",
    "exit_stage",
    "gate_outcome",
    "verdict",
    "kill_reason",
    "failure_signature",
)


def normalize_failure_reason(reason: object) -> str:
    """Collapse volatile details so repeated failures hash together deterministically."""
    text = str(reason or "").lower()
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                  "<uuid>", text)
    text = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[t ][0-9:.+-z]+)?\b", "<time>", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:?\d{2}|z)?\b",
                  "<time>", text)
    text = re.sub(r"(?<![a-z])'[^'\n]*'", "<q>", text)
    text = re.sub(r'"[^"\n]*"', "<q>", text)
    text = re.sub(r"`[^`\n]*`", "<q>", text)
    text = re.sub(r"(?<![a-z])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?%?", "<num>", text)
    text = re.sub(r"(/[^\s:;,]+)+", "<path>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unspecified failure"


def failure_signature(verdict: object, exit_stage: object, reason: object) -> str:
    category = normalize_failure_reason(reason)
    payload = f"{str(verdict or '')}\0{str(exit_stage or '')}\0{category}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _stage_keys(rec: dict | None) -> list[str]:
    stages = (rec or {}).get("stages") or {}
    if not isinstance(stages, dict):
        return []
    return [str(k) for k in stages.keys() if str(k).startswith("P")]


def _stage_payload(rec: dict | None, key: str) -> dict:
    stages = (rec or {}).get("stages") or {}
    payload = stages.get(key, {}) if isinstance(stages, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _data_missing(decision: Any, rec: dict | None) -> list[str]:
    metrics = _get(decision, "metrics", {}) or {}
    p8 = _stage_payload(rec, "P8")
    p6_data = _stage_payload(rec, "P6_data_availability")
    return (
        _strings(metrics.get("missing_series"))
        or _strings(p8.get("missing_series"))
        or _strings(p6_data.get("missing_series"))
    )


def _inputs_requested(rec: dict | None) -> list[str]:
    direct = _strings((rec or {}).get("inputs_requested"))
    if direct:
        return direct
    p6_data = _stage_payload(rec, "P6_data_availability")
    return _strings(p6_data.get("inputs_requested")) or _strings(p6_data.get("missing_series"))


def _gate_outcome(decision: Any, rec: dict | None) -> str:
    p8 = _stage_payload(rec, "P8")
    metrics = _get(decision, "metrics", {}) or {}
    for key in ("kill_reason", "skip_reason", "reason", "stage", "error"):
        value = p8.get(key)
        if value:
            return str(value)
    if metrics.get("skip_reason"):
        return str(metrics.get("skip_reason"))
    if metrics.get("stage"):
        return str(metrics.get("stage"))
    if _get(decision, "kill_reason"):
        return str(_get(decision, "kill_reason"))
    return str(_get(decision, "verdict", "") or "unknown")


def _exit_stage(decision: Any, rec: dict | None) -> str:
    p8 = _stage_payload(rec, "P8")
    verdict = str(_get(decision, "verdict", "") or p8.get("verdict") or "")
    kill_reason = str(_get(decision, "kill_reason", "") or p8.get("kill_reason") or "")
    if verdict == "needs_data":
        if "P6_data_availability" in _stage_keys(rec):
            return "P6_data_availability"
        return "P7"
    if verdict == "pending_module":
        return "P6"
    if verdict == "needs_review":
        if "P6_variable_coverage" in _stage_keys(rec):
            return "P6_variable_coverage"
        if "P6_auto_impl_no_progress" in _stage_keys(rec):
            return "P6_auto_impl_no_progress"
        return "P6"
    if verdict == "cannot_replicate" and kill_reason == "unfaithful_spec":
        return "P6_pre_fidelity"
    if verdict == "engine_error":
        stage = str(p8.get("stage") or (_get(decision, "metrics", {}) or {}).get("stage") or "")
        if stage.startswith("P"):
            return stage.split()[0]
        if stage == "module run":
            return "P7"
        return stage or "engine_error"
    if verdict == "kill":
        return {
            "unfalsifiable": "P3",
            "fee_curve": "P4",
            "dedup": "P5",
        }.get(kill_reason, "P8")
    keys = _stage_keys(rec)
    return keys[-1] if keys else "unknown"


def project_trace_record(claim: Any, decision: Any, rec: dict | None, run_log: dict | None) -> dict:
    """Project a uniform trace row from a Claim/Decision plus the run's stage record."""
    run_log = run_log or {}
    idempotency = run_log.get("idempotency", {}) if isinstance(run_log, dict) else {}
    metrics = _get(decision, "metrics", {}) or {}
    source_id = (
        _get(claim, "source_id", "")
        or run_log.get("source_id", "")
        or (rec or {}).get("source_id", "")
    )
    claim_type = (
        _get(claim, "resolved_claim_type", "")
        or metrics.get("claim_type")
        or (rec or {}).get("claim_type")
        or ""
    )
    verdict = _get(decision, "verdict", "")
    gate = _gate_outcome(decision, rec)
    exit_stage = _exit_stage(decision, rec)
    row = {
        "run_id": idempotency.get("run_id", "") or run_log.get("run_id", ""),
        "source_id": source_id,
        "claim_id": _get(decision, "claim_id", "") or _get(claim, "claim_id", "") or (rec or {}).get("claim_id", ""),
        "claim_type": str(claim_type or ""),
        "strategy_class": str(_get(claim, "applicable_strategy_class", "") or (rec or {}).get("strategy_class", "")),
        "inputs_requested": _inputs_requested(rec),
        "data_missing": _data_missing(decision, rec),
        "stages_reached": _stage_keys(rec),
        "exit_stage": exit_stage,
        "gate_outcome": gate,
        "verdict": verdict,
        "kill_reason": _get(decision, "kill_reason", None),
        "failure_signature": failure_signature(verdict, exit_stage, gate),
    }
    return {field: row.get(field) for field in TRACE_FIELDS}


def trace_from_decision_row(row: dict) -> dict:
    """Best-effort trace-shaped row for legacy decisions.jsonl fallback."""
    metrics = row.get("metrics") or {}
    verdict = row.get("verdict")
    gate = (
        row.get("kill_reason")
        or metrics.get("skip_reason")
        or metrics.get("stage")
        or verdict
        or "unknown"
    )
    exit_stage = metrics.get("stage") or ("P8" if verdict in {"kill", "watch", "underpowered", "research-supported"} else "unknown")
    missing = _strings(metrics.get("missing_series"))
    trace = {
        "run_id": row.get("run_id", ""),
        "source_id": row.get("source_id", ""),
        "claim_id": row.get("claim_id", ""),
        "claim_type": str(metrics.get("claim_type") or ""),
        "strategy_class": str(row.get("strategy_class") or ""),
        "inputs_requested": [],
        "data_missing": missing,
        "stages_reached": [],
        "exit_stage": str(exit_stage or "unknown"),
        "gate_outcome": str(gate),
        "verdict": verdict,
        "kill_reason": row.get("kill_reason"),
        "failure_signature": row.get("failure_signature") or failure_signature(verdict, exit_stage, gate),
    }
    return {field: trace.get(field) for field in TRACE_FIELDS}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # MC-1: fail open on a file-level read error (permission/encoding), like every sibling reader
    # (views._read_jsonl, learning._read_jsonl, proposals._read_rows). Otherwise a permission-restricted
    # traces.jsonl raises through `penrose triage` (a RAW TRACEBACK on the CLI) and the penrose_triage
    # MCP tool, breaking the fail-graceful contract that all other read-only surfaces uphold.
    try:
        text = path.read_text()
    except OSError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def load_trace_rows(traces_path: Path, decisions_path: Path) -> tuple[list[dict], str]:
    traces = read_jsonl(traces_path)
    if traces:
        return traces, str(traces_path)
    decisions = read_jsonl(decisions_path)
    if decisions:
        return [trace_from_decision_row(row) for row in decisions], str(decisions_path)
    return [], ""


def triage_report(rows: list[dict], *, top: int = 15, source: str | None = None) -> dict:
    filtered = [
        row for row in rows
        if not source or str(row.get("source_id") or "") == str(source)
    ]
    verdict_counts = Counter(str(row.get("verdict") or "unknown") for row in filtered)
    stage_counts = Counter(str(row.get("exit_stage") or "unknown") for row in filtered)
    clusters: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in filtered:
        sig = str(row.get("failure_signature") or failure_signature(
            row.get("verdict"), row.get("exit_stage"), row.get("gate_outcome")))
        grouped[sig].append(row)
    for sig, members in grouped.items():
        first = sorted(members, key=lambda r: str(r.get("claim_id") or ""))[0]
        clusters[sig] = {
            "failure_signature": sig,
            "count": len(members),
            "verdict": first.get("verdict"),
            "exit_stage": first.get("exit_stage"),
            "kill_reason": first.get("kill_reason"),
            "gate_outcome": first.get("gate_outcome"),
            "example_claim_id": first.get("claim_id"),
        }
    ordered_clusters = sorted(
        clusters.values(),
        key=lambda r: (-int(r["count"]), str(r["failure_signature"])),
    )[:max(0, int(top))]
    return {
        "status": "ok",
        "source": source,
        "total": len(filtered),
        "verdict_distribution": dict(sorted(verdict_counts.items())),
        "stage_dropoff": dict(sorted(stage_counts.items())),
        "failure_clusters": ordered_clusters,
    }
