"""Read-only data accessors shared by the CLI and the read-only MCP server.

These RETURN structured data (lists/dicts) and never print, mutate, run, or approve
anything. The CLI read commands (`penrose verdicts/data-requests/status`) and the
MCP tools both call these, so there is ONE read path and no drift. Fail-open: a
missing/corrupt file yields `[]`/a minimal dict, never a raise.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out: list[dict] = []
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        return []
    return out


_MAX_VERDICTS = 1000


def verdicts(limit: int = 20) -> list[dict]:
    """Recent backtested verdicts (oldest..newest), read-only, from the analysis index.

    `limit` is clamped to [1, 1000]; a non-positive limit falls back to the default 20
    (an agent cannot request an unbounded result)."""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = 20
    n = 20 if n <= 0 else min(n, _MAX_VERDICTS)
    rows = _read_jsonl(config.ANALYSIS_INDEX)[-n:]
    return [
        {
            "claim_id": r.get("claim_id"),
            "verdict": r.get("verdict"),
            "kill_reason": r.get("kill_reason"),
            "statement": r.get("statement"),
            "synthetic": bool(r.get("synthetic")),
            "dsr": r.get("dsr"),
            "psr": r.get("psr"),
            "resolution": r.get("resolution"),
        }
        for r in rows
    ]


def data_requests() -> list[dict]:
    """Open `needs_data` blockers, deduped by claim_id (latest), read-only."""
    rows = [r for r in _read_jsonl(config.DATA_REQUESTS) if r.get("status", "open") == "open"]
    latest = {r.get("claim_id"): r for r in rows}
    return [
        {
            "claim_id": r.get("claim_id"),
            "missing_series": list(r.get("missing_series", []) or []),
            "auto_fetch_attempted": r.get("auto_fetch_attempted"),
        }
        for r in latest.values()
    ]


def proposals() -> list[dict]:
    """Propose-only principle proposals (status: proposed), read-only.

    Promotion to the approved brain stays human (P9); this only reads the propose store.
    Fail-open to `[]`."""
    try:
        from .proposals import read_proposals
        return list(read_proposals() or [])
    except Exception:  # noqa: BLE001
        return []


def principles(limit: int = 50) -> list[dict]:
    """Distilled cross-run advisory principle candidates from the corpus, read-only.

    This is the agent-discussable "what candidates exist" surface. It only distills and
    returns proposals; it never writes the approved brain. `limit` is clamped to [1, 500];
    fail-open to `[]`."""
    try:
        from .learning import distill_contrastive_principles, distill_principles
        rows = list(distill_principles() or []) + list(distill_contrastive_principles() or [])
    except Exception:  # noqa: BLE001
        return []
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = 50
    n = 50 if n <= 0 else min(n, 500)
    return rows[:n]


def status() -> dict:
    """Pipeline status from the dashboard live.json, read-only."""
    live = Path(config.LIVE_JSON)
    if not live.exists():
        return {"pipeline_status": "idle", "note": "no dashboard/live.json"}
    try:
        d = json.loads(live.read_text())
    except (json.JSONDecodeError, OSError):
        return {"pipeline_status": "unknown"}
    return {
        "pipeline_status": d.get("pipeline_status", "unknown"),
        "status_badge": d.get("status_badge", ""),
        "updated_at": d.get("updated_at"),
        "stats": d.get("stats") or {},
    }


def triage(top: int = 15, source: str | None = None) -> dict:
    """Read-only failure-cluster analysis of the trace corpus: verdict distribution, per-stage
    drop-off, and top recurring failure signatures. Reads reports/traces.jsonl (decisions fallback).
    READ-ONLY; never writes anything."""
    from pathlib import Path
    from . import config
    from .trace import load_trace_rows, triage_report
    rows, loaded_from = load_trace_rows(Path(config.TRACES), Path(config.DECISIONS_LOG))
    if not rows:
        return {"status": "empty",
                "message": "No traces or decisions found yet (reports/traces.jsonl and decisions.jsonl empty)."}
    report = triage_report(rows, top=int(top), source=source)
    report["input"] = loaded_from
    return report
