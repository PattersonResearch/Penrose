"""Read-only data accessors shared by the CLI and the read-only MCP server.

These RETURN structured data (lists/dicts) and never print, mutate, run, or approve
anything. The CLI read commands (`penrose verdicts/data-requests/status`) and the
MCP tools both call these, so there is ONE read path and no drift. Fail-open: a
missing/corrupt file yields `[]`/a minimal dict, never a raise.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

from . import config
from .audit import verify_events


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


def _latest_audit_path() -> Path | None:
    audit_dir = Path(config.AUDIT)
    try:
        files = [p for p in audit_dir.glob("*.jsonl") if p.is_file()]
    except OSError:
        return None
    if not files:
        return None
    # deterministic under an mtime tie: break ties by filename so `penrose audit`
    # never picks an arbitrary run when two logs share a wall-clock second.
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def audit_summary(run_id=None, *, verify: bool = True) -> dict:
    """Read-only summary of a per-run audit log.

    Returns chain verification status, reproduction envelope, stage timings,
    gate outcome counts, and a simple stage drop-off count. Missing/corrupt
    files fail gracefully into a structured empty/error object.
    """
    if run_id is None:
        path = _latest_audit_path()
        if path is None:
            return {"status": "empty", "message": "No audit logs found yet."}
        run_id = path.stem
    else:
        path = Path(config.AUDIT) / f"{run_id}.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return {"status": "empty", "run_id": str(run_id), "message": f"No audit events found at {path}."}

    if verify:
        chain_ok, broken_seq = verify_events(rows)
    else:
        chain_ok, broken_seq = None, None

    # A non-dict row (tampered/truncated/foreign line) must never crash the summary — it is
    # already reflected as a broken chain above. Aggregate only over well-formed dict rows.
    malformed = sum(1 for row in rows if not isinstance(row, dict))
    dict_rows = [row for row in rows if isinstance(row, dict)]

    timings = defaultdict(float)
    gate_counts = Counter()
    drop_off = Counter()
    for row in dict_rows:
        stage = str(row.get("stage") or "unknown")
        if row.get("duration_ms") is not None:
            try:
                timings[stage] += float(row.get("duration_ms") or 0)
            except (TypeError, ValueError):
                pass
        if row.get("event") == "gate_outcome":
            detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
            gate_counts[str(detail.get("gate") or stage)] += 1
        if row.get("event") == "exit":
            drop_off[stage] += 1

    envelope = dict_rows[0].get("detail") if dict_rows and dict_rows[0].get("event") == "reproduction_envelope" else {}
    return {
        "status": "ok",
        "run_id": str(run_id),
        "path": str(path),
        "n_events": len(rows),
        "malformed_rows": malformed,
        "chain_ok": chain_ok,
        "chain_broken_seq": broken_seq,
        "envelope": envelope if isinstance(envelope, dict) else {},
        "stage_timings": dict(sorted(timings.items())),
        "gate_outcomes": dict(sorted(gate_counts.items())),
        "drop_off": dict(sorted(drop_off.items())),
    }
