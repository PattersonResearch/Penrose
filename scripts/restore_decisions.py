"""Restore a decision row recovered from an incident record (append-only P0 fix).

decisions.jsonl is append-only; nothing is ever removed (see `_supersede_decision_rows`
in pipeline/run.py). If a `--force` re-run, or any other bug before this fix landed, ever
erased a row from a live decisions.jsonl, the recovered record (verbatim, from wherever it
was rescued: a transcript, a report, a backup) can be re-appended here with an explicit
`restored_from_incident` provenance field, so the audit trail stays honest that this
particular row was RECONSTRUCTED, not originally written by a run.

This script only ever APPENDS to a decisions.jsonl file (default: this clone's
config.DECISIONS_LOG; pass --decisions-log to target another checkout's file). It never
touches any runtime directly. Pointing it at a production decisions.jsonl is an explicit
operator action, not performed automatically by this script.

Usage:
    python scripts/restore_decisions.py --record path/to/recovered_record.json
    python scripts/restore_decisions.py --record record.json --decisions-log /path/to/decisions.jsonl
    echo '{"decision_id": "...", ...}' | python scripts/restore_decisions.py --stdin
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_FIELDS = ("decision_id", "claim_id", "verdict")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_records(args) -> list[dict]:
    raw = sys.stdin.read() if args.stdin else Path(args.record).read_text()
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    raise SystemExit(
        f"restore_decisions: expected a JSON object or array of objects, got {type(parsed).__name__}"
    )


def _validate(record: dict) -> str | None:
    missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
    return f"record missing required field(s): {missing}" if missing else None


def _existing_restored_ids(path: Path) -> set[str]:
    """decision_ids that already carry a restored_from_incident marker in `path` —
    idempotency guard so re-running this script on the same record never duplicates it."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("restored_from_incident") and row.get("decision_id"):
            out.add(str(row["decision_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--record",
                    help="path to a JSON file with one recovered record, or a JSON array of records")
    ap.add_argument("--stdin", action="store_true",
                    help="read the record(s) as JSON from stdin instead of --record")
    ap.add_argument("--decisions-log", default=None,
                    help="decisions.jsonl to append to (default: this clone's config.DECISIONS_LOG)")
    ap.add_argument("--incident", default="decisions-restore",
                    help="short incident identifier stamped into restored_from_incident provenance")
    ap.add_argument("--note", default="",
                    help="optional free-text note stamped alongside the restored row")
    args = ap.parse_args()

    if not args.record and not args.stdin:
        ap.error("one of --record or --stdin is required")

    if args.decisions_log:
        path = Path(args.decisions_log)
    else:
        from penrose import config  # local import: only needed for the default path
        path = Path(config.DECISIONS_LOG)

    try:
        records = _load_records(args)
    except (json.JSONDecodeError, OSError) as e:
        print(f"restore_decisions: could not read record(s): {e}", file=sys.stderr)
        raise SystemExit(1)

    if not records:
        print("restore_decisions: no records to restore; nothing written.")
        return

    written = skipped_dupe = skipped_invalid = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same lock the engine's supersede path holds (decisions.jsonl.lock): the dedup scan
    # and the append happen under one exclusive lock, so a restore can neither interleave
    # with a live run's appends nor race a concurrent restore past the idempotency check.
    lock_path = Path(str(path) + ".lock")
    lock_file = lock_path.open("a+")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    already = _existing_restored_ids(path)
    with path.open("a") as f:  # APPEND ONLY — this script never rewrites or truncates
        for record in records:
            if not isinstance(record, dict):
                print(f"restore_decisions: skipping non-object record: {record!r}", file=sys.stderr)
                skipped_invalid += 1
                continue
            problem = _validate(record)
            if problem:
                print(f"restore_decisions: skipping record ({problem}): {record}", file=sys.stderr)
                skipped_invalid += 1
                continue
            decision_id = str(record["decision_id"])
            if decision_id in already:
                print(f"restore_decisions: {decision_id} already restored "
                      f"(restored_from_incident marker present); skipping duplicate.",
                      file=sys.stderr)
                skipped_dupe += 1
                continue
            row = dict(record)
            row["restored_from_incident"] = args.incident
            row["restored_at"] = _now()
            if args.note:
                row["restoration_note"] = args.note
            f.write(json.dumps(row, default=str) + "\n")
            already.add(decision_id)
            written += 1

    print(f"restore_decisions: wrote {written} row(s) to {path} "
          f"({skipped_dupe} already-restored duplicate(s) skipped, "
          f"{skipped_invalid} invalid record(s) skipped).")
    if written:
        print("NOTE: this appended only to the decisions.jsonl path above. Restoring a separate "
              "production runtime's decisions.jsonl is an explicit operator action, not performed "
              "automatically by this script.")


if __name__ == "__main__":
    main()
