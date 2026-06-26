"""Propose-only principle store.

This module is deliberately separate from ``PRINCIPLES_LOG`` and the trusted
BrainStore. Rows here are advisory proposals with ``status="proposed"``; P9
human approval remains the only promotion path.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Any

from . import config


_DEDUP_FIELDS = ("domain", "kill_reason", "statement")


def _proposal_key(row: dict) -> tuple[str, str, str]:
    return tuple(str(row.get(k) or "").strip() for k in _DEDUP_FIELDS)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return []
            if value.get("status") != "proposed":
                value = dict(value)
                value["status"] = "proposed"
            rows.append(value)
    except Exception:  # noqa: BLE001 - fail open on missing/corrupt/unreadable store
        return []
    return rows


def read_proposals(path: str | Path | None = None) -> list[dict]:
    """Read advisory proposals only.

    This is a READ-ONLY API for the propose-only store. It never writes
    ``config.PRINCIPLES_LOG`` and never writes the trusted BrainStore; proposal
    promotion still requires the existing human P9 review_queue -> approve path.
    Missing, empty, or corrupt stores fail open as ``[]``.
    """
    store = Path(path) if path is not None else config.PROPOSALS_LOG
    return _read_rows(store)


def _normalize(row: dict[str, Any], *, source: str, ts: str) -> dict:
    out = dict(row)
    out["source"] = str(out.get("source") or source)
    out["ts"] = str(out.get("ts") or ts)
    out["status"] = "proposed"
    if "supporting_kills" not in out and "supporting" in out:
        out["supporting_kills"] = list(out.get("supporting") or [])
    return out


def write_proposals(
    rows: Iterable[dict[str, Any]],
    *,
    path: str | Path | None = None,
    source: str = "distilled",
    ts: str | None = None,
) -> list[dict]:
    """Append/dedup proposed principles in the propose-only store.

    Existing and new rows are deduped by ``(domain, kill_reason, statement)``.
    The write is protected by a sibling lock file and committed with
    tmp+replace. On write/read failure this fails open and returns the currently
    readable rows or ``[]``; it never touches approved principle storage.
    """
    store = Path(path) if path is not None else config.PROPOSALS_LOG
    # P9 firewall, provable by construction: the propose-only store may NEVER be the
    # approved principle ledger or trusted brain state. A caller who points `path` at the
    # approved store is refused outright (it could otherwise clobber human-approved rows).
    if store.resolve() == Path(config.PRINCIPLES_LOG).resolve():
        raise ValueError(
            "write_proposals refuses to write the approved PRINCIPLES_LOG; "
            "proposals are advisory and promotion goes through human P9 review")
    stamp = ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    incoming = [_normalize(r, source=source, ts=stamp) for r in rows if isinstance(r, dict)]
    if not incoming:
        return read_proposals(store)

    store.parent.mkdir(parents=True, exist_ok=True)
    lock_path = store.with_suffix(store.suffix + ".lock")
    tmp_path = store.with_suffix(store.suffix + f".{os.getpid()}.tmp")
    try:
        import fcntl

        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            merged: dict[tuple[str, str, str], dict] = {}
            for row in _read_rows(store):
                key = _proposal_key(row)
                if all(key):
                    merged[key] = row
            for row in incoming:
                key = _proposal_key(row)
                if all(key):
                    merged[key] = row
            ordered = sorted(
                merged.values(),
                key=lambda r: (
                    str(r.get("domain") or ""),
                    str(r.get("kill_reason") or ""),
                    str(r.get("statement") or ""),
                ),
            )
            tmp_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered))
            os.replace(tmp_path, store)
            return ordered
    except Exception:  # noqa: BLE001
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return read_proposals(store)

