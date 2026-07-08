"""Propose-only principle store.

This module is deliberately separate from ``PRINCIPLES_LOG`` and the trusted
BrainStore. Rows here are advisory proposals with ``status="proposed"``; P9
human approval remains the only promotion path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Any

from . import config


def _proposal_key(row: dict) -> tuple[str, ...]:
    principle_id = str(row.get("principle_id") or "").strip()
    if principle_id:
        return (principle_id,)
    return (
        str(row.get("domain") or row.get("kill_domain") or "").strip(),
        str(row.get("kill_reason") or "").strip(),
        str(row.get("kind") or "recurrence").strip(),
    )


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        text = path.read_text()
    except Exception:  # noqa: BLE001 - file-level read failure fails open
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # CR4-1: skip a SINGLE corrupt/non-dict line rather than discarding the whole store — symmetric
        # with learning._read_jsonl (CR2-1). Otherwise one bad line in principles_proposed.jsonl would,
        # via write_proposals' merge, silently drop the OTHER-source rows (contrastive/manual) it should
        # have preserved.
        try:
            value = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(value, dict):
            continue
        if value.get("status") != "proposed":
            value = dict(value)
            value["status"] = "proposed"
        rows.append(value)
    return rows


def read_proposals(path: str | Path | None = None) -> list[dict]:
    """Read advisory proposals only.

    This is a READ-ONLY API for the propose-only store. It never writes
    ``config.PRINCIPLES_LOG`` and never writes the trusted BrainStore; proposal
    promotion still requires the existing human P9 review_queue -> approve path.
    Missing, empty, or corrupt stores fail open as ``[]``.
    """
    store = Path(path) if path is not None else config.PRINCIPLES_PROPOSED
    return _read_rows(store)


def _normalize(row: dict[str, Any], *, source: str, ts: str) -> dict:
    out = dict(row)
    out["source"] = str(out.get("source") or source)
    if ts or out.get("ts"):
        out["ts"] = str(out.get("ts") or ts)
    out["status"] = "proposed"
    if "supporting_kills" not in out and "supporting" in out:
        out["supporting_kills"] = list(out.get("supporting") or [])
    return out


def _targets_approved_ledger(store: Path) -> bool:
    """True if `store` would write the P9-approved principle ledger — under ANY alias.

    CR-1: `Path.resolve()` normalizes `..`/`.` and follows symlinks but does NOT canonicalize CASE.
    On a case-insensitive filesystem (macOS APFS default, Windows NTFS) a case-variant of the approved
    filename (`Principles.jsonl`) resolves to the SAME directory entry as `principles.jsonl` yet compares
    unequal by string, so a naive `resolve() ==` guard could be bypassed to clobber approved rows. We
    also catch same-inode aliases (hardlinks) via `samefile`. Legitimate proposal stores never share a
    name/inode with the approved ledger, so the extra strictness has no false-positive cost.
    """
    approved = Path(config.PRINCIPLES_LOG)
    try:
        sr, ar = store.resolve(), approved.resolve()
    except OSError:
        return False
    if sr == ar or str(sr).casefold() == str(ar).casefold():
        return True
    try:
        return store.exists() and approved.exists() and os.path.samefile(store, approved)
    except OSError:
        return False


def write_proposals(
    rows: Iterable[dict[str, Any]],
    *,
    path: str | Path | None = None,
    source: str = "distilled",
    ts: str | None = None,
    replace_source: bool = False,
) -> list[dict]:
    """Append/dedup proposed principles in the propose-only store.

    Existing and new rows are deduped by stable ``principle_id`` where present,
    with a legacy fallback to ``(domain, kill_reason, kind)``. When
    ``replace_source`` is true, existing rows from the same source that are not
    present in ``rows`` are removed; this lets the deterministic distill command
    update stale proposal counts without accumulating obsolete versions.
    The write is protected by a sibling lock file and committed with
    tmp+replace. On write/read failure this fails open and returns the currently
    readable rows or ``[]``; it never touches approved principle storage. The
    default store is ``config.PRINCIPLES_PROPOSED`` (``reports/principles_proposed.jsonl``),
    a human-review surface only. Promotion into the trusted brain is exclusively
    the human P9 approval path.
    """
    store = Path(path) if path is not None else config.PRINCIPLES_PROPOSED
    # P9 firewall, provable by construction: the propose-only store may NEVER be the
    # approved principle ledger or trusted brain state. A caller who points `path` at the
    # approved store is refused outright (it could otherwise clobber human-approved rows).
    if _targets_approved_ledger(store):
        raise ValueError(
            "write_proposals refuses to write the approved PRINCIPLES_LOG; "
            "proposals are advisory and promotion goes through human P9 review")
    stamp = ts or ""
    incoming = [_normalize(r, source=source, ts=stamp) for r in rows if isinstance(r, dict)]
    # An empty distill with replace_source MUST still purge this source's stale rows (CR-2): the
    # whole point of replace_source is to drop proposals whose supporting kills left the corpus.
    # Only short-circuit when there is genuinely nothing to do (no rows AND no purge requested).
    if not incoming and not replace_source:
        return read_proposals(store)
    incoming_keys = {_proposal_key(row) for row in incoming}

    store.parent.mkdir(parents=True, exist_ok=True)
    lock_path = store.with_suffix(store.suffix + ".lock")
    tmp_path = store.with_suffix(store.suffix + f".{os.getpid()}.tmp")
    try:
        import fcntl

        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            merged: dict[tuple[str, ...], dict] = {}
            for row in _read_rows(store):
                key = _proposal_key(row)
                if replace_source and row.get("source") == source and key not in incoming_keys:
                    continue
                if all(key):
                    merged[key] = row
            for row in incoming:
                key = _proposal_key(row)
                if all(key):
                    merged[key] = row
            ordered = sorted(
                merged.values(),
                key=lambda r: (
                    str(r.get("principle_id") or ""),
                    str(r.get("domain") or ""),
                    str(r.get("kill_reason") or ""),
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
