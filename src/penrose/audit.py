"""Hash-chained per-run audit event logs.

Audit events are observability only: they must never feed back into verdict
logic. Each emitted JSONL row includes wall-clock fields (`ts`, `duration_ms`)
and repeats them under `_volatile` for clarity, but the event content hash is
computed with `hash` omitted, `_volatile` omitted, and top-level `ts` /
`duration_ms` normalized to ``None``. This makes the hash chain reproducible
across reruns with identical semantic inputs even when clocks and timings
differ.
"""
from __future__ import annotations

import hashlib
import json
import platform as _platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from . import config


AUDIT_EVENT_FIELDS = (
    "seq", "run_id", "ts", "prev_hash", "hash", "principal", "stage", "event",
    "inputs_digest", "outputs_digest", "duration_ms", "detail",
)

CONFIG_FINGERPRINT_KEYS = (
    "DSR_DECISION",
    "DEFLATION_PRIOR",
    "BOOTSTRAP",
    "PERMUTATION",
    "REGIME_FRAGILITY",
    "FRAGILITY_GATE",
    "WALK_FORWARD",
    "CPCV",
    "ROBUSTNESS_GATES",
    "TAIL_RISK_GATE",
    "POWER",
    "IMPLAUSIBILITY",
)

_ZERO_HASH = "0" * 64
_MAX_DETAIL_ITEMS = 64
_MAX_STRING = 2048


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def canonical_json_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _bounded(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(value)[:_MAX_STRING]
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "...[truncated]"
    if isinstance(value, dict):
        out = {}
        for i, key in enumerate(sorted(value.keys(), key=str)):
            if i >= _MAX_DETAIL_ITEMS:
                out["_truncated_items"] = max(0, len(value) - _MAX_DETAIL_ITEMS)
                break
            out[str(key)] = _bounded(value[key], depth + 1)
        return out
    if isinstance(value, (set, frozenset)):
        # A set's iteration order is not stable across processes (PYTHONHASHSEED),
        # which would leak into the content hash. Sort deterministically by str.
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        seq = list(value)
        out = [_bounded(item, depth + 1) for item in seq[:_MAX_DETAIL_ITEMS]]
        if len(seq) > _MAX_DETAIL_ITEMS:
            out.append({"_truncated_items": len(seq) - _MAX_DETAIL_ITEMS})
        return out
    return value


def _hash_payload(event: dict) -> dict:
    payload = dict(event)
    payload.pop("hash", None)
    payload.pop("_volatile", None)
    payload["ts"] = None
    payload["duration_ms"] = None
    return payload


def event_hash(event: dict) -> str:
    return canonical_json_digest(_hash_payload(event))


def config_fingerprint() -> str:
    payload = {key: getattr(config, key, None) for key in CONFIG_FINGERPRINT_KEYS}
    return canonical_json_digest(payload)


def platform_tuple() -> dict:
    def _version(package: str) -> str:
        try:
            return metadata.version(package)
        except Exception:  # noqa: BLE001 - metadata varies by installation
            return "unknown"

    return {
        "os": _platform.platform(),
        "machine": _platform.machine() or "unknown",
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
    }


class AuditLog:
    """Append-only hash-chain writer for one Penrose run.

    Emission methods fail open: serialization and filesystem errors are logged to
    stderr at most and never propagate into caller control flow.
    """

    def __init__(self, run_id: str, principal: str, path: Path):
        self.run_id = str(run_id)
        self.principal = str(principal or "cli")
        self.path = Path(path)
        self._seq = 0
        self._head_hash = _ZERO_HASH

    def envelope(
        self,
        version: str,
        config_fingerprint: str,
        data_sources: Any,
        seeds: Any,
        platform: Any,
        reproducibility_class: str,
    ) -> None:
        detail = {
            "version": version,
            "config_fingerprint": config_fingerprint,
            "data_sources": data_sources,
            "seeds": seeds,
            "platform": platform,
            "reproducibility_class": reproducibility_class,
        }
        self._append("P0", "reproduction_envelope", detail=detail)

    def stage(
        self,
        stage: str,
        event: str,
        *,
        inputs: Any = None,
        outputs: Any = None,
        duration_ms: int | float | None = None,
        detail: Any = None,
    ) -> None:
        self._append(
            str(stage),
            str(event),
            inputs_digest=canonical_json_digest(inputs) if inputs is not None else None,
            outputs_digest=canonical_json_digest(outputs) if outputs is not None else None,
            duration_ms=duration_ms,
            detail=detail,
        )

    def head_hash(self) -> str:
        return self._head_hash

    def close(self) -> None:
        return None

    def _append(
        self,
        stage: str,
        event: str,
        *,
        inputs_digest: str | None = None,
        outputs_digest: str | None = None,
        duration_ms: int | float | None = None,
        detail: Any = None,
    ) -> None:
        try:
            volatile = {"ts": _now(), "duration_ms": duration_ms}
            row = {
                "seq": self._seq,
                "run_id": self.run_id,
                "ts": volatile["ts"],
                "prev_hash": self._head_hash,
                "hash": None,
                "principal": self.principal,
                "stage": stage,
                "event": event,
                "inputs_digest": inputs_digest,
                "outputs_digest": outputs_digest,
                "duration_ms": duration_ms,
                "detail": _bounded(detail or {}),
                "_volatile": volatile,
            }
            row["hash"] = event_hash(row)
            _append_jsonl(self.path, row)
            self._head_hash = row["hash"]
            self._seq += 1
        except Exception as exc:  # noqa: BLE001 - audit must fail open
            print(f"[penrose] audit emission failed: {exc}", file=sys.stderr)


def verify_events(rows: list[dict]) -> tuple[bool, int | None]:
    prev = _ZERO_HASH
    for expected_seq, row in enumerate(rows):
        if not isinstance(row, dict):  # tampered / truncated / foreign line -> chain is broken, not a crash
            return False, expected_seq
        if row.get("seq") != expected_seq:
            return False, expected_seq
        if row.get("prev_hash") != prev:
            return False, expected_seq
        if event_hash(row) != row.get("hash"):
            return False, expected_seq
        prev = str(row.get("hash") or "")
    return True, None
