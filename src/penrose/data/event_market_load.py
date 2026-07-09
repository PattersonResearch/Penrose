"""Load settled event-market bracket panels from declared tables.

Causality note (M-2): when a table declares an as-of timestamp column for its precomputed model
inputs, the loader ENFORCES that the stamp is no later than the decision time. But the ``underlying``
model inputs are an opaque dict ({mu, sigma}) and ``entry_price`` is a bare number, so with NO asof
column the loader CANNOT verify they were known at decision time — this is an opt-in check, not a
guarantee. For an UNTRUSTED external table, causality is a provenance/policy responsibility upstream;
the loader is a best-effort boundary, not a proof of no-lookahead. It logs a warning when a table
carries no asof column so the gap is visible.
"""
from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any

import pandas as pd

from .event_market import EVENT_MARKET_COLUMNS, EventMarketPanel, coerce_event_market_frame


class EventMarketDataUnavailable(RuntimeError):
    """Raised for declared event-market data that cannot be loaded."""


_PATH_KEYS = (
    "path",
    "table",
    "table_path",
    "event_market_path",
    "event_market_table",
)
_ASOF_COLUMNS = ("underlying_time", "underlying_asof", "asof_time", "model_time")


def load_event_market(spec: dict, data_dir) -> EventMarketPanel:
    """Build an ``EventMarketPanel`` from a declared CSV/parquet table.

    ``spec`` may declare the table either at top level or under ``event_market``.
    Relative paths are resolved under ``data_dir``. Missing tables and malformed
    declarations raise ``EventMarketDataUnavailable`` with a ``data_unavailable:``
    reason so the pipeline can route to ``needs_data`` instead of crashing.
    """
    path = _declared_path(spec, data_dir)
    if path is None:
        raise EventMarketDataUnavailable("data_unavailable: event_market_table")
    if not path.exists():
        raise EventMarketDataUnavailable(f"data_unavailable: event_market_table {path}")
    try:
        raw = _read_table(path)
    except EventMarketDataUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EventMarketDataUnavailable(
            f"data_unavailable: event_market_table {path}: {type(exc).__name__}: {exc}"
        ) from None

    raw = coerce_event_market_frame(raw, preserve_extra=_ASOF_COLUMNS)
    missing = [c for c in EVENT_MARKET_COLUMNS if c not in raw.columns]
    if missing:
        raise EventMarketDataUnavailable(
            "data_unavailable: event_market_table missing columns " + ", ".join(missing)
        )
    _reject_lookahead_underlying(raw)

    df = raw.loc[:, EVENT_MARKET_COLUMNS].copy()
    df["underlying"] = df["underlying"].map(_parse_underlying)
    name = str(_event_cfg(spec).get("name") or path.stem or "event_market")
    provenance = str(_event_cfg(spec).get("provenance") or f"event_market_table:{path}")
    try:
        return EventMarketPanel(name=name, data=df, provenance=provenance)
    except (TypeError, ValueError) as exc:
        raise EventMarketDataUnavailable(f"data_unavailable: event_market_table invalid: {exc}") from None


def _event_cfg(spec: dict | None) -> dict:
    if not isinstance(spec, dict):
        return {}
    cfg = spec.get("event_market")
    return cfg if isinstance(cfg, dict) else spec


def _declared_path(spec: dict | None, data_dir) -> Path | None:
    cfg = _event_cfg(spec)
    raw = None
    for key in _PATH_KEYS:
        if cfg.get(key):
            raw = cfg.get(key)
            break
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(data_dir) / path
    return path


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise EventMarketDataUnavailable(f"data_unavailable: unsupported event_market_table format {path.suffix}")


def _reject_lookahead_underlying(df: pd.DataFrame) -> None:
    present = [c for c in _ASOF_COLUMNS if c in df.columns]
    if not present:
        import sys
        print(
            "WARNING: event_market table declares no as-of column "
            f"({'/'.join(_ASOF_COLUMNS)}); underlying/entry_price causality is UNVERIFIED — "
            "no-lookahead relies on the source's provenance, not this loader.",
            file=sys.stderr,
        )
        return
    for col in present:
        decision = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
        asof = pd.to_datetime(df[col], utc=True, errors="coerce")
        if decision.isna().any() or asof.isna().any():
            raise EventMarketDataUnavailable(
                f"data_unavailable: event_market_table invalid {col}/decision_time"
            )
        if (asof > decision).any():
            raise EventMarketDataUnavailable(
                f"data_unavailable: event_market_table {col} after decision_time"
            )


def _parse_underlying(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value
