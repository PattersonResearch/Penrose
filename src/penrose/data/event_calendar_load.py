"""Load declared event calendars for event-study claim types."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class EventCalendarDataUnavailable(RuntimeError):
    """Raised for declared event calendars that cannot be loaded."""


@dataclass(frozen=True)
class EventCalendar:
    """A sorted, de-duplicated list of event timestamps."""

    name: str
    dates: pd.DatetimeIndex
    provenance: str
    entity: str | None = None


_PATH_KEYS = (
    "path",
    "table",
    "table_path",
    "calendar_path",
    "calendar_table",
    "event_calendar_path",
    "event_calendar_table",
)
_DATE_COLUMNS = ("date", "event_date", "event_time", "timestamp", "time", "datetime")


def load_event_calendar(spec: dict, data_dir) -> EventCalendar:
    """Build an ``EventCalendar`` from a declared CSV/parquet/json table.

    ``spec`` may declare the calendar either at top level or under
    ``event_calendar``. Relative paths resolve under ``data_dir``. Missing
    declarations raise ``EventCalendarDataUnavailable`` with a
    ``data_unavailable:`` reason so callers route to ``needs_data``.
    """
    cfg = _event_calendar_cfg(spec)
    path = _declared_path(cfg, data_dir)
    if path is None:
        raise EventCalendarDataUnavailable("data_unavailable: event_calendar")
    if not path.exists():
        raise EventCalendarDataUnavailable(f"data_unavailable: event_calendar {path}")
    try:
        raw = _read_table(path)
        dates = _event_dates(raw, cfg)
    except EventCalendarDataUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EventCalendarDataUnavailable(
            f"data_unavailable: event_calendar {path}: {type(exc).__name__}: {exc}"
        ) from None
    if len(dates) == 0:
        raise EventCalendarDataUnavailable(f"data_unavailable: event_calendar empty {path}")
    name = str(cfg.get("name") or path.stem or "event_calendar")
    provenance = str(cfg.get("provenance") or f"event_calendar_table:{path}")
    entity = str(cfg.get("entity") or "").strip() or None
    return EventCalendar(name=name, dates=dates, provenance=provenance, entity=entity)


def _event_calendar_cfg(spec: dict | None) -> dict:
    if not isinstance(spec, dict):
        return {}
    cfg = spec.get("event_calendar")
    if isinstance(cfg, dict):
        return dict(cfg)
    if cfg:
        return {"path": str(cfg)}
    return dict(spec)


def _declared_path(cfg: dict, data_dir) -> Path | None:
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
    raise EventCalendarDataUnavailable(
        f"data_unavailable: unsupported event_calendar format {path.suffix}"
    )


def _column(cfg: dict, columns, candidates: tuple[str, ...]) -> str | None:
    explicit = str(cfg.get("date_col") or cfg.get("event_date_col") or "").strip()
    if explicit:
        return explicit if explicit in columns else None
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _event_dates(raw: pd.DataFrame, cfg: dict) -> pd.DatetimeIndex:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DatetimeIndex([], tz="UTC")
    date_col = _column(cfg, raw.columns, _DATE_COLUMNS)
    if date_col is None:
        first = raw.columns[0]
        parsed = pd.to_datetime(raw[first], utc=True, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            date_col = first
    if date_col is None:
        raise EventCalendarDataUnavailable("data_unavailable: event_calendar missing date column")
    dates = pd.to_datetime(raw[date_col], utc=True, errors="coerce").dropna()
    if dates.empty:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(dates).sort_values().unique()


def event_calendar_declared(spec: dict | None) -> bool:
    """Return True when a spec declares an event-calendar table/path."""
    cfg = _event_calendar_cfg(spec)
    return any(bool(str(cfg.get(k) or "").strip()) for k in _PATH_KEYS)
