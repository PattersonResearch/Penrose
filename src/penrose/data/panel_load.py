"""Load declared date x entity panels for cross-sectional claim types."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .panel import Panel


class PanelDataUnavailable(RuntimeError):
    """Raised for declared panel data that cannot be loaded."""


_PATH_KEYS = ("path", "table", "table_path", "panel_path", "panel_table")
_DATE_COLUMNS = ("date", "timestamp", "time", "datetime")
_ENTITY_COLUMNS = ("entity", "asset", "ticker", "symbol", "permno", "id")
_VALUE_COLUMNS = ("value", "return", "returns", "characteristic", "signal")


def load_cross_sectional_sort_panels(spec: dict, data_dir) -> tuple[Panel, Panel]:
    """Load the returns and characteristic panels declared by a sort spec."""
    inputs = spec.get("panel_inputs") if isinstance(spec, dict) else None
    if not isinstance(inputs, dict):
        raise PanelDataUnavailable("data_unavailable: panel_inputs.returns, panel_inputs.characteristic")
    returns = load_panel_input(inputs.get("returns"), data_dir, role="returns", kind="return")
    characteristic = load_panel_input(
        inputs.get("characteristic"),
        data_dir,
        role="characteristic",
        kind="characteristic",
    )
    return returns, characteristic


def load_panel_input(declaration: Any, data_dir, *, role: str, kind: str) -> Panel:
    """Build a ``Panel`` from one declared CSV/parquet/json table.

    Declarations may be a string path/table or a dict with ``path``/``table`` plus
    optional ``date_col``, ``entity_col``, ``value_col``, ``name``, ``unit``, and
    ``provenance``. Wide tables use one date column and entity columns; long tables
    use date/entity/value columns.
    """
    cfg = _panel_cfg(declaration)
    path = _declared_path(cfg, data_dir)
    if path is None:
        raise PanelDataUnavailable(f"data_unavailable: panel_inputs.{role}")
    if not path.exists():
        raise PanelDataUnavailable(f"data_unavailable: panel_inputs.{role} {path}")
    if kind == "return":
        _require_survivorship_corrected(cfg, role)
    try:
        raw = _read_table(path)
        data = _panel_frame(raw, cfg, role)
        name = str(cfg.get("name") or role or path.stem)
        provenance = str(cfg.get("provenance") or f"panel_table:{path}")
        if kind == "return":
            provenance += "; survivorship=corrected"
        return Panel(
            name=name,
            data=data,
            provenance=provenance,
            kind=kind,
            unit=str(cfg.get("unit") or ("return" if kind == "return" else "")),
            note=str(cfg.get("note") or "declared panel input"),
        )
    except PanelDataUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PanelDataUnavailable(
            f"data_unavailable: panel_inputs.{role} {path}: {type(exc).__name__}: {exc}"
        ) from None


def _panel_cfg(declaration: Any) -> dict:
    if isinstance(declaration, dict):
        return dict(declaration)
    if declaration:
        return {"path": str(declaration)}
    return {}


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
    raise PanelDataUnavailable(f"data_unavailable: unsupported panel table format {path.suffix}")


def _column(cfg: dict, key: str, columns, candidates: tuple[str, ...]) -> str | None:
    explicit = str(cfg.get(key) or "").strip()
    if explicit:
        return explicit if explicit in columns else None
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _panel_frame(raw: pd.DataFrame, cfg: dict, role: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    date_col = _column(cfg, "date_col", raw.columns, _DATE_COLUMNS)
    entity_col = _column(cfg, "entity_col", raw.columns, _ENTITY_COLUMNS)
    value_col = _column(cfg, "value_col", raw.columns, _VALUE_COLUMNS)
    if date_col and entity_col and value_col:
        tmp = raw[[date_col, entity_col, value_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
        tmp[entity_col] = tmp[entity_col].astype(str)
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col, entity_col])
        if tmp.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        return tmp.pivot_table(
            index=date_col,
            columns=entity_col,
            values=value_col,
            aggfunc="last",
            sort=True,
        )
    if not date_col:
        first = raw.columns[0]
        parsed = pd.to_datetime(raw[first], utc=True, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            date_col = first
    if not date_col:
        raise PanelDataUnavailable(f"data_unavailable: panel_inputs.{role} missing date column")
    df = raw.copy()
    idx = pd.to_datetime(df.pop(date_col), utc=True, errors="coerce")
    df.index = idx
    df = df.loc[df.index.notna()]
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    return df


def _require_survivorship_corrected(cfg: dict, role: str) -> None:
    value = str(cfg.get("survivorship") or "").strip().lower()
    if not value:
        raise PanelDataUnavailable(
            f"data_unavailable: panel_inputs.{role} survivorship must be declared corrected"
        )
    if value != "corrected":
        raise PanelDataUnavailable(
            f"data_unavailable: panel_inputs.{role} survivorship={value} is not corrected"
        )
