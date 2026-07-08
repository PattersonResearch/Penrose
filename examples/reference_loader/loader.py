"""Reference PENROSE_DATA_DIR loader.

Point PENROSE_DATA_DIR at this directory to expose the local CSV series in
``data/`` through Penrose's scalar daily-series catalog contract.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CATALOG = yaml.safe_load((ROOT / "catalog.yaml").read_text(encoding="utf-8")) or {}
SERIES = CATALOG.get("series", {})


def available() -> list[str]:
    return sorted(SERIES)


def load_series(name: str) -> tuple[pd.Series, str] | None:
    meta = SERIES.get(name)
    if not meta:
        return None
    path = ROOT / str(meta["file"])
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
    except Exception:  # noqa: BLE001 - reference loaders must fail open
        return None
    if "value" not in frame or frame.empty:
        return None
    idx = pd.DatetimeIndex(frame["date"])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    series = pd.Series(frame["value"].astype(float).to_numpy(), index=idx.normalize(), name=name)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series, str(meta.get("provenance", f"reference-loader:{name}"))


def domain_of(name: str) -> str | None:
    meta = SERIES.get(name)
    return None if not meta else meta.get("domain")


def domains() -> list[str]:
    return sorted({m.get("domain") for m in SERIES.values() if m.get("domain")})


def describe(name: str) -> dict:
    meta = dict(SERIES.get(name) or {})
    if not meta:
        return {}
    meta["name"] = name
    return meta


def describe_brief(name: str) -> str:
    meta = SERIES.get(name) or {}
    domain = meta.get("domain", "unknown")
    unit = meta.get("unit", "")
    desc = meta.get("description", f"catalog series {name}")
    return f"{name} [{domain}, {unit}, {meta.get('provenance', 'unknown')}] - {desc}"
