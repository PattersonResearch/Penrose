"""pysystemtrade local futures adapter: BYO adjusted-price CSVs, as-displayed.

This adapter reads a user's local pysystemtrade-shaped data directory. It never
uses the network and never raises into bundle construction: absent directories,
missing instruments, and malformed CSVs all return unavailable/None.

Grade = "as_displayed": pysystemtrade adjusted futures prices are back-adjusted
continuous series, not point-in-time contract data.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Instrument codes are simple symbols (SP500, US10, CRUDE_W). Restrict to these so a crafted
# instrument name (e.g. "../../../secret") cannot traverse out of the data dir and read an
# arbitrary file. This is the security boundary; the path join alone is not enough.
_SAFE_INSTRUMENT = re.compile(r"[A-Za-z0-9_]+")

import pandas as pd

from ..granularity import check_granularity

NAME = "pysystemtrade"
PROVENANCE_GRADE = "as_displayed"


def _data_dir() -> Path:
    """Configured pysystemtrade root, from PENROSE_FUTURES_DIR or PYSYS_DIR."""
    return Path(os.environ.get("PENROSE_FUTURES_DIR") or os.environ.get("PYSYS_DIR") or "")


def _prices_dir() -> Path:
    return _data_dir() / "data" / "futures" / "adjusted_prices_csv"


def available() -> bool:
    """True iff the configured adjusted_prices_csv directory exists and is non-empty."""
    try:
        if not (os.environ.get("PENROSE_FUTURES_DIR") or os.environ.get("PYSYS_DIR")):
            return False
        d = _prices_dir()
        return d.is_dir() and any(d.iterdir())
    except Exception:  # noqa: BLE001 - a broken local dir is just unavailable
        return False


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) for a futures instrument.

    spec keys:
      instrument (required): pysystemtrade instrument code, e.g. "SP500".
      field      (optional): CSV value column; default "price".

    Raw rows are always checked for granularity and resampled to daily close
    before returning, so intraday adjusted-price files cannot enter a verdict as
    intraday bars through this adapter.
    """
    if not available() or not isinstance(spec, dict):
        return None
    instrument = str(spec.get("instrument") or "").strip()
    if not instrument or not _SAFE_INSTRUMENT.fullmatch(instrument):
        return None  # reject path-traversal / unsafe instrument names
    field = str(spec.get("field") or "price").strip() or "price"
    try:
        path = _prices_dir() / f"{instrument}.csv"
        if not path.exists() or not path.is_file():
            return None
        df = pd.read_csv(path)
        if df.empty or "DATETIME" not in df.columns or field not in df.columns:
            return None

        idx = pd.to_datetime(df["DATETIME"], utc=True, errors="coerce")
        vals = pd.to_numeric(df[field], errors="coerce")
        raw = pd.Series(vals.to_numpy(), index=idx, name=instrument).dropna()
        raw = raw[~raw.index.isna()]
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        if raw.empty:
            return None

        granularity = check_granularity(raw, expected="daily")
        daily = raw.resample("1D").last().dropna()
        daily = daily[~daily.index.duplicated(keep="last")].sort_index()
        if daily.empty:
            return None
        daily.name = instrument

        notes = ["back-adjusted continuous"]
        actual = granularity.get("actual")
        if actual and actual != "daily" and actual != "unknown":
            notes.append(f"resampled {actual}->daily")
        else:
            notes.append("resampled daily close")
        return daily, f"pysystemtrade-adjusted ({'; '.join(notes)})"
    except Exception:  # noqa: BLE001 - local data errors fail open
        return None
