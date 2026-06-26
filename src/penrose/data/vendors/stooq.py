"""Stooq keyless daily OHLCV adapter — free, as-displayed.

Stooq serves long-history daily CSV bars without an API key. This adapter is
intentionally tiny and fail-open: any network, parse, or bad-symbol error returns
None and never raises into bundle construction.
"""
from __future__ import annotations

from io import StringIO
import urllib.parse
import urllib.request

import pandas as pd

NAME = "stooq"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://stooq.com/q/d/l/"
_FIELDS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def available() -> bool:
    """Stooq is keyless; network failure is handled inside fetch()."""
    return True


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) for a Stooq symbol, or None.

    spec keys:
      symbol (required) — Stooq symbol, e.g. "spy.us" or "^spx".
      field  (optional) — one of open/high/low/close/volume; default close.
      start/end (optional) — ISO date bounds, filtered locally.

    Never raises — any error returns None.
    """
    if not isinstance(spec, dict):
        return None
    symbol = str(spec.get("symbol") or "").strip().lower()
    if not symbol:
        return None
    field = str(spec.get("field") or "close").strip().lower()
    column = _FIELDS.get(field, spec.get("field") or "Close")
    try:
        params = {"s": symbol, "i": "d"}
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
        df = pd.read_csv(StringIO(text))
        if df.empty or "Date" not in df.columns or column not in df.columns:
            return None
        idx = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.normalize()
        vals = pd.to_numeric(df[column], errors="coerce")
        s = pd.Series(vals.to_numpy(), index=idx, name=symbol).dropna()
        s = s[~s.index.isna()]
        if spec.get("start"):
            s = s[s.index >= pd.Timestamp(spec["start"], tz="UTC")]
        if spec.get("end"):
            s = s[s.index <= pd.Timestamp(spec["end"], tz="UTC")]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            return None
        return s, f"stooq-csv:{symbol}:{column}"
    except Exception:  # noqa: BLE001 — a data-fetch error never breaks a run
        return None
