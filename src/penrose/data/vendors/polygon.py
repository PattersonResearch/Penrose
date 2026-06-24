"""Polygon.io adapter — EXPERIMENTAL skeleton, NOT YET CERTIFIED.

BYO model: set `POLYGON_API_KEY` (https://polygon.io). With no key, every Polygon
series is UNAVAILABLE — fail-open, never a crash.

STATUS: framework-ready but NOT verified against a live key. The fetch() below is a
best-effort implementation of Polygon's aggregates (daily bars) endpoint; treat it
as experimental until a live-key integration test certifies it. Do NOT claim
"certified" for this vendor.

Grade = "as_displayed": daily OHLC bars as Polygon displays them (split/dividend
adjustment per `adjusted`), not point-in-time vintages.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "polygon"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://api.polygon.io/v2/aggs/ticker"


def available() -> bool:
    """True iff POLYGON_API_KEY is set. Uses stdlib urllib (no extra import gate)."""
    return bool(os.environ.get("POLYGON_API_KEY"))


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) or None. Never raises.

    EXPERIMENTAL — best-effort, not live-verified.
    spec keys: symbol/ticker (required), start, end (ISO dates), field (default
    "close" -> Polygon's "c"), adjusted (default True).
    """
    if not available() or not isinstance(spec, dict):
        return None
    ticker = spec.get("symbol") or spec.get("ticker")
    start = spec.get("start")
    end = spec.get("end")
    if not (ticker and start and end):
        return None
    field_map = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    col = field_map.get(spec.get("field", "close"), "c")
    try:
        adjusted = "true" if spec.get("adjusted", True) else "false"
        params = {"adjusted": adjusted, "sort": "asc", "limit": 50000,
                  "apiKey": os.environ["POLYGON_API_KEY"]}
        url = (f"{_BASE}/{urllib.parse.quote(str(ticker))}/range/1/day/"
               f"{start}/{end}?{urllib.parse.urlencode(params)}")
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode())
        rows = payload.get("results") or []
        if not rows:
            return None
        idx = pd.to_datetime([row["t"] for row in rows], unit="ms", utc=True).normalize()
        vals = [float(row[col]) for row in rows if col in row]
        if len(vals) != len(idx):
            return None
        s = pd.Series(vals, index=idx, name=str(ticker))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        return (s, f"polygon-api:{ticker}") if not s.empty else None
    except Exception:  # noqa: BLE001
        return None
