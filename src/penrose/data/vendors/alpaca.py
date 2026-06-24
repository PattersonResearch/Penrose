"""Alpaca Market Data adapter — EXPERIMENTAL skeleton, NOT YET CERTIFIED.

BYO model: set BOTH `ALPACA_API_KEY` and `ALPACA_API_SECRET` (https://alpaca.markets).
With either missing, every Alpaca series is UNAVAILABLE — fail-open, never a crash.

STATUS: framework-ready but NOT verified against a live key. The fetch() below is a
best-effort implementation of Alpaca's v2 stock daily bars endpoint; treat it as
experimental until a live-key integration test certifies it. Do NOT claim
"certified" for this vendor.

Grade = "as_displayed": daily OHLC bars as Alpaca displays them, not point-in-time
vintages.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "alpaca"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://data.alpaca.markets/v2/stocks"


def available() -> bool:
    """True iff BOTH ALPACA_API_KEY and ALPACA_API_SECRET are set."""
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET"))


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) or None. Never raises.

    EXPERIMENTAL — best-effort, not live-verified.
    spec keys: symbol (required), start, end (ISO dates), field (default "close" ->
    Alpaca's "c"), feed (default "iex").
    """
    if not available() or not isinstance(spec, dict):
        return None
    symbol = spec.get("symbol") or spec.get("ticker")
    if not symbol:
        return None
    field_map = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    col = field_map.get(spec.get("field", "close"), "c")
    try:
        headers = {
            "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"],
            "User-Agent": "penrose/0.1",
        }
        rows = []
        page_token = None
        for _ in range(50):                   # bounded pagination
            params = {"timeframe": "1Day", "limit": 10000,
                      "feed": spec.get("feed", "iex")}
            if spec.get("start"):
                params["start"] = spec["start"]
            if spec.get("end"):
                params["end"] = spec["end"]
            if page_token:
                params["page_token"] = page_token
            url = (f"{_BASE}/{urllib.parse.quote(str(symbol))}/bars?"
                   f"{urllib.parse.urlencode(params)}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode())
            rows.extend(payload.get("bars") or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        if not rows:
            return None
        idx = pd.to_datetime([row["t"] for row in rows], utc=True).normalize()
        vals = [float(row[col]) for row in rows if col in row]
        if len(vals) != len(idx):
            return None
        s = pd.Series(vals, index=idx, name=str(symbol))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        return (s, f"alpaca-api:{symbol}") if not s.empty else None
    except Exception:  # noqa: BLE001
        return None
