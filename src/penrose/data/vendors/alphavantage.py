"""Alpha Vantage adapter — CERTIFIED (free key), as-displayed daily equities.

BYO model: set `ALPHAVANTAGE_API_KEY` (free from https://www.alphavantage.co/support/#api-key).
With no key, every Alpha Vantage series is UNAVAILABLE — fail-open, never a crash.

STATUS: wired against the live TIME_SERIES_DAILY endpoint and verified by a key-gated
integration test (tests/test_vendors.py) that fetches a known symbol (SPY) and folds the
default `us_equity_spy` series into the bundle. With no key the test skips cleanly. (Note:
the free tier is rate-limited to ~25 requests/day; the adapter fails open on a rate-limit
note, so the series is simply absent rather than an error.)

Grade = "as_displayed": daily prices as Alpha Vantage displays them, not
point-in-time vintages.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "alphavantage"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://www.alphavantage.co/query"


def available() -> bool:
    """True iff ALPHAVANTAGE_API_KEY is set."""
    return bool(os.environ.get("ALPHAVANTAGE_API_KEY"))


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) or None. Never raises.

    EXPERIMENTAL — best-effort, not live-verified.
    spec keys: symbol (required), field (default "4. close"; "1. open" etc. also
    accepted), outputsize (default "full"), start/end (ISO dates) optionally clip.
    """
    if not available() or not isinstance(spec, dict):
        return None
    symbol = spec.get("symbol") or spec.get("ticker")
    if not symbol:
        return None
    field = spec.get("field", "4. close")
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": str(symbol),
            "outputsize": spec.get("outputsize", "full"),
            "apikey": os.environ["ALPHAVANTAGE_API_KEY"],
        }
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode())
        ts = payload.get("Time Series (Daily)")
        if not isinstance(ts, dict) or not ts:
            return None                       # also covers rate-limit / error notes
        idx, vals = [], []
        for day, fields in ts.items():
            if field not in fields:
                continue
            try:
                vals.append(float(fields[field]))
            except (TypeError, ValueError):
                continue
            idx.append(day)
        if not vals:
            return None
        index = pd.to_datetime(idx, utc=True).normalize()
        s = pd.Series(vals, index=index, name=str(symbol))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        if spec.get("start"):
            s = s[s.index >= pd.Timestamp(spec["start"], tz="UTC")]
        if spec.get("end"):
            s = s[s.index <= pd.Timestamp(spec["end"], tz="UTC")]
        return (s, f"alphavantage-api:{symbol}") if not s.empty else None
    except Exception:  # noqa: BLE001
        return None
