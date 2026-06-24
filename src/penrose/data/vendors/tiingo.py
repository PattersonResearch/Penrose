"""Tiingo adapter — EXPERIMENTAL skeleton, NOT YET CERTIFIED.

BYO model: set `TIINGO_API_KEY` (https://www.tiingo.com). With no key, every Tiingo
series is UNAVAILABLE — fail-open, never a crash.

STATUS: framework-ready but NOT verified against a live key. The fetch() below is a
best-effort implementation of Tiingo's daily end-of-day prices endpoint; treat it as
experimental until a live-key integration test certifies it. Do NOT claim
"certified" for this vendor.

Grade = "as_displayed": adjusted/raw daily EOD prices as Tiingo displays them, not
point-in-time vintages.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "tiingo"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://api.tiingo.com/tiingo/daily"


def available() -> bool:
    """True iff TIINGO_API_KEY is set."""
    return bool(os.environ.get("TIINGO_API_KEY"))


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) or None. Never raises.

    EXPERIMENTAL — best-effort, not live-verified.
    spec keys: symbol/ticker (required), start, end (ISO dates), field (default
    "close"; "adjClose" etc. also accepted as Tiingo column names).
    """
    if not available() or not isinstance(spec, dict):
        return None
    ticker = spec.get("symbol") or spec.get("ticker")
    if not ticker:
        return None
    field = spec.get("field", "close")
    try:
        params = {"token": os.environ["TIINGO_API_KEY"], "format": "json"}
        if spec.get("start"):
            params["startDate"] = spec["start"]
        if spec.get("end"):
            params["endDate"] = spec["end"]
        url = (f"{_BASE}/{urllib.parse.quote(str(ticker))}/prices?"
               f"{urllib.parse.urlencode(params)}")
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1",
                                                   "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rows = json.loads(r.read().decode())
        if not rows:
            return None
        idx, vals = [], []
        for row in rows:
            if field not in row or "date" in row and row["date"] is None:
                continue
            try:
                vals.append(float(row[field]))
            except (TypeError, ValueError):
                continue
            idx.append(row["date"])
        if not vals:
            return None
        index = pd.to_datetime(idx, utc=True).normalize()
        s = pd.Series(vals, index=index, name=str(ticker))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        return (s, f"tiingo-api:{ticker}") if not s.empty else None
    except Exception:  # noqa: BLE001
        return None
