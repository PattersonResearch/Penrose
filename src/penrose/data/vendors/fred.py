"""FRED (St. Louis Fed) adapter — CERTIFIED, free, as-displayed.

BYO model: set `FRED_API_KEY` (free from https://fredaccount.stlouisfed.org/apikeys).
With no key, every FRED series is simply UNAVAILABLE — fail-open, never a crash.

This is the one CERTIFIED adapter in the framework: it is wired against the live
FRED HTTP API and verified by a key-gated integration test (tests/test_vendors.py)
that fetches a known series (DGS10, the 10y treasury) and asserts a non-empty
Series. With no key the test skips cleanly.

Grade = "as_displayed": FRED returns values as currently displayed/revised, not a
point-in-time vintage. Honest, but not survivorship-aware (use ALFRED vintages for
that — out of scope here).

Implemented with stdlib `urllib` (no new hard dependency) against
https://api.stlouisfed.org/fred/series/observations.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "fred"
PROVENANCE_GRADE = "as_displayed"

_BASE = "https://api.stlouisfed.org/fred/series/observations"


def available() -> bool:
    """True iff FRED_API_KEY is set. urllib + pandas are stdlib/already-required."""
    return bool(os.environ.get("FRED_API_KEY"))


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) for a FRED series, or None.

    spec keys:
      series_id (required) — FRED series id, e.g. "DGS10".
      start (optional)     — ISO date "YYYY-MM-DD" lower bound (observation_start).
      end   (optional)     — ISO date upper bound (observation_end).

    Never raises — any error (no key, network, bad id) returns None.
    """
    if not available():
        return None
    if not isinstance(spec, dict):
        return None
    series_id = spec.get("series_id") or spec.get("symbol")
    if not series_id:
        return None
    try:
        params = {
            "series_id": series_id,
            "api_key": os.environ["FRED_API_KEY"],
            "file_type": "json",
        }
        if spec.get("start"):
            params["observation_start"] = spec["start"]
        if spec.get("end"):
            params["observation_end"] = spec["end"]
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode())
        obs = payload.get("observations") or []
        if not obs:
            return None
        idx, vals = [], []
        for o in obs:
            v = o.get("value")
            if v in (None, ".", ""):          # FRED marks missing days with "."
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
            idx.append(o.get("date"))
        if not vals:
            return None
        index = pd.to_datetime(idx, utc=True).normalize()
        s = pd.Series(vals, index=index, name=str(series_id))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        if s.empty:
            return None
        return s, f"fred-api:{series_id}"
    except Exception:  # noqa: BLE001 — a data-fetch error never breaks a run
        return None
