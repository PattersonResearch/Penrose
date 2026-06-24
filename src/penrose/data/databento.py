"""Databento BYO adapter — fetch point-in-time market data as penrose daily Series.

BYO model (mirrors BYO LLM tokens): the USER brings their own Databento account. Set
`DATABENTO_API_KEY` and `pip install databento`. With no key or package installed, every
Databento series is simply UNAVAILABLE — it never enters the bundle, the claim falls to an
honest `needs_data`, and a run never crashes. No license burden on penrose: the user's own
data entitlement covers their use.

Why this matters: Databento data is point-in-time and survivorship-aware, which directly
attacks LOOK-AHEAD — the bias penrose's whole robustness stack exists to catch. A kill
produced on point-in-time data is worth far more than one on free daily bars, so each series
carries provenance the (future) corpus can weight by.

Results are cached locally under `.databento_cache/`, keyed by the exact request, so re-runs
and the eval/dev loops never re-bill the user.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).resolve().parent / ".databento_cache"


def available() -> bool:
    """True iff a key is set AND the databento package is importable."""
    if not os.environ.get("DATABENTO_API_KEY"):
        return False
    try:
        import databento  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _cache_path(parts) -> Path:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]
    return _CACHE / f"{h}.parquet"


def fetch_daily(dataset: str, symbol: str, start: str, end: str, *,
                schema: str = "ohlcv-1d", field: str = "close",
                stype_in: str = "raw_symbol"):
    """Return (daily tz-naive pd.Series, provenance) for a Databento request, or None.

    dataset/symbol/schema/stype_in are Databento's own identifiers and depend on the user's
    entitlements — e.g. dataset 'GLBX.MDP3' (CME), continuous front-month 'ES.c.0' with
    stype_in='continuous', schema 'ohlcv-1d'. `field` is the OHLCV column to extract.
    Never raises — any error (no key, no entitlement, network) returns None.
    """
    if not os.environ.get("DATABENTO_API_KEY"):
        return None
    cpath = _cache_path([dataset, symbol, schema, start, end, field, stype_in])
    prov = f"databento:{dataset}:{schema} (point-in-time)"
    if cpath.exists():
        try:
            s = pd.read_parquet(cpath)["v"]
            s.name = f"{symbol}:{field}"
            return s, prov + " [cached]"
        except Exception:  # noqa: BLE001
            pass
    try:
        import databento as db
        client = db.Historical(os.environ["DATABENTO_API_KEY"])
        store = client.timeseries.get_range(
            dataset=dataset, symbols=[symbol], schema=schema,
            start=start, end=end, stype_in=stype_in)
        df = store.to_df()
        if df is None or len(df) == 0 or field not in getattr(df, "columns", []):
            return None
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s = pd.Series(pd.to_numeric(df[field].values, errors="coerce"), index=idx.normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        if s.empty:
            return None
        try:
            _CACHE.mkdir(exist_ok=True)
            s.to_frame("v").to_parquet(cpath)
        except Exception:  # noqa: BLE001
            pass
        s.name = f"{symbol}:{field}"
        return s, prov
    except Exception:  # noqa: BLE001 — a data-fetch error never breaks a run
        return None
