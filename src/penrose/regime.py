"""Point-in-time regime indicator series — pre-registered, trailing-only labels penrose may
condition on and partition the kill-lens by.

The hard rule (ROADMAP / regime design): regime is an INPUT the strategy READS, never a parameter
it FITS. These labels are computed ONCE from price, using STRICTLY TRAILING windows, so the label
at time t depends only on data up to t — point-in-time-correct by construction (no look-ahead). A
module conditioning on `vol_regime`/`trend_regime` cannot slide the boundary to flatter itself; and
feeding them into `robustness.regime_split` lets the falsifier catch edges concentrated in one
*market* regime (vol/trend), a fragility the calendar-only lens is blind to.

These are EXOGENOUS (derived from price, not from the strategy's own returns), which is what makes
them a legitimate partition/conditioning input rather than data-snooping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_regime(prices: pd.Series, window: int = 60, min_history: int = 120) -> pd.Series:
    """Trailing-volatility regime: low / mid / high, by the EXPANDING terciles of trailing realized
    vol (point-in-time — the cut at t uses only vol observed up to t). Returns a date->label Series.

    window      : trailing window for realized vol (bars)
    min_history : need at least this many vol obs before a regime is assigned (else NaN)
    """
    px = pd.Series(prices).astype(float).dropna()
    ret = np.log(px).diff()
    rv = ret.rolling(window).std()                  # trailing realized vol (uses only past)
    lo = rv.expanding(min_periods=min_history).quantile(1 / 3)   # point-in-time cutoffs
    hi = rv.expanding(min_periods=min_history).quantile(2 / 3)
    label = pd.Series(index=rv.index, dtype=object)
    label[rv <= lo] = "low_vol"
    label[rv >= hi] = "high_vol"
    label[(rv > lo) & (rv < hi)] = "mid_vol"
    return label.dropna()


def trend_regime(prices: pd.Series, ma: int = 200) -> pd.Series:
    """Trailing-trend regime: 'up' if price is above its trailing `ma`-bar moving average, else
    'down'. The MA at t uses only prices up to t (point-in-time). Returns a date->label Series."""
    px = pd.Series(prices).astype(float).dropna()
    trail_ma = px.rolling(ma).mean()                # trailing MA (uses only past)
    label = pd.Series(np.where(px > trail_ma, "uptrend", "downtrend"), index=px.index, dtype=object)
    return label[trail_ma.notna()]


def regime_schemes(prices: pd.Series) -> dict:
    """Convenience: both pre-registered regime label series, for passing to regime_split's
    `extra_schemes`. Empty if there isn't enough price history to compute them."""
    out = {}
    try:
        v = vol_regime(prices)
        if len(v) >= 40:
            out["vol_regime"] = v
    except Exception:  # noqa: BLE001
        pass
    try:
        t = trend_regime(prices)
        if len(t) >= 40:
            out["trend_regime"] = t
    except Exception:  # noqa: BLE001
        pass
    return out
