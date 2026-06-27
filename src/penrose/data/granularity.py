"""Data-granularity verification at the INPUT boundary.

A series fed to a backtest at the wrong sampling frequency (e.g. intraday bars consumed by a rule
written for daily data) silently corrupts every downstream statistic and yields a confident, wrong
verdict. This is the input-side analogue of the post-hoc `bars_per_year`-vs-calendar-span check that
P7 already applies to module OUTPUTS: here we verify the frequency of the data going IN.

Inference is from the median bar spacing of the DatetimeIndex (robust to gaps/holidays). Everything
fails open (returns `None`/`"unknown"`, never raises) and is advisory by default: a caller decides
whether a mismatch warns or refuses. Determinism: no clock, no randomness.
"""
from __future__ import annotations

import pandas as pd

# (inclusive upper bound on calendar bars/year, label), ascending. Bands are deliberately wide so
# trading-day daily (~252) and calendar daily (~365) both read as "daily".
_LABEL_BANDS = [
    (3.0, "annual"),
    (24.0, "monthly"),
    (90.0, "weekly"),
    (400.0, "daily"),
    (5000.0, "hourly"),
]
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def infer_bars_per_year(data) -> float | None:
    """Empirical bars/year from the median spacing of the index. None if not inferable."""
    idx = getattr(data, "index", data)
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return None
    deltas = pd.Series(idx).diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None
    median_s = float(deltas.median())
    if median_s <= 0:
        return None
    return _SECONDS_PER_YEAR / median_s


def frequency_label(bars_per_year: float | None) -> str:
    """Coarse frequency label from an empirical bars/year estimate."""
    if bars_per_year is None:
        return "unknown"
    for upper, label in _LABEL_BANDS:
        if bars_per_year <= upper:
            return label
    return "intraday"


def granularity_report(data) -> dict:
    """`{empirical_bars_per_year, frequency, n}` for a series or DatetimeIndex. Never raises."""
    bpy = infer_bars_per_year(data)
    idx = getattr(data, "index", data)
    n = len(idx) if hasattr(idx, "__len__") else 0
    return {
        "empirical_bars_per_year": round(bpy, 1) if bpy is not None else None,
        "frequency": frequency_label(bpy),
        "n": int(n),
    }


def check_granularity(data, expected: str = "daily") -> dict:
    """Compare a series' empirical frequency label to `expected`.

    Returns `{ok, expected, actual, empirical_bars_per_year, message}`. `ok` is True when the
    inferred label matches `expected`, or when the frequency cannot be inferred (fail-open: we do
    not block on unknowable granularity). Never raises.
    """
    rep = granularity_report(data)
    actual = rep["frequency"]
    bpy = rep["empirical_bars_per_year"]
    ok = actual == expected or actual == "unknown"
    if actual == "unknown":
        msg = "granularity unknown (too few/irregular points); not blocking."
    elif ok:
        msg = f"granularity OK: ~{bpy} bars/yr ({actual})."
    else:
        msg = (f"GRANULARITY MISMATCH: expected {expected}, series is ~{bpy} bars/yr ({actual}). "
               f"A rule written for {expected} data run on {actual} bars corrupts every statistic; "
               f"resample to {expected} or set the correct bars_per_year before testing.")
    return {"ok": ok, "expected": expected, "actual": actual,
            "empirical_bars_per_year": bpy, "message": msg}
