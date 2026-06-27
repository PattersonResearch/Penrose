"""Data-granularity verification at the input boundary. A wrong-frequency series (intraday bars
used by a daily rule) must be DETECTED, not silently corrupt the verdict. Deterministic."""
from __future__ import annotations

import pandas as pd

from penrose.data import granularity as g
from penrose.data.contract import DataBundle, Series, Unavailable


def _series(freq: str, periods: int) -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=periods, freq=freq, tz="UTC")
    return pd.Series(range(periods), index=idx, dtype=float)


def test_infers_daily_weekly_monthly_intraday():
    assert g.frequency_label(g.infer_bars_per_year(_series("B", 400))) == "daily"
    assert g.frequency_label(g.infer_bars_per_year(_series("D", 400))) == "daily"
    assert g.frequency_label(g.infer_bars_per_year(_series("W", 200))) == "weekly"
    assert g.frequency_label(g.infer_bars_per_year(_series("MS", 120))) == "monthly"
    assert g.frequency_label(g.infer_bars_per_year(_series("15min", 2000))) == "intraday"
    assert g.frequency_label(g.infer_bars_per_year(_series("h", 5000))) in {"hourly", "intraday"}


def test_check_flags_intraday_as_not_daily_and_passes_daily():
    daily = g.check_granularity(_series("B", 400), expected="daily")
    assert daily["ok"] and daily["actual"] == "daily"

    intraday = g.check_granularity(_series("15min", 3000), expected="daily")
    assert not intraday["ok"]
    assert intraday["actual"] == "intraday"
    assert "GRANULARITY MISMATCH" in intraday["message"]


def test_fail_open_on_uninferable_input():
    # too few points / not a DatetimeIndex -> unknown, never raises, never blocks
    assert g.infer_bars_per_year(pd.Series([1.0, 2.0])) is None
    chk = g.check_granularity(pd.Series([1.0, 2.0]), expected="daily")
    assert chk["ok"] and chk["actual"] == "unknown"


def test_bundle_granularity_warnings_surfaces_only_mismatches():
    bundle = DataBundle(series={
        "ok_daily": Series(name="ok_daily", data=_series("B", 300), provenance="test", unit="px"),
        "bad_intraday": Series(name="bad_intraday", data=_series("15min", 3000), provenance="test", unit="px"),
        "missing": Unavailable(name="missing", reason="no data"),
    })
    warns = bundle.granularity_warnings(expected="daily")
    names = {w["name"] for w in warns}
    assert names == {"bad_intraday"}            # daily ok, unavailable skipped
    assert warns[0]["actual"] == "intraday"
