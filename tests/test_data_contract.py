from __future__ import annotations

import pandas as pd

from penrose.data.contract import DataBundle, Series


def _series(name: str) -> Series:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
    return Series(name, pd.Series([1.0, 2.0], index=idx), "test", "u")


def test_unique_normalized_alias_resolves():
    bundle = DataBundle(series={"us_equity_spy": _series("us_equity_spy")})
    assert bundle.get("us-equity-SPY") is bundle.series["us_equity_spy"]


def test_ambiguous_or_unknown_alias_misses():
    bundle = DataBundle(series={
        "us_equity_spy": _series("us_equity_spy"),
        "spy_equity_us": _series("spy_equity_us"),
    })
    assert bundle.get("us-equity-SPY") is None
    assert bundle.get("us_equity_qqq") is None


def test_cross_bucket_qualifier_sibling_alias_misses():
    bundle = DataBundle(series={
        "us_equity_spy_close": _series("us_equity_spy_close"),
        "us_equity_spy_volume": _series("us_equity_spy_volume"),
    })
    assert bundle.get("us_equity_spy") is None
