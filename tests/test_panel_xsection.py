import numpy as np
import pandas as pd
import pytest

from penrose.data.panel import Panel
from penrose.data.xsection import (
    asof_panel,
    cross_sectional_rank,
    cross_sectional_zscore,
    form_factor,
    liquidity_screen,
    winsorize,
)


def test_panel_construction_coerces_naive_rejects_duplicate_columns_and_empty_coverage():
    dates = pd.date_range("2020-01-01", periods=3)
    p = Panel("p", pd.DataFrame({"a": [1, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]}, index=dates), "unit-test")
    assert str(p.data.index.tz) == "UTC"
    # all-NaN ENTITY 'b' is dropped; sparse DATES are KEPT (time axis preserved) -> 3 dates, 1 entity
    assert p.coverage == ("2020-01-01", "2020-01-03", 3, 1)

    dup = pd.DataFrame([[1.0, 2.0]], index=dates[:1])
    dup.columns = ["a", "a"]
    with pytest.raises(ValueError, match="columns must be unique"):
        Panel("dup", dup, "unit-test")

    empty = Panel("empty", pd.DataFrame(), "unit-test")
    assert empty.coverage == (None, None, 0, 0)


def test_winsorize_clips_bands_and_preserves_in_band_values():
    dates = pd.date_range("2020-01-01", periods=2, tz="UTC")
    p = Panel("x", pd.DataFrame({"a": [-10, 2], "b": [0.5, 20]}, index=dates), "raw")
    clipped = winsorize(p, -1, 3)
    assert clipped.data.loc[dates[0], "a"] == -1
    assert clipped.data.loc[dates[1], "b"] == 3
    assert clipped.data.loc[dates[0], "b"] == 0.5

    asym = winsorize(p, 0, 10)
    assert asym.data.loc[dates[0], "a"] == 0
    assert asym.data.loc[dates[1], "b"] == 10
    assert "winsorize" in asym.provenance


def test_rank_and_zscore_hand_checkable_with_deterministic_ties():
    dates = pd.date_range("2020-01-01", periods=3, tz="UTC")
    p = Panel(
        "x",
        pd.DataFrame(
            {
                "a": [1.0, 1.0, np.nan],
                "b": [1.0, 2.0, np.nan],
                "c": [3.0, 3.0, 5.0],
                "d": [4.0, np.nan, np.nan],
            },
            index=dates,
        ),
        "raw",
    )
    ranks = cross_sectional_rank(p, pct=False)
    assert ranks.data.loc[dates[0]].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    assert ranks.data.loc[dates[1], ["a", "b", "c"]].to_list() == [1.0, 2.0, 3.0]

    pct = cross_sectional_rank(p)
    assert pct.data.loc[dates[0], "d"] == 1.0

    z = cross_sectional_zscore(p)
    expected = (p.data.loc[dates[1], "a"] - 2.0) / 1.0
    assert z.data.loc[dates[1], "a"] == expected
    # a <2-observation date is KEPT with all-NaN values (spec), not dropped from the index
    assert dates[2] in z.data.index
    assert z.data.loc[dates[2]].isna().all()


def test_asof_panel_uses_last_filed_record_and_ignores_later_filing():
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"], tz="UTC")
    records = {
        "a": pd.DataFrame({"filed": ["2020-01-15", "2020-04-01"], "val": [10.0, 999.0]}),
        "b": pd.DataFrame({"filed": ["2020-02-01", "2020-03-15"], "val": [20.0, 30.0]}),
    }
    p = asof_panel(records, "book_to_market", dates)
    assert p.data.loc[pd.Timestamp("2020-03-31", tz="UTC"), "a"] == 10.0
    assert p.data.loc[pd.Timestamp("2020-01-31", tz="UTC"), "b"] != 20.0
    assert p.data.loc[pd.Timestamp("2020-02-29", tz="UTC"), "b"] == 20.0
    assert p.data.loc[pd.Timestamp("2020-03-31", tz="UTC"), "b"] == 30.0


def _factor_inputs():
    dates = pd.date_range("2020-01-31", "2020-03-10", tz="UTC")
    names = ["a", "b", "c", "d"]
    ret = pd.DataFrame(0.0, index=dates, columns=names)
    ret.loc[(ret.index > "2020-01-31") & (ret.index <= "2020-02-29"), ["c", "d"]] = 0.03
    ret.loc[(ret.index > "2020-01-31") & (ret.index <= "2020-02-29"), ["a", "b"]] = 0.01
    ret.loc[ret.index > "2020-02-29", ["a", "b"]] = 0.04
    ret.loc[ret.index > "2020-02-29", ["c", "d"]] = 0.02
    sig = pd.DataFrame(index=dates, columns=names, dtype=float)
    sig.loc[pd.Timestamp("2020-01-31", tz="UTC")] = [1.0, 2.0, 3.0, 4.0]
    sig.loc[pd.Timestamp("2020-02-29", tz="UTC")] = [4.0, 3.0, 2.0, 1.0]
    returns = Panel("returns", ret, "synthetic", kind="return")
    signal = Panel("signal", sig, "synthetic", kind="characteristic")
    return returns, signal


def test_form_factor_no_lookahead_membership_unchanged_when_future_returns_corrupted():
    returns, signal = _factor_inputs()
    base = form_factor(returns, signal, n_buckets=2, rebalance="ME", hold="1M", min_names=4)
    corrupted_data = returns.data.copy()
    corrupted_data.loc[corrupted_data.index > pd.Timestamp("2020-01-31", tz="UTC"), ["a", "d"]] += 100.0
    corrupted = Panel("returns", corrupted_data, "corrupted", kind="return")
    changed = form_factor(corrupted, signal, n_buckets=2, rebalance="ME", hold="1M", min_names=4)

    assert base.attrs["membership"] == changed.attrs["membership"]
    assert base.attrs["membership"]["2020-01-31"] == {"high": ["c", "d"], "low": ["a", "b"]}
    assert base.attrs["membership"]["2020-02-29"] == {"high": ["b", "a"], "low": ["d", "c"]}
    assert np.allclose(base.loc["2020-02-01":"2020-02-29"], 0.02)


def test_form_factor_weights_flip_sign_and_min_names_skip():
    returns, signal = _factor_inputs()
    weights = Panel(
        "weights",
        pd.DataFrame(
            {
                "a": [1.0, 9.0],
                "b": [9.0, 1.0],
                "c": [1.0, 9.0],
                "d": [9.0, 1.0],
            },
            index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"], tz="UTC"),
        ),
        "synthetic",
    )
    ew = form_factor(returns, signal, n_buckets=2, rebalance="ME", hold="1M", min_names=4)
    weighted = form_factor(returns, signal, n_buckets=2, rebalance="ME", hold="1M", weights=weights, min_names=4)
    flipped = form_factor(returns, signal, n_buckets=2, rebalance="ME", hold="1M", min_names=4, high_minus_low=False)
    skipped = form_factor(returns, signal, n_buckets=2, rebalance="ME", hold="1M", min_names=5)

    assert not weighted.empty
    assert np.allclose(flipped, -ew)
    assert skipped.empty
    assert "form_factor" in ew.attrs["provenance"]


def test_liquidity_screen_top_n_trailing_only():
    # 'a' is ILLIQUID before asof but its post-asof volume is HIGH on a MAJORITY of days, so a
    # leaky (full-history-median) implementation would pick 'a'; a trailing-only one must not.
    # (A single future outlier can't move a median, so the leak must be majority to be detectable.)
    dates = pd.date_range("2020-01-01", periods=10, tz="UTC")
    dvol = Panel(
        "dvol",
        pd.DataFrame(
            {
                "a": [10, 10, 10, 5000, 5000, 5000, 5000, 5000, 5000, 5000],  # low pre-asof, high after
                "b": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                "c": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
            },
            index=dates,
        ),
        "synthetic",
    )
    asof = dates[2]  # 2020-01-03: trailing window sees only a=10 (median 10) -> 'a' excluded
    picked = liquidity_screen(dvol, top_n=2, asof=asof, lookback="10D")
    assert picked == {"b", "c"}
    assert "a" not in picked, "trailing-only screen must ignore 'a' post-asof high volume (leak guard)"


def test_form_factor_blends_overlapping_holds():
    # Two monthly rebalances with OPPOSITE membership; 2-month holds overlap in March.
    # Jan cohort: high={c,d}(+2%) - low={a,b}(0) = +0.02.  Feb cohort (signal reversed):
    # high={a,b}(0) - low={c,d}(+2%) = -0.02.  A correct blend averages to ~0 on the overlap;
    # the old keep-first dedup would report only the Jan leg (+0.02).
    dates = pd.date_range("2020-01-31", "2020-04-15", freq="D", tz="UTC")
    ret = pd.DataFrame(0.0, index=dates, columns=["a", "b", "c", "d"])
    ret["c"] = ret["d"] = 0.02
    sig = pd.DataFrame(index=dates, columns=["a", "b", "c", "d"], dtype=float)
    jan = dates < pd.Timestamp("2020-02-01", tz="UTC")
    sig.loc[jan] = [1.0, 2.0, 3.0, 4.0]      # Jan: high={c,d}
    sig.loc[~jan] = [4.0, 3.0, 2.0, 1.0]     # Feb+: high={a,b}
    f = form_factor(Panel("r", ret, "syn", kind="return"),
                    Panel("s", sig, "syn", kind="characteristic"),
                    n_buckets=2, rebalance="ME", hold="2M", min_names=2, high_minus_low=True)
    mar = f[(f.index >= "2020-03-05") & (f.index <= "2020-03-20")]
    assert len(mar) > 0 and abs(float(mar.mean())) < 1e-9, \
        "overlapping opposite cohorts must blend to ~0, not keep-first"
