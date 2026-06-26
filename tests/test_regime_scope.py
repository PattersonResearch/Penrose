import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim
from penrose.pipeline import robustness as R
from penrose.pipeline import stages
from penrose import regime as regime_lib


def _claim(declared=None):
    return Claim(
        claim_id="regime-scope",
        statement="regime scoped claim",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit",
        source_span="span",
        claimed_metric_quote="quote",
        declared_regime=declared,
    )


def _survivor_bt(regime):
    return {
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 4.0,
        "n_oos": 1200,
        "oos_sharpe": 2.0,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.0, 1.1, 0.9], "consistent": True},
        "capacity_usd": 1_000_000,
        "bootstrap": {},
        "permutation": {},
        "regime": regime,
    }


def test_additivity_declared_none_is_exact_default_regime_result():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=180, freq="D", tz="UTC")
    net = pd.Series(rng.normal(0.006, 0.012, len(idx)), index=idx)
    labels = pd.Series(np.where(np.arange(len(idx)) % 3 == 0, "high_vol", "mid_vol"), index=idx)

    before = R.regime_split(net, 252.0, extra_schemes={"vol_regime": labels})
    after = R.regime_split(net, 252.0, extra_schemes={"vol_regime": labels}, declared=None)

    assert after == before
    assert "declared_regime" not in after


def test_regime_conditional_not_false_killed_and_scoped():
    idx = pd.date_range("2024-01-01", periods=180, freq="D", tz="UTC")
    labels = pd.Series(["high_vol"] * 140 + ["mid_vol"] * 40, index=idx)
    net = pd.Series(np.where(labels.eq("high_vol"), 0.020, -0.001), index=idx)

    undeclared = R.regime_split(net, 252.0, extra_schemes={"vol_regime": labels})
    declared = R.regime_split(
        net,
        252.0,
        extra_schemes={"vol_regime": labels},
        declared={"scheme": "vol_regime", "label": "high_vol"},
    )
    dec = stages.p8_verdict(_claim({"scheme": "vol_regime", "label": "high_vol"}),
                            _survivor_bt(declared), {"holdout_sharpe": 1.0}, False)

    assert undeclared["fragile"] is True
    assert declared["fragile"] is False
    assert declared["declared_exempted"] is True
    assert declared["adheres"] is True
    assert dec.verdict in {"watch", "research-supported"}
    assert "valid within declared regime: vol_regime=high_vol" in dec.rationale


def test_within_regime_fragility_still_bites():
    idx = pd.date_range("2024-01-01", periods=180, freq="D", tz="UTC")
    labels = pd.Series(["high_vol"] * 140 + ["mid_vol"] * 40, index=idx)
    net = pd.Series(np.select([idx.dayofweek == 0, idx.dayofweek >= 5],
                              [0.090, 0.008], default=0.0), index=idx)

    declared = R.regime_split(
        net,
        252.0,
        extra_schemes={"vol_regime": labels},
        declared={"scheme": "vol_regime", "label": "high_vol"},
    )
    dec = stages.p8_verdict(_claim({"scheme": "vol_regime", "label": "high_vol"}),
                            _survivor_bt(declared), {"holdout_sharpe": 1.0}, False)

    assert declared["fragile"] is True
    assert "day_of_week" in declared["fragile_reason"]
    assert dec.verdict == "kill"
    assert dec.kill_reason == "regime_fragile"


def test_adherence_mismatch_flagged():
    idx = pd.date_range("2024-01-01", periods=180, freq="D", tz="UTC")
    labels = pd.Series(["high_vol"] * 50 + ["low_vol"] * 130, index=idx)
    net = pd.Series(np.where(labels.eq("low_vol"), 0.020, -0.001), index=idx)

    declared = R.regime_split(
        net,
        252.0,
        extra_schemes={"vol_regime": labels},
        declared={"scheme": "vol_regime", "label": "high_vol"},
    )
    dec = stages.p8_verdict(_claim({"scheme": "vol_regime", "label": "high_vol"}),
                            _survivor_bt(declared), {"holdout_sharpe": 1.0}, False)

    assert declared["adheres"] is False
    assert declared["adherence"]["top_edge_label"] == "low_vol"
    assert dec.verdict == "kill"
    assert dec.kill_reason == "regime_mismatch"
    assert "regime_mismatch" in dec.rationale


def test_point_in_time_uses_trailing_regime_labels():
    idx = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    labels = regime_lib.vol_regime(prices, window=10, min_history=30)
    cutoff = labels.index[20]
    before = labels.loc[cutoff]

    future = pd.Series(np.linspace(1000.0, 2000.0, 40),
                       index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=40,
                                           freq="D", tz="UTC"))
    labels_with_future = regime_lib.vol_regime(pd.concat([prices, future]), window=10, min_history=30)

    assert labels_with_future.loc[cutoff] == before
