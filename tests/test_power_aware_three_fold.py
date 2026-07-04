from __future__ import annotations

import numpy as np

from penrose.brain import Claim
from penrose.pipeline import p7_backtest, stages


def _claim() -> Claim:
    return Claim("c", "s", "m", "scope", "1d", "src", "span", "metric")


def _fold_controlled_series(fold_means: list[float], seed: int = 7, fold_n: int = 333) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = []
    for mean in fold_means:
        noise = rng.normal(0.0, 1.0, fold_n)
        noise = (noise - noise.mean()) / noise.std(ddof=1)
        folds.append(noise + mean)
    return np.concatenate(folds + [np.array([0.05])])


def _bt(net: np.ndarray, *, n_oos: int = 300) -> dict:
    return {
        "psr": 0.95,
        "dsr": 0.95,
        "edge_t": 2.0,
        "n_oos": n_oos,
        "oos_sharpe": 1.0,
        "bars_per_year": 252,
        "per_trade_sharpe": 0.05,
        "three_fold": p7_backtest._three_fold(net),
        "capacity_usd": 1e6,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
    }


def test_true_marginal_edge_low_power_fold_failure_is_not_structural_kill():
    net = _fold_controlled_series([0.0832, -0.0164, 0.0832], seed=7)
    d = stages.p8_verdict(_claim(), _bt(net), {}, False)
    assert not (d.verdict == "kill" and d.kill_reason == "in_sample_only")
    assert d.verdict == "underpowered"
    assert d.kill_reason == "below_detection_floor"


def test_ambiguous_three_fold_uses_per_fold_power_even_when_overall_power_sufficient():
    # B-03: overall n_oos power can be sufficient while per-fold sign-stability power is not.
    net = _fold_controlled_series([0.0832, -0.0164, 0.0832], seed=7, fold_n=366)
    d = stages.p8_verdict(_claim(), _bt(net, n_oos=1100), {}, False)

    assert d.metrics["power_sufficient"] is True
    assert d.metrics["three_fold_power"] < stages.config.POWER["three_fold_min_power"]
    assert d.verdict == "underpowered"
    assert d.kill_reason == "below_detection_floor"


def test_significantly_negative_fold_stays_structural_kill():
    net = _fold_controlled_series([0.10, -0.12, 0.10], seed=7)
    d = stages.p8_verdict(_claim(), _bt(net), {}, False)
    assert d.verdict == "kill"
    assert d.kill_reason == "in_sample_only"
    assert d.metrics["min_fold_t"] <= -2.0


def test_three_fold_power_metrics_are_bounded():
    net = _fold_controlled_series([0.0832, -0.0164, 0.0832], seed=7)
    d = stages.p8_verdict(_claim(), _bt(net), {}, False)
    assert d.metrics["min_fold_t"] is not None
    assert d.metrics["three_fold_power"] is not None
    assert 0.0 <= d.metrics["three_fold_power"] <= 1.0
