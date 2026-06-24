from penrose.brain import Claim


def _claim():
    return Claim(
        claim_id="holdout-psr",
        statement="holdout psr gate test",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit",
        source_span="holdout psr gate test",
        claimed_metric_quote="",
    )


def _bt():
    return {
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 3.0,
        "n_oos": 1200,
        "n_trials": 3,
        "oos_sharpe": 2.0,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.2, 1.3, 1.1], "consistent": True},
        "capacity_usd": 1_000_000,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
    }


def test_positive_but_insignificant_holdout_lands_watch(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")

    decision = stages.p8_verdict(
        _claim(),
        _bt(),
        {"holdout_sharpe": 0.2, "holdout_psr": config.HOLDOUT_CONFIRM_PSR - 0.01, "nbars": 240},
        synthetic=False,
    )

    assert decision.verdict == "watch"
    assert "holdout did not confirm" in decision.rationale


def test_significant_holdout_can_reach_research_supported(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")

    decision = stages.p8_verdict(
        _claim(),
        _bt(),
        {"holdout_sharpe": 1.1, "holdout_psr": config.HOLDOUT_CONFIRM_PSR, "nbars": 240},
        synthetic=False,
    )

    assert decision.verdict == "research-supported"
