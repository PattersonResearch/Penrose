import numpy as np
import pandas as pd

from penrose.brain import Claim


def _claim():
    return Claim(
        claim_id="skip-holdout",
        statement="skip unreachable holdout test",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit",
        source_span="skip unreachable holdout test",
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


def _mres():
    idx = pd.date_range("2024-01-01", periods=240, freq="D", tz="UTC")
    rng = np.random.default_rng(404)
    net = pd.Series(0.01 + rng.normal(0.0, 0.001, len(idx)), index=idx)
    return {"net": net, "bars_per_year": 252.0}


def test_modeled_costs_skip_holdout_and_preserve_lock(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import p7_backtest as p7, run as run_mod, stages

    monkeypatch.delenv("PENROSE_HOLDOUT_LOCK", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    monkeypatch.setattr(config, "COST_PROVENANCE", "modeled")

    dec0 = stages.p8_verdict(_claim(), _bt(), {}, synthetic=False)
    dec, holdout = run_mod._maybe_consult_holdout(_claim(), _bt(), _mres(), dec0, synthetic=False)

    assert dec.verdict == "watch"
    assert holdout["not_consulted"] is True
    assert "preserved for a measured-cost run" in holdout["reason"]
    assert "preserved for a measured-cost run" in dec.rationale
    assert list((tmp_path / ".holdout" / "locks").glob("*.lock")) == []


def test_measured_costs_consult_holdout(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import p7_backtest as p7, run as run_mod, stages

    monkeypatch.delenv("PENROSE_HOLDOUT_LOCK", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")

    dec0 = stages.p8_verdict(_claim(), _bt(), {}, synthetic=False)
    dec, holdout = run_mod._maybe_consult_holdout(_claim(), _bt(), _mres(), dec0, synthetic=False)

    assert holdout.get("refused") is not True
    assert "holdout_sharpe" in holdout
    assert dec.verdict == "research-supported"
    assert len(list((tmp_path / ".holdout" / "locks").glob("*.lock"))) == 1
