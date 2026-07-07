import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose import config  # noqa: E402
from penrose.brain import Claim  # noqa: E402
from penrose.pipeline import robustness as R, stages  # noqa: E402


def _claim():
    return Claim("tail", "tail risk test", "", "", "", "unit", "", "")


def test_tail_metrics_near_constant_series_has_no_skew():
    # A float-constant series has sd ~ machine noise, not exactly 0; skew is UNDEFINED there.
    # Before the fix this returned a garbage skew (and a divide overflow). (swarm audit BUG #1)
    out = R.tail_metrics([0.01] * 100)
    assert out["skew"] is None
    assert out["asymmetric"] is False


def _survivor_bt(tail, *, research=False):
    return {
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 4.0,
        "n_oos": 2000,
        "bars_per_year": 252.0,
        "oos_sharpe": 2.0,
        "three_fold": {"folds": [1.0, 1.1, 0.9], "consistent": True},
        "capacity_usd": 1_000_000,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
        "tail": tail,
    }


def _widow_maker():
    return np.array([0.01] * 190 + [-0.20] * 10, dtype=float)


def _severe_skew_low_tail_ratio():
    return np.array([0.01] * 935 + [0.10] * 50 + [-0.90] * 15, dtype=float)


def _mild_skew_low_tail_ratio():
    return np.array([0.01] * 870 + [0.10] * 100 + [-0.15] * 30, dtype=float)


def test_widow_maker_tail_metrics_asymmetric():
    tail = R.tail_metrics(_widow_maker())
    assert tail["asymmetric"] is True
    assert tail["skew"] <= config.TAIL_RISK_GATE["max_skew"]
    assert tail["tail_ratio"] >= config.TAIL_RISK_GATE["min_tail_ratio"]


def test_severe_skew_alone_flags_tail_asymmetric(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    tail = R.tail_metrics(_severe_skew_low_tail_ratio())
    assert tail["skew"] <= config.TAIL_RISK_GATE["severe_skew"]
    assert tail["tail_ratio"] < config.TAIL_RISK_GATE["min_tail_ratio"]
    assert tail["asymmetric"] is True


def test_mild_skew_low_tail_ratio_not_asymmetric(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    tail = R.tail_metrics(_mild_skew_low_tail_ratio())
    assert config.TAIL_RISK_GATE["severe_skew"] < tail["skew"] <= config.TAIL_RISK_GATE["max_skew"]
    assert tail["tail_ratio"] < config.TAIL_RISK_GATE["min_tail_ratio"]
    assert tail["asymmetric"] is False


def test_moderate_skew_with_large_tail_ratio_still_flags(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -99.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    tail = R.tail_metrics(_widow_maker())
    assert tail["skew"] > config.TAIL_RISK_GATE["severe_skew"]
    assert tail["skew"] <= config.TAIL_RISK_GATE["max_skew"]
    assert tail["tail_ratio"] >= config.TAIL_RISK_GATE["min_tail_ratio"]
    assert tail["asymmetric"] is True


def test_symmetric_tail_metrics_not_asymmetric():
    rng = np.random.default_rng(7)
    tail = R.tail_metrics(rng.normal(0.0, 0.02, 400))
    assert tail["asymmetric"] is False


def test_tail_gate_default_on_caps_research_supported_to_watch(monkeypatch):
    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    bt = _survivor_bt(R.tail_metrics(_severe_skew_low_tail_ratio()), research=True)
    dec = stages.p8_verdict(_claim(), bt, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert "tail-asymmetric widow-maker warning" in dec.rationale
    assert "tail-asymmetric: capped to watch" in dec.rationale
    assert dec.metrics["tail_asymmetric"] is True
    assert dec.metrics["tail_skew"] == bt["tail"]["skew"]
    assert dec.metrics["tail_tail_ratio"] == bt["tail"]["tail_ratio"]
    assert dec.metrics["tail_max_loss"] == bt["tail"]["max_loss"]


def test_tail_gate_enabled_kills_widow_maker(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": False},
    )
    bt = _survivor_bt(R.tail_metrics(_widow_maker()))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "kill"
    assert dec.kill_reason == "tail_asymmetric"
    assert "tail-asymmetric widow-maker warning" in dec.rationale
    assert dec.metrics["tail_asymmetric"] is True


def test_tail_gate_enabled_ignores_symmetric(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": False},
    )
    rng = np.random.default_rng(11)
    bt = _survivor_bt(R.tail_metrics(rng.normal(0.01, 0.02, 400)))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert dec.metrics["tail_asymmetric"] is False


def test_tail_gate_cap_only_caps_research_supported_to_watch(monkeypatch):
    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    bt = _survivor_bt(R.tail_metrics(_widow_maker()), research=True)
    dec = stages.p8_verdict(_claim(), bt, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert "tail-asymmetric: capped to watch" in dec.rationale


def test_tail_warning_appears_when_verdict_already_watch(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    bt = _survivor_bt(R.tail_metrics(_severe_skew_low_tail_ratio()))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert "tail-asymmetric widow-maker warning" in dec.rationale
    assert "tail-asymmetric: capped to watch" not in dec.rationale
    assert dec.metrics["tail_asymmetric"] is True


def test_provided_series_tail_warning_has_preaggregated_caveat(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0,
         "min_tail_ratio": 3.0, "cap_only": True},
    )
    bt = dict(
        _survivor_bt(R.tail_metrics(_severe_skew_low_tail_ratio())),
        claim_type="provided_series_statistic",
    )
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "watch"
    assert "provided-series caveat: tested series is pre-aggregated" in dec.rationale
    assert "true tail risk requires per-trade reconstruction" in dec.rationale


def _kill_bt(tail):
    """A backtest whose statistical path is a KILL (3-fold sign-unstable), plus a tail flag."""
    bt = _survivor_bt(tail)
    bt["three_fold"] = {"folds": [1.2, -1.4, 1.1], "consistent": False}
    bt["psr"], bt["dsr"] = 0.2, 0.2
    return bt


def test_tail_gate_does_not_overwrite_a_prior_kill(monkeypatch):
    """J-1 (high): with cap_only=False the tail gate must NOT overwrite a prior structural kill /
    routing verdict; it only WARNS. Overwriting suppressed the power-aware underpowered relabel and
    masked the true kill_reason for principle mining."""
    monkeypatch.setattr(
        config, "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0, "severe_min_n": 40,
         "min_tail_ratio": 3.0, "cap_only": False},
    )
    dec = stages.p8_verdict(_claim(), _kill_bt(R.tail_metrics(_widow_maker())), {}, False)
    assert dec.verdict != "research-supported"
    assert dec.kill_reason != "tail_asymmetric"      # the tail gate did NOT overwrite the reason
    assert "widow-maker" in dec.rationale            # but the warning is still present


def test_tail_gate_cap_only_false_hard_kills_a_survivor(monkeypatch):
    """The cap_only=False dial still hard-kills an otherwise-supported SURVIVOR (the intended case)."""
    monkeypatch.setattr(
        config, "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "severe_skew": -3.0, "severe_min_n": 40,
         "min_tail_ratio": 3.0, "cap_only": False},
    )
    dec = stages.p8_verdict(_claim(), _survivor_bt(R.tail_metrics(_widow_maker()), research=True), {}, False)
    assert dec.verdict == "kill" and dec.kill_reason == "tail_asymmetric"
