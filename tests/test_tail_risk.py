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


def test_widow_maker_tail_metrics_asymmetric():
    tail = R.tail_metrics(_widow_maker())
    assert tail["asymmetric"] is True
    assert tail["skew"] <= config.TAIL_RISK_GATE["max_skew"]
    assert tail["tail_ratio"] >= config.TAIL_RISK_GATE["min_tail_ratio"]


def test_symmetric_tail_metrics_not_asymmetric():
    rng = np.random.default_rng(7)
    tail = R.tail_metrics(rng.normal(0.0, 0.02, 400))
    assert tail["asymmetric"] is False


def test_tail_gate_default_off_verdict_unchanged(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": False, "max_skew": -0.5, "min_tail_ratio": 3.0, "cap_only": False},
    )
    bt = _survivor_bt(R.tail_metrics(_widow_maker()))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None


def test_tail_gate_enabled_kills_widow_maker(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "min_tail_ratio": 3.0, "cap_only": False},
    )
    bt = _survivor_bt(R.tail_metrics(_widow_maker()))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "kill"
    assert dec.kill_reason == "tail_asymmetric"


def test_tail_gate_enabled_ignores_symmetric(monkeypatch):
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "min_tail_ratio": 3.0, "cap_only": False},
    )
    rng = np.random.default_rng(11)
    bt = _survivor_bt(R.tail_metrics(rng.normal(0.01, 0.02, 400)))
    dec = stages.p8_verdict(_claim(), bt, {}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None


def test_tail_gate_cap_only_caps_research_supported_to_watch(monkeypatch):
    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    monkeypatch.setattr(
        config,
        "TAIL_RISK_GATE",
        {"enabled": True, "max_skew": -0.5, "min_tail_ratio": 3.0, "cap_only": True},
    )
    bt = _survivor_bt(R.tail_metrics(_widow_maker()), research=True)
    dec = stages.p8_verdict(_claim(), bt, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert "tail-asymmetric: capped to watch" in dec.rationale
