"""Tests for the empirical robustness layer (bootstrap, permutation, walk-forward)
and the verdict-hardening gates in p8_verdict.

Run: PYTHONPATH=src:. python tests/test_robustness.py   (or `make test`)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.pipeline import robustness as R   # noqa: E402


def test_bootstrap_separates_signal_from_noise():
    rng = np.random.default_rng(7)
    # a real positive edge: mean clearly above 0 relative to its spread
    signal = rng.normal(0.02, 0.03, 200)
    bs = R.block_bootstrap(signal, bars_per_year=73, n_boot=1500, seed=1)
    assert not bs["edge_ci_includes_zero"], "clear edge should have CI excluding zero"
    assert bs["p_edge_gt0"] > 0.95

    noise = rng.normal(0.0, 0.05, 200)
    bn = R.block_bootstrap(noise, bars_per_year=73, n_boot=1500, seed=1)
    assert bn["edge_ci_includes_zero"], "pure noise should have CI including zero"
    print(f"ok: bootstrap separates signal (CI {bs['edge_ci']}) from noise (CI {bn['edge_ci']})")


def test_permutation_detects_real_alignment():
    rng = np.random.default_rng(3)
    pos = rng.choice([-1.0, 1.0], 200)
    payoff = pos * np.abs(rng.normal(0.04, 0.02, 200))     # position predicts payoff sign
    real = R.permutation_test(pos, payoff, cost_frac=0.0008, n_perm=1500, seed=2)
    assert real["p_value"] < 0.05, f"aligned signal should be significant, got {real}"

    payoff_rand = rng.normal(0.0, 0.04, 200)               # no relationship
    null = R.permutation_test(pos, payoff_rand, cost_frac=0.0008, n_perm=1500, seed=2)
    assert null["p_value"] > 0.10, f"random signal should not be significant, got {null}"
    print(f"ok: permutation p={real['p_value']} (aligned) vs {null['p_value']} (random)")


def test_max_drawdown():
    r = np.array([0.1, 0.1, -0.3, 0.05])     # equity 0.1,0.2,-0.1,-0.05; peak 0.2 -> trough -0.1 = 0.3
    assert abs(R._max_drawdown(r) - 0.3) < 1e-9
    print("ok: max drawdown computed correctly")


def test_walk_forward_runs_and_flags_consistency():
    rng = np.random.default_rng(11)
    n = 600
    sig = np.abs(rng.normal(0.05, 0.02, n))
    iv = np.full(n, 0.5)
    fut = iv + 0.6 * sig + rng.normal(0, 0.02, n)          # signal genuinely lifts realized vol
    frame = pd.DataFrame({"signal": sig, "fut_rv": fut, "iv": iv})
    wf = R.walk_forward_vol(frame, hold_days=5, cost_frac=0.0008, n_windows=4)
    assert wf["n_windows"] == 4 and wf["n_trades"] > 30
    assert isinstance(wf["per_window_sharpe"], list)
    print(f"ok: walk-forward ran {wf['n_windows']} windows, oos_sharpe={wf['oos_sharpe']}, "
          f"consistent={wf['consistent']}")


def test_capacity_ci_is_a_range():
    rng = np.random.default_rng(5)
    net = pd.Series(rng.normal(0.01, 0.04, 200))
    pos = pd.DataFrame({"VOL:BTC": np.abs(rng.normal(0.6, 0.2, 200))})
    cc = R.capacity_ci(net.values, pos, bars_per_year=73, impact_bps_per_1m=25.0, n_boot=400)
    assert cc and cc["capacity_lo"] <= cc["capacity_median"] <= cc["capacity_hi"]
    print(f"ok: capacity CI = [{cc['capacity_lo']:,} .. {cc['capacity_hi']:,}] (median {cc['capacity_median']:,})")


def test_capacity_ci_graceful_when_capacity_diverges():
    # Negligible turnover/impact sends modeled linear-impact capacity to +inf; capacity_ci
    # must return a graceful note dict, never raise OverflowError on int(inf). Regression for
    # a crash surfaced by refereeing low-turnover trend rules (slow EWMAC barely trades).
    rng = np.random.default_rng(7)
    net = pd.Series(np.abs(rng.normal(0.02, 0.01, 200)))          # positive edge in every resample
    pos = pd.DataFrame({"VOL:BTC": np.abs(rng.normal(0.6, 0.2, 200))})
    cc = R.capacity_ci(net.values, pos, bars_per_year=73, impact_bps_per_1m=1e-300, n_boot=200)
    assert cc is not None and "note" in cc and "capacity_lo" not in cc
    print("ok: capacity_ci returns a graceful note when capacity diverges")


def test_regime_fragility_permutation_confirms_planted_weekend_edge():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=220, freq="D")
    net = pd.Series(rng.normal(0.0, 0.004, len(idx)), index=idx)
    weekend = idx.dayofweek >= 5
    net[weekend] += rng.normal(0.05, 0.01, int(weekend.sum()))
    out = R.regime_split(net, bars_per_year=252)
    assert out["fragile"] is True
    assert out["fragility_p"]["weekend"] < 0.05
    assert "perm p=" in out["fragile_reason"]


def test_regime_fragility_permutation_rejects_marginal_noise_concentration():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    net = pd.Series(rng.normal(0.05, 1.0, len(idx)), index=idx)
    out = R.regime_split(net, bars_per_year=252)
    assert out["fragile"] is False
    assert out["fragility_p"]["day_of_week"] >= 0.05


def test_regime_fragility_permutation_is_deterministic():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    net = pd.Series(rng.normal(0.05, 1.0, len(idx)), index=idx)
    assert R.regime_split(net, bars_per_year=252) == R.regime_split(net, bars_per_year=252)


def test_verdict_gate_marks_ambiguous_thin_null_underpowered():
    from penrose.pipeline.stages import p8_verdict
    from penrose.brain import Claim
    claim = Claim("c1", "x", "m", "s", "h", "src", "span", "quote")
    # a bt that would PASS the analytic gate (high psr, stable folds) but whose
    # bootstrap edge CI straddles zero -> the empirical gate identifies an ambiguous null.
    bt = {"psr": 0.97, "dsr": 0.97, "edge_t": 1.4, "n_oos": 80, "oos_sharpe": 1.2,
          "three_fold": {"folds": [0.5, 0.4, 0.3], "consistent": True},
          "capacity_usd": 1_000_000,
          "bootstrap": {"edge_ci": [-0.002, 0.01], "edge_ci_includes_zero": True, "ci": 0.90},
          "permutation": {"p_value": 0.03}}
    dec = p8_verdict(claim, bt, holdout={"holdout_sharpe": 1.0}, synthetic=False)
    # n_oos=80 cannot resolve penrose's realistic 0.05 IC floor, so the power-aware
    # taxonomy correctly refuses to turn an ambiguous bootstrap null into "dead".
    assert dec.verdict == "underpowered" and dec.kill_reason == "below_detection_floor", dec.verdict
    assert "edge CI includes" in dec.rationale
    print(f"ok: empirical gate marked an analytically-passing thin claim underpowered "
          f"({dec.rationale[:60]}...)")


def test_unanchored_sources_cannot_reach_research_supported():
    from penrose import config
    from penrose.brain import Claim
    from penrose.pipeline.stages import p8_verdict

    bt = {"psr": 0.99, "dsr": 0.99, "edge_t": 4.0, "n_oos": 2000,
          "oos_sharpe": 2.0, "bars_per_year": 252,
          "three_fold": {"folds": [1.0, 1.1, 0.9], "consistent": True},
          "capacity_usd": 1_000_000, "bootstrap": {}, "permutation": {}, "regime": {}}
    old = config.COST_PROVENANCE
    config.COST_PROVENANCE = "measured"
    try:
        for source_type in ("generated_hypothesis", "chat"):
            claim = Claim("c1", "x", "m", "s", "h", "src", "span", "quote",
                          source_type=source_type)
            dec = p8_verdict(claim, bt, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
            assert dec.verdict == "watch"
            assert dec.metrics["fidelity_provenance"] == "self-authored-unanchored"
            assert "external anchor" in dec.rationale
    finally:
        config.COST_PROVENANCE = old


def test_v4a_tolerance_scales_with_reference_noise():
    from penrose.pipeline import v4a
    ref = {"avg_net_edge": 0.010, "oos_sharpe": 1.20, "capacity_usd": 1_000_000,
           "bootstrap": {"edge_ci": [0.004, 0.016], "sharpe_ci": [0.6, 1.8]}}  # half-widths: 0.006, 0.6
    near = {"avg_net_edge": 0.013, "oos_sharpe": 1.45, "capacity_usd": 3_000_000}
    far = {"avg_net_edge": -0.020, "oos_sharpe": -0.50, "capacity_usd": 50_000_000}
    assert v4a.within_tolerance(ref, near)["pass"], "a close candidate should calibrate"
    res_far = v4a.within_tolerance(ref, far)
    assert not res_far["pass"] and res_far["conclusive"], "a flipped/distant candidate should fail"
    # tolerance is the reference CI half-width, not a constant: widen the reference
    # CI and the same 'far' edge gap now fits
    ref_wide = dict(ref, bootstrap={"edge_ci": [-0.05, 0.07], "sharpe_ci": [-2.0, 4.4]})
    edge_only = {"avg_net_edge": 0.013, "oos_sharpe": 1.45, "capacity_usd": 3_000_000}
    assert v4a.within_tolerance(ref_wide, edge_only)["checks"]["edge"]["pass"]
    print("ok: V4a tolerance scales with the reference's own bootstrap CI half-width")


if __name__ == "__main__":
    test_bootstrap_separates_signal_from_noise()
    test_permutation_detects_real_alignment()
    test_max_drawdown()
    test_walk_forward_runs_and_flags_consistency()
    test_capacity_ci_is_a_range()
    test_capacity_ci_graceful_when_capacity_diverges()
    test_verdict_gate_marks_ambiguous_thin_null_underpowered()
    test_unanchored_sources_cannot_reach_research_supported()
    test_v4a_tolerance_scales_with_reference_noise()
    print("\nALL ROBUSTNESS TESTS PASSED")
