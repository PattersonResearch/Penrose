from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "worked_example_process_conditional.py"
spec = importlib.util.spec_from_file_location("worked_example_process_conditional", SCRIPT)
worked = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = worked
spec.loader.exec_module(worked)


@pytest.fixture(autouse=True)
def _isolated_holdout_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("PENROSE_HOLDOUT_LOCK", str(tmp_path / "worked-example-holdout.lock"))


def test_series_identical():
    example = worked.build_example(write_markdown=False)
    assert example.series_hash_a == example.series_hash_b


def test_process_conditional_divergence():
    example = worked.build_example(write_markdown=False)
    assert example.process_a.verdict in {"research-supported", "watch"}
    assert example.process_b.verdict == "kill"
    assert example.process_b.kill_reason in {"no_oos_edge", "low_edge_t"}
    assert example.process_a.verdict != example.process_b.verdict


def test_uses_real_deflation():
    example = worked.build_example(write_markdown=False)
    i = int(len(example.series) * worked.p7_backtest.IS_FRAC)
    o = int(len(example.series) * (worked.p7_backtest.IS_FRAC + worked.p7_backtest.OOS_FRAC))
    oos = example.series[i:o]
    direct_a = worked.stats.deflated_sharpe(
        oos, example.process_a.n_trials, example.process_a.ledger_sr_variance
    )
    direct_b = worked.stats.deflated_sharpe(
        oos, example.process_b.n_trials, example.process_b.ledger_sr_variance
    )
    assert np.isclose(example.process_a.dsr, round(direct_a, 4))
    assert np.isclose(example.process_b.dsr, round(direct_b, 4))
    assert example.process_a.decision.verdict == example.process_a.verdict
    assert example.process_b.decision.verdict == example.process_b.verdict
    assert example.process_a.bt["n_oos"] >= worked.config.DSR_DECISION["min_oos_bars"]
    assert example.process_b.bt["n_oos"] >= worked.config.DSR_DECISION["min_oos_bars"]
    assert worked.trial_stats is worked.p7_backtest._trial_stats


def test_determinism():
    first = worked.build_example(write_markdown=False)
    second = worked.build_example(write_markdown=False)
    assert (
        first.series_hash_a,
        first.process_a.dsr,
        first.process_b.dsr,
        first.process_a.verdict,
        first.process_b.verdict,
    ) == (
        second.series_hash_a,
        second.process_a.dsr,
        second.process_b.dsr,
        second.process_a.verdict,
        second.process_b.verdict,
    )


def test_only_process_differs():
    example = worked.build_example(write_markdown=False)
    assert example.series_hash_a == example.series_hash_b
    assert example.bars_per_year == worked.BARS_PER_YEAR
    assert example.thresholds == dict(worked.config.DSR_DECISION)
    assert example.process_a.search_denominator == 1
    assert example.process_b.search_denominator == 200
    assert example.process_a.n_trials != example.process_b.n_trials

    # The load-bearing claim: EVERY non-deflation gate is identical for A and B, so the verdict
    # flip is attributable to deflation alone (the only n_trials-dependent gate). Pin it.
    a_bt, b_bt = example.process_a.bt, example.process_b.bt
    assert a_bt["n_oos"] == b_bt["n_oos"]
    assert a_bt["psr"] == b_bt["psr"]                                  # PSR is not deflation-dependent
    assert a_bt["three_fold"] == b_bt["three_fold"]
    assert a_bt["regime"] == b_bt["regime"]
    assert a_bt["bootstrap"] == b_bt["bootstrap"]
    assert a_bt["permutation"] == b_bt["permutation"]
    assert a_bt["walk_forward"] == b_bt["walk_forward"]
    assert example.process_a.holdout == example.process_b.holdout
    # ...and ONLY the deflation-dependent quantities move:
    assert a_bt["n_trials"] != b_bt["n_trials"]
    assert a_bt["dsr"] != b_bt["dsr"]
    # the verdict must come from the real p8 decision object, not a local mapping
    assert example.process_a.verdict == example.process_a.decision.verdict
    assert example.process_b.verdict == example.process_b.decision.verdict
