import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim  # noqa: E402
from penrose.pipeline import p7_backtest as P7  # noqa: E402
from penrose.pipeline import robustness as R  # noqa: E402
from penrose.pipeline import stages  # noqa: E402


BPY = 252.0


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _claim() -> Claim:
    return Claim("cpcv", "cpcv test", "", "", "", "unit", "", "")


def _survivor_bt(cpcv: dict | None = None, *, n_oos: int = 1200) -> dict:
    return {
        "psr": 0.98,
        "dsr": 0.98,
        "edge_t": 3.0,
        "n_oos": n_oos,
        "bars_per_year": BPY,
        "oos_sharpe": 1.2,
        "is_sharpe": 1.0,
        "n_trials": 1,
        "three_fold": {"folds": [1.0, 1.1, 0.9], "consistent": True},
        "capacity_usd": 1_000_000,
        "bootstrap": {"edge_ci_includes_zero": False},
        "permutation": {"p_value": 0.01},
        "regime": {"fragile": False},
        "walk_forward": {},
        "capacity_ci": {},
        "cpcv": cpcv or {},
    }


def _overfit_seen() -> pd.Series:
    rng = np.random.default_rng(10)
    vals = np.r_[
        rng.normal(0.006, 0.001, 80),
        rng.normal(-0.008, 0.001, 80),
    ]
    return pd.Series(vals, index=_idx(len(vals)))


def _persistent_seen() -> pd.Series:
    rng = np.random.default_rng(11)
    vals = rng.normal(0.008, 0.001, 160)
    return pd.Series(vals, index=_idx(len(vals)))


def test_overfit_series_is_caught_and_demotes_borderline_survivor():
    cv = R.cpcv(_overfit_seen(), BPY, n_groups=8, k_test=2, embargo_frac=0.01, seed=3)
    assert cv["ran"] is True
    assert cv["prob_oos_loss"] >= 0.50
    assert cv["overfit"] is True
    assert cv["pbo"] is None

    dec = stages.p8_verdict(_claim(), _survivor_bt(cv), {}, synthetic=False)
    assert dec.verdict in {"kill", "underpowered"}
    assert "CPCV overfit" in dec.rationale
    assert dec.metrics["cpcv"]["pbo"] is None


def test_persistent_edge_survives_and_cpcv_does_not_change_verdict():
    cv = R.cpcv(_persistent_seen(), BPY, n_groups=8, k_test=2, embargo_frac=0.01, seed=3)
    assert cv["ran"] is True
    assert cv["prob_oos_loss"] == 0.0
    assert cv["overfit"] is False

    base = stages.p8_verdict(_claim(), _survivor_bt({}), {}, synthetic=False)
    with_cv = stages.p8_verdict(_claim(), _survivor_bt(cv), {}, synthetic=False)
    assert with_cv.verdict == base.verdict
    assert with_cv.kill_reason == base.kill_reason


def test_cpcv_never_promotes_existing_kill_or_watch():
    order = {"kill": 0, "underpowered": 1, "insufficient_data": 1, "watch": 2, "research-supported": 3}
    good_cv = R.cpcv(_persistent_seen(), BPY, n_groups=8, k_test=2, embargo_frac=0.01, seed=3)
    bad_cv = R.cpcv(_overfit_seen(), BPY, n_groups=8, k_test=2, embargo_frac=0.01, seed=3)

    dead = _survivor_bt(good_cv)
    dead["three_fold"] = {"folds": [1.0, -1.0, 1.0], "consistent": False}
    before_dead = stages.p8_verdict(_claim(), dict(dead, cpcv={}), {}, False)
    after_dead = stages.p8_verdict(_claim(), dead, {}, False)
    assert order[after_dead.verdict] <= order[before_dead.verdict]

    watch = _survivor_bt({})
    before_watch = stages.p8_verdict(_claim(), watch, {}, False)
    after_watch = stages.p8_verdict(_claim(), dict(watch, cpcv=bad_cv), {}, False)
    assert order[after_watch.verdict] <= order[before_watch.verdict]


def test_no_holdout_read_and_purge_embargo_are_applied():
    n = 200
    seen = np.random.default_rng(12).normal(0.004, 0.01, 160)
    tail_a = np.full(40, 0.5)
    tail_b = np.full(40, -0.5)
    idx = _idx(n)
    pos = pd.Series(1.0, index=idx)
    a = pd.Series(np.r_[seen, tail_a], index=idx)
    b = pd.Series(np.r_[seen, tail_b], index=idx)

    bt_a = P7.run_backtest("cpcv-holdout-a", a, pos, BPY, log=False)
    bt_b = P7.run_backtest("cpcv-holdout-b", b, pos, BPY, log=False)
    assert bt_a["cpcv"] == bt_b["cpcv"]
    assert bt_a["n_trials"] == bt_b["n_trials"]

    cv = R.cpcv(pd.Series(np.arange(50, dtype=float), index=_idx(50)),
                BPY, n_groups=5, k_test=2, embargo_frac=0.10, seed=0)
    first = cv["splits"][0]
    assert cv["embargo_n"] == 5
    assert first["test_groups"] == [0, 1]
    assert first["purged_n"] >= 1
    assert first["embargoed_n"] == 5
    assert first["train_n_after_purge_embargo"] < 30


def test_determinism_and_graceful_skip_are_inert():
    cv1 = R.cpcv(_overfit_seen(), BPY, n_groups=8, k_test=2, max_combos=12, seed=99)
    cv2 = R.cpcv(_overfit_seen(), BPY, n_groups=8, k_test=2, max_combos=12, seed=99)
    assert cv1 == cv2
    assert cv1["subsampled"] is True
    assert cv1["combos_used"] == 12
    assert cv1["combos_total"] == 28

    short = R.cpcv(pd.Series([0.1, -0.1, 0.2], index=_idx(3)), BPY)
    flat = R.cpcv(pd.Series([0.0] * 80, index=_idx(80)), BPY)
    assert short["ran"] is False and "too few" in short["reason"]
    assert flat["ran"] is False and "zero-variance" in flat["reason"]

    base = stages.p8_verdict(_claim(), _survivor_bt({}), {}, False)
    skipped = stages.p8_verdict(_claim(), _survivor_bt(short), {}, False)
    assert skipped.verdict == base.verdict
    assert skipped.kill_reason == base.kill_reason
