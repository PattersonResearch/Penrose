"""Calibration — persistence-matched nulls.

Destroys drift in real BTC returns while preserving either no persistence (per-period flips) or
within-block persistence (block flips). PASS means no drift-destroyed null is certified; the printed
watch+ gap quantifies how much the certifier responds to persistent direction.

Run: make calib-persistence  (or python scripts/calibration_persistence.py [N] [cost_bps])
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._calib_common import BARS_PER_YEAR, real_btc_returns, verdict_for_signal  # noqa: E402
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim  # noqa: E402

WATCH_PLUS = {"watch", "research-supported"}
HARD_FP = {"research-supported"}
BLOCK_LENGTHS = (6, 12, 24)


def per_period_flip(ret: pd.Series, rng: np.random.Generator) -> pd.Series:
    signs = rng.choice([-1.0, 1.0], size=len(ret))
    return pd.Series(ret.values * signs, index=ret.index, name="per_period_flip")


def block_flip(ret: pd.Series, rng: np.random.Generator, block_len: int) -> pd.Series:
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    signs = np.empty(len(ret), dtype=float)
    for start in range(0, len(ret), block_len):
        signs[start:start + block_len] = rng.choice([-1.0, 1.0])
    return pd.Series(ret.values * signs, index=ret.index, name=f"block_flip_{block_len}")


def dead_persistent_series(n: int = 1400, seed: int = 0, phi: float = 0.85) -> pd.Series:
    """Zero-drift AR(1) returns: exactly dead in mean, persistent in direction/vol."""
    if n < 40:
        raise ValueError("dead-state null requires at least 40 observations")
    rng = np.random.default_rng(seed)
    vals = np.zeros(n)
    for i in range(1, n):
        vals[i] = phi * vals[i - 1] + rng.normal(0.0, 0.01)
    vals = vals - vals.mean()
    return pd.Series(vals, index=pd.date_range("2018-01-01", periods=n, freq="D"))


def verdict_for_return_momentum(ret: pd.Series, cost_frac: float, *,
                                name: str = "persist") -> tuple[str, dict]:
    """Trade sign(prev return), realize current return."""
    if len(ret) < 40:
        raise ValueError("return series is too short for persistence calibration")
    sig = np.concatenate([[0.0], ret.values[:-1]])
    return verdict_for_signal(
        sig, ret, cost_frac,
        name=name, family="persistence::crypto", claim_id=name,
        statement="drift-destroyed persistence control", strategy_class="persistence")


def verdict_for_dead_persistent(seed: int = 0) -> str:
    net = dead_persistent_series(seed=seed)
    bt = P7.run_backtest(
        "dead_persistent", net, pd.Series(1.0, index=net.index), BARS_PER_YEAR,
        cost_frac=0.0, family="dead_persistent::crypto", log=False)
    claim = Claim(claim_id="dead", statement="dead persistent null", mechanism="", scope="",
                  horizon="", source_id="calib", source_span="", claimed_metric_quote="",
                  applicable_strategy_class="dead-null")
    return stages.p8_verdict(claim, bt, {}, synthetic=False).verdict


def run_persistence_battery(n_trials: int = 100, cost_frac: float = 0.0,
                            block_lengths: tuple[int, ...] = BLOCK_LENGTHS) -> dict:
    ret = real_btc_returns()
    if len(ret) < 300:
        raise ValueError("persistence calibration requires at least 300 return observations")
    rows: dict[str, Counter] = {"per-period": Counter()}
    for L in block_lengths:
        rows[f"block-{L}"] = Counter()
    for k in range(n_trials):
        pp = per_period_flip(ret, np.random.default_rng(40_000 + k))
        v, _ = verdict_for_return_momentum(pp, cost_frac, name="persist_pp")
        rows["per-period"][v] += 1
        for L in block_lengths:
            bf = block_flip(ret, np.random.default_rng(50_000 + L * 1000 + k), L)
            v, _ = verdict_for_return_momentum(bf, cost_frac, name=f"persist_b{L}")
            rows[f"block-{L}"][v] += 1
    return {"n": n_trials, "return_name": ret.name, "return_n": len(ret), "rows": rows}


def main() -> None:
    try:
        n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        cost_frac = (float(sys.argv[2]) if len(sys.argv) > 2 else 0.0) / 1e4
        res = run_persistence_battery(n_trials, cost_frac)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"persistence calibration unavailable: {e}") from None

    print(f"[persistence] real returns: {res['return_name']} n={res['return_n']} "
          f"trials={res['n']} cost={cost_frac*1e4:.1f}bps")
    print("\nnull          kill  underpwr  watch  res-sup  watch+%")
    print("-" * 58)
    hard = 0
    pp_watch = 0.0
    for label, tally in res["rows"].items():
        n = res["n"]
        watch_plus = sum(tally[v] for v in WATCH_PLUS)
        hard += sum(tally[v] for v in HARD_FP)
        if label == "per-period":
            pp_watch = 100 * watch_plus / n
        print(f"{label:12s} {tally.get('kill',0):5d} {tally.get('underpowered',0):9d} "
              f"{tally.get('watch',0):6d} {tally.get('research-supported',0):8d} "
              f"{100*watch_plus/n:7.1f}")
    print("-" * 58)
    print("Activation gap (watch+):")
    for label, tally in res["rows"].items():
        if not label.startswith("block-"):
            continue
        gap = 100 * sum(tally[v] for v in WATCH_PLUS) / res["n"] - pp_watch
        print(f"  {label} minus per-period: {gap:.1f} percentage points")
    print("\nHonest reading: Penrose responds to persistent multi-period direction; a block-preserving")
    print("null is therefore a disclosure control, not evidence of an economic mechanism by itself.")
    ok = hard == 0
    print(f"\nPERSISTENCE {'PASS' if ok else 'FAIL'}: "
          f"{'no drift-destroyed null was certified' if ok else 'a drift-destroyed null reached research-supported'}.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
