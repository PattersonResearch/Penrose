#!/usr/bin/env python
"""Calibration — the tail-risk / widow-maker control.

A widow-maker payoff (mostly small gains, rare large losses -> strongly negative skew, a left tail far
heavier than the right, yet a positive mean) is the canonical "looks great by Sharpe, then blows up"
trade. The opt-in tail-risk gate exists to flag exactly this shape. This control validates the DETECTOR
itself: it must flag widow-maker draws as tail-asymmetric and must NOT flag benign symmetric payoffs.
A detector that misses widow-makers (or false-flags benign returns) is a calibration failure.

Run:  make calib-tail   (or  python scripts/calibration_tail.py [N_per_arm])
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from penrose.pipeline.robustness import tail_metrics  # noqa: E402

N_BARS = 500


def widow_maker(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mostly small steady gains, rare large losses: positive mean, fat left tail."""
    r = rng.normal(0.0010, 0.004, n)
    n_shock = max(1, int(0.02 * n))
    idx = rng.choice(n, n_shock, replace=False)
    r[idx] = -rng.uniform(0.08, 0.20, n_shock)
    return r


def benign(n: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric returns with a small positive drift: the gate must leave these alone."""
    return rng.normal(0.0005, 0.02, n)


def main() -> int:
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    flagged_wm = sum(
        tail_metrics(widow_maker(N_BARS, np.random.default_rng(70_000 + k)))["asymmetric"]
        for k in range(n_per)
    )
    flagged_bn = sum(
        tail_metrics(benign(N_BARS, np.random.default_rng(80_000 + k)))["asymmetric"]
        for k in range(n_per)
    )
    catch = flagged_wm / n_per
    false = flagged_bn / n_per
    print("\n========== TAIL-RISK CALIBRATION ==========")
    print(f"widow-maker draws flagged tail-asymmetric : {flagged_wm}/{n_per} = {catch:.0%}")
    print(f"benign draws false-flagged                : {flagged_bn}/{n_per} = {false:.0%}")
    ok = catch >= 0.90 and false <= 0.05
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — the detector must catch widow-makers and spare benign payoffs")
    print("(the gate itself is opt-in; this validates the underlying tail detector)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
