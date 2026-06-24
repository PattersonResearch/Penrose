"""Calibration — detection-threshold SENSITIVITY sweep. The single-asset certify-IC floor (~0.15)
is not a fixed limitation; it is a function of statistical power. This maps how it moves with the
three levers that set power: sample size, transaction cost, (and see calibration_breadth for the
fourth, cross-sectional breadth). Complements the headline number with a defensible curve.

Construction matches calibration_injection (G-002-correct): the decision signal s[t] is paired with
its OWN forward return c*s[t] + eps[t+1] (no look-ahead), so labeled IC == effective IC. E2 off so
certification is reachable.

Findings (illustrative, 12 seeds/cell): floor falls with sample size (1yr~0.20, 5yr~0.15, 11yr~0.08)
and rises with cost (>=20bps -> 0.20). With cross-sectional breadth (calibration_breadth) it falls to
~0.02 at N=100. So the floor is a sample-power artifact; more time AND more breadth both pull it
toward the realistic 0.02-0.05 range.

Run:  python scripts/calibration_sensitivity.py [seeds]
"""
from __future__ import annotations

import os as _os
import sys
import tempfile as _tf
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")

from penrose import config                            # noqa: E402
config.COST_PROVENANCE = "measured"
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim                       # noqa: E402

IC_GRID = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)


def _claim():
    return Claim(claim_id="s", statement="", mechanism="", scope="", horizon="",
                 source_id="calib", source_span="", claimed_metric_quote="", applicable_strategy_class="s")


def _ar1(n, rng, phi=0.9):
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    return (s - s.mean()) / (s.std() or 1.0)


def _certify_rate(ic, T, cost_bps, seeds):
    cf = cost_bps / 1e4
    hits = 0
    for k in range(seeds):
        rng = np.random.default_rng(700 + k)
        s = _ar1(T, rng)
        eps = rng.normal(0, 0.01, T)
        c = ic * eps.std() / np.sqrt(max(1e-9, 1 - ic * ic))
        pos = np.sign(s[:-1])
        fwd = c * s[:-1] + eps[1:]                    # s[t] -> its own forward return (causal)
        turn = np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0
        idx = pd.date_range("2015-01-01", periods=T - 1, freq="D")
        net = pd.Series(pos * fwd - turn * cf, index=idx)
        bt = P7.run_backtest("s", net, pd.Series(1.0, index=idx), 252.0, cost_frac=cf,
                             payoff=pd.Series(fwd, index=idx), position_signed=pd.Series(pos, index=idx),
                             family="sens::x", log=False)
        d = stages.p8_verdict(_claim(), bt, {}, False)
        if d.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
            ho = P7.final_holdout_eval("s", net, 252.0, force=True)
            d = stages.p8_verdict(_claim(), bt, ho, False)
        if d.verdict == "research-supported":
            hits += 1
    return 100 * hits / seeds


def _floor(T, cost, seeds):
    for ic in IC_GRID:
        if _certify_rate(ic, T, cost, seeds) >= 50:
            return ic
    return None


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"[sensitivity] certify-IC floor (>=50% research-supported); {seeds} seeds/cell; single-asset.\n")
    print("vs SAMPLE SIZE (0 cost):")
    for T in [400, 800, 1500, 3000]:
        print(f"  T={T:>4} (~{T//252}yr daily): certify floor IC ~ {_floor(T, 0, seeds)}")
    print("vs COST (T=1500, ~5yr):")
    for cb in [0, 5, 10, 20]:
        print(f"  cost={cb:>2}bps: certify floor IC ~ {_floor(1500, cb, seeds)}")
    print("\nReading: the ~0.15 single-asset floor reflects ~5yr daily data + low cost. It falls with")
    print("sample size (->~0.08 at 11yr) and, cross-sectionally (calibration_breadth), to ~0.02 at N=100.")
    print("The detection limit is a sample-power artifact; more time AND more breadth lower it toward")
    print("the realistic 0.02-0.05 IC range. Higher cost raises it, as it should.")


if __name__ == "__main__":
    main()
