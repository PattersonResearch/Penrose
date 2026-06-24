"""Calibration — NEGATIVE CONTROL (placebo). The single most important property of a
falsification tool: it must NOT pass a no-edge signal. This measures penrose's FALSE-POSITIVE
rate by running many independent no-edge strategies through the REAL robustness stack.

Construction (honest specificity test):
  * Returns are REAL BTC daily log-returns (coinbase-live) — real fat tails, vol clustering.
  * The signal is an AR(1) process generated INDEPENDENTLY of those returns (seeded per trial),
    so by construction it has ZERO predictive edge. Strategy trades sign(signal_t) and realizes
    the real next-day return.
  * ZERO cost by default — the hardest test. With no cost drag, expected net is exactly 0, so a
    "pass" can only come from the statistical gates failing to reject noise (DSR multiple-testing
    deflation, 3-fold sign-stability, regime lens, bootstrap edge CI, permutation). Costs are not
    allowed to do the gates' job.
  * Each trial is scoped to a fresh `placebo::crypto` family and logged=False, so trials don't
    deflate each other and the real ledger is never touched — this measures the PER-STRATEGY
    false-positive rate (what one researcher testing one noise signal would see).

Verdict accounting:
  research-supported = CATASTROPHIC false positive (a no-edge signal certified)
  watch              = soft false positive (no-edge flagged as worth attention)
  kill / insufficient_data = correct rejection

A well-calibrated detector: ~0% research-supported, low watch rate.

Run:  make calib-placebo   (or python scripts/calibration_placebo.py [N] [cost_bps])
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

from penrose import config                            # noqa: E402
from penrose.data import client                       # noqa: E402
from penrose.data.contract import Series              # noqa: E402
from penrose.pipeline import p7_backtest, stages      # noqa: E402
from penrose.brain import Claim                       # noqa: E402
from scripts._calib_common import ar1_signal as _ar1_signal  # noqa: E402
from scripts._calib_common import real_btc_returns as _real_btc_returns  # noqa: E402
from scripts._calib_common import verdict_for_signal as _verdict_for_signal  # noqa: E402

BARS_PER_YEAR = 365.0
HARD_FP = {"research-supported"}
SOFT_FP = {"watch"}
CORRECT = {"kill", "insufficient_data"}


def _multi_test_demo(ret: pd.Series, cost_frac: float, n: int = 40) -> tuple[int, int]:
    """Multiple-testing realism. The main test scores each noise signal at n_trials=1 (per-test FP
    rate). Here we data-mine n noise signals in ONE family WITH logging, so the DSR trial-count +
    cross-trial-variance penalty GROWS as more are tested — the realistic case of a researcher
    trying many ideas. Deflation should push the flagged (watch+) rate toward 0. Returns
    (flagged_deflated, n)."""
    import tempfile
    tmp = Path(tempfile.mktemp(suffix="_mt.tsv"))
    old_ledger = p7_backtest.LEDGER
    p7_backtest.LEDGER = tmp
    flagged = 0
    try:
        for k in range(n):
            sig = _ar1_signal(len(ret), np.random.default_rng(30_000 + k))
            pos = np.sign(sig[:-1]); pay = ret.values[1:]; idx = ret.index[1:]
            turn = np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0
            net = pd.Series(pos * pay - turn.astype(float) * cost_frac, index=idx)
            bt = p7_backtest.run_backtest(
                f"mt_{k}", net, pd.Series(np.abs(pos), index=idx), BARS_PER_YEAR,
                payoff=pd.Series(pay, index=idx), position_signed=pd.Series(pos, index=idx),
                cost_frac=cost_frac, family="placebo_multitest::crypto", log=True)
            claim = Claim(claim_id="mt", statement="", mechanism="", scope="", horizon="",
                          source_id="calib", source_span="", claimed_metric_quote="",
                          applicable_strategy_class="placebo")
            if stages.p8_verdict(claim, bt, {}, synthetic=False).verdict in ("watch", "research-supported"):
                flagged += 1
    finally:
        p7_backtest.LEDGER = old_ledger
        try:
            tmp.unlink()
        except OSError:
            pass
    return flagged, n


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cost_frac = (float(sys.argv[2]) if len(sys.argv) > 2 else 0.0) / 1e4
    ret = _real_btc_returns()
    print(f"[placebo] real returns: {ret.name} n={len(ret)}  trials={n_trials}  "
          f"cost={cost_frac*1e4:.1f}bps (zero = hardest specificity test)")

    tally = Counter()
    fps = []
    dsrs = []
    for k in range(n_trials):
        rng = np.random.default_rng(10_000 + k)
        sig = _ar1_signal(len(ret), rng)
        v, bt = _verdict_for_signal(sig, ret, cost_frac)
        tally[v] += 1
        dsrs.append(bt.get("dsr") or 0.0)
        if v in HARD_FP or v in SOFT_FP:
            fps.append((k, v, bt.get("dsr"), bt.get("oos_sharpe"),
                        (bt.get("three_fold") or {}).get("folds")))

    # Negative control #2: an UNRELATED real series (weather) used as the BTC signal. We use the
    # DAILY CHANGE in temperature, not the level: a raw temp is almost always positive, so
    # sign(temp) is a constant +1 -> a degenerate, non-trading position that lands insufficient_data
    # (the position never flips). Temperature CHANGES oscillate around zero, so the signal actually
    # trades — and since weather has no relation to BTC returns, the honest verdict is a clean kill.
    print("\n[placebo] unrelated-real-series controls (real series -> BTC returns):")
    b = client.fetch_bundle()
    for wk in ("weather_temp_ny", "weather_temp_lax", "weather_temp_chi"):
        wv = b.series.get(wk)
        if not isinstance(wv, Series):
            continue
        w = wv.data.astype(float).reindex(ret.index).ffill().bfill().diff().fillna(0.0)
        v, bt = _verdict_for_signal(w.values, ret, cost_frac)
        print(f"    {wk:18s} -> {v:18s} DSR={bt.get('dsr')} OOS_Sh={bt.get('oos_sharpe')}")

    # Multiple-testing realism: data-mine many noise signals in ONE family so DSR deflates.
    _mt_flagged, _mt_n = _multi_test_demo(ret, cost_frac, n=min(40, max(20, n_trials // 3)))
    print(f"\n[placebo] multiple-testing realism ({_mt_n} noise signals data-mined in one family):")
    print(f"    flagged (watch+) WITH DSR deflation: {_mt_flagged}/{_mt_n} = "
          f"{100*_mt_flagged/_mt_n:.0f}%  (vs ~4% per-test — the correction should push it toward 0)")

    hard = sum(tally[v] for v in HARD_FP)
    soft = sum(tally[v] for v in SOFT_FP)
    print("\n========== PLACEBO CALIBRATION ==========")
    print(f"verdict distribution over {n_trials} no-edge trials: {dict(tally)}")
    print(f"DSR over noise: mean={np.mean(dsrs):.3f} p95={np.percentile(dsrs,95):.3f} "
          f"max={np.max(dsrs):.3f}  (research-supported needs >0.95)")
    print(f"HARD false positives (research-supported): {hard}/{n_trials} = {100*hard/n_trials:.1f}%")
    print(f"SOFT false positives (watch):              {soft}/{n_trials} = {100*soft/n_trials:.1f}%")
    if fps:
        print("  flagged trials (seed, verdict, dsr, oos_sharpe, 3fold):")
        for f in fps[:12]:
            print("   ", f)
    ok = hard == 0
    print(f"\nPLACEBO {'PASS' if ok else 'FAIL'}: "
          f"{'no no-edge signal was certified' if ok else 'a NO-EDGE signal reached research-supported (calibration failure)'}.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
