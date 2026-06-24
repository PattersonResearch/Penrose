"""Calibration: NATIVE-BREADTH recalibration (the IC-floor thesis).

The single-asset BTC injection showed penrose only certifies above IC ~0.15. The thesis here is
that floor is partly an ARTIFACT of the lowest-power regime: by the fundamental law of active
management, IR = IC * sqrt(breadth). A single asset traded daily has breadth ~252/yr, so a real
IC=0.05 edge is only Sharpe ~0.8 there — uncertifiable. The SAME edge tested CROSS-SECTIONALLY at
native breadth (N names) is Sharpe ~ IC*sqrt(N*252) — easily resolvable for N in the hundreds.

This harness measures that directly through penrose's REAL backtest + verdict stack: it injects a
known cross-sectional IC into an N-asset panel, forms a dollar-neutral z-weighted long-short
portfolio (decide at t on signal_t, realize r_{t+1} — no look-ahead), runs it through
run_backtest + p8_verdict, and sweeps IC x breadth. Output: the certification IC threshold per
breadth — it should fall ~1/sqrt(N), pulling real 0.03-0.05 edges from `underpowered` into testable.

E2 is turned off (measured-cost regime) so certification is reachable; zero cost = pure detection
floor. Run:  make calib-breadth
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

import os as _os, tempfile as _tf
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")
_os.path.exists(_os.environ["PENROSE_HOLDOUT_LOCK"]) and _os.unlink(_os.environ["PENROSE_HOLDOUT_LOCK"])
from penrose import config                            # noqa: E402
config.COST_PROVENANCE = "measured"                  # E2 off: let the certify path open (breadth test)
from penrose.pipeline import p7_backtest, stages      # noqa: E402
from penrose.brain import Claim                       # noqa: E402

BARS_PER_YEAR = 252.0
IC_GRID = [0.0, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]
BREADTHS = [1, 6, 25, 100]
T = 1400                                             # periods (so even N=1 has a fair n)


def _xs_trial(n_assets: int, ic: float, rng: np.random.Generator) -> str:
    """Inject cross-sectional IC into an N-asset panel; trade a dollar-neutral z-weighted book."""
    sig = rng.standard_normal((T, n_assets))
    if n_assets > 1:                                  # standardize cross-sectionally each period
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-9)
    eps = rng.standard_normal((T, n_assets))
    c = ic / np.sqrt(max(1e-9, 1 - ic * ic)) if ic > 0 else 0.0
    # fwd_t = the return EARNED over (t, t+1] by acting on signal_t; sig_t predicts it with corr=ic,
    # eps is the unpredictable part (not known at decision time -> causal, no look-ahead). The signal
    # is IID, so we must NOT shift sig vs return (a shift needs a PERSISTENT signal to carry the edge
    # across the lag — that was the bug). Pair each sig_t with its own forward return.
    fwd = c * sig + eps
    if n_assets > 1:
        w = sig / (np.abs(sig).sum(axis=1, keepdims=True) + 1e-9)   # dollar-neutral, unit gross
    else:
        w = np.sign(sig)                              # single asset: directional bet
    port = (w * fwd).sum(axis=1)
    idx = pd.date_range("2021-01-01", periods=len(port), freq="D")
    net = pd.Series(port, index=idx)
    pos = pd.Series(np.abs(w).sum(axis=1), index=idx)
    bt = p7_backtest.run_backtest("xs", net, pos, BARS_PER_YEAR, cost_frac=0.0,
                                  family=f"breadth_{n_assets}::xs", log=False)
    claim = Claim(claim_id="xs", statement="", mechanism="", scope="", horizon="",
                  source_id="calib", source_span="", claimed_metric_quote="",
                  applicable_strategy_class="xs")
    dec = stages.p8_verdict(claim, bt, {}, synthetic=False)
    if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
        ho = p7_backtest.final_holdout_eval("xs", net, BARS_PER_YEAR, force=True)
        dec = stages.p8_verdict(claim, bt, ho, synthetic=False)
    return dec.verdict


def main() -> None:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"[breadth] cross-sectional injection; T={T} periods; seeds/cell={per}; E2 off\n")
    print(f"{'breadth N':>9} | " + " ".join(f"IC={ic:<4}" for ic in IC_GRID) + " | certify-IC floor")
    print("-" * 88)
    floors = {}
    for N in BREADTHS:
        cells = []
        cert_floor = None
        for ic in IC_GRID:
            t = Counter()
            for k in range(per):
                t[_xs_trial(N, ic, np.random.default_rng(90_000 + N * 1000 + int(ic * 1000) * 7 + k))] += 1
            cert = 100 * t["research-supported"] / per
            cells.append(f"{cert:>3.0f}%")
            if cert_floor is None and cert >= 50 and ic > 0:
                cert_floor = ic
        floors[N] = cert_floor
        print(f"{N:>9} | " + " ".join(f"{c:<6}" for c in cells) + f" | {cert_floor}")
    print("-" * 88)
    print("certify-IC floor vs breadth (>=50% research-supported):")
    for N in BREADTHS:
        print(f"    N={N:>4}: IC ~ {floors[N]}")
    print("\nReadings:")
    print("  * IC=0 must certify 0% at every breadth (no edge, no false certification).")
    print("  * The floor should FALL ~1/sqrt(N): single-asset ~0.12-0.18, hundreds of names -> ~0.03-0.05.")
    print("  * This is the IC-floor thesis made measurable: penrose's '0.15 floor' was a single-asset")
    print("    artifact; at native cross-sectional breadth, realistic 0.03-0.05 edges become testable")
    print("    (so they earn a real verdict, not `underpowered`).")


if __name__ == "__main__":
    main()
