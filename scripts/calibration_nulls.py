"""Calibration — the 5-null falsification battery.

A trustworthy falsifier must reject a strategy that shows performance on data with NO exploitable
signal but REALISTIC market mechanics. The placebo (calibration_placebo) is one such null;
this extends to five, each stripping economic signal while preserving a different
data mechanic that naive backtests mistake for edge:

  A. white_noise        — i.i.d. returns; pure selection bias (does multiple-testing deflation hold?)
  B. regime_switch_vol  — two-state vol (calm/stress) Markov; no mean signal (regime artifacts)
  C. bid_ask_bounce     — microstructure placebo: price bounces between bid/ask -> spurious
                          mean-reversion a careless strategy "trades" (catches timing/leakage)
  D. zero_alpha_factor  — a real-looking factor with ZERO true forward IC (cross-sectional null)
  E. garch              — GARCH(1,1) vol clustering, zero mean (fat tails + clustering, no edge)

For each null we generate N independent draws, trade the natural naive strategy on each, and run
it through penrose's real backtest + power-aware verdict. PASS = ~0% reach research-supported
(the catastrophic false positive); low watch rate. A null that certifies is a calibration failure.

Run:  make calib-nulls   (or python scripts/calibration_nulls.py [N_per_null])
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
config.COST_PROVENANCE = "measured"                  # E2 off: any certification must be earned by the gates
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim                       # noqa: E402

BARS_PER_YEAR = 252.0
T = 1400


def _claim() -> Claim:
    return Claim(claim_id="null", statement="no-edge null", mechanism="", scope="", horizon="",
                 source_id="calib", source_span="", claimed_metric_quote="",
                 applicable_strategy_class="null")


COST_BPS = 5.0   # realistic round-trip per unit turnover (liquid equities). The battery MUST charge
                 # cost: the bid-ask-bounce microstructure null is a high-turnover spurious edge the
                 # statistical gates alone cannot reject — only cost falsifies it (verified: 8% FP at
                 # 0bps -> 0% at >=2bps). Running zero-cost both hid this and gave a luck-dependent pass.


def _verdict(net: pd.Series, payoff=None, pos_signed=None) -> str:
    idx = net.index
    cost = COST_BPS / 1e4
    # Subtract turnover cost from the realized P&L (run_backtest takes net as-is; it does NOT
    # re-derive net from positions+cost). High-turnover nulls (bounce) are thereby falsified.
    if pos_signed is not None:
        turn = pos_signed.reindex(idx).diff().abs().fillna(pos_signed.reindex(idx).abs())
        net = net - turn * cost
    bt = P7.run_backtest("null", net, pd.Series(1.0, index=idx), BARS_PER_YEAR, cost_frac=cost,
                         payoff=payoff, position_signed=pos_signed, family="nulls::x", log=False)
    dec = stages.p8_verdict(_claim(), bt, {}, synthetic=False)
    if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
        ho = P7.final_holdout_eval("null", net, BARS_PER_YEAR, force=True)
        dec = stages.p8_verdict(_claim(), bt, ho, synthetic=False)
    return dec.verdict


# --- the five null data-generating processes + their natural naive strategy ------------------
def null_white_noise(rng):
    r = rng.normal(0, 0.01, T)
    sig = np.sign(_ar1(rng))                          # persistent signal, independent of r -> no edge
    idx = _idx(); net = pd.Series(sig[:-1] * r[1:], index=idx[1:])
    return net, pd.Series(r[1:], index=idx[1:]), pd.Series(sig[:-1], index=idx[1:])


def null_regime_switch_vol(rng):
    # 2-state Markov vol, zero mean in both states (no exploitable mean signal)
    state, vols = 0, (0.006, 0.025)
    r = np.empty(T)
    for t in range(T):
        if rng.random() < 0.03:
            state ^= 1
        r[t] = rng.normal(0, vols[state])
    sig = np.sign(_ar1(rng))
    idx = _idx(); return pd.Series(sig[:-1] * r[1:], index=idx[1:]), pd.Series(r[1:], index=idx[1:]), pd.Series(sig[:-1], index=idx[1:])


def null_bid_ask_bounce(rng):
    # microstructure: efficient price is a flat random walk with ZERO drift; observed price bounces
    # +/- half-spread each bar -> spurious negative autocorrelation a reversal strategy "exploits".
    half_spread = 0.0008
    eff = np.cumsum(rng.normal(0, 0.004, T))          # zero-drift efficient log-price
    obs = eff + half_spread * rng.choice([-1, 1], T)  # bid-ask bounce
    px = np.exp(obs)
    ret = pd.Series(px, index=_idx()).pct_change().dropna()
    # naive 1-bar reversal: bet against the last move (the classic bounce trap)
    sig = -np.sign(ret.shift(1).fillna(0.0))
    net = sig * ret
    return net.dropna(), ret.reindex(net.dropna().index), sig.reindex(net.dropna().index)


def null_zero_alpha_factor(rng):
    # cross-sectional factor with ZERO true forward IC: signal & return independent across 50 names
    N = 50
    sig = rng.standard_normal((T, N)); sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-9)
    fwd = rng.standard_normal((T, N))                 # independent of sig -> IC 0
    w = sig / (np.abs(sig).sum(1, keepdims=True) + 1e-9)
    port = (w * fwd).sum(1)
    idx = _idx(); return pd.Series(port, index=idx), None, None


def null_garch(rng):
    # GARCH(1,1) vol clustering, zero mean -> fat tails + clustering, no edge
    w_, a_, b_ = 1e-6, 0.08, 0.90
    var = w_ / (1 - a_ - b_)
    r = np.empty(T)
    for t in range(T):
        var = w_ + a_ * (r[t - 1] ** 2 if t else 0.0) + b_ * var
        r[t] = rng.normal(0, np.sqrt(var))
    sig = np.sign(_ar1(rng))
    idx = _idx(); return pd.Series(sig[:-1] * r[1:], index=idx[1:]), pd.Series(r[1:], index=idx[1:]), pd.Series(sig[:-1], index=idx[1:])


def _ar1(rng, phi=0.9):
    s = np.zeros(T)
    for i in range(1, T):
        s[i] = phi * s[i - 1] + rng.normal()
    return s


def _idx():
    return pd.date_range("2015-01-01", periods=T, freq="D")


NULLS = [("A white_noise", null_white_noise), ("B regime_switch_vol", null_regime_switch_vol),
         ("C bid_ask_bounce", null_bid_ask_bounce), ("D zero_alpha_factor", null_zero_alpha_factor),
         ("E garch", null_garch)]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"[5-null] each null: {n} independent draws, naive strategy, real verdict (zero cost).\n")
    print(f"{'null':22s} {'kill':>5} {'underpwr':>9} {'watch':>6} {'res-sup':>8}  hard-FP%")
    print("-" * 64)
    overall_hard = 0
    for ni, (label, fn) in enumerate(NULLS):
        t = Counter()
        for k in range(n):
            # G-001: Python's hash() is per-process randomized (PYTHONHASHSEED) -> non-reproducible
            # calibration. Seed deterministically from the null's INDEX instead, so `make calib-nulls`
            # samples the same draws every run and a borderline PASS/FAIL can't flip on seed drift.
            rng = np.random.default_rng(20_000 + ni * 100_000 + k)
            out = fn(rng)
            net = out[0] if isinstance(out, tuple) else out
            payoff = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            ps = out[2] if isinstance(out, tuple) and len(out) > 2 else None
            t[_verdict(net, payoff, ps)] += 1
        hard = t.get("research-supported", 0)
        overall_hard += hard
        print(f"{label:22s} {t.get('kill',0):>5} {t.get('underpowered',0):>9} {t.get('watch',0):>6} "
              f"{hard:>8}  {100*hard/n:.0f}%")
    print("-" * 64)
    print(f"\nHARD false positives (research-supported on a NULL): {overall_hard}/{len(NULLS)*n} = "
          f"{100*overall_hard/(len(NULLS)*n):.1f}%")
    print(f"VERDICT: {'PASS — no null certified' if overall_hard == 0 else 'FAIL — a NULL reached research-supported'}")
    print("\nThis is the 5-null battery: a trustworthy falsifier must not certify")
    print("any data-generating process with no exploitable signal, even with realistic mechanics.")


if __name__ == "__main__":
    main()
