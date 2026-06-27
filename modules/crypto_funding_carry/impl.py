"""crypto_funding_carry — OPERATOR-WRITTEN (trusted, in-process) module.

Tests the classic crypto **perpetual funding-carry** claim on REAL data
(binance-funding rates from the penrose-data catalog — provenance != synthetic):

    "A delta-neutral position that harvests the perpetual-swap funding rate earns a
     positive risk-adjusted return net of trading costs."

A delta-neutral carry is long spot + short perp (or the reverse). Because the two legs'
price moves cancel, the position's PnL is, to first order, JUST the funding stream it
collects. So the faithful payoff per period is the funding rate itself, signed by which
side we hold.

Faithfulness / no look-ahead (this is the whole point — penrose judges the translation):
  * The position for period t is chosen from funding OBSERVED UP TO t-1 only
    (`signal_t = sign(funding[t-1])`). We never peek at funding[t] to decide.
  * The realized carry over period t is `signal_t * funding[t]` — we receive the rate
    iff we positioned on the correct side. Funding is highly autocorrelated, so the
    past-sign rule is a real, tradeable signal, not a fitted one.
  * Trading cost (a full delta-neutral rebalance: unwind + reopen, both legs, both
    sides) is charged ONLY when the position flips, i.e. on turnover.

This module is deliberately simple and un-tuned: there are no free parameters to overfit,
which is exactly what makes its verdict trustworthy. Whatever DSR / 3-fold / regime /
bootstrap / holdout say, they say about the carry premium itself.

Signature: run(bundle, claim, cost_frac) -> contract dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Registry metadata (pipeline.run discovers modules via these):
__module_id__ = "crypto_funding_carry"
__strategy_class__ = "crypto-funding-carry"
__description__ = ("Harvests the crypto perpetual-swap funding rate with a delta-neutral "
                   "long-spot/short-perp carry. Use ONLY for claims that the funding rate "
                   "itself carries a harvestable risk premium — not for directional, "
                   "basis-term-structure, or cross-sectional factor claims.")
__strategy_class_aliases__ = ["crypto-funding-carry"]

BARS_PER_YEAR = 365.0          # one funding-carry decision per day; daily settlement

# Real funding series in the catalog, in preference order (BTC is the deepest/most liquid).
_FUNDING_KEYS = ("funding_btc", "funding_eth", "funding_sol")


def run(bundle, claim, cost_frac: float) -> dict:
    funding = None
    for key in _FUNDING_KEYS:
        s = bundle.get(key)
        if s is not None and getattr(s, "available", False) and getattr(s, "data", None) is not None:
            funding = s.data.dropna()
            break
    if funding is None or len(funding) < 60:
        # No real funding history -> data blocker, NOT a falsified claim.
        return {"ok": False, "reason": "data_unavailable: no funding series with >=60 obs"}

    funding = funding.sort_index()
    r = funding.to_numpy(dtype="float64")

    # signal_t = sign(funding[t-1])  -> decision uses only past funding (no look-ahead).
    # realized carry_t = signal_t * funding[t]  -> we receive the rate if positioned right.
    sig = np.sign(r[:-1])                 # length n-1, indexed to decide period t (=1..n-1)
    pay = r[1:]                           # funding realized over those same periods
    idx = funding.index[1:]

    carry = sig * pay                     # gross delta-neutral carry per period
    flip = np.abs(np.diff(np.concatenate([[0.0], sig]))) > 0      # position changed -> rebalance
    cost = flip.astype(float) * float(cost_frac)                  # full rebalance cost on flip only
    net = carry - cost

    if len(net) < 30:
        return {"ok": False, "reason": "data_unavailable: too few carry periods", "n": len(net)}

    net_s = pd.Series(net, index=idx, name="net")
    pos_signed = pd.Series(sig, index=idx, name="position_signed")
    positions = pd.Series(np.abs(sig), index=idx, name="positions")   # turnover/capacity (unit notional)
    payoff = pd.Series(pay, index=idx, name="payoff")

    return {"ok": True, "net": net_s, "positions": positions,
            "payoff": payoff, "position_signed": pos_signed,
            "bars_per_year": BARS_PER_YEAR, "n_trades": int(len(net_s))}
