"""macro_vol_btc — tradeable translation of 2604.01431v1's BTC channels.

Times the BTC volatility risk premium with a Kalshi macro signal:
  * standardize the signal on in-sample data (no look-ahead),
  * every 5 trading days enter a vol position sized by clip(z, -1, 1),
  * payoff per unit vega = pos * (realized_vol[t..t+5] - implied_vol[t]) - cost.

Returns the per-trade net series + positions for P7 to score under DSR / 3-fold /
locked holdout / capacity.

Two run signatures supported (the registry picks whichever the caller uses):
  * run(bundle, channel: 'fed' | 'recession', cost_frac)        — legacy
  * run(bundle, claim, cost_frac)                                — generic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNEL_SIGNAL = {"fed": "kxfed_signal", "recession": "kxrecssnber_signal"}
HOLD_DAYS = 5
BARS_PER_YEAR = 365 / HOLD_DAYS          # ~73 non-overlapping 5d trades / yr

# Registry metadata (penrose pipeline.run discovers modules via these):
__module_id__ = "macro_vol_btc"
# Accept both the canonical config.STRATEGY_CLASS_VOL name and the natural
# LLM-extracted phrasing. The registry checks all aliases.
__strategy_class__ = "macro-signal-volatility-forecast"
__description__ = ("Times the crypto volatility risk premium (realized vs implied vol) "
                   "with a macro prediction-market signal: daily probability changes in "
                   "Kalshi macro contracts (Fed rate, CPI, recession). Use ONLY for claims "
                   "that a macro / prediction-market signal forecasts crypto (BTC/ETH/SOL) "
                   "realized volatility — not for microstructure, options-IV, or pure "
                   "cross-sectional equity claims.")
__strategy_class_aliases__ = [
    # NARROW on purpose (A-016): this module is BTC-only + Fed/recession-signal only. The
    # broad "Volatility Forecasting" aliases collapsed unrelated ETH/CPI/equity vol claims
    # onto it -> meaningless verdicts. Only the specific macro-vol-forecast class routes here;
    # anything else generates its own spec/module.
    "macro-signal-volatility-forecast",
]


def _align(bundle, sig_name: str | None = None) -> pd.DataFrame:
    """Align the inputs for ONE channel. Only the channel's own signal is required — NOT both
    channels. Requiring kxfed AND kxrecssnber on every date over-constrained the join: on real
    data the two Kalshi channels barely overlap, collapsing a 73-trade Fed test to ~19. Each
    channel is tested on its own signal ∩ realized-vol ∩ implied-vol."""
    keys = (["kxfed_signal", "kxrecssnber_signal"] if sig_name is None else [sig_name])
    keys = keys + ["btc_realized_vol_5d", "btc_implied_vol"]
    cols = {}
    for key in keys:
        s = bundle.get(key)
        cols[key] = s.data if (s is not None and getattr(s, "available", False)) else None
    df = pd.DataFrame({k: v for k, v in cols.items() if v is not None})
    return df.dropna()


def run(bundle, channel: str, cost_frac: float, is_frac: float = 0.50) -> dict:
    """Produce per-trade net returns + positions for one channel."""
    sig_name = CHANNEL_SIGNAL[channel]
    df = _align(bundle, sig_name)
    if sig_name not in df.columns or len(df) < 60:
        return {"ok": False, "reason": "aligned data too short", "n": len(df)}

    # realized vol over the NEXT HOLD_DAYS window, observed at settlement (no look-ahead
    # at entry: position uses only signal up to t; payoff realizes over [t, t+HOLD]).
    fut_rv = df["btc_realized_vol_5d"].shift(-HOLD_DAYS)
    iv_entry = df["btc_implied_vol"]

    # in-sample standardization params (fit on first is_frac only)
    cut = int(len(df) * is_frac)
    mu = df[sig_name].iloc[:cut].mean()
    sd = df[sig_name].iloc[:cut].std(ddof=1) or 1.0
    z = ((df[sig_name] - mu) / sd).clip(-1, 1)

    # non-overlapping every HOLD_DAYS
    idx = np.arange(0, len(df) - HOLD_DAYS, HOLD_DAYS)
    rows = []
    for i in idx:
        ts = df.index[i]
        pos = float(z.iloc[i])
        payoff = pos * (float(fut_rv.iloc[i]) - float(iv_entry.iloc[i]))
        net = payoff - abs(pos) * cost_frac          # cost scales with position size
        rows.append((ts, net, pos))
    if not rows:
        return {"ok": False, "reason": "no trades", "n": 0}

    net = pd.Series([r[1] for r in rows], index=[r[0] for r in rows], name="net")
    pos = pd.Series([abs(r[2]) for r in rows], index=[r[0] for r in rows], name="pos")
    return {"ok": True, "net": net, "positions": pos,
            "bars_per_year": BARS_PER_YEAR, "n_trades": len(net)}
