"""Bracket-market backtest front end.

This module turns settled event-market brackets into the same time-indexed
net-P&L series P7 already consumes. It does not route claims, load venue data,
or change the downstream robustness stack.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .. import config
from ..data.event_market import EventMarketPanel


PricingModel = Callable[[Any, float, float, dict[str, Any]], float]


def calc_ev(prob: float, entry_price: float) -> float:
    """Expected value of buying a binary bracket at ``entry_price``."""
    p = _probability(prob, "prob")
    price = _probability(entry_price, "entry_price")
    return p - price


def kalshi_taker_fee(entry_price: float) -> float:
    """Modeled Kalshi taker fee for a one-dollar binary contract."""
    price = _probability(entry_price, "entry_price")
    cfg = config.FEE_CURVE["kalshi"]
    return float(cfg["fee_rate"]) * price * (1.0 - price) * float(cfg["C"])


def run_event_market_backtest(
    panel: EventMarketPanel,
    p_model: PricingModel,
    params: dict[str, Any] | None = None,
    *,
    min_ev: float = 0.0,
    max_price: float = 1.0,
    kelly_fraction: float = 1.0,
    size_cap: float = 1.0,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series, float, dict[str, float | int]]:
    """Walk an EventMarketPanel and emit a P7-compatible net-P&L series.

    ``p_model`` receives only ``underlying``, bracket strikes, and declared
    params. Rows are evaluated in causal decision-time order. ``seed`` is part
    of the public signature for deterministic future tie-breaking; this first
    increment uses no randomness.
    """
    del seed
    if not isinstance(panel, EventMarketPanel):
        raise TypeError("run_event_market_backtest panel must be an EventMarketPanel")
    if not callable(p_model):
        raise TypeError("run_event_market_backtest p_model must be callable")
    max_price = _probability(max_price, "max_price")
    if kelly_fraction < 0:
        raise ValueError("run_event_market_backtest kelly_fraction must be >= 0")
    if size_cap < 0:
        raise ValueError("run_event_market_backtest size_cap must be >= 0")

    stats = _zero_stats(int(panel.data["event_id"].nunique()) if len(panel.data) else 0)
    empty = (_empty_net_series("event_market_net"), _empty_net_series("event_market_position"),
             _DEFAULT_BARS_PER_YEAR, stats)
    if len(panel.data) == 0:
        return empty

    rows = panel.data.sort_values(
        ["decision_time", "event_id", "strike_low", "strike_high"],
        kind="mergesort",
    )
    params = dict(params or {})
    nets: list[float] = []
    sizes: list[float] = []
    close_times: list[pd.Timestamp] = []

    for _, row in rows.iterrows():
        prob = _probability(
            p_model(row["underlying"], float(row["strike_low"]), float(row["strike_high"]), params),
            "p_model probability",
        )
        price = float(row["entry_price"])
        ev = calc_ev(prob, price)
        if ev < min_ev or price > max_price:
            continue
        size = _kelly_size(prob, price, kelly_fraction, size_cap)
        if size <= 0.0:
            # L-2: a zero-size position is not a trade — appending it would inflate the trade
            # count and the t-stat denominator with a zero-contribution observation.
            continue
        net = size * ((int(row["outcome"]) - price) - kalshi_taker_fee(price))
        nets.append(float(net))
        sizes.append(float(size))
        close_times.append(pd.Timestamp(row["close_time"]).tz_convert("UTC"))

    if not nets:
        return empty

    # L-1: emit EVERYTHING run_backtest needs — the net series, the POSITIONS (the Kelly sizes, which
    # P7 uses for turnover/capacity), and bars_per_year. This is a per-TRADE series, so the correct
    # annualizer is trades-per-year (NOT the calendar 252 other paths use); returning it here stops an
    # integrator from silently mis-annualizing Sharpe/DSR and false-killing a real bracket edge.
    index = pd.DatetimeIndex(close_times, tz="UTC", name="close_time")
    frame = pd.DataFrame({"net": nets, "position": sizes}, index=index).sort_index(kind="mergesort")
    net_series = frame["net"].rename("event_market_net")
    pos_series = frame["position"].rename("event_market_position")
    bars_per_year = _trades_per_year(net_series.index, len(net_series))
    return net_series, pos_series, bars_per_year, {
        "n_trades": int(len(net_series)),
        "n_events": stats["n_events"],
        "total_net": float(net_series.sum()),
        "mean_net": float(net_series.mean()),
        "bars_per_year": float(bars_per_year),
    }


# Fallback annualizer when trades-per-year can't be estimated (a single trade, or all at one instant):
# 1.0 = no annualization, the honest/conservative choice rather than a fabricated calendar rate.
_DEFAULT_BARS_PER_YEAR = 1.0


def _trades_per_year(index: pd.DatetimeIndex, n_trades: int) -> float:
    """Annualizer for a per-trade net series: trades observed per calendar year."""
    if n_trades <= 1:
        return _DEFAULT_BARS_PER_YEAR
    span_days = (index.max() - index.min()).total_seconds() / 86400.0
    years = span_days / 365.25
    if years <= 0:
        return _DEFAULT_BARS_PER_YEAR
    return float(n_trades) / years


def _kelly_size(prob: float, entry_price: float, fraction: float, cap: float) -> float:
    if cap <= 0 or fraction <= 0:
        return 0.0
    if entry_price >= 1.0:
        return 0.0
    full_kelly = max(0.0, (prob - entry_price) / (1.0 - entry_price))
    return min(float(cap), float(fraction) * full_kelly)


def _probability(value: float, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric probability in [0, 1]") from None
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return x


def _empty_net_series(name: str = "event_market_net") -> pd.Series:
    return pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([], tz="UTC", name="close_time"),
        name=name,
    )


def _zero_stats(n_events: int = 0) -> dict[str, float | int]:
    return {"n_trades": 0, "n_events": int(n_events), "total_net": 0.0, "mean_net": 0.0,
            "bars_per_year": _DEFAULT_BARS_PER_YEAR}
