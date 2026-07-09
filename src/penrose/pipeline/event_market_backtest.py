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


def run_weather_tail_fade_backtest(
    panel: EventMarketPanel,
    *,
    fee_coeff: float | None = None,
    capacity_frac: float = 0.10,
    max_pair_gross: float | None = None,
    pair_cities: tuple[str, str] | list[str] | None = None,
    portfolio_notional: float = 1000.0,
    min_open_interest: float = 0.0,
    weighting: str = "capacity",
) -> tuple[pd.Series, pd.DataFrame, float, dict[str, Any]]:
    """Reconstruct the Kalshi weather tail-fade primitive from resolved brackets.

    For every traded tail bracket, the primitive holds NO from ``p_close`` to
    settlement. Daily portfolio returns are absolute reconstructed P&L divided
    by a declared portfolio notional, so thin capacity genuinely reduces the
    executable signal instead of being normalized away.
    """
    if not isinstance(panel, EventMarketPanel):
        raise TypeError("run_weather_tail_fade_backtest panel must be an EventMarketPanel")
    if capacity_frac < 0:
        raise ValueError("weather tail-fade capacity_frac must be >= 0")
    if portfolio_notional <= 0:
        raise ValueError("weather tail-fade portfolio_notional must be > 0")
    if max_pair_gross is not None and max_pair_gross < 0:
        raise ValueError("weather tail-fade max_pair_gross must be >= 0")
    coeff = float(config.FEE_CURVE["kalshi"]["fee_rate"] if fee_coeff is None else fee_coeff)
    if coeff < 0:
        raise ValueError("weather tail-fade fee_coeff must be >= 0")

    stats = _zero_weather_stats(int(panel.data["event_id"].nunique()) if len(panel.data) else 0)
    if len(panel.data) == 0:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="close_date")
        return (
            pd.Series(dtype=float, index=empty_idx, name="kalshi_weather_tail_fade_net"),
            pd.DataFrame(index=empty_idx),
            _DEFAULT_BARS_PER_YEAR,
            stats,
        )

    rows = panel.data.sort_values(
        ["close_time", "event_id", "strike_low", "strike_high"],
        kind="mergesort",
    )
    trades: list[dict[str, Any]] = []
    missing_fields: set[str] = set()
    gross_before_pair_cap = 0.0
    gross_after_pair_cap = 0.0
    raw_capacity = 0.0

    for _, row in rows.iterrows():
        underlying = row["underlying"] if isinstance(row["underlying"], dict) else {}
        city = str(underlying.get("city") or underlying.get("underlying") or row["event_id"]).strip()
        ticker = str(underlying.get("ticker") or row["event_id"]).strip()
        is_tail = _bool_value(underlying.get("is_tail"), "is_tail", missing_fields)
        volume = _float_field(underlying.get("volume"), "volume", missing_fields)
        open_interest = _float_field(underlying.get("open_interest"), "open_interest", missing_fields)
        if missing_fields:
            continue
        if not is_tail or volume <= 0.0:
            continue
        # Liquidity floor: only trade genuinely liquid markets. The implausible Sharpe on the full
        # universe is an artifact of harvesting the thinnest tradeable brackets; a real book can only
        # deploy where there is depth. Below the floor -> not tradeable (skip), which shrinks breadth,
        # reduces diversification, and pulls the Sharpe toward a deployable number.
        if open_interest < float(min_open_interest):
            continue
        price = _probability(float(row["entry_price"]), "p_close")
        capacity = float(capacity_frac) * min(volume, open_interest)
        raw_capacity += max(0.0, capacity)
        if capacity <= 0.0:
            continue
        outcome = int(row["outcome"])
        pnl_no = price if outcome == 0 else -(1.0 - price)
        fee = coeff * price * (1.0 - price)
        close_time = pd.Timestamp(row["close_time"])
        close_time = close_time.tz_convert("UTC") if close_time.tzinfo else close_time.tz_localize("UTC")
        trades.append({
            "close_date": close_time.normalize(),
            "city": city,
            "ticker": ticker,
            "price": price,
            "outcome": outcome,
            "pnl_no": float(pnl_no),
            "fee": float(fee),
            "net_unit": float(pnl_no - fee),
            "capacity_notional": float(capacity),
            "scaled_notional": float(capacity),
        })

    if missing_fields:
        raise ValueError("weather tail-fade rows missing fields: " + ", ".join(sorted(missing_fields)))
    if not trades:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="close_date")
        stats.update({
            "n_trades": 0,
            "n_events": int(panel.data["event_id"].nunique()),
            "raw_capacity_notional": float(raw_capacity),
            "capacity_frac": float(capacity_frac),
            "fee_coeff": float(coeff),
            "portfolio_notional": float(portfolio_notional),
        })
        return (
            pd.Series(dtype=float, index=empty_idx, name="kalshi_weather_tail_fade_net"),
            pd.DataFrame(index=empty_idx),
            _DEFAULT_BARS_PER_YEAR,
            stats,
        )

    trade_frame = pd.DataFrame(trades)
    if pair_cities and max_pair_gross is not None:
        pair = {str(c).strip() for c in list(pair_cities)[:2] if str(c).strip()}
        if pair:
            for day, idx in trade_frame.groupby("close_date", sort=True).groups.items():
                del day
                day_rows = trade_frame.loc[idx]
                pair_mask = day_rows["city"].astype(str).isin(pair)
                pair_gross = float(day_rows.loc[pair_mask, "scaled_notional"].sum())
                gross_before_pair_cap += pair_gross
                cap = float(max_pair_gross) * float(portfolio_notional)
                if pair_gross > cap > 0.0:
                    scale = cap / pair_gross
                    trade_frame.loc[day_rows.loc[pair_mask].index, "scaled_notional"] *= scale
                gross_after_pair_cap += float(trade_frame.loc[day_rows.loc[pair_mask].index, "scaled_notional"].sum())
        else:
            gross_before_pair_cap = gross_after_pair_cap = 0.0

    trade_frame["weighted_pnl"] = trade_frame["scaled_notional"] * trade_frame["net_unit"]
    daily = trade_frame.groupby("close_date", sort=True)
    if str(weighting).strip().lower() == "equal":
        # Diversified deployment: EQUAL risk-weight across the (liquidity-floored) tradeable markets each day,
        # not capacity-proportional. Capacity-weighting concentrates the book in the few gigantic markets
        # (OI up to ~500k) and manufactures an implausibly smooth Sharpe; a real book spreads risk across the
        # many liquid markets it can trade. The per-market capacity cap still gates WHICH markets are tradeable
        # (via min_open_interest); within that set each contributes equal risk.
        net = daily["net_unit"].mean().rename("kalshi_weather_tail_fade_net")
    else:
        net = (daily["weighted_pnl"].sum() / float(portfolio_notional)).rename(
            "kalshi_weather_tail_fade_net"
        )
    net.index = pd.DatetimeIndex(net.index, tz="UTC", name="close_date")
    city_positions = (
        trade_frame.pivot_table(
            index="close_date",
            columns="city",
            values="scaled_notional",
            aggfunc="sum",
            fill_value=0.0,
        )
        / float(portfolio_notional)
    )
    city_positions.index = pd.DatetimeIndex(city_positions.index, tz="UTC", name="close_date")
    city_positions = city_positions.reindex(net.index).fillna(0.0).sort_index(axis=1)
    bars_per_year = _trades_per_year(net.index, len(net))
    total_fee = float((trade_frame["scaled_notional"] * trade_frame["fee"]).sum() / float(portfolio_notional))
    gross_after = float(trade_frame["scaled_notional"].sum())
    stats = {
        "n_trades": int(len(trade_frame)),
        "n_events": int(panel.data["event_id"].nunique()),
        "n_days": int(len(net)),
        "total_net": float(net.sum()),
        "mean_net": float(net.mean()),
        "bars_per_year": float(bars_per_year),
        "fee_coeff": float(coeff),
        "total_fee": total_fee,
        "capacity_frac": float(capacity_frac),
        "portfolio_notional": float(portfolio_notional),
        "raw_capacity_notional": float(raw_capacity),
        "gross_notional_after_caps": gross_after,
        "capacity_binding": bool(gross_after < raw_capacity or raw_capacity < float(portfolio_notional) * len(net)),
        "max_pair_gross": None if max_pair_gross is None else float(max_pair_gross),
        "pair_cities": list(pair_cities or []),
        "pair_gross_before_cap": float(gross_before_pair_cap),
        "pair_gross_after_cap": float(gross_after_pair_cap),
        "per_position_mean_net_unit": float(trade_frame["net_unit"].mean()),
        "per_position_win_rate": float((trade_frame["net_unit"] > 0).mean()),
    }
    return net.sort_index(kind="mergesort"), city_positions, bars_per_year, stats


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


def _float_field(value: Any, name: str, missing: set[str]) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        missing.add(name)
        return 0.0
    if pd.isna(x):
        missing.add(name)
        return 0.0
    return x


def _bool_value(value: Any, name: str, missing: set[str]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    missing.add(name)
    return False


def _empty_net_series(name: str = "event_market_net") -> pd.Series:
    return pd.Series(
        dtype=float,
        index=pd.DatetimeIndex([], tz="UTC", name="close_time"),
        name=name,
    )


def _zero_stats(n_events: int = 0) -> dict[str, float | int]:
    return {"n_trades": 0, "n_events": int(n_events), "total_net": 0.0, "mean_net": 0.0,
            "bars_per_year": _DEFAULT_BARS_PER_YEAR}


def _zero_weather_stats(n_events: int = 0) -> dict[str, Any]:
    return {
        "n_trades": 0,
        "n_events": int(n_events),
        "n_days": 0,
        "total_net": 0.0,
        "mean_net": 0.0,
        "bars_per_year": _DEFAULT_BARS_PER_YEAR,
        "fee_coeff": float(config.FEE_CURVE["kalshi"]["fee_rate"]),
        "total_fee": 0.0,
        "capacity_frac": 0.10,
        "portfolio_notional": 1000.0,
        "raw_capacity_notional": 0.0,
        "gross_notional_after_caps": 0.0,
        "capacity_binding": False,
        "max_pair_gross": None,
        "pair_cities": [],
        "pair_gross_before_cap": 0.0,
        "pair_gross_after_cap": 0.0,
    }
