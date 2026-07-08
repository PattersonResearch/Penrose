"""Deterministic execution for event-study claims."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..data.event_calendar_load import (
    EventCalendarDataUnavailable,
    load_event_calendar,
)
from .factor_spanning import _input_name as _factor_input_name
from .predictive_regression import _observations_per_year
from .provided_series import _finite_datetime_series, _series_data


def _input_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("series")
            or value.get("return_series")
            or value.get("returns")
            or value.get("asset_returns")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or ""
        ).strip()
    return str(value or "").strip()


def _return_series_input(spec: dict) -> Any:
    value = (
        spec.get("return_series")
        or spec.get("returns")
        or spec.get("asset_returns")
        or spec.get("target")
        or ""
    )
    inputs = spec.get("inputs") or []
    if not value and isinstance(inputs, list) and inputs:
        value = inputs[0]
    return value


def _market_series_input(spec: dict) -> Any:
    value = spec.get("market_series") or spec.get("market_returns") or ""
    inputs = spec.get("inputs") or []
    if not value and isinstance(inputs, list) and len(inputs) >= 2:
        value = inputs[1]
    return value


def _parse_window(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, dict):
        value = value.get("days") or value.get("window") or [value.get("start"), value.get("end")]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            start = int(value[0])
            end = int(value[1])
            if start <= end:
                return start, end
        except (TypeError, ValueError):
            pass
    text = str(value or "").strip().lower()
    if text:
        import re

        nums = re.findall(r"[+-]?\d+", text)
        if len(nums) >= 2:
            start, end = int(nums[0]), int(nums[1])
            if start <= end:
                return start, end
    return default


def _parse_estimation_window(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("days") or value.get("periods") or value.get("length") or value.get("value")
    try:
        out = int(value)
        return max(5, out)
    except (TypeError, ValueError):
        return 60


def _baseline_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("family") or value.get("model") or value.get("baseline")
    text = str(value or "mean_adjusted").strip().lower().replace("-", "_")
    if text in {"market", "market_model", "ols_market_model"}:
        return "market_model"
    return "mean_adjusted"


def _utc_indexed(series: pd.Series) -> pd.Series:
    out = series.sort_index(kind="mergesort")
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.Series(out.to_numpy(dtype="float64"), index=idx, name=out.name)


def _event_positions(index: pd.DatetimeIndex, event_date: pd.Timestamp,
                     start_offset: int, end_offset: int) -> pd.DatetimeIndex:
    pos = index.searchsorted(event_date, side="left")
    start = pos + start_offset
    end = pos + end_offset
    if start < 0 or end >= len(index) or start > end:
        return pd.DatetimeIndex([], tz=index.tz)
    return index[start:end + 1]


def _market_model_params(ret: pd.Series, market: pd.Series) -> tuple[float, float] | None:
    frame = pd.DataFrame({"r": ret, "m": market}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5:
        return None
    x = frame["m"].to_numpy(dtype="float64")
    y = frame["r"].to_numpy(dtype="float64")
    x_var = float(np.var(x, ddof=1))
    if not math.isfinite(x_var) or x_var <= 0.0:
        return None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x_var)
    alpha = float(np.mean(y) - beta * np.mean(x))
    return alpha, beta


def _event_car_rows(
    returns: pd.Series,
    events: pd.DatetimeIndex,
    *,
    event_window: tuple[int, int],
    estimation_window: int,
    baseline: str,
    market: pd.Series | None = None,
) -> tuple[pd.Series, list[dict]]:
    idx = pd.DatetimeIndex(returns.index).sort_values()
    ret = returns.reindex(idx)
    market_aligned = market.reindex(idx) if market is not None else None
    cars: list[float] = []
    event_index: list[pd.Timestamp] = []
    rows: list[dict] = []
    last_accepted_window_end: pd.Timestamp | None = None
    start_offset, end_offset = event_window

    for raw_event in events:
        event_date = pd.Timestamp(raw_event)
        if event_date.tzinfo is None:
            event_date = event_date.tz_localize("UTC")
        else:
            event_date = event_date.tz_convert("UTC")
        event_window_idx = _event_positions(idx, event_date, start_offset, end_offset)
        if len(event_window_idx) == 0:
            rows.append({"event_date": event_date.isoformat(), "skipped": "event_window_out_of_range"})
            continue
        event_start = event_window_idx[0]
        event_end = event_window_idx[-1]
        if last_accepted_window_end is not None and idx.searchsorted(event_start, side="left") < idx.searchsorted(last_accepted_window_end, side="right"):
            rows.append({"event_date": event_date.isoformat(), "skipped": "overlaps_prior_event_window"})
            continue
        # ES-1: anchor the estimation window at the EVENT WINDOW START, not the event date. For a negative
        # start_offset (e.g. a [-5,+5] pre-event-drift window — the standard event-study convention) the
        # event window begins BEFORE event_date, so ending the baseline at event_date would fit it on
        # returns inside the event window and contaminate the CAR. Ending at event_start keeps the estimation
        # window strictly before the event window for every start_offset sign.
        estimation_end_pos = idx.searchsorted(event_start, side="left")
        estimation_start_pos = estimation_end_pos - int(estimation_window)
        if estimation_start_pos < 0 or estimation_end_pos <= estimation_start_pos:
            rows.append({"event_date": event_date.isoformat(), "skipped": "insufficient_pre_event_estimation_window"})
            continue
        estimation_idx = idx[estimation_start_pos:estimation_end_pos]
        if (
            last_accepted_window_end is not None
            and idx.searchsorted(estimation_idx[0], side="left")
            <= idx.searchsorted(last_accepted_window_end, side="left")
        ):
            rows.append({"event_date": event_date.isoformat(), "skipped": "estimation_window_overlaps_prior_event_window"})
            continue
        estimation_returns = ret.reindex(estimation_idx).dropna()
        if len(estimation_returns) < int(estimation_window):
            rows.append({"event_date": event_date.isoformat(), "skipped": "incomplete_pre_event_estimation_window"})
            continue
        event_returns = ret.reindex(event_window_idx).dropna()
        if len(event_returns) != len(event_window_idx):
            rows.append({"event_date": event_date.isoformat(), "skipped": "incomplete_event_window"})
            continue
        if baseline == "market_model":
            if market_aligned is None:
                rows.append({"event_date": event_date.isoformat(), "skipped": "missing_market_series"})
                continue
            params = _market_model_params(estimation_returns, market_aligned.reindex(estimation_idx))
            if params is None:
                rows.append({"event_date": event_date.isoformat(), "skipped": "degenerate_market_model"})
                continue
            alpha, beta = params
            expected = alpha + beta * market_aligned.reindex(event_window_idx)
            if expected.isna().any():
                rows.append({"event_date": event_date.isoformat(), "skipped": "incomplete_market_event_window"})
                continue
            abnormal = event_returns - expected.reindex(event_returns.index)
            baseline_detail = {"alpha": alpha, "beta": beta}
        else:
            mean_return = float(estimation_returns.mean())
            abnormal = event_returns - mean_return
            baseline_detail = {"mean": mean_return}
        car = float(abnormal.sum())
        cars.append(car)
        event_index.append(event_date)
        rows.append({
            "event_date": event_date.isoformat(),
            "event_window_start": event_start.isoformat(),
            "event_window_end": event_end.isoformat(),
            "estimation_start": estimation_idx[0].isoformat(),
            "estimation_end": estimation_idx[-1].isoformat(),
            "car": car,
            "baseline": baseline_detail,
        })
        last_accepted_window_end = event_end

    return pd.Series(cars, index=pd.DatetimeIndex(event_index), name="event_study_car"), rows


@dataclass
class EventStudyModule:
    """Trusted module object for one declared event-study spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"event_study_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "event_study"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic event-study executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        return_raw = _return_series_input(self.spec)
        return_name = _input_name(return_raw)
        if not return_name:
            return {"ok": False, "reason": "data_unavailable: return_series"}
        raw_returns = _series_data(bundle, return_name)
        returns = _finite_datetime_series(raw_returns) if raw_returns is not None else None
        if returns is None:
            return {"ok": False, "reason": f"data_unavailable: {return_name}"}
        returns = _utc_indexed(returns)
        event_window = _parse_window(self.spec.get("window") or self.spec.get("event_window"), (0, 5))
        estimation_window = _parse_estimation_window(
            self.spec.get("estimation_window") or self.spec.get("baseline_window")
        )
        baseline = _baseline_name(self.spec.get("baseline"))
        market = None
        market_name = ""
        if baseline == "market_model":
            market_raw = _market_series_input(self.spec)
            market_name = _factor_input_name(market_raw)
            if not market_name:
                return {"ok": False, "reason": "data_unavailable: market_series"}
            raw_market = _series_data(bundle, market_name)
            market = _finite_datetime_series(raw_market) if raw_market is not None else None
            if market is None:
                return {"ok": False, "reason": f"data_unavailable: {market_name}"}
            market = _utc_indexed(market)
        try:
            calendar = load_event_calendar(self.spec, self.spec.get("data_dir") or config.DATA_DIR)
        except EventCalendarDataUnavailable as exc:
            return {"ok": False, "reason": str(exc)}
        try:
            net, rows = _event_car_rows(
                returns,
                calendar.dates,
                event_window=event_window,
                estimation_window=estimation_window,
                baseline=baseline,
                market=market,
            )
        except (TypeError, ValueError) as exc:
            return {"ok": False, "reason": f"event_study_spec_invalid: {exc}"}
        if len(net) < 20:
            return {"ok": False, "reason": f"data_unavailable: insufficient_events ({len(net)})"}
        bars_per_year = max(1.0, _observations_per_year(net.index, len(net)))
        positions = pd.Series(1.0, index=net.index, name="event_study_position")
        return {
            "ok": True,
            "net": net,
            "positions": positions,
            "bars_per_year": float(bars_per_year),
            "n_trades": int(len(net)),
            "event_study": {
                "return_series": return_name,
                "market_series": market_name or None,
                "event_calendar": calendar.name,
                "event_calendar_provenance": calendar.provenance,
                "window": list(event_window),
                "estimation_window": int(estimation_window),
                "baseline": baseline,
                "bars_per_year": float(bars_per_year),
                "n_events_declared": int(len(calendar.dates)),
                "n_events_used": int(len(net)),
                "events": rows,
            },
        }


def build_module(spec: dict, claim) -> EventStudyModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "event_study"
    )
    return EventStudyModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
