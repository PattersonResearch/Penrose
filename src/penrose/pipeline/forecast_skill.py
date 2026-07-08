"""Deterministic execution for forecast-skill claims.

This trusted module tests whether a declared model forecast ``F_t`` predicts a
declared realized target ``Y_t`` more accurately than a declared benchmark
forecast ``B_t``. It emits the per-period squared-loss differential:

``net_t = (B_t - Y_t)^2 - (F_t - Y_t)^2``

Positive ``net_t`` means the model beats the benchmark for that target period.
The ordinary P7 DSR / sqrt(n) t-stat on this loss-differential series is the
Diebold-Mariano equal-predictive-accuracy test.

FSK-2: for an ``h``-step-ahead forecast (h>1) the per-period loss differentials
OVERLAP and are serially correlated up to ``h-1`` lags, which would make the
naive sqrt(n) standard error anti-conservative (over-stating the t-stat by
~sqrt(h)). To keep the DM test honest, the executor emits NON-OVERLAPPING loss
differentials — subsampling at positions ``0, h, 2h, ...`` — so each observation
is independent. (For nested models the Clark-West adjustment remains a future
refinement.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .predictive_regression import _observations_per_year, _parse_horizon
from .provided_series import _finite_datetime_series, _series_data


def _input_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("series")
            or value.get("forecast_series")
            or value.get("target_series")
            or value.get("benchmark_series")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or ""
        ).strip()
    return str(value or "").strip()


def _model_forecast_input(spec: dict) -> Any:
    value = (
        spec.get("model_forecast")
        or spec.get("forecast")
        or spec.get("model_forecast_series")
        or spec.get("forecast_series")
        or ""
    )
    inputs = spec.get("inputs") or []
    if not value and isinstance(inputs, list) and inputs:
        value = inputs[0]
    return value


def _target_input(spec: dict) -> Any:
    value = spec.get("target") or spec.get("target_series") or spec.get("realized_target") or ""
    inputs = spec.get("inputs") or []
    if not value and isinstance(inputs, list) and len(inputs) >= 2:
        value = inputs[1]
    return value


def _benchmark_spec(spec: dict) -> Any:
    return (
        spec.get("benchmark_forecast")
        or spec.get("benchmark")
        or spec.get("benchmark_series")
        or {}
    )


def _benchmark_method(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("method") or value.get("family") or value.get("benchmark") or value.get("kind")
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"random_walk", "rw", "naive", "persistence", "last_value"}:
        return "random_walk"
    if text in {"historical_mean", "expanding_mean", "mean", "expanding_historical_mean"}:
        return "historical_mean"
    return ""


def _construct_implied_benchmark(target: pd.Series, method: str) -> pd.Series:
    """Build a declared implied benchmark using only target values through t-1."""
    method = _benchmark_method(method)
    y = target.sort_index(kind="mergesort")
    if method == "random_walk":
        return y.shift(1).rename("forecast_skill_random_walk_benchmark")
    if method == "historical_mean":
        return y.expanding(min_periods=1).mean().shift(1).rename(
            "forecast_skill_historical_mean_benchmark"
        )
    raise ValueError(f"unsupported benchmark method: {method or 'missing'}")


def _aligned_forecast_frame(model: pd.Series, target: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"model": model, "target": target, "benchmark": benchmark})
    frame = frame.sort_index(kind="mergesort").replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["model", "target", "benchmark"])


@dataclass
class ForecastSkillModule:
    """Trusted module object for one declared forecast-skill spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"forecast_skill_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "forecast_skill"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic forecast-skill loss-differential executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        model_raw = _model_forecast_input(self.spec)
        target_raw = _target_input(self.spec)
        model_name = _input_name(model_raw)
        target_name = _input_name(target_raw)
        if not model_name or not target_name:
            return {"ok": False, "reason": "data_unavailable: model_forecast_and_target_required"}

        raw_model = _series_data(bundle, model_name)
        raw_target = _series_data(bundle, target_name)
        model = _finite_datetime_series(raw_model) if raw_model is not None else None
        target = _finite_datetime_series(raw_target) if raw_target is not None else None
        missing = []
        if model is None:
            missing.append(model_name)
        if target is None:
            missing.append(target_name)
        if missing:
            return {"ok": False, "reason": "data_unavailable: " + ", ".join(missing)}

        benchmark_raw = _benchmark_spec(self.spec)
        benchmark_name = _input_name(benchmark_raw)
        benchmark_method = _benchmark_method(benchmark_raw)
        benchmark_kind = "explicit"
        if benchmark_name:
            raw_benchmark = _series_data(bundle, benchmark_name)
            benchmark = _finite_datetime_series(raw_benchmark) if raw_benchmark is not None else None
            if benchmark is None:
                return {"ok": False, "reason": f"data_unavailable: {benchmark_name}"}
        elif benchmark_method:
            benchmark_kind = "implied"
            try:
                benchmark = _construct_implied_benchmark(target, benchmark_method)
            except ValueError as exc:
                return {"ok": False, "reason": f"forecast_skill_spec_invalid: {exc}"}
        else:
            return {"ok": False, "reason": "data_unavailable: declared_benchmark_required"}

        aligned = _aligned_forecast_frame(model, target, benchmark)
        # FSK-2: for an h-step-ahead forecast, consecutive loss differentials overlap (serial correlation
        # up to h-1 lags) and would inflate the DM t-stat by ~sqrt(h). Emit NON-OVERLAPPING observations
        # (subsample every h) so the sqrt(n) standard error is honest. h=1 leaves the series unchanged.
        horizon = _parse_horizon(
            self.spec.get("horizon") or self.spec.get("h") or getattr(claim, "horizon", None))
        if horizon > 1:
            aligned = aligned.iloc[::horizon].copy()
        n = int(len(aligned))
        if n < 20:
            return {"ok": False, "reason": f"data_unavailable: insufficient_aligned_observations ({n})"}

        model_loss = (aligned["model"] - aligned["target"]) ** 2
        benchmark_loss = (aligned["benchmark"] - aligned["target"]) ** 2
        net = (benchmark_loss - model_loss).rename("forecast_skill_loss_differential")
        positions = pd.Series(1.0, index=net.index, name="forecast_skill_unit_position")
        bars_per_year = max(1.0, _observations_per_year(net.index, n))
        return {
            "ok": True,
            "net": net,
            "positions": positions,
            "bars_per_year": float(bars_per_year),
            "n_trades": int(len(net)),
            "forecast_skill": {
                "model_forecast": model_name,
                "target": target_name,
                "benchmark": benchmark_name or benchmark_method,
                "benchmark_kind": benchmark_kind,
                "loss": str(self.spec.get("loss") or "squared_error"),
                "bars_per_year": float(bars_per_year),
                "mean_loss_differential": float(net.mean()),
                "n_observations": int(len(net)),
            },
        }


def build_module(spec: dict, claim) -> ForecastSkillModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "forecast_skill"
    )
    return ForecastSkillModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
