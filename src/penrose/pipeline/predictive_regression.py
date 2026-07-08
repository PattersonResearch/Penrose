"""Deterministic execution for predictive-regression claims.

This trusted module tests a declared predictor/target relationship directly and
then emits the same P7 triple used by trading, provided-series, and event-market
paths. For a horizon ``h`` it aligns ``X_t`` with ``Y_{t+h}``, keeps only
non-overlapping rows at positions ``0, h, 2h, ...``, freezes the sign and
z-score moments on the in-sample prefix rows whose target timestamp is still in
sample, and emits:

``net_t = sign(cov_IS(X_t, Y_{t+h})) * zscore_IS(X_t) * zscore_IS(Y_{t+h})``
``positions_t = sign(cov_IS(X_t, Y_{t+h})) * zscore_IS(X_t)``

The emitted observations are already non-overlapping. ``bars_per_year`` is the
observed sampling rate of that emitted series, not divided by ``h`` again, so
P7's ``sqrt(n)`` statistics are based on the same independent row count that is
actually emitted.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import p7_backtest
from .provided_series import _declared_inputs, _finite_datetime_series, _series_data


def _input_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("series")
            or value.get("base_series")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or ""
        ).strip()
    return str(value or "").strip()


def _regression_inputs(spec: dict) -> tuple[Any, Any]:
    predictor = (
        spec.get("predictor")
        or spec.get("predictor_series")
        or spec.get("x")
        or ""
    )
    target = (
        spec.get("target")
        or spec.get("target_series")
        or spec.get("y")
        or ""
    )
    inputs = _declared_inputs(spec)
    if not predictor and inputs:
        predictor = _input_name(inputs[0])
    if not target and len(inputs) >= 2:
        target = _input_name(inputs[1])
    return predictor, target


def _parse_horizon(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("periods") or value.get("days") or value.get("h") or value.get("value")
    if isinstance(value, (int, np.integer)):
        return max(1, int(value))
    if isinstance(value, float) and math.isfinite(value):
        return max(1, int(value))
    text = str(value or "").strip().lower()
    if not text:
        return 1
    m = re.search(r"(\d+)", text)
    return max(1, int(m.group(1))) if m else 1


def _derived_display_name(value: dict) -> str:
    transform = str(value.get("transform") or "").strip()
    base = str(value.get("base_series") or "").strip()
    if transform == "realized_vol":
        return f"realized_vol({base},{int(value.get('window') or 1)})"
    if transform:
        return f"{transform}({base})"
    return base


def _realized_vol(price: pd.Series, window: int) -> pd.Series:
    """Forward realized volatility: std of log returns over [t+1..t+window]."""
    w = max(1, int(window))
    logret = np.log(price).diff()
    future = pd.concat([logret.shift(-i) for i in range(1, w + 1)], axis=1)
    return future.std(axis=1, ddof=0).rename(f"realized_vol_{w}")


def _materialize_derived(series: pd.Series, spec: dict, horizon: int) -> pd.Series | None:
    transform = str(spec.get("transform") or "").strip()
    if transform == "realized_vol":
        return _realized_vol(series, int(spec.get("window") or horizon))
    if transform == "returns":
        return series.pct_change().rename("returns")
    if transform == "log_returns":
        return np.log(series).diff().rename("log_returns")
    return None


def _materialize_input(bundle, raw: Any, horizon: int) -> tuple[str, pd.Series | None, bool]:
    horizon_encoded = False
    if isinstance(raw, dict) and raw.get("kind") == "derived_series":
        base_name = _input_name(raw)
        base = _series_data(bundle, base_name)
        clean = _finite_datetime_series(base) if base is not None else None
        if clean is None:
            return base_name, None, False
        derived = _materialize_derived(clean, raw, horizon)
        name = _derived_display_name(raw)
        horizon_encoded = bool(raw.get("horizon_encoded") and raw.get("transform") == "realized_vol")
        return name, _finite_datetime_series(derived) if derived is not None else None, horizon_encoded
    name = _input_name(raw)
    data = _series_data(bundle, name)
    return name, _finite_datetime_series(data) if data is not None else None, False


def _horizon_encoded_realized_vol_window(raw: Any, horizon: int) -> int | None:
    if not (
        isinstance(raw, dict)
        and raw.get("kind") == "derived_series"
        and raw.get("transform") == "realized_vol"
        and raw.get("horizon_encoded")
    ):
        return None
    return max(1, int(raw.get("window") or horizon))


def _aligned_xy(predictor: pd.Series, target: pd.Series, horizon: int,
                *, target_horizon_encoded: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame({"x": predictor, "y": target}).sort_index(kind="mergesort")
    frame["target_time"] = pd.Series(frame.index, index=frame.index).shift(-horizon)
    if target_horizon_encoded:
        frame["y_h"] = frame["y"]
    else:
        frame["y_h"] = frame["y"].shift(-horizon)
    clean = frame[["x", "y_h", "target_time"]].replace([np.inf, -np.inf], np.nan)
    return clean.dropna(subset=["x", "y_h", "target_time"])


def _non_overlapping_xy(aligned: pd.DataFrame, horizon: int) -> pd.DataFrame:
    step = max(1, int(horizon))
    return aligned.iloc[::step].copy()


def _observations_per_year(index: pd.DatetimeIndex, n: int) -> float:
    if n <= 1:
        return 1.0
    span_days = (index.max() - index.min()).total_seconds() / 86400.0
    years = span_days / 365.25
    if years <= 0:
        return 1.0
    return float(n) / years


def _ols_stats(x: np.ndarray, y: np.ndarray, sign: float) -> dict:
    n = int(len(x))
    if n < 3:
        return {"beta": None, "t_stat": None, "r2": None, "n_is": n, "sign": sign}
    x_center = x - float(np.mean(x))
    y_center = y - float(np.mean(y))
    x_ss = float(np.dot(x_center, x_center))
    y_ss = float(np.dot(y_center, y_center))
    if x_ss <= 0.0 or y_ss <= 0.0:
        return {"beta": None, "t_stat": None, "r2": None, "n_is": n, "sign": sign}
    cov_num = float(np.dot(x_center, y_center))
    beta = cov_num / x_ss
    corr = cov_num / math.sqrt(x_ss * y_ss)
    corr = max(-1.0, min(1.0, corr))
    denom = max(1e-12, 1.0 - corr * corr)
    t_stat = corr * math.sqrt((n - 2) / denom)
    return {
        "beta": float(beta),
        "t_stat": float(t_stat),
        "r2": float(corr * corr),
        "n_is": n,
        "sign": sign,
    }


@dataclass
class PredictiveRegressionModule:
    """Trusted module object for one declared predictive-regression spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"predictive_regression_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "predictive_regression"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic predictive-regression executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        predictor_raw, target_raw = _regression_inputs(self.spec)
        if not _input_name(predictor_raw) or not _input_name(target_raw):
            return {"ok": False, "reason": "data_unavailable: predictor_and_target_required"}

        horizon = _parse_horizon(self.spec.get("horizon") or self.spec.get("h"))
        target_window = _horizon_encoded_realized_vol_window(target_raw, horizon)
        if target_window is not None and target_window != horizon:
            return {
                "ok": False,
                "needs_review": True,
                "reason": (
                    f"derived target window {target_window} != regression horizon {horizon}; "
                    "independent-observation invariant cannot hold"
                ),
            }
        predictor_name, x, _predictor_horizon_encoded = _materialize_input(
            bundle, predictor_raw, horizon)
        target_name, y, target_horizon_encoded = _materialize_input(
            bundle, target_raw, horizon)
        missing = []
        if x is None:
            missing.append(predictor_name)
        if y is None:
            missing.append(target_name)
        if missing:
            return {"ok": False, "reason": "data_unavailable: " + ", ".join(missing)}

        aligned = _non_overlapping_xy(
            _aligned_xy(x, y, horizon, target_horizon_encoded=target_horizon_encoded),
            horizon,
        )
        n = int(len(aligned))
        if n < 20:
            return {"ok": False, "reason": f"data_unavailable: insufficient_aligned_observations ({n})"}

        i = int(n * p7_backtest.IS_FRAC)
        is_frame = aligned.iloc[:i]
        if i < n:
            oos_start = aligned.index[i]
            moment_frame = is_frame[is_frame["target_time"] < oos_start]
        else:
            moment_frame = is_frame
        if len(moment_frame) < 3:
            return {"ok": False, "reason": "data_unavailable: insufficient_in_sample_observations"}

        x_is = moment_frame["x"].to_numpy(dtype="float64")
        y_is = moment_frame["y_h"].to_numpy(dtype="float64")
        x_mean = float(np.mean(x_is))
        y_mean = float(np.mean(y_is))
        x_sd = float(np.std(x_is, ddof=1))
        y_sd = float(np.std(y_is, ddof=1))
        if x_sd <= 0.0 or y_sd <= 0.0 or not (math.isfinite(x_sd) and math.isfinite(y_sd)):
            return {"ok": False, "reason": "predictive_regression_spec_invalid: degenerate_is_moments"}

        cov = float(np.cov(x_is, y_is, ddof=1)[0, 1])
        sign = 1.0 if cov > 0 else -1.0 if cov < 0 else 0.0
        zx = (aligned["x"] - x_mean) / x_sd
        zy = (aligned["y_h"] - y_mean) / y_sd
        net = (sign * zx * zy).rename("predictive_regression_net")
        positions = (sign * zx).rename("predictive_regression_position")
        # The series has already been h-subsampled above. Dividing by h again would
        # understate the emitted observation rate and would not affect P7's sqrt(n)
        # gates, which consume len(net) directly.
        bars_per_year = max(1.0, _observations_per_year(aligned.index, n))

        stats = _ols_stats(x_is, y_is, sign)
        stats.update({
            "predictor": predictor_name,
            "target": target_name,
            "horizon": int(horizon),
            "target_horizon_encoded": bool(target_horizon_encoded),
            "x_mean_is": x_mean,
            "x_sd_is": x_sd,
            "y_mean_is": y_mean,
            "y_sd_is": y_sd,
            "bars_per_year": float(bars_per_year),
            "n_is_moments": int(len(moment_frame)),
        })
        return {
            "ok": True,
            "net": net,
            "positions": positions,
            "bars_per_year": float(bars_per_year),
            "n_trades": int(len(net)),
            "regression": stats,
        }


def build_module(spec: dict, claim) -> PredictiveRegressionModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "predictive_regression"
    )
    return PredictiveRegressionModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
