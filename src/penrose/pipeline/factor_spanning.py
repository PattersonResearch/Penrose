"""Deterministic execution for factor-spanning claims.

This trusted module tests whether a declared candidate factor has alpha after
controlling for a declared benchmark set. It fits the spanning regression on the
in-sample prefix only, freezes those betas, and emits the benchmark-hedged
residual return series to the ordinary P7 stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import p7_backtest
from .predictive_regression import _observations_per_year
from .provided_series import _finite_datetime_series, _series_data


_BENCHMARK_SET_DEFAULTS = {
    "capm": ["us_equity_ff3_mkt_rf"],
    "ff3": ["us_equity_ff3_mkt_rf", "us_equity_ff3_smb", "us_equity_ff3_hml"],
    "ff5": [
        "us_equity_ff5_mkt_rf",
        "us_equity_ff5_smb",
        "us_equity_ff5_hml",
        "us_equity_ff5_rmw",
        "us_equity_ff5_cma",
    ],
    "carhart": [
        "us_equity_ff3_mkt_rf",
        "us_equity_ff3_smb",
        "us_equity_ff3_hml",
        "us_equity_momentum_wml",
    ],
}


def _input_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("series")
            or value.get("candidate_factor")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or ""
        ).strip()
    return str(value or "").strip()


def _candidate_factor(spec: dict) -> Any:
    candidate = (
        spec.get("candidate_factor")
        or spec.get("candidate")
        or spec.get("factor")
        or spec.get("factor_series")
        or ""
    )
    inputs = spec.get("inputs") or []
    if not candidate and inputs:
        candidate = inputs[0]
    return candidate


def _benchmark_factors(spec: dict) -> list[Any]:
    explicit = (
        spec.get("benchmark_factors")
        or spec.get("benchmarks")
        or spec.get("controls")
        or []
    )
    if isinstance(explicit, str):
        explicit = [explicit]
    out = [x for x in explicit if _input_name(x)]
    if out:
        return out
    inputs = spec.get("inputs") or []
    if isinstance(inputs, list) and len(inputs) > 1:
        return [x for x in inputs[1:] if _input_name(x)]
    benchmark_set = str(spec.get("benchmark_set") or spec.get("model") or "").strip().lower()
    return list(_BENCHMARK_SET_DEFAULTS.get(benchmark_set, []))


def _aligned_factor_frame(candidate: pd.Series, benchmarks: dict[str, pd.Series]) -> pd.DataFrame:
    data = {"candidate": candidate}
    data.update(benchmarks)
    frame = pd.DataFrame(data).sort_index(kind="mergesort")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _ols_stats(x: np.ndarray, y: np.ndarray, benchmark_names: list[str]) -> dict:
    n = int(len(y))
    k = int(x.shape[1]) if x.ndim == 2 else 0
    if n <= k + 1 or k < 1:
        return {
            "alpha": None,
            "alpha_t_stat": None,
            "betas": {},
            "r2": None,
            "n_is": n,
        }
    design = np.column_stack([np.ones(n), x])
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "alpha": None,
            "alpha_t_stat": None,
            "betas": {},
            "r2": None,
            "n_is": n,
        }
    fitted = design @ coef
    resid = y - fitted
    ss_res = float(np.dot(resid, resid))
    y_center = y - float(np.mean(y))
    ss_tot = float(np.dot(y_center, y_center))
    df = n - k - 1
    if df <= 0 or ss_tot <= 0.0:
        alpha_t = None
    else:
        sigma2 = ss_res / df
        try:
            xtx_inv = np.linalg.pinv(design.T @ design)
            se_alpha = math.sqrt(max(0.0, float(sigma2 * xtx_inv[0, 0])))
            alpha_t = float(coef[0] / se_alpha) if se_alpha > 0.0 else None
        except np.linalg.LinAlgError:
            alpha_t = None
    return {
        "alpha": float(coef[0]),
        "alpha_t_stat": alpha_t,
        "betas": {
            name: float(beta)
            for name, beta in zip(benchmark_names, coef[1:], strict=False)
        },
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "n_is": n,
    }


@dataclass
class FactorSpanningModule:
    """Trusted module object for one declared factor-spanning spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"factor_spanning_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "factor_spanning"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic factor-spanning executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        candidate_raw = _candidate_factor(self.spec)
        benchmark_raw = _benchmark_factors(self.spec)
        candidate_name = _input_name(candidate_raw)
        benchmark_names = [_input_name(x) for x in benchmark_raw if _input_name(x)]
        if not candidate_name or not benchmark_names:
            return {"ok": False, "reason": "data_unavailable: candidate_and_benchmarks_required"}

        candidate_raw_data = _series_data(bundle, candidate_name)
        candidate = _finite_datetime_series(candidate_raw_data) if candidate_raw_data is not None else None
        benchmarks: dict[str, pd.Series] = {}
        missing = []
        if candidate is None:
            missing.append(candidate_name)
        for name in benchmark_names:
            raw_data = _series_data(bundle, name)
            clean = _finite_datetime_series(raw_data) if raw_data is not None else None
            if clean is None:
                missing.append(name)
            else:
                benchmarks[name] = clean
        if missing:
            return {"ok": False, "reason": "data_unavailable: " + ", ".join(missing)}

        aligned = _aligned_factor_frame(candidate, benchmarks)
        n = int(len(aligned))
        if n < 20:
            return {"ok": False, "reason": f"data_unavailable: insufficient_aligned_observations ({n})"}

        i = int(n * p7_backtest.IS_FRAC)
        if i <= len(benchmark_names) + 1:
            return {"ok": False, "reason": "data_unavailable: insufficient_in_sample_observations"}
        is_frame = aligned.iloc[:i]
        x_is = is_frame[benchmark_names].to_numpy(dtype="float64")
        y_is = is_frame["candidate"].to_numpy(dtype="float64")
        stats = _ols_stats(x_is, y_is, benchmark_names)
        betas = stats.get("betas") or {}
        if len(betas) != len(benchmark_names):
            return {"ok": False, "reason": "factor_spanning_spec_invalid: degenerate_is_regression"}

        beta_vec = np.array([float(betas[name]) for name in benchmark_names], dtype="float64")
        x_all = aligned[benchmark_names].to_numpy(dtype="float64")
        residual = aligned["candidate"].to_numpy(dtype="float64") - (x_all @ beta_vec)
        net = pd.Series(residual, index=aligned.index, name="factor_spanning_net")
        bars_per_year = max(1.0, _observations_per_year(aligned.index, n))

        exposures = {"candidate": 1.0}
        exposures.update({name: -float(betas[name]) for name in benchmark_names})
        positions = pd.DataFrame(
            {name: float(exposure) for name, exposure in exposures.items()},
            index=aligned.index,
        )
        gross_exposure = float(positions.abs().sum(axis=1).iloc[0])
        stats.update({
            "candidate": candidate_name,
            "benchmarks": list(benchmark_names),
            "benchmark_set": str(self.spec.get("benchmark_set") or ""),
            "bars_per_year": float(bars_per_year),
            "n_is_moments": int(len(is_frame)),
            "is_start": aligned.index[0].isoformat(),
            "is_end": aligned.index[i - 1].isoformat(),
            "oos_start": aligned.index[i].isoformat() if i < n else None,
            "position_exposures": exposures,
            "position_gross_exposure": gross_exposure,
        })
        return {
            "ok": True,
            "net": net,
            "positions": positions,
            "bars_per_year": float(bars_per_year),
            "n_trades": int(len(net)),
            "regression": stats,
            "position_exposures": exposures,
        }


def build_module(spec: dict, claim) -> FactorSpanningModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "factor_spanning"
    )
    return FactorSpanningModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
