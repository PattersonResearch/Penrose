"""Deterministic execution for provided-series statistic claims.

This claim type is already a pre-computed sample: each declared input series is a
per-observation net-P&L series. The executor therefore only pools those declared
observations and hands them to the normal P7/P8 statistic/verdict path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _declared_inputs(spec: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in spec.get("inputs") or []:
        key = str(name or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _series_data(bundle, name: str):
    item = bundle.get(name) if hasattr(bundle, "get") else None
    if item is None or getattr(item, "available", True) is False:
        return None
    data = getattr(item, "data", item)
    return data if isinstance(data, pd.Series) else None


def _finite_datetime_series(data: pd.Series) -> pd.Series | None:
    if not isinstance(data.index, pd.DatetimeIndex):
        return None
    vals = pd.to_numeric(data, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = vals.dropna()
    if len(vals) == 0:
        return None
    return pd.Series(vals.to_numpy(dtype="float64"), index=vals.index)


@dataclass
class ProvidedSeriesStatisticModule:
    """Trusted module object for one provided-series statistic spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"provided_series_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "provided_series_statistic"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic pooled provided-series statistic executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        inputs = _declared_inputs(self.spec)
        if not inputs:
            return {"ok": False, "reason": "data_unavailable: no_declared_inputs"}

        pooled: list[pd.Series] = []
        unavailable: list[str] = []
        for name in inputs:
            data = _series_data(bundle, name)
            clean = _finite_datetime_series(data) if data is not None else None
            if clean is None:
                unavailable.append(name)
            else:
                pooled.append(clean)

        if unavailable:
            return {"ok": False, "reason": "data_unavailable: " + ", ".join(unavailable)}

        net = pd.concat(pooled).sort_index(kind="mergesort")
        positions = pd.Series(1.0, index=net.index)
        return {
            "ok": True,
            "net": net,
            "positions": positions,
            "bars_per_year": 1.0,
            "n_trades": int(len(net)),
        }


def build_module(spec: dict, claim) -> ProvidedSeriesStatisticModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "provided_series_statistic"
    )
    return ProvidedSeriesStatisticModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
