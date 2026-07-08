"""Deterministic execution for cross-sectional sort claims."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .. import config
from ..data.panel_load import PanelDataUnavailable, load_cross_sectional_sort_panels
from ..data.xsection import _offset_alias, form_factor
from .predictive_regression import _observations_per_year


def _parse_n_buckets(value) -> int:
    try:
        return max(2, int(value))
    except (TypeError, ValueError):
        return 10


def _cadence_bars_per_year(rule: str, index: pd.DatetimeIndex, n: int) -> float:
    text = str(rule or "").strip().upper()
    if text in {"D", "1D"}:
        return 365.25
    if text in {"B", "1B"}:
        return 252.0
    if text in {"W", "1W", "W-FRI", "W-MON", "W-SUN"} or text.startswith("W-"):
        return 52.0
    if text in {"M", "ME", "1M", "1ME"}:
        return 12.0
    m = re.fullmatch(r"(\d+)M(?:E)?", text)
    if m:
        return max(1.0, 12.0 / int(m.group(1)))
    if text in {"Q", "QE", "1Q", "1QE"}:
        return 4.0
    if text in {"Y", "YE", "A", "AE", "1Y", "1YE", "1A", "1AE"}:
        return 1.0
    return max(1.0, _observations_per_year(index, n))


def _synthesize_positions(net: pd.Series, membership: dict, *, hold: str) -> pd.DataFrame:
    if net.empty:
        return pd.DataFrame(index=net.index)
    pieces: list[pd.DataFrame] = []
    max_date = net.index.max()
    for raw_rb, legs in sorted((membership or {}).items()):
        rb = pd.Timestamp(raw_rb)
        if rb.tzinfo is None:
            rb = rb.tz_localize("UTC")
        else:
            rb = rb.tz_convert("UTC")
        idx = net.index[(net.index > rb) & (net.index <= min(rb + _offset_alias(hold), max_date))]
        if len(idx) == 0:
            continue
        high = [str(x) for x in legs.get("high", [])]
        low = [str(x) for x in legs.get("low", [])]
        weights: dict[str, float] = {}
        if high:
            for name in high:
                weights[name] = weights.get(name, 0.0) + 0.5 / len(high)
        if low:
            for name in low:
                weights[name] = weights.get(name, 0.0) - 0.5 / len(low)
        if weights:
            pieces.append(pd.DataFrame(weights, index=idx))
    if not pieces:
        return pd.DataFrame({"cross_sectional_sort": 1.0}, index=net.index)
    positions = pd.concat(pieces).groupby(level=0).mean().sort_index()
    return positions.reindex(net.index).fillna(0.0)


@dataclass
class CrossSectionalSortModule:
    """Trusted module object for one declared cross-sectional sort spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"cross_sectional_sort_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "cross_sectional_sort"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic cross-sectional sort executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        try:
            returns, characteristic = load_cross_sectional_sort_panels(
                self.spec,
                self.spec.get("data_dir") or config.DATA_DIR,
            )
            n_buckets = _parse_n_buckets(self.spec.get("n_buckets") or self.spec.get("buckets"))
            rebalance = str(self.spec.get("rebalance") or "ME")
            hold = str(self.spec.get("hold") or "1M")
            min_names = int(self.spec.get("min_names") or max(2, n_buckets * 2))
            net = form_factor(
                returns,
                characteristic,
                n_buckets=n_buckets,
                rebalance=rebalance,
                hold=hold,
                min_names=min_names,
            ).rename("cross_sectional_sort_net")
            positions = _synthesize_positions(net, net.attrs.get("membership") or {}, hold=hold)
            bars_per_year = _cadence_bars_per_year(rebalance, net.index, len(net))
            return {
                "ok": True,
                "net": net,
                "positions": positions,
                "bars_per_year": float(bars_per_year),
                "n_trades": int(len(net)),
                "sort": {
                    "returns_panel": returns.name,
                    "characteristic_panel": characteristic.name,
                    "characteristic": str(self.spec.get("characteristic") or characteristic.name),
                    "n_buckets": int(n_buckets),
                    "rebalance": rebalance,
                    "hold": hold,
                    "min_names": int(min_names),
                    "membership_rebalances": int(len(net.attrs.get("membership") or {})),
                    "membership": net.attrs.get("membership") or {},
                },
            }
        except PanelDataUnavailable as exc:
            return {"ok": False, "reason": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "reason": f"cross_sectional_sort_spec_invalid: {exc}"}


def build_module(spec: dict, claim) -> CrossSectionalSortModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "cross_sectional_sort"
    )
    return CrossSectionalSortModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
