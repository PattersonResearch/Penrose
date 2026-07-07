"""Deterministic execution for event-market bracket strategy claims."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .. import config
from ..data.event_market_load import EventMarketDataUnavailable, load_event_market
from .event_market_backtest import run_event_market_backtest


_ALLOWED_PRICING_FAMILIES = {"normal_bracket"}


@dataclass
class EventMarketStrategyModule:
    """Trusted module object for one declared event-market bracket spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __supports_param_override__ = True
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"event_market_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "event_market_strategy"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic event-market bracket executor."

    def evaluate(self, param_override: dict | None = None) -> dict:
        panel = load_event_market(self.spec, self.spec.get("data_dir") or config.DATA_DIR)
        pricing = _pricing_config(self.spec)
        family = str(pricing.get("family") or pricing.get("model") or "").strip()
        if family not in _ALLOWED_PRICING_FAMILIES:
            raise ValueError(f"unsupported event-market pricing model: {family or '<missing>'}")
        params = dict(pricing.get("params") or {})
        entry = _entry_config(self.spec)
        if param_override:
            _merge_param_override(params, entry, param_override)
        net, positions, bars_per_year, stats = run_event_market_backtest(
            panel,
            _normal_bracket_probability,
            params=params,
            min_ev=float(entry.get("min_ev", 0.0)),
            max_price=float(entry.get("max_price", 1.0)),
            kelly_fraction=float(entry.get("kelly_fraction", 1.0)),
            size_cap=float(entry.get("size_cap", 1.0)),
            seed=int(entry.get("seed", 0)),
        )
        return {
            "net": net,
            "positions": positions,
            "bars_per_year": float(bars_per_year),
            "n_trades": int(stats.get("n_trades", len(net))),
        }

    def run(self, bundle, claim, cost_frac, param_override: dict | None = None):  # noqa: ARG002
        try:
            return {"ok": True, **self.evaluate(param_override=param_override)}
        except EventMarketDataUnavailable as exc:
            return {"ok": False, "reason": str(exc)}
        except (TypeError, ValueError) as exc:
            # M-5: a degenerate pricing model (sigma<=0, non-numeric mu/sigma) is a SPEC defect, not
            # missing data — do NOT use the `data_unavailable:` prefix (which would park it at
            # needs_data and break the re-run loop). Route it via the module-failure path instead.
            return {"ok": False, "reason": f"event_market_spec_invalid: {exc}"}


def build_module(spec: dict, claim) -> EventMarketStrategyModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "event_market_strategy"
    )
    return EventMarketStrategyModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )


def _pricing_config(spec: dict) -> dict:
    pricing = spec.get("pricing_model") or spec.get("pricing") or spec.get("model") or {}
    return pricing if isinstance(pricing, dict) else {"family": str(pricing)}


def _entry_config(spec: dict) -> dict:
    entry = spec.get("entry") or spec.get("entry_rule") or spec.get("sizing") or {}
    if not isinstance(entry, dict):
        return {}
    out = dict(entry)
    sizing = spec.get("sizing")
    if isinstance(sizing, dict):
        out.update(sizing)
    return out


_ENTRY_PARAM_KEYS = {"min_ev", "max_price", "kelly_fraction", "size_cap", "seed"}


def _merge_param_override(params: dict, entry: dict, param_override: dict) -> None:
    """Apply declared-grid overrides to event-market pricing or entry params."""
    for key, value in dict(param_override or {}).items():
        if key in _ENTRY_PARAM_KEYS:
            entry[key] = value
        else:
            params[key] = value


def _normal_bracket_probability(underlying: Any, strike_low: float, strike_high: float,
                                params: dict[str, Any]) -> float:
    mu, sigma = _mu_sigma(underlying, params)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("normal_bracket sigma must be finite and > 0")
    hi = _normal_cdf((float(strike_high) - mu) / sigma)
    lo = _normal_cdf((float(strike_low) - mu) / sigma)
    return min(1.0, max(0.0, hi - lo))


def _mu_sigma(underlying: Any, params: dict[str, Any] | None = None) -> tuple[float, float]:
    params = params or {}
    if isinstance(underlying, dict):
        mu = params.get("mu", params.get("forecast", params.get(
            "spot", underlying.get("mu", underlying.get("forecast", underlying.get("spot"))))))
        sigma = params.get("sigma", params.get("vol", underlying.get("sigma", underlying.get("vol"))))
    elif isinstance(underlying, (list, tuple)) and len(underlying) >= 2:
        mu = params.get("mu", params.get("forecast", params.get("spot", underlying[0])))
        sigma = params.get("sigma", params.get("vol", underlying[1]))
    else:
        raise ValueError("normal_bracket underlying must declare mu/forecast/spot and sigma")
    try:
        return float(mu), float(sigma)
    except (TypeError, ValueError):
        raise ValueError("normal_bracket underlying mu/sigma must be numeric") from None


def _normal_cdf(z: float) -> float:
    if z == math.inf:
        return 1.0
    if z == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))
