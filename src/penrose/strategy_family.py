"""Structured strategy-family identity.

This is advisory metadata: it scopes learning/distillation surfaces, not verdict gates.
"""
from __future__ import annotations

import re
from typing import Any


METHODS = {"single", "regime_blend", "ensemble", "overlay"}

_METHOD_ALIASES = {
    "single": "single",
    "standalone": "single",
    "regime": "regime_blend",
    "regime_blend": "regime_blend",
    "regime_blends": "regime_blend",
    "regime-blend": "regime_blend",
    "regime blend": "regime_blend",
    "regime blended": "regime_blend",
    "regime conditioned blend": "regime_blend",
    "ensemble": "ensemble",
    "ensembles": "ensemble",
    "blend": "ensemble",
    "basket": "ensemble",
    "overlay": "overlay",
    "overlays": "overlay",
}

_COMPONENT_ALIASES = {
    "carry": "carry",
    "funding": "carry",
    "funding_carry": "carry",
    "basis": "carry",
    "trend": "trend",
    "trend_following": "trend",
    "trendfollowing": "trend",
    "ewmac": "trend",
    "moving_average": "trend",
    "moving_average_crossover": "trend",
    "momentum": "momentum",
    "value": "value",
    "reversal": "mean_reversion",
    "mean_reversion": "mean_reversion",
    "vol": "volatility",
    "volatility": "volatility",
    "vrp": "volatility",
    "microstructure": "microstructure",
    "macro": "macro",
    "tail": "tail",
    "stat_arb": "stat_arb",
    "arbitrage": "arbitrage",
}

_COMPONENT_PATTERNS = [
    ("carry", ("funding carry", "carry", "funding", "basis")),
    ("trend", ("trend-following", "trend following", "trend", "ewmac",
               "moving average crossover", "moving-average crossover", "moving average")),
    ("momentum", ("momentum",)),
    ("value", ("value",)),
    ("mean_reversion", ("mean reversion", "mean-reversion", "reversal")),
    ("volatility", ("volatility", "realized vol", "implied vol", "vrp", "garch")),
    ("microstructure", ("order book", "liquidation", "microstructure", "depth")),
    ("macro", ("macro", "cpi", "fed", "recession", "inflation", "treasury")),
    ("tail", ("tail", "skew", "crash")),
    ("stat_arb", ("stat arb", "statistical arbitrage")),
    ("arbitrage", ("arbitrage",)),
]


def _slug(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def canonical_component(value: Any) -> str:
    slug = _slug(value)
    return _COMPONENT_ALIASES.get(slug, slug)


def canonical_method(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    slug = _slug(value)
    return _METHOD_ALIASES.get(text) or _METHOD_ALIASES.get(slug) or ""


def normalize_strategy_family(raw: Any) -> dict | None:
    """Return {"components": [...], "method": "..."} or None when unparseable."""
    if not isinstance(raw, dict):
        return None
    components_raw = raw.get("components")
    method_raw = raw.get("method")
    if not isinstance(components_raw, (list, tuple)):
        return None
    components = sorted({
        c for c in (canonical_component(x) for x in components_raw)
        if c
    })
    method = canonical_method(method_raw)
    if not components or method not in METHODS:
        return None
    return {"components": components, "method": method}


def _claim_text(claim: Any, source: Any | None = None) -> str:
    parts = [
        getattr(claim, "applicable_strategy_class", ""),
        getattr(claim, "statement", ""),
        getattr(claim, "mechanism", ""),
        getattr(claim, "source_span", ""),
        getattr(claim, "claimed_metric_quote", ""),
        getattr(source, "text", "") if source is not None else "",
    ]
    return " ".join(str(p or "") for p in parts).lower()


def infer_components_from_claim(claim: Any, source: Any | None = None) -> list[str]:
    text = _claim_text(claim, source)
    found: set[str] = set()
    for component, patterns in _COMPONENT_PATTERNS:
        if any(p in text for p in patterns):
            found.add(component)
    if found:
        return sorted(found)
    fallback = canonical_component(getattr(claim, "applicable_strategy_class", ""))
    return [fallback or "unspecified"]


def infer_method_from_claim(claim: Any, source: Any | None = None) -> str:
    text = _claim_text(claim, source)
    if any(p in text for p in ("regime blend", "regime-blend", "regime_blend",
                               "regime conditioned blend", "regime-conditioned blend")):
        return "regime_blend"
    if any(p in text for p in ("ensemble", "basket", "blend of", "blended")):
        return "ensemble"
    if "overlay" in text:
        return "overlay"
    return "single"


def declared_strategy_family(
    claim: Any,
    source: Any | None = None,
    raw: Any | None = None,
) -> dict:
    """Normalize a declared family or build the conservative simple default."""
    normalized = normalize_strategy_family(raw)
    if normalized is not None:
        return normalized
    return {
        "components": infer_components_from_claim(claim, source),
        "method": infer_method_from_claim(claim, source),
    }


def family_key(family: dict) -> str:
    normalized = normalize_strategy_family(family)
    if normalized is None:
        return ""
    return "+".join(normalized["components"]) + "::" + normalized["method"]
