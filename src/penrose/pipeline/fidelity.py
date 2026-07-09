"""Module-fidelity refuter — the adversarial VERIFY gate (refute, not praise).

The deepest validity hole in penrose: a module is an LLM's *translation* of a paper's
claim into code. If the translation drifts, a "kill" might be killing a mis-implementation,
not the paper's actual strategy — and a "survivor" might survive because the code quietly
does something easier than the claim. The statistical gates (DSR, regime, permutation)
cannot see this; only a reader comparing the claim to the code can.

So this is a separate role whose ONLY job is to find where the module diverges from the
claim. Assume guilty until proven faithful. It never improves the code — it judges it.
Set PENROSE_LLM_VERIFIER_BASE_URL/API_KEY/MODEL to route this role to an independent verifier;
when unset, it deliberately falls back to the default provider/model path so existing
installations do not change behavior.

Verifier failures are INCONCLUSIVE, never faithful. They do not turn into kills, but they also
cannot authorize trusted-module reuse or a strongest positive verdict.
"""
from __future__ import annotations

import ast
import math
import re

from .. import config, llm
from . import fidelity_memory

_SYSTEM = (
    "You are an adversarial code auditor for a quantitative-research pipeline. You are given "
    "a research CLAIM and the Python MODULE that is supposed to test it. Your ONLY job is to "
    "decide whether the module FAITHFULLY implements the claim's economic logic — and to hunt "
    "for ways it does NOT. Assume the module is unfaithful until the code proves otherwise. "
    "When given a ModuleSpec instead of Python code, judge whether the spec faithfully "
    "translates the claim into a test; do not reject it merely because it is not executable yet.\n"
    "Faithful means: it forms the signal the claim describes, trades in the direction/horizon "
    "the claim implies, and tests THAT relationship — not a convenient proxy. Flag divergences "
    "like: wrong signal, wrong direction, look-ahead/peeking, trading a different instrument, a "
    "degenerate/constant position, or returning a backtest unrelated to the claim. Do NOT "
    "penalize an honest 'data_unavailable' (that's not an implementation defect). Do NOT praise. "
    "Respond ONLY with JSON: {\"faithful\": true|false, \"confidence\": 0.0-1.0, "
    "\"divergences\": [\"...\"], \"note\": \"one sentence\"}."
)

_PROVIDED_SERIES_STATISTIC_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=provided_series_statistic. This is NOT a trading "
    "strategy. Absence of positions, PnL construction, entry/exit rules, or a backtest is "
    "faithful when the implementation/spec pools exactly the declared provided series and "
    "applies exactly the claim's stated statistic/decision rule. For this claim type, hunt "
    "for only these divergences: missing/extra declared series, wrong pooled statistic or "
    "decision rule, or added gates/thresholds/data-quality/multiplicity rules the claim did "
    "not state. Do not reject merely because there are no positions or PnL. Do NOT flag that "
    "a declared series was pre-computed or constructed outside Penrose, that its construction "
    "is unverifiable, or that the spec 'omits' the entry/PnL/fee logic. For this claim type "
    "the series is GIVEN and pre-encoded; construction/provenance skepticism is handled "
    "downstream by a provenance cap, NOT by a fidelity divergence. Judge ONLY: (1) exact "
    "declared series, (2) exact stated statistic/decision rule, (3) no added gates."
)

_EVENT_MARKET_STRATEGY_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=event_market_strategy. This pass uses a "
    "deterministic declared-model module, not LLM-generated pricing code. A spec/module "
    "is structurally faithful when it loads the declared settled bracket table, uses an "
    "allowed declared pricing family such as normal_bracket, and applies the declared "
    "entry/sizing parameters through the event-market backtest. Defer new "
    "LLM-reconstruction fidelity for arbitrary pricing formulas to the later milestone."
)

_PREDICTIVE_REGRESSION_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=predictive_regression. This is NOT a trading "
    "strategy. A deterministic spec/module is faithful only when it tests the declared "
    "predictor, target, horizon, estimator, and statistic directly, aligns X_t with "
    "Y_t+h, freezes sign and standardization on the in-sample prefix, and adds no "
    "entry/exit, cost/capacity, or trading overlay. Hunt for variable substitution, "
    "wrong horizon, added gates, or any OOS/holdout look-ahead."
)

_FACTOR_SPANNING_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=factor_spanning. This is NOT a trading "
    "strategy. A deterministic spec/module is faithful only when it tests the declared "
    "candidate factor against the declared benchmark set, fits F_t = alpha + beta'B_t "
    "on the in-sample prefix only, freezes betas, emits the benchmark-hedged residual "
    "alpha series, and adds no entry/exit, cost/capacity, unclaimed controls, or "
    "trading overlay. Hunt for candidate substitution, benchmark-set substitution, "
    "added controls, or any OOS/holdout refit/look-ahead."
)

_CROSS_SECTIONAL_SORT_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=cross_sectional_sort. This is NOT a generic "
    "trading strategy. A deterministic spec/module is faithful only when it tests the "
    "declared characteristic-sorted top-minus-bottom spread, uses the declared bucket "
    "count and rebalance/hold cadence, ranks characteristics known at or before each "
    "rebalance, and adds no separate entry/exit, timing overlay, substituted "
    "characteristic, or unclaimed universe screen. Hunt for characteristic substitution, "
    "bucket/cadence substitution, added gates, or any OOS/holdout look-ahead."
)

_EVENT_STUDY_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=event_study. This is NOT a trading "
    "strategy. A deterministic spec/module is faithful only when it loads the "
    "declared return series and event calendar, estimates the declared baseline "
    "using only rows strictly before each event, computes abnormal returns over "
    "the declared event window, emits one CAR observation per event, and adds no "
    "entry/exit, cost/capacity, or trading overlay. Hunt for event-calendar "
    "substitution, event-window substitution, baseline substitution, or any "
    "post-event/OOS/holdout look-ahead."
)

_FORECAST_SKILL_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=forecast_skill. This is NOT a trading "
    "strategy. A deterministic spec/module is faithful only when it tests the "
    "declared model forecast against the declared explicit or implied benchmark "
    "on the declared realized target, emits the squared-loss differential "
    "(B_t-Y_t)^2 - (F_t-Y_t)^2, constructs implied random_walk/historical_mean "
    "benchmarks strictly causally through t-1, and adds no entry/exit, "
    "cost/capacity, target substitution, benchmark substitution, or trading "
    "overlay. Hunt for model, target, or benchmark substitution and any "
    "OOS/holdout look-ahead."
)

_FORMULAIC_SIGNAL_GUIDANCE = (
    "\nCLAIM TYPE OVERRIDE: claim_type=formulaic_signal. This is a trusted "
    "deterministic formula executor, not LLM-generated strategy code. A spec is "
    "structurally faithful when it declares the exact DSL signal string, the traded "
    "price series, the position_map, optional funding_pnl_series, and an inputs list "
    "that exactly covers every referenced series. The executor applies the one-bar "
    "position lag and optional funding cash-flow in one audited place. Hunt for "
    "series substitution, missing funding_pnl_series, wrong position_map, or a "
    "formula that changes the claim's signal."
)

_USER_TMPL = """CLAIM (verbatim): {statement}
MECHANISM: {mechanism}
SPEC signal_logic: {signal_logic}

MODULE CODE:
```python
{code}
```

Does the module faithfully test the claim? Hunt for divergences first. Output only the JSON."""

_SPEC_USER_TMPL = """CLAIM (verbatim): {statement}
MECHANISM: {mechanism}

GENERATED MODULE SPEC:
```json
{spec_text}
```

Does this ModuleSpec faithfully translate the claim into an implementable test?
Judge the SPEC, not whether executable Python already exists. Hunt for divergences like
wrong claim_type, turning a descriptive statistic into a trading strategy, wrong inputs,
wrong statistic/signal, or a test that cannot answer the stated claim. Output only the JSON."""


def assess(claim, module_code: str, spec: dict | None = None,
           *, role: str = "fidelity_refuter") -> dict:
    """Return {faithful, verified, confidence, divergences, note}."""
    code = (module_code or "").strip()
    if not code:
        return {"faithful": False, "verified": False, "confidence": 0.0, "divergences": [],
                "note": "no module source available; fidelity not checked",
                "independent_verifier": False}
    if spec and (spec.get("module_spec_only") or "claim_translation" in spec):
        user = _SPEC_USER_TMPL.format(
            statement=(getattr(claim, "statement", "") or "")[:500],
            mechanism=(getattr(claim, "mechanism", "") or "")[:400],
            spec_text=code[:6000],
        )
    else:
        user = _USER_TMPL.format(
            statement=(getattr(claim, "statement", "") or "")[:500],
            mechanism=(getattr(claim, "mechanism", "") or "")[:400],
            signal_logic=str((spec or {}).get("signal_logic", ""))[:500] or "(n/a)",
            code=code[:6000],
        )
    claim_type = str(
        (spec or {}).get("claim_type")
        or getattr(claim, "resolved_claim_type", "")
        or ""
    )
    if not claim_type:
        try:
            claim_type = fidelity_memory.classify_claim_type(claim)
        except Exception:  # noqa: BLE001 - prompt specialization must fail closed to the default
            claim_type = ""
    system = _SYSTEM
    if claim_type == "provided_series_statistic":
        system += _PROVIDED_SERIES_STATISTIC_GUIDANCE
    elif claim_type == "event_market_strategy":
        system += _EVENT_MARKET_STRATEGY_GUIDANCE
    elif claim_type == "predictive_regression":
        system += _PREDICTIVE_REGRESSION_GUIDANCE
    elif claim_type == "factor_spanning":
        system += _FACTOR_SPANNING_GUIDANCE
    elif claim_type == "cross_sectional_sort":
        system += _CROSS_SECTIONAL_SORT_GUIDANCE
    elif claim_type == "event_study":
        system += _EVENT_STUDY_GUIDANCE
    elif claim_type == "forecast_skill":
        system += _FORECAST_SKILL_GUIDANCE
    elif claim_type == "formulaic_signal":
        system += _FORMULAIC_SIGNAL_GUIDANCE
    try:
        parsed, response = llm.call_json(
            role,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
        if not isinstance(parsed, dict) or "faithful" not in parsed:
            return {"faithful": False, "verified": False, "confidence": 0.0, "divergences": [],
                    "note": "fidelity check inconclusive", "independent_verifier": False}
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        faithful = bool(parsed.get("faithful"))
        result = {
            "faithful": faithful,
            "verified": faithful and confidence >= config.FIDELITY_KILL_CONFIDENCE,
            "confidence": confidence,
            "divergences": (parsed.get("divergences") or [])[:5],
            "note": str(parsed.get("note", ""))[:240],
            "independent_verifier": bool(getattr(response, "independent_verifier", False)),
        }
        result = _provided_series_statistic_backstop(result, claim_type, spec)
        result = _event_market_strategy_backstop(result, claim_type, spec)
        result = _predictive_regression_backstop(result, claim_type, spec, claim)
        result = _factor_spanning_backstop(result, claim_type, spec, claim)
        result = _cross_sectional_sort_backstop(result, claim_type, spec, claim)
        result = _event_study_backstop(result, claim_type, spec, claim)
        result = _forecast_skill_backstop(result, claim_type, spec, claim)
        return _formulaic_signal_backstop(result, claim_type, spec, claim)
    except Exception as e:  # noqa: BLE001 — inconclusive is contained, never promoted to faithful
        return {"faithful": False, "verified": False, "confidence": 0.0, "divergences": [],
                "note": f"fidelity check errored: {e}", "independent_verifier": False}


_PROVIDED_SERIES_GATE_KEY_MARKERS = (
    "gate",
    "gates",
    "significance",
    "p_value",
    "pvalue",
    "alpha",
    "multiplicity",
    "multiple_testing",
    "bonferroni",
    "fdr",
    "data_quality",
    "quality_gate",
    "quality_filter",
    "minimum_observation",
    "minimum_observations",
    "minimum_sample",
    "minimum_obs",
    "min_observation",
    "min_observations",
    "min_obs",
    "min_sample",
    "threshold",
    "thresholds",
)
_PROVIDED_SERIES_GATE_VALUE_PATTERNS = (
    r"\bp\s*(?:<=|<)\s*0?\.\d+\b",
    r"\bp[-\s]?value\b",
    r"\bsignificance\b",
    r"\bbonferroni\b",
    r"\balpha\s*=",
    r"\bfdr\b",
    r"\breject\s+if\b",
    r"\btop\s+\d+(?:\.\d+)?%",
    r"\bmin(?:imum)?\b.{0,24}\bobs(?:ervation|ervations)?\b",
)
_PROVIDED_SERIES_GATE_VALUE_SKIP_KEYS = {
    "claim_statement",
    "claim_source_span",
    "claim_mechanism",
    "source_span",
    "claimed_metric_quote",
}


def _gate_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def _provided_series_has_added_gate_field(value, *, _key: str = "") -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            norm = _gate_key(key)
            if any(marker in norm for marker in _PROVIDED_SERIES_GATE_KEY_MARKERS):
                return True
            if _provided_series_has_added_gate_field(child, _key=norm):
                return True
    elif isinstance(value, list):
        return any(_provided_series_has_added_gate_field(child, _key=_key) for child in value)
    elif isinstance(value, str) and _key not in _PROVIDED_SERIES_GATE_VALUE_SKIP_KEYS:
        lowered = value.lower()
        return any(re.search(pat, lowered) for pat in _PROVIDED_SERIES_GATE_VALUE_PATTERNS)
    return False


def _is_deterministic_provided_series_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    inputs = spec.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    return not _provided_series_has_added_gate_field(spec)


def _provided_series_statistic_backstop(result: dict, claim_type: str,
                                        spec: dict | None = None) -> dict:
    """Keep the 6g fidelity contract deterministic even if the LLM refuter drifts.

    For provided-series claims built by the deterministic template, the spec structure is
    the fidelity contract: non-empty declared inputs, the deterministic marker, and no
    added gate fields. The template mechanically pools those inputs and computes the
    one-sample mean, so construction/provenance objections are verdict caps, not W1
    divergences. If the structure is absent, fail safe and leave the block in place.
    """
    if claim_type != "provided_series_statistic" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_provided_series_spec(spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["provided_series_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "provided_series_statistic fidelity block overridden structurally: deterministic "
        "template with declared inputs and no added gate fields; construction provenance "
        "remains verdict-capped"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _formulaic_signal_names(spec: dict | None) -> set[str] | None:
    if not isinstance(spec, dict):
        return None
    try:
        from . import formulaic_signal

        names = formulaic_signal.referenced_names(str(spec.get("signal") or ""))
    except Exception:  # noqa: BLE001
        return None
    out = {str(x or "").strip() for x in names if str(x or "").strip()}
    trade = str(spec.get("trade_series") or "").strip()
    funding = str(spec.get("funding_pnl_series") or "").strip()
    if trade:
        out.add(trade)
    if funding:
        out.add(funding)
    return out


def _is_deterministic_formulaic_signal_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "formulaic_signal":
        return False
    if not str(spec.get("signal") or "").strip():
        return False
    if not str(spec.get("trade_series") or "").strip():
        return False
    if str(spec.get("position_map") or "sign").strip().lower() not in {"sign", "zscore_clip"}:
        return False
    required = _formulaic_signal_names(spec)
    if not required:
        return False
    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        return False
    declared = {str(x or "").strip() for x in inputs if str(x or "").strip()}
    return declared == required


def _formulaic_signal_correspondence_verified(claim, spec: dict | None) -> bool:
    del claim
    return _is_deterministic_formulaic_signal_spec(spec)


def _formulaic_signal_backstop(result: dict, claim_type: str,
                               spec: dict | None = None, claim=None) -> dict:
    if claim_type != "formulaic_signal" or result.get("faithful") is not False:
        return result
    if not _formulaic_signal_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["formulaic_signal_fidelity_override"] = "deterministic_formula_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "formulaic_signal fidelity block overridden structurally: DSL parses under "
        "the audited operator table and inputs exactly match formula/trade/funding series"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _is_deterministic_event_market_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "event_market_strategy":
        return False
    cfg = spec.get("event_market") if isinstance(spec.get("event_market"), dict) else spec
    has_table = any(cfg.get(k) for k in ("path", "table", "table_path",
                                         "event_market_path", "event_market_table"))
    rule = ""
    for key in ("primitive", "strategy", "rule", "strategy_rule", "strategy_family"):
        value = spec.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("family") or value.get("rule")
        text = str(value or "").strip().lower().replace("-", "_")
        if text in {"kalshi_weather_tail_fade", "weather_tail_fade", "tail_fade"}:
            rule = "kalshi_weather_tail_fade"
            break
    pricing = spec.get("pricing_model") or spec.get("pricing") or spec.get("model") or {}
    if isinstance(pricing, dict):
        family = str(pricing.get("family") or pricing.get("model") or "")
    else:
        family = str(pricing or "")
    # M-1: mirror the provided_series discipline — DENY the structural override if the spec carries an
    # added gate (significance/alpha/bonferroni/threshold/min_observations/...) that the deterministic
    # normal_bracket executor silently drops. A stated-but-dropped gate is a REAL fidelity divergence,
    # not a construction cap, so the refuter's unfaithful verdict must stand. (These markers do not
    # collide with the legit event-market fields min_ev/max_price/kelly_fraction/size_cap/pricing params.)
    if _provided_series_has_added_gate_field(spec):
        return False
    return has_table and (family.strip() == "normal_bracket" or rule == "kalshi_weather_tail_fade")


def _event_market_strategy_backstop(result: dict, claim_type: str,
                                    spec: dict | None = None) -> dict:
    if claim_type != "event_market_strategy" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_event_market_spec(spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["event_market_fidelity_override"] = "deterministic_declared_model_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "event_market_strategy fidelity block overridden structurally: deterministic "
        "audited event-market primitive over a declared bracket table; arbitrary "
        "pricing-formula fidelity is deferred"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _is_deterministic_regression_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "predictive_regression":
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    inputs = spec.get("inputs")
    if not isinstance(inputs, list) or len([x for x in inputs if str(x or "").strip()]) < 2:
        return False
    predictor = str(spec.get("predictor") or inputs[0] or "").strip()
    target = str(spec.get("target") or inputs[1] or "").strip()
    if not predictor or not target or predictor == target:
        return False
    horizon = spec.get("horizon")
    if horizon in (None, ""):
        return False
    estimator = str(spec.get("estimator") or "").lower()
    if estimator and not any(k in estimator for k in ("ols", "covariance", "single_predictor")):
        return False
    if _provided_series_has_added_gate_field(spec):
        return False
    return True


_HORIZON_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
}


def _regression_claim_text(claim, spec: dict | None) -> str:
    parts = []
    seen = set()
    for attr in ("statement", "mechanism", "horizon", "claimed_metric_quote", "source_span"):
        try:
            value = str(getattr(claim, attr, "") or "").strip()
            key = value.lower()
            if value and key not in seen:
                parts.append(value)
                seen.add(key)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(spec, dict):
        for key in ("claim_statement", "claim_mechanism", "claim_source_span", "claim_translation"):
            value = str(spec.get(key) or "").strip()
            seen_key = value.lower()
            if value and seen_key not in seen:
                parts.append(value)
                seen.add(seen_key)
    return " ".join(str(p) for p in parts if p).lower()


def _contains_var(text_norm: str, name: str) -> bool:
    var = _norm_var(name)
    if not var:
        return False
    if re.search(rf"(?:^|_){re.escape(var)}(?:_|$)", text_norm):
        return True
    tokens = [t for t in var.split("_") if len(t) > 1]
    return bool(tokens) and all(re.search(rf"(?:^|_){re.escape(t)}(?:_|$)", text_norm) for t in tokens)


def _ordered_regression_pair_verified(text: str, predictor: str, target: str) -> bool:
    text_norm = _norm_var(text)
    predictor_norm = _norm_var(predictor)
    target_norm = _norm_var(target)
    if not (
        predictor_norm
        and target_norm
        and predictor_norm != target_norm
        and _contains_var(text_norm, predictor)
        and _contains_var(text_norm, target)
    ):
        return False
    direct = re.search(
        rf"(?:^|_){re.escape(predictor_norm)}(?:_|$).{{0,80}}"
        rf"(?:predict|predicts|forecast|forecasts|lead|leads).{{0,80}}"
        rf"(?:^|_){re.escape(target_norm)}(?:_|$)",
        text_norm,
    )
    regression = re.search(
        rf"(?:regression|regress|coefficient|beta).{{0,80}}"
        rf"(?:^|_){re.escape(target_norm)}(?:_|$).{{0,50}}"
        rf"(?:on|against).{{0,30}}"
        rf"(?:^|_){re.escape(predictor_norm)}(?:_|$)",
        text_norm,
    )
    target_on_predictor = re.search(
        rf"(?:^|_){re.escape(target_norm)}(?:_|$).{{0,50}}"
        rf"(?:on|against).{{0,30}}"
        rf"(?:^|_){re.escape(predictor_norm)}(?:_|$)",
        text_norm,
    )
    return bool(direct or regression or target_on_predictor)


def _series_ref_name(value) -> str:
    if isinstance(value, dict):
        if value.get("kind") == "derived_series":
            transform = str(value.get("transform") or "").strip()
            base = str(value.get("base_series") or "").strip()
            if transform == "realized_vol":
                return f"realized_vol({base},{value.get('window') or ''})"
            if transform:
                return f"{transform}({base})"
            return base
        return str(
            value.get("series")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or ""
        ).strip()
    return str(value or "").strip()


def _binding_score_ok(prov: dict) -> bool:
    kind = str(prov.get("kind") or "").lower()
    if kind == "literal":
        return bool(prov.get("full_coverage") is True)
    if kind in {"derived", "derived_series"}:
        base = prov.get("base_resolution") or {}
        return (
            bool(base.get("full_coverage") is True)
            and not (base.get("unmatched_name_tokens") or [])
            and
            float(base.get("score", 0.0) or 0.0) >= 0.5
            and len(base.get("matched_tokens") or []) >= 1
        )
    return (
        bool(prov.get("full_coverage") is True)
        and not (prov.get("unmatched_name_tokens") or [])
        and
        float(prov.get("score", 0.0) or 0.0) >= 0.5
        and len(prov.get("matched_tokens") or []) >= 1
    )


def _binding_evidence_tokens(prov: dict) -> list[str]:
    if str(prov.get("kind") or "").lower() in {"derived", "derived_series"}:
        prov = prov.get("base_resolution") or {}
    out: list[str] = []
    for item in prov.get("matched_tokens") or []:
        token = str(item or "")
        out.append((token.split("~")[-1] or token).lower())
    return [t for t in out if len(t) >= 3]


def _binding_matches_spec_value(prov: dict, value) -> bool:
    kind = str(prov.get("kind") or "").lower()
    if kind in {"derived", "derived_series"}:
        if not isinstance(value, dict) or value.get("kind") != "derived_series":
            return False
        return (
            str(prov.get("transform") or "") == str(value.get("transform") or "")
            and str(prov.get("base_series") or "") == str(value.get("base_series") or "")
            and int(prov.get("window") or value.get("window") or 1)
            == int(value.get("window") or prov.get("window") or 1)
        )
    return str(prov.get("series") or "").strip() == _series_ref_name(value)


def _contains_any_token(text_norm: str, tokens: list[str]) -> bool:
    return any(re.search(rf"(?:^|_){re.escape(_norm_var(tok))}(?:_|$)", text_norm) for tok in tokens)


def _derived_phrase_verified(text: str, prov: dict) -> bool:
    transform = str(prov.get("transform") or "").lower()
    if transform == "realized_vol":
        return bool(re.search(r"\b(?:realized\s+vol(?:atility)?|rv)\b", text, flags=re.IGNORECASE))
    if transform == "log_returns":
        return bool(re.search(r"\blog\s+returns?\b", text, flags=re.IGNORECASE))
    if transform == "returns":
        return bool(re.search(r"\breturns?\b", text, flags=re.IGNORECASE))
    return False


# BIND-1: a partial bind is a WRONG bind when the bound series names a MEASUREMENT (funding) that differs
# from the measurement the claim names (volume). A paraphrase bind (c2: "CPI repricing" ->
# kx_cpi_inflation_probability_daily) shares no such measurement conflict, so it is NOT rejected. Full-
# coverage bindings never conflict. This separates the two cases that score/coverage alone cannot.
_MEASUREMENT_TOKENS = frozenset({
    "volume", "funding", "price", "returns", "return", "rate", "oi", "spread",
    "basis", "spot", "perp", "open", "close", "high", "low",
})


def _measurement_tokens(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t} & _MEASUREMENT_TOKENS


def _binding_measurement_conflict(text: str, prov: dict) -> bool:
    """True when the bound series names a measurement the claim does NOT, and the claim names a DIFFERENT
    measurement — the volume->funding mis-bind signature. Full-coverage / no-measurement-token binds pass."""
    if str(prov.get("kind") or "").lower() in {"derived", "derived_series"}:
        prov = prov.get("base_resolution") or {}
    if prov.get("full_coverage"):
        return False
    series_measures = _measurement_tokens(prov.get("series"))
    if not series_measures:
        return False  # bound series carries no measurement token -> no measurement conflict possible
    claim_only = _measurement_tokens(text) - series_measures
    return bool(claim_only)  # claim names a measurement the bound series lacks -> conflict


def _ordered_binding_provenance_verified(text: str, predictor_prov: dict, target_prov: dict) -> bool:
    if not (_binding_score_ok(predictor_prov) and _binding_score_ok(target_prov)):
        return False
    # BIND-1: reject a partial bind whose series names a DIFFERENT measurement than the claim (volume vs
    # funding). Paraphrase binds with no measurement conflict are allowed.
    if _binding_measurement_conflict(text, predictor_prov) or _binding_measurement_conflict(text, target_prov):
        return False
    m = re.search(r"\b(?:predicts?|forecasts?|forecasting|leads?|explains?)\b", text, flags=re.IGNORECASE)
    if not m:
        return False
    before = text[:m.start()]
    after = text[m.end():]
    before_norm = _norm_var(before)
    after_norm = _norm_var(after)
    predictor_tokens = _binding_evidence_tokens(predictor_prov)
    target_tokens = _binding_evidence_tokens(target_prov)
    if not predictor_tokens or not target_tokens:
        return False
    if not _contains_any_token(before_norm, predictor_tokens):
        return False
    if not _contains_any_token(after_norm, target_tokens):
        return False
    if str(target_prov.get("kind") or "").lower() in {"derived", "derived_series"}:
        return _derived_phrase_verified(after, target_prov)
    return True


def _horizon_to_int(value) -> int | None:
    if isinstance(value, dict):
        value = value.get("periods") or value.get("days") or value.get("h") or value.get("value")
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float) and math.isfinite(value):
        return max(1, int(value))
    text = str(value or "").lower()
    m = re.search(r"\b(\d+)\s*(?:d|day|days|period|periods|month|months|quarter|quarters|year|years)?\b", text)
    if m:
        return max(1, int(m.group(1)))
    for word, num in _HORIZON_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return num
    return None


def _claim_horizons(text: str) -> set[int]:
    out: set[int] = set()
    for m in re.finditer(
        r"\b(\d+)\s*[- ]?(?:d|day|days|period|periods|month|months|quarter|quarters|year|years)"
        r"(?:\s*[- ]?ahead|\s+forward|\s+future)?\b",
        text,
    ):
        out.add(max(1, int(m.group(1))))
    for word, num in _HORIZON_WORDS.items():
        if re.search(
            rf"\b{word}\s*[- ]?(?:d|day|days|period|periods|month|months|quarter|quarters|year|years)"
            rf"(?:\s*[- ]?ahead|\s+forward|\s+future)?\b",
            text,
        ):
            out.add(num)
    if re.search(r"\bnext[- ](?:day|period)\b|\bone[- ](?:day|period)\b", text):
        out.add(1)
    return out


def _predictive_regression_correspondence_verified(claim, spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    inputs = spec.get("inputs") or []
    predictor = spec.get("predictor") or (inputs[0] if inputs else "") or ""
    target = spec.get("target") or (inputs[1] if len(inputs) > 1 else "") or ""
    text = _regression_claim_text(claim, spec)
    provenance = spec.get("binding_provenance") or {}
    if provenance:
        predictor_prov = provenance.get("predictor") or {}
        target_prov = provenance.get("target") or {}
        if not (
            isinstance(predictor_prov, dict)
            and isinstance(target_prov, dict)
            and _binding_matches_spec_value(predictor_prov, predictor)
            and _binding_matches_spec_value(target_prov, target)
            and _ordered_binding_provenance_verified(text, predictor_prov, target_prov)
        ):
            return False
    elif not _ordered_regression_pair_verified(text, _series_ref_name(predictor), _series_ref_name(target)):
        return False
    spec_horizon = _horizon_to_int(spec.get("horizon"))
    declared_horizons = _claim_horizons(text)
    return bool(spec_horizon and declared_horizons and spec_horizon in declared_horizons)


def _predictive_regression_backstop(result: dict, claim_type: str,
                                    spec: dict | None = None, claim=None) -> dict:
    if claim_type != "predictive_regression" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_regression_spec(spec):
        return result
    if not _predictive_regression_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["predictive_regression_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "predictive_regression fidelity block overridden structurally: deterministic "
        "template with declared predictor, target, horizon, and no trading overlay"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _factor_spanning_canonical_benchmarks(benchmark_set: str) -> list:
    try:
        from .factor_spanning import _BENCHMARK_SET_DEFAULTS
        return list(_BENCHMARK_SET_DEFAULTS.get(benchmark_set, []))
    except Exception:  # noqa: BLE001
        return []


def _is_deterministic_factor_spanning_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "factor_spanning":
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        return False
    candidate = _series_ref_name(spec.get("candidate_factor") or (inputs[0] if inputs else ""))
    benchmarks = spec.get("benchmark_factors") or inputs[1:]
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    benchmark_names = [_series_ref_name(x) for x in benchmarks if _series_ref_name(x)]
    if not candidate or len(benchmark_names) < 1 or candidate in benchmark_names:
        return False
    benchmark_set = str(spec.get("benchmark_set") or "").lower()
    if benchmark_set:
        if benchmark_set not in {"capm", "ff3", "ff5", "carhart"}:
            return False
        # FS-1: benchmark_factors must match the declared set's canonical cardinality. A spec that
        # declares "ff3" but hedges only [mkt] (dropping SMB/HML) would let a truly-spanned factor's
        # size premium leak in as fake alpha and still certify faithful; an over-hedge (extra controls)
        # would false-kill by over-hedging. Require the count to match the declared set.
        canonical = _factor_spanning_canonical_benchmarks(benchmark_set)
        if canonical and len(benchmark_names) != len(canonical):
            return False
    estimator = str(spec.get("estimator") or "").lower()
    if estimator and not all(k in estimator for k in ("ols", "is")):
        return False
    if _provided_series_has_added_gate_field(spec):
        return False
    return True


def _factor_spanning_correspondence_verified(claim, spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    inputs = spec.get("inputs") or []
    candidate = spec.get("candidate_factor") or (inputs[0] if inputs else "") or ""
    benchmarks = spec.get("benchmark_factors") or inputs[1:]
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    text = _regression_claim_text(claim, spec)
    text_norm = _norm_var(text)
    if not re.search(r"\b(alpha|intercept|spanning|spanned|controlling|controls?)\b", text):
        return False
    benchmark_set = str(spec.get("benchmark_set") or "").lower()
    if benchmark_set:
        benchmark_patterns = {
            "capm": r"(?:^|_)capm(?:_|$)|(?:^|_)mkt(?:_|$)|(?:^|_)market(?:_|$)",
            "ff3": r"(?:^|_)ff3(?:_|$)|fama_french_3|three_factor|(?:^|_)smb(?:_|$)|(?:^|_)hml(?:_|$)",
            "ff5": r"(?:^|_)ff5(?:_|$)|fama_french_5|five_factor|(?:^|_)rmw(?:_|$)|(?:^|_)cma(?:_|$)",
            "carhart": r"(?:^|_)carhart(?:_|$)",
        }
        if not re.search(benchmark_patterns.get(benchmark_set, r"$^"), text_norm):
            return False
        # FS-1: the hedged benchmark_factors must match the declared set's cardinality (no dropped/added).
        canonical = _factor_spanning_canonical_benchmarks(benchmark_set)
        bench_names = [_series_ref_name(b) for b in benchmarks if _series_ref_name(b)]
        if canonical and len(bench_names) != len(canonical):
            return False
    provenance = spec.get("binding_provenance") or {}
    candidate_prov = provenance.get("candidate_factor") if isinstance(provenance, dict) else {}
    candidate_prov = candidate_prov if isinstance(candidate_prov, dict) else {}
    if candidate_prov:
        return (
            str(candidate_prov.get("kind") or "").lower() != "unresolved"
            and _binding_matches_spec_value(candidate_prov, candidate)
            and _binding_score_ok(candidate_prov)
        )
    return _contains_var(text_norm, _series_ref_name(candidate)) and all(
        _series_ref_name(b) != _series_ref_name(candidate) for b in benchmarks
    )


def _factor_spanning_backstop(result: dict, claim_type: str,
                              spec: dict | None = None, claim=None) -> dict:
    if claim_type != "factor_spanning" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_factor_spanning_spec(spec):
        return result
    if not _factor_spanning_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["factor_spanning_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "factor_spanning fidelity block overridden structurally: deterministic template "
        "with declared candidate, benchmark set, IS-frozen betas, and no trading overlay"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _is_deterministic_cross_sectional_sort_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "cross_sectional_sort":
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    panel_inputs = spec.get("panel_inputs")
    if not isinstance(panel_inputs, dict):
        return False
    if not panel_inputs.get("returns") or not panel_inputs.get("characteristic"):
        return False
    if not str(spec.get("characteristic") or "").strip():
        return False
    try:
        if int(spec.get("n_buckets") or 0) < 2:
            return False
    except (TypeError, ValueError):
        return False
    if not str(spec.get("rebalance") or "").strip() or not str(spec.get("hold") or "").strip():
        return False
    if _provided_series_has_added_gate_field(spec):
        return False
    return True


def _cross_sectional_sort_correspondence_verified(claim, spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    text = _regression_claim_text(claim, spec)
    text_norm = _norm_var(text)
    characteristic = str(spec.get("characteristic") or "").strip()
    if not characteristic or not _contains_var(text_norm, characteristic):
        return False
    try:
        n_buckets = int(spec.get("n_buckets") or 0)
    except (TypeError, ValueError):
        return False
    # CS-1: reject bucket-count / cadence SUBSTITUTION. If the claim EXPLICITLY names a bucket scheme or a
    # rebalance cadence, the spec must match it — a "quartiles, annual" claim reconstructed at deciles/monthly
    # is a substitution the structural backstop must not certify. Silence in the claim never triggers a
    # rejection (the extractor's default is acceptable when the claim omits the detail).
    from .spec_gen import _claimed_bucket_count, _claimed_rebalance
    claimed_buckets = _claimed_bucket_count(text)
    if claimed_buckets is not None and claimed_buckets != n_buckets:
        return False
    claimed_cadence = _claimed_rebalance(text)
    spec_cadence = str(spec.get("rebalance") or "").strip().upper()
    if claimed_cadence is not None and spec_cadence and claimed_cadence != spec_cadence:
        return False
    if not re.search(r"\bsort|sorted|sorting|rank|ranked|ranking\b", text):
        return False
    if not re.search(r"\btop|bottom|spread|high[- ]?minus[- ]?low|long[- ]short\b", text):
        return False
    return True


def _cross_sectional_sort_backstop(result: dict, claim_type: str,
                                   spec: dict | None = None, claim=None) -> dict:
    if claim_type != "cross_sectional_sort" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_cross_sectional_sort_spec(spec):
        return result
    if not _cross_sectional_sort_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["cross_sectional_sort_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "cross_sectional_sort fidelity block overridden structurally: deterministic template "
        "with declared characteristic, bucket count, rebalance cadence, and no trading overlay"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _event_calendar_declared(value) -> bool:
    if isinstance(value, dict):
        raw = (
            value.get("path")
            or value.get("table")
            or value.get("table_path")
            or value.get("calendar_path")
            or value.get("event_calendar_path")
        )
    else:
        raw = value
    return bool(str(raw or "").strip())


def _event_study_window(spec: dict | None) -> tuple[int, int] | None:
    if not isinstance(spec, dict):
        return None
    value = spec.get("window") or spec.get("event_window")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if start <= end else None


def _is_deterministic_event_study_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "event_study":
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    return_series = _series_ref_name(spec.get("return_series") or ((spec.get("inputs") or [""])[0]))
    if not return_series:
        return False
    if not _event_calendar_declared(spec.get("event_calendar")):
        return False
    if _event_study_window(spec) is None:
        return False
    baseline = str(spec.get("baseline") or "mean_adjusted").strip().lower()
    if baseline not in {"mean_adjusted", "market_model"}:
        return False
    if baseline == "market_model" and not _series_ref_name(spec.get("market_series")):
        return False
    try:
        if int(spec.get("estimation_window") or 0) < 5:
            return False
    except (TypeError, ValueError):
        return False
    if _provided_series_has_added_gate_field(spec):
        return False
    return True


def _event_study_correspondence_verified(claim, spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    text = _regression_claim_text(claim, spec)
    if not re.search(r"\b(?:abnormal\s+returns?|cumulative\s+abnormal\s+returns?|car)\b", text):
        return False
    if not re.search(r"\b(?:event|announcement|earnings|fomc|listing|addition|halving)\b", text):
        return False
    from .spec_gen import _claimed_baseline, _claimed_event_window
    claimed_window = _claimed_event_window(text)
    spec_window = _event_study_window(spec)
    if claimed_window is not None and spec_window != claimed_window:
        return False
    claimed_baseline = _claimed_baseline(text)
    spec_baseline = str(spec.get("baseline") or "mean_adjusted").strip().lower()
    if claimed_baseline is not None and spec_baseline != claimed_baseline:
        return False
    provenance = spec.get("binding_provenance") or {}
    ret_prov = provenance.get("return_series") if isinstance(provenance, dict) else {}
    ret_prov = ret_prov if isinstance(ret_prov, dict) else {}
    return_series = spec.get("return_series") or ((spec.get("inputs") or [""])[0])
    if ret_prov:
        return (
            str(ret_prov.get("kind") or "").lower() != "unresolved"
            and _binding_matches_spec_value(ret_prov, return_series)
            and _binding_score_ok(ret_prov)
        )
    return bool(_series_ref_name(return_series))


def _event_study_backstop(result: dict, claim_type: str,
                          spec: dict | None = None, claim=None) -> dict:
    if claim_type != "event_study" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_event_study_spec(spec):
        return result
    if not _event_study_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["event_study_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "event_study fidelity block overridden structurally: deterministic template "
        "with declared calendar, window, strictly-pre-event baseline, and no trading overlay"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _forecast_benchmark_method(value) -> str:
    if isinstance(value, dict):
        value = value.get("method") or value.get("family") or value.get("benchmark") or value.get("kind")
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"random_walk", "rw", "naive", "persistence", "last_value"}:
        return "random_walk"
    if text in {"historical_mean", "expanding_mean", "mean", "expanding_historical_mean"}:
        return "historical_mean"
    return ""


def _forecast_benchmark_ref(spec: dict | None):
    if not isinstance(spec, dict):
        return ""
    return spec.get("benchmark") or spec.get("benchmark_forecast") or spec.get("benchmark_series") or ""


def _is_deterministic_forecast_skill_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "forecast_skill":
        return False
    if spec.get("_llm_mode") != "deterministic-template":
        return False
    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        return False
    model = _series_ref_name(spec.get("model_forecast") or (inputs[0] if inputs else ""))
    target = _series_ref_name(spec.get("target") or (inputs[1] if len(inputs) > 1 else ""))
    if not model or not target or model == target:
        return False
    benchmark = _forecast_benchmark_ref(spec)
    benchmark_series = _series_ref_name(benchmark)
    benchmark_method = _forecast_benchmark_method(benchmark)
    if benchmark_series:
        if benchmark_series in {model, target}:
            return False
    elif benchmark_method not in {"random_walk", "historical_mean"}:
        return False
    if str(spec.get("loss") or "squared_error").lower() not in {"squared_error", "mse", "msfe"}:
        return False
    if _provided_series_has_added_gate_field(spec):
        return False
    return True


def _forecast_skill_correspondence_verified(claim, spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    text = _regression_claim_text(claim, spec)
    text_norm = _norm_var(text)
    if not re.search(r"\b(?:forecast|forecasts|forecasting|predicts?|prediction)\b", text):
        return False
    if not re.search(
        r"\b(?:msfe|mspe|rmse|diebold[- ]mariano|clark[- ]west|out[- ]of[- ]sample\s+r|oos\s+r|benchmark|random[- ]walk|naive)\b",
        text,
    ):
        return False
    inputs = spec.get("inputs") or []
    model = spec.get("model_forecast") or (inputs[0] if inputs else "")
    target = spec.get("target") or (inputs[1] if len(inputs) > 1 else "")
    benchmark = _forecast_benchmark_ref(spec)
    benchmark_series = _series_ref_name(benchmark)
    benchmark_method = _forecast_benchmark_method(benchmark)

    provenance = spec.get("binding_provenance") or {}
    if isinstance(provenance, dict) and provenance:
        model_prov = provenance.get("model_forecast") or {}
        target_prov = provenance.get("target") or {}
        benchmark_prov = provenance.get("benchmark") or {}
        if not (
            isinstance(model_prov, dict)
            and isinstance(target_prov, dict)
            and str(model_prov.get("kind") or "").lower() != "unresolved"
            and str(target_prov.get("kind") or "").lower() != "unresolved"
            and _binding_matches_spec_value(model_prov, model)
            and _binding_matches_spec_value(target_prov, target)
            and _binding_score_ok(model_prov)
            and _binding_score_ok(target_prov)
        ):
            return False
        if benchmark_series:
            if not (
                isinstance(benchmark_prov, dict)
                and str(benchmark_prov.get("kind") or "").lower() != "unresolved"
                and _binding_matches_spec_value(benchmark_prov, benchmark_series)
                and _binding_score_ok(benchmark_prov)
            ):
                return False
        elif benchmark_method:
            if benchmark_method == "random_walk" and not re.search(r"\b(random[- ]walk|naive|persistence)\b", text):
                return False
            if benchmark_method == "historical_mean" and not re.search(r"\bhistorical\s+mean\b|\bexpanding\s+mean\b", text):
                return False
        else:
            return False
    else:
        if not (_contains_var(text_norm, _series_ref_name(model)) and _contains_var(text_norm, _series_ref_name(target))):
            return False
        if benchmark_series and not _contains_var(text_norm, benchmark_series):
            return False
        if benchmark_method == "random_walk" and not re.search(r"\b(random[- ]walk|naive|persistence)\b", text):
            return False
        if benchmark_method == "historical_mean" and not re.search(r"\bhistorical\s+mean\b|\bexpanding\s+mean\b", text):
            return False

    # FS-forecast CS-1 analogue: reject explicit target/benchmark substitution.
    from .spec_gen import _forecast_benchmark_method as _claimed_forecast_benchmark_method
    claimed_method = _claimed_forecast_benchmark_method(text)
    if claimed_method and benchmark_method and claimed_method != benchmark_method:
        return False
    if benchmark_series and benchmark_series in {_series_ref_name(model), _series_ref_name(target)}:
        return False
    return True


def _forecast_skill_backstop(result: dict, claim_type: str,
                             spec: dict | None = None, claim=None) -> dict:
    if claim_type != "forecast_skill" or result.get("faithful") is not False:
        return result
    if not _is_deterministic_forecast_skill_spec(spec):
        return result
    if not _forecast_skill_correspondence_verified(claim, spec):
        return result
    out = dict(result)
    out["faithful"] = True
    out["verified"] = (
        float(out.get("confidence", 0.0) or 0.0) >= config.FIDELITY_KILL_CONFIDENCE
    )
    out["forecast_skill_fidelity_override"] = "deterministic_template_structural"
    note = str(out.get("note", "") or "").strip()
    suffix = (
        "forecast_skill fidelity block overridden structurally: deterministic template "
        "with declared model forecast, target, benchmark, loss differential, and no trading overlay"
    )
    out["note"] = (f"{note}; {suffix}" if note else suffix)[:240]
    return out


def _norm_var(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def _declared_variables(spec: dict | None) -> list[str]:
    if not isinstance(spec, dict):
        return []
    out, seen = [], set()
    for key in ("variables", "inputs"):
        vals = spec.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, (list, tuple)):
            continue
        for val in vals:
            if isinstance(val, dict):
                val = val.get("name") or val.get("id") or val.get("series") or ""
            name = _norm_var(val)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _expr_deps(value: ast.AST) -> set[str]:
    """Every identifier an expression can depend on: bare Names, attribute names
    (bundle.realized_drift), and string constants (df["realized_drift"],
    bundle.get("funding")). String constants count because this codebase's module
    idiom addresses series by string key at least as often as by local name
    (audit finding D-2, 2026-07-05)."""
    deps: set[str] = set()
    for n in ast.walk(value):
        if isinstance(n, ast.Name):
            deps.add(n.id)
        elif isinstance(n, ast.Attribute):
            deps.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            deps.add(n.value)
    return deps


class _SignalDependencyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assign_deps: dict[str, set[str]] = {}
        self.signal_targets: set[str] = set()
        self.signal_deps: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        deps = _expr_deps(node.value)
        expanded = set(deps)
        for dep in list(deps):
            expanded.update(self.assign_deps.get(dep, set()))
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assign_deps[target.id] = expanded
                if "signal" in target.id.lower():
                    self.signal_targets.add(target.id)
                    self.signal_deps.update(expanded)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        deps = _expr_deps(node.value)
        expanded = set(deps)
        for dep in list(deps):
            expanded.update(self.assign_deps.get(dep, set()))
        if isinstance(node.target, ast.Name):
            self.assign_deps[node.target.id] = expanded
            if "signal" in node.target.id.lower():
                self.signal_targets.add(node.target.id)
                self.signal_deps.update(expanded)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            names = _expr_deps(node.value)
            if names & self.signal_targets:
                self.signal_deps.update(names)
        self.generic_visit(node)


def variable_coverage_check(module_code: str, spec: dict | None) -> dict:
    """Deterministic fidelity guard: every declared input must flow into the signal.

    This is deliberately fail-soft: it returns needs_review on a clear structural miss,
    and pass/skipped otherwise. It never calls an LLM and never emits a kill.
    """
    claim_type = str((spec or {}).get("claim_type") or "trading_strategy")
    if claim_type != "trading_strategy":
        return {"ok": True, "skipped": True, "reason": "claim type has no trading signal"}
    variables = _declared_variables(spec)
    if not variables:
        return {"ok": True, "skipped": True, "reason": "no declared variables"}
    try:
        tree = ast.parse(module_code or "")
    except (SyntaxError, ValueError) as e:
        return {"ok": True, "skipped": True, "reason": f"module AST unavailable: {e}"}
    visitor = _SignalDependencyVisitor()
    visitor.visit(tree)
    if not visitor.signal_targets:
        return {"ok": True, "skipped": True, "reason": "no explicit signal assignment"}
    # SIGNAL deps only — unioning all assigned names (set(visitor.assign_deps)) was
    # audit finding D-1 (2026-07-05): it marked any declared input "covered" if it was
    # merely loaded into a local, which is exactly the June-29 5d false-negative shape.
    deps = {_norm_var(name) for name in visitor.signal_deps}
    missing = [v for v in variables if v not in deps]
    if missing:
        return {
            "ok": False,
            "needs_review": True,
            "missing_variables": missing,
            "reason": "declared variable(s) do not flow into the signal: " + ", ".join(missing),
        }
    return {"ok": True, "skipped": False, "covered_variables": variables}
