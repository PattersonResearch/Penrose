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
        return _event_market_strategy_backstop(result, claim_type, spec)
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


def _is_deterministic_event_market_spec(spec: dict | None) -> bool:
    if not isinstance(spec, dict):
        return False
    if spec.get("claim_type") != "event_market_strategy":
        return False
    cfg = spec.get("event_market") if isinstance(spec.get("event_market"), dict) else spec
    has_table = any(cfg.get(k) for k in ("path", "table", "table_path",
                                         "event_market_path", "event_market_table"))
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
    return has_table and family.strip() == "normal_bracket"


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
        "normal_bracket pricing over a declared bracket table; arbitrary pricing-formula "
        "fidelity is deferred"
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
