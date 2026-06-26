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

from .. import config, llm

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
    try:
        parsed, response = llm.call_json(
            role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
        if not isinstance(parsed, dict) or "faithful" not in parsed:
            return {"faithful": False, "verified": False, "confidence": 0.0, "divergences": [],
                    "note": "fidelity check inconclusive", "independent_verifier": False}
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        faithful = bool(parsed.get("faithful"))
        return {
            "faithful": faithful,
            "verified": faithful and confidence >= config.FIDELITY_KILL_CONFIDENCE,
            "confidence": confidence,
            "divergences": (parsed.get("divergences") or [])[:5],
            "note": str(parsed.get("note", ""))[:240],
            "independent_verifier": bool(getattr(response, "independent_verifier", False)),
        }
    except Exception as e:  # noqa: BLE001 — inconclusive is contained, never promoted to faithful
        return {"faithful": False, "verified": False, "confidence": 0.0, "divergences": [],
                "note": f"fidelity check errored: {e}", "independent_verifier": False}
