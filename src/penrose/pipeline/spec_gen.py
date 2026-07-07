"""LLM-driven ModuleSpec generation.

When P6 routing finds no registered module for a claim's strategy class (cold-
start registry, or a genuinely novel strategy class), this generates a
ModuleSpec YAML the operator can review and implement (or hand to an agent
swarm). One-shot insertion: the spec lands in modules/_specs/, the
pipeline continues, and the operator reviews it asynchronously.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import config, llm
from ..brain import Claim
from ..data.contract import load_catalog_loader
from . import fidelity_memory
from .p1_ingest import IngestedSource


SPEC_SYSTEM = (
    "You are a research-engine module spec generator. Given a falsifiable claim "
    "extracted from a paper, you produce a ModuleSpec — a precise, implementable "
    "description of how to test the exact claim. If claim_type is "
    "descriptive_statistical, specify the statistic to compute and its uncertainty; "
    "do NOT translate it into a trading strategy. If claim_type is trading_strategy, "
    "act as a translator, not a co-author: preserve the claim's signal exactly, do "
    "not add unclaimed gates or kill conditions, and choose one explicit conventional "
    "default for any unspecified required parameter. Describe the signal, positions, "
    "PnL, DSR / 3-fold / locked-holdout / fee + slippage + capacity discipline. If "
    "claim_type is structural_proposition, be "
    "honest about what cannot be operationalized. Be concrete: specify the inputs "
    "required (use the data contract vocabulary), the kill criterion, and the "
    "unknowns. If a claim requires data that is not in the available data vocabulary, "
    "declare the true required input using a clear catalog-style series name and list "
    "it in unknowns as unavailable; never fabricate, synthesize, derive, or proxy a "
    "missing input from other series to make the spec runnable. The operator (or an "
    "agent swarm) will implement your spec; vagueness "
    "costs them time. Respond strictly in JSON."
)

TRADING_SPEC_USER_TMPL = """Generate a ModuleSpec for this claim:

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Translator-not-co-author constraints for trading_strategy claims:
- Do NOT add any significance test, data-quality filter, minimum-observation gate, or
  rejection/kill condition that the claim itself does not state. Required cost/fee
  modeling is fine; invented pass/fail gates are not.
- Preserve the claim's exact signal definition. If the claim states an absolute
  directional rule, do not convert it to a cross-sectional, relative-value, ranked,
  demeaned, normalized, or otherwise transformed signal unless the claim says to.
  Likewise, do not convert a relative/ranked claim into an absolute directional rule.
- For any required parameter the claim leaves unspecified (lookback, exit rule, sign
  convention, venue), commit to one conventional default and state it explicitly. Never
  emit a menu of alternatives in one field, and never leave a required field unresolved.
- Data fidelity is part of faithful translation. If the exact input the claim requires is
  not present in AVAILABLE DATA SERIES, do NOT synthesize, fabricate, derive, blend, or
  proxy it from other series. Declare the true required input in `inputs` with a clear
  catalog-style name (for example `bnb_spot_daily`) and include an `unknowns` entry saying
  that series is unavailable. A faithful-but-data-blocked spec is correct.

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "claim_translation": str (how the academic claim becomes a tradeable, falsifiable test),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "signal_logic": str (pseudocode or precise description of how the module computes positions),
  "kill_criterion": str (concrete: which metric thresholds kill the strategy),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""

DESCRIPTIVE_SPEC_USER_TMPL = """Generate a ModuleSpec for this descriptive statistical claim.
The implementation must compute and test the stated statistic directly. Do not invent
entry rules, positions, PnL, Sharpe, or a conditional trading strategy unless the claim
itself explicitly says to trade.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "claim_type": "descriptive_statistical",
  "claim_translation": str (the exact statistic/hypothesis to test, e.g. unconditional mean + CI),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "statistic_logic": str (precise description of the statistic, sample, uncertainty, and test),
  "signal_logic": str ("descriptive_statistical: compute the statistic directly; no positions/PnL"),
  "kill_criterion": str (concrete: what statistical result fails to replicate the stated claim),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""

STRUCTURAL_SPEC_USER_TMPL = """Generate a ModuleSpec for this structural proposition.
If it cannot be operationalized against available data without inventing a trading
strategy, say so plainly in claim_translation and unknowns.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "claim_type": "structural_proposition",
  "claim_translation": str (what can and cannot be falsified),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "signal_logic": str ("structural_proposition: cannot create positions unless explicitly claimed"),
  "kill_criterion": str (concrete falsification criterion, or "cannot_operationalize"),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""


def classify_claim_type(claim: Claim, source: IngestedSource | None = None) -> str:
    return fidelity_memory.classify_claim_type(claim, source)


def _catalog_vocab() -> str:
    """Newline list of catalog series. Fail-open to an omitted prompt block."""
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        if not hasattr(catalog, "available"):
            return ""
        if hasattr(catalog, "describe_brief"):
            lines = [catalog.describe_brief(n) for n in catalog.available()]
        else:
            lines = list(catalog.available())
        return "\n".join(f"  - {line}" for line in lines)
    except Exception:  # noqa: BLE001
        return ""


def generate_spec(claim: Claim, source: IngestedSource,
                  *, use_llm: bool = True,
                  prior_divergences: list[str] | None = None) -> dict:
    """Generate a ModuleSpec for a claim. Returns the spec dict + writes YAML to disk.

    The YAML lands at modules/_specs/<claim_id>.yaml. The operator reviews it
    asynchronously and (if accepted) moves it under modules/<module_id>/ and
    implements impl.py — at which point the registry picks it up on the next run.
    """
    specs_dir = config.MODULES / "_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    out_path = specs_dir / f"{claim.claim_id}.yaml"

    claim_type = classify_claim_type(claim, source)
    if claim_type == "provided_series_statistic":
        # 6g: a mechanical translation, not a creative one -- never touches the LLM, so it
        # can neither invent gates (over-specification, EXP-1) nor fall back to an empty
        # stub (under-specification, EXP-1b). See _provided_series_stat_spec.
        spec = _provided_series_stat_spec(claim, source)
    elif not use_llm:
        spec = _stub_spec(claim, source)
    else:
        try:
            spec = _llm_spec(claim, source, claim_type=claim_type,
                             prior_divergences=prior_divergences)
        except Exception as e:  # noqa: BLE001
            # never let spec generation failure block the pipeline — fall back to stub
            spec = _stub_spec(claim, source)
            spec["_llm_error"] = str(e)
    spec.setdefault("claim_type", claim_type)

    spec["_generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec["_claim_id"] = claim.claim_id
    spec["_source_id"] = source.source_id
    spec["_path"] = str(out_path)

    _write_yaml(out_path, spec)
    return spec


def _immediate_correction_guidance(prior_divergences: list[str] | None) -> str:
    items = [str(d).strip() for d in (prior_divergences or []) if str(d).strip()]
    if not items:
        return ""
    bullets = "\n".join(f"- {item[:500]}" for item in items[:8])
    return (
        "YOUR PREVIOUS SPEC FOR THIS CLAIM WAS REJECTED AS UNFAITHFUL for these "
        "specific reasons:\n"
        f"{bullets}\n"
        "Produce a corrected spec that FIXES exactly these, WITHOUT introducing any new "
        "divergence. Do not over-correct into a different claim.\n"
    )


def _base_prompt(claim: Claim, source: IngestedSource, claim_type: str,
                 prior_divergences: list[str] | None = None) -> str:
    tmpl = {
        "descriptive_statistical": DESCRIPTIVE_SPEC_USER_TMPL,
        "structural_proposition": STRUCTURAL_SPEC_USER_TMPL,
    }.get(claim_type, TRADING_SPEC_USER_TMPL)
    user = tmpl.format(
        statement=claim.statement[:600],
        mechanism=(claim.mechanism or "")[:300],
        scope=(claim.scope or "")[:200],
        horizon=(claim.horizon or "")[:80],
        strategy_class=claim.applicable_strategy_class or "unspecified",
        claim_type=claim_type,
        source_span=claim.source_span[:400],
        claimed_metric=(claim.claimed_metric_quote or "")[:200],
    )
    guidance = fidelity_memory.rejection_guidance(
        claim.applicable_strategy_class or "unspecified", claim_type)
    if guidance:
        user = guidance + "\n" + user
    correction = _immediate_correction_guidance(prior_divergences)
    if correction:
        user = correction + "\n" + user
    vocab = _catalog_vocab()
    if vocab:
        user = (
            "AVAILABLE DATA SERIES (use these EXACT names in `inputs` when they fit the claim):\n"
            + vocab + "\n\n" + user
            + "\n\nUse inputs drawn from the AVAILABLE DATA SERIES list when one exactly satisfies "
            "the claim. If the claim requires an input that is absent from this list, do NOT "
            "fabricate, synthesize, derive, or proxy it from listed series. Instead, declare the "
            "true required input in `inputs` using a clear catalog-style name and list it in "
            "`unknowns` as unavailable."
        )
    return user


def _llm_spec(claim: Claim, source: IngestedSource, *, claim_type: str | None = None,
              prior_divergences: list[str] | None = None) -> dict:
    claim_type = claim_type or classify_claim_type(claim)
    user = _base_prompt(claim, source, claim_type, prior_divergences=prior_divergences)
    parsed, _ = llm.call_json(
        "module_spec_generator",
        [{"role": "system", "content": SPEC_SYSTEM},
         {"role": "user",   "content": user}],
        temperature=0.1,
    )
    # ensure required keys exist
    required = ["module_id", "strategy_class", "claim_translation", "inputs",
                "signal_logic", "kill_criterion", "unknowns"]
    for k in required:
        if k not in parsed:
            parsed[k] = "" if k != "inputs" and k != "unknowns" else []
    # Audit D-4 (2026-07-05): claim_type is a GATE-SELECTION switch (fidelity override +
    # variable-coverage skip). It must come ONLY from the deterministic classifier —
    # never from the LLM's spec JSON, which an over-eager model could set to
    # provided_series_statistic and loosen two gates at once. Log any disagreement.
    llm_claim_type = parsed.get("claim_type")
    if llm_claim_type and llm_claim_type != claim_type:
        print(f"[penrose] spec LLM proposed claim_type={llm_claim_type!r}; "
              f"keeping deterministic {claim_type!r}")
    parsed["claim_type"] = claim_type
    parsed["version"] = 0
    parsed["status"] = "spec-only"
    parsed["source"] = source.source_id
    return parsed


_SERIES_DECISION_EXCLUSION_CUES = (
    "not part of the decision",
    "reported, not",
    "for reference",
    "benchmark",
    "control series",
    "control:",
    "comparison",
    "compared to",
    "baseline",
    "reference series",
    "context (",
    "descriptive only",
    "not a gate",
    "reported only",
    "informational",
    "for context",
    "as context",
    "contextual",
    "as reference",
    "we also report",
    "reported alongside",
    "illustrative",
)


def _series_text_units(text: str) -> list[str]:
    """Text units for the decision-exclusion scan. Split on BLANK-line paragraph
    boundaries and JOIN soft-wrapped lines within each paragraph first, so a cue like
    "Context (reported, not part of the decision):" stays attached to the series names
    that wrap onto the following physical line(s). Emit each joined paragraph AND its
    sentences, so a series named in a context paragraph/sentence is correctly excluded."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        joined = " ".join(para.split())     # collapse soft wraps + whitespace
        if not joined:
            continue
        units.append(joined)
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip())
    return units


def _declared_series_from_text(claim: Claim, source: IngestedSource | None = None) -> list[str]:
    """Deterministically pull any catalog series names that literally appear in the
    claim/source text, excluding names that appear only in context/reference units.
    No invention, no LLM: a name is included only if it is BOTH a real catalog series
    AND already named by the claim/source -- the same discipline as the #5a
    catalog-vocabulary guard, applied without a model in the loop. Fail-open to [] on
    any catalog error."""
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        names = list(catalog.available()) if hasattr(catalog, "available") else []
    except Exception:  # noqa: BLE001
        return []
    text = "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])
    matched = set()
    for unit in _series_text_units(text):
        lowered = unit.lower()
        if any(cue in lowered for cue in _SERIES_DECISION_EXCLUSION_CUES):
            continue
        matched.update(
            n for n in names
            if n and re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", unit)
        )
    return sorted(matched)


def _provided_series_stat_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic (non-LLM) ModuleSpec for a `provided_series_statistic` claim (6g):
    "test the statistic of a provided/pre-computed series." This claim class needs no
    creative translation -- pool the claim's own declared series and apply exactly the
    claim's own stated decision rule. Building it as a template rather than an LLM
    generation is what makes both failure modes structurally impossible: there is no
    generation step to invent a p<=0.05 gate, a data-quality kill, or a menu of deflation
    methods (over-specification, EXP-1), and no generation step to fall back to an empty
    stub (under-specification, EXP-1b).
    """
    inputs = _declared_series_from_text(claim, source)
    decision_rule = (claim.claimed_metric_quote or claim.statement or "").strip()
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": claim.applicable_strategy_class or "unspecified",
        "claim_type": "provided_series_statistic",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic one-sample test on the claim's own declared/provided series: pool "
            "ALL declared series into one sample (no re-grouping, no sub-bucketing), compute "
            "the sample mean and its one-sample test/CI, and apply EXACTLY the claim's stated "
            "decision rule. No positions, no PnL simulation, and no extra pass/fail filters "
            "or deflation method beyond what the claim itself declares."
        ),
        "inputs": inputs,
        "statistic_logic": (
            "one_sample_mean_test: pool the declared series into one sample; compute the mean, "
            "its standard error, and a one-sample test against the claim's own stated decision "
            "rule. If the claim declares a single cohort/deflation family, use exactly that one "
            "method; never offer a menu of methods, and never add extra pass/fail filters the "
            "claim did not state."
        ),
        "signal_logic": "provided_series_statistic: compute the statistic directly; no positions/PnL.",
        "kill_criterion": (
            f"the claim's own stated decision rule is not met: {decision_rule}"[:400]
            if decision_rule else
            "the claim's own stated decision rule is not met (see claim_statement)"
        ),
        "expected_data_needs": "as declared by the claim; no additional window/frequency invented",
        "unknowns": (
            [] if inputs else
            ["no catalog series literally matched the claim text; operator must confirm the "
             "series names before implementing"]
        ),
        "implementation_notes": (
            "Deterministically generated (no LLM) for a provided-series-statistics claim: this "
            "claim type is a mechanical translation, not a creative one -- preserve the stated "
            "decision rule exactly."
        ),
        "_llm_mode": "deterministic-template",
    }


def _stub_spec(claim: Claim, source: IngestedSource) -> dict:
    claim_type = classify_claim_type(claim)
    if claim_type == "descriptive_statistical":
        translation = (
            "OFFLINE STUB — operator must compute and test the descriptive statistic "
            "stated by the claim directly. Re-run with PENROSE_LLM_API_KEY set to get "
            "an LLM-generated translation."
        )
        signal_logic = "descriptive_statistical: compute the stated statistic directly; no positions/PnL."
        kill_criterion = "The stated statistic does not replicate with appropriate uncertainty bounds."
    elif claim_type == "structural_proposition":
        translation = "OFFLINE STUB — operator must determine whether this structural proposition is operationalizable."
        signal_logic = "structural_proposition: cannot create positions unless explicitly claimed."
        kill_criterion = "cannot_operationalize unless a direct falsification test is specified"
    else:
        translation = (
            "OFFLINE STUB — operator must translate this claim to a tradeable test. "
            "Re-run with PENROSE_LLM_API_KEY set to get an LLM-generated translation."
        )
        signal_logic = "OFFLINE STUB — not generated."
        kill_criterion = "OOS DSR/PSR < 0.90 OR 3-fold sign-unstable OR edge_t < 1.0"
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": claim.applicable_strategy_class or "unspecified",
        "claim_type": claim_type,
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": translation,
        "inputs": [],
        "signal_logic": signal_logic,
        "kill_criterion": kill_criterion,
        "unknowns": ["auto-generated stub spec; operator review required"],
        "_llm_mode": "stub",
    }


def _write_yaml(path: Path, spec: dict) -> None:
    # strip internal keys starting with _ for the YAML body; keep _path separate
    body = {k: v for k, v in spec.items() if not k.startswith("_")}
    header = (
        f"# auto-generated ModuleSpec (F10) — review and implement\n"
        f"# claim: {spec.get('_claim_id', '?')} · source: {spec.get('_source_id', '?')}\n"
        f"# generated: {spec.get('_generated_at', '?')}\n\n"
    )
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False))
