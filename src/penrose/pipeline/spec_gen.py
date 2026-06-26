"""LLM-driven ModuleSpec generation.

When P6 routing finds no registered module for a claim's strategy class (cold-
start registry, or a genuinely novel strategy class), this generates a
ModuleSpec YAML the operator can review and implement (or hand to an agent
swarm). One-shot insertion: the spec lands in modules/_specs/, the
pipeline continues, and the operator reviews it asynchronously.
"""
from __future__ import annotations

import json
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
    "describe the signal, positions, PnL, DSR / 3-fold / locked-holdout / fee + "
    "slippage + capacity discipline. If claim_type is structural_proposition, be "
    "honest about what cannot be operationalized. Be concrete: specify the inputs "
    "required (use the data contract vocabulary), the kill criterion, and the "
    "unknowns. The operator (or an agent swarm) will implement your spec; vagueness "
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


def classify_claim_type(claim: Claim) -> str:
    return fidelity_memory.classify_claim_type(claim)


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
                  *, use_llm: bool = True) -> dict:
    """Generate a ModuleSpec for a claim. Returns the spec dict + writes YAML to disk.

    The YAML lands at modules/_specs/<claim_id>.yaml. The operator reviews it
    asynchronously and (if accepted) moves it under modules/<module_id>/ and
    implements impl.py — at which point the registry picks it up on the next run.
    """
    specs_dir = config.MODULES / "_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    out_path = specs_dir / f"{claim.claim_id}.yaml"

    claim_type = classify_claim_type(claim)
    if not use_llm:
        spec = _stub_spec(claim, source)
    else:
        try:
            spec = _llm_spec(claim, source, claim_type=claim_type)
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


def _base_prompt(claim: Claim, source: IngestedSource, claim_type: str) -> str:
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
    vocab = _catalog_vocab()
    if vocab:
        user = (
            "AVAILABLE DATA SERIES (use these EXACT names in `inputs` when they fit the claim; "
            "only invent a new logical name if NOTHING here matches):\n"
            + vocab + "\n\n" + user
            + "\n\nPrefer inputs drawn from the AVAILABLE DATA SERIES list when one is provided; "
            "exact-match the listed names."
        )
    return user


def _llm_spec(claim: Claim, source: IngestedSource, *, claim_type: str | None = None) -> dict:
    claim_type = claim_type or classify_claim_type(claim)
    user = _base_prompt(claim, source, claim_type)
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
    parsed["claim_type"] = parsed.get("claim_type") or claim_type
    parsed["version"] = 0
    parsed["status"] = "spec-only"
    parsed["source"] = source.source_id
    return parsed


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
