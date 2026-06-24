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
from .p1_ingest import IngestedSource


SPEC_SYSTEM = (
    "You are a research-engine module spec generator. Given a falsifiable claim "
    "extracted from a paper, you produce a ModuleSpec — a precise, implementable "
    "description of how to test the claim as a tradeable strategy under DSR / "
    "3-fold / locked-holdout / fee + slippage + capacity discipline. Be concrete: "
    "specify the signal logic, the inputs required (use the data contract vocabulary), "
    "the kill criterion, and the unknowns. The operator (or an agent swarm) will "
    "implement your spec; vagueness costs them time. Respond strictly in JSON."
)

SPEC_USER_TMPL = """Generate a ModuleSpec for this claim:

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}

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

    if not use_llm:
        spec = _stub_spec(claim, source)
    else:
        try:
            spec = _llm_spec(claim, source)
        except Exception as e:  # noqa: BLE001
            # never let spec generation failure block the pipeline — fall back to stub
            spec = _stub_spec(claim, source)
            spec["_llm_error"] = str(e)

    spec["_generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec["_claim_id"] = claim.claim_id
    spec["_source_id"] = source.source_id
    spec["_path"] = str(out_path)

    _write_yaml(out_path, spec)
    return spec


def _llm_spec(claim: Claim, source: IngestedSource) -> dict:
    user = SPEC_USER_TMPL.format(
        statement=claim.statement[:600],
        mechanism=(claim.mechanism or "")[:300],
        scope=(claim.scope or "")[:200],
        horizon=(claim.horizon or "")[:80],
        strategy_class=claim.applicable_strategy_class or "unspecified",
        source_span=claim.source_span[:400],
        claimed_metric=(claim.claimed_metric_quote or "")[:200],
    )
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
    parsed["version"] = 0
    parsed["status"] = "spec-only"
    parsed["source"] = source.source_id
    return parsed


def _stub_spec(claim: Claim, source: IngestedSource) -> dict:
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": claim.applicable_strategy_class or "unspecified",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "OFFLINE STUB — operator must translate this claim to a tradeable test. "
            "Re-run with PENROSE_LLM_API_KEY set to get an LLM-generated translation."
        ),
        "inputs": [],
        "signal_logic": "OFFLINE STUB — not generated.",
        "kill_criterion": "OOS DSR/PSR < 0.90 OR 3-fold sign-unstable OR edge_t < 1.0",
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
