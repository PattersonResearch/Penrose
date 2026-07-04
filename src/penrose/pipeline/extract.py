"""P2 claim extraction + P3 falsifiability — LLM-driven.

Both roles use json_mode and require verbatim source_spans (anti-
hallucination). Falls back to operator-supplied claims.py when no API key is
present (offline / cold-box dev). The fallback path is explicit and flagged in
the decision record so a hand-authored run is never mistaken for an LLM run.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from typing import Optional

from .. import llm
from ..brain import Claim


# --- P2: claim extraction --------------------------------------------------- #

P2_SYSTEM = (
    "You are a research-engine claim extractor. You read an academic paper and "
    "extract every falsifiable forecasting claim the paper makes — not its summary, "
    "not its motivation, only the testable predictions. For each claim you must "
    "quote the exact sentence(s) it came from (the source_span). If you cannot "
    "quote the sentence, do not emit the claim. Never invent metrics or numbers; "
    "if the paper states a metric, copy it verbatim into claimed_metric_quote. "
    "Respond strictly in JSON."
)

P2_USER_TMPL = """Paper: {title}
Source ID: {source_id}

Extract falsifiable forecasting claims. Each claim must be a directional,
testable prediction with a measurable horizon — not narrative, not motivation,
not literature review. Reject hand-waving.

Output JSON: {{"claims": [{{"statement": str, "mechanism": str, "scope": str,
"horizon": str, "source_span": str (verbatim quote), "claimed_metric_quote": str
or null, "applicable_strategy_class": str, "expected_edge": float or null,
"sample_period": {{"start": str, "end": str}} or null}}]}}
expected_edge is the paper's own claimed net edge per trade as a fraction,
only if numerically stated; else null.
sample_period is the paper's own evaluation window (data start/end),
verbatim-derived; null if not stated.
{vocab}
Paper text (truncated):
---
{body}
---
"""

# Defensive: cap source-text size to avoid blowing context. Most of the value is
# in abstract + intro + findings anyway. The deep_reader role handles full text.
MAX_CHARS_PER_PAPER = 24_000


def _norm_ws(s: str) -> str:
    """Collapse all runs of whitespace to single spaces (case-sensitive)."""
    return " ".join((s or "").split())


def span_in_text(span: str, text: str) -> bool:
    """Verbatim-span guarantee: does `span` actually occur in `text`?

    Robust to minor whitespace differences: both sides are whitespace-normalized
    before the substring check. An empty span never matches. This is what stops a
    hallucinated / injected / scanned-PDF claim from passing the gate with a
    fabricated source_span that isn't really in the paper.
    """
    span_n = _norm_ws(span)
    if not span_n:
        return False
    return span_n in _norm_ws(text)


def _expected_edge_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        edge = float(value)
    except (TypeError, ValueError):
        return None
    return edge if math.isfinite(edge) and edge >= 0 else None


def extract_claims(source, known_classes: dict | None = None) -> tuple[list[Claim], dict]:
    """Run the P2 LLM role over an IngestedSource; return claims + provenance.

    known_classes ({class_name: description}) are the strategy classes that already have
    a module. The LLM is told to REUSE an existing class name verbatim when a claim
    genuinely fits it (so P6 routes it to that module and it backtests), else propose a
    new one. This is what lets the LLM path produce real backtests, not just specs.
    """
    body = source.text[:MAX_CHARS_PER_PAPER]
    vocab = ""
    if known_classes:
        lines = "\n".join(f'- "{k}": {v}' for k, v in known_classes.items())
        vocab = ("\nEXISTING strategy classes (a module already implements each). If a claim "
                 "CLEARLY fits one, set applicable_strategy_class to its EXACT name (verbatim) "
                 "so it routes to that module and gets backtested. Otherwise propose a new "
                 "concise class.\n" + lines + "\n")
    user = P2_USER_TMPL.format(title=source.title, source_id=source.source_id,
                               body=body, vocab=vocab)
    parsed, resp = llm.call_json(
        "claim_extractor",
        [{"role": "system", "content": P2_SYSTEM},
         {"role": "user",   "content": user}],
        temperature=0.1,
        timeout=240,   # P2 over a full paper with a thinking model legitimately runs long
    )

    raw = parsed.get("claims", []) if isinstance(parsed, dict) else []
    claims: list[Claim] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            continue
        span = (c.get("source_span") or "").strip()
        statement = (c.get("statement") or "").strip()
        if not span or not statement:
            # anti-hallucination gate: drop anything without a verbatim span
            continue
        if not span_in_text(span, source.text):
            # Verbatim-span guarantee: the span must actually occur in the
            # source text. A non-empty-but-fabricated span (hallucinated, injected,
            # or from a scanned/empty PDF) is dropped here, not trusted.
            continue
        try:
            claim = Claim(
                claim_id=f"{source.source_id}-c{i+1}",
                statement=statement,
                mechanism=(c.get("mechanism") or "").strip(),
                scope=(c.get("scope") or "").strip(),
                horizon=(c.get("horizon") or "").strip(),
                source_id=source.source_id,
                source_span=span,
                # B-014: claimed_metric_quote is also "verbatim" per the prompt — verify it occurs in
                # the source text; drop a fabricated metric (it flows into spec_gen/report otherwise).
                claimed_metric_quote=(lambda q: q if (q and span_in_text(q, source.text)) else "")(
                    (c.get("claimed_metric_quote") or "").strip()),
                applicable_strategy_class=(c.get("applicable_strategy_class")
                                           or "unspecified"),
                sample_period=c.get("sample_period"),
                expected_edge=_expected_edge_or_none(c.get("expected_edge")),
            )
        except ValueError:
            claim = Claim(
                claim_id=f"{source.source_id}-c{i+1}",
                statement=statement,
                mechanism=(c.get("mechanism") or "").strip(),
                scope=(c.get("scope") or "").strip(),
                horizon=(c.get("horizon") or "").strip(),
                source_id=source.source_id,
                source_span=span,
                claimed_metric_quote=(lambda q: q if (q and span_in_text(q, source.text)) else "")(
                    (c.get("claimed_metric_quote") or "").strip()),
                applicable_strategy_class=(c.get("applicable_strategy_class")
                                           or "unspecified"),
                expected_edge=_expected_edge_or_none(c.get("expected_edge")),
            )
        claims.append(claim)

    prov = {
        "role": "claim_extractor", "model": resp.model, "in_tokens": resp.in_tokens,
        "out_tokens": resp.out_tokens, "cost_usd": round(resp.cost_usd, 5),
        "cached": resp.cached, "n_extracted": len(claims),
        "n_rejected_no_span": len(raw) - len(claims),
        "input_chars": len(body), "truncated": len(source.text) > len(body),
    }
    return claims, prov


def fallback_claims(source) -> tuple[list[Claim], dict]:
    """Offline path: try to import a hand-authored claims.py for this source.

    Used when no LLM API key is configured. The decision record carries
    `extracted_by: "manual-fallback"` so a hand-authored run is never mistaken
    for an LLM-driven one.
    """
    try:
        from . import claims as manual  # noqa: PLC0415
    except ImportError as e:
        return [], {"extracted_by": "manual-fallback", "error": str(e),
                    "n_extracted": 0}

    if manual.SOURCE_ID != source.source_id:
        return [], {"extracted_by": "manual-fallback",
                    "error": f"claims.py is for {manual.SOURCE_ID}, "
                             f"not {source.source_id}",
                    "n_extracted": 0}

    return list(manual.CLAIMS), {"extracted_by": "manual-fallback",
                                 "n_extracted": len(manual.CLAIMS)}


# --- P3: falsifiability classifier ------------------------------------------ #

P3_SYSTEM = (
    "You are a falsifiability classifier for a research pipeline. You decide "
    "whether a claim can be tested deterministically (via a backtest against "
    "data) or whether it is qualitative narrative that no historical data can "
    "falsify. Be strict: 'companies with strong management outperform' is "
    "qualitative-only. 'Daily |Δprob| in KXFED predicts 5-day-ahead BTC vol' "
    "is generated-module-testable. Never call anything testable that lacks a "
    "measurable outcome and a time horizon. Respond strictly in JSON."
)

P3_USER_TMPL = """Classify this claim:
{{
  "statement": "{statement}",
  "mechanism": "{mechanism}",
  "scope":    "{scope}",
  "horizon":  "{horizon}"
}}

Output JSON: {{"route": "deterministic-testable" | "generated-module-testable" |
"qualitative-only" | "unfalsifiable", "reason": str, "note": str}}
"""


def classify_claim(claim: Claim) -> dict:
    """Run the P3 LLM role; return route/reason/note dict."""
    user = P3_USER_TMPL.format(
        statement=claim.statement[:500],
        mechanism=(claim.mechanism or "")[:300],
        scope=(claim.scope or "")[:200],
        horizon=(claim.horizon or "")[:80],
    )
    parsed, resp = llm.call_json(
        "falsifiability_classifier",
        [{"role": "system", "content": P3_SYSTEM},
         {"role": "user",   "content": user}],
        temperature=0.0,
    )
    route = parsed.get("route", "unfalsifiable") if isinstance(parsed, dict) else "unfalsifiable"
    if route not in ("deterministic-testable", "generated-module-testable",
                     "qualitative-only", "unfalsifiable"):
        route = "unfalsifiable"
    return {
        "stage": "P3", "route": route, "killed": route == "unfalsifiable",
        "reason": "unfalsifiable" if route == "unfalsifiable" else None,
        "note": parsed.get("note", "") if isinstance(parsed, dict) else "",
        "_llm": {"model": resp.model, "cost_usd": round(resp.cost_usd, 5),
                 "cached": resp.cached},
    }


def classify_claim_stub(claim: Claim) -> dict:
    """Offline fallback: assume testable. Flags the assumption."""
    return {"stage": "P3", "route": "generated-module-testable", "killed": False,
            "reason": None,
            "note": "OFFLINE: LLM unavailable; assumed testable. Re-run with API key.",
            "_llm": {"model": "stub", "cost_usd": 0.0, "cached": False}}
