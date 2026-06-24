"""Relevance gate (pre-P2): is this paper testable against penrose's data domains?

The loop's paper-grabber can pull off-domain papers (a CERN physics paper slipped
through once). Extracting claims + auto-implementing modules for a paper whose data we
can never source just burns LLM budget and clutters the data-request backlog. This is a
cheap pre-screen on the abstract: if NO claim is plausibly testable with the data we have
(or could realistically collect), the paper is flagged off_domain and skipped before the
expensive stages.

Fail-OPEN: any error (LLM down, parse fail) returns relevant=True — we never silently
drop a paper because the screener hiccuped; the worst case is we fall back to the old
behavior (extract + needs_data).
"""
from __future__ import annotations

from .. import llm

# The data domains penrose can test against today (catalog) or plausibly collect.
DATA_DOMAINS = [
    "crypto spot prices (BTC, ETH, SOL, LINK, AVAX, ADA — daily)",
    "crypto perpetual-futures funding rates (BTC, ETH, SOL)",
    "crypto realized & implied volatility (BTC)",
    "prediction-market probability series (Kalshi, Polymarket — macro/event contracts)",
    "daily weather temperatures (US cities)",
]

_SYSTEM = (
    "You are a relevance gate for a quantitative-research pipeline. The pipeline can only "
    "backtest a claim if it can be expressed against one of these data domains:\n"
    + "\n".join(f"  - {d}" for d in DATA_DOMAINS)
    + "\nGiven a paper's title and abstract, decide whether AT LEAST ONE of its empirical "
    "claims could plausibly be tested with that data (directly, or via a close proxy we "
    "could collect). A paper about, say, nuclear physics, corporate-bond default intensity, "
    "or pure martingale theory with no tradable empirical claim is NOT relevant. A paper "
    "about crypto returns/volatility/funding, prediction-market efficiency, or weather "
    "markets IS relevant. Respond ONLY with JSON: "
    '{"relevant": true|false, "domains": [<matching domain keywords>], '
    '"reason": "<one sentence>"}.'
)


def screen(title: str, text: str, *, role: str = "falsifiability_classifier") -> dict:
    """Return {'relevant': bool, 'domains': [...], 'reason': str}. Fails open."""
    abstract = (text or "")[:2800]
    user = f"TITLE: {title or '(untitled)'}\n\nABSTRACT / OPENING:\n{abstract}"
    try:
        parsed, _ = llm.call_json(
            role,
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
        if not isinstance(parsed, dict) or "relevant" not in parsed:
            return {"relevant": True, "domains": [], "reason": "screen inconclusive (fail-open)"}
        return {
            "relevant": bool(parsed.get("relevant")),
            "domains": parsed.get("domains") or [],
            "reason": str(parsed.get("reason", ""))[:200],
        }
    except Exception as e:  # noqa: BLE001 — never drop a paper on a screener error
        return {"relevant": True, "domains": [], "reason": f"screen errored, fail-open: {e}"}
