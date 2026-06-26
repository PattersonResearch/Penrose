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
from ..data.contract import load_catalog_loader

# Permanent floor: live/synthetic-fallback domains data/client.py serves even with
# no catalog. The catalog only unions onto this floor; it never shrinks it.
_BASE_DOMAINS = [
    "crypto spot prices (BTC, ETH, SOL, LINK, AVAX, ADA — daily)",
    "crypto perpetual-futures funding rates (BTC, ETH, SOL)",
    "crypto realized & implied volatility (BTC)",
    "prediction-market probability series (Kalshi, Polymarket — macro/event contracts)",
    "daily weather temperatures (US cities)",
]

_DOMAIN_PHRASING = {
    "rates": "sovereign interest rates / treasury yields (e.g. US 10y, 3m T-bill — daily)",
    "inflation": "inflation expectations / breakevens (e.g. 10y breakeven — daily)",
    "equity": "equity indices / ETFs (e.g. SPY, QQQ, IWM — daily)",
    "commodity": "commodity / metal ETFs (e.g. gold — daily)",
    "crypto-mktcap": "crypto market-cap series (BTC/ETH/SOL — daily)",
    "weather-forecast": "weather forecast skill / model error (multi-model ensemble vs actuals)",
    "weather-market": "weather prediction-market edge & realized PnL series",
}


def _catalog_domains() -> tuple[str, ...]:
    """Distinct catalog domains, phrased for the prompt. Fail-open on any error."""
    from .. import config
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        tags = catalog.domains() if hasattr(catalog, "domains") else []
    except Exception:  # noqa: BLE001 — never break the gate on a catalog hiccup
        return ()
    out = []
    for tag in tags:
        phrase = _DOMAIN_PHRASING.get(tag)
        if phrase:
            out.append(phrase)
    return tuple(out)


def _data_domains() -> list[str]:
    seen, merged = set(), []
    for domain in (*_BASE_DOMAINS, *_catalog_domains()):
        if domain not in seen:
            seen.add(domain)
            merged.append(domain)
    return merged


def _system_prompt() -> str:
    return (
        "You are a relevance gate for a quantitative-research pipeline. The pipeline can only "
        "backtest a claim if it can be expressed against one of these data domains:\n"
        + "\n".join(f"  - {d}" for d in _data_domains())
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
        system_prompt = _system_prompt()
        parsed, _ = llm.call_json(
            role,
            [{"role": "system", "content": system_prompt},
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
