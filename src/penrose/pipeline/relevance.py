"""Scope gate (pre-P2): does this paper contain a falsifiable quantitative performance claim?

The loop's paper-grabber can pull genuinely out-of-scope papers (a CERN physics paper slipped
through once). Extracting claims + auto-implementing modules for a paper with NO tradable
performance claim just burns LLM budget. This is a cheap pre-screen on the abstract that judges
SCOPE, not data availability: a paper with at least one falsifiable performance claim is relevant
even when we do not currently hold its data — that becomes a `needs_data` result downstream, NOT an
off-domain rejection. Only papers with no tradable empirical performance claim at all (pure theory,
a governance/survey study, physics) are flagged off_domain and skipped before the expensive stages.
The currently-held data domains are still passed to the model as INFORMATIONAL context (so it can
tag which a claim maps to), but absence from that list is never grounds for off_domain.

Fail-OPEN: any error (LLM down, parse fail) returns relevant=True — we never silently drop a paper
because the screener hiccuped; the worst case is we fall back to extract + needs_data.
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
        "You are a SCOPE gate for a quantitative-research falsification pipeline. The pipeline "
        "evaluates falsifiable claims about investment PERFORMANCE: abnormal or risk-adjusted "
        "return, return predictability, a tradeable signal, a factor, a trading strategy, or any "
        "claim that a rule produces a measurable edge in asset returns.\n\n"
        "Given a paper's title and abstract, decide whether it contains AT LEAST ONE such "
        "falsifiable, quantitative performance claim that could in principle be tested on market "
        "data. Judge SCOPE, NOT data availability: do NOT answer 'not relevant' merely because the "
        "specific dataset would be hard to obtain or is outside what the pipeline currently holds "
        "— whether the data exists is decided LATER in the pipeline (it becomes a 'needs_data' "
        "result, not an off-domain rejection).\n\n"
        "RELEVANT (in scope): a momentum, value, carry, reversal, or anomaly claim in ANY asset "
        "class (equities, futures, FX, crypto, rates, commodities); a return-predictability or "
        "market-timing claim; a trading-strategy performance claim; a prediction-market efficiency "
        "or volatility-edge claim. In scope even if the asset class is one the pipeline does not "
        "currently have data for.\n\n"
        "NOT RELEVANT (out of scope): a paper with NO tradable, quantitative performance claim at "
        "all — pure theory or mathematics, a corporate-governance or survey study, an accounting "
        "or causal-inference result with no return prediction, physics, etc.\n\n"
        "For context only, the pipeline currently holds or can readily collect data for:\n"
        + "\n".join(f"  - {d}" for d in _data_domains())
        + "\n(This list is INFORMATIONAL — a claim outside it is still in scope; it just routes to "
        "needs_data later.)\n\n"
        "Respond ONLY with JSON: "
        '{"relevant": true|false, "domains": [<held-data domains the claim maps to, if any>], '
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
