"""Brain connection-discovery — find structure across the verdict corpus (kills, principles, links).

HARD RULE — INFORM, NEVER GATE (user directive, 2026-06-20). This module is READ-ONLY and ADVISORY.
It consumes verdict records and emits connections (clusters / cross-domain links / principles /
similarity edges) as METADATA. It MUST NEVER feed back into the verdict path (p8_verdict / run): a
new claim is always tested independently on its own data; the corpus only contextualizes the result
for the human, it never pre-rejects an idea. Guardrails baked in:
  * Principles are drawn ONLY from genuine STRUCTURAL kills (in_sample_only / regime_fragile /
    walk_forward_drift / no_signal_alignment). NEVER from `underpowered`
    (below_detection_floor): "we couldn't resolve it" must not become "this neighborhood is dead."
  * Every principle carries n, the power of its supporting kills, a date, and a confidence that
    DECAYS with age — a kill is conditional ("died under THESE conditions"), not eternal.
  * Similarity links FLAG + surface the DIFFERENCES; they never imply "skip testing this."
This file deliberately imports nothing from pipeline.stages / pipeline.run, so it CANNOT be wired
into the verdict logic by accident.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

from .strategy_family import family_key, normalize_strategy_family
# Genuine STRUCTURAL kills — power-INDEPENDENT (an edge that is actually broken). Only these inform
# suppressive priors. `below_detection_floor` (underpowered) is excluded ON PURPOSE.
STRUCTURAL_KILLS = {"in_sample_only", "regime_fragile", "walk_forward_drift",
                    "no_signal_alignment", "tail_asymmetric"}

# Coarse research-domain inference from a claim's text — for CROSS-DOMAIN links (a failure mode that
# crosses research areas is more interesting than one confined to a single paper).
_DOMAIN_KEYWORDS = [
    ("crypto-volatility", ("volatility", "realized vol", "implied vol", "vrp", "dvol", "garch")),
    ("funding-carry", ("funding", "perpetual", "perp", "basis", "carry")),
    ("pm-microstructure", ("order book", "liquidation", "resolution", "halt", "margin", "depth", "perpetual futures on")),
    ("prediction-market", ("kalshi", "polymarket", "prediction market", "binary", "election")),
    ("crypto-equity", ("equity", "equities", "correlation", "asset class", "inflation", "breakeven")),
    ("trend-following", ("trend-following", "trend following", "ewmac", "moving-average crossover", "moving average crossover")),
    ("cross-sectional-factor", ("momentum", "reversal", "value", "accruals", "size", "factor", "anomaly", "long-short")),
    ("macro-signal", ("fed", "cpi", "recession", "macro", "rate")),
]


def infer_domain(text: str, fallback: str = "other") -> str:
    t = (text or "").lower()
    for dom, kws in _DOMAIN_KEYWORDS:
        if any(k in t for k in kws):
            return dom
    return fallback


@dataclass
class Record:
    """One normalized verdict from the corpus."""
    id: str
    domain: str
    verdict: str
    kill_reason: Optional[str]
    statement: str
    structural: bool                 # is this a genuine structural kill (informs priors)?
    power_sufficient: Optional[bool]
    date: str = ""                   # ISO; for confidence decay
    synthetic: bool = False
    claim_id: Optional[str] = None
    strategy_family: Optional[dict] = None


@dataclass
class Connections:
    failure_clusters: list = field(default_factory=list)
    cross_domain: list = field(default_factory=list)
    principles: list = field(default_factory=list)
    contrastive: list = field(default_factory=list)
    similarity_links: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _year(date: str) -> Optional[int]:
    m = re.search(r"(20\d\d)", date or "")
    return int(m.group(1)) if m else None


def failure_clusters(records: list[Record]) -> list[dict]:
    """Group genuine structural kills by (kill_reason, domain). Foundation of everything else."""
    groups = defaultdict(list)
    for r in records:
        if r.structural:
            groups[(r.kill_reason, r.domain)].append(r.id)
    out = [{"kill_reason": kr, "domain": dom, "n": len(ids), "members": ids}
           for (kr, dom), ids in groups.items() if len(ids) >= 2]
    return sorted(out, key=lambda c: -c["n"])


def cross_domain_links(records: list[Record]) -> list[dict]:
    """Same structural failure mode appearing across DIFFERENT domains — the surprising connections.
    'regime fragility killed both a crypto-vol claim and a PM-microstructure claim.'"""
    by_reason = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.structural:
            by_reason[r.kill_reason][r.domain].append(r.id)
    out = []
    for kr, doms in by_reason.items():
        if len(doms) >= 2:
            out.append({"kill_reason": kr, "domains": sorted(doms),
                        "n_domains": len(doms),
                        "examples": {d: ids[:3] for d, ids in doms.items()},
                        "total": sum(len(v) for v in doms.values())})
    return sorted(out, key=lambda c: -c["n_domains"])


def _support_key(r: Record) -> str:
    return str(r.claim_id or r.id)


def _support_sort_key(r: Record) -> tuple[str, str]:
    return (str(r.claim_id or r.id), str(r.id))


def _unique_structural_records(records: list[Record]) -> list[Record]:
    by_support: dict[str, Record] = {}
    for r in records:
        if not r.structural:
            continue
        support = _support_key(r)
        if support and support not in by_support:
            by_support[support] = r
    return sorted(by_support.values(), key=_support_sort_key)


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_+.-]+", "-", str(value or "").lower()).strip("-")


def _plural_method(method: str) -> str:
    return method + "s"


def _principle_confidence(rs: list[Record], current_year: int, half_life_years: float) -> tuple[int, float]:
    years = [y for y in (_year(r.date) for r in rs) if y]
    newest = max(years) if years else current_year
    age = max(0, current_year - newest)
    confidence = round(0.5 * (0.5 ** (age / half_life_years)), 3)
    return newest, confidence


def _principle_row(
    *,
    level: str,
    label: str,
    kill_reason: str,
    rs: list[Record],
    current_year: int,
    half_life_years: float,
    strategy_family: dict | None = None,
    domain: str | None = None,
) -> dict:
    rs = sorted(rs, key=_support_sort_key)
    newest, confidence = _principle_confidence(rs, current_year, half_life_years)
    example_claim_ids = sorted({_support_key(r) for r in rs if _support_key(r)})[:5]
    supporting = [r.id for r in rs]
    if level == "exact":
        components = "+".join(strategy_family["components"]) if strategy_family else label
        method = strategy_family["method"] if strategy_family else "single"
        statement = (
            f"{components} {_plural_method(method)} fail with '{kill_reason}': "
            f"{len(rs)} independent claims died this way. Treat matching composite claims as "
            "a review prior, then independently test the next claim."
        )
        meaning = (
            f"{components} {_plural_method(method)} repeatedly die with {kill_reason}; "
            "treat the exact family as a review prior, then independently test the next claim."
        )
        pid = f"principle-family-{_sanitize_id(label)}-{_sanitize_id(kill_reason)}"
    elif level == "method":
        statement = (
            f"{_plural_method(label)} fail with '{kill_reason}': {len(rs)} independent claims "
            "died this way. Treat matching method claims as a review prior, then independently "
            "test the next claim."
        )
        meaning = (
            f"{_plural_method(label)} repeatedly die with {kill_reason}; treat the method as a "
            "review prior, then independently test the next claim."
        )
        pid = f"principle-method-{_sanitize_id(label)}-{_sanitize_id(kill_reason)}"
    elif level == "component":
        statement = (
            f"{label}-containing strategies fail with '{kill_reason}': {len(rs)} independent "
            "claims died this way. Treat matching component claims as a review prior, then "
            "independently test the next claim."
        )
        meaning = (
            f"{label}-containing strategies repeatedly die with {kill_reason}; treat the component "
            "as a review prior, then independently test the next claim."
        )
        pid = f"principle-component-{_sanitize_id(label)}-{_sanitize_id(kill_reason)}"
    else:
        statement = (
            f"In the {label} domain, claims that fail with '{kill_reason}' recur: "
            f"{len(rs)} independent claims died this way. Treat new {label} claims as "
            f"likely to share this failure mode - but TEST them; this is a prior, not a verdict."
        )
        meaning = (
            f"{label} claims repeatedly die with {kill_reason}; treat the pattern as a review prior, "
            "then independently test the next claim."
        )
        pid = f"principle-{label}-{kill_reason}"
    row = {
        "principle_id": pid,
        "statement": statement,
        "domain": domain or label,
        "kill_reason": kill_reason,
        "family_level": level,
        "n_observations": len(rs),
        "supporting_kill_count": len(rs),
        "example_claim_ids": example_claim_ids,
        "meaning": meaning,
        "supporting": supporting,
        "newest_evidence_year": newest,
        "confidence": confidence,
        "caveat": "Advisory only (inform-never-gate). Built from structural kills, excludes "
                  "underpowered. Confidence decays with age; a stale kill does not bar a new test.",
    }
    if strategy_family is not None:
        row["strategy_family"] = strategy_family
    return row


def _family_or_none(r: Record) -> dict | None:
    return normalize_strategy_family(r.strategy_family)


def propose_principles(records: list[Record], current_year: int = 2026,
                       min_kills: Optional[int] = None, half_life_years: float = 4.0,
                       min_obs: Optional[int] = None) -> list[dict]:
    """Power-aware principles from STRUCTURAL kills only. Confidence decays with the age of the
    supporting evidence (kills are conditional, not eternal). NO LLM here - a deterministic,
    auditable statement; an LLM phrasing pass can sit on top later, gated by reproduce-not-trust.

    This is a cross-run proposal distiller: it groups all recorded structural kills by the same
    reason and domain/class across the full corpus. The result is advisory only. Promotion into
    ``principles.jsonl`` or the trusted BrainStore remains a human P9 action.
    """
    threshold = min_kills if min_kills is not None else (min_obs if min_obs is not None else 3)
    try:
        threshold = max(3, int(threshold))
    except (TypeError, ValueError):
        threshold = 3
    principles = []
    assigned: set[str] = set()
    structural = _unique_structural_records(records)

    exact_groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    fallback_groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in structural:
        # `r.structural` is the correct (and only needed) guardrail: an `underpowered` verdict
        # has structural=False and is already excluded. A genuine STRUCTURAL kill (e.g. 3-fold
        # sign-instability) is power-INDEPENDENT evidence the edge is broken, so it counts toward a
        # principle even on a thin sample. (Do NOT also exclude on power_sufficient — that wrongly
        # drops legitimate structural kills and was a real bug found in testing.)
        fam = _family_or_none(r)
        if fam is None:
            fallback_groups[(str(r.kill_reason), r.domain)].append(r)
            continue
        exact_groups[(str(r.kill_reason), family_key(fam))].append(r)

    for (kr, key), rs in sorted(exact_groups.items()):
        if len(rs) < threshold:
            continue
        fam = _family_or_none(rs[0])
        if fam is None:
            continue
        principles.append(_principle_row(
            level="exact", label=key, kill_reason=kr, rs=rs,
            current_year=current_year, half_life_years=half_life_years,
            strategy_family=fam, domain=key,
        ))
        assigned.update(_support_key(r) for r in rs)

    method_groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in structural:
        if _support_key(r) in assigned:
            continue
        fam = _family_or_none(r)
        if fam is not None:
            method_groups[(str(r.kill_reason), fam["method"])].append(r)

    for (kr, method), rs in sorted(method_groups.items()):
        if len(rs) < threshold:
            continue
        principles.append(_principle_row(
            level="method", label=method, kill_reason=kr, rs=rs,
            current_year=current_year, half_life_years=half_life_years,
            domain=method,
        ))
        assigned.update(_support_key(r) for r in rs)

    component_candidates: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in structural:
        if _support_key(r) in assigned:
            continue
        fam = _family_or_none(r)
        if fam is None:
            continue
        for component in fam["components"]:
            component_candidates[(str(r.kill_reason), component)].append(r)
    while True:
        available = {
            key: [r for r in rs if _support_key(r) not in assigned]
            for key, rs in component_candidates.items()
        }
        eligible = [
            (key, sorted(rs, key=_support_sort_key))
            for key, rs in available.items()
            if len(rs) >= threshold
        ]
        if not eligible:
            break
        (kr, component), rs = sorted(eligible, key=lambda item: (-len(item[1]), item[0]))[0]
        principles.append(_principle_row(
            level="component", label=component, kill_reason=kr, rs=rs,
            current_year=current_year, half_life_years=half_life_years,
            domain=component,
        ))
        assigned.update(_support_key(r) for r in rs)

    for (kr, dom), rs in sorted(fallback_groups.items()):
        if len(rs) < threshold:
            continue
        principles.append(_principle_row(
            level="domain", label=dom, kill_reason=kr, rs=rs,
            current_year=current_year, half_life_years=half_life_years,
            domain=dom,
        ))

    return sorted(principles, key=lambda p: (
        -p["n_observations"], p["family_level"], p["domain"], p["kill_reason"]
    ))


def propose_contrastive_principles(records: list[Record], current_year: int = 2026,
                                   min_kills: int = 3, min_survivors: int = 2) -> list[dict]:
    """Learn from the boundary between what SURVIVES and what dies.

    For each recurring structural failure mode in one domain, contrast it with
    other domains that survive to watch/research-supported. Advisory only.
    """
    kills_by_reason_domain = defaultdict(list)
    survivors_by_domain = defaultdict(list)
    for r in records:
        if r.structural and r.kill_reason:
            kills_by_reason_domain[(r.kill_reason, r.domain)].append(r.id)
        if r.verdict in ("watch", "research-supported"):
            survivors_by_domain[r.domain].append(r.id)

    principles = []
    for (reason, kdom), kill_ids in kills_by_reason_domain.items():
        n_kills = len(kill_ids)
        if n_kills < min_kills:
            continue
        survivor_domains = {
            dom: ids
            for dom, ids in survivors_by_domain.items()
            if dom != kdom and len(ids) >= min_survivors
        }
        if not survivor_domains:
            continue
        principles.append({
            "principle_id": f"contrast-{kdom}-{reason}",
            "kind": "contrastive",
            "statement": (
                f"'{reason}' recurs in the {kdom} domain ({n_kills} structural kills), but claims in "
                f"{sorted(survivor_domains)} survive to watch+ - {reason} is specific to {kdom}, not "
                f"universal. In that context prefer the surviving class - but TEST it; this is a prior."
            ),
            "kill_reason": reason,
            "kill_domain": kdom,
            "n_kills": n_kills,
            "survivor_domains": {d: ids[:3] for d, ids in survivor_domains.items()},
            "n_survivor_domains": len(survivor_domains),
            "caveat": "Advisory only (inform-never-gate). Learns from the survivor-vs-kill boundary; "
                      "contextual, not a guarantee.",
        })
    return sorted(principles, key=lambda p: -p["n_kills"])


def similarity_links(records: list[Record], threshold: float = 0.6, max_links: int = 50) -> list[dict]:
    """Advisory text-similarity edges between claim statements. FLAGS related prior verdicts and
    surfaces the DIFFERENCE — never implies 'skip testing'. (Lightweight difflib; embeddings later.)"""
    out = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if not a.statement or not b.statement:
                continue
            sim = difflib.SequenceMatcher(None, a.statement.lower(), b.statement.lower()).ratio()
            if sim >= threshold:
                out.append({"a": a.id, "b": b.id, "similarity": round(sim, 3),
                            "a_verdict": a.verdict, "b_verdict": b.verdict,
                            "note": "RELATED prior verdict — review the difference; does NOT mean skip testing."})
    return sorted(out, key=lambda l: -l["similarity"])[:max_links]


def discover(records: list[Record], current_year: int = 2026) -> Connections:
    """Run the full advisory connection-discovery over a verdict corpus."""
    structural = [r for r in records if r.structural]
    c = Connections(
        failure_clusters=failure_clusters(records),
        cross_domain=cross_domain_links(records),
        principles=propose_principles(records, current_year=current_year),
        contrastive=propose_contrastive_principles(records, current_year=current_year),
        similarity_links=similarity_links(records),
        stats={"n_records": len(records), "n_structural_kills": len(structural),
               "n_underpowered": sum(1 for r in records if r.power_sufficient is False),
               "verdicts": dict(Counter(r.verdict for r in records)),
               "domains": dict(Counter(r.domain for r in records))},
    )
    return c
