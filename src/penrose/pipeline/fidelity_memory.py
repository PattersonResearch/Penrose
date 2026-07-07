"""Claim-type routing and fidelity-rejection memory for generated modules.

This module is intentionally small and deterministic. The classifier is a
fail-open heuristic: when it cannot find a clear shape cue, it returns today's
implicit default, ``trading_strategy``.
"""
from __future__ import annotations

import fcntl
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config

CLAIM_TYPES = frozenset({
    "descriptive_statistical",
    "trading_strategy",
    "structural_proposition",
    "provided_series_statistic",
})
DEFAULT_CLAIM_TYPE = "trading_strategy"

# 6g: "test the statistic of a provided/pre-computed series" is a FIRST-CLASS claim type,
# distinct from (and checked before) descriptive_statistical/trading_strategy. It fires on
# claims that declare a single pooled/cohort statistic over series the claim itself names
# (e.g. a pre-registered cohort-mean test) -- the exact shape that made spec-gen either
# invent gates the claim never stated (over-specification) or fall back to an empty
# trading-strategy stub (under-specification), because a provided-series-statistics claim
# was otherwise misrouted as trading_strategy.
_PROVIDED_SERIES_STAT_STRONG_PATTERNS = [
    r"\bone\s+(?:declared\s+)?deflation cohort\b",
    r"\bsingle\s+(?:declared\s+)?deflation cohort\b",
]
_PROVIDED_SERIES_STAT_PROVENANCE_PATTERNS = [
    r"\bdeclared series\b",
    r"\bprovided series\b",
    r"\bpre-?computed series\b",
]
_PROVIDED_SERIES_STAT_WEAK_PATTERNS = [
    r"\bpooled\s+(?:one[- ]sample\s+)?statistic\b",
    r"\bcohort[- ]level mean\b",
    r"\bcohort[- ]mean\b",
    r"\bone[- ]sample\b",
    r"\bpooled mean\b",
    r"\bsingle (?:pooled )?statistic\b",
    r"\bone (?:pooled )?statistic\b",
]
_PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS = [
    r"\balready\s+(?:pre-?)?encodes?\b.{0,40}\b(?:p&l|pnl|net|return|returns|profit|edge)\b",
    r"\bseries\b.{0,40}\balready\s+encodes?\b",
]
_PROVIDED_SERIES_DECLARATION_PATTERNS = [
    *_PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS,
    r"\b(?:provided|pre-?computed|declared|pre-?encoded)\s+(?:net\s+)?(?:p&l|pnl|return|returns|series)\b",
    r"\bpool these\s+\d*\s*(?:[\w&.-]+\s+)*series\b",
]
_TRADING_CONSTRUCTION_PATTERNS = [
    r"\bgo\s+long\b|\bgo\s+short\b|\blong[- ]short\b|\blong[- ]only\b|\bshort[- ]only\b",
    r"\bentry\b|\bexit\b",
    r"\bposition\b|\bpositions\b",
    r"\bsignal\b",
    r"\bmomentum\b",
    r"\brebalance\b|\brebalanced\b|\brebalancing\b",
]

_DESCRIPTIVE_PATTERNS = [
    r"\bunconditional\s+(?:mean|average|bias|frequency|correlation)\b",
    r"\b(?:mean|average)\s+(?:bias|return|effect|difference)\b",
    r"\bcorrelation\b",
    r"\bfrequency\b",
    r"\bfraction\b",
    r"\bpercent(?:age)?\s+of\b",
    r"\bobservations?\b",
    r"\bci\b|\bconfidence interval\b",
]
_TRADING_PATTERNS = [
    r"\bsignal\b",
    r"\bentry\b|\bexit\b",
    r"\bposition\b|\bpositions\b",
    r"\bpnl\b|\bp&l\b",
    r"\bsharpe\b",
    r"\btrade\b|\btrading\b",
    r"\bmomentum\b",
    r"\blong\b|\bshort\b",
    r"\bstrategy\b",
]
_STRUCTURAL_PATTERNS = [
    r"\bmarket structure\b",
    r"\bmicrostructure\b",
    r"\bcauses?\b",
    r"\bmechanism\b",
    r"\binstitutional\b",
    r"\bshould\b",
]

_PREREGISTERED_SINGLE_STAT_PATTERNS = [
    r"\b(?:exactly\s+)?one\s+pre[- ]?registered\s+statistic\b",
    r"\bsingle\s+pre[- ]?registered\s+statistic\b",
    r"\b(?:one|single)\s+(?:declared\s+)?deflation cohort\b",
    r"\bcounts?\s+as\s+one\s+pre[- ]?registered\s+search\b",
    r"\bpre[- ]?registered\s+search\s+denominator\s*(?:of\s+|is\s+|=\s*)1\b",
    r"\b(?:deflation\s+)?cohort\s+denominator\s*(?:of\s+|is\s+|=\s*)1\b",
]
_SINGLE_POOLED_TEST_ASSERTION = (
    r"\b(?:one|single)\s+(?:pooled\s+)?test\b|"
    r"\bpooled\s+test\b.{0,40}\b(?:one|single)\b"
)
_PREREGISTRATION_CONTEXT_PATTERN = (
    r"\bpre[- ]?registered\b|"
    r"\bdeflation cohort\b|"
    r"\bpre[- ]?registered search\b|"
    r"\b(?:exactly\s+)?one\s+pre[- ]?registered\s+statistic\b|"
    r"\bsingle\s+pre[- ]?registered\s+statistic\b"
)


def _claim_source_text(claim, source=None) -> str:
    parts = []
    for attr in ("statement", "mechanism", "claimed_metric_quote", "source_span"):
        try:
            parts.append(getattr(claim, attr, "") or "")
        except Exception:  # noqa: BLE001
            pass
    try:
        parts.append(getattr(source, "text", "") or "")
    except Exception:  # noqa: BLE001
        pass
    if isinstance(source, dict):
        try:
            parts.append(str(source.get("text") or ""))
        except Exception:  # noqa: BLE001
            pass
    return " ".join(str(p) for p in parts if p).lower()


def is_preregistered_single_cohort(claim, source=None) -> bool:
    """True only for an explicit one-cohort pre-registration assertion.

    This is verdict-integrity bookkeeping, not claim routing. Generic statistical
    prose such as "one-sample t-test" or "pooled mean" must not earn the reduced
    deflation denominator.
    """
    try:
        text = _claim_source_text(claim, source)
        if not text.strip():
            return False
        if any(re.search(pat, text) for pat in _PREREGISTERED_SINGLE_STAT_PATTERNS):
            return True
        return bool(
            re.search(r"\bno multiplicity correction\b", text)
            and re.search(_SINGLE_POOLED_TEST_ASSERTION, text)
            and re.search(_PREREGISTRATION_CONTEXT_PATTERN, text)
        )
    except Exception:  # noqa: BLE001
        return False


def classify_claim_type(claim, source=None) -> str:
    """Return a deterministic claim type, failing open to trading_strategy.

    Classification is keyword/regex-based over the ENGLISH claim text. Non-English
    claims, or unusual phrasings that match no cue, fall through to the
    `trading_strategy` default — the conservative fail-open (the claim is tested as
    a strategy rather than mis-specialized), never a crash.
    """
    try:
        claim_text = " ".join([
            getattr(claim, "statement", "") or "",
            getattr(claim, "mechanism", "") or "",
            getattr(claim, "claimed_metric_quote", "") or "",
            getattr(claim, "source_span", "") or "",
        ]).lower()
    except Exception:  # noqa: BLE001
        return DEFAULT_CLAIM_TYPE
    if not claim_text.strip():
        return DEFAULT_CLAIM_TYPE

    # Checked FIRST and independently of the descriptive/trading tally, but only for
    # explicit provided-series declarations. Ambiguous pooled/cohort/one-sample prose is
    # handled by the declaration-gated weak branch below.
    if any(re.search(pat, claim_text) for pat in _PROVIDED_SERIES_STAT_STRONG_PATTERNS):
        return "provided_series_statistic"
    if any(re.search(pat, claim_text) for pat in _PROVIDED_SERIES_STAT_PROVENANCE_PATTERNS):
        trading_construction = sum(
            1 for pat in _TRADING_CONSTRUCTION_PATTERNS if re.search(pat, claim_text)
        )
        high_confidence_encoded = any(
            re.search(pat, claim_text)
            for pat in _PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS
        )
        if trading_construction >= 2 and not high_confidence_encoded:
            return "trading_strategy"
        return "provided_series_statistic"
    source_text = (getattr(source, "text", "") or "").lower()
    stat_text = " ".join([claim_text, source_text])
    has_weak_stat = any(
        re.search(pat, stat_text) for pat in _PROVIDED_SERIES_STAT_WEAK_PATTERNS
    )
    if has_weak_stat:
        declaration_text = stat_text
        if any(re.search(pat, declaration_text) for pat in _PROVIDED_SERIES_DECLARATION_PATTERNS):
            trading_construction = sum(
                1 for pat in _TRADING_CONSTRUCTION_PATTERNS if re.search(pat, claim_text)
            )
            high_confidence_encoded = any(
                re.search(pat, declaration_text)
                for pat in _PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS
            )
            if trading_construction >= 2 and not high_confidence_encoded:
                return "trading_strategy"
            return "provided_series_statistic"

    descriptive = sum(1 for pat in _DESCRIPTIVE_PATTERNS if re.search(pat, claim_text))
    trading = sum(1 for pat in _TRADING_PATTERNS if re.search(pat, claim_text))
    structural = sum(1 for pat in _STRUCTURAL_PATTERNS if re.search(pat, claim_text))

    if descriptive and descriptive >= trading:
        return "descriptive_statistical"
    if trading:
        return "trading_strategy"
    if structural:
        return "structural_proposition"
    return DEFAULT_CLAIM_TYPE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store_path() -> Path:
    return Path(getattr(config, "FIDELITY_REJECTIONS", config.REPORTS / "fidelity_rejections.jsonl"))


def append_rejection(*, strategy_class: str, claim_type: str, divergences, note: str = "") -> None:
    """Persist one faithful=false rejection using flock + tmp + replace.

    Corrupt or missing existing rows are ignored rather than surfacing into the
    pipeline. This is learning feedback, not a verdict dependency.
    """
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy_class": str(strategy_class or "unspecified"),
        "claim_type": claim_type if claim_type in CLAIM_TYPES else DEFAULT_CLAIM_TYPE,
        "divergences": [str(d)[:500] for d in (divergences or []) if str(d).strip()][:5],
        "note": str(note or "")[:500],
        "ts": _now(),
    }
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows: list[str] = []
        if path.exists():
            rows = [line for line in path.read_text().splitlines() if line.strip()]
        rows.append(json.dumps(row, sort_keys=True))
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("\n".join(rows) + "\n")
        tmp.replace(path)


def rejection_guidance(strategy_class: str, claim_type: str, *, limit: int = 3) -> str:
    """Return a capped prompt block for prior divergences, or "" fail-open."""
    path = _store_path()
    if not path.exists():
        return ""
    try:
        wanted_class = str(strategy_class or "").strip()
        wanted_type = claim_type if claim_type in CLAIM_TYPES else DEFAULT_CLAIM_TYPE
        seen: set[str] = set()
        items: list[str] = []
        for line in reversed(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("strategy_class") != wanted_class and row.get("claim_type") != wanted_type:
                continue
            for div in row.get("divergences") or []:
                div = str(div).strip()
                if div and div not in seen:
                    seen.add(div)
                    items.append(div[:300])
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break
        if not items:
            return ""
        bullets = "\n".join(f"- {item}" for item in items)
        return (
            "AVOID THESE PAST FIDELITY FAILURES for this strategy/claim type:\n"
            f"{bullets}\n"
        )
    except Exception:  # noqa: BLE001
        return ""
