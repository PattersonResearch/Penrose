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
})
DEFAULT_CLAIM_TYPE = "trading_strategy"

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


def classify_claim_type(claim) -> str:
    """Return a deterministic claim type, failing open to trading_strategy."""
    try:
        text = " ".join([
            getattr(claim, "statement", "") or "",
            getattr(claim, "mechanism", "") or "",
            getattr(claim, "claimed_metric_quote", "") or "",
            getattr(claim, "source_span", "") or "",
        ]).lower()
    except Exception:  # noqa: BLE001
        return DEFAULT_CLAIM_TYPE
    if not text.strip():
        return DEFAULT_CLAIM_TYPE

    descriptive = sum(1 for pat in _DESCRIPTIVE_PATTERNS if re.search(pat, text))
    trading = sum(1 for pat in _TRADING_PATTERNS if re.search(pat, text))
    structural = sum(1 for pat in _STRUCTURAL_PATTERNS if re.search(pat, text))

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
