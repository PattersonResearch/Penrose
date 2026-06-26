"""penrose brain layer — atoms in penrose's native BrainStore.

The two-credential firewall is modeled here as two client classes:

  * BrainReader      — read-only. Retrieval, dedup, dreaming, chat pre-flight.
                       Physically cannot call put/link.
  * PromotionClient  — read-write. ONLY constructed inside P9 after a human
                       approves a proposal. This is the single knowledge-write path.

The external brain runtime is retired; this module routes the public API to the
in-repo SQLite BrainStore. Atoms keep slug `atoms/penrose/<kind>/<id>`,
frontmatter with type=atom, kind, scope/source_id=penrose_pm, trust, status, and
typed edges.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from . import brainstore, config
from .proposals import read_proposals

SOURCE_TYPES = frozenset({
    "external_source", "generated_hypothesis", "synthesized_hypothesis",
    "confirmation", "chat",
})
UNANCHORED_SOURCE_TYPES = frozenset({"generated_hypothesis", "synthesized_hypothesis", "chat"})


def validate_source_type(value: str) -> str:
    if value not in SOURCE_TYPES:
        raise ValueError(f"unsupported source_type {value!r}; expected one of {sorted(SOURCE_TYPES)}")
    return value


def source_is_unanchored(value: str) -> bool:
    return validate_source_type(value) in UNANCHORED_SOURCE_TYPES


def _frontmatter(d: dict) -> str:
    lines = ["---"]
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def slug(kind: str, ident: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in ident.lower())
    return f"atoms/penrose/{kind}/{safe}"


class BrainReader:
    """Read-only access. Raises if anyone tries to use it to write."""

    def get(self, slug_: str) -> Optional[str]:
        return brainstore.get(slug_)

    def search(self, query: str, limit: int = 10) -> str:
        return brainstore.search(query, n=limit)

    def graph(self, slug_: str, depth: int = 2) -> str:
        return brainstore.graph(slug_, depth=depth)

    def list(self, prefix: str = "atoms/penrose", n: int = 200) -> str:
        return brainstore.list(prefix=prefix, n=n)


class PromotionClient(BrainReader):
    """Read-write. Constructed ONLY after human approval at P9."""

    def __init__(self, approved_by: str):
        if not str(approved_by or "").strip():
            raise PermissionError("PromotionClient requires an approver (P9 gate).")
        self.approved_by = approved_by

    def put_atom(self, kind: str, ident: str, body: str, **frontmatter: Any) -> dict:
        fm = {"type": "atom", "kind": kind, "scope": config.SCOPE,
              "source_id": config.SCOPE, "status": "active",
              "reviewed_by": self.approved_by, **frontmatter}
        s = slug(kind, ident)
        content = _frontmatter(fm) + "\n\n" + body.strip() + "\n"
        ok = brainstore._put(s, content)
        return {"slug": s, "ok": ok, "out": "", "err": "" if ok else "brainstore put failed"}

    def link(self, from_slug: str, to_slug: str, link_type: str) -> bool:
        return brainstore._link(from_slug, to_slug, link_type)


# --- atom dataclasses (the penrose_pm schema) --------------------------------

@dataclass
class Claim:
    claim_id: str
    statement: str
    mechanism: str
    scope: str
    horizon: str
    source_id: str
    source_span: str            # verbatim quote from the paper (P2 anti-hallucination)
    claimed_metric_quote: str
    applicable_strategy_class: str = config.STRATEGY_CLASS_VOL
    source_type: str = "external_source"   # external_source | generated_hypothesis | chat
    search_cohort_id: Optional[str] = None
    search_denominator: Optional[int] = None
    raw_hypothesis_id: Optional[str] = None
    data_provenance: dict = field(default_factory=dict)
    declared_regime: Optional[dict] = None

    def __post_init__(self) -> None:
        self.source_type = validate_source_type(self.source_type)
        if self.declared_regime is not None:
            if not isinstance(self.declared_regime, dict):
                raise ValueError("declared_regime must be a mapping with scheme and label")
            scheme = self.declared_regime.get("scheme")
            label = self.declared_regime.get("label")
            if not isinstance(scheme, str) or not scheme.strip():
                raise ValueError("declared_regime.scheme must be a non-empty string")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("declared_regime.label must be a non-empty string")
            self.declared_regime = {"scheme": scheme.strip().lower(),
                                    "label": label.strip().lower()}


@dataclass
class Decision:
    decision_id: str
    claim_id: str
    verdict: str                # kill | watch | research-supported | insufficient_data
    kill_reason: Optional[str]  # S6 enum value
    rationale: str
    metrics: dict = field(default_factory=dict)
    revisit_at: Optional[str] = None
    verified_by_human: bool = False


@dataclass
class Principle:
    principle_id: str
    statement: str
    supporting_kills: list[str]
    applicable_strategy_classes: list[str]
    n_observations: int
    confidence: float
    status: str = "proposed"


KILL_REASONS = [
    "unfalsifiable", "fee_curve", "dedup", "no_oos_edge", "in_sample_only",
    "negative_dsr", "capacity_too_small", "data_unavailable", "execution_infeasible",
    # future: emerge from regime-conditional analysis (P7), not pre-baked detection.
    # "regime_dependent", "loss_clustering", — add when regime splits are live and
    # a real pattern is observed, then extract as a principle.
]
