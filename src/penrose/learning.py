"""Advisory cross-run learning surfaces.

No function here feeds verdict logic or writes approved brain state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import brain_connect, config
from .brain_connect import Record
from .strategy_family import normalize_strategy_family


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except Exception:  # noqa: BLE001 - file-level read failure (permissions/encoding) fails open
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # CR2-1: skip a SINGLE corrupt/non-dict line rather than discarding the whole corpus. One bad
        # line (partial write, concurrent-writer race, manual edit) must not make a full corpus read
        # as empty — which downstream would misread as "no supporting kills" and purge good proposals.
        try:
            value = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _analysis_by_claim(path: Path) -> dict[str, dict]:
    rows = _read_jsonl(path)
    out: dict[str, dict] = {}
    for row in rows:
        claim_id = str(row.get("claim_id") or "")
        if claim_id:
            out[claim_id] = row
    return out


def _record_from_decision(row: dict, supplement: dict | None = None) -> Record:
    sup = supplement or {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if not metrics and isinstance(sup.get("metrics"), dict):
        metrics = sup.get("metrics") or {}
    statement = str(
        row.get("statement")
        or row.get("claim_statement")
        or sup.get("statement")
        or row.get("claim_id")
        or ""
    )
    text = " ".join(str(x or "") for x in (
        statement,
        row.get("source_title"),
        sup.get("source_title"),
        row.get("domain"),
        sup.get("domain"),
    ))
    data_prov = sup.get("data_provenance") if isinstance(sup.get("data_provenance"), dict) else {}
    strategy_family = (
        normalize_strategy_family(row.get("strategy_family"))
        or normalize_strategy_family(data_prov.get("strategy_family_structured"))
        or normalize_strategy_family(sup.get("strategy_family"))
    )
    domain = str(
        row.get("domain")
        or sup.get("domain")
        or data_prov.get("data_domain")
        or brain_connect.infer_domain(text)
    )
    verdict = str(row.get("verdict") or "")
    kill_reason = row.get("kill_reason")
    claim_id = str(row.get("claim_id") or "")
    return Record(
        id=str(row.get("decision_id") or row.get("claim_id") or ""),
        domain=domain,
        verdict=verdict,
        kill_reason=str(kill_reason) if kill_reason is not None else None,
        statement=statement[:240],
        structural=(verdict == "kill" and kill_reason in brain_connect.STRUCTURAL_KILLS),
        power_sufficient=metrics.get("power_sufficient"),
        date=str(row.get("logged_at") or row.get("run_at") or sup.get("run_at") or ""),
        synthetic=bool(row.get("synthetic") or sup.get("synthetic")),
        claim_id=claim_id or str(row.get("decision_id") or ""),
        strategy_family=strategy_family,
    )


def load_decision_records(
    decisions_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
) -> list[Record]:
    """Load the already-recorded decisions corpus as advisory Records.

    Missing or corrupt inputs fail open as ``[]``. This reads only historical
    ledgers and has no path to the confirmation reserve or verdict pipeline.
    """
    dpath = Path(decisions_path) if decisions_path is not None else config.DECISIONS_LOG
    apath = Path(analysis_path) if analysis_path is not None else config.ANALYSIS_INDEX
    decisions = _read_jsonl(dpath)
    if not decisions:
        return []
    supplements = _analysis_by_claim(apath)
    records = []
    for row in decisions:
        claim_id = str(row.get("claim_id") or "")
        records.append(_record_from_decision(row, supplements.get(claim_id)))
    return records


def _proposal_from_principle(row: dict) -> dict:
    out = dict(row)
    if "supporting_kills" not in out:
        out["supporting_kills"] = list(out.get("supporting") or [])
    if "supporting_kill_count" not in out:
        out["supporting_kill_count"] = int(out.get("n_observations") or len(out["supporting_kills"]))
    if "example_claim_ids" not in out:
        out["example_claim_ids"] = list(out["supporting_kills"][:5])
    out["source"] = "distilled"
    out["status"] = "proposed"
    return out


def _principle_min_kills() -> int:
    try:
        return max(3, int(config.PRINCIPLE_MIN_KILLS))
    except (TypeError, ValueError):
        return 3


def distill_principles(
    records: Iterable[Record] | None = None,
    *,
    decisions_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
    current_year: int = 2026,
) -> list[dict]:
    """Distill cross-run advisory principle proposals from the full corpus.

    This reuses ``brain_connect.propose_principles`` over all recorded decisions,
    not the conservative same-run ``stages.propose_principle`` rule. It returns
    proposed rows only and never writes approved principle storage. Human P9
    approval is still required before any principle can enter ``principles.jsonl``
    or the trusted BrainStore.
    """
    recs = list(records) if records is not None else load_decision_records(decisions_path, analysis_path)
    if not recs:
        return []
    return [
        _proposal_from_principle(p)
        for p in brain_connect.propose_principles(
            recs,
            current_year=current_year,
            min_kills=_principle_min_kills(),
            half_life_years=config.CORPUS_HALF_LIFE_YEARS,
        )
    ]


def persist_distilled_proposals(decisions_path: str | Path | None = None) -> dict:
    """Distill cross-run proposals and PERSIST them to the propose-only store (status: proposed).

    P9-safe: never writes the approved brain / principles.jsonl. Shared by the ``penrose distill`` CLI and
    the MCP ``penrose_mine_principles`` tool so the CR2-1 guard lives in ONE place: the replace_source
    purge is gated on corpus readability — if the decisions file is non-empty but yields zero records
    (a transient read failure, not a genuine empty corpus), the store is PRESERVED, not purged.
    """
    from .proposals import write_proposals, read_proposals
    recs = load_decision_records(decisions_path)
    rows = distill_principles(recs)
    dpath = Path(decisions_path) if decisions_path is not None else getattr(config, "DECISIONS_LOG", None)
    corpus_unreadable = bool(
        dpath is not None and dpath.exists() and dpath.stat().st_size > 0 and not recs)
    if corpus_unreadable:
        stored = read_proposals()
        return {"distilled": rows, "stored": len(stored), "status": "proposed",
                "warning": "decisions corpus present but unreadable; distilled proposals preserved (not purged)"}
    stored = write_proposals(rows, source="distilled", replace_source=True)
    return {"distilled": rows, "stored": len(stored), "status": "proposed"}


def distill_contrastive_principles(
    records: Iterable[Record] | None = None,
    *,
    decisions_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
    current_year: int = 2026,
) -> list[dict]:
    """Distill survivor-vs-kill boundary principle proposals from the corpus.

    Proposed rows only; this never writes approved principle storage.
    """
    recs = list(records) if records is not None else load_decision_records(decisions_path, analysis_path)
    if not recs:
        return []
    out = []
    for row in brain_connect.propose_contrastive_principles(recs, current_year=current_year):
        proposal = dict(row)
        proposal["source"] = "distilled-contrastive"
        proposal["status"] = "proposed"
        out.append(proposal)
    return out
