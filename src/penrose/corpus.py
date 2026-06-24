"""Deterministic, provenance-linked abstraction corpus (advisory; inform-never-gate)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .brain import source_is_unanchored


def _anchored_safe(source_type) -> bool:
    """True only for a recognized, anchored source_type. Unknown/malformed -> False (excluded):
    fail-soft (never raises during a corpus rebuild) and fail-closed (firewall-safe). Mirrors the
    defensive posture in retrieval._eligible."""
    try:
        return not source_is_unanchored(str(source_type or "external_source"))
    except Exception:  # noqa: BLE001
        return False


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _domain(c: dict) -> str:
    p = c.get("data_provenance") or {}
    return str(p.get("data_domain") or p.get("domain") or "general")


def _family(c: dict) -> str:
    p = c.get("data_provenance") or {}
    return str(p.get("strategy_family") or p.get("family") or _domain(c))


def _direction(c: dict) -> str:
    direction = str(c.get("evidence_direction", "unknown")).lower()
    return direction if direction in {"positive", "negative"} else "unknown"


def _id(prefix: str, parts) -> str:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
    return f"{prefix}-{h}"


def _claim_key(c: dict) -> str:
    text = str(c.get("reusable_principle") or c.get("statement") or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _confidence(rows: list[dict], now_year: int) -> float:
    vals = []
    for row in rows:
        try:
            year = int(str(row.get("created_at", ""))[:4])
        except Exception:
            year = now_year
        vals.append(0.5 ** (max(0, now_year - year) / config.CORPUS_HALF_LIFE_YEARS))
    return round(sum(vals) / max(1, len(vals)), 4)


def _footprint(rows: list[dict]) -> dict:
    domains, datasets, periods = set(), set(), []
    for row in rows:
        p = row.get("data_provenance") or {}
        row_domains = p.get("data_domains") or (
            [p.get("data_domain")] if p.get("data_domain") else [])
        domains.update(map(str, row_domains))
        datasets.update(map(str, p.get("datasets") or []))
        periods.extend(x for x in (p.get("periods") or []) if isinstance(x, dict))
    periods = sorted({(str(x.get("start", "")), str(x.get("end", ""))) for x in periods})
    return {"data_domains": sorted(domains), "datasets": sorted(datasets),
            "periods": [{"start": a, "end": b} for a, b in periods if a and b]}


def build(concepts: list[dict], *, min_support: int | None = None,
          current_year: int | None = None) -> dict:
    min_support = int(min_support or config.CORPUS_MIN_SUPPORT)
    current_year = current_year or datetime.now(timezone.utc).year
    nodes, edges = [], []
    for c in sorted(concepts, key=lambda x: str(x.get("concept_id"))):
        nodes.append({**c, "node_id": c.get("concept_id"), "level": "observation"})
    recurring = defaultdict(list)
    for c in concepts:
        if _direction(c) == "unknown":
            continue
        recurring[(_claim_key(c), _direction(c))].append(c)
    specific_claims = []
    for (key, direction), rows in sorted(recurring.items()):
        rows = sorted(rows, key=lambda x: str(x.get("concept_id")))
        if not key or len(rows) < 2:
            continue
        node = {"node_id": _id("specific", [key, direction]),
                "level": "specific_claim", "direction": direction,
                "statement": str(rows[0].get("reusable_principle")
                                 or rows[0].get("statement", "")),
                "support_count": len(rows),
                "provenance": [r.get("concept_id") for r in rows],
                "caveat": "Recurring observation, not an independent verdict."}
        specific_claims.append(node); nodes.append(node)
        edges += [{"from": r.get("concept_id"), "to": node["node_id"], "type": "recurs_as"}
                  for r in rows]
    grouped = defaultdict(list)
    for c in concepts:
        if _direction(c) == "unknown":
            continue
        grouped[(_family(c), _direction(c))].append(c)
    families = []
    for (family, direction), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: str(x.get("concept_id")))
        if len(rows) < min_support:
            continue
        node = {"node_id": _id("family", [family, direction]),
                "level": "family_principle", "family": family, "direction": direction,
                "statement": f"{direction.title()} evidence recurs in {family} under recorded boundaries.",
                "support_count": len(rows), "confidence": _confidence(rows, current_year),
                "provenance": [r.get("concept_id") for r in rows],
                "data_provenance": _footprint(rows),
                "caveat": "Advisory prior only; every new claim is tested independently."}
        families.append(node); nodes.append(node)
        edges += [{"from": r.get("concept_id"), "to": node["node_id"], "type": "supports"}
                  for r in rows]
    mechanisms = []
    by_direction = defaultdict(list)
    for node in families:
        by_direction[node["direction"]].append(node)
    for direction, rows in sorted(by_direction.items()):
        domains = {next((_domain(c) for c in concepts if c.get("concept_id") in r["provenance"]),
                        r["family"]) for r in rows}
        if len(domains) < 2:
            continue
        node = {"node_id": _id("cross-family", [direction, sorted(domains)]),
                "level": "cross_family_mechanism", "direction": direction,
                "statement": f"A {direction} mechanism recurs across distinct data families.",
                "families": sorted(r["family"] for r in rows), "data_domains": sorted(domains),
                "support_count": sum(r["support_count"] for r in rows),
                "provenance": [r["node_id"] for r in rows],
                "data_provenance": _footprint(rows),
                "caveat": "Candidate synthesis input, not a verdict."}
        mechanisms.append(node); nodes.append(node)
        edges += [{"from": r["node_id"], "to": node["node_id"], "type": "abstracts_to"}
                  for r in rows]
    generated_at = max((str(c.get("created_at", "")) for c in concepts),
                       default="1970-01-01T00:00:00+00:00")
    return {"schema_version": 1, "generated_at": generated_at,
            "levels": {"observation": len(concepts), "specific_claim": len(specific_claims),
                       "family_principle": len(families),
                       "cross_family_mechanism": len(mechanisms)},
            "nodes": nodes, "edges": edges}


def reserve_intersects(provenance: dict, reserve: dict | None = None) -> bool:
    reserve = reserve or config.CONFIRMATION_RESERVE
    epochs = reserve.get("epochs") or [reserve]
    for epoch in epochs:
        for key in ("data_domains", "datasets"):
            left = provenance.get(key) or ([provenance.get(key[:-1])] if provenance.get(key[:-1]) else [])
            if set(map(str, left)) & set(map(str, epoch.get(key, []))):
                return True
        for left in provenance.get("periods", []) or []:
            for right in epoch.get("periods", []) or [epoch]:
                ls, le = str(left.get("start", "")), str(left.get("end", ""))
                rs, re = str(right.get("start", "")), str(right.get("end", ""))
                if ls and le and rs and re and max(ls, rs) <= min(le, re):
                    return True
    return False


def configured_reserve_epochs(reserve: dict | None = None) -> list[dict]:
    reserve = reserve or config.CONFIRMATION_RESERVE
    out = []
    for i, epoch in enumerate(reserve.get("epochs") or []):
        start, end = str(epoch.get("start", "")), str(epoch.get("end", ""))
        epoch_id = str(epoch.get("epoch_id", "")).strip()
        if epoch_id and start and end and start <= end:
            out.append({**epoch, "epoch_id": epoch_id, "start": start, "end": end})
    if len({x["epoch_id"] for x in out}) != len(out):
        return []
    ordered = sorted(out, key=lambda x: (x["start"], x["end"], x["epoch_id"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] <= left["end"]:
            return []
    return out


def provenance_checkable(provenance: dict, reserve: dict | None = None) -> bool:
    """True when lineage contains enough dimensions to prove reserve independence."""
    epochs = configured_reserve_epochs(reserve)
    if not epochs or not provenance.get("periods"):
        return False
    if any(epoch.get("datasets") for epoch in epochs) and not provenance.get("datasets"):
        return False
    if any(epoch.get("data_domains") for epoch in epochs):
        domains = provenance.get("data_domains") or (
            [provenance.get("data_domain")] if provenance.get("data_domain") else [])
        if not domains:
            return False
    return True


def build_files(concepts_path: Path | None = None) -> dict:
    if not configured_reserve_epochs():
        raise RuntimeError(
            "confirmation reserve is unconfigured: define distinct epochs before synthesis")
    concepts = [
        c for c in _read(Path(concepts_path or config.CONCEPTS))
        if provenance_checkable(c.get("data_provenance") or {})
        and not reserve_intersects(c.get("data_provenance") or {})
        and _anchored_safe(c.get("source_type"))
    ]
    graph = build(concepts)
    config.CORPUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_JSON.write_text(json.dumps(graph, indent=2, default=str))
    config.CORPUS_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_GRAPH.write_text("".join(json.dumps(e) + "\n" for e in graph["edges"]))
    return graph
