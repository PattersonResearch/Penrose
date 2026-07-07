"""Deterministic calibration for planted-principle hypothesis creation recovery."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from penrose.corpus import build  # noqa: E402
from penrose.synthesize import normalize  # noqa: E402

PLANTED_PRINCIPLE = (
    "positive evidence recurs when signal strength is conditioned on a lagged "
    "liquidity stress boundary"
)


def _buildable_spec(series: str = "stress_signal") -> dict:
    return {
        "signal": f"zscore({series}, window) - ma({series}, slow_window)",
        "series": [series],
        "params": {"window": 20, "slow_window": 60},
        "param_grid": {"window": [10, 20, 40], "slow_window": [40, 60, 120]},
        "conditioning": "lag(liquidity_stress, 1) > threshold",
        "entry_exit": "enter when signal > 1; exit after horizon or signal <= 0",
        "horizon": "1d",
    }


def planted_concepts() -> list[dict]:
    rows = []
    for family, domain, dataset in [
        ("term_carry", "rates", "rates-discovery"),
        ("volatility_breakout", "crypto_vol", "crypto-vol-discovery"),
    ]:
        for i in range(3):
            rows.append({
                "concept_id": f"plant-{family}-{i}",
                "source_claim_id": f"kill-{family}-{i}",
                "statement": f"{family} killed experiment {i} exposed the planted principle",
                "reusable_principle": PLANTED_PRINCIPLE,
                "surviving_explanation": "",
                "evidence_direction": "positive",
                "created_at": "2026-01-01T00:00:00+00:00",
                "source_type": "external_source",
                "data_provenance": {
                    "strategy_family": family,
                    "data_domain": domain,
                    "data_domains": [domain],
                    "datasets": [dataset],
                    "periods": [{"start": "2020-01-01", "end": "2022-12-31"}],
                },
            })
    return rows


def noise_concepts() -> list[dict]:
    rows = []
    for i in range(6):
        rows.append({
            "concept_id": f"noise-{i}",
            "source_claim_id": f"noise-kill-{i}",
            "statement": f"unrelated noisy killed observation {i}",
            "reusable_principle": f"unrelated one-off noise principle {i}",
            "surviving_explanation": "",
            "evidence_direction": "positive",
            "created_at": "2026-01-01T00:00:00+00:00",
            "source_type": "external_source",
            "data_provenance": {
                "strategy_family": f"noise-family-{i}",
                "data_domain": f"noise-domain-{i % 2}",
                "data_domains": [f"noise-domain-{i % 2}"],
                "datasets": [f"noise-dataset-{i}"],
                "periods": [{"start": "2020-01-01", "end": "2022-12-31"}],
            },
        })
    return rows


def cross_family_nodes(graph: dict) -> list[dict]:
    return [n for n in graph["nodes"] if n.get("level") == "cross_family_mechanism"]


def traced_concept_ids(graph: dict, mechanism: dict) -> set[str]:
    family_nodes = {
        n["node_id"]: n for n in graph["nodes"] if n.get("level") == "family_principle"
    }
    out: set[str] = set()
    for family_node_id in mechanism["provenance"]:
        out.update(family_nodes[family_node_id]["provenance"])
    return out


def recovery_check() -> tuple[bool, dict, dict]:
    concepts = planted_concepts()
    graph = build(concepts, min_support=3, current_year=2026)
    mechanisms = cross_family_nodes(graph)
    if len(mechanisms) != 1:
        return False, graph, {}
    mechanism = mechanisms[0]
    ok = (
        mechanism["direction"] == "positive"
        and set(mechanism["families"]) == {"term_carry", "volatility_breakout"}
        and set(mechanism["data_domains"]) == {"rates", "crypto_vol"}
        and traced_concept_ids(graph, mechanism) == {c["concept_id"] for c in concepts}
    )
    return ok, graph, mechanism


def noise_check() -> bool:
    graph = build(noise_concepts(), min_support=3, current_year=2026)
    return cross_family_nodes(graph) == []


def grounding_check(graph: dict, mechanism: dict) -> bool:
    raw = [{
        "statement": "Grounded planted candidate hypothesis",
        "mechanism": "planted cross-family mechanism",
        "scope": "rates and crypto volatility",
        "horizon": "1d",
        "strategy_class": "stress-conditioned-signal",
        "candidate_class": "testable_now",
        "inspired_by": [mechanism["node_id"]],
        "falsifier": "no OOS lift after conditioning",
        "spec": _buildable_spec(),
    }, {
        "statement": "Hallucinated planted candidate hypothesis",
        "mechanism": "fake mechanism",
        "scope": "synthetic",
        "horizon": "1d",
        "strategy_class": "fake",
        "candidate_class": "testable_now",
        "inspired_by": ["cross-family-does-not-exist"],
        "falsifier": "not applicable",
    }, {
        "statement": "Grounded planted candidate hypothesis",
        "mechanism": "duplicate statement",
        "scope": "rates and crypto volatility",
        "horizon": "1d",
        "strategy_class": "stress-conditioned-signal",
        "candidate_class": "testable_now",
        "inspired_by": [mechanism["node_id"]],
        "falsifier": "no OOS lift after conditioning",
        "spec": _buildable_spec(),
    }]
    _claims, normalized = normalize("calib-hypothesis-recovery", raw, {"nodes": graph["nodes"]})
    return (
        normalized[0]["grounded"] is True
        and normalized[0]["admitted"] is True
        and normalized[1]["grounded"] is False
        and normalized[1]["admitted"] is False
        and normalized[2]["duplicate_in_run"] is True
        and normalized[2]["admitted"] is False
    )


def deterministic_check() -> bool:
    first = build(planted_concepts(), min_support=3, current_year=2026)
    second = build(list(reversed(planted_concepts())), min_support=3, current_year=2026)
    sig = lambda g: sorted((n["node_id"], n["level"]) for n in g["nodes"])
    return sig(first) == sig(second)


def main() -> None:
    recovered, graph, mechanism = recovery_check()
    noise_silent = noise_check()
    grounding = recovered and grounding_check(graph, mechanism)
    deterministic = deterministic_check()

    rows = [
        ("planted cross-family principle recovered", recovered),
        ("noise corpus yields zero cross-family nodes", noise_silent),
        ("grounding firewall rejects ungrounded candidate", grounding),
        ("recovery deterministic", deterministic),
    ]
    for name, ok in rows:
        print(f"{name}: {'OK' if ok else 'FAIL'}")
    if recovered:
        print(f"recovered node: {mechanism['node_id']}")
        print(f"provenance: {mechanism['provenance']}")
    raise SystemExit(0 if all(ok for _, ok in rows) else 1)


if __name__ == "__main__":
    main()
