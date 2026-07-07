"""Planted-principle recovery tests for candidate hypothesis creation.

These tests make the honest claim only: the deterministic abstraction
(`penrose.corpus.build`) can recover a planted cross-family relationship from
past killed concepts and `penrose.synthesize.normalize` admits only candidates
grounded in real corpus node ids. They do not claim profitability; every
candidate hypothesis still faces the Referee, confirmation firewall, and P9.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PLANTED_PRINCIPLE = (
    "positive evidence recurs when signal strength is conditioned on a lagged "
    "liquidity stress boundary"
)


def buildable_spec(series: str = "stress_signal") -> dict:
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
    specs = [
        ("term_carry", "rates", "rates-discovery"),
        ("volatility_breakout", "crypto_vol", "crypto-vol-discovery"),
    ]
    for family, domain, dataset in specs:
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


def planted_recovery() -> tuple[dict, dict, list[dict]]:
    from penrose.corpus import build

    concepts = planted_concepts()
    graph = build(concepts, min_support=3, current_year=2026)
    mechanisms = cross_family_nodes(graph)
    assert len(mechanisms) == 1
    return graph, mechanisms[0], concepts


def traced_planted_concept_ids(graph: dict, mechanism: dict) -> set[str]:
    family_nodes = {
        n["node_id"]: n for n in graph["nodes"] if n.get("level") == "family_principle"
    }
    concept_ids: set[str] = set()
    for family_node_id in mechanism["provenance"]:
        concept_ids.update(family_nodes[family_node_id]["provenance"])
    return concept_ids


def test_recovers_planted_cross_family_principle():
    graph, mechanism, concepts = planted_recovery()

    assert mechanism["direction"] == "positive"
    assert set(mechanism["families"]) == {"term_carry", "volatility_breakout"}
    assert set(mechanism["data_domains"]) == {"rates", "crypto_vol"}
    assert mechanism["support_count"] == 6

    traced_ids = traced_planted_concept_ids(graph, mechanism)
    planted_ids = {c["concept_id"] for c in concepts}
    assert traced_ids == planted_ids
    assert {
        c["data_provenance"]["strategy_family"] for c in concepts
        if c["concept_id"] in traced_ids
    } == {"term_carry", "volatility_breakout"}
    assert {
        c["reusable_principle"] for c in concepts if c["concept_id"] in traced_ids
    } == {PLANTED_PRINCIPLE}


def test_noise_corpus_yields_no_cross_family_node():
    from penrose.corpus import build

    graph = build(noise_concepts(), min_support=3, current_year=2026)
    assert cross_family_nodes(graph) == []


def test_recovery_is_deterministic():
    from penrose.corpus import build

    first = build(planted_concepts(), min_support=3, current_year=2026)
    second = build(list(reversed(planted_concepts())), min_support=3, current_year=2026)

    def signature(graph: dict) -> list[tuple[str, str]]:
        return sorted((n["node_id"], n["level"]) for n in graph["nodes"])

    assert signature(first) == signature(second)


def test_normalize_admits_grounded_candidate():
    from penrose.synthesize import normalize

    graph, mechanism, _ = planted_recovery()
    _claims, normalized = normalize("hypothesis-recovery", [{
        "statement": "Liquidity stress conditioning improves the next testable signal family",
        "mechanism": "planted cross-family mechanism",
        "scope": "rates and crypto volatility",
        "horizon": "1d",
        "strategy_class": "stress-conditioned-signal",
        "candidate_class": "testable_now",
        "inspired_by": [mechanism["node_id"]],
        "falsifier": "no OOS lift after conditioning",
        "spec": buildable_spec(),
    }], {"nodes": graph["nodes"]})

    assert normalized[0]["grounded"] is True
    assert normalized[0]["admitted"] is True
    assert normalized[0]["data_provenance"]["corpus_nodes"] == [mechanism["node_id"]]


def test_normalize_rejects_ungrounded_candidate():
    from penrose.synthesize import normalize

    graph, _mechanism, _ = planted_recovery()
    _claims, normalized = normalize("hypothesis-recovery", [{
        "statement": "A hallucinated node should not ground a candidate hypothesis",
        "mechanism": "fake mechanism",
        "scope": "synthetic",
        "horizon": "1d",
        "strategy_class": "fake",
        "candidate_class": "testable_now",
        "inspired_by": ["cross-family-does-not-exist"],
        "falsifier": "not applicable",
    }], {"nodes": graph["nodes"]})

    assert normalized[0]["grounded"] is False
    assert normalized[0]["admitted"] is False
    assert normalized[0]["data_provenance"]["corpus_nodes"] == []


def test_normalize_rejects_duplicate():
    from penrose.synthesize import normalize

    graph, mechanism, _ = planted_recovery()
    raw = [{
        "statement": "Duplicate candidate statement",
        "mechanism": "planted mechanism",
        "scope": "rates and crypto volatility",
        "horizon": "1d",
        "strategy_class": "stress-conditioned-signal",
        "candidate_class": "testable_now",
        "inspired_by": [mechanism["node_id"]],
        "falsifier": "no OOS lift",
        "spec": buildable_spec(),
    }, {
        "statement": "Duplicate candidate statement",
        "mechanism": "same planted mechanism",
        "scope": "rates and crypto volatility",
        "horizon": "1d",
        "strategy_class": "stress-conditioned-signal",
        "candidate_class": "testable_now",
        "inspired_by": [mechanism["node_id"]],
        "falsifier": "no OOS lift",
        "spec": buildable_spec(),
    }]
    _claims, normalized = normalize("hypothesis-recovery", raw, {"nodes": graph["nodes"]})

    assert normalized[0]["admitted"] is True
    assert normalized[1]["grounded"] is True
    assert normalized[1]["duplicate_in_run"] is True
    assert normalized[1]["admitted"] is False


def test_llm_synthesizer_grounds_in_planted_principle(tmp_path, monkeypatch):
    if not os.environ.get("PENROSE_LLM_API_KEY"):
        pytest.skip("PENROSE_LLM_API_KEY not configured; deterministic recovery is covered above")

    from penrose import config
    from penrose.synthesize import run_synthesis

    concepts_path = tmp_path / "concepts.jsonl"
    concepts_path.write_text("".join(json.dumps(c) + "\n" for c in planted_concepts()))
    monkeypatch.setattr(config, "CONCEPTS", concepts_path)
    monkeypatch.setattr(config, "CORPUS_JSON", tmp_path / "corpus.json")
    monkeypatch.setattr(config, "CORPUS_GRAPH", tmp_path / "corpus_graph.jsonl")
    monkeypatch.setattr(config, "SYNTHESIS_ARCHIVES", tmp_path / "syntheses")
    monkeypatch.setattr(config, "SYNTHESIS_RUNS", tmp_path / "synthesis_runs.jsonl")
    # PEN-17: this test intentionally exercises the end-to-end synthesis entry point.
    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)
    monkeypatch.setattr(config, "CONFIRMATION_RESERVE", {
        "reserve_id": "hypothesis-recovery-reserve",
        "epochs": [{
            "epoch_id": "reserve-2024",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "data_domains": ["reserved-holdout"],
            "datasets": ["reserved-holdout"],
        }],
    })

    expected_graph, mechanism, _ = planted_recovery()
    manifest = run_synthesis(n=3, generate_only=True, run_id="hypothesis-llm-recovery")
    normalized_path = Path(manifest["artifact_dir"]) / "candidates.normalized.jsonl"
    normalized = [json.loads(line) for line in normalized_path.read_text().splitlines()]

    assert any(
        row["grounded"] and mechanism["node_id"] in row["data_provenance"]["corpus_nodes"]
        for row in normalized
    ), {
        "expected_node": mechanism["node_id"],
        "available_nodes": [
            n["node_id"] for n in expected_graph["nodes"]
            if n.get("level") == "cross_family_mechanism"
        ],
        "normalized": normalized,
    }
