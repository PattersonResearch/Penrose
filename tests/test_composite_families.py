from __future__ import annotations

import json
from pathlib import Path

from penrose.brain import Claim
from penrose.brain_connect import Record, propose_principles
from penrose.pipeline import spec_gen
from penrose.pipeline.p1_ingest import IngestedSource


def _source(text: str = "") -> IngestedSource:
    return IngestedSource(
        source_id="unit",
        title="unit",
        text=text,
        n_pages=1,
        n_chars=len(text),
        text_sha256="x",
        injection_flags=[],
    )


def _claim(statement: str, strategy_class: str, claim_id: str = "c1") -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=statement,
        mechanism="",
        scope="crypto",
        horizon="1d",
        source_id="unit",
        source_span=statement,
        claimed_metric_quote="",
        applicable_strategy_class=strategy_class,
    )


def _record(
    rid: str,
    family: dict | None,
    *,
    reason: str = "regime_fragile",
    domain: str = "funding-carry",
) -> Record:
    return Record(
        id=rid,
        claim_id=rid,
        domain=domain,
        verdict="kill",
        kill_reason=reason,
        statement="fixture claim",
        structural=True,
        power_sufficient=True,
        date="2026-01-01T00:00:00Z",
        strategy_family=family,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_spec_gen_declares_structured_composite_and_simple_family(tmp_path, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "MODULES", Path(tmp_path) / "modules")

    composite = spec_gen.generate_spec(
        _claim(
            "A carry and trend regime blend predicts next-day crypto returns.",
            "carry_trend_regime_blend",
            claim_id="composite",
        ),
        _source(),
        use_llm=False,
    )
    simple = spec_gen.generate_spec(
        _claim(
            "Momentum predicts next-day crypto returns.",
            "momentum",
            claim_id="simple",
        ),
        _source(),
        use_llm=False,
    )

    assert composite["strategy_family"] == {
        "components": ["carry", "trend"],
        "method": "regime_blend",
    }
    assert simple["strategy_family"] == {
        "components": ["momentum"],
        "method": "single",
    }


def test_hierarchical_clustering_emits_exact_composite_when_supported():
    family = {"components": ["trend", "carry"], "method": "regime_blend"}
    records = [_record(f"ct-{i}", family) for i in range(3)]

    principles = propose_principles(records, min_kills=3)

    assert len(principles) == 1
    principle = principles[0]
    assert principle["family_level"] == "exact"
    assert principle["strategy_family"] == {
        "components": ["carry", "trend"],
        "method": "regime_blend",
    }
    assert principle["supporting_kill_count"] == 3
    assert "carry+trend" in principle["statement"]
    assert "regime_blends" in principle["statement"]


def test_sparse_exact_composites_roll_up_to_method_without_generic_hybrid_bucket():
    carry_trend = {"components": ["carry", "trend"], "method": "regime_blend"}
    momentum_value = {"components": ["momentum", "value"], "method": "regime_blend"}
    records = (
        [_record(f"ct-{i}", carry_trend) for i in range(2)]
        + [_record(f"mv-{i}", momentum_value) for i in range(2)]
    )

    principles = propose_principles(records, min_kills=3)

    assert len(principles) == 1
    principle = principles[0]
    assert principle["family_level"] == "method"
    assert principle["domain"] == "regime_blend"
    assert principle["supporting_kill_count"] == 4
    assert "regime_blends" in principle["statement"]
    assert "hybrid" not in principle["statement"].lower()


def test_each_kill_supports_exactly_one_emitted_principle():
    carry_trend = {"components": ["carry", "trend"], "method": "regime_blend"}
    momentum_value = {"components": ["momentum", "value"], "method": "regime_blend"}
    records = (
        [_record(f"ct-{i}", carry_trend) for i in range(3)]
        + [_record(f"mv-{i}", momentum_value) for i in range(3)]
    )

    principles = propose_principles(records, min_kills=3)

    assert {p["family_level"] for p in principles} == {"exact"}
    supporting = [kill for p in principles for kill in p["supporting"]]
    assert len(supporting) == 6
    assert len(set(supporting)) == 6
    assert sum(p["supporting_kill_count"] for p in principles) == 6


def test_missing_or_unparseable_strategy_family_falls_back_to_domain_behavior():
    records = [
        _record("legacy-0", None, reason="in_sample_only", domain="funding-carry"),
        _record("legacy-1", {"components": "carry", "method": "single"},
                reason="in_sample_only", domain="funding-carry"),
        _record("legacy-2", None, reason="in_sample_only", domain="funding-carry"),
    ]

    principles = propose_principles(records, min_kills=3)

    assert len(principles) == 1
    principle = principles[0]
    assert principle["family_level"] == "domain"
    assert principle["domain"] == "funding-carry"
    assert principle["principle_id"] == "principle-funding-carry-in_sample_only"


def test_structured_family_distillation_stays_propose_only(tmp_path, monkeypatch):
    from penrose import config
    from penrose.learning import distill_principles

    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(config, "PRINCIPLE_MIN_KILLS", 3)

    family = {"components": ["carry", "trend"], "method": "regime_blend"}
    _write_jsonl(decisions, [{
        "decision_id": f"d{i}",
        "claim_id": f"c{i}",
        "statement": "carry trend regime blend",
        "verdict": "kill",
        "kill_reason": "regime_fragile",
        "metrics": {"power_sufficient": True},
        "logged_at": "2026-01-01T00:00:00Z",
        "strategy_family": family,
    } for i in range(3)])

    proposals = distill_principles()

    assert len(proposals) == 1
    assert proposals[0]["status"] == "proposed"
    assert proposals[0]["source"] == "distilled"
    assert proposals[0]["family_level"] == "exact"
