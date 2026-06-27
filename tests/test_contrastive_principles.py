from __future__ import annotations

from penrose.brain_connect import Record, propose_contrastive_principles


def _record(
    rid: str,
    domain: str,
    verdict: str,
    kill_reason: str | None = None,
    statement: str = "fixture claim",
) -> Record:
    return Record(
        id=rid,
        domain=domain,
        verdict=verdict,
        kill_reason=kill_reason,
        statement=statement,
        structural=(verdict == "kill" and kill_reason is not None),
        power_sufficient=True,
        date="2026-01-01T00:00:00Z",
    )


def test_carver_shaped_corpus_yields_trend_vs_carry_contrastive_principle():
    records = [
        _record(f"trend-kill-{i}", "trend-following", "kill", "regime_fragile")
        for i in range(3)
    ] + [
        _record(f"carry-watch-{i}", "funding-carry", "watch")
        for i in range(2)
    ]

    out = propose_contrastive_principles(records)

    assert len(out) == 1
    principle = out[0]
    assert principle["kind"] == "contrastive"
    assert principle["kill_domain"] == "trend-following"
    assert principle["kill_reason"] == "regime_fragile"
    assert "funding-carry" in principle["survivor_domains"]
    assert principle["survivor_domains"]["funding-carry"] == ["carry-watch-0", "carry-watch-1"]


def test_contrastive_principles_require_survivors_in_other_domains():
    records = [
        _record(f"trend-kill-{i}", "trend-following", "kill", "regime_fragile")
        for i in range(3)
    ]

    assert propose_contrastive_principles(records) == []


def test_contrastive_principles_require_min_kills():
    records = [
        _record(f"trend-kill-{i}", "trend-following", "kill", "regime_fragile")
        for i in range(2)
    ] + [
        _record(f"carry-watch-{i}", "funding-carry", "watch")
        for i in range(2)
    ]

    assert propose_contrastive_principles(records) == []


def test_views_principles_composes_recurrence_then_contrastive(monkeypatch):
    from penrose import learning, views

    recurrence = {"principle_id": "principle-funding-carry-in_sample_only"}
    contrastive = {"principle_id": "contrast-trend-following-regime_fragile", "kind": "contrastive"}
    monkeypatch.setattr(learning, "distill_principles", lambda: [recurrence])
    monkeypatch.setattr(learning, "distill_contrastive_principles", lambda: [contrastive])

    assert views.principles(limit=10) == [recurrence, contrastive]
    assert views.principles(limit=1) == [recurrence]
