from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _kill(i: int, *, run_id: str = "run", reason: str = "in_sample_only") -> dict:
    return {
        "decision_id": f"{run_id}-d{i}",
        "claim_id": f"{run_id}-claim-{i}",
        "run_id": run_id,
        "statement": "BTC perpetual funding carry is strong in-sample and predicts returns.",
        "verdict": "kill",
        "kill_reason": reason,
        "metrics": {"power_sufficient": True},
        "logged_at": "2026-01-01T00:00:00Z",
    }


def _configure(tmp_path, monkeypatch):
    from penrose import config

    decisions = tmp_path / "decisions.jsonl"
    proposed = tmp_path / "reports" / "principles_proposed.jsonl"
    approved = tmp_path / "principles.jsonl"
    approved.write_text("approved sentinel\n")
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(config, "PRINCIPLES_PROPOSED", proposed)
    monkeypatch.setattr(config, "PRINCIPLES_LOG", approved)
    monkeypatch.setattr(config, "PRINCIPLE_MIN_KILLS", 3)
    return decisions, proposed, approved


def test_cross_run_same_class_kills_yield_proposal_when_per_run_rule_misses(tmp_path, monkeypatch):
    from penrose.brain import Decision
    from penrose.learning import distill_principles
    from penrose.pipeline import stages

    decisions, _, _ = _configure(tmp_path, monkeypatch)
    rows = [_kill(0, run_id=f"run-{i}") for i in range(3)]
    _write_jsonl(decisions, rows)

    proposals = distill_principles()

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["status"] == "proposed"
    assert proposal["domain"] == "funding-carry"
    assert proposal["kill_reason"] == "in_sample_only"
    assert proposal["supporting_kill_count"] == 3
    assert proposal["example_claim_ids"] == ["run-0-claim-0", "run-1-claim-0", "run-2-claim-0"]
    assert "review prior" in proposal["meaning"]

    for row in rows:
        per_run = [Decision(
            decision_id=row["decision_id"],
            claim_id=row["claim_id"],
            verdict=row["verdict"],
            kill_reason=row["kill_reason"],
            rationale="fixture",
        )]
        assert stages.propose_principle(per_run) is None


def test_cross_run_principle_floor_stays_at_three(tmp_path, monkeypatch):
    from penrose import config
    from penrose.learning import distill_principles

    decisions, _, _ = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "PRINCIPLE_MIN_KILLS", 2)
    _write_jsonl(decisions, [_kill(0, run_id="run-a"), _kill(0, run_id="run-b")])

    assert distill_principles() == []


def test_distill_cli_persists_proposed_schema_and_leaves_approved_untouched(tmp_path, monkeypatch, capsys):
    from penrose import cli

    decisions, proposed, approved = _configure(tmp_path, monkeypatch)
    _write_jsonl(decisions, [_kill(0, run_id=f"run-{i}") for i in range(3)])

    assert cli.main(["distill", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "proposed"
    assert proposed.exists()
    rows = [json.loads(line) for line in proposed.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "proposed"
    assert row["supporting_kill_count"] == 3
    assert row["example_claim_ids"] == ["run-0-claim-0", "run-1-claim-0", "run-2-claim-0"]
    assert "meaning" in row
    assert "ts" not in row
    assert approved.read_text() == "approved sentinel\n"
    assert len(payload["distilled"]) == 1


def test_distill_cli_is_idempotent(tmp_path, monkeypatch, capsys):
    from penrose import cli

    decisions, proposed, _ = _configure(tmp_path, monkeypatch)
    _write_jsonl(decisions, [_kill(0, run_id=f"run-{i}") for i in range(3)])

    assert cli.main(["distill", "--json"]) == 0
    capsys.readouterr()
    first = proposed.read_text()
    assert cli.main(["distill", "--json"]) == 0
    capsys.readouterr()

    assert proposed.read_text() == first
    assert len([line for line in first.splitlines() if line.strip()]) == 1


def test_empty_or_malformed_corpus_fails_open(tmp_path, monkeypatch):
    from penrose.learning import distill_principles

    decisions, _, _ = _configure(tmp_path, monkeypatch)

    assert distill_principles() == []
    decisions.write_text("{bad json\n")
    assert distill_principles() == []


def test_p9_firewall_refuses_case_variant_of_approved_ledger_cr1(tmp_path, monkeypatch):
    """CR-1: on a case-insensitive FS a case-variant of the approved ledger must be refused, not clobber it."""
    from penrose import config, proposals
    approved = tmp_path / "principles.jsonl"
    approved.write_text('{"status":"approved","principle_id":"sentinel"}\n')
    monkeypatch.setattr(config, "PRINCIPLES_LOG", approved)
    import pytest
    with pytest.raises(ValueError):
        proposals.write_proposals([{"principle_id": "x", "domain": "d", "kill_reason": "k"}],
                                  path=tmp_path / "Principles.jsonl", source="distilled")
    assert "sentinel" in approved.read_text()  # approved rows untouched


def test_empty_distill_purges_stale_proposals_cr2(tmp_path):
    """CR-2: an empty distill with replace_source purges this source's stale rows (corpus lost support)."""
    from penrose import proposals
    store = tmp_path / "proposed.jsonl"
    proposals.write_proposals([{"principle_id": "p1", "domain": "d", "kill_reason": "in_sample_only"}],
                              path=store, source="distilled", replace_source=True)
    assert len(proposals.read_proposals(store)) == 1
    proposals.write_proposals([], path=store, source="distilled", replace_source=True)
    assert len(proposals.read_proposals(store)) == 0


def test_corpus_reader_skips_corrupt_line_not_whole_file_cr2r2(tmp_path):
    """CR2-1: one corrupt line must not make a full corpus read as empty (which would wipe proposals)."""
    from penrose.learning import _read_jsonl
    p = tmp_path / "decisions.jsonl"
    p.write_text('{"a": 1}\n{corrupt not json\n{"b": 2}\n')
    assert _read_jsonl(p) == [{"a": 1}, {"b": 2}]  # bad line skipped, valid rows kept


def test_proposals_store_reader_skips_corrupt_line_cr4(tmp_path):
    """CR4-1: one corrupt line in the proposals store must not discard the other rows (symmetric w/ CR2-1)."""
    from penrose import proposals
    store = tmp_path / "proposed.jsonl"
    store.write_text('{"principle_id":"a","status":"proposed"}\n{corrupt\n{"principle_id":"b","status":"proposed"}\n')
    assert len(proposals._read_rows(store)) == 2
    # and a subsequent write preserves the survivors rather than wiping them
    proposals.write_proposals([{"principle_id": "c", "domain": "d", "kill_reason": "k"}],
                              path=store, source="other")
    ids = {r["principle_id"] for r in proposals.read_proposals(store)}
    assert {"a", "b", "c"} <= ids
