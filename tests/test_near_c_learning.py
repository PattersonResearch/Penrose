from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


def test_propose_only_store_read_api_never_touches_approved_principles(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import read_proposals
    from penrose.proposals import write_proposals

    proposals = tmp_path / "reports" / "principles_proposed.jsonl"
    approved = tmp_path / "principles.jsonl"
    approved.write_text("sentinel approved store\n")
    monkeypatch.setattr(config, "PRINCIPLES_PROPOSED", proposals)
    monkeypatch.setattr(config, "PRINCIPLES_LOG", approved)

    assert read_proposals() == []
    written = write_proposals([{
        "statement": "In funding-carry, in_sample_only failures recur.",
        "domain": "funding-carry",
        "kill_reason": "in_sample_only",
        "n_observations": 3,
        "confidence": 0.5,
    }], ts="2026-06-25T00:00:00Z")

    assert len(written) == 1
    rows = read_proposals()
    assert rows[0]["status"] == "proposed"
    assert rows[0]["source"] == "distilled"
    assert approved.read_text() == "sentinel approved store\n"


def test_proposals_store_corrupt_or_missing_fails_open(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import read_proposals

    store = tmp_path / "reports" / "principles_proposed.jsonl"
    monkeypatch.setattr(config, "PRINCIPLES_PROPOSED", store)

    assert read_proposals() == []
    store.parent.mkdir(parents=True)
    store.write_text("{bad json\n")
    assert read_proposals() == []


def test_proposals_store_is_gitignored():
    check = subprocess.run(
        ["git", "check-ignore", "reports/principles_proposed.jsonl"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert check.returncode == 0
    assert check.stdout.strip() == "reports/principles_proposed.jsonl"


def test_distill_principles_cross_run_finds_what_per_run_rule_misses(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import Decision
    from penrose.learning import distill_principles
    from penrose.pipeline import stages

    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    # PEN-17: this test intentionally exercises the distillation entry point.
    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)

    rows = []
    for run_idx in range(3):
        for claim_idx in range(2):
            rows.append({
                "decision_id": f"run{run_idx}-c{claim_idx}-d1",
                "claim_id": f"run{run_idx}-c{claim_idx}",
                "run_id": f"run{run_idx}",
                "statement": "BTC perpetual funding carry predicts next day returns.",
                "verdict": "kill",
                "kill_reason": "in_sample_only",
                "metrics": {"power_sufficient": True},
                "logged_at": f"2026-01-0{run_idx + 1}T00:00:00Z",
            })
    _write_jsonl(decisions, rows)

    proposals = distill_principles()

    assert proposals
    assert proposals[0]["domain"] == "funding-carry"
    assert proposals[0]["kill_reason"] == "in_sample_only"
    assert proposals[0]["n_observations"] >= 3
    assert proposals[0]["status"] == "proposed"

    for run_idx in range(3):
        per_run = [
            Decision(
                decision_id=r["decision_id"],
                claim_id=r["claim_id"],
                verdict=r["verdict"],
                kill_reason=r["kill_reason"],
                rationale="fixture",
            )
            for r in rows
            if r["run_id"] == f"run{run_idx}"
        ]
        assert stages.propose_principle(per_run) is None


def test_distill_principles_empty_or_corrupt_corpus_fails_open(tmp_path, monkeypatch):
    from penrose import config
    from penrose.learning import distill_principles

    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    # PEN-17: this test intentionally exercises the distillation entry point.
    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)

    assert distill_principles() == []
    decisions.write_text("{bad json\n")
    assert distill_principles() == []


def test_cli_proposals_reads_and_principles_distills_to_propose_only_store(tmp_path, monkeypatch, capsys):
    from penrose import cli, config

    proposals = tmp_path / "reports" / "principles_proposed.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    approved = tmp_path / "principles.jsonl"
    approved.write_text("approved sentinel\n")
    monkeypatch.setattr(config, "PRINCIPLES_PROPOSED", proposals)
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(config, "PRINCIPLES_LOG", approved)
    # PEN-17: this test intentionally exercises CLI distillation.
    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)

    _write_jsonl(decisions, [{
        "decision_id": f"d{i}",
        "claim_id": f"c{i}",
        "statement": "BTC perpetual funding carry predicts next day returns.",
        "verdict": "kill",
        "kill_reason": "in_sample_only",
        "metrics": {"power_sufficient": True},
        "logged_at": "2026-01-01T00:00:00Z",
    } for i in range(3)])

    assert cli.main(["principles", "--json"]) == 0
    distilled = json.loads(capsys.readouterr().out)
    assert distilled["status"] == "proposed"
    assert len(distilled["distilled"]) == 1

    assert cli.main(["proposals", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["status"] == "proposed"
    assert approved.read_text() == "approved sentinel\n"
