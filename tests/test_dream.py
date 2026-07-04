import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def test_run_id_rejects_paths():
    from penrose.dream import _validate_run_id

    for bad in ("../escape", "/tmp/escape", "a/b", "..", "a..b"):
        with pytest.raises(ValueError):
            _validate_run_id(bad)
    assert _validate_run_id("dream-20260621.safe_1") == "dream-20260621.safe_1"


def test_candidate_archive_is_immutable(tmp_path):
    from penrose import dream

    manifest = dream.create_manifest(
        run_id="immutable-test", generation_budget=1, model="test",
        corpus_snapshot_hash="abc", root=tmp_path / "immutable-test")
    dream.record_candidates(manifest, [{"statement": "first"}])
    dream.record_candidates(manifest, [{"statement": "first"}])
    with pytest.raises(RuntimeError):
        dream.record_candidates(manifest, [{"statement": "different"}])


def test_interrupted_registered_run_resumes(tmp_path, monkeypatch):
    from penrose import config, dream
    from penrose.pipeline import p7_backtest as p7

    archive_root = tmp_path / "dreams"
    run_root = archive_root / "resume-test"
    monkeypatch.setattr(config, "DREAM_ARCHIVES", archive_root)
    monkeypatch.setattr(config, "DREAM_RUNS", tmp_path / "dream_runs.jsonl")
    # PEN-17: this test exercises the end-to-end dream entry point intentionally.
    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)
    monkeypatch.setattr(p7, "LEDGER", tmp_path / "ledger.tsv")

    dream.create_manifest(
        run_id="resume-test", generation_budget=2, model="test",
        corpus_snapshot_hash="abc", root=run_root)
    monkeypatch.setattr(
        dream, "build_evidence_packet", lambda: {"eligible_verdicts": [], "snapshot_hash": "abc"})
    monkeypatch.setattr(
        dream, "capability_manifest", lambda: {"series": [], "trusted_strategy_classes": []})
    monkeypatch.setattr(
        dream, "generate_candidates",
        lambda n, evidence, capabilities: (
            [
                {"statement": "candidate one", "strategy_class": "carry-a",
                 "candidate_class": "testable_now"},
                {"statement": "candidate two", "strategy_class": "carry-b",
                 "candidate_class": "conceptual_only"},
            ],
            {"model": "test", "in_tokens": 0, "out_tokens": 0, "cost_usd": 0,
             "cached": False, "prompt_hash": "prompt"},
        ),
    )

    result = dream.run_dream(n=2, generate_only=True, run_id="resume-test")
    assert result["status"] == "generated_only"
    assert result["candidates_generated"] == 2
    assert (run_root / "candidates.raw.jsonl").exists()
    assert json.loads((run_root / "manifest.json").read_text())["status"] == "generated_only"
