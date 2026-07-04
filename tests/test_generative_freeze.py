import pytest


FROZEN = "the generative layer is frozen (PEN-17)"


def test_pen17_run_dream_frozen_before_filesystem_write(tmp_path, monkeypatch):
    from penrose import config
    from penrose.dream import run_dream

    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", False)
    monkeypatch.setattr(config, "DREAM_ARCHIVES", tmp_path / "dreams")

    with pytest.raises(RuntimeError, match="generative layer is frozen"):
        run_dream(n=1, generate_only=True, run_id="frozen-test")

    assert not (tmp_path / "dreams" / "frozen-test").exists()


def test_pen17_run_synthesis_frozen_before_filesystem_write(tmp_path, monkeypatch):
    from penrose import config
    from penrose.synthesize import run_synthesis

    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", False)
    monkeypatch.setattr(config, "SYNTHESIS_ARCHIVES", tmp_path / "syntheses")

    with pytest.raises(RuntimeError, match="generative layer is frozen"):
        run_synthesis(n=1, generate_only=True, run_id="frozen-synth")

    assert not (tmp_path / "syntheses" / "frozen-synth").exists()


def test_pen17_run_dream_enabled_uses_existing_path(tmp_path, monkeypatch):
    from penrose import config, dream

    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)
    monkeypatch.setattr(config, "DREAM_ARCHIVES", tmp_path / "dreams")
    monkeypatch.setattr(config, "DREAM_RUNS", tmp_path / "dream_runs.jsonl")
    monkeypatch.setattr(
        dream, "build_evidence_packet", lambda: {"eligible_verdicts": [], "snapshot_hash": "abc"})
    monkeypatch.setattr(
        dream, "capability_manifest", lambda: {"series": [], "trusted_strategy_classes": []})
    monkeypatch.setattr(
        dream, "generate_candidates",
        lambda n, evidence, capabilities: (
            [{"statement": "candidate", "candidate_class": "conceptual_only"}],
            {"model": "test", "in_tokens": 0, "out_tokens": 0, "cost_usd": 0,
             "cached": False, "prompt_hash": "prompt"},
        ),
    )

    out = dream.run_dream(n=1, generate_only=True, run_id="enabled-test")
    assert out["status"] == "generated_only"
    assert out["candidates_generated"] == 1


def test_pen17_distill_principles_guard_and_override(monkeypatch):
    from penrose import config
    from penrose.learning import distill_principles

    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", False)
    with pytest.raises(RuntimeError, match="generative layer is frozen"):
        distill_principles(records=[])

    monkeypatch.setattr(config, "GENERATIVE_LAYER_ENABLED", True)
    assert distill_principles(records=[]) == []
