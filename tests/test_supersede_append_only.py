"""FIX 1 / FIX 5c regression tests (v0.4.1): decisions.jsonl is append-only.

Covers the data-loss regression this fix closes: a --force re-run of an already-decided
claim must NEVER delete or rewrite a prior decisions.jsonl line, even when the re-run
produces zero new decisions or crashes mid-way.
"""
import json
from pathlib import Path

import pytest


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- unit-level: _supersede_decision_rows itself ---------------------------------- #

def test_supersede_never_truncates_or_rewrites_the_file(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)

    old_kill = {"decision_id": "source-c1-d1", "claim_id": "source-c1", "source_id": "source",
                "run_id": "old-run", "verdict": "kill", "kill_reason": "in_sample_only"}
    old_needs_data = {"decision_id": "source-c2-d1", "claim_id": "source-c2", "source_id": "source",
                      "run_id": "old-run", "verdict": "needs_data"}
    other_source = {"decision_id": "other-c1-d1", "claim_id": "other-c1", "source_id": "other",
                    "run_id": "old-run", "verdict": "kill"}
    for row in (old_kill, old_needs_data, other_source):
        runmod._append_jsonl(path, row)

    original_write_text = Path.write_text
    original_replace = Path.replace

    def guarded_write_text(self, *args, **kwargs):
        if self == path:
            raise AssertionError("supersede must never call write_text on decisions.jsonl")
        return original_write_text(self, *args, **kwargs)

    def guarded_replace(self, target):
        if Path(target) == path or self == path:
            raise AssertionError("supersede must never replace() decisions.jsonl (no rewrite)")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "replace", guarded_replace)

    marked = runmod._supersede_decision_rows("source", "new-run")

    rows = _read_rows(path)
    # the three ORIGINAL lines are still there, byte-for-byte, untouched
    assert old_kill in rows
    assert old_needs_data in rows
    assert other_source in rows
    # two new supersession markers were appended for "source"'s two prior decisions
    assert marked == 2
    markers = [r for r in rows if r.get("type") == "supersession_marker"]
    assert len(markers) == 2
    marked_ids = {m["decision_id"] for m in markers}
    assert marked_ids == {"source-c1-d1", "source-c2-d1"}
    for m in markers:
        assert m["verdict"] == "superseded"
        assert m["superseded_by_run_id"] == "new-run"
    # the OTHER source's row is untouched: no marker minted for it
    assert not any(m["decision_id"] == "other-c1-d1" for m in markers)


def test_supersede_does_not_remark_an_already_superseded_row(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)
    runmod._append_jsonl(path, {"decision_id": "source-c1-d1", "claim_id": "source-c1",
                                "source_id": "source", "run_id": "old-run", "verdict": "kill"})

    first = runmod._supersede_decision_rows("source", "run-2")
    assert first == 1
    second = runmod._supersede_decision_rows("source", "run-3")
    assert second == 0  # already superseded; no duplicate marker

    rows = _read_rows(path)
    markers = [r for r in rows if r.get("type") == "supersession_marker"]
    assert len(markers) == 1


def test_supersede_skips_rows_already_the_latest_state_for_their_identity(tmp_path, monkeypatch):
    """The common re-verdict-the-SAME-claim case: a new row with the SAME decision_id as a
    prior one (this run re-decided the same claim) needs no marker at all -- append order
    already makes the new row the latest state, and the old row's bytes are untouched."""
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)
    old_row = {"decision_id": "source-c1-d1", "claim_id": "source-c1", "source_id": "source",
              "run_id": "old-run", "verdict": "kill"}
    new_row = {"decision_id": "source-c1-d1", "claim_id": "source-c1", "source_id": "source",
              "run_id": "new-run", "verdict": "watch"}
    runmod._append_jsonl(path, old_row)
    runmod._append_jsonl(path, new_row)

    marked = runmod._supersede_decision_rows("source", "new-run")

    rows = _read_rows(path)
    assert old_row in rows and new_row in rows      # BOTH physical lines preserved
    assert marked == 0                              # nothing needed marking
    assert not any(r.get("type") == "supersession_marker" for r in rows)


def test_legacy_rows_without_source_id_matched_by_claim_prefix(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)
    legacy = {"claim_id": "legacy_source-c1", "verdict": "kill"}       # no decision_id, no source_id
    other = {"claim_id": "other_source-c1", "verdict": "kill"}
    runmod._append_jsonl(path, legacy)
    runmod._append_jsonl(path, other)

    marked = runmod._supersede_decision_rows("legacy_source", "new-run")

    rows = _read_rows(path)
    assert legacy in rows and other in rows          # both original lines preserved verbatim
    assert marked == 1
    markers = [r for r in rows if r.get("type") == "supersession_marker"]
    assert len(markers) == 1
    assert markers[0]["claim_id"] == "legacy_source-c1"
    assert not any(m["claim_id"] == "other_source-c1" for m in markers)


# --- integration-level: run_source through the whole pipeline ---------------------- #

class _TinyBundle:
    series = {}
    requested_window = None

    def provenance_summary(self):
        return {}

    def any_synthetic(self):
        return False

    def reset_access(self):
        pass

    def accessed_synthetic(self):
        return False


def _isolate(tmp_path, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "CONCEPTS", tmp_path / "reports" / "concepts.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", tmp_path / "processed_papers.json")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", tmp_path / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", False)
    (tmp_path / "modules").mkdir()


def test_forced_rerun_that_crashes_mid_way_preserves_prior_rows(tmp_path, monkeypatch):
    """FIX 5c: a --force re-run that raises before completion must not lose the prior
    decision (already covered indirectly by test_phase2a_robustness.py; re-asserted here
    against the new non-destructive supersede for the specific data-loss incident shape)."""
    from penrose import config
    from penrose.brain import Claim
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift_claim.md"
    paper.write_text("Funding-rate vs realized-drift momentum claim text.\n")

    prior = {"decision_id": "funding_drift_claim-c3-d1", "claim_id": "funding_drift_claim-c3",
            "source_id": "funding_drift_claim", "run_id": "old-run", "verdict": "kill",
            "kill_reason": "in_sample_only"}
    runmod._append_jsonl(config.DECISIONS_LOG, prior)

    claim = Claim(claim_id="funding_drift_claim-c3", statement="unit claim", mechanism="unit",
                 scope="unit", horizon="1d", source_id="funding_drift_claim",
                 source_span="Funding-rate vs realized-drift momentum claim text.",
                 claimed_metric_quote="", applicable_strategy_class="unit_class")
    monkeypatch.setattr(runmod.dataclient, "fetch_bundle",
                        lambda: (_ for _ in ()).throw(RuntimeError("bundle exploded")))

    with pytest.raises(RuntimeError, match="bundle exploded"):
        runmod.run_source(paper, use_llm=False, claims_override=[claim], force=True)

    rows = _read_rows(config.DECISIONS_LOG)
    assert rows == [prior]   # untouched -- not even a marker, since nothing new was written


def test_forced_rerun_with_zero_extraction_preserves_prior_rows(tmp_path, monkeypatch):
    """FIX 1 + FIX 5c: the exact incident shape -- a --force re-run whose extraction
    (claims_override=[]) produces zero replacement decisions must never call supersede,
    so the prior row for this source is preserved untouched."""
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift_claim.md"
    paper.write_text("Funding-rate vs realized-drift momentum claim text.\n")

    prior = {"decision_id": "funding_drift_claim-c3-d1", "claim_id": "funding_drift_claim-c3",
            "source_id": "funding_drift_claim", "run_id": "old-run", "verdict": "kill",
            "kill_reason": "in_sample_only"}
    runmod._append_jsonl(config.DECISIONS_LOG, prior)

    out = runmod.run_source(paper, use_llm=False, claims_override=[], force=True)

    rows = _read_rows(config.DECISIONS_LOG)
    assert prior in rows                       # preserved, byte-for-byte
    assert not any(r.get("type") == "supersession_marker" for r in rows)
    assert out["note"] == "no claims extracted; pipeline ends here"


def test_forced_rerun_with_different_verdict_preserves_old_row_alongside_new(tmp_path, monkeypatch):
    """A successful --force re-run of the SAME claim that reaches a DIFFERENT verdict must
    preserve the old row physically (not just semantically) -- the append-only guarantee
    holds even when the new run legitimately supersedes the old conclusion."""
    from penrose import config
    from penrose.brain import Claim, Decision
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "synthetic_source.md"
    paper.write_text("Synthetic source\n")
    claim = Claim(claim_id="synthetic_source-c1", statement="unit claim", mechanism="unit",
                 scope="unit", horizon="1d", source_id="synthetic_source",
                 source_span="Synthetic source", claimed_metric_quote="",
                 applicable_strategy_class="unit_class")

    prior = {"decision_id": "synthetic_source-c1-d1", "claim_id": "synthetic_source-c1",
            "source_id": "synthetic_source", "run_id": "old-run", "verdict": "kill",
            "kill_reason": "in_sample_only"}
    runmod._append_jsonl(config.DECISIONS_LOG, prior)

    import types
    module = types.SimpleNamespace(
        __strategy_class__="unit_class", __module_id__="unit_module", __auto_generated__=False,
        run=lambda bundle, claim, cost: {"ok": True, "net": [0.1], "positions": [1.0],
                                         "bars_per_year": 252.0})
    runmod.REGISTRY.clear()
    runmod.REGISTRY["unit_class"] = module
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest",
                        lambda *a, **k: {"psr": 0.91, "dsr": 0.91, "n_oos": 200, "oos_sharpe": 1.0,
                                        "capacity_usd": 1_000_000, "three_fold": {}, "bootstrap": {},
                                        "permutation": {}, "regime": {}})
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda claim, bt, holdout, synthetic: Decision(
                            decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                            verdict="watch", kill_reason=None, rationale="unit watch, revised",
                            metrics={"psr": bt["psr"], "dsr": bt["dsr"]}))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")

    out = runmod.run_source(paper, use_llm=False, claims_override=[claim],
                            bundle_override=_TinyBundle(), force=True)

    rows = _read_rows(config.DECISIONS_LOG)
    assert prior in rows                                          # old "kill" row: preserved verbatim
    new_rows = [r for r in rows if r.get("run_id") == out["idempotency"]["run_id"]]
    assert len(new_rows) == 1 and new_rows[0]["verdict"] == "watch"  # new "watch" row: written
    # both rows share the same decision_id (deterministic id) -- the file now has the full
    # history for this claim, oldest first, nothing removed
    same_id = [r for r in rows if r.get("decision_id") == "synthetic_source-c1-d1"]
    assert len(same_id) == 2
