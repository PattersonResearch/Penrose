import inspect
import json
import types
from pathlib import Path


def test_append_jsonl_creates_missing_dirs(tmp_path):
    from penrose.pipeline.run import _append_jsonl

    out = tmp_path / "missing" / "nested" / "rows.jsonl"
    _append_jsonl(out, {"ok": True})

    assert out.exists()
    assert json.loads(out.read_text().strip()) == {"ok": True}


def _write_module(root: Path, name: str, body: str) -> None:
    mod = root / name
    mod.mkdir()
    (mod / "impl.py").write_text(body)


def test_same_owner_hyphen_underscore_aliases_do_not_warn(tmp_path, monkeypatch, capsys):
    from penrose import config
    from penrose.pipeline import run as runmod

    _write_module(
        tmp_path,
        "same_owner",
        "__module_id__ = 'same_owner'\n"
        "__strategy_class__ = 'crypto_funding_carry'\n"
        "__strategy_class_aliases__ = ['crypto-funding-carry']\n"
        "def run(bundle, claim, cost):\n"
        "    return {'ok': False, 'reason': 'data_unavailable: unit'}\n",
    )
    monkeypatch.setattr(config, "MODULES", tmp_path)
    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    runmod._register_known_modules()

    err = capsys.readouterr().err
    assert "strategy_class alias collision" not in err
    assert runmod.REGISTRY["crypto-funding-carry"] is runmod.REGISTRY["crypto_funding_carry"]


def test_distinct_owner_canonical_alias_collision_still_warns(tmp_path, monkeypatch, capsys):
    from penrose import config
    from penrose.pipeline import run as runmod

    _write_module(
        tmp_path,
        "owner_a",
        "__module_id__ = 'owner_a'\n"
        "__strategy_class__ = 'macro_signal_volatility_forecast'\n"
        "def run(bundle, claim, cost):\n"
        "    return {'ok': False, 'reason': 'data_unavailable: unit'}\n",
    )
    _write_module(
        tmp_path,
        "owner_b",
        "__module_id__ = 'owner_b'\n"
        "__strategy_class__ = 'macro-signal-volatility-forecast'\n"
        "def run(bundle, claim, cost):\n"
        "    return {'ok': False, 'reason': 'data_unavailable: unit'}\n",
    )
    monkeypatch.setattr(config, "MODULES", tmp_path)
    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    runmod._register_known_modules()

    err = capsys.readouterr().err
    assert "strategy_class alias collision" in err
    assert "owner_a" in err


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


def test_rerun_supersedes_same_source_decisions_only(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import Claim, Decision
    from penrose.pipeline import run as runmod

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

    paper = tmp_path / "synthetic_source.md"
    paper.write_text("Synthetic source\n")
    claim = Claim(
        claim_id="synthetic_source-c1",
        statement="unit claim",
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="synthetic_source",
        source_span="Synthetic source",
        claimed_metric_quote="",
        applicable_strategy_class="unit_class",
    )

    module = types.SimpleNamespace(
        __strategy_class__="unit_class",
        __module_id__="unit_module",
        __auto_generated__=False,
        run=lambda bundle, claim, cost: {"ok": True, "net": [0.1], "positions": [1.0],
                                         "bars_per_year": 252.0},
    )
    runmod.REGISTRY.clear()
    runmod.REGISTRY["unit_class"] = module
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest",
                        lambda *a, **k: {"psr": 0.91, "dsr": 0.91, "n_oos": 200,
                                         "oos_sharpe": 1.0, "capacity_usd": 1_000_000,
                                         "three_fold": {}, "bootstrap": {}, "permutation": {},
                                         "regime": {}})
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda claim, bt, holdout, synthetic: Decision(
                            decision_id=f"{claim.claim_id}-d1",
                            claim_id=claim.claim_id,
                            verdict="watch",
                            kill_reason=None,
                            rationale="unit watch",
                            metrics={"psr": bt["psr"], "dsr": bt["dsr"]},
                        ))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")

    runmod._append_jsonl(config.DECISIONS_LOG, {
        "decision_id": "other-c1-d1",
        "claim_id": "other-c1",
        "source_id": "other_source",
        "verdict": "kill",
    })

    first = runmod.run_source(
        paper, use_llm=False, claims_override=[claim], bundle_override=_TinyBundle())
    skipped = runmod.run_source(
        paper, use_llm=False, claims_override=[claim], bundle_override=_TinyBundle())
    forced = runmod.run_source(
        paper, use_llm=False, claims_override=[claim], bundle_override=_TinyBundle(), force=True)

    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines()
            if line.strip()]
    own = [row for row in rows if row.get("source_id") == "synthetic_source"]
    other = [row for row in rows if row.get("source_id") == "other_source"]

    assert len(own) == 1
    assert own[0]["claim_id"] == "synthetic_source-c1"
    assert own[0]["verdict"] == "watch"
    assert own[0]["run_id"] == forced["idempotency"]["run_id"]
    assert len(other) == 1 and other[0]["claim_id"] == "other-c1"
    assert skipped["note"] == "already processed (unchanged); use --force to re-run"
    assert skipped["idempotency"]["skipped"] is True
    assert "decisions" not in skipped
    assert first["decisions"] == forced["decisions"]


def test_crash_before_completion_preserves_prior_decision_rows(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import Claim
    from penrose.pipeline import run as runmod

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", tmp_path / "processed_papers.json")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", tmp_path / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    (tmp_path / "modules").mkdir()

    paper = tmp_path / "crash_source.md"
    paper.write_text("Crash source\n")
    prior = {
        "decision_id": "crash_source-c1-d1",
        "claim_id": "crash_source-c1",
        "source_id": "crash_source",
        "run_id": "old-run",
        "verdict": "watch",
    }
    runmod._append_jsonl(config.DECISIONS_LOG, prior)
    claim = Claim(
        claim_id="crash_source-c1",
        statement="unit claim",
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="crash_source",
        source_span="Crash source",
        claimed_metric_quote="",
        applicable_strategy_class="unit_class",
    )
    monkeypatch.setattr(runmod.dataclient, "fetch_bundle",
                        lambda: (_ for _ in ()).throw(RuntimeError("bundle exploded")))

    try:
        runmod.run_source(paper, use_llm=False, claims_override=[claim], force=True)
    except RuntimeError as exc:
        assert "bundle exploded" in str(exc)
    else:
        raise AssertionError("run_source should have raised before completion")

    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines()
            if line.strip()]
    assert rows == [prior]


def test_decision_supersede_uses_atomic_tmp_replace(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)
    runmod._append_jsonl(path, {"claim_id": "source-c1", "source_id": "source",
                                "run_id": "old", "verdict": "kill"})
    runmod._append_jsonl(path, {"claim_id": "source-c1", "source_id": "source",
                                "run_id": "new", "verdict": "watch"})
    runmod._append_jsonl(path, {"claim_id": "other-c1", "source_id": "other",
                                "run_id": "old", "verdict": "kill"})

    original_write_text = Path.write_text
    original_replace = Path.replace
    replaced = []

    def guarded_write_text(self, *args, **kwargs):
        if self == path:
            raise AssertionError("supersede must not truncate decisions.jsonl directly")
        return original_write_text(self, *args, **kwargs)

    def spy_replace(self, target):
        replaced.append((self, Path(target)))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "replace", spy_replace)

    removed = runmod._supersede_decision_rows("source", "new")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert removed == 1
    assert [row["run_id"] for row in rows if row["source_id"] == "source"] == ["new"]
    assert any(row["source_id"] == "other" for row in rows)
    assert replaced and replaced[0][0].name.startswith("decisions.jsonl.")
    assert replaced[0][1] == path
    source = inspect.getsource(runmod._supersede_decision_rows)
    assert "fcntl.flock" in source
    assert "tmp.replace(path)" in source


def test_legacy_decision_without_source_id_is_superseded_by_claim_prefix(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "DECISIONS_LOG", path)
    runmod._append_jsonl(path, {"claim_id": "legacy_source-c1", "verdict": "kill"})
    runmod._append_jsonl(path, {"claim_id": "legacy_source-c1", "source_id": "legacy_source",
                                "run_id": "new", "verdict": "watch"})
    runmod._append_jsonl(path, {"claim_id": "other_source-c1", "verdict": "kill"})

    removed = runmod._supersede_decision_rows("legacy_source", "new")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert removed == 1
    assert rows == [
        {"claim_id": "legacy_source-c1", "source_id": "legacy_source",
         "run_id": "new", "verdict": "watch"},
        {"claim_id": "other_source-c1", "verdict": "kill"},
    ]


def test_fidelity_timeout_records_unknown_and_run_continues(tmp_path, monkeypatch):
    from penrose import config
    from penrose import concepts
    from penrose.brain import Claim, Decision
    from penrose.pipeline import extract
    from penrose.pipeline import run as runmod

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", tmp_path / "processed_papers.json")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", tmp_path / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    monkeypatch.setattr(config, "FIDELITY_CHECK", True)
    (tmp_path / "modules").mkdir()

    module_path = tmp_path / "unit_module.py"
    module_path.write_text("def run(bundle, claim, cost):\n    return {}\n")
    module = types.SimpleNamespace(
        __strategy_class__="unit_class",
        __module_id__="unit_module",
        __auto_generated__=False,
        __file__=str(module_path),
        run=lambda bundle, claim, cost: {"ok": True, "net": [0.1], "positions": [1.0],
                                         "bars_per_year": 252.0},
    )
    runmod.REGISTRY.clear()
    runmod.REGISTRY["unit_class"] = module
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    paper = tmp_path / "fidelity_timeout.md"
    paper.write_text("Fidelity timeout\n")
    claim = Claim(
        claim_id="fidelity_timeout-c1",
        statement="unit claim",
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="fidelity_timeout",
        source_span="Fidelity timeout",
        claimed_metric_quote="",
        applicable_strategy_class="unit_class",
    )

    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest",
                        lambda *a, **k: {"psr": 0.99, "dsr": 0.99, "n_oos": 2000,
                                         "oos_sharpe": 2.0, "capacity_usd": 1_000_000,
                                         "three_fold": {}, "bootstrap": {}, "permutation": {},
                                         "regime": {}})
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda claim, bt, holdout, synthetic: Decision(
                            decision_id=f"{claim.claim_id}-d1",
                            claim_id=claim.claim_id,
                            verdict="research-supported",
                            kill_reason=None,
                            rationale="unit supported",
                            metrics={"psr": bt["psr"], "dsr": bt["dsr"]},
                        ))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    calls = {"n": 0}

    def fake_fidelity(claim, module_code):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"faithful": True, "verified": True, "checked": True,
                    "confidence": 0.9, "divergences": [], "note": "route ok"}
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(runmod.fidelity, "assess", fake_fidelity)

    out = runmod.run_source(
        paper, use_llm=True, claims_override=[claim], bundle_override=_TinyBundle())

    assert out["decisions"] == [{
        "claim_id": "fidelity_timeout-c1",
        "verdict": "watch",
        "kill_reason": None,
    }]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines()
            if line.strip()]
    assert len(rows) == 1
    assert rows[0]["verdict"] == "watch"
    assert rows[0]["metrics"]["fidelity"]["checked"] is False
    assert rows[0]["metrics"]["fidelity_provenance"] == "unknown"
    assert "TimeoutError" in rows[0]["metrics"]["fidelity"]["error"]


def test_max_claims_limits_processed_claims(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import Claim
    from penrose.pipeline import run as runmod

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
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

    paper = tmp_path / "max_claims.md"
    paper.write_text("Max claims\n")
    claims = [
        Claim(f"max_claims-c{i}", f"claim {i}", "unit", "unit", "1d", "max_claims",
              "Max claims", "", applicable_strategy_class="missing")
        for i in range(3)
    ]

    out = runmod.run_source(
        paper, use_llm=False, claims_override=claims, bundle_override=_TinyBundle(), max_claims=1)

    assert out["max_claims"] == 1
    assert out["claims_extracted"] == 1
    assert out["decisions"] == [{
        "claim_id": "max_claims-c0",
        "verdict": "pending_module",
        "kill_reason": None,
    }]
