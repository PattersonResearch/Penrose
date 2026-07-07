import json
import types
from pathlib import Path

import pandas as pd


def _claim(statement, *, claim_id="c1", strategy_class="weather_bias"):
    from penrose.brain import Claim

    return Claim(
        claim_id=claim_id,
        statement=statement,
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="paper",
        source_span=statement,
        claimed_metric_quote="",
        applicable_strategy_class=strategy_class,
    )


def _source():
    from penrose.pipeline.p1_ingest import IngestedSource

    return IngestedSource(
        source_id="paper",
        title="paper",
        text="unit",
        n_pages=1,
        n_chars=4,
        text_sha256="abc",
        injection_flags=[],
    )


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


def _patch_run_paths(tmp_path, monkeypatch):
    from penrose import config

    tmp_path.mkdir(parents=True, exist_ok=True)
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
    monkeypatch.setattr(config, "AUTO_MODULES", tmp_path / "modules" / "_auto")
    monkeypatch.setattr(config, "FIDELITY_REJECTIONS", tmp_path / "reports" / "fidelity_rejections.jsonl")
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", True)
    monkeypatch.setattr(config, "FIDELITY_CHECK", True)
    (tmp_path / "modules").mkdir(parents=True)


def test_r1_claim_type_classifier_and_descriptive_template(monkeypatch):
    from penrose.pipeline import spec_gen

    desc = _claim("The unconditional mean ensemble bias is +0.96°F over 1,700 observations.")
    trading = _claim("A 12 month momentum signal enters long positions and earns positive Sharpe.")
    unclear = _claim("", strategy_class="")

    assert spec_gen.classify_claim_type(desc) == "descriptive_statistical"
    assert spec_gen.classify_claim_type(trading) == "trading_strategy"
    assert spec_gen.classify_claim_type(unclear) == "trading_strategy"

    monkeypatch.setattr(spec_gen, "_catalog_vocab", lambda: "")
    prompt = spec_gen._base_prompt(desc, _source(), "descriptive_statistical")
    assert "claim_type: descriptive_statistical" in prompt
    assert "compute and test the stated statistic directly" in prompt
    assert "Do not invent" in prompt

    spec = spec_gen.generate_spec(desc, _source(), use_llm=False)
    assert spec["claim_type"] == "descriptive_statistical"
    assert "descriptive statistic" in spec["claim_translation"]
    assert "no positions/PnL" in spec["signal_logic"]


def test_r2_rejection_memory_persists_and_injects_prompt(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import fidelity_memory, spec_gen

    store = tmp_path / "reports" / "fidelity_rejections.jsonl"
    monkeypatch.setattr(config, "FIDELITY_REJECTIONS", store)
    monkeypatch.setattr(spec_gen, "_catalog_vocab", lambda: "")

    claim = _claim("The unconditional mean ensemble bias is +0.96°F.", strategy_class="weather_bias")
    baseline = spec_gen._base_prompt(claim, _source(), "descriptive_statistical")
    assert "AVOID THESE PAST FIDELITY FAILURES" not in baseline

    fidelity_memory.append_rejection(
        strategy_class="weather_bias",
        claim_type="descriptive_statistical",
        divergences=["module never computes the unconditional mean"],
        note="unfaithful",
    )

    rows = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    assert rows[0]["strategy_class"] == "weather_bias"
    assert rows[0]["claim_type"] == "descriptive_statistical"
    assert rows[0]["divergences"] == ["module never computes the unconditional mean"]
    assert rows[0]["ts"]

    prompt = spec_gen._base_prompt(claim, _source(), "descriptive_statistical")
    assert "AVOID THESE PAST FIDELITY FAILURES" in prompt
    assert "module never computes the unconditional mean" in prompt

    monkeypatch.setattr(config, "FIDELITY_REJECTIONS", tmp_path / "reports" / "missing.jsonl")
    no_store = spec_gen._base_prompt(claim, _source(), "descriptive_statistical")
    assert no_store == baseline


def _run_precheck_case(tmp_path, monkeypatch, precheck_result):
    from penrose import concepts, config
    from penrose.brain import Decision
    from penrose.pipeline import extract
    from penrose.pipeline import run as runmod

    _patch_run_paths(tmp_path, monkeypatch)
    paper = tmp_path / "paper.md"
    paper.write_text("paper")
    claim = _claim("A momentum signal enters long positions and earns positive Sharpe.", claim_id="paper-c1")
    spec = {
        "module_id": "auto_unit",
        "strategy_class": "momentum_unit",
        "claim_type": "trading_strategy",
        "claim_translation": "trade momentum",
        "inputs": [],
        "signal_logic": "momentum signal",
        "kill_criterion": "unit",
        "_path": str(tmp_path / "modules" / "_specs" / "paper-c1.yaml"),
    }
    module_path = tmp_path / "impl.py"
    module_path.write_text("__strategy_class__='momentum_unit'\ndef run(bundle, claim, cost): return {}\n")
    auto_module = types.SimpleNamespace(
        __strategy_class__="momentum_unit",
        __module_id__="auto_unit",
        __auto_generated__=True,
        __file__=str(module_path),
    )
    net = pd.Series([0.01] * 20, index=pd.date_range("2020-01-01", periods=20, freq="D"))
    positions = pd.Series([1.0] * 20, index=net.index)
    calls = {"impl": 0, "sandbox": 0, "p7": 0, "fid": 0}

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.spec_gen, "generate_spec", lambda *a, **k: dict(spec))
    monkeypatch.setattr(runmod.sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(runmod.sandbox, "ensure_image", lambda: True)

    def fake_try_implement(*args, **kwargs):
        calls["impl"] += 1
        return {"ok": True, "module": auto_module, "module_id": "auto_unit", "validation": {}}

    def fake_sandbox(*args, **kwargs):
        calls["sandbox"] += 1
        return {"ok": True, "net": net, "positions": positions, "bars_per_year": 252.0}

    def fake_backtest(*args, **kwargs):
        calls["p7"] += 1
        return {"psr": 0.99, "dsr": 0.99, "n_oos": 200, "oos_sharpe": 1.0,
                "capacity_usd": 1_000_000, "three_fold": {}, "bootstrap": {},
                "permutation": {}, "regime": {}}

    def fake_fidelity(*args, **kwargs):
        calls["fid"] += 1
        precheck_results = precheck_result if isinstance(precheck_result, list) else [precheck_result]
        if calls["fid"] <= len(precheck_results):
            result = precheck_results[calls["fid"] - 1]
            if isinstance(result, Exception):
                raise result
            return result
        return {"faithful": True, "verified": True, "checked": True,
                "confidence": 0.9, "divergences": [], "note": "post ok"}

    monkeypatch.setattr(runmod.impl_gen, "try_implement", fake_try_implement)
    monkeypatch.setattr(runmod.sandbox, "run_in_container", fake_sandbox)
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest", fake_backtest)
    monkeypatch.setattr(runmod.fidelity, "assess", fake_fidelity)
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
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(
        paper, use_llm=True, claims_override=[claim], bundle_override=_TinyBundle(), force=True)
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines()
            if line.strip()]
    return out, rows, calls


def test_r3_unfaithful_precheck_blocks_before_auto_impl_and_backtest(tmp_path, monkeypatch):
    from penrose.pipeline import run as runmod

    unfaithful = {"faithful": False, "verified": False, "checked": True, "confidence": 0.92,
                  "divergences": ["spec translates unconditional mean into momentum trading"],
                  "note": "unfaithful spec"}
    out, rows, calls = _run_precheck_case(
        tmp_path,
        monkeypatch,
        [unfaithful] * runmod.SPEC_SELF_CORRECTION_MAX_ATTEMPTS,
    )

    assert out["decisions"] == [{
        "claim_id": "paper-c1",
        "verdict": "cannot_replicate",
        "kill_reason": "unfaithful_spec",
    }]
    assert rows[0]["verdict"] == "cannot_replicate"
    assert rows[0]["kill_reason"] == "unfaithful_spec"
    assert calls["impl"] == 0
    assert calls["sandbox"] == 0
    assert calls["p7"] == 0
    assert calls["fid"] == runmod.SPEC_SELF_CORRECTION_MAX_ATTEMPTS


def test_r3_faithful_or_error_precheck_falls_through_to_backtest(tmp_path, monkeypatch):
    _, faithful_rows, faithful_calls = _run_precheck_case(
        tmp_path / "faithful",
        monkeypatch,
        {"faithful": True, "verified": True, "checked": True, "confidence": 0.91,
         "divergences": [], "note": "spec ok"},
    )
    assert faithful_rows[0]["verdict"] == "watch"
    assert faithful_calls["impl"] == 1
    assert faithful_calls["sandbox"] == 1
    assert faithful_calls["p7"] == 1

    _, error_rows, error_calls = _run_precheck_case(
        tmp_path / "error",
        monkeypatch,
        TimeoutError("refuter timeout"),
    )
    assert error_rows[0]["verdict"] == "watch"
    assert error_calls["impl"] == 1
    assert error_calls["sandbox"] == 1
    assert error_calls["p7"] == 1
