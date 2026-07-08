import json
import types

import pandas as pd


def _claim(statement="Realized_drift minus funding predicts returns."):
    from penrose.brain import Claim

    return Claim(
        claim_id="paper-c1",
        statement=statement,
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="paper",
        source_span=statement,
        claimed_metric_quote="",
        applicable_strategy_class="unit_strategy",
    )


def _isolate(tmp_path, monkeypatch):
    from penrose import config

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
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", True)
    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    (tmp_path / "modules").mkdir(parents=True, exist_ok=True)


class _Bundle:
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


def test_w1_fidelity_prompt_is_claim_type_aware(monkeypatch):
    from penrose.pipeline import fidelity

    calls = []

    def fake_call_json(role, messages, **kwargs):
        calls.append(messages[0]["content"])
        return ({"faithful": True, "confidence": 0.95, "divergences": [], "note": "ok"},
                types.SimpleNamespace(independent_verifier=False))

    monkeypatch.setattr(fidelity.llm, "call_json", fake_call_json)
    fidelity.assess(
        _claim("The pooled mean of declared P&L series is greater than zero."),
        "{}",
        spec={"claim_type": "provided_series_statistic", "module_spec_only": True},
    )
    fidelity.assess(_claim(), "def run(bundle, claim, cost): return {}", spec={"claim_type": "trading_strategy"})

    assert "Absence of positions, PnL construction" in calls[0]
    assert "Do not reject merely because there are no positions or PnL" in calls[0]
    assert "CLAIM TYPE OVERRIDE: claim_type=provided_series_statistic" not in calls[1]


def test_w2_zero_claim_retry_then_engine_error_logs_both_attempts(tmp_path, monkeypatch):
    from penrose import config
    from penrose.llm import LLMResponse
    from penrose.pipeline import extract, run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift.md"
    paper.write_text("This research note claims realized drift minus funding predicts returns. " * 4)

    call_count = {"n": 0}

    def fake_call_json(role, messages, **kwargs):
        call_count["n"] += 1
        assert role == "claim_extractor"
        if call_count["n"] == 2:
            assert "this document contains research/trading claims" in messages[1]["content"]
        resp = LLMResponse(text='{"claims":[]}', model=f"resolved-{call_count['n']}",
                           in_tokens=1, out_tokens=1, cost_usd=0.0, elapsed_s=0.0)
        return {"claims": []}, resp

    monkeypatch.setattr(extract.llm, "call_json", fake_call_json)
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})

    out = runmod.run_source(paper, use_llm=True, bundle_override=_Bundle())

    assert out.get("engine_error") is True
    assert [a["n_extracted"] for a in out["p2"]["attempts"]] == [0, 0]
    assert [a["resolved_model"] for a in out["p2"]["attempts"]] == ["resolved-1", "resolved-2"]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert rows[0]["verdict"] == "engine_error"
    assert "retry-on-zero" in rows[0]["rationale"]


def test_w2_lenient_json_recovers_fences_and_outer_object():
    from penrose.llm import _parse_or_repair

    assert _parse_or_repair('```json\n{"claims": []}\n```') == {"claims": []}
    assert _parse_or_repair('analysis first\n{"claims": [{"statement": "x"}]}\ntrailing') == {
        "claims": [{"statement": "x"}]
    }


def test_w3_variable_coverage_missing_variable_routes_needs_review(tmp_path, monkeypatch):
    from penrose import config
    from penrose.brain import Decision
    from penrose.pipeline import extract, run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "paper.md"
    paper.write_text("paper")
    claim = _claim()
    spec = {"module_id": "unit", "strategy_class": "unit_strategy", "claim_type": "trading_strategy",
            "inputs": ["realized_drift", "funding"], "signal_logic": "realized_drift - funding"}
    impl_path = tmp_path / "impl.py"
    impl_path.write_text(
        "__strategy_class__='unit_strategy'\n"
        "def run(bundle, claim, cost):\n"
        "    funding = 1\n"
        "    signal = funding - 0\n"
        "    return {}\n"
    )
    module = types.SimpleNamespace(__strategy_class__="unit_strategy", __module_id__="unit",
                                   __auto_generated__=True, __file__=str(impl_path))
    idx = pd.date_range("2020-01-01", periods=20)
    monkeypatch.setattr(extract, "classify_claim", lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                                                  "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup", lambda claim, reader: {"stage": "P5", "killed": False})
    monkeypatch.setattr(runmod.spec_gen, "generate_spec", lambda *a, **k: dict(spec))
    monkeypatch.setattr(runmod.sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(runmod.sandbox, "ensure_image", lambda: True)
    monkeypatch.setattr(runmod.impl_gen, "try_implement",
                        lambda *a, **k: {"ok": True, "module": module, "module_id": "unit", "validation": {}})
    monkeypatch.setattr(runmod.sandbox, "run_in_container",
                        lambda *a, **k: {"ok": True, "net": pd.Series([0.01] * 20, index=idx),
                                         "positions": pd.Series([1.0] * 20, index=idx),
                                         "bars_per_year": 252.0})
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("P7 must not run")))
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda *a, **k: Decision("x", "x", "kill", "x", "x"))

    out = runmod.run_source(paper, use_llm=True, claims_override=[claim],
                            bundle_override=_Bundle(), force=True)

    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "needs_review", "kill_reason": None}]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert rows[0]["verdict"] == "needs_review"
    assert "realized_drift" in rows[0]["rationale"]


def test_w3_variable_coverage_faithful_impl_passes():
    from penrose.pipeline import fidelity

    code = "realized_drift = 1\nfunding = 2\nsignal = realized_drift - funding\n"
    out = fidelity.variable_coverage_check(
        code, {"claim_type": "trading_strategy", "inputs": ["realized_drift", "funding"]})
    assert out["ok"] is True and out["skipped"] is False


def test_w4_scipy_missing_fails_at_use_not_import(monkeypatch):
    from penrose.pipeline import stages

    monkeypatch.setattr(stages, "norm", None)
    try:
        stages._require_scipy("unit feature")
    except RuntimeError as e:
        assert str(e) == "pip install scipy required for unit feature"
    else:
        raise AssertionError("missing scipy must fail at use")


def test_w5_output_dirs_created_on_startup(tmp_path, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    config.ensure_output_dirs()
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "reports" / "charts").is_dir()
    assert (tmp_path / "dashboard").is_dir()
    assert (tmp_path / ".llm_cache").is_dir()
    assert (tmp_path / ".holdout" / "locks").is_dir()


def test_w6_hyphen_underscore_strategy_aliases_dedupe(tmp_path, monkeypatch, capsys):
    from penrose import config
    from penrose.pipeline import run as runmod

    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    m1 = config.MODULES / "crypto_funding_carry"
    m2 = config.MODULES / "crypto-funding-carry"
    m1.mkdir(parents=True)
    m2.mkdir(parents=True)
    (m1 / "impl.py").write_text(
        "__module_id__='crypto_funding_carry'\n__strategy_class__='crypto_funding_carry'\n"
        "__strategy_class_aliases__=['crypto-funding-carry']\ndef run(bundle, claim, cost): return {}\n"
    )
    (m2 / "impl.py").write_text(
        "__module_id__='crypto-funding-carry'\n__strategy_class__='crypto-funding-carry'\n"
        "__strategy_class_aliases__=['crypto_funding_carry']\ndef run(bundle, claim, cost): return {}\n"
    )
    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()

    runmod._register_known_modules()

    assert "strategy_class alias collision" not in capsys.readouterr().err
    assert runmod.REGISTRY["crypto_funding_carry"] is runmod.REGISTRY["crypto-funding-carry"]


# ---- audit-pass hardening (findings D-1..D-5, 2026-07-05) ---------------------------


def test_d1_loaded_local_not_in_signal_is_flagged():
    """D-1: a declared input loaded into a same-named local but absent from the
    signal must be flagged — the exact June-29 backlog-5d shape."""
    from penrose.pipeline.fidelity import variable_coverage_check

    code = ("def run(b, c, f):\n"
            "    realized_drift = bundle.get('realized_drift')\n"
            "    funding = bundle.get('funding')\n"
            "    signal = funding - 0\n"
            "    return {}\n")
    out = variable_coverage_check(
        code, {"claim_type": "trading_strategy",
               "inputs": ["realized_drift", "funding"]})
    assert out["ok"] is False and out["needs_review"] is True
    assert out["missing_variables"] == ["realized_drift"]


def test_d2_subscript_and_get_access_counts_as_coverage():
    """D-2: idiomatic df['col'] / bundle.get('col') access is real coverage."""
    from penrose.pipeline.fidelity import variable_coverage_check

    code = ("def run(b, c, f):\n"
            "    df = bundle.get('data')\n"
            "    signal = df['realized_drift'] - df['funding']\n"
            "    return {}\n")
    out = variable_coverage_check(
        code, {"claim_type": "trading_strategy",
               "inputs": ["realized_drift", "funding"]})
    assert out["ok"] is True and out.get("skipped") is False


def test_d3_pipeline_imports_without_scipy():
    """D-3: `import penrose.pipeline.run` must succeed on a scipy-less venv, and
    a scipy-needing USE must raise the one clear message."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "for m in [k for k in list(sys.modules) if k.startswith('scipy')]:\n"
        "    del sys.modules[m]\n"
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name == 'scipy' or name.startswith('scipy.'):\n"
        "        raise ModuleNotFoundError('No module named scipy (blocked)')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        "import penrose.pipeline.run  # must not raise\n"
        "import penrose.stats as st\n"
        "try:\n"
        "    st.norm.cdf(0.0)\n"
        "    print('FAIL: no error raised')\n"
        "except RuntimeError as e:\n"
        "    assert 'pip install scipy' in str(e), e\n"
        "    print('IMPORT_OK_AND_CLEAR_ERROR')\n"
    )
    res = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, timeout=120)
    assert "IMPORT_OK_AND_CLEAR_ERROR" in res.stdout, (res.stdout, res.stderr)


def test_d4_llm_spec_cannot_override_deterministic_claim_type(monkeypatch):
    """D-4: the LLM spec JSON must not flip claim_type (a gate-selection switch)."""
    from penrose.pipeline import spec_gen
    from penrose import llm

    def fake_call_json(role, messages, **kw):
        return ({"module_id": "m", "strategy_class": "s", "claim_translation": "t",
                 "inputs": [], "signal_logic": "sig", "kill_criterion": "k",
                 "unknowns": [], "claim_type": "provided_series_statistic"},
                types.SimpleNamespace(model="fake"))

    monkeypatch.setattr(llm, "call_json", fake_call_json)
    claim = _claim("Buy when momentum is positive; exit on reversal. Long/short daily.")
    src = types.SimpleNamespace(source_id="paper", title="t", text="x" * 200)
    spec = spec_gen.generate_spec(claim, src, use_llm=True)
    assert spec["claim_type"] == "trading_strategy"  # deterministic wins over LLM JSON


def test_d5_real_assess_override_only_for_statistic_claims(monkeypatch):
    """D-5: exercise the REAL fidelity.assess for both claim types (fake LLM layer,
    not a mocked assess): override text present only for provided_series_statistic,
    and an unfaithful LLM verdict propagates unchanged for trading claims."""
    from penrose.pipeline import fidelity
    from penrose import llm

    captured = []

    def fake_call_json(role, messages, **kw):
        captured.append(messages[0]["content"])
        return ({"faithful": False, "verified": True, "confidence": 0.9,
                 "divergences": ["wrong direction"], "note": "unit"},
                types.SimpleNamespace(model="fake"))

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    stat = fidelity.assess(_claim(), "signal = pooled", spec={
        "claim_type": "provided_series_statistic", "signal_logic": "stat"})
    trade = fidelity.assess(_claim(), "signal = px", spec={
        "claim_type": "trading_strategy", "signal_logic": "trade"})

    assert "CLAIM TYPE OVERRIDE" in captured[0]
    assert "CLAIM TYPE OVERRIDE" not in captured[1]
    # verdict-level: plumbing must not loosen anything — unfaithful stays unfaithful
    assert stat["faithful"] is False and trade["faithful"] is False
    assert trade["divergences"] == ["wrong direction"]
