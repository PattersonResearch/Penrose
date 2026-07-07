import json


def _claim():
    from penrose.brain import Claim

    statement = "Delta-neutral carry works across BTC and BNB using spot and perp prices."
    return Claim(
        claim_id="paper-c1",
        statement=statement,
        mechanism="carry",
        scope="crypto spot and perpetual futures",
        horizon="daily",
        source_id="paper",
        source_span=statement,
        claimed_metric_quote="positive DSR",
        applicable_strategy_class="carry_unit",
    )


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


def _isolate(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import extract, run as runmod

    monkeypatch.setattr(config, "ROOT", tmp_path)
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
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(config, "FIDELITY_REJECTIONS", tmp_path / "reports" / "fidelity_rejections.jsonl")
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", False)
    monkeypatch.setattr(config, "FIDELITY_CHECK", True)
    (tmp_path / "modules").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runmod, "REGISTRY", {})
    monkeypatch.setattr(runmod, "_REGISTRY_ALIAS_OWNERS", {})
    monkeypatch.setattr(runmod, "_REGISTRY_CANONICAL_OWNERS", {})
    monkeypatch.setattr(runmod, "_REGISTRY_CANONICAL_MODULES", {})
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.stages, "propose_principle", lambda decisions: None)
    monkeypatch.setattr(runmod, "write_report", lambda *a, **k: tmp_path / "report.md")
    monkeypatch.setattr(runmod, "_write_live", lambda *a, **k: None)


def _spec(tmp_path, inputs):
    return {
        "module_id": "carry_unit",
        "strategy_class": "carry_unit",
        "claim_type": "trading_strategy",
        "claim_translation": "Test the stated delta-neutral carry rule.",
        "inputs": list(inputs),
        "signal_logic": "Use declared inputs exactly.",
        "kill_criterion": "claim's own decision rule fails",
        "unknowns": [],
        "_path": str(tmp_path / "carry_unit.yaml"),
    }


def _run_with_spec(tmp_path, monkeypatch, spec):
    from penrose.pipeline import run as runmod

    paper = tmp_path / "paper.md"
    paper.write_text("paper")
    monkeypatch.setattr(runmod.spec_gen, "generate_spec", lambda *a, **k: dict(spec))
    return runmod.run_source(
        paper, use_llm=True, claims_override=[_claim()],
        bundle_override=_Bundle(), force=True)


def test_pre_fidelity_missing_spec_input_routes_needs_data(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    spec = _spec(tmp_path, ["btc_spot_daily", "bnb_spot_daily"])
    monkeypatch.setattr(runmod, "_missing_spec_inputs_from_bundle",
                        lambda spec, bundle: ["bnb_spot_daily"])
    monkeypatch.setattr(runmod, "_assess_spec_fidelity_safe",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("fidelity loop must be skipped")))

    out = _run_with_spec(tmp_path, monkeypatch, spec)

    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "needs_data",
                                 "kill_reason": None}]
    assert out["claims"][0]["stages"]["P8"]["missing_series"] == ["bnb_spot_daily"]
    assert "P6_pre_fidelity_attempts" not in out["claims"][0]["stages"]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines()
            if line.strip()]
    assert rows[-1]["verdict"] == "needs_data"
    assert rows[-1]["metrics"]["missing_series"] == ["bnb_spot_daily"]
    assert rows[-1]["kill_reason"] is None
    assert "cannot_replicate" not in {row["verdict"] for row in rows}


def test_pre_fidelity_available_spec_inputs_do_not_route_needs_data(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    spec = _spec(tmp_path, ["btc_spot_daily", "eth_spot_daily"])
    fidelity_calls = []
    monkeypatch.setattr(runmod, "_missing_spec_inputs_from_bundle", lambda spec, bundle: [])
    monkeypatch.setattr(runmod, "_assess_spec_fidelity_safe",
                        lambda claim, spec: fidelity_calls.append(spec) or {
                            "faithful": True, "verified": True, "checked": True,
                            "confidence": 0.95, "divergences": [], "note": "faithful"})

    out = _run_with_spec(tmp_path, monkeypatch, spec)

    assert len(fidelity_calls) == 1
    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "pending_module",
                                 "kill_reason": None}]
    assert not config.DATA_REQUESTS.exists()


def test_pre_fidelity_availability_error_fails_open(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    spec = _spec(tmp_path, ["bnb_spot_daily"])
    fidelity_calls = []

    def unavailable_oracle(spec, bundle):
        raise RuntimeError("availability unavailable")

    monkeypatch.setattr(runmod, "_missing_spec_inputs_from_bundle", unavailable_oracle)
    monkeypatch.setattr(runmod, "_assess_spec_fidelity_safe",
                        lambda claim, spec: fidelity_calls.append(spec) or {
                            "faithful": True, "verified": True, "checked": True,
                            "confidence": 0.95, "divergences": [], "note": "faithful"})

    out = _run_with_spec(tmp_path, monkeypatch, spec)

    assert len(fidelity_calls) == 1
    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "pending_module",
                                 "kill_reason": None}]
    assert not config.DATA_REQUESTS.exists()
