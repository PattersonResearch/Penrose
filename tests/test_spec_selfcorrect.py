import json
import types


def _claim():
    from penrose.brain import Claim

    statement = "Go long BTC perps when funding is negative; exit when funding turns positive."
    return Claim(
        claim_id="paper-c1",
        statement=statement,
        mechanism="unit",
        scope="BTC perpetuals",
        horizon="1d",
        source_id="paper",
        source_span=statement,
        claimed_metric_quote="positive DSR",
        applicable_strategy_class="funding_drift",
    )


def _source():
    from penrose.pipeline.p1_ingest import IngestedSource

    return IngestedSource(
        source_id="paper", title="paper", text="unit", n_pages=1, n_chars=4,
        text_sha256="abc", injection_flags=[])


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
    monkeypatch.setattr(runmod, "write_report", lambda *a, **k: tmp_path / "report.md")
    monkeypatch.setattr(runmod, "_write_live", lambda *a, **k: None)


def _spec(n, tmp_path):
    return {
        "module_id": f"spec_{n}",
        "strategy_class": "funding_drift",
        "claim_type": "trading_strategy",
        "claim_translation": f"translation {n}",
        "inputs": ["funding"],
        "signal_logic": f"signal {n}",
        "kill_criterion": "claim's own decision rule fails",
        "unknowns": [],
        "_path": str(tmp_path / f"spec_{n}.yaml"),
    }


def _unfaithful(divergences):
    return {
        "faithful": False,
        "verified": True,
        "checked": True,
        "confidence": 0.91,
        "divergences": divergences,
        "note": "; ".join(divergences),
    }


def test_spec_self_correction_retries_then_succeeds(tmp_path, monkeypatch):
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "paper.md"
    paper.write_text("paper")
    divergences = ["converted absolute funding sign into cross-sectional ranking"]
    calls = []
    fids = [
        _unfaithful(divergences),
        {"faithful": True, "verified": True, "checked": True, "confidence": 0.95,
         "divergences": [], "note": "faithful"},
    ]

    def fake_generate_spec(claim, source, *, use_llm=True, prior_divergences=None):
        calls.append({"use_llm": use_llm, "prior_divergences": prior_divergences})
        return _spec(len(calls), tmp_path)

    monkeypatch.setattr(runmod.spec_gen, "generate_spec", fake_generate_spec)
    monkeypatch.setattr(runmod, "_assess_spec_fidelity_safe",
                        lambda claim, spec: fids.pop(0))

    out = runmod.run_source(paper, use_llm=True, claims_override=[_claim()],
                            bundle_override=_Bundle(), force=True)

    assert len(calls) == 2
    assert calls[1]["prior_divergences"] == divergences
    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "pending_module",
                                 "kill_reason": None}]
    attempts = out["claims"][0]["stages"]["P6_pre_fidelity_attempts"]
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["blocked"] is True
    assert attempts[0]["divergences"] == divergences
    assert attempts[1]["blocked"] is False


def test_spec_self_correction_exhausts_bounded_attempts(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "paper.md"
    paper.write_text("paper")
    calls = []
    divergences = ["left sign convention unresolved"]

    def fake_generate_spec(claim, source, *, use_llm=True, prior_divergences=None):
        calls.append(prior_divergences)
        return _spec(len(calls), tmp_path)

    monkeypatch.setattr(runmod.spec_gen, "generate_spec", fake_generate_spec)
    monkeypatch.setattr(runmod, "_assess_spec_fidelity_safe",
                        lambda claim, spec: _unfaithful(divergences))

    out = runmod.run_source(paper, use_llm=True, claims_override=[_claim()],
                            bundle_override=_Bundle(), force=True)

    assert len(calls) == runmod.SPEC_SELF_CORRECTION_MAX_ATTEMPTS
    assert out["decisions"] == [{"claim_id": "paper-c1", "verdict": "cannot_replicate",
                                 "kill_reason": "unfaithful_spec"}]
    assert len([c for c in out["claims"] if c["claim_id"] == "paper-c1"]) == 1
    attempts = out["claims"][0]["stages"]["P6_pre_fidelity_attempts"]
    assert len(attempts) == runmod.SPEC_SELF_CORRECTION_MAX_ATTEMPTS
    assert all(a["blocked"] is True for a in attempts)

    rows = [json.loads(line) for line in config.FIDELITY_REJECTIONS.read_text().splitlines()
            if line.strip()]
    assert len(rows) == runmod.SPEC_SELF_CORRECTION_MAX_ATTEMPTS
    assert rows[-1]["divergences"] == divergences


def test_generate_spec_threads_prior_divergences_into_prompt(tmp_path, monkeypatch):
    from penrose import config, llm
    from penrose.pipeline import spec_gen

    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    monkeypatch.setattr(spec_gen, "_catalog_vocab", lambda: "")
    monkeypatch.setattr(spec_gen.fidelity_memory, "rejection_guidance", lambda *a, **k: "")
    captured = {}

    def fake_call_json(role, messages, **kwargs):
        captured["user"] = messages[1]["content"]
        return ({
            "module_id": "funding_drift",
            "strategy_class": "funding_drift",
            "claim_translation": "translation",
            "inputs": ["funding"],
            "signal_logic": "long when funding is negative",
            "kill_criterion": "claim's own decision rule fails",
            "unknowns": [],
        }, types.SimpleNamespace(model="fake"))

    monkeypatch.setattr(llm, "call_json", fake_call_json)
    divergences = [
        "cross-sectional mean-subtraction changes the signal",
        "emitted a menu of exit rules",
    ]

    spec_gen.generate_spec(_claim(), _source(), use_llm=True,
                           prior_divergences=divergences)

    prompt = captured["user"]
    assert "YOUR PREVIOUS SPEC FOR THIS CLAIM WAS REJECTED AS UNFAITHFUL" in prompt
    assert divergences[0] in prompt
    assert divergences[1] in prompt
    assert "Do NOT add any significance test" in prompt
    assert "Never" in prompt and "menu of alternatives" in prompt
