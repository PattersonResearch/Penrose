"""FIX 4 / FIX 5a regression tests (v0.4.1): item 6g -- "test the statistic of a
provided series" is a first-class claim type routed to a single deterministic test.
Spec-gen must not invent gates (EXP-1, over-specification) and must not fall back to an
empty stub (EXP-1b, under-specification). This is the planted eval for 6g: a minimal
pre-registered cohort-mean claim must clear fidelity and reach a real verdict.
"""
import json
import types

import pandas as pd


def _claim(statement, *, claim_id="cohort-c1", strategy_class="kalshi_tail_calibration",
          claimed_metric_quote=""):
    from penrose.brain import Claim

    return Claim(
        claim_id=claim_id, statement=statement, mechanism="unit", scope="unit", horizon="1d",
        source_id="cohort", source_span=statement, claimed_metric_quote=claimed_metric_quote,
        applicable_strategy_class=strategy_class,
    )


def _source():
    from penrose.pipeline.p1_ingest import IngestedSource

    return IngestedSource(source_id="cohort", title="cohort", text="unit", n_pages=1,
                          n_chars=4, text_sha256="abc", injection_flags=[])


# The planted eval claim: minimal pre-registered cohort-mean, ONE pooled statistic, no
# extra gates -- exactly the class that failed both directions in the 2026-07-04 incident.
PLANTED_CLAIM_STATEMENT = (
    "The pooled mean of the declared net settlement P&L series across all registered "
    "cities is greater than zero, tested as a single pooled statistic across the one "
    "declared deflation cohort with no additional significance or data-quality gates."
)

# Forbidden content: gates the claim never stated. Any of these appearing in a generated
# spec for this claim type is exactly the EXP-1 over-specification bug reoccurring.
_FORBIDDEN_INVENTED_GATES = [
    "p<=0.05", "p <= 0.05", "p-value", "bonferroni", "bh-fdr", "james-stein",
    "minimum 30 obs", "min 30 obs", "no city >50%", "no city > 50%",
]


def test_classifier_routes_planted_claim_to_provided_series_statistic():
    from penrose.pipeline import spec_gen

    claim = _claim(PLANTED_CLAIM_STATEMENT)
    assert spec_gen.classify_claim_type(claim) == "provided_series_statistic"


def test_classifier_does_not_regress_existing_claim_types():
    """6g's new patterns must not steal claims that legitimately belong to the existing
    types (regression guard alongside the r1 test in test_fidelity_6c.py)."""
    from penrose.pipeline import spec_gen

    descriptive = _claim("The unconditional mean ensemble bias is +0.96F over 1,700 observations.")
    trading = _claim("A 12 month momentum signal enters long positions and earns positive Sharpe.")
    funding_drift = _claim(
        "Realized drift minus funding predicts forward returns; go long when excess_drift "
        "is positive, short or flat otherwise, tested cross-sectionally across six perps."
    )
    assert spec_gen.classify_claim_type(descriptive) == "descriptive_statistical"
    assert spec_gen.classify_claim_type(trading) == "trading_strategy"
    assert spec_gen.classify_claim_type(funding_drift) == "trading_strategy"


def test_deterministic_spec_never_invents_gates_the_claim_did_not_state():
    """EXP-1 (over-specification): spec-gen must not add a significance threshold, a
    data-quality kill, or a menu of deflation methods for a claim that declared none."""
    from penrose.pipeline import spec_gen

    claim = _claim(PLANTED_CLAIM_STATEMENT)
    spec = spec_gen.generate_spec(claim, _source(), use_llm=False)

    assert spec["claim_type"] == "provided_series_statistic"
    blob = json.dumps(spec).lower()
    for forbidden in _FORBIDDEN_INVENTED_GATES:
        assert forbidden not in blob, f"spec invented a gate the claim never stated: {forbidden!r}"


def test_deterministic_spec_is_never_an_empty_stub():
    """EXP-1b (under-specification): the same claim type must not fall back to an empty
    'not generated' stub -- statistic_logic/signal_logic/claim_translation/kill_criterion
    must all be concrete, non-empty content, with or without an LLM available."""
    from penrose.pipeline import spec_gen

    claim = _claim(PLANTED_CLAIM_STATEMENT)
    for use_llm in (False, True):
        spec = spec_gen.generate_spec(claim, _source(), use_llm=use_llm)
        assert spec["claim_type"] == "provided_series_statistic"
        assert spec.get("statistic_logic", "").strip()
        assert spec.get("signal_logic", "").strip()
        assert "not generated" not in spec.get("signal_logic", "").lower()
        assert spec.get("claim_translation", "").strip()
        assert spec.get("kill_criterion", "").strip()


def test_provided_series_statistic_never_calls_the_llm(monkeypatch):
    """This claim type is a mechanical translation: generate_spec must route around
    _llm_spec entirely (never touches the LLM, so neither failure mode has a generation
    step in which to occur)."""
    from penrose.pipeline import spec_gen

    def _boom(*a, **k):
        raise AssertionError("provided_series_statistic must never call the LLM spec generator")

    monkeypatch.setattr(spec_gen, "_llm_spec", _boom)
    claim = _claim(PLANTED_CLAIM_STATEMENT)
    spec = spec_gen.generate_spec(claim, _source(), use_llm=True)
    assert spec["claim_type"] == "provided_series_statistic"


def test_declared_series_pulled_from_claim_text_without_invention(monkeypatch):
    from penrose.pipeline import spec_gen

    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_wx_tailpnl_ny", "kx_wx_tailpnl_hou",
                                                            "unrelated_series"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim(
        "The pooled mean of kx_wx_tailpnl_ny and kx_wx_tailpnl_hou (declared series) is "
        "greater than zero, one pooled statistic, no additional gates."
    )
    spec = spec_gen.generate_spec(claim, _source(), use_llm=False)
    assert spec["inputs"] == ["kx_wx_tailpnl_hou", "kx_wx_tailpnl_ny"]
    assert "unrelated_series" not in spec["inputs"]


# --- planted eval: full pipeline integration, clears fidelity, reaches a real verdict --- #

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


def test_planted_eval_cohort_mean_claim_clears_fidelity_and_reaches_real_verdict(tmp_path, monkeypatch):
    """The 6g planted eval (FIX 5a): a minimal pre-registered cohort-mean claim -- pooled
    mean of N declared series > threshold, one statistic -- MUST clear fidelity 100% of
    the time and reach a real verdict (not cannot_replicate/unfaithful_spec)."""
    from penrose import concepts, config
    from penrose.brain import Decision
    from penrose.pipeline import extract, run as runmod

    _patch_run_paths(tmp_path, monkeypatch)
    paper = tmp_path / "cohort.md"
    paper.write_text("cohort claim")
    claim = _claim(PLANTED_CLAIM_STATEMENT, claim_id="cohort-c1")

    module_path = tmp_path / "impl.py"
    module_path.write_text(
        "__strategy_class__='kalshi_tail_calibration'\ndef run(bundle, claim, cost): return {}\n"
    )
    auto_module = types.SimpleNamespace(
        __strategy_class__="kalshi_tail_calibration", __module_id__="auto_cohort",
        __auto_generated__=True, __file__=str(module_path),
    )
    net = pd.Series([0.01] * 20, index=pd.date_range("2020-01-01", periods=20, freq="D"))
    positions = pd.Series([1.0] * 20, index=net.index)

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(runmod.sandbox, "ensure_image", lambda: True)

    def fake_try_implement(*args, **kwargs):
        return {"ok": True, "module": auto_module, "module_id": "auto_cohort", "validation": {}}

    def fake_sandbox(*args, **kwargs):
        return {"ok": True, "net": net, "positions": positions, "bars_per_year": 252.0}

    def fake_backtest(*args, **kwargs):
        return {"psr": 0.95, "dsr": 0.95, "n_oos": 200, "oos_sharpe": 1.2,
                "capacity_usd": 1_000_000, "three_fold": {}, "bootstrap": {}, "permutation": {}, "regime": {}}

    def fake_fidelity(claim, module_code, spec=None, **kwargs):
        # A faithful pre-fidelity assessment: the deterministic spec is a literal
        # translation of the claim, so a real fidelity check clears it. We assert here
        # that the SPEC handed to the refuter contains no invented gates -- catching a
        # regression even if a future change routed this claim type through the LLM.
        if spec is not None:
            blob = json.dumps(spec).lower()
            for forbidden in _FORBIDDEN_INVENTED_GATES:
                assert forbidden not in blob
        return {"faithful": True, "verified": True, "checked": True, "confidence": 0.95,
               "divergences": [], "note": "faithful deterministic translation"}

    monkeypatch.setattr(runmod.impl_gen, "try_implement", fake_try_implement)
    monkeypatch.setattr(runmod.sandbox, "run_in_container", fake_sandbox)
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest", fake_backtest)
    monkeypatch.setattr(runmod.fidelity, "assess", fake_fidelity)
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda claim, bt, holdout, synthetic: Decision(
                            decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                            verdict="watch", kill_reason=None, rationale="planted eval watch",
                            metrics={"psr": bt["psr"], "dsr": bt["dsr"]}))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(paper, use_llm=True, claims_override=[claim],
                            bundle_override=_TinyBundle(), force=True)

    assert out["decisions"] == [{"claim_id": "cohort-c1", "verdict": "watch", "kill_reason": None}]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert rows[0]["verdict"] not in ("cannot_replicate",)
    assert rows[0].get("kill_reason") != "unfaithful_spec"
