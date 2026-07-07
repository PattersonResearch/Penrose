"""FIX 4 / FIX 5a regression tests (v0.4.1): item 6g -- "test the statistic of a
provided series" is a first-class claim type routed to a single deterministic test.
Spec-gen must not invent gates (EXP-1, over-specification) and must not fall back to an
empty stub (EXP-1b, under-specification). This is the planted eval for 6g: a minimal
pre-registered cohort-mean claim must clear fidelity and reach a real verdict.
"""
import json
import os
import tempfile
import types

import numpy as np
import pandas as pd


def _claim(statement, *, claim_id="cohort-c1", strategy_class="kalshi_tail_calibration",
          claimed_metric_quote="", source_span=None):
    from penrose.brain import Claim

    return Claim(
        claim_id=claim_id, statement=statement, mechanism="unit", scope="unit", horizon="1d",
        source_id="cohort", source_span=source_span if source_span is not None else statement,
        claimed_metric_quote=claimed_metric_quote,
        applicable_strategy_class=strategy_class,
    )


def _source(text="unit"):
    from penrose.pipeline.p1_ingest import IngestedSource

    return IngestedSource(source_id="cohort", title="cohort", text=text, n_pages=1,
                          n_chars=len(text), text_sha256="abc", injection_flags=[])


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


def test_classifier_requires_provided_declaration_for_weak_statistic_patterns():
    from penrose.pipeline import spec_gen

    source_excerpt = (
        "Pool these 8 net-tail-P&L series (each already encodes the per-market net P&L "
        "above, held to settlement) into a pooled sample and run a single one-sided "
        "one-sample t-test on that pooled sample."
    )
    provided = _claim(
        "A strategy that buys the mispriced side of every settled tail market "
        "(YES when p<=0.10, NO when p>=0.90) and holds to settlement earns a "
        "positive mean net P&L after the fee",
        source_span="single one-sided one-sample t-test on daily net P&L",
        claimed_metric_quote="single one-sided one-sample t-test on daily net P&L",
    )
    long_short = _claim(
        "A long-short momentum strategy earns positive returns. We validate with a "
        "one-sample t-test on daily P&L.",
        strategy_class="momentum",
    )
    carry = _claim(
        "A carry strategy earns a pooled mean spread of 3bps per day.",
        strategy_class="carry",
    )

    assert spec_gen.classify_claim_type(provided, _source(source_excerpt)) == (
        "provided_series_statistic"
    )
    assert spec_gen.classify_claim_type(long_short) == "trading_strategy"
    assert spec_gen.classify_claim_type(carry) == "trading_strategy"


def test_exp1b_tail_market_claim_stays_provided_series_statistic():
    from penrose.pipeline import spec_gen

    statement = (
        "A strategy that buys the mispriced side of every settled tail market "
        "(YES when p<=0.10, NO when p>=0.90) and holds to settlement earns a "
        "positive mean net P&L after the fee"
    )
    source_excerpt = (
        "Pool these 8 net-tail-P&L series (each already encodes the per-market net P&L "
        "above, held to settlement) into a pooled sample and run a single one-sided "
        "one-sample t-test on that pooled sample."
    )

    assert spec_gen.classify_claim_type(_claim(statement), _source(source_excerpt)) == (
        "provided_series_statistic"
    )


def test_ambiguous_strong_stat_phrases_do_not_steal_trading_claims():
    from penrose.pipeline import spec_gen

    cohort_mean = _claim(
        "A long-short BTC momentum strategy works; we validate the edge with a "
        "cohort-mean spread of small-cap tokens daily.",
        strategy_class="momentum",
    )
    pooled_stat = _claim(
        "A long-only equity carry strategy works; we confirm with a pooled statistic "
        "of net P&L.",
        strategy_class="carry",
    )

    assert spec_gen.classify_claim_type(cohort_mean) == "trading_strategy"
    assert spec_gen.classify_claim_type(pooled_stat) == "trading_strategy"


def test_incidental_declaration_prose_does_not_flip_trading_claims():
    from penrose.pipeline import spec_gen

    claim = _claim(
        "A long-short BTC momentum signal is validated with a one-sample t-test on "
        "daily P&L.",
        strategy_class="momentum",
    )

    assert spec_gen.classify_claim_type(
        claim,
        _source("These series of trades are pooled to compute Sharpe."),
    ) == "trading_strategy"
    assert spec_gen.classify_claim_type(
        claim,
        _source("Each signal encodes information about future returns."),
    ) == "trading_strategy"


def test_trading_veto_blocks_weak_stat_with_non_encoded_series_declaration():
    from penrose.pipeline import spec_gen

    claim = _claim(
        "A long-short BTC momentum signal enters on breakout and exits on reversal; "
        "we validate the edge with a pooled mean.",
        strategy_class="momentum",
    )

    assert spec_gen.classify_claim_type(
        claim,
        _source("Pool these 8 momentum series into a pooled sample."),
    ) == "trading_strategy"


def test_classifier_strong_provided_series_pattern_needs_no_extra_declaration():
    from penrose.pipeline import spec_gen

    claim = _claim("Evaluate the registered claims as one declared deflation cohort.")

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


def test_provided_series_fidelity_structural_override_for_deterministic_template(monkeypatch):
    from penrose import llm
    from penrose.pipeline import fidelity

    def fake_call_json(role, messages, **kw):
        return ({"faithful": False, "confidence": 0.70,
                 "divergences": [
                     "the spec reduces a claim whose series must be constructed from a "
                     "stated entry rule and exact fee to an opaque one-sample mean test, "
                     "omitting the P&L/fee logic"
                 ],
                 "note": "construction unverifiable"},
                types.SimpleNamespace(independent_verifier=False))

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = fidelity.assess(_claim(PLANTED_CLAIM_STATEMENT), "{}", spec={
        "claim_type": "provided_series_statistic",
        "module_spec_only": True,
        "inputs": ["kx_tail_a"],
        "_llm_mode": "deterministic-template",
    })

    assert out["faithful"] is True
    assert out["verified"] is True
    assert out["provided_series_fidelity_override"] == "deterministic_template_structural"
    assert "overridden structurally" in out["note"]


def test_provided_series_fidelity_structural_override_fails_safe_without_template(monkeypatch):
    from penrose import llm
    from penrose.pipeline import fidelity

    def fake_call_json(role, messages, **kw):
        return ({"faithful": False, "confidence": 0.90,
                 "divergences": ["spec pools an extra undeclared series and adds a p<=0.05 gate"],
                 "note": "real W1 divergence"},
                types.SimpleNamespace(independent_verifier=False))

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = fidelity.assess(_claim(PLANTED_CLAIM_STATEMENT), "{}", spec={
        "claim_type": "provided_series_statistic",
        "module_spec_only": True,
        "inputs": [],
        "_llm_mode": "deterministic-template",
    })

    assert out["faithful"] is False
    assert out["verified"] is False
    assert "provided_series_fidelity_override" not in out


def test_provided_series_fidelity_structural_override_fails_safe_with_gate_field(monkeypatch):
    from penrose import llm
    from penrose.pipeline import fidelity

    def fake_call_json(role, messages, **kw):
        return ({"faithful": False, "confidence": 0.90,
                 "divergences": ["construction is unverifiable"],
                 "note": "blocked"},
                types.SimpleNamespace(independent_verifier=False))

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = fidelity.assess(_claim(PLANTED_CLAIM_STATEMENT), "{}", spec={
        "claim_type": "provided_series_statistic",
        "module_spec_only": True,
        "inputs": ["kx_tail_a"],
        "_llm_mode": "deterministic-template",
        "significance_gate": "p <= 0.05",
    })

    assert out["faithful"] is False
    assert "provided_series_fidelity_override" not in out


def test_deterministic_provided_series_structure_rejects_gate_values_and_abbrev_keys():
    from penrose.pipeline import fidelity

    clean = {
        "claim_type": "provided_series_statistic",
        "module_spec_only": True,
        "inputs": ["kx_tail_a"],
        "_llm_mode": "deterministic-template",
        "claim_statement": (
            "A strategy buys YES when p<=0.10 and NO when p>=0.90, using already "
            "encoded net P&L series."
        ),
        "claim_translation": "pool declared series and apply the stated decision rule",
    }

    assert fidelity._is_deterministic_provided_series_spec(clean) is True
    assert fidelity._is_deterministic_provided_series_spec({
        **clean,
        "rejection_rule": "p<0.05",
    }) is False
    assert fidelity._is_deterministic_provided_series_spec({
        **clean,
        "min_obs": 100,
    }) is False


def test_provided_series_fidelity_backstop_does_not_affect_trading_claims(monkeypatch):
    from penrose import llm
    from penrose.pipeline import fidelity

    def fake_call_json(role, messages, **kw):
        return ({"faithful": False, "confidence": 0.90,
                 "divergences": ["construction is unverifiable and P&L/fee logic is omitted"],
                 "note": "construction objection"},
                types.SimpleNamespace(independent_verifier=False))

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = fidelity.assess(
        _claim("A 12 month momentum signal enters long positions and earns positive Sharpe."),
        "def run(bundle, claim, cost): return {}",
        spec={"claim_type": "trading_strategy"},
    )

    assert out["faithful"] is False
    assert "provided_series_fidelity_override" not in out


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


def test_declared_series_scans_source_and_excludes_context_units(monkeypatch):
    from penrose.pipeline import spec_gen

    decision_series = [
        "kx_wx_tailpnl_atl",
        "kx_wx_tailpnl_bos",
        "kx_wx_tailpnl_chi",
        "kx_wx_tailpnl_dal",
        "kx_wx_tailpnl_den",
        "kx_wx_tailpnl_lax",
        "kx_wx_tailpnl_mia",
        "kx_wx_tailpnl_sea",
    ]
    context_series = ["kx_wx_abserr_ny", "kx_wx_abserr_hou"]
    fake_catalog = types.SimpleNamespace(available=lambda: decision_series + context_series)
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    source = _source(
        "## Data\n\n"
        f"Pool these series: {' '.join(decision_series)}.\n\n"
        f"Context (reported, not part of the decision): {' '.join(context_series)}.\n"
    )

    claim = _claim(PLANTED_CLAIM_STATEMENT)

    assert spec_gen._declared_series_from_text(claim, source) == sorted(decision_series)


def test_declared_series_excludes_benchmark_comparison_units(monkeypatch):
    from penrose.pipeline import spec_gen

    fake_catalog = types.SimpleNamespace(available=lambda: ["decision_tail_pnl", "benchmark_tail_pnl"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    source = _source(
        "Decision series: decision_tail_pnl.\n\n"
        "Benchmark: benchmark_tail_pnl for comparison."
    )

    assert spec_gen._declared_series_from_text(_claim(PLANTED_CLAIM_STATEMENT), source) == [
        "decision_tail_pnl"
    ]


def test_declared_series_excludes_new_context_cue_units(monkeypatch):
    from penrose.pipeline import spec_gen

    fake_catalog = types.SimpleNamespace(available=lambda: ["decision_tail_pnl", "context_tail_pnl"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    source = _source(
        "Decision series: decision_tail_pnl.\n\n"
        "We also report context_tail_pnl as reference; it is illustrative."
    )

    assert spec_gen._declared_series_from_text(_claim(PLANTED_CLAIM_STATEMENT), source) == [
        "decision_tail_pnl"
    ]


def test_declared_series_short_keys_match_only_whole_tokens(monkeypatch):
    from penrose.pipeline import spec_gen

    fake_catalog = types.SimpleNamespace(available=lambda: ["rv"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim("The served sample is live and does not name the short catalog key.")

    assert spec_gen._declared_series_from_text(claim) == []


def test_declared_series_never_invents_uncataloged_names(monkeypatch):
    from penrose.pipeline import spec_gen

    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_wx_tailpnl_ny"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim(
        "The pooled mean of kx_wx_tailpnl_ny and kx_wx_tailpnl_missing is greater than zero."
    )

    assert spec_gen._declared_series_from_text(claim) == ["kx_wx_tailpnl_ny"]


def test_declared_series_fail_open_on_catalog_error(monkeypatch):
    from penrose.pipeline import spec_gen

    def _raise(_data_dir):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(spec_gen, "load_catalog_loader", _raise)
    claim = _claim("The pooled mean of kx_wx_tailpnl_ny is greater than zero.")

    assert spec_gen._declared_series_from_text(claim) == []


def test_provided_series_spec_uses_source_decision_inputs_only(monkeypatch):
    from penrose.pipeline import spec_gen

    decision_series = [
        "kx_wx_tailpnl_atl",
        "kx_wx_tailpnl_bos",
        "kx_wx_tailpnl_chi",
        "kx_wx_tailpnl_dal",
        "kx_wx_tailpnl_den",
        "kx_wx_tailpnl_lax",
        "kx_wx_tailpnl_mia",
        "kx_wx_tailpnl_sea",
    ]
    context_series = ["kx_wx_abserr_ny", "kx_wx_abserr_hou"]
    fake_catalog = types.SimpleNamespace(available=lambda: decision_series + context_series)
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    source = _source(
        "## Data\n\n"
        f"Pool these series: {' '.join(decision_series)}.\n\n"
        f"Context (reported, not part of the decision): {' '.join(context_series)}.\n"
    )

    spec = spec_gen._provided_series_stat_spec(_claim(PLANTED_CLAIM_STATEMENT), source)

    assert spec["inputs"] == sorted(decision_series)
    assert spec["unknowns"] == []


def _research_supported_bt(claim_type="trading_strategy"):
    return {
        "claim_type": claim_type,
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 3.0,
        "n_oos": 1200,
        "n_trials": 1,
        "oos_sharpe": 2.0,
        "is_sharpe": 1.5,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.1, 1.2, 1.0], "consistent": True},
        "capacity_usd": 1_000_000,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
        "walk_forward": {},
        "capacity_ci": {},
        "cost_sensitivity": {},
        "cpcv": {},
    }


def test_provided_series_verdict_caps_research_supported_to_watch(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    dec = stages.p8_verdict(
        _claim(PLANTED_CLAIM_STATEMENT),
        _research_supported_bt("provided_series_statistic"),
        {"holdout_sharpe": 1.0, "holdout_psr": config.HOLDOUT_CONFIRM_PSR},
        synthetic=False,
    )

    assert dec.verdict == "watch"
    assert dec.kill_reason is None
    assert dec.metrics["provided_series_provenance"] == "unverified_construction"
    assert dec.metrics["claim_type"] == "provided_series_statistic"
    assert "pre-computed series it did not construct" in dec.rationale


def test_provided_series_verdict_cap_does_not_change_kill(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    bt = dict(_research_supported_bt("provided_series_statistic"), psr=0.10, dsr=0.10)
    dec = stages.p8_verdict(
        _claim(PLANTED_CLAIM_STATEMENT),
        bt,
        {"holdout_sharpe": 1.0, "holdout_psr": config.HOLDOUT_CONFIRM_PSR},
        synthetic=False,
    )

    assert dec.verdict == "kill"
    assert dec.metrics["provided_series_provenance"] == "unverified_construction"
    assert "pre-computed series it did not construct" not in dec.rationale


def test_non_provided_claim_can_still_reach_research_supported(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    dec = stages.p8_verdict(
        _claim("A 12 month momentum signal enters long positions and earns positive Sharpe."),
        _research_supported_bt("trading_strategy"),
        {"holdout_sharpe": 1.0, "holdout_psr": config.HOLDOUT_CONFIRM_PSR},
        synthetic=False,
    )

    assert dec.verdict == "research-supported"
    assert dec.metrics["provided_series_provenance"] is None


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


def _contract_bundle(series: dict[str, pd.Series]):
    from penrose.data.contract import DataBundle, Series

    return DataBundle(series={
        name: Series(name=name, data=data, provenance="unit", unit="net_pnl")
        for name, data in series.items()
    })


def _patch_fast_robustness(monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "BOOTSTRAP", {"n_boot": 80, "ci": 0.90, "block": None, "seed": 0})
    monkeypatch.setattr(config, "REGIME_FRAGILITY", {"n_perm": 40, "p_kill": 0.05, "seed": 0})
    monkeypatch.setattr(config, "CPCV", {"n_groups": 6, "k_test": 2, "embargo_frac": 0.01,
                                         "max_combos": 30, "seed": 0,
                                         "overfit_prob_kill": 0.50, "min_paths": 4})


def test_provided_series_deterministic_module_pools_declared_inputs_without_resampling():
    from penrose.pipeline import impl_gen, provided_series

    idx_a = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-04", "2024-01-05"])
    idx_b = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-06", "2024-01-07"])
    bundle = _contract_bundle({
        "kx_tail_a": pd.Series([0.01, -0.02, 0.03, 0.01, -0.01], index=idx_a),
        "kx_tail_b": pd.Series([0.02, 0.04, -0.01, 0.03, 0.01], index=idx_b),
    })
    claim = _claim("The pooled mean of kx_tail_a and kx_tail_b provided series is greater than zero.")
    spec = {"module_id": "auto_pool", "strategy_class": "kalshi_tail_calibration",
            "claim_type": "provided_series_statistic", "inputs": ["kx_tail_a", "kx_tail_b"]}

    module = provided_series.build_module(spec, claim)
    out = module.run(bundle, claim, 0.0)

    assert out["ok"] is True
    assert len(out["net"]) == 10
    assert out["n_trades"] == 10
    assert out["bars_per_year"] == 1.0
    assert out["positions"].index.equals(out["net"].index)
    assert out["net"].index.duplicated().any()
    assert getattr(module, "__auto_generated__", None) is False
    assert module.__module_id__ == "auto_pool"
    assert module.__strategy_class__ == "kalshi_tail_calibration"

    module_file = tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        with module_file:
            module_file.write(
                "import pandas as pd\n"
                "__module_id__='auto_pool'\n"
                "__strategy_class__='kalshi_tail_calibration'\n"
                "def run(bundle, claim, cost_frac):\n"
                "    a = bundle.get('kx_tail_a').data\n"
                "    b = bundle.get('kx_tail_b').data\n"
                "    net = pd.concat([a, b]).sort_index(kind='mergesort')\n"
                "    return {'ok': True, 'net': net, 'positions': pd.Series(1.0, index=net.index), "
                "'bars_per_year': 1.0, 'n_trades': len(net)}\n"
            )
        ok, why = impl_gen._validate_module(tmp.name, "auto_pool", bundle, claim, 0.0)
        assert ok is True, why
    finally:
        os.unlink(tmp.name)


def _patch_run_paths(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import p7_backtest

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
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
    monkeypatch.setattr(p7_backtest, "LEDGER", tmp_path / "backtest_ledger.tsv")
    (tmp_path / "modules").mkdir(parents=True)


def test_planted_eval_cohort_mean_claim_clears_fidelity_and_reaches_real_verdict(tmp_path, monkeypatch):
    """The 6g planted eval (FIX 5a): a minimal pre-registered cohort-mean claim -- pooled
    mean of N declared series > threshold, one statistic -- MUST clear fidelity 100% of
    the time and reach a real verdict (not cannot_replicate/unfaithful_spec)."""
    from penrose import concepts, config
    from penrose.pipeline import extract, run as runmod, spec_gen

    _patch_run_paths(tmp_path, monkeypatch)
    _patch_fast_robustness(monkeypatch)
    paper = tmp_path / "cohort.md"
    paper.write_text("Pool declared series kx_tail_a and kx_tail_b as one statistic.")
    claim = _claim(PLANTED_CLAIM_STATEMENT, claim_id="cohort-c1")
    net_a = pd.Series(np.linspace(0.006, 0.014, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    net_b = pd.Series(np.linspace(0.005, 0.013, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_tail_a", "kx_tail_b"])

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})

    def fake_try_implement(*args, **kwargs):
        raise AssertionError("provided_series_statistic must skip LLM/Docker auto-implementation")

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
    monkeypatch.setattr(runmod.fidelity, "assess", fake_fidelity)
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(paper, use_llm=True, claims_override=[claim],
                            bundle_override=_contract_bundle({"kx_tail_a": net_a, "kx_tail_b": net_b}),
                            force=True)

    assert out["decisions"][0]["claim_id"] == "cohort-c1"
    assert out["decisions"][0]["verdict"] in {"kill", "underpowered", "watch"}
    assert out["decisions"][0]["verdict"] != "pending_module"
    assert out["decisions"][0]["verdict"] != "cannot_replicate"
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert rows[0]["verdict"] not in ("cannot_replicate",)
    assert rows[0].get("kill_reason") != "unfaithful_spec"
    assert rows[0]["verdict"] != "research-supported"
    assert rows[0]["metrics"]["claim_type"] == "provided_series_statistic"
    assert rows[0]["metrics"]["provided_series_provenance"] == "unverified_construction"
    assert rows[0]["metrics"]["n_trials"] == 1
    assert rows[0]["metrics"]["n_trades"] == 120
    ledger = pd.read_csv(tmp_path / "backtest_ledger.tsv", sep="\t")
    assert int(ledger.iloc[0]["search_denominator"]) == 1
    assert ledger.iloc[0]["generation_source"] == "provided_series_statistic"


def test_exp1b_source_only_provided_type_is_threaded_through_verdict(tmp_path, monkeypatch):
    """EXP-1b: the extracted claim looks like a trading strategy without source, while the
    source declares that Penrose must test already-encoded provided P&L series."""
    from penrose import concepts, config
    from penrose.pipeline import extract, fidelity, run as runmod, spec_gen

    _patch_run_paths(tmp_path, monkeypatch)
    _patch_fast_robustness(monkeypatch)
    source_text = (
        "Pool these 8 kx_tail_a and kx_tail_b net-tail-P&L series (each already encodes "
        "the per-market net P&L above, held to settlement) into a pooled sample and run "
        "a single one-sided one-sample t-test on that pooled sample. There is exactly "
        "one pre-registered statistic; because it is a single pooled test, no "
        "multiplicity correction applies."
    )
    paper = tmp_path / "exp1b.md"
    paper.write_text(source_text)
    statement = (
        "A strategy that buys the mispriced side of settled tail markets "
        "earns a positive mean net P&L"
    )
    claim = _claim(statement, claim_id="exp-1b", claimed_metric_quote="")
    net_a = pd.Series(np.linspace(0.006, 0.014, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    net_b = pd.Series(np.linspace(0.005, 0.013, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_tail_a", "kx_tail_b"])

    assert spec_gen.classify_claim_type(claim) == "trading_strategy"
    assert spec_gen.classify_claim_type(claim, _source(source_text)) == (
        "provided_series_statistic"
    )

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.impl_gen, "try_implement",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("provided_series_statistic must skip auto-implementation")))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    def false_unfaithful(*args, **kwargs):
        return (
            {"faithful": False, "confidence": 0.95,
             "divergences": ["trading construction not implemented"],
             "note": "missing trading construction"},
            types.SimpleNamespace(independent_verifier=True),
        )

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)

    out = runmod.run_source(paper, use_llm=True, claims_override=[claim],
                            bundle_override=_contract_bundle({"kx_tail_a": net_a, "kx_tail_b": net_b}),
                            force=True)

    assert getattr(claim, "resolved_claim_type") == "provided_series_statistic"
    assert runmod._generation_source_for(claim) == "provided_series_statistic"
    assert out["decisions"][0]["verdict"] in {"kill", "underpowered", "watch"}
    assert out["decisions"][0]["verdict"] != "cannot_replicate"
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    metrics = rows[0]["metrics"]
    assert rows[0]["kill_reason"] != "unfaithful_module"
    assert metrics["claim_type"] == "provided_series_statistic"
    assert metrics["provided_series_provenance"] == "unverified_construction"
    assert metrics["n_trials"] == 1
    assert metrics["regime"]["n_partitions"] == 0
    assert metrics["fidelity"]["faithful"] is True
    assert metrics["fidelity"]["provided_series_fidelity_override"] == (
        "deterministic_template_structural"
    )
    ledger = pd.read_csv(tmp_path / "backtest_ledger.tsv", sep="\t")
    assert int(ledger.iloc[0]["search_denominator"]) == 1
    assert ledger.iloc[0]["generation_source"] == "provided_series_statistic"


def test_source_only_provided_without_preregistration_keeps_normal_deflation(tmp_path, monkeypatch):
    from penrose import concepts, config
    from penrose.pipeline import extract, run as runmod, spec_gen

    _patch_run_paths(tmp_path, monkeypatch)
    _patch_fast_robustness(monkeypatch)
    source_text = (
        "Pool these kx_tail_a and kx_tail_b net-tail-P&L series (each already encodes "
        "the per-market net P&L above, held to settlement) into a pooled sample and run "
        "a one-sided one-sample t-test on that pooled sample."
    )
    paper = tmp_path / "exp1b_not_preregistered.md"
    paper.write_text(source_text)
    claim = _claim(
        "A strategy that buys the mispriced side of settled tail markets earns a "
        "positive mean net P&L",
        claim_id="exp-1b-open-cohort",
    )
    net_a = pd.Series(np.linspace(0.006, 0.014, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    net_b = pd.Series(np.linspace(0.005, 0.013, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_tail_a", "kx_tail_b"])

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.fidelity, "assess",
                        lambda *a, **k: {"faithful": True, "verified": True, "checked": True,
                                         "confidence": 0.95, "divergences": [],
                                         "note": "faithful"})
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(paper, use_llm=True, claims_override=[claim],
                            bundle_override=_contract_bundle({"kx_tail_a": net_a, "kx_tail_b": net_b}),
                            force=True)

    assert out["decisions"][0]["verdict"] in {"kill", "underpowered", "watch"}
    assert out["decisions"][0]["verdict"] != "cannot_replicate"
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    metrics = rows[0]["metrics"]
    assert metrics["claim_type"] == "provided_series_statistic"
    assert metrics["provided_series_provenance"] == "unverified_construction"
    assert metrics["n_trials"] != 1
    assert int(metrics["regime"].get("n_partitions", 0)) > 0
    ledger = pd.read_csv(tmp_path / "backtest_ledger.tsv", sep="\t")
    assert ledger.iloc[0]["generation_source"] == "provided_series_statistic"


def test_provided_series_missing_declared_input_routes_to_needs_data(tmp_path, monkeypatch):
    from penrose import concepts, config
    from penrose.pipeline import extract, run as runmod, spec_gen

    _patch_run_paths(tmp_path, monkeypatch)
    paper = tmp_path / "cohort_missing.md"
    paper.write_text("Pool declared series kx_tail_a and kx_tail_missing as one statistic.")
    claim = _claim(
        "The pooled mean of declared series kx_tail_a and kx_tail_missing is greater than zero.",
        claim_id="cohort-missing",
    )
    net_a = pd.Series(np.linspace(-0.01, 0.02, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    fake_catalog = types.SimpleNamespace(available=lambda: ["kx_tail_a", "kx_tail_missing"])

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    monkeypatch.setattr(extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.impl_gen, "try_implement",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("provided_series_statistic must skip auto-impl")))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(
        paper, use_llm=False, claims_override=[claim],
        bundle_override=_contract_bundle({"kx_tail_a": net_a}),
        force=True,
    )

    assert out["decisions"] == [{
        "claim_id": "cohort-missing", "verdict": "needs_data", "kill_reason": None,
    }]
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert rows[0]["verdict"] == "needs_data"
    assert rows[0]["metrics"]["missing_series"] == ["kx_tail_missing"]
    reqs = [json.loads(line) for line in config.DATA_REQUESTS.read_text().splitlines() if line.strip()]
    assert reqs[0]["missing_series"] == ["kx_tail_missing"]


def test_non_provided_claim_still_uses_normal_auto_impl_path(tmp_path, monkeypatch):
    from penrose import concepts
    from penrose.pipeline import run as runmod

    _patch_run_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(runmod.config, "FIDELITY_CHECK", False)
    paper = tmp_path / "momentum.md"
    paper.write_text("A 12 month momentum signal enters long positions and earns positive Sharpe.")
    claim = _claim(
        "A 12 month momentum signal enters long positions and earns positive Sharpe.",
        claim_id="momentum-c1",
        strategy_class="unit_momentum",
    )
    calls = []

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(runmod.sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(runmod.sandbox, "ensure_image", lambda: True)

    def fake_try_implement(spec, claim, bundle, cost_frac, **kwargs):
        calls.append((spec.get("claim_type"), claim.claim_id))
        return {"ok": False, "reason": "unit auto impl declined"}

    monkeypatch.setattr(runmod.impl_gen, "try_implement", fake_try_implement)
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(
        paper, use_llm=False, claims_override=[claim], bundle_override=_contract_bundle({}),
        force=True,
    )

    assert calls == [("trading_strategy", "momentum-c1")]
    assert out["decisions"] == [{
        "claim_id": "momentum-c1", "verdict": "pending_module", "kill_reason": None,
    }]
