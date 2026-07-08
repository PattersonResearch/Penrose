import json
import numpy as np
import pandas as pd
import types

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import spec_gen
from penrose.pipeline import fidelity, fidelity_memory
from penrose.pipeline import predictive_regression
from penrose.pipeline import p7_backtest as P7
from penrose.pipeline import run as runmod
from penrose.pipeline.human_review import human_review_explanation
from penrose.pipeline.p1_ingest import IngestedSource


def _claim() -> Claim:
    return Claim(
        claim_id="pr-test",
        statement="predictor forecasts target 3-day-ahead with a regression coefficient",
        mechanism="predictive regression",
        scope="synthetic",
        horizon="3 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t-stat",
        applicable_strategy_class="predictive_regression",
    )


def _bundle(x: pd.Series, y: pd.Series) -> DataBundle:
    return DataBundle(series={
        "predictor": Series("predictor", x, "test", "z"),
        "target": Series("target", y, "test", "z"),
    })


def _source(text: str = "") -> IngestedSource:
    return IngestedSource(
        source_id="test",
        title="test",
        text=text,
        n_pages=1,
        n_chars=len(text),
        text_sha256="abc",
        injection_flags=[],
    )


def _spec(horizon: int = 3) -> dict:
    return {
        "module_id": "test_predictive_regression",
        "strategy_class": "predictive_regression",
        "claim_type": "predictive_regression",
        "inputs": ["predictor", "target"],
        "predictor": "predictor",
        "target": "target",
        "horizon": horizon,
        "estimator": "single_predictor_ols_covariance_sign",
    }


def test_series_resolver_kxcpi_phrasings_and_negative_cases():
    names = [
        "kxcpi_abs_prob_change_daily",
        "kxabc_abs_prob_change_daily",
        "sol_spot_daily",
    ]
    phrases = [
        "KXCPI absolute probability change signal",
        "absolute probability change for KXCPI",
        "KXCPI abs prob change daily",
        "Kalshi KXCPI probability change",
        "KXCPI absolute prob delta",
    ]
    for phrase in phrases:
        out = spec_gen.resolve_series_from_prose(phrase, names)
        assert out is not None, phrase
        assert out["series"] == "kxcpi_abs_prob_change_daily"
        assert out["score"] >= 0.5

    assert spec_gen.resolve_series_from_prose("unrelated macro weather prose", names) is None


def test_derived_resolver_sol_realized_vol_and_abbreviation():
    names = ["sol_spot_daily", "btc_spot_daily", "kxcpi_abs_prob_change_daily"]
    out = spec_gen.resolve_derived_series("Solana realized volatility", names, 5)
    assert out is not None
    assert out["transform"] == "realized_vol"
    assert out["base_series"] == "sol_spot_daily"
    assert out["window"] == 5
    assert out["horizon_encoded"] is True

    rv = spec_gen.resolve_derived_series("SOL RV", names, 3)
    assert rv is not None
    assert rv["transform"] == "realized_vol"
    assert rv["base_series"] == "sol_spot_daily"
    assert rv["window"] == 3


def test_predictive_spec_binds_kxcpi_signal_to_derived_sol_rv(monkeypatch):
    fake_catalog = types.SimpleNamespace(
        available=lambda: ["kxcpi_abs_prob_change_daily", "sol_spot_daily"]
    )
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = Claim(
        claim_id="kxcpi-sol-rv",
        statement=(
            "The KXCPI signal negatively predicts five-day ahead realized volatility "
            "for Solana in-sample with t=-2.55 and p=0.011."
        ),
        mechanism="predictive regression",
        scope="Solana",
        horizon="5 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t=-2.55",
        applicable_strategy_class="predictive_regression",
    )

    spec = spec_gen._predictive_regression_spec(claim, _source())

    assert spec["predictor"] == "kxcpi_abs_prob_change_daily"
    assert spec["target"]["transform"] == "realized_vol"
    assert spec["target"]["base_series"] == "sol_spot_daily"
    assert spec["target"]["window"] == 5
    assert spec["unknowns"] == []
    assert spec["binding_provenance"]["predictor"]["score"] >= 0.5
    assert spec["binding_provenance"]["target"]["base_resolution"]["score"] >= 0.5


def test_predictive_regression_fidelity_uses_binding_provenance_for_derived_target(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["spec-only deterministic module"],
            "note": "structural false positive",
        }, response)

    fake_catalog = types.SimpleNamespace(
        available=lambda: ["kxcpi_abs_prob_change_daily", "sol_spot_daily"]
    )
    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = Claim(
        claim_id="kxcpi-sol-rv",
        statement=(
            "The KXCPI signal negatively predicts five-day ahead realized volatility "
            "for Solana in-sample with t=-2.55 and p=0.011."
        ),
        mechanism="predictive regression",
        scope="Solana",
        horizon="5 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t=-2.55",
        applicable_strategy_class="predictive_regression",
    )
    spec = spec_gen._predictive_regression_spec(claim, _source())

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=spec)

    assert out["faithful"] is True
    assert out["predictive_regression_fidelity_override"] == "deterministic_template_structural"


def test_predictive_regression_fidelity_rejects_low_confidence_binding(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["low-confidence binding"],
            "note": "wrong variables",
        }, response)

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    claim = _claim()
    spec = {
        **_spec(3),
        "_llm_mode": "deterministic-template",
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
        "binding_provenance": {
            "predictor": {
                "kind": "prose",
                "series": "predictor",
                "score": 0.25,
                "matched_tokens": ["predictor"],
            },
            "target": {
                "kind": "prose",
                "series": "target",
                "score": 1.0,
                "matched_tokens": ["target", "forecast"],
            },
        },
    }

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=spec)

    assert out["faithful"] is False
    assert "predictive_regression_fidelity_override" not in out


def test_predictive_regression_unresolved_binding_routes_needs_review():
    claim = Claim(
        claim_id="pr-unresolved",
        statement="The unresolved inflation channel forecasts target 3-day-ahead with a regression beta.",
        mechanism="predictive regression",
        scope="synthetic",
        horizon="3 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t-stat",
        applicable_strategy_class="predictive_regression",
    )
    spec = {
        **_spec(3),
        "_llm_mode": "deterministic-template",
        "inputs": ["target"],
        "predictor": "",
        "target": "target",
        "unknowns": ["unresolved binding: predictor series was not resolved"],
        "binding_provenance": {
            "predictor": {
                "kind": "unresolved",
                "description": "unresolved inflation channel",
                "confirmed": False,
            },
            "target": {
                "kind": "literal",
                "series": "target",
                "score": 1.0,
                "matched_tokens": ["target"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
                "description": "target",
            },
        },
    }

    review = runmod._predictive_regression_binding_review(claim, spec)

    assert review is not None
    assert review["reason"] == "predictive_regression_binding_unresolved"
    explanation = review["explanation"]
    assert "unresolved inflation channel" in explanation["why"]
    assert "candidate `target`" in explanation["why"]
    assert "Confirm predictor=" in explanation["action"]


def test_predictive_regression_unconfirmed_binding_routes_needs_review_not_unfaithful():
    claim = Claim(
        claim_id="pr-unconfirmed",
        statement="Bitcoin perp volume predicts five-day ahead realized volatility for Solana.",
        mechanism="predictive regression",
        scope="Solana",
        horizon="5 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t-stat",
        applicable_strategy_class="predictive_regression",
    )
    target = {
        "kind": "derived_series",
        "transform": "realized_vol",
        "base_series": "sol_spot_daily",
        "window": 5,
        "horizon_encoded": True,
    }
    spec = {
        **_spec(5),
        "_llm_mode": "deterministic-template",
        "inputs": ["btc_perp_funding_daily", target],
        "predictor": "btc_perp_funding_daily",
        "target": target,
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
        "unknowns": [],
        "binding_provenance": {
            "predictor": {
                "kind": "prose",
                "series": "btc_perp_funding_daily",
                "score": 0.67,
                "matched_tokens": ["btc", "perp"],
                "full_coverage": False,
                "unmatched_name_tokens": ["funding"],
                "description": "Bitcoin perp volume",
            },
            "target": {
                "kind": "derived_series",
                "transform": "realized_vol",
                "base_series": "sol_spot_daily",
                "window": 5,
                "base_resolution": {
                    "series": "sol_spot_daily",
                    "score": 1.0,
                    "matched_tokens": ["sol~solana"],
                    "full_coverage": True,
                    "unmatched_name_tokens": [],
                },
                "description": "five-day ahead realized volatility for Solana",
            },
        },
    }

    review = runmod._predictive_regression_binding_review(claim, spec)

    assert review is not None
    assert review["reason"] == "predictive_regression_binding_unconfirmed"
    assert review["detail"]["confirmed"]["predictor"] is False
    assert "candidate `btc_perp_funding_daily` (confidence 0.67" in review["explanation"]["why"]
    assert "cannot_replicate" not in review["explanation"]["what"]


def test_predictive_regression_confirmed_binding_proceeds_to_execution_path():
    claim = _claim()
    spec = {
        **_spec(3),
        "_llm_mode": "deterministic-template",
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
        "unknowns": [],
        "binding_provenance": {
            "predictor": {
                "kind": "literal",
                "series": "predictor",
                "score": 1.0,
                "matched_tokens": ["predictor"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
                "description": "predictor",
            },
            "target": {
                "kind": "literal",
                "series": "target",
                "score": 1.0,
                "matched_tokens": ["target"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
                "description": "target",
            },
        },
    }

    assert runmod._predictive_regression_binding_review(claim, spec) is None


def test_human_review_explanation_for_binding_is_operator_readable():
    out = human_review_explanation(
        "predictive_regression_binding_uncertain",
        {
            "predictor": "kx_cpi_inflation_probability_daily",
            "target": {
                "kind": "derived_series",
                "transform": "realized_vol",
                "base_series": "eth_spot_daily",
                "window": 5,
            },
            "binding_provenance": {
                "predictor": {
                    "kind": "prose",
                    "series": "kx_cpi_inflation_probability_daily",
                    "score": 0.5,
                    "matched_tokens": ["cpi", "inflation"],
                    "description": "the inflation channel, measured by CPI repricing on KXCPI contracts",
                },
                "target": {
                    "kind": "derived_series",
                    "transform": "realized_vol",
                    "base_series": "eth_spot_daily",
                    "window": 5,
                    "base_resolution": {"score": 1.0, "matched_tokens": ["eth~ethereum"]},
                    "description": "Ethereum 5-day-ahead realized volatility",
                },
            },
            "confirmed": {"predictor": False, "target": True},
        },
    )

    assert out["what"].startswith("Routed to human review")
    assert "CPI repricing" in out["why"]
    assert "candidate `kx_cpi_inflation_probability_daily` (confidence 0.50" in out["why"]
    assert "realized_vol(eth_spot_daily, 5)" in out["why"]
    assert "Confirm predictor=kx_cpi_inflation_probability_daily" in out["action"]


def test_needs_review_attaches_human_review_to_decision_and_queue(tmp_path, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    claim = _claim()
    review = {
        "what": "Routed to human review - I could not confirm the data binding.",
        "why": "Predictor prose has an unconfirmed candidate.",
        "action": "Confirm predictor=foo and target=bar, then re-run.",
    }

    dec = runmod._needs_review(
        claim,
        "predictive_regression_binding_unconfirmed",
        {"claim_id": claim.claim_id, "stages": {}},
        {"claims": [], "source_id": "test", "idempotency": {"run_id": "unit"}},
        metrics={"binding_uncertainty": "predictive_regression_binding_unconfirmed"},
        review=review,
    )

    assert dec.verdict == "needs_review"
    assert dec.kill_reason is None
    assert dec.metrics["human_review"] == review
    assert "Predictor prose has an unconfirmed candidate" in dec.rationale
    queued = json.loads((tmp_path / "review_queue.jsonl").read_text().splitlines()[-1])
    assert queued["human_review"] == review
    assert queued["metrics"]["human_review"] == review


def test_predictive_regression_horizon_alignment_contract():
    n, h = 80, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = pd.Series(np.linspace(-2.0, 2.0, n), index=idx)
    y_vals = np.zeros(n)
    y_vals[h:] = 10.0 * x.to_numpy()[:-h]
    y = pd.Series(y_vals, index=idx)

    module = predictive_regression.build_module(_spec(h), _claim())
    out = module.run(_bundle(x, y), _claim(), 0.0)

    assert out["ok"] is True
    emitted_positions = list(range(0, n - h, h))
    assert len(out["net"]) == len(emitted_positions)
    assert out["net"].index[0] == idx[0]
    assert out["net"].index[-1] == idx[emitted_positions[-1]]
    assert set(out["net"].index.to_series().diff().dropna().dt.days) == {h}
    assert out["regression"]["horizon"] == h
    assert out["regression"]["sign"] == 1.0
    assert abs(out["regression"]["beta"] - 10.0) < 1e-9
    expected_bpy = predictive_regression._observations_per_year(out["net"].index, len(out["net"]))
    assert out["bars_per_year"] == expected_bpy


def test_predictive_regression_standardization_is_frozen_to_is_prefix():
    n, h = 120, 5
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(11)
    x = pd.Series(rng.normal(0.0, 1.0, n), index=idx)
    aligned_y = 0.6 * x.to_numpy() + 0.4 * rng.normal(0.0, 1.0, n)
    y = pd.Series(np.concatenate([np.zeros(h), aligned_y[:-h]]), index=idx)

    module = predictive_regression.build_module(_spec(h), _claim())
    base = module.run(_bundle(x, y), _claim(), 0.0)
    assert base["ok"] is True

    x_changed = x.copy()
    y_changed = y.copy()
    emitted_idx = idx[:n - h:h]
    is_cut = int(len(emitted_idx) * P7.IS_FRAC)
    oos_start = emitted_idx[is_cut]
    x_changed.loc[oos_start:] = x_changed.loc[oos_start:] * 100.0 + 5000.0
    y_changed.loc[oos_start:] = y_changed.loc[oos_start:] * -100.0 - 7000.0
    changed = module.run(_bundle(x_changed, y_changed), _claim(), 0.0)
    assert changed["ok"] is True

    assert base["regression"]["x_mean_is"] == changed["regression"]["x_mean_is"]
    assert base["regression"]["x_sd_is"] == changed["regression"]["x_sd_is"]
    assert base["regression"]["y_mean_is"] == changed["regression"]["y_mean_is"]
    assert base["regression"]["y_sd_is"] == changed["regression"]["y_sd_is"]
    assert base["regression"]["n_is_moments"] == is_cut - 1
    pd.testing.assert_series_equal(
        base["net"].iloc[:base["regression"]["n_is_moments"]],
        changed["net"].iloc[:changed["regression"]["n_is_moments"]],
    )


def test_forward_realized_vol_target_is_not_double_shifted_and_keeps_is_clean():
    n, h = 120, 5
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(17)
    x = pd.Series(rng.normal(0.0, 1.0, n), index=idx)
    logret = pd.Series(0.01 + 0.002 * np.sin(np.arange(n)), index=idx)
    price = pd.Series(np.exp(logret.cumsum()), index=idx)
    target_spec = {
        "kind": "derived_series",
        "transform": "realized_vol",
        "base_series": "sol_spot_daily",
        "window": h,
        "horizon_encoded": True,
    }
    spec = {**_spec(h), "inputs": ["predictor", target_spec], "target": target_spec}
    bundle = DataBundle(series={
        "predictor": Series("predictor", x, "test", "z"),
        "sol_spot_daily": Series("sol_spot_daily", price, "test", "price"),
    })

    rv = predictive_regression._realized_vol(price, h)
    changed_future = price.copy()
    changed_future.iloc[h + 1] *= 50.0
    changed_rv = predictive_regression._realized_vol(changed_future, h)
    assert rv.iloc[0] == changed_rv.iloc[0]
    changed_inside = price.copy()
    changed_inside.iloc[h] *= 50.0
    inside_rv = predictive_regression._realized_vol(changed_inside, h)
    assert rv.iloc[0] != inside_rv.iloc[0]

    module = predictive_regression.build_module(spec, _claim())
    out = module.run(bundle, _claim(), 0.0)
    assert out["ok"] is True
    assert out["regression"]["target_horizon_encoded"] is True

    aligned = predictive_regression._non_overlapping_xy(
        predictive_regression._aligned_xy(
            x,
            rv.dropna(),
            h,
            target_horizon_encoded=True,
        ),
        h,
    )
    is_cut = int(len(aligned) * P7.IS_FRAC)
    oos_start = aligned.index[is_cut]
    moment_frame = aligned.iloc[:is_cut]
    moment_frame = moment_frame[moment_frame["target_time"] < oos_start]
    assert out["regression"]["y_mean_is"] == float(moment_frame["y_h"].mean())

    changed_price = price.copy()
    changed_price.loc[oos_start:] = changed_price.loc[oos_start:] * 100.0
    changed_bundle = DataBundle(series={
        "predictor": Series("predictor", x, "test", "z"),
        "sol_spot_daily": Series("sol_spot_daily", changed_price, "test", "price"),
    })
    changed = module.run(changed_bundle, _claim(), 0.0)
    assert changed["ok"] is True
    assert out["regression"]["x_mean_is"] == changed["regression"]["x_mean_is"]
    assert out["regression"]["x_sd_is"] == changed["regression"]["x_sd_is"]
    assert out["regression"]["y_mean_is"] == changed["regression"]["y_mean_is"]
    assert out["regression"]["y_sd_is"] == changed["regression"]["y_sd_is"]


def test_predictive_regression_classifier_does_not_steal_trading_claim():
    claim = Claim(
        claim_id="pr-route",
        statement=(
            "The signal forecasts next-day returns; IS Sharpe 1.4, OOS 0.6, "
            "regression beta 0.3."
        ),
        mechanism="Use the signal as a trading strategy.",
        scope="synthetic",
        horizon="1 day",
        source_id="test",
        source_span="",
        claimed_metric_quote="Sharpe",
    )

    assert fidelity_memory.classify_claim_type(claim) == "trading_strategy"


def test_predictive_regression_fidelity_backstop_rejects_swapped_inputs(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["predictor and target are swapped"],
            "note": "wrong direction",
        }, response)

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    claim = _claim()
    swapped = {
        **_spec(3),
        "_llm_mode": "deterministic-template",
        "inputs": ["target", "predictor"],
        "predictor": "target",
        "target": "predictor",
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
    }

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=swapped)

    assert out["faithful"] is False
    assert "predictive_regression_fidelity_override" not in out


def test_predictive_regression_fidelity_backstop_confirms_ordered_inputs(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["deterministic module is spec-only"],
            "note": "structural false positive",
        }, response)

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    claim = _claim()
    spec = {
        **_spec(3),
        "_llm_mode": "deterministic-template",
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
    }

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=spec)

    assert out["faithful"] is True
    assert out["predictive_regression_fidelity_override"] == "deterministic_template_structural"


def test_unanchored_predictive_regression_survivor_caps_at_watch(monkeypatch):
    from penrose import config
    from penrose.pipeline import stages

    monkeypatch.setattr(config, "COST_PROVENANCE", "measured")
    claim = Claim(
        claim_id="pr-unanchored",
        statement="predictor forecasts target one-day-ahead with a regression beta",
        mechanism="predictive regression",
        scope="synthetic",
        horizon="1 day",
        source_id="dream",
        source_span="",
        claimed_metric_quote="t-stat",
        applicable_strategy_class="predictive_regression",
        source_type="generated_hypothesis",
    )
    bt = {
        "claim_type": "predictive_regression",
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 4.0,
        "n_oos": 1200,
        "oos_sharpe": 2.0,
        "bars_per_year": 52,
        "three_fold": {"folds": [1.0, 1.1, 0.9], "consistent": True},
        "capacity_usd": None,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
    }

    dec = stages.p8_verdict(
        claim,
        bt,
        {"holdout_sharpe": 1.0, "holdout_psr": config.HOLDOUT_CONFIRM_PSR},
        synthetic=False,
    )

    assert dec.verdict == "watch"
    assert dec.metrics["fidelity_provenance"] == "self-authored-unanchored"
    assert "external anchor" in dec.rationale


class _ClaimStub:
    def __init__(self, statement, mechanism=""):
        self.statement = statement
        self.mechanism = mechanism
        self.claimed_metric_quote = ""
        self.source_span = ""


def test_classifier_routes_regression_vs_strategy_pr2_pr3():
    """PR2-2 + PR3-1: strategy-framed claims -> trading_strategy; genuine regression claims (incl.
    spelled-out horizons, non-enumerated targets, incidental 'returns'/'trading') -> predictive_regression."""
    from penrose.pipeline.fidelity_memory import classify_claim_type as cc
    # genuine regressions (must route to predictive_regression)
    assert cc(_ClaimStub("The KXCPI signal negatively predicts five-day ahead realized volatility for Solana in-sample with t=-2.55 and p=0.011.")) == "predictive_regression"
    assert cc(_ClaimStub("The yield curve slope forecasts next-quarter GDP growth, regression beta 0.4, t-stat 2.8 out-of-sample.")) == "predictive_regression"
    assert cc(_ClaimStub("VRP forecasts next-month realized volatility, regression beta 0.5, t-stat 3.", "returns are not the target here")) == "predictive_regression"
    assert cc(_ClaimStub("The signal forecasts next-week realized volatility via volatility trading desks, regression coefficient 0.3, t-stat 2.5 out-of-sample.")) == "predictive_regression"
    assert cc(_ClaimStub("The dividend yield predicts next-quarter stock returns, regression t-stat 3.0 out-of-sample.")) == "predictive_regression"
    # strategy-framed (must NOT be stolen by the regression path)
    assert cc(_ClaimStub("Our signal forecasts next-day returns; in-sample Sharpe 1.4, out-of-sample 0.6, regression beta 0.3.")) == "trading_strategy"
    assert cc(_ClaimStub("A long-short momentum strategy earns Sharpe 2.0 with position sizing by volatility.")) == "trading_strategy"


def test_classifier_pr4_edge_cases():
    """PR4: strategy-metric steals (IR/turnover) -> trading; horizon phrasings (5-day horizon, monthly
    horizon, next 5 days, hyphenated '5-day ahead') and OLS/alpha-intercept regressions -> regression."""
    from penrose.pipeline.fidelity_memory import classify_claim_type as cc
    reg = [
        "X predicts Y over a 5-day horizon, regression beta 0.3, t-stat 2.8.",
        "X predicts Y at a monthly horizon, regression beta 0.3, t-stat 2.8.",
        "X predicts Y for the next 5 days, regression beta 0.3, t-stat 2.8.",
        "X forecasts Y 5-day ahead, OLS coefficient significant at 1%.",
        "X predicts Y 5-day ahead; the regression alpha (intercept) is 0.01 and beta is 0.3, t-stat 2.9.",
    ]
    for s in reg:
        assert cc(_ClaimStub(s)) == "predictive_regression", s
    trad = [
        "Our signal forecasts next-day returns with information ratio 1.2, turnover 5x; regression beta 0.3, t-stat 2.8.",
    ]
    for s in trad:
        assert cc(_ClaimStub(s)) == "trading_strategy", s


def test_prose_binding_derived_targets_and_synonyms():
    """Binding layer: prose predictor -> catalog series; prose derived target (realized vol / returns) ->
    derived spec on the right base series, incl. full-name->ticker synonyms (Chainlink->link, Bitcoin->btc)
    and multi-spot-variant disambiguation (sol_spot_daily preferred over sol_okx_spot_daily/price.*)."""
    from penrose.pipeline import spec_gen as SG
    names = [
        "kxcpi_abs_prob_change_daily", "kxcpi_abs_probability_change_daily",
        "sol_spot_daily", "sol_okx_spot_daily", "price.sol_usd_spot_daily",
        "eth_spot_daily", "btc_spot_daily", "link_spot_daily",
        "funding_btc_native_5y", "btc_perp_volume_daily",
    ]
    # predictor prose -> series
    r = SG.resolve_series_from_prose("KXCPI absolute probability change signal", names)
    assert r and r["series"].startswith("kxcpi_abs_prob"), r
    # derived targets (the 3 KXCPI coins)
    for coin, base in [("Solana", "sol_spot_daily"), ("Ethereum", "eth_spot_daily"), ("Chainlink", "link_spot_daily")]:
        d = SG.resolve_derived_series(f"{coin} realized volatility", names, 5)
        assert d and d["transform"] == "realized_vol" and d["base_series"] == base, (coin, d)
    # multi-variant disambiguation -> canonical bare spot series
    alias = SG._resolve_spot_alias_from_prose("Solana", names)
    assert alias and alias["series"] == "sol_spot_daily", alias


def test_bind1_partial_binding_does_not_auto_certify():
    """BIND-1: a partial (2-of-3 token) predictor binding (full_coverage=False) must NOT pass the
    deterministic correspondence override; only a full-coverage binding does."""
    from penrose.pipeline import fidelity as F
    partial = {"series": "btc_perp_funding_daily", "score": 0.67,
               "matched_tokens": ["btc", "perp"], "full_coverage": False}
    full = {"series": "kxcpi_abs_probability_change_daily", "score": 1.0,
            "matched_tokens": ["kxcpi", "abs~absolute", "probability", "change"], "full_coverage": True}
    target = {"kind": "derived_series", "transform": "realized_vol", "base_series": "sol_spot_daily",
              "window": 5, "base_resolution": {"series": "sol_spot_daily", "score": 1.0,
              "matched_tokens": ["sol~solana", "spot~asset"], "full_coverage": True}}
    wrong = "bitcoin perp volume predicts five-day ahead realized volatility for solana"
    good = "kxcpi absolute probability change signal predicts five-day ahead realized volatility for solana"
    assert F._ordered_binding_provenance_verified(wrong, partial, target) is False
    assert F._ordered_binding_provenance_verified(good, full, target) is True
