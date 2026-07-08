import numpy as np
import pandas as pd
import types

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import fidelity, fidelity_memory, forecast_skill, spec_gen
from penrose.pipeline import run as runmod
from penrose.pipeline.human_review import human_review_explanation
from penrose.pipeline.p1_ingest import IngestedSource


def _claim() -> Claim:
    return Claim(
        claim_id="fs-test",
        statement=(
            "The model_forecast forecasts target out-of-sample and beats the random-walk "
            "benchmark with lower MSFE."
        ),
        mechanism="forecast accuracy comparison",
        scope="synthetic",
        horizon="1 day",
        source_id="test",
        source_span="",
        claimed_metric_quote="MSFE",
        applicable_strategy_class="forecast_skill",
    )


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


def _spec() -> dict:
    return {
        "module_id": "test_forecast_skill",
        "strategy_class": "forecast_skill",
        "claim_type": "forecast_skill",
        "inputs": ["model_forecast", "target"],
        "model_forecast": "model_forecast",
        "target": "target",
        "benchmark": {"kind": "implied", "method": "random_walk"},
        "loss": "squared_error",
        "_llm_mode": "deterministic-template",
        "claim_statement": _claim().statement,
    }


def test_classifier_routes_forecast_skill_before_predictive_regression():
    claim = _claim()
    assert fidelity_memory.classify_claim_type(claim) == "forecast_skill"

    msfe_ratio = Claim(
        claim_id="fs-msfe",
        statement="The KXCPI model forecasts volatility out-of-sample with MSFE=0.959 and p=0.010.",
        mechanism="forecast accuracy comparison",
        scope="synthetic",
        horizon="1 day",
        source_id="test",
        source_span="",
        claimed_metric_quote="MSFE=0.959",
        applicable_strategy_class="forecast_skill",
    )
    assert fidelity_memory.classify_claim_type(msfe_ratio) == "forecast_skill"

    regression = Claim(
        claim_id="pr",
        statement="predictor forecasts target 3-day-ahead with regression beta t-statistic 2.4.",
        mechanism="predictive regression",
        scope="synthetic",
        horizon="3 days",
        source_id="test",
        source_span="",
        claimed_metric_quote="t-stat",
        applicable_strategy_class="predictive_regression",
    )
    assert fidelity_memory.classify_claim_type(regression) == "predictive_regression"


def test_constructed_benchmark_is_strictly_causal():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    y = pd.Series(np.arange(8, dtype=float), index=idx)

    rw = forecast_skill._construct_implied_benchmark(y, "random_walk")
    hist = forecast_skill._construct_implied_benchmark(y, "historical_mean")

    mutated = y.copy()
    mutated.loc[idx[4:]] = 1000.0
    rw_mut = forecast_skill._construct_implied_benchmark(mutated, "random_walk")
    hist_mut = forecast_skill._construct_implied_benchmark(mutated, "historical_mean")

    assert rw.loc[idx[4]] == rw_mut.loc[idx[4]]
    assert hist.loc[idx[4]] == hist_mut.loc[idx[4]]
    assert rw.loc[idx[4]] == y.loc[idx[3]]
    assert hist.loc[idx[4]] == y.iloc[:4].mean()


def test_executor_emits_loss_differential_against_implied_benchmark():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    target = pd.Series(np.linspace(0.0, 39.0, 40), index=idx)
    model = target.copy()
    bundle = DataBundle(series={
        "model_forecast": Series("model_forecast", model, "test", "forecast"),
        "target": Series("target", target, "test", "target"),
    })
    out = forecast_skill.build_module(_spec(), _claim()).run(bundle, _claim(), 0.0)

    assert out["ok"] is True
    first_idx = out["net"].index[0]
    assert first_idx == idx[1]
    assert out["net"].loc[first_idx] == 1.0
    assert out["positions"].eq(1.0).all()
    assert out["forecast_skill"]["benchmark_kind"] == "implied"


def test_forecast_skill_spec_binds_declared_random_walk(monkeypatch):
    fake_catalog = types.SimpleNamespace(available=lambda: ["model_forecast", "target"])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim()

    spec = spec_gen._forecast_skill_spec(claim, _source())

    assert spec["model_forecast"] == "model_forecast"
    assert spec["target"] == "target"
    assert spec["benchmark"] == {"kind": "implied", "method": "random_walk"}
    assert spec["unknowns"] == []


def test_forecast_skill_binding_review_requires_declared_benchmark():
    claim = _claim()
    spec = {
        **_spec(),
        "benchmark": {},
        "unknowns": ["benchmark forecast was not resolved"],
        "binding_provenance": {
            "model_forecast": {"kind": "literal", "series": "model_forecast", "full_coverage": True},
            "target": {"kind": "literal", "series": "target", "full_coverage": True},
            "benchmark": {"kind": "unresolved", "confirmed": False},
        },
    }

    review = runmod._forecast_skill_binding_review(claim, spec)

    assert review is not None
    assert review["reason"] == "forecast_skill_binding_unresolved"
    assert review["explanation"] == human_review_explanation(
        "forecast_skill_binding_uncertain", review["detail"]
    )


def test_forecast_skill_fidelity_rejects_benchmark_substitution(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["benchmark substitution"],
            "note": "wrong benchmark",
        }, response)

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    claim = _claim()
    bad = {
        **_spec(),
        "benchmark": {"kind": "implied", "method": "historical_mean"},
        "binding_provenance": {
            "model_forecast": {
                "kind": "literal",
                "series": "model_forecast",
                "score": 1.0,
                "matched_tokens": ["model", "forecast"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
            },
            "target": {
                "kind": "literal",
                "series": "target",
                "score": 1.0,
                "matched_tokens": ["target"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
            },
            "benchmark": {"kind": "implied", "method": "historical_mean", "confirmed": True},
        },
    }

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=bad)

    assert out["faithful"] is False
    assert "forecast_skill_fidelity_override" not in out


def test_fsk2_multistep_horizon_subsamples_non_overlapping():
    """FSK-2: for an h-step-ahead forecast the executor emits NON-OVERLAPPING loss differentials
    (subsample every h) so overlapping serial correlation can't inflate the DM t-stat."""
    import numpy as np
    import pandas as pd
    from penrose.pipeline import forecast_skill as FS
    from penrose.data.contract import DataBundle, Series

    idx = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, 500), index=idx)
    f = pd.Series(rng.normal(0, 1, 500), index=idx)
    bundle = DataBundle(series={"f": Series("f", f, "t", "z"), "y": Series("y", y, "t", "z")})

    class _Claim:
        claim_id = "fsk"
        horizon = "5 days"
        statement = ""
        mechanism = ""

    def _spec(h):
        return {"claim_type": "forecast_skill", "module_id": "m", "strategy_class": "forecast_skill",
                "model_forecast": "f", "target": "y", "benchmark": {"method": "random_walk"}, "horizon": h}

    out1 = FS.build_module(_spec("1 day"), _Claim()).run(bundle, _Claim(), 0.0)
    out5 = FS.build_module(_spec("5 days"), _Claim()).run(bundle, _Claim(), 0.0)
    assert out1["n_trades"] > 480                       # h=1: ~all aligned obs
    assert 90 <= out5["n_trades"] <= 110                # h=5: ~n/5 non-overlapping
