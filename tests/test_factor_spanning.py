import numpy as np
import pandas as pd
import types

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import factor_spanning
from penrose.pipeline import fidelity, fidelity_memory, spec_gen
from penrose.pipeline import p7_backtest as P7
from penrose.pipeline import run as runmod
from penrose.pipeline.p1_ingest import IngestedSource


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _claim(statement: str | None = None) -> Claim:
    return Claim(
        claim_id="fs-test",
        statement=statement or "The candidate factor earns alpha after controlling for FF3 factors.",
        mechanism="factor spanning regression",
        scope="synthetic",
        horizon="daily",
        source_id="test",
        source_span="",
        claimed_metric_quote="alpha t-stat",
        applicable_strategy_class="factor_spanning",
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


def _spec(**overrides) -> dict:
    spec = {
        "module_id": "test_factor_spanning",
        "strategy_class": "factor_spanning",
        "claim_type": "factor_spanning",
        "inputs": ["candidate", "b1", "b2"],
        "candidate_factor": "candidate",
        "benchmark_set": "ff3",
        "benchmark_factors": ["b1", "b2"],
        "estimator": "multivariate_ols_is_frozen_betas",
        "_llm_mode": "deterministic-template",
    }
    spec.update(overrides)
    return spec


def _bundle(candidate: pd.Series, b1: pd.Series, b2: pd.Series) -> DataBundle:
    return DataBundle(series={
        "candidate": Series("candidate", candidate, "test", "ret"),
        "b1": Series("b1", b1, "test", "ret"),
        "b2": Series("b2", b2, "test", "ret"),
    })


def test_factor_spanning_executor_emits_residual_alpha_and_positions():
    rng = np.random.default_rng(11)
    n = 800
    idx = _idx(n)
    b1 = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    b2 = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    eps = pd.Series(rng.normal(0.0, 0.0005, n), index=idx)
    candidate = 0.001 + 0.3 * b1 + 0.2 * b2 + eps

    module = factor_spanning.build_module(_spec(), _claim())
    out = module.run(_bundle(candidate, b1, b2), _claim(), 0.0)

    assert out["ok"] is True
    assert out["n_trades"] == n
    assert out["bars_per_year"] > 250
    assert abs(out["regression"]["alpha"] - 0.001) < 0.0001
    assert abs(out["regression"]["betas"]["b1"] - 0.3) < 0.02
    assert abs(out["regression"]["betas"]["b2"] - 0.2) < 0.02
    assert abs(float(out["net"].mean()) - 0.001) < 0.0002
    assert out["position_exposures"]["candidate"] == 1.0
    assert out["position_exposures"]["b1"] < 0.0
    assert out["positions"].index.equals(out["net"].index)
    assert list(out["positions"].columns) == ["candidate", "b1", "b2"]
    assert float(out["positions"]["candidate"].iloc[0]) == 1.0
    assert float(out["positions"].abs().sum(axis=1).iloc[0]) > 1.0


def test_factor_spanning_freezes_betas_on_in_sample_prefix_no_lookahead():
    rng = np.random.default_rng(12)
    n = 1000
    idx = _idx(n)
    b1 = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    b2 = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    candidate = pd.Series(index=idx, dtype=float)
    cut = int(n * P7.IS_FRAC)
    candidate.iloc[:cut] = 0.4 * b1.iloc[:cut] + 0.1 * b2.iloc[:cut]
    candidate.iloc[cut:] = 1.3 * b1.iloc[cut:] - 0.7 * b2.iloc[cut:]

    module = factor_spanning.build_module(_spec(), _claim())
    out = module.run(_bundle(candidate, b1, b2), _claim(), 0.0)

    assert out["ok"] is True
    assert abs(out["regression"]["betas"]["b1"] - 0.4) < 1e-10
    assert abs(out["regression"]["betas"]["b2"] - 0.1) < 1e-10
    assert out["regression"]["oos_start"] == idx[cut].isoformat()


def test_factor_spanning_aligns_candidate_and_benchmarks_by_common_dates():
    idx = _idx(50)
    b1 = pd.Series(np.linspace(0.0, 1.0, 50), index=idx)
    b2 = pd.Series(np.linspace(1.0, 0.0, 50), index=idx)
    candidate = 0.2 * b1 + 0.1 * b2
    b2 = b2.iloc[5:]

    module = factor_spanning.build_module(_spec(), _claim())
    out = module.run(_bundle(candidate, b1, b2), _claim(), 0.0)

    assert out["ok"] is True
    assert out["n_trades"] == 45
    assert out["net"].index.min() == idx[5]


def test_factor_spanning_classifier_and_deterministic_spec(monkeypatch):
    fake_catalog = types.SimpleNamespace(
        available=lambda: [
            "my_candidate_factor",
            "us_equity_ff3_mkt_rf",
            "us_equity_ff3_smb",
            "us_equity_ff3_hml",
        ]
    )
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim(
        "my_candidate_factor earns alpha after controlling for Fama-French three-factor benchmarks."
    )

    assert fidelity_memory.classify_claim_type(claim) == "factor_spanning"
    assert fidelity_memory.classify_claim_type(
        _claim("A factor momentum trading strategy goes long winners and short losers.")
    ) == "trading_strategy"

    spec = spec_gen._factor_spanning_spec(claim, _source())
    assert spec["claim_type"] == "factor_spanning"
    assert spec["candidate_factor"] == "my_candidate_factor"
    assert spec["benchmark_set"] == "ff3"
    assert spec["benchmark_factors"] == [
        "us_equity_ff3_mkt_rf",
        "us_equity_ff3_smb",
        "us_equity_ff3_hml",
    ]


def test_factor_spanning_fidelity_backstop_and_binding_review(monkeypatch):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["deterministic spec-only module"],
            "note": "structural false positive",
        }, response)

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    claim = _claim("candidate earns alpha after controlling for FF3 factors.")
    spec = {
        # ff3-consistent benchmark cardinality (FS-1): a declared benchmark_set must list its full
        # canonical factor count (stub module below, so no bundle data needed).
        **_spec(inputs=["candidate", "b1", "b2", "b3"], benchmark_factors=["b1", "b2", "b3"]),
        "claim_statement": claim.statement,
        "claim_mechanism": claim.mechanism,
        "binding_provenance": {
            "candidate_factor": {
                "kind": "literal",
                "series": "candidate",
                "score": 1.0,
                "matched_tokens": ["candidate"],
                "full_coverage": True,
                "unmatched_name_tokens": [],
                "description": "candidate",
            },
            "benchmark_set": {"kind": "benchmark_set", "confirmed": True},
        },
    }

    out = fidelity.assess(claim, "def run(bundle, claim, cost): return {}", spec=spec)
    assert out["faithful"] is True
    assert out["factor_spanning_fidelity_override"] == "deterministic_template_structural"

    unresolved = {
        **spec,
        "inputs": ["b1", "b2"],
        "candidate_factor": "",
        "unknowns": ["candidate factor series was not resolved from literal/prose bindings"],
        "binding_provenance": {
            "candidate_factor": {"kind": "unresolved", "description": "mystery factor"},
            "benchmark_set": {"kind": "benchmark_set", "confirmed": True},
        },
    }
    review = runmod._factor_spanning_binding_review(claim, unresolved)
    assert review is not None
    assert review["reason"] == "factor_spanning_binding_unresolved"
    assert "mystery factor" in review["explanation"]["why"]


def test_fs3_strategy_framed_candidate_detected():
    """FS-3: a self-described tradeable-strategy candidate is detected (-> qualified/capped verdict); a
    plain factor is not."""
    from penrose.pipeline import stages

    class _C:
        def __init__(self, s, m=""):
            self.statement = s
            self.mechanism = m
    assert stages._factor_spanning_candidate_is_strategy(
        _C("Our long-short momentum strategy, rebalanced monthly, generates alpha after controlling for FF3.")) is True
    assert stages._factor_spanning_candidate_is_strategy(
        _C("The quality factor earns alpha after controlling for FF3.")) is False
