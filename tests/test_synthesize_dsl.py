import numpy as np
import pandas as pd
import pytest

from penrose import synthesize
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import formulaic_signal, spec_gen


def _caps():
    return {"series": ["funding_btc", "btc_perp_close_daily_5y"]}


def _candidate(signal: str, *, trade_series: str = "btc_perp_close_daily_5y") -> dict:
    return {
        "statement": "A synthesized funding signal predicts next-day crypto perpetual returns.",
        "mechanism": "tentative funding pressure mechanism",
        "scope": "BTC perpetual futures",
        "horizon": "daily",
        "strategy_class": "funding_pressure",
        "candidate_class": "testable_now",
        "required_series": [],
        "inspired_by": ["node-1"],
        "boundaries": [],
        "falsifier": "net return fails production robustness gates",
        "spec": {
            "signal": signal,
            "trade_series": trade_series,
            "position_map": "zscore_clip",
            "params": {"lookback": 20},
            "param_grid": {"lookback": [10, 20, 60]},
            "conditioning": None,
            "entry_exit": "Hold one-bar-lagged clipped signal exposure to trade_series returns.",
            "horizon": "daily",
        },
    }


def _graph():
    return {
        "nodes": [
            {
                "node_id": "node-1",
                "level": "family_principle",
                "data_provenance": {
                    "data_domains": ["crypto"],
                    "datasets": ["unit"],
                    "periods": [{"start": "2020-01-01", "end": "2020-04-30"}],
                },
            }
        ]
    }


@pytest.mark.parametrize(
    ("signal", "reason"),
    [
        ("rank(funding_btc)", "signal not valid DSL: unsupported function rank"),
        ("ma(funding_btc, 5)", "signal not valid DSL: unsupported function ma"),
        ("zscore(close, 20)", "unknown series: close"),
    ],
)
def test_synth_reconstructability_rejects_non_dsl_or_unknown_series(signal, reason):
    ok, got = synthesize._reconstructability(_candidate(signal), _caps())
    assert ok is False
    assert got == reason


@pytest.mark.parametrize("signal", ["returns(x)", "sign(x, y)", "zscore(x)"])
def test_formulaic_validate_signal_rejects_bad_arity(signal):
    with pytest.raises(formulaic_signal.FormulaicSignalError):
        formulaic_signal.validate_signal(signal)


def test_formulaic_validate_signal_accepts_declared_arity():
    formulaic_signal.validate_signal("returns(x, 14)")


def test_synthesized_dsl_candidate_emits_runnable_formulaic_spec():
    candidate = _candidate("zscore(funding_btc, 20) - rolling_mean(funding_btc, 60)")

    ok, reason = synthesize._reconstructability(candidate, _caps())
    assert ok is True
    assert reason == ""

    claims, normalized = synthesize.normalize("unit-synth", [candidate], _graph(), _caps())
    row = normalized[0]
    assert row["admitted"] is True
    emitted = row["pipeline_spec"]
    assert emitted["claim_type"] == "formulaic_signal"
    assert emitted["signal"] == "zscore(funding_btc, 20) - rolling_mean(funding_btc, 60)"
    assert emitted["trade_series"] == "btc_perp_close_daily_5y"
    assert emitted["position_map"] == "zscore_clip"
    assert emitted["inputs"] == ["btc_perp_close_daily_5y", "funding_btc"]
    assert emitted["grid"] == {"lookback": [10, 20, 60]}
    assert emitted["param_grid"] == {"lookback": [10, 20, 60]}

    generated = spec_gen._embedded_formulaic_signal_spec(claims[0])
    assert generated is not None
    assert generated["claim_type"] == "formulaic_signal"
    assert generated["param_grid"] == {"lookback": [10, 20, 60]}

    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    funding = pd.Series(np.sin(np.arange(90) / 4.0), index=idx)
    trade = pd.Series(100.0 + np.cumsum(np.cos(np.arange(90) / 6.0)), index=idx)
    bundle = DataBundle(series={
        "funding_btc": Series("funding_btc", funding, "unit", "synthetic"),
        "btc_perp_close_daily_5y": Series(
            "btc_perp_close_daily_5y", trade, "unit", "synthetic"
        ),
    })
    out = formulaic_signal.build_module(generated, claims[0]).run(bundle, claims[0], 0.0)
    assert out["ok"] is True
    assert len(out["net"]) > 0
