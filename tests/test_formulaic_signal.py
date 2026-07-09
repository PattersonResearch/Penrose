import numpy as np
import pandas as pd
import pytest

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import fidelity, fidelity_memory, formulaic_signal, spec_gen


def _claim(statement: str = "signal: price(sig); trade_series: trade") -> Claim:
    return Claim(
        claim_id="unit_formulaic",
        statement=statement,
        mechanism="",
        scope="",
        horizon="",
        source_id="unit",
        source_span=statement,
        claimed_metric_quote="",
        applicable_strategy_class="formulaic_signal",
    )


def _series(name: str, values) -> Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return Series(name, pd.Series(values, index=idx), "unit", "test")


def test_formulaic_executor_applies_one_bar_lag_and_funding_cashflow():
    trade = np.array([100.0, 101.0, 99.0, 100.0, 102.0, 101.0])
    sig = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    funding = np.array([0.0, 0.001, 0.002, -0.001, 0.003, 0.0])
    spec = {
        "claim_type": "formulaic_signal",
        "strategy_class": "formulaic_signal",
        "trade_series": "trade",
        "signal": "price(sig)",
        "position_map": "sign",
        "funding_pnl_series": "fund",
        "inputs": ["sig", "trade", "fund"],
    }
    bundle = DataBundle(series={
        "trade": _series("trade", trade),
        "sig": _series("sig", sig),
        "fund": _series("fund", funding),
    })
    out = formulaic_signal.build_module(spec, _claim()).run(bundle, _claim(), 0.0)
    assert out["ok"] is True
    idx = bundle.series["trade"].data.index.tz_localize("UTC")
    ret = pd.Series(trade, index=idx).pct_change()
    pos = pd.Series(sig, index=idx).shift(1)
    expected = (pos * (ret - pd.Series(funding, index=idx))).dropna()
    pd.testing.assert_series_equal(out["net"], expected.rename("formulaic_signal_net"))


@pytest.mark.parametrize(
    "expr",
    [
        "lag(sig, -1)",
        "sig.__class__",
        "sig[0]",
        "(lambda x: x)(sig)",
        "rolling_max(sig, 5)",
    ],
)
def test_formulaic_parser_rejects_unsafe_or_unapproved_syntax(expr):
    spec = {
        "claim_type": "formulaic_signal",
        "trade_series": "trade",
        "signal": expr,
        "inputs": ["sig", "trade"],
    }
    bundle = DataBundle(series={"trade": _series("trade", [100, 101, 102]), "sig": _series("sig", [1, 2, 3])})
    out = formulaic_signal.build_module(spec, _claim()).run(bundle, _claim(), 0.0)
    assert out["ok"] is False
    assert out["reason"].startswith("formulaic_signal_invalid:")


def test_formulaic_classifier_spec_and_fidelity_correspondence():
    statement = (
        "Formulaic crypto rule; signal: returns(trade, 5) - rolling_sum(fund, 5); "
        "trade_series: trade; funding_pnl_series: fund; position_map: sign"
    )
    claim = _claim(statement)
    assert fidelity_memory.classify_claim_type(claim) == "formulaic_signal"
    source = type("Source", (), {"source_id": "unit_source", "text": ""})()
    spec = spec_gen._formulaic_spec(claim, source)
    assert spec["claim_type"] == "formulaic_signal"
    assert set(spec["inputs"]) == {"trade", "fund"}
    assert fidelity._is_deterministic_formulaic_signal_spec(spec) is True
    bad = dict(spec, inputs=["trade"])
    assert fidelity._is_deterministic_formulaic_signal_spec(bad) is False
