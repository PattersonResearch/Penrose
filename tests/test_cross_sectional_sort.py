import json
import types

import numpy as np
import pandas as pd
import pytest

from penrose.brain import Claim
from penrose.data.panel_load import PanelDataUnavailable, load_cross_sectional_sort_panels
from penrose.pipeline import cross_sectional_sort, fidelity, fidelity_memory, run as runmod, spec_gen
from penrose.pipeline.p1_ingest import IngestedSource


def _dates(n: int = 24) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME", tz="UTC")


def _claim(statement: str | None = None) -> Claim:
    return Claim(
        claim_id="css-test",
        statement=statement or (
            "Stocks sorted by book-to-market into deciles earn a top-minus-bottom spread."
        ),
        mechanism="cross-sectional characteristic sort",
        scope="synthetic equities",
        horizon="monthly",
        source_id="test",
        source_span="",
        claimed_metric_quote="mean spread",
        applicable_strategy_class="cross_sectional_sort",
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


def _write_wide(path, data: pd.DataFrame) -> None:
    out = data.copy()
    out.insert(0, "date", out.index.astype(str))
    out.to_csv(path, index=False)


def _spec(tmp_path, returns: pd.DataFrame, characteristic: pd.DataFrame, **overrides) -> dict:
    ret_path = tmp_path / "returns.csv"
    char_path = tmp_path / "characteristic.csv"
    _write_wide(ret_path, returns)
    _write_wide(char_path, characteristic)
    spec = {
        "module_id": "unit_cross_sectional_sort",
        "strategy_class": "cross_sectional_sort",
        "claim_type": "cross_sectional_sort",
        "panel_inputs": {
            "returns": {"path": str(ret_path), "survivorship": "corrected", "name": "returns"},
            "characteristic": {"path": str(char_path), "name": "book_to_market"},
        },
        "characteristic": "book-to-market",
        "n_buckets": 2,
        "rebalance": "ME",
        "hold": "1M",
        "min_names": 4,
        "_llm_mode": "deterministic-template",
    }
    spec.update(overrides)
    return spec


def _small_panels():
    dates = _dates(6)
    names = ["a", "b", "c", "d"]
    characteristic = pd.DataFrame(index=dates, columns=names, dtype=float)
    characteristic.iloc[:] = [1.0, 2.0, 3.0, 4.0]
    returns = pd.DataFrame(0.0, index=dates, columns=names)
    returns.iloc[1, :] = [0.00, 0.01, 0.03, 0.04]
    returns.iloc[2:, :] = 0.01
    return returns, characteristic


def test_panel_loader_reads_declared_wide_tables_and_requires_survivorship(tmp_path):
    returns, characteristic = _small_panels()
    spec = _spec(tmp_path, returns, characteristic)

    ret_panel, char_panel = load_cross_sectional_sort_panels(spec, tmp_path)

    assert ret_panel.kind == "return"
    assert char_panel.kind == "characteristic"
    assert ret_panel.coverage[3] == 4
    assert "survivorship=corrected" in ret_panel.provenance

    bad = {**spec, "panel_inputs": dict(spec["panel_inputs"])}
    bad["panel_inputs"]["returns"] = dict(bad["panel_inputs"]["returns"])
    bad["panel_inputs"]["returns"].pop("survivorship")
    with pytest.raises(PanelDataUnavailable, match="survivorship must be declared"):
        load_cross_sectional_sort_panels(bad, tmp_path)


def test_cross_sectional_sort_executor_emits_form_factor_spread_and_positions(tmp_path):
    returns, characteristic = _small_panels()
    module = cross_sectional_sort.build_module(_spec(tmp_path, returns, characteristic), _claim())

    out = module.run(None, _claim(), 0.0)

    assert out["ok"] is True
    assert out["bars_per_year"] == 12.0
    assert out["n_trades"] > 0
    assert abs(float(out["net"].iloc[0]) - 0.03) < 1e-12
    assert out["positions"].index.equals(out["net"].index)
    assert abs(float(out["positions"].abs().sum(axis=1).iloc[0]) - 1.0) < 1e-12
    assert out["sort"]["membership"]["2020-01-31"] == {"high": ["c", "d"], "low": ["a", "b"]}


def test_cross_sectional_sort_ranking_uses_characteristics_known_at_rebalance(tmp_path):
    returns, characteristic = _small_panels()
    base = cross_sectional_sort.build_module(
        _spec(tmp_path, returns, characteristic), _claim()
    ).run(None, _claim(), 0.0)

    changed_characteristic = characteristic.copy()
    changed_characteristic.iloc[1:, :] = [4.0, 3.0, 2.0, 1.0]
    changed = cross_sectional_sort.build_module(
        _spec(tmp_path, returns, changed_characteristic), _claim()
    ).run(None, _claim(), 0.0)

    assert base["ok"] is True and changed["ok"] is True
    assert base["sort"]["membership"]["2020-01-31"] == changed["sort"]["membership"]["2020-01-31"]
    assert base["net"].iloc[0] == changed["net"].iloc[0]


def test_cross_sectional_sort_survivorship_includes_delisted_entity_while_alive(tmp_path):
    dates = _dates(5)
    names = ["a", "b", "c", "delisted"]
    characteristic = pd.DataFrame(index=dates, columns=names, dtype=float)
    characteristic.iloc[:] = [1.0, 2.0, 3.0, 4.0]
    returns = pd.DataFrame(0.0, index=dates, columns=names)
    returns.loc[dates[1], "delisted"] = 0.10
    returns.loc[dates[2]:, "delisted"] = np.nan

    module = cross_sectional_sort.build_module(_spec(tmp_path, returns, characteristic), _claim())
    out = module.run(None, _claim(), 0.0)

    assert out["ok"] is True
    assert "delisted" in out["sort"]["membership"]["2020-01-31"]["high"]
    assert "delisted" in out["positions"].columns
    assert out["positions"].loc[dates[1], "delisted"] > 0


def test_cross_sectional_sort_classifier_spec_and_strategy_guard(monkeypatch):
    fake_catalog = types.SimpleNamespace(available=lambda: [])
    monkeypatch.setattr(spec_gen, "load_catalog_loader", lambda data_dir: fake_catalog)
    claim = _claim("Stocks sorted by book-to-market into deciles earn a top-minus-bottom spread.")

    assert fidelity_memory.classify_claim_type(claim) == "cross_sectional_sort"
    assert fidelity_memory.classify_claim_type(
        _claim("A long-short momentum trading strategy has Sharpe 1.2 after rebalancing monthly.")
    ) == "trading_strategy"

    spec = spec_gen._cross_sectional_sort_spec(claim, _source())
    assert spec["claim_type"] == "cross_sectional_sort"
    assert spec["characteristic"] == "book-to-market"
    assert spec["n_buckets"] == 10
    assert spec["panel_inputs"]["returns"]["survivorship"] == "corrected"
    assert runmod._cross_sectional_sort_binding_review(claim, spec)["reason"] == (
        "cross_sectional_sort_binding_unconfirmed"
    )


def test_cross_sectional_sort_fidelity_backstop(monkeypatch, tmp_path):
    def false_unfaithful(*args, **kwargs):
        del args, kwargs
        response = type("Response", (), {"independent_verifier": True})()
        return ({
            "faithful": False,
            "confidence": 0.99,
            "divergences": ["deterministic spec-only module"],
            "note": "structural false positive",
        }, response)

    returns, characteristic = _small_panels()
    spec = {
        **_spec(tmp_path, returns, characteristic, n_buckets=10, min_names=4),
        "claim_statement": "Stocks sorted by book-to-market into deciles earn a top-minus-bottom spread.",
        "claim_mechanism": "cross-sectional sort",
    }
    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)

    out = fidelity.assess(_claim(spec["claim_statement"]), "def run(bundle, claim, cost): return {}", spec=spec)

    assert out["faithful"] is True
    assert out["cross_sectional_sort_fidelity_override"] == "deterministic_template_structural"


def test_missing_spec_inputs_short_circuits_panels_not_bundle(tmp_path):
    returns, characteristic = _small_panels()
    spec = _spec(tmp_path, returns, characteristic)
    bundle = types.SimpleNamespace(get=lambda name: (_ for _ in ()).throw(AssertionError(name)))

    assert runmod._missing_spec_inputs_from_bundle(spec, bundle) == []


def test_cs1_bucket_and_cadence_substitution_rejected():
    """CS-1: the fidelity correspondence rejects a spec whose bucket count or rebalance cadence contradicts
    an explicitly-named scheme in the claim (a quartiles/annual claim reconstructed at deciles/monthly)."""
    from penrose.pipeline import fidelity as F

    class _C:
        def __init__(self, s):
            self.statement = s
            self.mechanism = ""
            self.claimed_metric_quote = ""
            self.source_span = ""

    def _spec(char="momentum", n=10, reb="ME"):
        return {"claim_type": "cross_sectional_sort", "characteristic": char,
                "n_buckets": n, "rebalance": reb, "_llm_mode": "deterministic-template"}

    quart = _C("Stocks sorted by momentum into quartiles earn a top-minus-bottom long-short spread.")
    assert F._cross_sectional_sort_correspondence_verified(quart, _spec(n=10)) is False   # sub
    assert F._cross_sectional_sort_correspondence_verified(quart, _spec(n=4)) is True      # match
    annual = _C("Stocks sorted by book_to_market into deciles annually earn a top minus bottom long-short spread.")
    assert F._cross_sectional_sort_correspondence_verified(
        annual, _spec(char="book_to_market", n=10, reb="ME")) is False                     # cadence sub
    assert F._cross_sectional_sort_correspondence_verified(
        annual, _spec(char="book_to_market", n=10, reb="YE")) is True                      # cadence match
