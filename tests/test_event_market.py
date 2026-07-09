from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from penrose.data.event_market import EVENT_MARKET_COLUMNS, EventMarketPanel
from penrose.data.event_market_load import EventMarketDataUnavailable, load_event_market
from penrose.pipeline import event_market
from penrose.pipeline.event_market_backtest import run_event_market_backtest, run_weather_tail_fade_backtest


def _panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e2", "e1", "e3"],
            "decision_time": [
                "2024-01-02 09:30",
                "2024-01-01 09:30",
                pd.Timestamp("2024-01-03 09:30", tz="America/New_York"),
            ],
            "close_time": [
                "2024-01-02 21:00",
                "2024-01-01 21:00",
                pd.Timestamp("2024-01-03 21:00", tz="America/New_York"),
            ],
            "strike_low": [0.0, 0.0, 0.0],
            "strike_high": [10.0, 10.0, 10.0],
            "entry_price": [0.50, 0.40, 0.35],
            "outcome": [1, 1, 0],
            "underlying": [0.70, 0.80, 0.20],
        }
    )


def test_event_market_contract_rejects_invalid_outcome():
    df = _panel_frame()
    df.loc[0, "outcome"] = 2
    with pytest.raises(ValueError, match="outcome must be in"):
        EventMarketPanel("bad", df, "unit-test")


def test_event_market_contract_rejects_entry_price_outside_unit_interval():
    df = _panel_frame()
    df.loc[0, "entry_price"] = 1.01
    with pytest.raises(ValueError, match="entry_price must be in"):
        EventMarketPanel("bad", df, "unit-test")


def test_event_market_contract_rejects_close_before_decision():
    df = _panel_frame()
    df.loc[0, "close_time"] = "2024-01-02 08:00"
    with pytest.raises(ValueError, match="close_time must be >= decision_time"):
        EventMarketPanel("bad", df, "unit-test")


def test_event_market_contract_sorts_by_decision_time_and_localizes_utc():
    panel = EventMarketPanel("ok", _panel_frame(), "unit-test")
    assert list(panel.data["event_id"]) == ["e1", "e2", "e3"]
    assert list(panel.data.columns) == EVENT_MARKET_COLUMNS
    assert str(panel.data["decision_time"].dt.tz) == "UTC"
    assert str(panel.data["close_time"].dt.tz) == "UTC"
    assert panel.kind == "event_market"
    assert panel.coverage == ("2024-01-01T09:30:00+00:00", "2024-01-03T14:30:00+00:00", 3, 3)


def _synthetic_panel(n: int = 80) -> EventMarketPanel:
    rows = []
    for i in range(n):
        prob = 0.70 if i % 2 == 0 else 0.30
        price = 0.50
        outcome = 1 if (i % 10) < int(prob * 10) else 0
        rows.append(
            {
                "event_id": f"event-{i:03d}",
                "decision_time": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "close_time": pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(hours=i),
                "strike_low": 0.0,
                "strike_high": 1.0,
                "entry_price": price,
                "outcome": outcome,
                "underlying": prob,
            }
        )
    return EventMarketPanel("synthetic", pd.DataFrame(rows), "unit-test")


def _truth_model(underlying, strike_low, strike_high, params):
    del strike_low, strike_high, params
    return float(underlying)


def _bad_model(underlying, strike_low, strike_high, params):
    del underlying, strike_low, strike_high, params
    return 0.71


def test_event_market_backtest_calibrated_model_positive_mean_net():
    net, positions, bars_per_year, stats = run_event_market_backtest(
        _synthetic_panel(), _truth_model, min_ev=0.05, max_price=0.75,
        kelly_fraction=1.0, size_cap=1.0,
    )
    assert stats["n_trades"] == 40
    assert stats["n_events"] == 80
    assert stats["mean_net"] > 0
    assert float(net.mean()) > 0
    # L-1: positions returned + aligned to net; bars_per_year is trades-per-year (>> a calendar 252).
    assert list(positions.index) == list(net.index)
    assert (positions > 0).all()
    assert bars_per_year > 1000  # 40 trades over ~2 days -> thousands/yr, not 252
    assert stats["bars_per_year"] == bars_per_year


def test_event_market_backtest_no_edge_model_nonpositive_after_fee():
    net, positions, _, stats = run_event_market_backtest(
        _synthetic_panel(), _bad_model, min_ev=0.05, max_price=0.75,
        kelly_fraction=1.0, size_cap=1.0,
    )
    assert stats["n_trades"] == 80
    assert stats["mean_net"] <= 0
    assert float(net.mean()) <= 0


def test_event_market_backtest_skips_zero_size_trades():
    # L-2: a well-calibrated model (prob == market price) has zero edge -> zero Kelly size -> no trades,
    # rather than a pile of zero-net observations inflating the trade count / t-stat denominator.
    net, positions, _, stats = run_event_market_backtest(
        _synthetic_panel(), lambda u, lo, hi, p: 0.50, min_ev=0.0, max_price=1.0,
    )
    assert stats["n_trades"] == 0
    assert net.empty and positions.empty


def test_event_market_backtest_empty_panel_no_crash():
    empty = EventMarketPanel("empty", pd.DataFrame(), "unit-test")
    net, positions, bars_per_year, stats = run_event_market_backtest(empty, _truth_model)
    assert net.empty and positions.empty
    assert str(net.index.tz) == "UTC"
    assert stats["n_trades"] == 0 and bars_per_year == 1.0


def test_event_market_backtest_returns_utc_close_time_series_for_p7_shape():
    net, positions, _, stats = run_event_market_backtest(_synthetic_panel(20), _truth_model, min_ev=0.05)
    assert stats["n_trades"] == 10
    assert str(net.index.tz) == "UTC"
    assert net.index.name == "close_time"
    assert np.isfinite(net.mean())
    assert np.isfinite(net.std(ddof=1))


def test_event_market_backtest_deterministic_same_inputs():
    panel = _synthetic_panel()
    a_net, a_pos, a_bpy, a_stats = run_event_market_backtest(panel, _truth_model, min_ev=0.05, seed=123)
    b_net, b_pos, b_bpy, b_stats = run_event_market_backtest(panel, _truth_model, min_ev=0.05, seed=123)
    pd.testing.assert_series_equal(a_net, b_net)
    pd.testing.assert_series_equal(a_pos, b_pos)
    assert a_bpy == b_bpy and a_stats == b_stats


def _normal_table(n: int = 40, *, entry_offset: float = -0.18) -> pd.DataFrame:
    import math

    low, high = -0.5, 1.0
    prob = 0.5 * (1.0 + math.erf(high / math.sqrt(2.0))) - 0.5 * (1.0 + math.erf(low / math.sqrt(2.0)))
    wins_per_10 = int(round(prob * 10))
    rows = []
    for i in range(n):
        decision = pd.Timestamp("2024-02-01", tz="UTC") + pd.Timedelta(days=i)
        rows.append({
            "event_id": f"normal-{i:03d}",
            "decision_time": decision.isoformat(),
            "close_time": (decision + pd.Timedelta(hours=8)).isoformat(),
            "strike_low": low,
            "strike_high": high,
            "entry_price": prob + entry_offset,
            "outcome": 1 if (i % 10) < wins_per_10 else 0,
            "underlying": json.dumps({"mu": 0.0, "sigma": 1.0}),
            "underlying_time": decision.isoformat(),
        })
    return pd.DataFrame(rows)


def _event_spec(path, *, data_dir=None, entry_offset: float = -0.18):
    del entry_offset
    spec = {
        "module_id": "unit_event_market",
        "strategy_class": "unit_event_market",
        "claim_type": "event_market_strategy",
        "event_market": {"path": str(path), "name": "unit_brackets"},
        "pricing_model": {"family": "normal_bracket"},
        "entry": {"min_ev": 0.05, "max_price": 0.90, "kelly_fraction": 0.75, "size_cap": 1.0},
    }
    if data_dir is not None:
        spec["data_dir"] = str(data_dir)
    return spec


def test_load_event_market_builds_panel_and_rejects_lookahead(tmp_path):
    table = tmp_path / "brackets.csv"
    _normal_table().to_csv(table, index=False)

    panel = load_event_market(_event_spec("brackets.csv", data_dir=tmp_path), tmp_path)
    assert isinstance(panel, EventMarketPanel)
    assert panel.kind == "event_market"
    assert panel.data["underlying"].map(lambda x: x["sigma"]).eq(1.0).all()
    assert panel.coverage[2:] == (40, 40)

    bad = _normal_table()
    bad.loc[0, "underlying_time"] = (
        pd.Timestamp(bad.loc[0, "decision_time"]) + pd.Timedelta(seconds=1)
    ).isoformat()
    bad_path = tmp_path / "lookahead.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(EventMarketDataUnavailable, match="after decision_time"):
        load_event_market(_event_spec(bad_path), tmp_path)


def test_load_event_market_missing_declared_table_is_graceful(tmp_path):
    with pytest.raises(EventMarketDataUnavailable, match="data_unavailable: event_market_table"):
        load_event_market(_event_spec("missing.csv", data_dir=tmp_path), tmp_path)


def test_event_market_module_evaluate_returns_p7_shape_and_is_deterministic(tmp_path):
    table = tmp_path / "brackets.csv"
    _normal_table(80).to_csv(table, index=False)
    claim = type("Claim", (), {"claim_id": "em-c1", "applicable_strategy_class": "unit_event_market"})()
    module = event_market.build_module(_event_spec(table), claim)

    out_a = module.evaluate()
    out_b = module.evaluate()

    assert set(out_a) == {"net", "positions", "bars_per_year", "n_trades"}
    assert out_a["n_trades"] > 0
    assert float(out_a["net"].mean()) > 0
    assert out_a["positions"].index.equals(out_a["net"].index)
    assert getattr(module, "__auto_generated__", None) is False
    pd.testing.assert_series_equal(out_a["net"], out_b["net"])
    pd.testing.assert_series_equal(out_a["positions"], out_b["positions"])
    assert out_a["bars_per_year"] == out_b["bars_per_year"]
    assert out_a["n_trades"] == out_b["n_trades"]


def test_event_market_module_param_override_changes_entry_threshold(tmp_path):
    table = tmp_path / "brackets.csv"
    _normal_table(80).to_csv(table, index=False)
    claim = type("Claim", (), {"claim_id": "em-c1", "applicable_strategy_class": "unit_event_market"})()
    module = event_market.build_module(_event_spec(table), claim)

    base = module.run(None, claim, 0.0)
    blocked = module.run(None, claim, 0.0, param_override={"min_ev": 0.50})

    assert getattr(module, "__supports_param_override__", None) is True
    assert base["ok"] is True and base["n_trades"] > 0
    assert blocked["ok"] is True and blocked["n_trades"] == 0


def test_event_market_module_param_override_changes_normal_bracket_params(tmp_path):
    table = tmp_path / "brackets.csv"
    _normal_table(80).to_csv(table, index=False)
    claim = type("Claim", (), {"claim_id": "em-c1", "applicable_strategy_class": "unit_event_market"})()
    module = event_market.build_module(_event_spec(table), claim)

    base = module.run(None, claim, 0.0)
    shifted = module.run(None, claim, 0.0, param_override={"mu": -10.0})

    assert base["ok"] is True and base["n_trades"] > 0
    assert shifted["ok"] is True and shifted["n_trades"] == 0


def test_event_market_module_noedge_returns_no_positive_net(tmp_path):
    table = tmp_path / "brackets.csv"
    _normal_table(80, entry_offset=0.0).to_csv(table, index=False)
    claim = type("Claim", (), {"claim_id": "em-c1", "applicable_strategy_class": "unit_event_market"})()
    spec = _event_spec(table)
    spec["entry"]["min_ev"] = 0.0
    module = event_market.build_module(spec, claim)

    out = module.evaluate()

    assert out["n_trades"] == 0
    assert float(out["net"].sum()) <= 0.0


def _weather_tail_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["KXW-CHI-1", "KXW-NYC-1", "KXW-BOS-1"],
        "city": ["CHI", "NYC", "BOS"],
        "close_date": ["2024-07-01", "2024-07-02", "2024-07-02"],
        "p_close": [0.25, 0.20, 0.20],
        "outcome": [0, 0, 1],
        "volume": [100.0, 100.0, 100.0],
        "open_interest": [50.0, 100.0, 100.0],
        "is_tail": [True, True, True],
        "underlying_time": ["2024-07-01", "2024-07-02", "2024-07-02"],
    })


def test_weather_tail_raw_table_loads_through_event_market_contract(tmp_path):
    table = tmp_path / "weather.csv"
    _weather_tail_rows().to_csv(table, index=False)

    panel = load_event_market({
        "claim_type": "event_market_strategy",
        "event_market": {"path": str(table), "name": "weather"},
    }, tmp_path)

    assert isinstance(panel, EventMarketPanel)
    assert list(panel.data.columns) == EVENT_MARKET_COLUMNS
    assert panel.data.loc[0, "event_id"] == "KXW-CHI-1"
    assert panel.data.loc[0, "underlying"]["city"] == "CHI"
    assert panel.data.loc[0, "underlying"]["open_interest"] == 50.0


def test_weather_tail_fade_reconstructs_fee_capacity_and_pair_cap():
    panel = EventMarketPanel("weather", _weather_tail_rows(), "unit-test")

    net, positions, bars_per_year, stats = run_weather_tail_fade_backtest(
        panel,
        fee_coeff=0.07,
        capacity_frac=0.10,
        max_pair_gross=0.01,
        pair_cities=["NYC", "BOS"],
        portfolio_notional=1000.0,
    )

    fee_25 = 0.07 * 0.25 * 0.75
    expected_day1 = (5.0 * (0.25 - fee_25)) / 1000.0
    fee_20 = 0.07 * 0.20 * 0.80
    expected_day2 = (5.0 * (0.20 - fee_20) + 5.0 * (-0.80 - fee_20)) / 1000.0
    assert np.isclose(float(net.iloc[0]), expected_day1)
    assert np.isclose(float(net.iloc[1]), expected_day2)
    assert np.isclose(float(positions.loc[net.index[1], "NYC"]), 0.005)
    assert np.isclose(float(positions.loc[net.index[1], "BOS"]), 0.005)
    assert stats["pair_gross_before_cap"] == 20.0
    assert stats["pair_gross_after_cap"] == 10.0
    assert bars_per_year > 300


def test_weather_tail_module_reports_measured_cost_provenance(tmp_path):
    table = tmp_path / "weather.csv"
    _weather_tail_rows().to_csv(table, index=False)
    claim = type("Claim", (), {"claim_id": "weather-c1", "applicable_strategy_class": "kalshi_weather_tail_fade"})()
    spec = {
        "module_id": "unit_weather_tail",
        "strategy_class": "kalshi_weather_tail_fade",
        "claim_type": "event_market_strategy",
        "primitive": "kalshi_weather_tail_fade",
        "event_market": {"path": str(table), "name": "weather"},
        "entry": {"capacity_frac": 0.10, "fee_coeff": 0.07, "portfolio_notional": 1000.0},
    }

    out = event_market.build_module(spec, claim).run(None, claim, 0.0)

    assert out["ok"] is True
    assert out["cost_provenance"] == "measured"
    assert out["capacity_provenance"] == "reconstructed_from_volume_open_interest"
    assert out["event_market"]["primitive"] == "kalshi_weather_tail_fade"


def test_event_market_run_missing_table_returns_data_unavailable(tmp_path):
    claim = type("Claim", (), {"claim_id": "em-c1", "applicable_strategy_class": "unit_event_market"})()
    module = event_market.build_module(_event_spec("missing.csv", data_dir=tmp_path), claim)

    out = module.run(None, claim, 0.0)

    assert out["ok"] is False
    assert out["reason"].startswith("data_unavailable: event_market_table")


def test_event_market_strategy_routing_reaches_deterministic_builder(tmp_path, monkeypatch):
    from penrose import concepts, config
    from penrose.brain import Claim
    from penrose.data.contract import DataBundle
    from penrose.pipeline import fidelity_memory, p7_backtest, run as runmod

    for attr, path in {
        "DECISIONS_LOG": tmp_path / "decisions.jsonl",
        "REVIEW_QUEUE": tmp_path / "review_queue.jsonl",
        "DATA_REQUESTS": tmp_path / "data_requests.jsonl",
        "ANALYSIS_INDEX": tmp_path / "reports" / "analysis_index.jsonl",
        "PROCESSED_PAPERS": tmp_path / "processed_papers.json",
        "REPORTS": tmp_path / "reports",
        "LIVE_JSON": tmp_path / "dashboard" / "live.json",
        "PROGRESS_JSON": tmp_path / "dashboard" / "progress.json",
        "ARCHIVES": tmp_path / "archives",
        "LLM_CACHE_DIR": tmp_path / ".llm_cache",
        "MODULES": tmp_path / "modules",
        "AUTO_MODULES": tmp_path / "modules" / "_auto",
        "FIDELITY_REJECTIONS": tmp_path / "reports" / "fidelity_rejections.jsonl",
    }.items():
        monkeypatch.setattr(config, attr, path)
    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", True)
    monkeypatch.setattr(p7_backtest, "LEDGER", tmp_path / "backtest_ledger.tsv")
    (tmp_path / "modules").mkdir(parents=True)

    statement = (
        "A Kalshi prediction-market bracket strategy uses the declared bracket pricing "
        "model normal_bracket with mu and sigma to buy underpriced strikes."
    )
    claim = Claim(
        claim_id="em-route",
        statement=statement,
        mechanism="declared pricing model normal_bracket",
        scope="unit",
        horizon="settlement",
        source_id="em",
        source_span=statement,
        claimed_metric_quote="declared bracket pricing model",
        applicable_strategy_class="unit_event_market",
    )
    assert fidelity_memory.classify_claim_type(claim) == "event_market_strategy"

    paper = tmp_path / "paper.md"
    paper.write_text(statement)
    table = tmp_path / "brackets.csv"
    _normal_table(100).to_csv(table, index=False)
    spec = _event_spec(table)
    calls = []

    class FakeEventModule:
        __module_id__ = "unit_event_market"
        __strategy_class__ = "unit_event_market"
        __auto_generated__ = False
        __file__ = __file__

        def run(self, bundle, claim, cost_frac):
            idx = pd.date_range("2024-01-01", periods=100, freq="D")
            net = pd.Series(np.full(100, 0.01), index=idx)
            return {
                "ok": True,
                "net": net,
                "positions": pd.Series(1.0, index=idx),
                "bars_per_year": 252.0,
                "n_trades": 100,
            }

    def fake_build_module(got_spec, got_claim):
        calls.append((got_spec.get("claim_type"), got_claim.claim_id))
        return FakeEventModule()

    runmod.REGISTRY.clear()
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_MODULES.clear()
    monkeypatch.setattr(runmod.spec_gen, "generate_spec", lambda *a, **k: dict(spec))
    monkeypatch.setattr(runmod.event_market, "build_module", fake_build_module)
    monkeypatch.setattr(runmod.impl_gen, "try_implement",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("event_market_strategy must skip auto-impl")))
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(concepts, "extract_and_append", lambda *a, **k: None)

    out = runmod.run_source(
        paper,
        use_llm=False,
        claims_override=[claim],
        bundle_override=DataBundle(series={}),
        force=True,
    )

    assert calls == [("event_market_strategy", "em-route")]
    assert out["decisions"][0]["claim_id"] == "em-route"
    assert out["decisions"][0]["verdict"] in {"kill", "underpowered", "watch", "research-supported", "needs_review"}


def test_event_market_fidelity_structural_override_for_declared_normal_model(tmp_path, monkeypatch):
    from penrose.pipeline import fidelity

    table = tmp_path / "brackets.csv"
    _normal_table(10).to_csv(table, index=False)
    claim = type(
        "Claim",
        (),
        {
            "statement": "Kalshi event-market brackets use declared pricing model normal_bracket.",
            "mechanism": "normal_bracket declared model",
            "resolved_claim_type": "event_market_strategy",
        },
    )()

    def false_unfaithful(*args, **kwargs):
        return ({"faithful": False, "confidence": 0.95,
                 "divergences": ["verifier drift"], "note": "verifier drift"},
                type("Resp", (), {"independent_verifier": False})())

    monkeypatch.setattr(fidelity.llm, "call_json", false_unfaithful)
    out = fidelity.assess(claim, "def run(): pass", spec=_event_spec(table))

    assert out["faithful"] is True
    assert out["verified"] is True
    assert out["event_market_fidelity_override"] == "deterministic_declared_model_structural"


def test_event_market_deterministic_spec_denied_when_added_gate_present():
    """M-1: a spec that declares a bracket table + normal_bracket BUT also carries an added gate the
    deterministic executor drops must NOT qualify for the structural fidelity override (mirrors the
    provided_series discipline) — the refuter's unfaithful verdict must stand."""
    from penrose.pipeline import fidelity
    clean = {"claim_type": "event_market_strategy", "event_market": {"table": "x.parquet"},
             "pricing_model": {"family": "normal_bracket"}, "entry": {"min_ev": 0.1, "max_price": 0.8}}
    assert fidelity._is_deterministic_event_market_spec(clean) is True
    gated = dict(clean); gated["significance"] = {"alpha": 0.05, "method": "bonferroni"}
    assert fidelity._is_deterministic_event_market_spec(gated) is False
