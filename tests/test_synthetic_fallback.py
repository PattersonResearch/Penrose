import json
from pathlib import Path

import pandas as pd
import pytest


def test_synthetic_macro_fallback_is_edge_free(tmp_path, monkeypatch):
    from modules.macro_vol_btc import impl
    from penrose import config
    from penrose.brain import Claim
    from penrose.data import client
    from penrose.data.contract import DataBundle
    from penrose.pipeline import p7_backtest, stages

    old_ledger = p7_backtest.LEDGER
    old_seed = client._SEED
    p7_backtest.LEDGER = tmp_path / "ledger.tsv"
    claim = Claim(
        "pen-11-syn",
        "synthetic macro signal forecasts future BTC realized volatility",
        "",
        "",
        "5d",
        "unit",
        "span",
        "",
        applicable_strategy_class="macro-signal-volatility-forecast",
    )
    try:
        survivors = 0
        for seed in range(20):
            client._SEED = 20260618 + seed
            syn = client._synthetic_bundle("2023-01-01", "2026-03-31", None)
            bundle = DataBundle(series={
                "kxfed_signal": syn["kxfed_signal"],
                "btc_realized_vol_5d": syn["btc_realized_vol_5d"],
                "btc_implied_vol": syn["btc_implied_vol_syn"],
            })
            mres = impl.run(
                bundle, "fed",
                config.VOL_TRADE_COST["deribit_roundtrip_bps_of_vega"] / 1e4)
            assert mres["ok"]
            bt = p7_backtest.run_backtest(
                f"pen11-syn-{seed}", mres["net"], mres["positions"], mres["bars_per_year"],
                log=False, family="pen11::synthetic")
            dec = stages.p8_verdict(claim, bt, {}, synthetic=True)
            if dec.verdict in {"watch", "research-supported"}:
                survivors += 1
        # PEN-11: the synthetic fallback is an edge-free control; allow at most one fluke.
        assert survivors <= 1
    finally:
        client._SEED = old_seed
        p7_backtest.LEDGER = old_ledger


def test_staged_paper_offline_fallback_verdict_is_underpowered(tmp_path):
    from modules.macro_vol_btc import impl
    from penrose import config
    from penrose.data import client
    from penrose.data.contract import DataBundle
    # `claims` is an operator artifact (hand-authored staged papers) that is not part of a cold or
    # public clone; skip this operator-specific acceptance test when it is absent. The edge-free
    # property itself is covered by test_synthetic_macro_fallback_is_edge_free above.
    claims = pytest.importorskip("penrose.pipeline.claims")
    from penrose.pipeline import p7_backtest, stages

    old_ledger = p7_backtest.LEDGER
    p7_backtest.LEDGER = tmp_path / "ledger.tsv"
    try:
        syn = client._synthetic_bundle("2023-01-01", "2026-03-31", None)
        bundle = DataBundle(series={
            "kxfed_signal": syn["kxfed_signal"],
            "kxrecssnber_signal": syn["kxrecssnber_signal"],
            "btc_realized_vol_5d": syn["btc_realized_vol_5d"],
            "btc_implied_vol": syn["btc_implied_vol_syn"],
        }, fallback_substitutions=["kxfed_signal", "kxrecssnber_signal", "btc_implied_vol"])
        channels = {
            "2604.01431v1-btc-fed": "fed",
            "2604.01431v1-btc-recession": "recession",
        }
        observed = {}
        for claim in claims.CLAIMS:
            mres = impl.run(
                bundle, channels[claim.claim_id],
                config.VOL_TRADE_COST["deribit_roundtrip_bps_of_vega"] / 1e4)
            assert mres["ok"]
            bt = p7_backtest.run_backtest(
                claim.claim_id, mres["net"], mres["positions"], mres["bars_per_year"],
                log=False, family="pen11::staged-paper")
            dec = stages.p8_verdict(claim, bt, {}, synthetic=True)
            observed[claim.claim_id] = (dec.verdict, dec.kill_reason)

        # PEN-11: observed after the fallback became edge-free. Pin the honest
        # offline staged-paper result rather than preserving the former planted discovery.
        assert observed == {
            "2604.01431v1-btc-fed": ("underpowered", "below_detection_floor"),
            "2604.01431v1-btc-recession": ("underpowered", "below_detection_floor"),
        }
    finally:
        p7_backtest.LEDGER = old_ledger


def test_fetch_bundle_warns_and_records_synthetic_substitutions(tmp_path, monkeypatch, capsys):
    from penrose import config
    from penrose.data import client
    from penrose.data.contract import Unavailable

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "missing-catalog")
    monkeypatch.setattr(client, "_binance_btc_daily",
                        lambda start, end: Unavailable("btc_price", "offline"))
    monkeypatch.setattr(client, "_deribit_dvol_daily",
                        lambda start, end: Unavailable("btc_implied_vol", "offline"))

    bundle = client.fetch_bundle("2023-01-01", "2023-04-30")
    err = capsys.readouterr().err

    assert "substituting EDGE-FREE synthetic" in err
    assert "kxfed_signal" in bundle.fallback_substitutions
    assert "kxrecssnber_signal" in bundle.fallback_substitutions
    assert "btc_implied_vol" in bundle.fallback_substitutions
    assert bundle.series["kxfed_signal"].note == (
        "EDGE-FREE synthetic fallback; any edge found on this series is a bug")


def _isolate_run_outputs(runmod, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runmod.config, "ROOT", tmp_path)
    monkeypatch.setattr(runmod.config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(runmod.config, "PROCESSED_PAPERS", tmp_path / "processed.json")
    monkeypatch.setattr(runmod.config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(runmod.config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(runmod.config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(runmod.config, "ANALYSIS_INDEX", tmp_path / "analysis.jsonl")
    monkeypatch.setattr(runmod.config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(runmod.config, "LIVE_JSON", tmp_path / "live.json")
    monkeypatch.setattr(runmod.config, "PROGRESS_JSON", tmp_path / "progress.json")
    monkeypatch.setattr(runmod.config, "MODULES", tmp_path / "modules")


def test_analysis_record_includes_fallback_substitutions(tmp_path, monkeypatch):
    from penrose.brain import Claim, Decision
    from penrose.data.contract import DataBundle
    from penrose.pipeline import run as runmod

    class Module:
        def run(self, bundle, claim, cost_frac):
            idx = pd.date_range("2024-01-01", periods=6, tz="UTC")
            net = pd.Series([0.01, 0.02, -0.01, 0.0, 0.01, -0.02], index=idx)
            return {"ok": True, "net": net, "positions": net.abs(), "bars_per_year": 252.0}

    _isolate_run_outputs(runmod, monkeypatch, tmp_path)
    paper = tmp_path / "paper.md"
    paper.write_text("Unit signal predicts one-day returns.")
    claim = Claim(
        "pen-11-c1", "Unit signal predicts one-day returns.", "", "", "1 day",
        "paper", "Unit signal predicts one-day returns.", "",
        applicable_strategy_class="pen11_unit")
    bundle = DataBundle(
        requested_window=("2024-01-01", "2024-01-03"),
        fallback_substitutions=["kxfed_signal", "btc_implied_vol"],
    )

    monkeypatch.setattr(runmod, "_register_known_modules", lambda: None)
    monkeypatch.setattr(runmod, "REGISTRY", {"pen11_unit": Module()})
    monkeypatch.setattr(runmod.stages, "p5_dedup", lambda claim, reader: {"stage": "P5", "killed": False})
    monkeypatch.setattr(runmod.p7_backtest, "run_backtest", lambda *a, **k: {
        "dsr": 0.1, "psr": 0.1, "n_oos": 3, "bars_per_year": 252.0,
        "three_fold": {}, "bootstrap": {}, "permutation": {}, "regime": {},
    })
    monkeypatch.setattr(runmod.stages, "p8_verdict", lambda claim, bt, holdout, synthetic: Decision(
        decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
        verdict="watch", kill_reason=None, rationale="unit verdict", metrics={}))
    monkeypatch.setattr(runmod.stages, "propose_principle", lambda decisions: None)
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")
    monkeypatch.setattr(runmod, "write_report", lambda *a, **k: tmp_path / "report.md")

    runmod.run_source(paper, use_llm=False, claims_override=[claim],
                      bundle_override=bundle, force=True)

    row = json.loads((tmp_path / "analysis.jsonl").read_text().splitlines()[-1])
    assert row["data_provenance"]["fallback_substitutions"] == [
        "kxfed_signal", "btc_implied_vol"]
