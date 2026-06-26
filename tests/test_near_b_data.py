from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


def _load_catalog_loader():
    from penrose import config
    from penrose.data.contract import load_catalog_loader

    catalog_dir = config.DATA_DIR          # the public PENROSE_DATA_DIR contract
    loader_path = catalog_dir / "loader.py"
    if not loader_path.exists():
        pytest.skip("catalog loader not present")
    return load_catalog_loader(catalog_dir)


def test_panel_adapter_aggregates_synthetic_parquet(tmp_path, monkeypatch):
    loader = _load_catalog_loader()
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "event_ts": pd.to_datetime([
            "2024-01-01 09:00Z", "2024-01-01 18:00Z",
            "2024-01-02 10:00Z", "2024-01-02 12:00Z",
            "2024-01-03 10:00Z",
        ]),
        "market": ["target", "target", "target", "other", "target"],
        "value": [1.0, 3.0, 5.0, 100.0, 7.0],
    }).to_parquet(panel_path)
    repo = os.path.relpath(tmp_path, loader.DEV)

    base = {
        "repo": repo,
        "path": panel_path.name,
        "adapter": "panel",
        "date_col": "event_ts",
        "value_col": "value",
        "filter_col": "market",
        "filter_val": "target",
        "provenance": "fixture",
    }
    monkeypatch.setattr(loader, "SERIES", {
        "panel_last": {**base, "agg": "last"},
        "panel_sum": {**base, "agg": "sum"},
        "panel_mean": {**base, "agg": "mean"},
        "panel_count": {**base, "agg": "count"},
    })

    expected = {
        "panel_last": [3.0, 5.0, 7.0],
        "panel_sum": [4.0, 5.0, 7.0],
        "panel_mean": [2.0, 5.0, 7.0],
        "panel_count": [2.0, 1.0, 1.0],
    }
    for name, values in expected.items():
        result = loader.load_series(name)
        assert result is not None
        s, prov = result
        assert prov == "fixture"
        assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert list(s.astype(float)) == values


def test_panel_adapter_fails_open_on_missing_column_and_empty_filter(tmp_path, monkeypatch):
    loader = _load_catalog_loader()
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "event_ts": pd.to_datetime(["2024-01-01 09:00Z"]),
        "market": ["target"],
        "value": [1.0],
    }).to_parquet(panel_path)
    repo = os.path.relpath(tmp_path, loader.DEV)
    monkeypatch.setattr(loader, "SERIES", {
        "missing_col": {
            "repo": repo, "path": panel_path.name, "adapter": "panel",
            "date_col": "missing", "value_col": "value",
        },
        "empty_filter": {
            "repo": repo, "path": panel_path.name, "adapter": "panel",
            "date_col": "event_ts", "value_col": "value",
            "filter_col": "market", "filter_val": "absent",
        },
    })

    assert loader.load_series("missing_col") is None
    assert loader.load_series("empty_filter") is None


def _isolate_run_outputs(monkeypatch, tmp_path):
    from penrose import config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", tmp_path / "processed_papers.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", tmp_path / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")


class _UnitVendor:
    NAME = "unitvendor"
    PROVENANCE_GRADE = "as_displayed"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def available(self):
        return True

    def fetch(self, spec):
        self.calls += 1
        if self.result == "raise":
            raise RuntimeError("network down")
        return self.result


class _NeedsSeriesModule:
    __strategy_class__ = "near_b_unit"
    __module_id__ = "near_b_unit"

    def __init__(self, missing="unit_test_vendor_series"):
        self.missing = missing
        self.calls = 0

    def run(self, bundle, claim, cost_frac):
        self.calls += 1
        s = bundle.get(self.missing)
        if s is None:
            return {"ok": False, "reason": f"data_unavailable: {self.missing}"}
        net = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
        return {"ok": True, "net": net, "positions": net.abs(), "bars_per_year": 252.0}


def _run_single_claim(tmp_path, monkeypatch, module):
    from penrose.brain import Claim, Decision
    from penrose.data.contract import DataBundle
    from penrose.pipeline import run as runmod

    _isolate_run_outputs(monkeypatch, tmp_path)
    paper = tmp_path / "paper.md"
    paper.write_text("Daily test series predicts returns over one day.")
    claim = Claim(
        claim_id="near-b-c1",
        statement="Daily test series predicts returns over one day.",
        mechanism="unit",
        scope="unit",
        horizon="1 day",
        source_id="paper",
        source_span="Daily test series predicts returns over one day.",
        claimed_metric_quote="",
        applicable_strategy_class="near_b_unit",
    )
    monkeypatch.setattr(runmod, "_register_known_modules", lambda: None)
    monkeypatch.setattr(runmod, "REGISTRY", {"near_b_unit": module})
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
    return runmod.run_source(
        paper, use_llm=False, claims_override=[claim],
        bundle_override=DataBundle(requested_window=("2024-01-01", "2024-01-03")),
        force=True,
    )


def test_needs_data_auto_fetches_resolvable_series_and_retests(tmp_path, monkeypatch):
    from penrose.data import vendors

    fetched = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    vendor = _UnitVendor((fetched, "unit-vendor"))
    monkeypatch.setattr(vendors, "ADAPTERS", {"unitvendor": vendor})
    monkeypatch.setattr(vendors, "DEFAULT_SERIES", {
        "unit_test_vendor_series": {"vendor": "unitvendor", "unit": "u"}
    })
    module = _NeedsSeriesModule()

    out = _run_single_claim(tmp_path, monkeypatch, module)

    assert out["decisions"] == [{"claim_id": "near-b-c1", "verdict": "watch", "kill_reason": None}]
    assert module.calls == 2
    assert vendor.calls == 1
    assert "unit_test_vendor_series" in out["provenance"]


def test_needs_data_unresolvable_series_stays_needs_data(tmp_path, monkeypatch):
    from penrose.data import vendors

    vendor = _UnitVendor(None)
    monkeypatch.setattr(vendors, "ADAPTERS", {"unitvendor": vendor})
    monkeypatch.setattr(vendors, "DEFAULT_SERIES", {
        "different_vendor_series": {"vendor": "unitvendor", "unit": "u"}
    })
    module = _NeedsSeriesModule("unknown_or_ambiguous_series")

    out = _run_single_claim(tmp_path, monkeypatch, module)

    assert out["decisions"][0]["verdict"] == "needs_data"
    assert out["decisions"][0]["kill_reason"] is None
    assert module.calls == 1
    assert vendor.calls == 0


@pytest.mark.parametrize("fetch_result", ["raise", None])
def test_needs_data_fetch_failure_falls_back_gracefully(tmp_path, monkeypatch, fetch_result):
    from penrose.data import vendors

    vendor = _UnitVendor(fetch_result)
    monkeypatch.setattr(vendors, "ADAPTERS", {"unitvendor": vendor})
    monkeypatch.setattr(vendors, "DEFAULT_SERIES", {
        "unit_test_vendor_series": {"vendor": "unitvendor", "unit": "u"}
    })
    module = _NeedsSeriesModule()

    out = _run_single_claim(tmp_path, monkeypatch, module)

    assert out["decisions"][0]["verdict"] == "needs_data"
    assert module.calls == 1
    assert vendor.calls == 1
    assert out["claims"][0]["stages"]["P7_auto_fetch"]["attempted"] == ["unit_test_vendor_series"]
    data_request = json.loads((tmp_path / "data_requests.jsonl").read_text().splitlines()[-1])
    decision_log = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert data_request["auto_fetch_attempted"] == ["unit_test_vendor_series"]
    assert decision_log["metrics"]["auto_fetch_attempted"] == ["unit_test_vendor_series"]
    assert "Auto-fetch attempted: unit_test_vendor_series" in decision_log["rationale"]


def test_vendor_resolver_rejects_ambiguous_alias(monkeypatch):
    from penrose.data import vendors

    vendor = _UnitVendor(None)
    monkeypatch.setattr(vendors, "ADAPTERS", {"unitvendor": vendor})
    monkeypatch.setattr(vendors, "DEFAULT_SERIES", {
        "vendor_series": {"vendor": "unitvendor", "unit": "u"},
        "series_vendor": {"vendor": "unitvendor", "unit": "u"},
    })

    assert vendors.resolve_vendor_spec("vendor_series") is None
