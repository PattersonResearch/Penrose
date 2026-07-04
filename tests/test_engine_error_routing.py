import json
from pathlib import Path

import pandas as pd


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


def _claim(strategy_class="engine_error_unit"):
    from penrose.brain import Claim

    return Claim(
        claim_id="pen-09-c1",
        statement="Unit signal predicts one-day returns.",
        mechanism="unit",
        scope="unit",
        horizon="1 day",
        source_id="paper",
        source_span="Unit signal predicts one-day returns.",
        claimed_metric_quote="",
        applicable_strategy_class=strategy_class,
    )


def _run_single_claim(tmp_path, monkeypatch, module, *, patch_backtest=True):
    from penrose.brain import Decision
    from penrose.data.contract import DataBundle
    from penrose.pipeline import run as runmod

    _isolate_run_outputs(runmod, monkeypatch, tmp_path)
    paper = tmp_path / "paper.md"
    paper.write_text("Unit signal predicts one-day returns.")
    monkeypatch.setattr(runmod, "_register_known_modules", lambda: None)
    monkeypatch.setattr(runmod, "REGISTRY", {"engine_error_unit": module})
    monkeypatch.setattr(runmod.stages, "p5_dedup", lambda claim, reader: {"stage": "P5", "killed": False})
    if patch_backtest:
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
        paper, use_llm=False, claims_override=[_claim()],
        bundle_override=DataBundle(requested_window=("2024-01-01", "2024-01-03")),
        force=True,
    )


class _RaisingModule:
    def run(self, bundle, claim, cost_frac):
        raise KeyError("boom")


class _RaisingDataUnavailableModule:
    def run(self, bundle, claim, cost_frac):
        raise KeyError("data_unavailable: eth_spot")


class _NeedsDataModule:
    def run(self, bundle, claim, cost_frac):
        return {"ok": False, "reason": "data_unavailable: eth_spot_daily"}


class _OkModule:
    def run(self, bundle, claim, cost_frac):
        idx = pd.date_range("2024-01-01", periods=6, tz="UTC")
        net = pd.Series([0.01, 0.02, -0.01, 0.0, 0.01, -0.02], index=idx)
        return {"ok": True, "net": net, "positions": net.abs(), "bars_per_year": 252.0}


def test_raising_module_routes_engine_error_without_data_request(tmp_path, monkeypatch):
    from penrose.pipeline import run as runmod

    sentinel = '{"request_id":"keep"}\n'
    _isolate_run_outputs(runmod, monkeypatch, tmp_path)
    (tmp_path / "data_requests.jsonl").write_text(sentinel)

    out = _run_single_claim(tmp_path, monkeypatch, _RaisingModule())

    assert out["decisions"][0]["verdict"] == "engine_error"
    assert (tmp_path / "data_requests.jsonl").read_text() == sentinel
    review = json.loads((tmp_path / "review_queue.jsonl").read_text().splitlines()[-1])
    assert review["type"] == "engine_error"
    assert review["status"] == "pending"


def test_raising_data_unavailable_keyerror_stays_engine_error(tmp_path, monkeypatch):
    # B-01: engine_error-wrapped data_unavailable text is not a module data request.
    out = _run_single_claim(tmp_path, monkeypatch, _RaisingDataUnavailableModule())

    assert out["decisions"][0]["verdict"] == "engine_error"
    assert not (tmp_path / "data_requests.jsonl").exists()
    review = json.loads((tmp_path / "review_queue.jsonl").read_text().splitlines()[-1])
    assert review["type"] == "engine_error"
    assert review["status"] == "pending"


def test_module_data_unavailable_still_routes_needs_data(tmp_path, monkeypatch):
    out = _run_single_claim(tmp_path, monkeypatch, _NeedsDataModule())

    assert out["decisions"][0]["verdict"] == "needs_data"
    req = json.loads((tmp_path / "data_requests.jsonl").read_text().splitlines()[-1])
    assert req["missing_series"] == ["eth_spot_daily"]


def test_backtest_exception_routes_engine_error_not_needs_data(tmp_path, monkeypatch):
    from penrose.pipeline import run as runmod

    def raise_backtest(*args, **kwargs):
        raise RuntimeError("bad stats")

    monkeypatch.setattr(runmod.p7_backtest, "run_backtest", raise_backtest)
    out = _run_single_claim(tmp_path, monkeypatch, _OkModule(), patch_backtest=False)

    assert out["decisions"][0]["verdict"] == "engine_error"
    assert not (tmp_path / "data_requests.jsonl").exists()
    decision = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert decision["metrics"]["stage"] == "P7 backtest"


def test_review_list_and_approve_handle_engine_errors(tmp_path, monkeypatch, capsys):
    from penrose import config
    from penrose.pipeline import p9_review

    queue = tmp_path / "review_queue.jsonl"
    monkeypatch.setattr(config, "REVIEW_QUEUE", queue)
    queue.write_text(json.dumps({
        "type": "engine_error",
        "status": "pending",
        "claim_id": "pen-09-c1",
        "rationale": "engine error during module run: RuntimeError",
    }) + "\n")

    p9_review.cmd_list(None)
    assert "ENGINE ERROR pen-09-c1" in capsys.readouterr().out

    class Args:
        idx = 0
        approver = "unit"

    try:
        p9_review.cmd_approve(Args())
    except SystemExit as e:
        assert "engine errors are fixed in code" in str(e)
    else:
        raise AssertionError("engine_error approval must be refused")
