import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series
from penrose.pipeline import p7_backtest as p7
from penrose.pipeline import run as runmod


class _LiveModule:
    __strategy_class__ = "unit-live-reader"
    __module_id__ = "unit-live-reader"

    @staticmethod
    def run(bundle, _claim, _cost_frac):
        return _module_result(bundle.get("live_edge").data)


class _SyntheticModule:
    __strategy_class__ = "unit-synthetic-reader"
    __module_id__ = "unit-synthetic-reader"

    @staticmethod
    def run(bundle, _claim, _cost_frac):
        return _module_result(bundle.get("synthetic_edge").data)


def _module_result(net):
    net = net.astype(float)
    return {
        "ok": True,
        "net": net,
        "positions": pd.Series(1.0, index=net.index),
        "bars_per_year": 252.0,
    }


def _claim(claim_id: str, cls: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=f"Unit claim {claim_id}",
        mechanism="deterministic fixture",
        scope="unit",
        horizon="daily",
        source_id="parallel-unit",
        source_span="fixture",
        claimed_metric_quote="deterministic fixture",
        applicable_strategy_class=cls,
    )


def _claims():
    return [
        _claim("parallel-unit-c1", "unit-live-reader"),
        _claim("parallel-unit-c2", "unit-synthetic-reader"),
        _claim("parallel-unit-c3", "unit-live-reader"),
        _claim("parallel-unit-c4", "unit-synthetic-reader"),
    ]


def _bundle() -> DataBundle:
    idx = pd.date_range("2020-01-01", periods=360, freq="D", tz="UTC")
    live = pd.Series(np.sin(np.arange(len(idx)) / 11.0) * 0.002 + 0.0002, index=idx)
    synth = pd.Series(np.cos(np.arange(len(idx)) / 13.0) * 0.002 + 0.0001, index=idx)
    return DataBundle(series={
        "live_edge": Series("live_edge", live, "vendor-live", "return"),
        "synthetic_edge": Series("synthetic_edge", synth, "synthetic", "return"),
    })


def _isolate(tmp_path, monkeypatch, name: str):
    root = tmp_path / name
    monkeypatch.setattr(runmod.config, "ARCHIVES", root / "archives")
    monkeypatch.setattr(runmod.config, "PROCESSED_PAPERS", root / "processed.json")
    monkeypatch.setattr(runmod.config, "REVIEW_QUEUE", root / "review_queue.jsonl")
    monkeypatch.setattr(runmod.config, "DATA_REQUESTS", root / "data_requests.jsonl")
    monkeypatch.setattr(runmod.config, "DECISIONS_LOG", root / "decisions.jsonl")
    monkeypatch.setattr(runmod.config, "ANALYSIS_INDEX", root / "reports" / "analysis_index.jsonl")
    monkeypatch.setattr(runmod.config, "REPORTS", root / "reports")
    monkeypatch.setattr(runmod.config, "LIVE_JSON", root / "live.json")
    monkeypatch.setattr(runmod.config, "PROGRESS_JSON", root / "progress.json")
    monkeypatch.setattr(runmod.config, "CONCEPTS", root / "reports" / "concepts.jsonl")
    monkeypatch.setattr(runmod.config, "MODULES", root / "modules")
    return root


def _install_unit_registry(monkeypatch):
    monkeypatch.setattr(runmod, "_register_known_modules", lambda: None)
    monkeypatch.setattr(runmod, "REGISTRY", {
        "unit-live-reader": _LiveModule,
        "unit-synthetic-reader": _SyntheticModule,
    })
    monkeypatch.setattr(runmod, "_REGISTRY_ALIAS_OWNERS", {})
    monkeypatch.setattr(runmod, "_REGISTRY_CANONICAL_OWNERS", {})
    monkeypatch.setattr(runmod.extract, "classify_claim_stub",
                        lambda _claim: {"killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda _claim, _reader: {"killed": False, "reason": None, "note": ""})


def _decision_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("claim_id", "").startswith("parallel-unit-c"):
                rows.append(row)
    return rows


def _verdict_signature(root: Path):
    rows = _decision_rows(root / "decisions.jsonl")

    def _sig(row):
        m = row.get("metrics") or {}
        # P-4: assert MORE than verdict/kill_reason — every verdict-affecting field a concurrency race
        # could flip must be in the equivalence net, else a future shared-state divergence ships green.
        return (
            row.get("claim_id"), row.get("verdict"), row.get("kill_reason"),
            m.get("n_trials"), m.get("dsr"), m.get("psr"), m.get("edge_t"),
            m.get("oos_sharpe"), m.get("synthetic"),
            (m.get("tail") or {}).get("asymmetric"),
            (m.get("parameter_fragility") or {}).get("fragile"),
            # NOTE: corpus_isolation is deliberately EXCLUDED — it is an advisory nearest-neighbor lookup
            # whose slug list depends on accumulated concept IDs (content-hashed), so it is not
            # byte-deterministic even serial-vs-serial and does not determine the verdict. Asserting it
            # would flake without catching a real divergence. All hard verdict fields above ARE asserted.
        )

    return [_sig(row) for row in rows]


def _run_fixture(tmp_path, monkeypatch, name: str, workers: int):
    root = _isolate(tmp_path, monkeypatch, name)
    old_ledger = p7.LEDGER
    p7.LEDGER = root / "ledger.tsv"
    _install_unit_registry(monkeypatch)
    paper = root / "paper.txt"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("parallel unit fixture")
    try:
        runmod.run_source(
            paper,
            use_llm=False,
            claims_override=_claims(),
            bundle_override=_bundle(),
            force=True,
            max_claim_workers=workers,
        )
    finally:
        p7.LEDGER = old_ledger
    return root


def test_parallel_verdicts_match_serial_byte_critical_fields(tmp_path, monkeypatch):
    serial = _run_fixture(tmp_path, monkeypatch, "serial", workers=1)
    parallel = _run_fixture(tmp_path, monkeypatch, "parallel", workers=4)

    assert _verdict_signature(parallel) == _verdict_signature(serial)


def test_parallel_bundle_access_is_per_claim(tmp_path, monkeypatch):
    root = _run_fixture(tmp_path, monkeypatch, "access", workers=4)
    rows = [
        json.loads(line)
        for line in (root / "reports" / "analysis_index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_claim = {row["claim_id"]: row for row in rows}

    assert by_claim["parallel-unit-c1"]["data_provenance"]["datasets"] == ["live_edge"]
    assert by_claim["parallel-unit-c1"]["synthetic"] is False
    assert by_claim["parallel-unit-c2"]["data_provenance"]["datasets"] == ["synthetic_edge"]
    assert by_claim["parallel-unit-c2"]["synthetic"] is True


def test_run_claim_tasks_isolates_worker_crash():
    """P-1: a worker raising in an unguarded spot must NOT abort the pool and discard sibling workers'
    results — it is converted to a per-item error result via on_error, and every other item survives."""
    def fn(item):
        if item == 2:
            raise RuntimeError("boom")
        return {"claim_i": item, "ok": True}

    def on_error(item, exc):
        return {"claim_i": item, "worker_error": str(exc)}

    out = runmod._run_claim_tasks([1, 2, 3, 4], worker_count=4, fn=fn, on_error=on_error)
    by_ci = {r["claim_i"]: r for r in out}
    assert len(out) == 4                                      # nothing dropped
    assert by_ci[1]["ok"] is True and by_ci[3]["ok"] is True and by_ci[4]["ok"] is True  # siblings survive
    assert by_ci[2]["worker_error"] == "boom"                 # crash isolated to its own item
