import json

from penrose.brain import Claim, Decision
from penrose.trace import TRACE_FIELDS, failure_signature, project_trace_record


def _claim(claim_id="trace-c1"):
    return Claim(
        claim_id=claim_id,
        statement="fixture claim",
        mechanism="fixture",
        scope="unit",
        horizon="daily",
        source_id="trace-source",
        source_span="fixture claim",
        claimed_metric_quote="Sharpe 1",
        applicable_strategy_class="trace-class",
    )


def _rec(claim_id="trace-c1"):
    return {
        "claim_id": claim_id,
        "statement": "fixture claim",
        "inputs_requested": ["phantom_series", "btc_close_daily"],
        "stages": {
            "P1": {"sanitized": True},
            "P3": {"killed": False},
            "P4": {"killed": False},
            "P5": {"killed": False},
            "P6_data_availability": {
                "blocked": True,
                "inputs_requested": ["phantom_series", "btc_close_daily"],
                "missing_series": ["phantom_series"],
            },
            "P8": {"verdict": "needs_data", "missing_series": ["phantom_series"]},
        },
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_trace_record_projection_and_emission(tmp_path, monkeypatch):
    from penrose.pipeline import run as runmod

    trace_path = tmp_path / "reports" / "traces.jsonl"
    monkeypatch.setattr(runmod.config, "TRACES", trace_path)

    claim = _claim()
    dec = Decision(
        decision_id="trace-c1-d1",
        claim_id="trace-c1",
        verdict="needs_data",
        kill_reason=None,
        rationale="untestable",
        metrics={"missing_series": ["phantom_series"]},
    )
    run_log = {"source_id": "trace-source", "idempotency": {"run_id": "run-1"}}
    row = project_trace_record(claim, dec, _rec(), run_log)

    assert tuple(row.keys()) == TRACE_FIELDS
    assert row["run_id"] == "run-1"
    assert row["source_id"] == "trace-source"
    assert row["claim_id"] == "trace-c1"
    assert row["claim_type"] == ""
    assert row["strategy_class"] == "trace-class"
    assert row["inputs_requested"] == ["phantom_series", "btc_close_daily"]
    assert row["data_missing"] == ["phantom_series"]
    assert row["stages_reached"] == ["P1", "P3", "P4", "P5", "P6_data_availability", "P8"]
    assert row["exit_stage"] == "P6_data_availability"
    assert row["gate_outcome"] == "needs_data"
    assert row["verdict"] == "needs_data"
    assert len(row["failure_signature"]) == 16

    runmod._emit_trace(claim, dec, _rec(), run_log)
    emitted = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    assert emitted == [row]


def test_triage_text_clusters_recurring_signatures(tmp_path, monkeypatch, capsys):
    from penrose import cli, config

    traces = tmp_path / "reports" / "traces.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(config, "TRACES", traces)
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    sig = failure_signature("needs_data", "P7", "data_unavailable: int(<q>) failed")
    _write_jsonl(traces, [
        {
            "run_id": "r1", "source_id": "s1", "claim_id": "c1",
            "claim_type": "", "strategy_class": "x", "inputs_requested": [],
            "data_missing": ["phantom_series"], "stages_reached": ["P1", "P7", "P8"],
            "exit_stage": "P7", "gate_outcome": "data_unavailable: int('abc') failed",
            "verdict": "needs_data", "kill_reason": None, "failure_signature": sig,
        },
        {
            "run_id": "r1", "source_id": "s1", "claim_id": "c2",
            "claim_type": "", "strategy_class": "x", "inputs_requested": [],
            "data_missing": ["phantom_series"], "stages_reached": ["P1", "P7", "P8"],
            "exit_stage": "P7", "gate_outcome": "data_unavailable: int('xyz') failed",
            "verdict": "needs_data", "kill_reason": None, "failure_signature": sig,
        },
        {
            "run_id": "r1", "source_id": "s1", "claim_id": "c3",
            "claim_type": "", "strategy_class": "x", "inputs_requested": [],
            "data_missing": [], "stages_reached": ["P1", "P8"],
            "exit_stage": "P8", "gate_outcome": "no_oos_edge",
            "verdict": "kill", "kill_reason": "no_oos_edge",
            "failure_signature": failure_signature("kill", "P8", "no_oos_edge"),
        },
    ])

    assert cli.main(["triage", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "Verdict distribution:" in out
    assert "needs_data" in out
    assert "Per-stage drop-off:" in out
    assert "Top recurring failure clusters:" in out
    assert sig in out
    assert "    2 " in out


def test_triage_json_output_is_valid_and_stable(tmp_path, monkeypatch, capsys):
    from penrose import cli, config

    traces = tmp_path / "reports" / "traces.jsonl"
    monkeypatch.setattr(config, "TRACES", traces)
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    sig = failure_signature("pending_module", "P6", "module_unavailable")
    _write_jsonl(traces, [
        {
            "run_id": "r1", "source_id": "s1", "claim_id": "c2",
            "claim_type": "", "strategy_class": "x", "inputs_requested": [],
            "data_missing": [], "stages_reached": ["P1", "P6", "P8"],
            "exit_stage": "P6", "gate_outcome": "module_unavailable",
            "verdict": "pending_module", "kill_reason": None, "failure_signature": sig,
        },
        {
            "run_id": "r1", "source_id": "s1", "claim_id": "c1",
            "claim_type": "", "strategy_class": "x", "inputs_requested": [],
            "data_missing": [], "stages_reached": ["P1", "P6", "P8"],
            "exit_stage": "P6", "gate_outcome": "module_unavailable",
            "verdict": "pending_module", "kill_reason": None, "failure_signature": sig,
        },
    ])

    assert cli.main(["triage", "--json", "--top", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "failure_clusters": [{
            "count": 2,
            "example_claim_id": "c1",
            "exit_stage": "P6",
            "failure_signature": sig,
            "gate_outcome": "module_unavailable",
            "kill_reason": None,
            "verdict": "pending_module",
        }],
        "input": str(traces),
        "source": None,
        "stage_dropoff": {"P6": 2},
        "status": "ok",
        "total": 2,
        "verdict_distribution": {"pending_module": 2},
    }


def test_triage_empty_absent_state_is_graceful(tmp_path, monkeypatch, capsys):
    from penrose import cli, config

    monkeypatch.setattr(config, "TRACES", tmp_path / "reports" / "traces.jsonl")
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")

    assert cli.main(["triage"]) == 0
    out = capsys.readouterr().out
    assert "No traces or decisions found yet" in out
    assert "Traceback" not in out

    assert cli.main(["triage", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "empty"


def test_trace_read_jsonl_fails_open_on_oserror_mc1(tmp_path):
    """MC-1: a permission-restricted traces file must fail open ([]), not raise a traceback through
    penrose triage / the penrose_triage MCP tool."""
    import os
    from penrose.trace import read_jsonl
    p = tmp_path / "traces.jsonl"
    p.write_text('{"a": 1}\n')
    os.chmod(p, 0o000)
    try:
        assert read_jsonl(p) == []
    finally:
        os.chmod(p, 0o600)
