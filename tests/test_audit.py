import json
import uuid


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_envelope_is_seq_zero(tmp_path):
    from penrose.audit import AuditLog

    path = tmp_path / "audit.jsonl"
    log = AuditLog("run-1", "cli", path)
    log.envelope("0.8.0", "fp", {}, {"seed": 0}, {"os": "test"}, "OPEN")
    row = _rows(path)[0]
    assert row["seq"] == 0
    assert row["event"] == "reproduction_envelope"
    assert row["prev_hash"] == "0" * 64


def test_chain_link_tamper_detection(tmp_path):
    from penrose.audit import AuditLog, verify_events

    path = tmp_path / "audit.jsonl"
    log = AuditLog("run-1", "cli", path)
    log.envelope("0.8.0", "fp", {}, {}, {}, "OPEN")
    log.stage("P1", "exit", detail={"ok": True})
    log.stage("P2", "exit", detail={"ok": True})
    rows = _rows(path)
    rows[1]["detail"]["ok"] = False
    ok, seq = verify_events(rows)
    assert ok is False
    assert seq == 1


def test_hashes_deterministic_across_wall_clock_and_duration(tmp_path):
    from penrose.audit import AuditLog

    path1 = tmp_path / "a.jsonl"
    path2 = tmp_path / "b.jsonl"
    a = AuditLog("same-run", "cli", path1)
    b = AuditLog("same-run", "cli", path2)
    envelope = ("0.8.0", "fp", {"source": "s"}, {"seed": 0}, {"os": "test"}, "OPEN")
    a.envelope(*envelope)
    b.envelope(*envelope)
    a.stage("P1", "exit", outputs={"b": 2, "a": 1}, duration_ms=1, detail={"x": "y"})
    b.stage("P1", "exit", outputs={"a": 1, "b": 2}, duration_ms=999, detail={"x": "y"})
    hashes1 = [row["hash"] for row in _rows(path1)]
    hashes2 = [row["hash"] for row in _rows(path2)]
    assert hashes1 == hashes2


def test_stage_fail_open_for_nonserializable_and_internal_error(tmp_path, monkeypatch):
    from penrose import audit

    path = tmp_path / "audit.jsonl"
    log = audit.AuditLog("run-1", "cli", path)
    log.stage("P1", "exit", detail={"object": object()})
    assert _rows(path)[0]["detail"]["object"].startswith("<object object")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(audit, "_append_jsonl", boom)
    log.stage("P2", "exit", detail={"still": "does not raise"})


def test_canonical_json_digest_stable_under_key_reordering():
    from penrose.audit import canonical_json_digest

    assert canonical_json_digest({"b": 2, "a": {"d": 4, "c": 3}}) == canonical_json_digest(
        {"a": {"c": 3, "d": 4}, "b": 2}
    )


def test_audit_summary_verifies_and_aggregates(tmp_path, monkeypatch):
    from penrose import config
    from penrose.audit import AuditLog
    from penrose.views import audit_summary

    monkeypatch.setattr(config, "AUDIT", tmp_path / "audit")
    path = config.AUDIT / "run-1.jsonl"
    log = AuditLog("run-1", "cli", path)
    log.envelope("0.8.0", "fp", {}, {}, {"os": "test"}, "OPEN")
    log.stage("P1", "exit", duration_ms=10, detail={"ok": True})
    log.stage("P1", "exit", duration_ms=15, detail={"ok": True})
    log.stage("gate", "gate_outcome", detail={"gate": "deflation", "verdict": "kill"})

    summary = audit_summary("run-1")
    assert summary["chain_ok"] is True
    assert summary["chain_broken_seq"] is None
    assert summary["stage_timings"]["P1"] == 25
    assert summary["gate_outcomes"]["deflation"] == 1
    assert summary["envelope"]["version"] == "0.8.0"


def _isolate_run_paths(base, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "DECISIONS_LOG", base / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", base / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", base / "data_requests.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", base / "processed_papers.json")
    monkeypatch.setattr(config, "REPORTS", base / "reports")
    monkeypatch.setattr(config, "AUDIT", base / "reports" / "audit")
    monkeypatch.setattr(config, "TRACES", base / "reports" / "traces.jsonl")
    monkeypatch.setattr(config, "LIVE_JSON", base / "dashboard" / "live.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", base / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "ARCHIVES", base / "archives")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", base / ".llm_cache")
    monkeypatch.setattr(config, "MODULES", base / "modules")
    monkeypatch.setattr(config, "AUTO_MODULES", base / "modules" / "_auto")
    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    (base / "modules").mkdir(parents=True, exist_ok=True)


class _UUID:
    hex = "abc123"


def _run_zero_claim_source(base, monkeypatch, *, audit_enabled):
    from penrose.pipeline import run as runmod

    _isolate_run_paths(base, monkeypatch)
    monkeypatch.setattr(runmod.uuid, "uuid4", lambda: _UUID())
    monkeypatch.setattr(runmod, "_now", lambda: "2026-01-01T00:00:00+00:00")
    if not audit_enabled:
        monkeypatch.setattr(runmod, "_finalize_audit_run", lambda *a, **k: None)
    paper = base / "paper.md"
    paper.write_text("Tiny staged source.")
    out = runmod.run_source(paper, use_llm=False, claims_override=[], force=True)
    return out


def test_run_source_audit_is_additive_for_trace_and_decision_logs(tmp_path, monkeypatch):
    from penrose import config

    baseline = tmp_path / "baseline"
    audited = tmp_path / "audited"

    _run_zero_claim_source(baseline, monkeypatch, audit_enabled=False)
    baseline_decisions = (config.DECISIONS_LOG.read_bytes()
                          if config.DECISIONS_LOG.exists() else b"")
    baseline_traces = (config.TRACES.read_bytes() if config.TRACES.exists() else b"")

    monkeypatch.undo()
    out = _run_zero_claim_source(audited, monkeypatch, audit_enabled=True)
    audited_decisions = config.DECISIONS_LOG.read_bytes() if config.DECISIONS_LOG.exists() else b""
    audited_traces = config.TRACES.read_bytes() if config.TRACES.exists() else b""

    assert audited_decisions == baseline_decisions
    assert audited_traces == baseline_traces
    assert out["audit_head_hash"]
    assert out["audit_path"]
    assert (config.AUDIT / f"{out['idempotency']['run_id']}.jsonl").exists()


def test_verify_events_treats_non_dict_row_as_broken_not_crash():
    # A tampered/truncated/foreign valid-JSON line (e.g. a bare list) must read as a broken chain,
    # never raise. Regression for the GLM-audit BUG A (audit_summary crashed on `list.get`).
    from penrose.audit import AuditLog, verify_events

    import tempfile, os
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    p = d / "run.jsonl"
    log = AuditLog("run", "cli", p)
    log.envelope("0.8.0", "fp", {}, {}, {"os": "t"}, "OPEN")
    log.stage("P1", "exit", detail={"ok": True})
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows.append([1, 2, 3])  # non-dict foreign row
    ok, seq = verify_events(rows)
    assert ok is False and seq == 2


def test_audit_summary_does_not_crash_on_non_dict_row(tmp_path, monkeypatch):
    from penrose import config, views
    from penrose.audit import AuditLog

    monkeypatch.setattr(config, "AUDIT", tmp_path)
    p = tmp_path / "runX.jsonl"
    log = AuditLog("runX", "cli", p)
    log.envelope("0.8.0", "fp", {}, {}, {"os": "t"}, "OPEN")
    log.stage("P1", "exit", detail={"ok": True})
    with p.open("a") as f:
        f.write(json.dumps([1, 2, 3]) + "\n")  # tampered non-dict row
    out = views.audit_summary("runX")            # must not raise
    assert out["status"] == "ok"
    assert out["chain_ok"] is False
    assert out["malformed_rows"] == 1


def test_set_in_detail_hashes_deterministically():
    # A set's iteration order is process-dependent; _bounded must sort it so the content hash is
    # reproducible. Regression for GLM-audit BUG B (set leaked into the hash).
    from penrose.audit import _bounded, canonical_json_digest

    a = _bounded({"tags": {"z", "a", "m", "q"}})
    b = _bounded({"tags": {"q", "m", "a", "z"}})
    assert canonical_json_digest(a) == canonical_json_digest(b)
    assert a["tags"] == ["a", "m", "q", "z"]
