import json
import os
import subprocess
import sys


def test_run_json_prints_one_parseable_object(tmp_path, monkeypatch, capsys):
    from penrose import cli
    from penrose.pipeline import run as runmod

    paper = tmp_path / "paper.md"
    paper.write_text("# paper\n")

    def fake_run_source(path, **kwargs):
        assert path == paper
        assert kwargs["claims_override"] is None
        return {
            "run_at": "2026-01-01T00:00:00Z",
            "source_id": "paper",
            "decisions": [{"claim_id": "c1", "verdict": "kill", "kill_reason": "no_edge"}],
            "principle_proposed": None,
        }

    monkeypatch.setattr(runmod, "run_source", fake_run_source)
    assert cli.main(["run", "--paper", str(paper), "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    out = json.loads(captured.out)
    assert out == {
        "source_id": "paper",
        "verdicts": [{"claim_id": "c1", "verdict": "kill", "kill_reason": "no_edge"}],
        "principle": None,
        "status": "complete",
    }


def test_run_claims_loads_claim_objects_and_bypasses_extraction(tmp_path, monkeypatch, capsys):
    from penrose import cli
    from penrose.pipeline import extract
    from penrose.pipeline import run as runmod

    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps([{
        "claim_id": "injected-c1",
        "statement": "daily BTC carry predicts next day returns",
        "mechanism": "carry",
        "scope": "BTC",
        "horizon": "1d",
        "source_span": "daily BTC carry predicts next day returns",
        "claimed_metric_quote": "Sharpe 1.0",
        "applicable_strategy_class": "unknown-test-class",
        "declared_regime": {"scheme": "BTC_VOL_REGIME", "label": "HIGH"},
    }]))

    def fail_extract(*args, **kwargs):
        raise AssertionError("P2 extraction should not be called for --claims")

    def fake_run_source(path, **kwargs):
        assert path == claims_path
        claims = kwargs["claims_override"]
        assert len(claims) == 1
        assert claims[0].claim_id == "injected-c1"
        assert claims[0].source_id == "claims"
        assert claims[0].declared_regime == {"scheme": "btc_vol_regime", "label": "high"}
        return {
            "run_at": "2026-01-01T00:00:00Z",
            "source_id": "claims",
            "p2": {"mode": "source-adapter"},
            "claims": [{"claim_id": "injected-c1"}],
            "decisions": [{"claim_id": "injected-c1", "verdict": "needs_data",
                           "kill_reason": None}],
            "principle_proposed": None,
        }

    monkeypatch.setattr(extract, "extract_claims", fail_extract)
    monkeypatch.setattr(runmod, "run_source", fake_run_source)
    assert cli.main(["run", "--claims", str(claims_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdicts"] == [{"claim_id": "injected-c1", "verdict": "needs_data",
                                "kill_reason": None}]


def test_run_claims_malformed_file_errors_gracefully(tmp_path, capsys):
    from penrose import cli

    claims_path = tmp_path / "claims.json"
    claims_path.write_text("{bad json")

    assert cli.main(["run", "--claims", str(claims_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("penrose: claims file invalid: malformed JSON")
    assert captured.err == ""
    assert "Traceback" not in captured.out

    assert cli.main(["run", "--claims", str(claims_path), "--json"]) == 1
    err = json.loads(capsys.readouterr().out)
    assert err["status"] == "error"
    assert err["error"].startswith("claims file invalid: malformed JSON")

    claims_path.write_text(json.dumps([{"claim_id": "missing-required-fields"}]))
    assert cli.main(["run", "--claims", str(claims_path)]) == 1
    schema_error = capsys.readouterr().out
    assert schema_error.startswith("penrose: claims file invalid: claim 1 missing required field")
    assert "Traceback" not in schema_error


def test_run_without_new_flags_preserves_pipeline_run_delegation(monkeypatch):
    from penrose import cli
    from penrose.pipeline import run as runmod

    seen = {}

    def fake_main():
        seen["argv"] = list(sys.argv)

    monkeypatch.setattr(runmod, "main", fake_main)
    assert cli.main(["run", "--paper", "x.md", "--no-llm", "--force", "--max-claims", "2"]) == 0
    assert seen["argv"] == [
        "penrose-run", "--paper", "x.md", "--no-llm", "--force", "--max-claims", "2",
    ]


def test_make_eval_honors_py_and_cosmetic_docs_are_resolved():
    dry = subprocess.run(
        ["make", "-n", "eval", "PY=X"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert "X scripts/eval_suite.py" in dry.stdout

    # Internally the public-facing Makefile is `Makefile.public`; the public build
    # ships it AS `Makefile`. Read whichever exists so the public green bar is clean.
    mk = "Makefile.public" if os.path.exists("Makefile.public") else "Makefile"
    phony = next(
        line for line in open(mk).read().splitlines()
        if line.startswith(".PHONY:") or "calib-persistence" in line)
    assert "calib-persistence" in phony or "calib-persistence" in open(mk).read()
    assert "docs/STRESS_TESTING.md" in open("README.md").read()
