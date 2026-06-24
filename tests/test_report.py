from penrose import config
from penrose.brain import Claim, Decision
from penrose.report import write_report


def test_report_notes_inert_deflation_when_dsr_equals_psr_despite_partitions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    claim = Claim("c1", "single claim with regime partitions", "", "", "", "unit", "span", "")
    decision = Decision(
        decision_id="d1",
        claim_id="c1",
        verdict="watch",
        kill_reason=None,
        rationale="unit test",
        metrics={"psr": 0.941234, "dsr": 0.941234, "n_trials": 4, "sr_var": 0.0},
    )

    path = write_report("unit-report", "Unit Report", [claim], [decision], {}, None)

    text = path.read_text()
    assert "**Deflation note:** scored by PSR" in text
