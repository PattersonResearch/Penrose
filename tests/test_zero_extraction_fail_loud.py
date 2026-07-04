"""FIX 2 / FIX 3 regression tests (v0.4.1): a 0-claim extraction result must never look
like a silent success (2026-07-04 incident: funding_drift_claim, EXP-1/EXP-1b).
"""
import json

import pytest


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _isolate(tmp_path, monkeypatch):
    from penrose import config

    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "REVIEW_QUEUE", tmp_path / "review_queue.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "PROCESSED_PAPERS", tmp_path / "processed_papers.json")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "dashboard" / "live.json")
    monkeypatch.setattr(config, "PROGRESS_JSON", tmp_path / "dashboard" / "progress.json")
    monkeypatch.setattr(config, "ARCHIVES", tmp_path / "archives")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setattr(config, "MODULES", tmp_path / "modules")
    monkeypatch.setattr(config, "AUTO_IMPLEMENT_MODULES", False)
    (tmp_path / "modules").mkdir()


# --- unit level: _zero_extraction_is_suspicious ------------------------------------ #

class _Src:
    def __init__(self, n_chars):
        self.n_chars = n_chars


def test_fallback_after_error_is_always_suspicious():
    from penrose.pipeline import run as runmod

    assert runmod._zero_extraction_is_suspicious(
        _Src(10), {"mode": "fallback-after-error", "extraction_error": "network error"}) is True
    # even a tiny source: an engine failure is an engine failure regardless of size
    assert runmod._zero_extraction_is_suspicious(
        _Src(1), {"mode": "fallback-after-error"}) is True


def test_llm_zero_extraction_on_nontrivial_source_is_suspicious():
    from penrose.pipeline import run as runmod

    big = runmod.MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION + 50
    assert runmod._zero_extraction_is_suspicious(_Src(big), {"mode": "llm", "n_extracted": 0}) is True


def test_llm_zero_extraction_on_trivial_source_is_quiet():
    from penrose.pipeline import run as runmod

    tiny = max(0, runmod.MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION - 20)
    assert runmod._zero_extraction_is_suspicious(_Src(tiny), {"mode": "llm", "n_extracted": 0}) is False


def test_manual_mode_zero_extraction_is_quiet():
    """An operator-chosen offline run (no LLM, no claims.py) is a deliberate, known state,
    not a suspicious engine failure -- never flagged loudly."""
    from penrose.pipeline import run as runmod

    big = runmod.MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION + 500
    assert runmod._zero_extraction_is_suspicious(_Src(big), {"mode": "manual"}) is False


def test_source_adapter_zero_extraction_is_quiet():
    """A caller explicitly passing claims_override=[] made a deliberate choice; not suspicious."""
    from penrose.pipeline import run as runmod

    big = runmod.MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION + 500
    assert runmod._zero_extraction_is_suspicious(_Src(big), {"mode": "source-adapter"}) is False


def test_is_suspicious_fails_open_never_raises():
    from penrose.pipeline import run as runmod

    assert runmod._zero_extraction_is_suspicious(object(), None) is False
    assert runmod._zero_extraction_is_suspicious(None, {"mode": "llm"}) is False


# --- integration level: run_source through the real zero-claims branch ------------- #

def test_extraction_failure_writes_loud_engine_error_not_silent_success(tmp_path, monkeypatch):
    """Reproduces the funding_drift_claim shape: extract_claims raises (transient LLM/
    network failure) -> run_source falls back to fallback_claims -> no claims.py exists
    for this source -> zero claims. This must surface as a loud engine_error, never a
    quiet 'no claims extracted' success."""
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift_claim.md"
    paper.write_text(
        "On crypto perpetual futures, the funding rate is the market's implied expected "
        "return. excess_drift = realized_return - funding_implied_drift. Go long when "
        "excess_drift > 0; short or flat when excess_drift < 0. Test cross-sectionally "
        "across BTC, ETH, SOL, BNB, XRP, DOGE perps.\n"
    )

    def _boom(*args, **kwargs):
        raise RuntimeError(
            "role claim_extractor returned un-parseable/empty JSON after 3 attempts "
            "(last body 0 chars, finish_reason=length)"
        )

    monkeypatch.setattr(runmod.extract, "extract_claims", _boom)
    monkeypatch.setattr(runmod.extract, "fallback_claims",
                        lambda source: ([], {"extracted_by": "manual-fallback",
                                            "error": "no claims.py for this source",
                                            "n_extracted": 0}))
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})

    out = runmod.run_source(paper, use_llm=True)

    assert out.get("engine_error") is True
    assert out["p2"]["mode"] == "fallback-after-error"

    rows = _read_rows(config.DECISIONS_LOG)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "engine_error"
    assert "extraction" in rows[0]["rationale"].lower()

    review = _read_rows(config.REVIEW_QUEUE)
    assert len(review) == 1
    assert review[0]["type"] == "engine_error"


def test_llm_genuine_zero_claims_on_nontrivial_paper_is_also_loud(tmp_path, monkeypatch):
    """Even when the LLM call SUCCEEDS but returns zero claims for a substantial paper
    (the EXP-1 admin-vocabulary misread class), the run must not look like a clean
    success either -- it needs a human to confirm the judgment."""
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "admin_styled.md"
    paper.write_text(
        "decision_id: kalshi_tail_calibration-c1\nverdict: cannot_replicate\n"
        "kill_reason: unfaithful_spec\nsupersedes: kalshi_tail_calibration.md (frozen)\n"
        "This pre-registration amendment records the prior run's verdict for audit "
        "purposes and documents the supersession chain before any new adjudication.\n" * 3
    )

    monkeypatch.setattr(runmod.extract, "extract_claims",
                        lambda source, known_classes=None: ([], {"n_extracted": 0}))
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})

    out = runmod.run_source(paper, use_llm=True)

    assert out.get("engine_error") is True
    rows = _read_rows(config.DECISIONS_LOG)
    assert len(rows) == 1 and rows[0]["verdict"] == "engine_error"


def test_trivial_empty_source_stays_quiet(tmp_path, monkeypatch):
    """A genuinely trivial/blank source (nothing to extract) must NOT be flagged loudly —
    the guard is for suspicious failures on real content, not every empty test fixture."""
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "blank.md"
    paper.write_text("x")

    monkeypatch.setattr(runmod.extract, "extract_claims",
                        lambda source, known_classes=None: ([], {"n_extracted": 0}))
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})

    out = runmod.run_source(paper, use_llm=True)

    assert not out.get("engine_error")
    assert out["note"] == "no claims extracted; pipeline ends here"
    rows = _read_rows(config.DECISIONS_LOG) if config.DECISIONS_LOG.exists() else []
    assert rows == []


def test_extraction_failure_does_not_supersede_prior_decisions(tmp_path, monkeypatch):
    """FIX 1 + FIX 2 together: the exact 2026-07-04 incident. A --force re-run whose
    extraction fails must both (a) surface loudly and (b) never call
    _supersede_decision_rows, so the prior decision for this source is untouched."""
    from penrose import config
    from penrose.pipeline import run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift_claim.md"
    paper.write_text(
        "On crypto perpetual futures, the funding rate is the market's implied expected "
        "return over the funding window and realized drift leads the convergence.\n"
    )

    prior = {"decision_id": "funding_drift_claim-c3-d1", "claim_id": "funding_drift_claim-c3",
            "source_id": "funding_drift_claim", "run_id": "old-run", "verdict": "kill",
            "kill_reason": "in_sample_only"}
    runmod._append_jsonl(config.DECISIONS_LOG, prior)

    monkeypatch.setattr(runmod.extract, "extract_claims",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network error")))
    monkeypatch.setattr(runmod.extract, "fallback_claims",
                        lambda source: ([], {"extracted_by": "manual-fallback", "n_extracted": 0}))
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})

    out = runmod.run_source(paper, use_llm=True, force=True)

    rows = _read_rows(config.DECISIONS_LOG)
    assert prior in rows                                     # untouched
    assert not any(r.get("type") == "supersession_marker" for r in rows)
    assert any(r.get("verdict") == "engine_error" for r in rows)
    assert out["idempotency"]["superseded_decisions"] == 0
