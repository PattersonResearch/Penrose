"""FIX 5b regression test (v0.4.1): a known-good, June-style claim must extract to >=1 claim
and produce a non-empty adjudication path, i.e. it must never again silently produce zero
claims / zero decisions the way an empty extraction could before the loud-empty-run fix.

The claim text below is an inline fixture (hermetic and offline; not read from disk at test
time)."""
import json

import pandas as pd

FUNDING_DRIFT_CLAIM_TEXT = """# Funding-rate vs realized-drift momentum (crypto perpetual futures)

## Hypothesis

On crypto perpetual futures, the funding rate is the market's implied expected return --
the breakeven drift a long must earn over the funding period to be +EV. The signal
compares funding to the realized drift over the same window:

- When realized drift exceeds the funding-implied drift, the winning side has no
  incentive to close; the imbalance resolves as the losing side capitulates and funding
  rises -> directional continuation.
- When realized drift falls short of funding (paying more than earned), the paying side
  capitulates -> reversal / exhaustion.

Funding is a lagging indicator, so realized drift leads the convergence. This is a
momentum / trend-confirmation effect -- explicitly NOT funding-as-mean-reversion (the
naive RSI read), and NOT a funding-carry harvest (it trades WITH the earning side, the
opposite sign to carry).

## Signal

excess_drift = realized_return - funding_implied_drift, per asset, over matched windows.
Go long when excess_drift > 0 (momentum confirmed by carry); short or flat when
excess_drift < 0. Test cross-sectionally across the perps below (the breadth is the
power source).

## Data -- use these exact catalog series

- Funding (Binance, per-8h rate): funding_btc, funding_eth, funding_sol, funding_bnb,
  funding_xrp, funding_doge.
- Price (OKX perp close): btc_perp_hourly, eth_perp_hourly, sol_perp_hourly,
  bnb_perp_hourly, xrp_perp_hourly, doge_perp_hourly.
- Matched coverage 2024-12-23 to 2026-06-16, daily resolution via the catalog.

## Falsification notes

- Model trading fees and the funding cost itself.
- Expect one-regime / underpowered concerns; the 6-perp cross-section is the intended
  power source.
- Tail: a trend-continuation long is convex (positive skew), not a bounded-up/
  unbounded-down widow-maker.
"""

# A verbatim sentence from the fixture, used as the P2 anti-hallucination source_span.
_SOURCE_SPAN = (
    "excess_drift = realized_return - funding_implied_drift, per asset, over matched windows."
)


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


def test_funding_drift_claim_type_is_trading_strategy_not_misrouted():
    """Regression guard: this is a real trading-strategy claim (signal, long/short,
    cross-sectional test) and must not be swept into the new 6g provided_series_statistic
    routing (which is for "test the statistic of a provided series", a different shape)."""
    from penrose.pipeline import spec_gen

    from penrose.brain import Claim
    claim = Claim(
        claim_id="funding_drift_claim-c1",
        statement="Realized drift minus funding predicts forward returns via excess_drift.",
        mechanism="momentum/trend-confirmation via funding-drift convergence",
        scope="crypto perpetual futures, 6-perp cross-section",
        horizon="daily",
        source_id="funding_drift_claim",
        source_span=_SOURCE_SPAN,
        claimed_metric_quote="",
        applicable_strategy_class="crypto_funding_drift_momentum",
    )
    assert spec_gen.classify_claim_type(claim) == "trading_strategy"


def test_p2_extraction_yields_at_least_one_claim(monkeypatch):
    """Exercises the REAL extract.extract_claims parsing/anti-hallucination logic (only
    the LLM transport is mocked) against the funding_drift_claim fixture text. Must
    extract >=1 claim -- this is the exact paper that silently extracted zero claims
    under the v0.4.0 --force regression."""
    from penrose.llm import LLMResponse
    from penrose.pipeline import extract
    from penrose.pipeline.p1_ingest import IngestedSource

    source = IngestedSource(
        source_id="funding_drift_claim", title="Funding-rate vs realized-drift momentum",
        text=FUNDING_DRIFT_CLAIM_TEXT, n_pages=1, n_chars=len(FUNDING_DRIFT_CLAIM_TEXT),
        text_sha256="deadbeef", injection_flags=[],
    )
    canned = {
        "claims": [{
            "statement": "Realized drift minus funding predicts forward returns via excess_drift.",
            "mechanism": "momentum/trend-confirmation via funding-drift convergence",
            "scope": "crypto perpetual futures, 6-perp cross-section",
            "horizon": "daily",
            "source_span": _SOURCE_SPAN,
            "claimed_metric_quote": None,
            "applicable_strategy_class": "crypto_funding_drift_momentum",
            "expected_edge": None,
            "sample_period": {"start": "2024-12-23", "end": "2026-06-16"},
        }]
    }

    def fake_call_json(role, messages, **kwargs):
        assert role == "claim_extractor"
        resp = LLMResponse(text=json.dumps(canned), model="test-model", in_tokens=10,
                           out_tokens=10, cost_usd=0.0, elapsed_s=0.01)
        return canned, resp

    monkeypatch.setattr(extract.llm, "call_json", fake_call_json)

    claims, prov = extract.extract_claims(source, known_classes={})

    assert len(claims) >= 1
    assert claims[0].source_id == "funding_drift_claim"
    assert claims[0].sample_period == {"start": "2024-12-23", "end": "2026-06-16"}
    assert prov["n_extracted"] >= 1


def test_funding_drift_claim_produces_non_empty_adjudication_path(tmp_path, monkeypatch):
    """End-to-end: the funding_drift_claim fixture, run through run_source with the real
    extractor (LLM transport mocked) and a mocked downstream module, must reach >=1
    decision -- never the silent zero-decision regression."""
    import types

    from penrose.brain import Decision
    from penrose.llm import LLMResponse
    from penrose.pipeline import extract, run as runmod

    _isolate(tmp_path, monkeypatch)
    paper = tmp_path / "funding_drift_claim.md"
    paper.write_text(FUNDING_DRIFT_CLAIM_TEXT)

    canned = {
        "claims": [{
            "statement": "Realized drift minus funding predicts forward returns via excess_drift.",
            "mechanism": "momentum/trend-confirmation via funding-drift convergence",
            "scope": "crypto perpetual futures, 6-perp cross-section",
            "horizon": "daily",
            "source_span": _SOURCE_SPAN,
            "claimed_metric_quote": None,
            "applicable_strategy_class": "crypto_funding_drift_momentum",
            "expected_edge": None,
            "sample_period": None,
        }]
    }

    def fake_call_json(role, messages, **kwargs):
        resp = LLMResponse(text=json.dumps(canned), model="test-model", in_tokens=10,
                           out_tokens=10, cost_usd=0.0, elapsed_s=0.01)
        return canned, resp

    monkeypatch.setattr(extract.llm, "call_json", fake_call_json)
    monkeypatch.setattr(runmod.relevance, "screen", lambda *a, **k: {"relevant": True})
    monkeypatch.setattr(runmod.extract, "classify_claim",
                        lambda claim: {"stage": "P3", "route": "generated-module-testable",
                                       "killed": False, "reason": None, "note": ""})
    monkeypatch.setattr(runmod.stages, "p5_dedup",
                        lambda claim, reader: {"stage": "P5", "killed": False, "reason": None})

    module = types.SimpleNamespace(
        __strategy_class__="crypto_funding_drift_momentum", __module_id__="unit_module",
        __auto_generated__=False,
        run=lambda bundle, claim, cost: {"ok": True, "net": pd.Series([0.01] * 20),
                                         "positions": pd.Series([1.0] * 20), "bars_per_year": 365.0},
    )
    runmod.REGISTRY.clear()
    runmod.REGISTRY["crypto_funding_drift_momentum"] = module
    runmod._REGISTRY_ALIAS_OWNERS.clear()
    runmod._REGISTRY_CANONICAL_OWNERS.clear()

    monkeypatch.setattr(runmod.p7_backtest, "run_backtest",
                        lambda *a, **k: {"psr": 0.2, "dsr": 0.2, "n_oos": 200, "oos_sharpe": 0.1,
                                        "capacity_usd": 1_000_000, "three_fold": {}, "bootstrap": {},
                                        "permutation": {}, "regime": {}})
    monkeypatch.setattr(runmod.stages, "p8_verdict",
                        lambda claim, bt, holdout, synthetic: Decision(
                            decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                            verdict="kill", kill_reason="no_oos_edge", rationale="unit kill",
                            metrics={"psr": bt["psr"], "dsr": bt["dsr"]}))
    monkeypatch.setattr(runmod.charts, "render_backtest_chart", lambda *a, **k: "")

    class _TinyBundle:
        series = {}
        requested_window = None

        def provenance_summary(self):
            return {}

        def any_synthetic(self):
            return False

        def reset_access(self):
            pass

        def accessed_synthetic(self):
            return False

    out = runmod.run_source(paper, use_llm=True, bundle_override=_TinyBundle())

    assert out.get("engine_error") is not True
    assert len(out.get("decisions", [])) >= 1
    from penrose import config
    rows = [json.loads(line) for line in config.DECISIONS_LOG.read_text().splitlines() if line.strip()]
    assert len(rows) >= 1
    assert rows[0]["verdict"] != "engine_error"
