"""Smoke + behaviour tests for the penrose v1 pipeline.

Run: PYTHONPATH=src:. python tests/test_pipeline.py
(or `make test`). Uses the real data client (live BTC/DVOL + synthetic macro
signal) so it doubles as an integration check.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def test_data_contract_provenance():
    from penrose.data import client
    b = client.fetch_bundle()
    prov = b.provenance_summary()
    assert "btc_price" in prov and "kxfed_signal" in prov
    # every series declares a provenance, never a silent null
    for k, v in prov.items():
        assert "provenance" in v, k
    print("ok: data contract attaches provenance to every series")


def test_fee_curve_peaks_at_50c():
    from penrose import stats as H
    f50 = H.pm_fee_frac(0.5, 0.07, 1.0)
    ftail = H.pm_fee_frac(0.05, 0.07, 1.0)
    assert f50 > ftail * 5, "fee must peak at 50c and vanish at the tails"
    print(f"ok: fee curve peaks at 50c ({f50:.4f}) vs tail ({ftail:.4f})")


def test_pipeline_runs_and_proposes_only():
    """Smoke test: pipeline runs the staged paper, produces proposals only.

    Uses manual fallback path (--no-llm) so it works in any environment. The
    full LLM path is exercised in dev/integration; this test verifies the
    plumbing, firewall, and queue semantics.
    """
    from penrose.pipeline import run as run_mod
    from pathlib import Path
    paper = Path(__file__).resolve().parents[1] / "2604.01431v1.pdf"
    if not paper.exists():
        print("skip: staged paper not present at repo root")
        return
    out = run_mod.run_source(paper, use_llm=False)
    assert out.get("source_id"), "source_id must be set"
    assert "claims" in out, "claims list must exist (even if empty)"
    from penrose import config
    assert config.REVIEW_QUEUE.exists(), "proposals must land in the review queue"
    import json
    rows = [json.loads(l) for l in config.REVIEW_QUEUE.read_text().splitlines() if l.strip()]
    assert any(r.get("status") == "pending" for r in rows), "proposals start pending"
    print(f"ok: pipeline ran source {out['source_id']}; all proposals pending human commit")


def test_missing_series_drops_phantom_tokens():
    """needs_data must log real series only — never prose/status tokens an auto-impl emits.

    Regression for the live case where a run logged missing_series ['cannot_operationalize',
    'the']: bare prose / status markers must be dropped, real (even uncatalogued) series kept.
    """
    from penrose.pipeline.run import _parse_missing_series as f
    assert f("data_unavailable: crsp_daily_stock_returns") == ["crsp_daily_stock_returns"]
    assert f("data_unavailable: crsp_dlyret, the") == ["crsp_dlyret"]   # drop the article
    assert f("data_unavailable: us_breakeven_10y; long") == ["us_breakeven_10y"]  # drop 'long'
    # all-phantom reasons fall back to a single 'unspecified', never prose
    assert f("data_unavailable: cannot_operationalize, the") == ["unspecified"]
    assert f("data_unavailable: the realized loser returns") == ["unspecified"]


def test_holdout_is_single_use():
    from penrose import config
    from penrose.pipeline import p7_backtest as p7
    import pandas as pd, numpy as np
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        old_root = config.ROOT
        old_holdout_dir = config.HOLDOUT_DIR
        config.ROOT = Path(d)
        config.HOLDOUT_DIR = Path(d) / ".holdout"
        try:
            idx = pd.date_range("2023-01-01", periods=200, freq="5D", tz="UTC")
            net = pd.Series(np.random.default_rng(1).normal(0.01, 0.05, 200), index=idx)
            lock = p7._claim_holdout_lock("unit-test")
            if lock.exists():
                lock.unlink()
            first = p7.final_holdout_eval("unit-test", net, 73.0)
            second = p7.final_holdout_eval("unit-test", net, 73.0)
            assert "holdout_sharpe" in first, "first holdout call should evaluate"
            assert second.get("refused"), "second holdout call must refuse (single-use, S4)"
            assert lock.exists()
        finally:
            config.ROOT = old_root
            config.HOLDOUT_DIR = old_holdout_dir
    print("ok: locked holdout is single-use (refuses the second peek)")


def test_cost_sensitivity_breakeven():
    import numpy as np
    import pandas as pd
    from penrose import config
    from penrose.brain import Claim
    from penrose.pipeline import p7_backtest as p7, stages

    rng = np.random.default_rng(123)
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    signed = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=idx)
    gross = pd.Series(0.0018 + rng.normal(0, 0.0007, n), index=idx)
    payoff = signed * gross
    turn = signed.diff().abs().fillna(signed.abs())
    cost = 0.0005
    net = signed * payoff - turn * cost
    bt = p7.run_backtest(
        "unit-cost", net, turn, 252.0, log=False, payoff=payoff,
        position_signed=signed, cost_frac=cost, family="unit::cost")
    cs = bt.get("cost_sensitivity") or {}
    assert cs["configured_cost_frac"] == cost
    assert cs["breakeven_cost_frac"] is not None
    assert cs["breakeven_cost_frac"] > cost
    assert cs["margin"] == round(cs["breakeven_cost_frac"] / cost, 4)

    claim = Claim(claim_id="cost", statement="cost test", mechanism="", scope="", horizon="",
                  source_id="unit", source_span="", claimed_metric_quote="")
    base = stages.p8_verdict(claim, bt, {}, synthetic=False)
    old_gate = dict(config.COST_SENSITIVITY_GATE)
    try:
        config.COST_SENSITIVITY_GATE.update({"enabled": True, "min_margin": cs["margin"] + 0.1})
        gated = stages.p8_verdict(claim, bt, {}, synthetic=False)
    finally:
        config.COST_SENSITIVITY_GATE.clear()
        config.COST_SENSITIVITY_GATE.update(old_gate)
    assert base.verdict == "watch"
    assert gated.verdict == "kill"
    assert gated.kill_reason == "cost_sensitive"


def test_underpowered_resolution():
    from penrose.brain import Claim
    from penrose.pipeline import stages

    claim = Claim(claim_id="power", statement="power test", mechanism="", scope="", horizon="",
                  source_id="unit", source_span="", claimed_metric_quote="")
    base = {
        "psr": 0.97, "dsr": 0.80, "edge_t": 3.0, "oos_sharpe": 1.5,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.0, 1.0, 1.0], "consistent": True},
        "capacity_usd": 1e6, "bootstrap": {}, "permutation": {}, "regime": {},
    }
    thin = dict(base, n_oos=60)
    d_thin = stages.p8_verdict(claim, thin, {}, False)
    assert d_thin.verdict == "underpowered"
    res = d_thin.metrics.get("resolution")
    assert res is not None
    assert res["current_oos_bars"] == thin["n_oos"]
    assert res["needed_oos_bars"] > thin["n_oos"]
    assert res["more_oos_bars_needed"] == res["needed_oos_bars"] - thin["n_oos"]
    assert res["more_oos_bars_needed"] >= 0
    assert res["needed_breadth_n"] >= 1
    assert res["current_mde_ic"] == d_thin.metrics["mde_ic"]
    assert "breadth via IR" in res["basis"]
    assert "more OOS trades" in d_thin.rationale

    insufficient = stages.p8_verdict(claim, dict(base, n_oos=20), {}, False)
    assert insufficient.verdict == "insufficient_data"
    insuff_res = insufficient.metrics.get("resolution")
    assert insuff_res is not None
    assert insuff_res["current_oos_bars"] == 20
    assert insuff_res["more_oos_bars_needed"] >= 0
    assert "more OOS trades" in insufficient.rationale

    thinner = stages.p8_verdict(claim, dict(base, n_oos=30), {}, False)
    thicker = stages.p8_verdict(claim, dict(base, n_oos=120), {}, False)
    assert thinner.metrics["resolution"]["needed_breadth_n"] >= res["needed_breadth_n"]
    assert res["needed_breadth_n"] >= thicker.metrics["resolution"]["needed_breadth_n"]

    powered = stages.p8_verdict(claim, dict(base, n_oos=1100), {}, False)
    assert powered.verdict == "kill"
    assert powered.metrics.get("resolution") is None


class _ReaderFixture:
    def __init__(self, listed="", hits=""):
        self._listed = listed
        self._hits = hits

    def list(self, prefix="atoms/penrose", n=200):
        return self._listed

    def search(self, query, limit=10):
        return self._hits


def test_corpus_isolation_empty_is_inert():
    from penrose.brain import Claim
    from penrose.pipeline import stages

    claim = Claim(claim_id="iso", statement="BTC volatility carry", mechanism="volatility carry",
                  scope="", horizon="", source_id="unit", source_span="", claimed_metric_quote="")
    bt = {
        "psr": 0.97, "dsr": 0.80, "edge_t": 3.0, "n_oos": 1100, "oos_sharpe": 1.5,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.0, 1.0, 1.0], "consistent": True},
        "capacity_usd": 1e6, "bootstrap": {}, "permutation": {}, "regime": {},
    }
    before = stages.p8_verdict(claim, bt, {}, False)
    advisory = stages.corpus_isolation(claim, _ReaderFixture())
    after = stages.p8_verdict(claim, bt, {}, False)
    after.metrics["corpus_isolation"] = advisory
    assert advisory["isolation_score"] is None
    assert advisory["advisory"] == "corpus empty; no isolation signal"
    assert before.verdict == after.verdict
    assert before.kill_reason == after.kill_reason


def test_corpus_isolation_with_fixture():
    from penrose.brain import Claim
    from penrose.pipeline import stages

    claim = Claim(claim_id="iso2", statement="BTC volatility carry", mechanism="volatility carry",
                  scope="", horizon="", source_id="unit", source_span="", claimed_metric_quote="")
    reader = _ReaderFixture(
        listed="atoms/penrose/claim/old :: Old volatility carry result\n",
        hits=("atoms/penrose/claim/old :: BTC volatility carry invalidation :: 0.91\n"
              "atoms/penrose/principle/vol :: volatility mechanism family :: 0.72\n"),
    )
    advisory = stages.corpus_isolation(claim, reader)
    assert advisory["neighbor_count"] == 2
    assert advisory["mechanism_family_present"] is True
    assert advisory["isolation_score"] is not None
    assert advisory["nearest"][0]["slug"] == "atoms/penrose/claim/old"


if __name__ == "__main__":
    test_data_contract_provenance()
    test_fee_curve_peaks_at_50c()
    test_pipeline_runs_and_proposes_only()
    test_holdout_is_single_use()
    print("\nALL TESTS PASSED")
