import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim
from penrose.data.contract import DataBundle
from penrose.pipeline import p7_backtest as p7
from penrose.pipeline import run as runmod
from penrose.pipeline import stages


class _Module:
    __strategy_class__ = "unit-deflation"
    __module_id__ = "unit-deflation"

    @staticmethod
    def run(_bundle, _claim, _cost_frac):
        net = _net()
        return {
            "ok": True,
            "net": net,
            "positions": pd.Series(1.0, index=net.index),
            "bars_per_year": 252.0,
        }


class _AbortRun(BaseException):
    pass


class _AbortModule(_Module):
    @staticmethod
    def run(_bundle, _claim, _cost_frac):
        raise _AbortRun("simulated paper abort")


def _claim(claim_id: str, cls: str = "unit-deflation") -> Claim:
    return Claim(
        claim_id=claim_id,
        statement="BTC cohort deflation test",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class=cls,
    )


def _net(seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=240, freq="D")
    return pd.Series(rng.normal(0.002, 0.004, len(idx)), index=idx)


def _score(claim: Claim, family: str, *, log: bool = True) -> tuple[dict, str]:
    net = _net()
    bt = p7.run_backtest(
        claim.claim_id,
        net,
        pd.Series(1.0, index=net.index),
        252.0,
        family=family,
        search_cohort_id=claim.search_cohort_id,
        search_denominator=claim.search_denominator,
        log=log,
    )
    dec = stages.p8_verdict(claim, bt, {}, synthetic=False)
    return bt, dec.verdict


def _ready(claims: list[Claim]) -> list[dict]:
    return [{"claim": c, "module": _Module()} for c in claims]


def test_run_cohort_makes_same_strategy_order_independent(tmp_path, monkeypatch):
    old = p7.LEDGER
    try:
        first_ledger = tmp_path / "first.tsv"
        p7.LEDGER = first_ledger
        target_first = _claim("target")
        first_claims = [target_first, *[_claim(f"filler-{i}") for i in range(4)]]
        first_ready = _ready(first_claims)
        runmod._register_run_cohorts(first_ready, "unit-paper")
        first_family = first_ready[0]["family"]
        first_bt, first_verdict = _score(target_first, first_family)

        p7.LEDGER = tmp_path / "last.tsv"
        target_last = _claim("target")
        last_claims = [*[_claim(f"filler-{i}") for i in range(4)], target_last]
        last_ready = _ready(last_claims)
        runmod._register_run_cohorts(last_ready, "unit-paper")
        last_family = last_ready[-1]["family"]
        for item in last_ready[:-1]:
            _score(item["claim"], item["family"])
        last_bt, last_verdict = _score(target_last, last_family)
    finally:
        p7.LEDGER = old

    first_family_n = first_bt["n_trials"] - int(first_bt["regime"].get("n_partitions", 0))
    last_family_n = last_bt["n_trials"] - int(last_bt["regime"].get("n_partitions", 0))
    assert first_bt["n_trials"] == last_bt["n_trials"]
    assert first_family_n == last_family_n == 5
    assert first_verdict == last_verdict


def test_trial_stats_never_discounts_distinct_strategies_or_unrelated_family(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        p7.register_trials([
            {
                "strategy": f"registered-{i}",
                "family": "unit::crypto",
                "generation_source": "paper",
                "search_cohort_id": "small-cohort",
                "search_denominator": 2,
            }
            for i in range(2)
        ])
        for i in range(5):
            p7._append_ledger(
                f"distinct-{i}",
                {"per_trade_sharpe": 0.1, "dsr": 0.5, "n": i + 1},
                "unit::crypto",
            )
        n, _ = p7._trial_stats("unit::crypto", "distinct-4")
        unrelated_n, _ = p7._trial_stats("other::crypto", "other")
    finally:
        p7.LEDGER = old

    assert n == 7
    assert unrelated_n == 1


def test_single_claim_cohort_is_byte_identical_to_running_count(tmp_path):
    old = p7.LEDGER
    try:
        claim = _claim("single")
        family = runmod._family(claim, _Module())

        p7.LEDGER = tmp_path / "baseline.tsv"
        baseline, _ = _score(claim, family, log=False)

        p7.LEDGER = tmp_path / "cohort.tsv"
        cohort_claim = _claim("single")
        ready = _ready([cohort_claim])
        runmod._register_run_cohorts(ready, "unit-paper")
        cohort, _ = _score(cohort_claim, ready[0]["family"], log=False)
    finally:
        p7.LEDGER = old

    assert cohort == baseline
    assert cohort["n_trials"] - int(cohort["regime"].get("n_partitions", 0)) == 1


def test_run_cohort_registration_failure_falls_back_to_running_count(tmp_path, monkeypatch):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    claims = [_claim("a"), _claim("b")]
    ready = _ready(claims)

    def _boom(_rows):
        raise OSError("simulated ledger failure")

    monkeypatch.setattr(runmod.p7_backtest, "register_trials", _boom)
    try:
        runmod._register_run_cohorts(ready, "unit-paper")
        assert claims[0].search_cohort_id is None
        assert claims[0].search_denominator is None
        bt, _ = _score(claims[0], runmod._family(claims[0], _Module()), log=False)
    finally:
        p7.LEDGER = old

    assert bt["n_trials"] - int(bt["regime"].get("n_partitions", 0)) == 1


def test_run_cohort_ignores_stale_claim_denominator(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    stale = _claim("stale")
    stale.search_cohort_id = "old-cohort"
    stale.search_denominator = 99
    fresh = _claim("fresh")
    ready = _ready([stale, fresh])
    try:
        runmod._register_run_cohorts(ready, "unit-paper")
        n, _ = p7._trial_stats(ready[0]["family"], "stale")
    finally:
        p7.LEDGER = old

    assert stale.search_denominator == 2
    assert stale.search_cohort_id != "old-cohort"
    assert n == 2


def test_run_source_cleans_abandoned_paper_cohort_before_later_same_family_run(tmp_path, monkeypatch):
    old_ledger = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    _isolate_run_outputs(tmp_path, monkeypatch)
    _install_registry(monkeypatch, _AbortModule())
    paper = tmp_path / "run-a.txt"
    paper.write_text("BTC cohort cleanup test")
    claims = [_claim(f"run-a-{i}") for i in range(5)]

    try:
        with pytest_raises_abort():
            runmod.run_source(
                paper,
                use_llm=False,
                claims_override=claims,
                bundle_override=DataBundle(),
                force=True,
            )
        ledger = p7._canonicalize_ledger()
        assert ledger.empty

        later_claims = [_claim(f"run-b-{i}") for i in range(2)]
        later_ready = _ready(later_claims)
        runmod._register_run_cohorts(later_ready, "run-b")
        bt, _ = _score(later_claims[0], later_ready[0]["family"], log=False)
    finally:
        p7.LEDGER = old_ledger

    assert bt["n_trials"] - int(bt["regime"].get("n_partitions", 0)) == 2


def test_scored_prior_paper_strategy_still_counts_for_later_same_family_run(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        prior = _claim("prior-scored")
        prior_ready = _ready([prior])
        runmod._register_run_cohorts(prior_ready, "prior-paper")
        _score(prior, prior_ready[0]["family"])
        runmod._cleanup_run_cohorts(prior_ready)

        later_claims = [_claim("later-a"), _claim("later-b")]
        later_ready = _ready(later_claims)
        runmod._register_run_cohorts(later_ready, "later-paper")
        bt, _ = _score(later_claims[0], later_ready[0]["family"], log=False)
    finally:
        p7.LEDGER = old

    assert bt["n_trials"] - int(bt["regime"].get("n_partitions", 0)) == 3


class pytest_raises_abort:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, _tb):
        assert exc_type is _AbortRun
        return True


def _install_registry(monkeypatch, module) -> None:
    monkeypatch.setattr(runmod, "_register_known_modules", lambda: None)
    monkeypatch.setattr(runmod, "REGISTRY", {"unit-deflation": module})
    monkeypatch.setattr(runmod, "_REGISTRY_ALIAS_OWNERS", {})
    monkeypatch.setattr(runmod, "_REGISTRY_CANONICAL_OWNERS", {})


def _isolate_run_outputs(tmp_path, monkeypatch) -> None:
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
