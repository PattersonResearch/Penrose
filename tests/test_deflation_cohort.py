import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim
from penrose import config
from penrose.data.contract import DataBundle
from penrose.pipeline import fidelity_memory
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


def _provided_claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement="The pooled mean of declared provided series is greater than zero.",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="one pooled statistic across declared series",
        applicable_strategy_class="unit-provided-series",
    )


def _provided_preregistered_claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=(
            "The declared provided series has exactly one pre-registered statistic; "
            "because it is a single pooled test, no multiplicity correction applies."
        ),
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="single pooled test",
        applicable_strategy_class="unit-provided-series",
    )


def _exp1b_claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=(
            "A strategy that buys the mispriced side of settled tail markets "
            "earns a positive mean net P&L"
        ),
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class="unit-provided-series",
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
    # PEN-06: paper cohorts below the external effective-trials prior floor at 10.
    assert first_family_n == last_family_n == config.DEFLATION_PRIOR["external_min_trials"]
    assert first_bt["dsr"] == last_bt["dsr"]
    assert first_verdict == last_verdict


def test_is_preregistered_single_cohort_requires_explicit_single_cohort_assertion():
    assert fidelity_memory.is_preregistered_single_cohort(_provided_preregistered_claim("exp-1b"))

    momentum = Claim(
        claim_id="momentum",
        statement="Momentum was validated with a one-sample t-test on daily P&L.",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class="unit-provided-series",
    )
    pooled_mean = Claim(
        claim_id="pooled",
        statement="We compute a pooled mean over the declared series.",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class="unit-provided-series",
    )
    assert not fidelity_memory.is_preregistered_single_cohort(momentum)
    assert not fidelity_memory.is_preregistered_single_cohort(pooled_mean)


def test_exp1b_keeps_provided_series_and_singleton_preregistration():
    source = SimpleNamespace(text=(
        "Pool these 8 net-tail-P&L series (each already encodes the per-market net P&L "
        "above, held to settlement) into a pooled sample and run a single one-sided "
        "one-sample t-test on that pooled sample. There is exactly one pre-registered "
        "statistic; because it is a single pooled test, no multiplicity correction applies."
    ))
    claim = _exp1b_claim("exp-1b")

    assert fidelity_memory.classify_claim_type(claim, source) == "provided_series_statistic"
    assert fidelity_memory.is_preregistered_single_cohort(claim, source) is True


def test_preregistered_single_cohort_rejects_generic_statistical_prose():
    generic_sources = [
        "the F-statistic has denominator of 1 degree of freedom",
        "the denominator is 1 for this unit-root test",
        "we report one test, with no multiplicity correction",
    ]
    for text in generic_sources:
        assert fidelity_memory.is_preregistered_single_cohort(
            _provided_claim("generic"), SimpleNamespace(text=text)
        ) is False

    assert fidelity_memory.is_preregistered_single_cohort(
        _provided_claim("explicit-stat"),
        SimpleNamespace(text="exactly one pre-registered statistic; single pooled test"),
    ) is True
    assert fidelity_memory.is_preregistered_single_cohort(
        _provided_claim("explicit-search"),
        SimpleNamespace(text="This counts as one pre-registered search."),
    ) is True


def test_provided_series_provenance_phrase_obeys_trading_veto():
    claim = Claim(
        claim_id="momentum-provenance",
        statement="The long-short momentum strategy earns positive excess returns",
        mechanism=(
            "We go long the top decile and go short the bottom decile, rebalanced "
            "monthly, using the provided series of excess returns from CRSP"
        ),
        scope="",
        horizon="",
        source_id="unit-source",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class="momentum",
    )
    source = SimpleNamespace(text="F-tests whose denominator of 1 degree of freedom are reported.")

    assert fidelity_memory.classify_claim_type(claim, source) == "trading_strategy"
    assert fidelity_memory.is_preregistered_single_cohort(claim, source) is False


def test_provided_series_without_preregistration_uses_normal_cohort_deflation(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        claims = [_provided_claim("provided-a"), _provided_claim("provided-b")]
        ready = _ready(claims)
        runmod._register_run_cohorts(ready, "unit-paper")
        rows = p7._canonicalize_ledger()
        n_a, _ = p7._trial_stats(
            ready[0]["family"], "provided-a",
            registered_trials=claims[0].search_denominator,
            generation_source="provided_series_statistic",
            search_cohort_id=claims[0].search_cohort_id,
            preregistered_single_cohort=False,
        )
        net = _net()
        bt = p7.run_backtest(
            "provided-a",
            net,
            pd.Series(1.0, index=net.index),
            252.0,
            family=ready[0]["family"],
            generation_source="provided_series_statistic",
            search_cohort_id=claims[0].search_cohort_id,
            search_denominator=claims[0].search_denominator,
            preregistered_single_cohort=False,
            log=False,
        )
    finally:
        p7.LEDGER = old

    assert fidelity_memory.classify_claim_type(claims[0]) == "provided_series_statistic"
    assert not fidelity_memory.is_preregistered_single_cohort(claims[0])
    assert claims[0].search_denominator == 2
    assert claims[1].search_denominator == 2
    assert set(rows["search_denominator"].astype(int)) == {2}
    assert set(rows["generation_source"].astype(str)) == {"provided_series_statistic"}
    assert ready[0]["family"] == ready[1]["family"]
    assert n_a == config.DEFLATION_PRIOR["external_min_trials"]
    assert bt["n_trials"] > 1
    assert int(bt["regime"].get("n_partitions", 0)) > 0


def test_generic_denominator_prose_does_not_create_singleton_provided_deflation(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    source = SimpleNamespace(text="F-tests whose denominator of 1 degree of freedom are reported.")
    try:
        claims = [_provided_claim("provided-a"), _provided_claim("provided-b")]
        ready = _ready(claims)
        runmod._register_run_cohorts(ready, "unit-paper", source=source)
        net = _net()
        bt = p7.run_backtest(
            "provided-a",
            net,
            pd.Series(1.0, index=net.index),
            252.0,
            family=ready[0]["family"],
            generation_source="provided_series_statistic",
            search_cohort_id=claims[0].search_cohort_id,
            search_denominator=claims[0].search_denominator,
            preregistered_single_cohort=fidelity_memory.is_preregistered_single_cohort(
                claims[0], source
            ),
            log=False,
        )
    finally:
        p7.LEDGER = old

    assert fidelity_memory.classify_claim_type(claims[0], source) == "provided_series_statistic"
    assert fidelity_memory.is_preregistered_single_cohort(claims[0], source) is False
    assert claims[0].search_denominator == 2
    assert ready[0]["family"] == ready[1]["family"]
    assert bt["n_trials"] != 1
    assert int(bt["regime"].get("n_partitions", 0)) > 0


def test_preregistered_provided_series_keeps_singleton_deflation_break(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        claims = [
            _provided_preregistered_claim("provided-a"),
            _provided_preregistered_claim("provided-b"),
        ]
        ready = _ready(claims)
        runmod._register_run_cohorts(ready, "unit-paper")
        rows = p7._canonicalize_ledger()
        n_a, _ = p7._trial_stats(
            ready[0]["family"], "provided-a",
            registered_trials=claims[0].search_denominator,
            generation_source="provided_series_statistic",
            search_cohort_id=claims[0].search_cohort_id,
            preregistered_single_cohort=True,
        )
        net = _net()
        bt = p7.run_backtest(
            "provided-a",
            net,
            pd.Series(1.0, index=net.index),
            252.0,
            family=ready[0]["family"],
            generation_source="provided_series_statistic",
            search_cohort_id=claims[0].search_cohort_id,
            search_denominator=claims[0].search_denominator,
            preregistered_single_cohort=True,
            log=False,
        )
    finally:
        p7.LEDGER = old

    assert claims[0].search_denominator == 1
    assert claims[1].search_denominator == 1
    assert set(rows["search_denominator"].astype(int)) == {1}
    assert ready[0]["family"] != ready[1]["family"]
    assert n_a == 1
    assert bt["regime"] == {
        "n_partitions": 0,
        "fragile": False,
        "provided_series_statistic": True,
    }
    assert bt["n_trials"] == 1


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
        n, _ = p7._trial_stats("unit::crypto", "distinct-4", generation_source="generated")
        unrelated_n, _ = p7._trial_stats("other::crypto", "other", generation_source="generated")
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
    # PEN-06: paper-source single claims now use the external effective-trials floor.
    assert (
        cohort["n_trials"] - int(cohort["regime"].get("n_partitions", 0))
        == config.DEFLATION_PRIOR["external_min_trials"]
    )


def test_declared_grid_size_counts_product_fail_open_and_caps():
    assert p7.declared_grid_size({
        "param_grid": {"window": [10, 20, 60], "thr": [1, 2]},
    }) == 6
    assert p7.declared_grid_size(None) == 1
    assert p7.declared_grid_size({}) == 1
    assert p7.declared_grid_size({"param_grid": {}}) == 1
    assert p7.declared_grid_size({"param_grid": {"window": []}}) == 1
    assert p7.declared_grid_size({"param_grid": {"window": "10,20,60"}}) == 1
    assert p7.declared_grid_size({
        "param_grid": {"window": list(range(101)), "thr": list(range(101))},
    }) == p7.DECLARED_GRID_SIZE_CAP


def test_declared_parameter_grid_floors_trials_and_deflates_more(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        net = _net()
        positions = pd.Series(1.0, index=net.index)
        baseline = p7.run_backtest(
            "grid-baseline", net, positions, 252.0, family="unit::grid", log=False)
        unit_grid = p7.run_backtest(
            "grid-unit", net, positions, 252.0, family="unit::grid",
            declared_grid_size=1, log=False)
        wide_grid = p7.run_backtest(
            "grid-wide", net, positions, 252.0, family="unit::grid",
            declared_grid_size=20, log=False)
    finally:
        p7.LEDGER = old

    base_family_n = baseline["n_trials"] - int(baseline["regime"].get("n_partitions", 0))
    wide_family_n = wide_grid["n_trials"] - int(wide_grid["regime"].get("n_partitions", 0))
    assert unit_grid["n_trials"] == baseline["n_trials"]
    assert unit_grid["dsr"] == baseline["dsr"]
    assert baseline["declared_grid_size"] == 1
    assert wide_grid["declared_grid_size"] == 20
    assert wide_family_n >= 20
    assert wide_family_n > base_family_n
    assert wide_grid["n_trials"] > baseline["n_trials"]
    assert wide_grid["dsr"] < baseline["dsr"]


def test_preregistered_provided_series_ignores_declared_grid_floor(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        n, _ = p7._trial_stats(
            "provided-series::claim", "provided-grid",
            generation_source="provided_series_statistic",
            preregistered_single_cohort=True,
            declared_grid_size=20,
        )
        net = _net()
        bt = p7.run_backtest(
            "provided-grid",
            net,
            pd.Series(1.0, index=net.index),
            252.0,
            family="provided-series::claim",
            generation_source="provided_series_statistic",
            preregistered_single_cohort=True,
            declared_grid_size=20,
            log=False,
        )
    finally:
        p7.LEDGER = old

    assert n == 1
    assert bt["declared_grid_size"] == 20
    assert bt["regime"] == {
        "n_partitions": 0,
        "fragile": False,
        "provided_series_statistic": True,
    }
    assert bt["n_trials"] == 1


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

    assert (
        bt["n_trials"] - int(bt["regime"].get("n_partitions", 0))
        == config.DEFLATION_PRIOR["external_min_trials"]
    )


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
        n, _ = p7._trial_stats(ready[0]["family"], "stale", generation_source="generated")
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

    assert (
        bt["n_trials"] - int(bt["regime"].get("n_partitions", 0))
        == config.DEFLATION_PRIOR["external_min_trials"]
    )


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

    assert (
        bt["n_trials"] - int(bt["regime"].get("n_partitions", 0))
        == config.DEFLATION_PRIOR["external_min_trials"]
    )


def test_deflation_prior_floor_applies_for_paper_source(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        bt, _ = _score(_claim("paper-floor"), "unit::crypto", log=False)
    finally:
        p7.LEDGER = old

    assert bt["n_trials"] - int(bt["regime"].get("n_partitions", 0)) >= 10
    assert bt["dsr"] < bt["psr"]


def test_deflation_prior_floor_not_applied_for_generated_source(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        n, var = p7._trial_stats("generated::crypto", "candidate", generation_source="generated")
    finally:
        p7.LEDGER = old

    assert n == config.DEFLATION_PRIOR["generated_min_trials"]
    assert var == config.DEFLATION_PRIOR["sr_var_prior"]


def test_variance_prior_active_with_no_scored_siblings(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        n, var = p7._trial_stats("unit::crypto", "first-paper")
    finally:
        p7.LEDGER = old

    assert n == config.DEFLATION_PRIOR["external_min_trials"]
    assert var == config.DEFLATION_PRIOR["sr_var_prior"]


def test_current_cohort_excluded_from_empirical_variance(tmp_path):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        p7._append_ledger(
            "prior-a", {"per_trade_sharpe": 0.01, "dsr": 0.1, "n": 10},
            "unit::crypto", search_cohort_id="prior")
        p7._append_ledger(
            "prior-b", {"per_trade_sharpe": 0.03, "dsr": 0.1, "n": 10},
            "unit::crypto", search_cohort_id="prior")
        p7._append_ledger(
            "current-a", {"per_trade_sharpe": 9.0, "dsr": 0.1, "n": 10},
            "unit::crypto", search_cohort_id="current")
        _, var_without_current = p7._trial_stats(
            "unit::crypto", "current-b", search_cohort_id="current")
        _, var_with_current = p7._trial_stats("unit::crypto", "current-b")
    finally:
        p7.LEDGER = old

    assert var_without_current < var_with_current
    assert var_without_current == config.DEFLATION_PRIOR["sr_var_prior"]


def test_trial_registration_cannot_shrink_or_overwrite_scored_rows(tmp_path, capsys):
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        p7.register_trials([{
            "strategy": "s1",
            "family": "unit::crypto",
            "generation_source": "paper",
            "search_cohort_id": "declared",
            "search_denominator": 500,
        }])
        p7.register_trials([{
            "strategy": "s1",
            "family": "unit::crypto",
            "generation_source": "paper",
            "search_cohort_id": "declared",
            "search_denominator": 5,
        }])
        row = p7._canonicalize_ledger().iloc[0]
        n, _ = p7._trial_stats("unit::crypto", "s1", generation_source="generated")

        p7._append_ledger(
            "s1",
            {"per_trade_sharpe": 0.12, "dsr": 0.4, "n": 100},
            "unit::crypto",
            generation_source="paper",
            search_cohort_id="declared",
            search_denominator=500,
        )
        p7.register_trials([{
            "strategy": "s1",
            "family": "unit::crypto",
            "generation_source": "paper",
            "search_cohort_id": "declared",
            "search_denominator": 5,
        }])
        scored = p7._canonicalize_ledger().iloc[0]
        warning = capsys.readouterr().err
    finally:
        p7.LEDGER = old

    assert int(row["search_denominator"]) == 500
    assert n == 500
    assert float(scored["per_trade_sharpe"]) == 0.12
    assert int(scored["search_denominator"]) == 500
    assert "registration for scored strategy 's1' ignored" in warning


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


def test_declared_grid_size_deflation_floor(tmp_path):
    """#58 core: a declared param grid floors n_trials (charges the parameter search); grid<=1 is inert.
    Isolates the global ledger so the floor — not accumulated history — determines n_trials."""
    import numpy as np, pandas as pd
    from penrose.pipeline import p7_backtest as P
    assert P.declared_grid_size({"param_grid": {"w": [10, 20, 60], "t": [1, 2]}}) == 6
    assert P.declared_grid_size({}) == 1 and P.declared_grid_size({"param_grid": "x"}) == 1
    idx = pd.date_range("2023-01-01", periods=200, freq="D", tz="UTC")
    net = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200), index=idx)
    pos = pd.Series(1.0, index=idx)
    old = P.LEDGER
    try:
        P.LEDGER = tmp_path / "base.tsv"
        base = P.run_backtest("g1", net, pos, 252.0, log=False, declared_grid_size=1)
        P.LEDGER = tmp_path / "grid.tsv"
        grid = P.run_backtest("g2", net, pos, 252.0, log=False, declared_grid_size=30)
    finally:
        P.LEDGER = old
    assert grid["n_trials"] >= 30            # the declared 30-wide grid floors n_trials
    assert grid["n_trials"] > base["n_trials"]  # and deflates strictly more than the no-grid run
