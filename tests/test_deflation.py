import pandas as pd
from penrose.pipeline import p7_backtest as P7


def test_absurd_cohort_denominator_is_capped(tmp_path, monkeypatch):
    """A poisoned ledger row (1e12 trials) must not annihilate the DSR — clamp per-cohort denominator."""
    led = tmp_path / "backtest_ledger.tsv"
    led.write_text(
        "strategy\tfamily\tgeneration_source\tsearch_cohort_id\tsearch_denominator\tper_trade_sharpe\tdsr\tn\n"
        "evil\tcrypto::crypto\tpaper\tevil-cohort\t1000000000000\t0.1\t0.5\t100\n"
    )
    monkeypatch.setattr(P7, "LEDGER", led)
    n_trials, _ = P7._trial_stats(family=None, strategy="weather_x", declared_grid_size=1)
    assert n_trials <= P7._MAX_COHORT_DENOMINATOR + 10  # clamped, not 1e12
