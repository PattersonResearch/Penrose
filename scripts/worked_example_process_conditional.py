#!/usr/bin/env python
"""Deterministic worked example: same returns, two search processes."""
from __future__ import annotations

import hashlib
import math
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from penrose import config, stats  # noqa: E402
from penrose.brain import Claim, Decision  # noqa: E402
from penrose.pipeline import p7_backtest, stages  # noqa: E402
from penrose.pipeline.p7_backtest import _trial_stats as trial_stats  # noqa: E402

BARS_PER_YEAR = 252.0
COST_FRAC = 0.0008
FAMILY = "worked-example::process-conditional"
WINNER = "worked-example-winner"
COHORT_A = "worked-example-preregistered"
COHORT_B = "worked-example-200"
N_TRIALS_B = 200
SR_VARIANCE_B = 0.0005
SEED = 20260623
N_BARS = 4000
EFFECT = 0.055


@dataclass(frozen=True)
class ProcessResult:
    label: str
    search_denominator: int
    ledger_sr_variance: float
    bt: dict[str, Any]
    holdout: dict[str, Any]
    decision: Decision

    @property
    def n_trials(self) -> int:
        return int(self.bt["n_trials"])

    @property
    def n_oos(self) -> int:
        return int(self.bt["n_oos"])

    @property
    def psr(self) -> float:
        return float(self.bt["psr"])

    @property
    def dsr(self) -> float:
        return float(self.bt["dsr"])

    @property
    def verdict(self) -> str:
        return self.decision.verdict

    @property
    def kill_reason(self) -> str | None:
        return self.decision.kill_reason


@dataclass(frozen=True)
class WorkedExample:
    series: np.ndarray
    series_hash_a: str
    series_hash_b: str
    bars_per_year: float
    full_sharpe: float
    thresholds: dict[str, Any]
    process_a: ProcessResult
    process_b: ProcessResult


@contextmanager
def _temporary_p7_ledger(path: Path):
    old = p7_backtest.LEDGER
    p7_backtest.LEDGER = path
    try:
        yield
    finally:
        p7_backtest.LEDGER = old


def _claim() -> Claim:
    return Claim(
        claim_id="worked-example-process-conditional",
        statement="Constructed returns illustrate process-conditional DSR deflation.",
        mechanism="deterministic worked example",
        scope="synthetic returns",
        horizon="daily",
        source_id="worked-example",
        source_span="synthetic worked example",
        claimed_metric_quote="same R, different disclosed search denominator",
    )


def _series_hash(net: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(net, dtype=np.float64).tobytes()).hexdigest()


def _base_series(length: int = N_BARS) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    z = rng.normal(0.0, 1.0, length)
    return (z - z.mean()) / z.std(ddof=1)


def _candidate_series() -> np.ndarray:
    return ((_base_series() + EFFECT) * 0.01).astype(np.float64)


def _index(length: int) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", periods=length, freq="D", tz="UTC")


def _positions(length: int, index: pd.DatetimeIndex) -> pd.Series:
    pos = np.where(np.arange(length) % 2 == 0, 1.0, -1.0)
    return pd.Series(pos, index=index)


def _payoff_for_exact_net(net: np.ndarray, positions: pd.Series) -> pd.Series:
    # Makes position * payoff - cost exactly equal to the supplied net series.
    return pd.Series(positions.to_numpy() * (net + COST_FRAC), index=positions.index)


def _walk_forward_frame(length: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    t = np.arange(length)
    return pd.DataFrame(
        {
            "signal": np.linspace(-2.0, 2.0, length),
            "fut_rv": 0.02 + 0.002 * np.sin(t * 0.17),
            "iv": 0.0,
        },
        index=index,
    )


def _per_trade_sharpe(net: np.ndarray) -> float:
    return float(np.mean(net) / np.std(net, ddof=1))


def _ledger_rows_for_a(net: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": WINNER,
                "family": FAMILY,
                "generation_source": "worked_example",
                "search_cohort_id": COHORT_A,
                "search_denominator": 1,
                "per_trade_sharpe": _per_trade_sharpe(net),
                "dsr": "",
                "n": "",
            }
        ]
    )


def _ledger_rows_for_b(net: np.ndarray) -> pd.DataFrame:
    winner_sr = _per_trade_sharpe(net)
    deviations = np.r_[0.0, -np.linspace(0.01, 1.0, N_TRIALS_B - 1)]
    scale = math.sqrt(SR_VARIANCE_B / float(np.var(deviations, ddof=1)))
    trial_srs = winner_sr + deviations * scale
    rows = []
    for i, sr in enumerate(trial_srs):
        rows.append(
            {
                "strategy": WINNER if i == 0 else f"worked-example-variant-{i:03d}",
                "family": FAMILY,
                "generation_source": "worked_example",
                "search_cohort_id": COHORT_B,
                "search_denominator": N_TRIALS_B,
                "per_trade_sharpe": float(sr),
                "dsr": "",
                "n": "",
            }
        )
    return pd.DataFrame(rows)


def _write_ledger(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def _score_process(
    net: np.ndarray,
    ledger: pd.DataFrame,
    ledger_path: Path,
    label: str,
    search_denominator: int,
) -> ProcessResult:
    index = _index(len(net))
    net_series = pd.Series(net, index=index)
    positions = _positions(len(net), index)
    _write_ledger(ledger, ledger_path)
    _, ledger_sr_variance = trial_stats(FAMILY, WINNER)
    bt = p7_backtest.run_backtest(
        WINNER,
        net_series,
        positions,
        BARS_PER_YEAR,
        log=False,
        payoff=_payoff_for_exact_net(net, positions),
        position_signed=positions,
        cost_frac=COST_FRAC,
        wf_frame=_walk_forward_frame(len(net), index),
        family=FAMILY,
    )
    holdout = p7_backtest.final_holdout_eval(WINNER, net_series, BARS_PER_YEAR, force=True)
    decision = stages.p8_verdict(_claim(), bt, holdout, synthetic=False)
    return ProcessResult(
        label=label,
        search_denominator=search_denominator,
        ledger_sr_variance=float(ledger_sr_variance),
        bt=bt,
        holdout=holdout,
        decision=decision,
    )


def build_example(write_markdown: bool = True) -> WorkedExample:
    net_a = _candidate_series()
    net_b = net_a.copy()
    sha_a = _series_hash(net_a)
    sha_b = _series_hash(net_b)
    assert sha_a == sha_b, "Process A and B net return series are not byte-identical"

    with tempfile.TemporaryDirectory(prefix="penrose-worked-example-") as tmp:
        ledger_path = Path(tmp) / "backtest_ledger.tsv"
        with _temporary_p7_ledger(ledger_path):
            process_a = _score_process(net_a, _ledger_rows_for_a(net_a), ledger_path, "A", 1)
            process_b = _score_process(
                net_b, _ledger_rows_for_b(net_b), ledger_path, "B", N_TRIALS_B
            )

    assert process_a.verdict in {"research-supported", "watch"}, process_a
    assert process_b.verdict == "kill", process_b
    assert process_b.kill_reason in {"no_oos_edge", "negative_dsr"}, process_b
    assert process_a.verdict != process_b.verdict, (process_a, process_b)

    example = WorkedExample(
        series=net_a,
        series_hash_a=sha_a,
        series_hash_b=sha_b,
        bars_per_year=BARS_PER_YEAR,
        full_sharpe=float(stats.sharpe(net_a, BARS_PER_YEAR)),
        thresholds=dict(config.DSR_DECISION),
        process_a=process_a,
        process_b=process_b,
    )
    if write_markdown:
        write_worked_example(example)
    return example


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _gate_lines(result: ProcessResult) -> list[str]:
    bt = result.bt
    return [
        f"- {result.label} folds: {bt['three_fold']}",
        f"- {result.label} regime: fragile={bt['regime'].get('fragile')} "
        f"n_partitions={bt['regime'].get('n_partitions')}",
        f"- {result.label} bootstrap: edge_ci={bt['bootstrap'].get('edge_ci')} "
        f"includes_zero={bt['bootstrap'].get('edge_ci_includes_zero')}",
        f"- {result.label} permutation: p_value={bt['permutation'].get('p_value')}",
        f"- {result.label} walk_forward: consistent={bt['walk_forward'].get('consistent')} "
        f"(per-window Sharpe magnitudes are an artifact of the synthetic walk-forward frame, "
        f"not R's full-series Sharpe; the gate only checks consistency)",
        f"- {result.label} holdout: {_plain(result.holdout)}",
        f"- {result.label} p8 rationale: {result.decision.rationale}",
    ]


def render_markdown(example: WorkedExample) -> str:
    a, b = example.process_a, example.process_b
    gates = "\n".join(_gate_lines(a) + _gate_lines(b))
    return f"""# Worked Example: Process-Conditional Verdict

This is a constructed, isolated illustration of Penrose's deflation gate running inside the real P7/P8 pipeline. It is not alpha, not a real strategy, and not evidence of a profitable trading rule. The same byte-identical return series `R` is scored twice through `run_backtest(...)` and `stages.p8_verdict(...)`; only the disclosed trial ledger changes.

By construction, `R` is tuned to sit just inside the boundary: Process B's deflated DSR lands just **below** the kill threshold while A stays above it, so the flip is deliberately a near-threshold case (a small change to the effect size would move B back across the line). This is intentional — the point is to isolate deflation as the *only* gate that differs, not to claim a wide margin.

Full verdicts include more than DSR: minimum OOS bars, 3-fold sign stability, regime fragility, bootstrap edge CI, permutation alignment, walk-forward consistency, holdout confirmation, and provenance caps. In this construction those non-deflation gates pass for both processes; Process B is killed only after the ledger deflates the OOS-slice DSR.

Same byte-identical return series:

- SHA-256: `{example.series_hash_a}`
- Length: `{len(example.series)}`
- OOS bars: `{a.n_oos}`
- Full-series annualized Sharpe, descriptive only: `{_fmt(example.full_sharpe)}`
- DSR thresholds: kill below `{example.thresholds["kill_below_psr"]}`, watch band `{example.thresholds["watch_band"]}`

| Process | Search lineage | Series hash | Bars/year | n_oos | n_trials | ledger sr_variance | OOS PSR | OOS DSR | Real P8 verdict | kill_reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A: preregistered one hypothesis | {a.search_denominator} | `{example.series_hash_a}` | {example.bars_per_year:.0f} | {a.n_oos} | {a.n_trials} | {_fmt(a.ledger_sr_variance)} | {_fmt(a.psr)} | {_fmt(a.dsr)} | {a.verdict} | {a.kill_reason} |
| B: selected best of 200 | {b.search_denominator} | `{example.series_hash_b}` | {example.bars_per_year:.0f} | {b.n_oos} | {b.n_trials} | {_fmt(b.ledger_sr_variance)} | {_fmt(b.psr)} | {_fmt(b.dsr)} | {b.verdict} | {b.kill_reason} |

Gate output:

{gates}

Why the verdict changes:

1. A normal backtester sees only `R`; Penrose also sees the search lineage that produced `R`.
2. P7 computes PSR/DSR on the OOS slice, not the full series, and adds regime partitions to the ledger trial count.
3. The returns, dates, bars/year, costs, robustness inputs, and holdout are unchanged. Process B's 200-trial ledger raises the deflation denominator and cross-trial Sharpe variance, dropping only B below the P8 DSR kill threshold.
"""


def write_worked_example(example: WorkedExample) -> None:
    (ROOT / "docs" / "WORKED_EXAMPLE.md").write_text(render_markdown(example))


def render_audit(example: WorkedExample) -> str:
    a, b = example.process_a, example.process_b
    lines = [
        "Penrose process-conditional worked example",
        f"series_sha256        : {example.series_hash_a}",
        f"series_hash_equal    : {example.series_hash_a == example.series_hash_b}",
        f"length               : {len(example.series)}",
        f"bars_per_year        : {example.bars_per_year:.0f}",
        f"full_sharpe_desc     : {_fmt(example.full_sharpe)}",
        "",
        "process                         search_denominator  n_oos  n_trials  ledger_sr_var  PSR       DSR       verdict  kill_reason",
        f"A: preregistered one hypothesis {a.search_denominator:>18}  {a.n_oos:>5}  {a.n_trials:>8}  {_fmt(a.ledger_sr_variance):>13}  {_fmt(a.psr):>8}  {_fmt(a.dsr):>8}  {a.verdict:<8} {a.kill_reason}",
        f"B: selected best of 200         {b.search_denominator:>18}  {b.n_oos:>5}  {b.n_trials:>8}  {_fmt(b.ledger_sr_variance):>13}  {_fmt(b.psr):>8}  {_fmt(b.dsr):>8}  {b.verdict:<8} {b.kill_reason}",
        "",
        "Gate output:",
        *_gate_lines(a),
        *_gate_lines(b),
        "",
        "Wrote WORKED_EXAMPLE.md",
    ]
    return "\n".join(lines)


def main() -> int:
    example = build_example(write_markdown=True)
    print(render_audit(example))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
