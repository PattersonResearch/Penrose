# Worked Example: Process-Conditional Verdict

This is a constructed, isolated illustration of Penrose's deflation gate running inside the real P7/P8 pipeline. It is not alpha, not a real strategy, and not evidence of a profitable trading rule. The same byte-identical return series `R` is scored twice through `run_backtest(...)` and `stages.p8_verdict(...)`; only the disclosed trial ledger changes.

By construction, `R` is tuned to sit just inside the boundary: Process B's deflated DSR lands just **below** the kill threshold while A stays above it, so the flip is deliberately a near-threshold case (a small change to the effect size would move B back across the line). This is intentional — the point is to isolate deflation as the *only* gate that differs, not to claim a wide margin.

Full verdicts include more than DSR: minimum OOS bars, 3-fold sign stability, regime fragility, bootstrap edge CI, permutation alignment, walk-forward consistency, holdout confirmation, and provenance caps. In this construction those non-deflation gates pass for both processes; Process B is killed only after the ledger deflates the OOS-slice DSR.

Same byte-identical return series:

- SHA-256: `a4c125c10ab55b515612347f0185523df6fe065b6e0d9d950ca675ff055b19cd`
- Length: `4000`
- OOS bars: `1200`
- Full-series annualized Sharpe, descriptive only: `0.873098`
- DSR thresholds: kill below `0.9`, watch band `(0.9, 0.95)`

| Process | Search lineage | Series hash | Bars/year | n_oos | n_trials | ledger sr_variance | OOS PSR | OOS DSR | Real P8 verdict | kill_reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A: preregistered one hypothesis | 1 | `a4c125c10ab55b515612347f0185523df6fe065b6e0d9d950ca675ff055b19cd` | 252 | 1200 | 10 | 0.000000 | 0.999700 | 0.999700 | watch | None |
| B: selected best of 200 | 200 | `a4c125c10ab55b515612347f0185523df6fe065b6e0d9d950ca675ff055b19cd` | 252 | 1200 | 209 | 0.000500 | 0.999700 | 0.897600 | kill | no_oos_edge |

Gate output:

- A folds: {'folds': [0.128, 2.199, 3.082], 'consistent': True}
- A regime: fragile=False n_partitions=9
- A bootstrap: edge_ci=[0.00054, 0.00148] includes_zero=False
- A permutation: p_value=0.0005
- A walk_forward: consistent=True (per-window Sharpe magnitudes are an artifact of the synthetic walk-forward frame, not R's full-series Sharpe; the gate only checks consistency)
- A holdout: {'holdout_sharpe': 0.798, 'holdout_psr': 0.9221, 'nbars': 800}
- A p8 rationale: costs/capacity are MODELED placeholders — capped at watch until measured (E2)
- B folds: {'folds': [0.128, 2.199, 3.082], 'consistent': True}
- B regime: fragile=False n_partitions=9
- B bootstrap: edge_ci=[0.00054, 0.00148] includes_zero=False
- B permutation: p_value=0.0005
- B walk_forward: consistent=True (per-window Sharpe magnitudes are an artifact of the synthetic walk-forward frame, not R's full-series Sharpe; the gate only checks consistency)
- B holdout: {'holdout_sharpe': 0.798, 'holdout_psr': 0.9221, 'nbars': 800}
- B p8 rationale: OOS score 0.898 (<0.9), edge_t 3.43

Why the verdict changes:

1. A normal backtester sees only `R`; Penrose also sees the search lineage that produced `R`.
2. P7 computes PSR/DSR on the OOS slice, not the full series, and adds regime partitions to the ledger trial count.
3. The returns, dates, bars/year, costs, robustness inputs, and holdout are unchanged. Process B's 200-trial ledger raises the deflation denominator and cross-trial Sharpe variance, dropping only B below the P8 DSR kill threshold.
