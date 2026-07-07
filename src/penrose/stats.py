"""Instrument-agnostic risk statistics — the ungameable metrics penrose scores verdicts with.

Self-contained statistics so penrose
is SELF-CONTAINED: a fresh `pip install penrose` no longer needs the sibling repo on the path
(that coupling was the sole blocker found by the clone-and-run test). These are the exact
functions penrose uses — copied VERBATIM, not reimplemented, so verdicts are bit-identical
(`make eval` is the parity guard: it must stay green after vendoring).

Only the five functions penrose actually consumes are here:
  sharpe, probabilistic_sharpe, deflated_sharpe (the multiple-testing DSR), pm_fee_frac
  (Polymarket dynamic fee curve), and _capacity_usd (linear-impact capacity floor).

The DSR is the whole point — Bailey & López de Prado's multiple-testing-aware deflated Sharpe.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # W4/6h + audit D-3: importable on a scipy-less venv; fail clearly at USE.
    from scipy.stats import norm as _scipy_norm
except (ModuleNotFoundError, ImportError):  # pragma: no cover — exercised via subprocess test
    _scipy_norm = None


class _NormProxy:
    """Defers scipy to first use so `import penrose.stats` (and everything that
    transitively pulls it, e.g. penrose.pipeline.run) imports on a lean venv. Any
    actual statistical call without scipy raises one clear, actionable message."""

    def __getattr__(self, attr: str):
        if _scipy_norm is None:
            raise RuntimeError(
                "pip install scipy required for Sharpe/DSR statistics "
                f"(scipy.stats.norm.{attr} was called but scipy is not installed)"
            )
        return getattr(_scipy_norm, attr)


norm = _NormProxy() if _scipy_norm is None else _scipy_norm

EULER = 0.5772156649


def sharpe(net: np.ndarray, bars_per_year: float) -> float:
    if len(net) < 20 or net.std() == 0:
        return float("nan")
    return net.mean() / net.std(ddof=1) * math.sqrt(bars_per_year)


def probabilistic_sharpe(net: np.ndarray, sr_benchmark: float = 0.0) -> float:
    n = len(net)
    if n < 20 or net.std() == 0:
        return float("nan")
    sr = net.mean() / net.std(ddof=1)
    s = pd.Series(net)
    g3, g4 = s.skew(), s.kurt() + 3.0
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe(net: np.ndarray, n_trials: int, sr_variance: float) -> float:
    """Deflated Sharpe Ratio — the multiple-testing correction."""
    if n_trials < 2 or sr_variance <= 0:
        return probabilistic_sharpe(net)
    e1 = norm.ppf(1 - 1.0 / n_trials)
    e2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    sr_star = math.sqrt(sr_variance) * ((1 - EULER) * e1 + EULER * e2)
    return probabilistic_sharpe(net, sr_benchmark=sr_star)


def pm_fee_frac(p, fee_rate: float, C: float = 1.0):
    """Polymarket dynamic fee as a fraction of size: C*feeRate*p*(1-p).
    ~0 near 0c/100c (cheap tails), maximal at 50c. Symmetric in p<->1-p."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return C * fee_rate * p * (1.0 - p)


def _capacity_usd(positions: pd.DataFrame, ann_ret: float, bpy: float,
                  impact_coef_bps_per_1m: float):
    """Notional $ at which a linear market-impact model erases the net edge.

    Model (explicit ASSUMPTION): trading $N incurs slippage of `impact_coef` bps per $1M
    traded, on TOP of the flat cost already charged. Beyond N* the extra slippage alone eats
    the whole net return. A capacity floor, not a promise of fills.
        N* = ann_ret / (annual_turnover_fraction * impact_per_$).
    """
    if ann_ret is None or ann_ret <= 0 or impact_coef_bps_per_1m <= 0:
        return None
    turn_bar = positions.diff().abs().sum(axis=1).dropna().mean()   # fraction of book/bar
    if not turn_bar or turn_bar <= 0:
        return None
    turn_ann = turn_bar * bpy
    impact_per_dollar = (impact_coef_bps_per_1m / 1e4) / 1e6        # frac cost per $ traded
    cap = ann_ret / (turn_ann * impact_per_dollar)
    return int(round(cap, -3)) if math.isfinite(cap) else None
