"""Mechanical competing-explanation lenses.

These refine the concept record only. They do not mutate or replace the P8 headline verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline.robustness import regime_split


def visible_inputs(net, *, market=None, momentum=None, simpler_net=None,
                   visible_frac: float = 0.80) -> dict:
    """Trim every explanation input to the Referee-visible IS+OOS window."""
    net = pd.Series(net)
    visible = net.iloc[:int(len(net) * visible_frac)]
    def align(value):
        if value is None:
            return None
        try:
            return pd.Series(value).reindex(visible.index)
        except Exception:
            return pd.Series(value).iloc[:len(visible)]
    return {"net": visible, "market": align(market), "momentum": align(momentum),
            "simpler_net": align(simpler_net)}


def exposure_decomposition(net, market=None, momentum=None) -> dict:
    y = pd.Series(net, dtype=float).dropna()
    factors = []
    names = []
    for name, factor in (("market", market), ("momentum", momentum)):
        if factor is not None:
            factors.append(pd.Series(factor, dtype=float).reindex(y.index))
            names.append(name)
    if len(y) < 10 or not factors:
        return {"applicable": False, "reason": "need >=10 aligned returns and one factor"}
    frame = pd.concat([y.rename("y"), *[x.rename(n) for x, n in zip(factors, names)]], axis=1).dropna()
    if len(frame) < 10:
        return {"applicable": False, "reason": "too few aligned observations"}
    x = np.column_stack([np.ones(len(frame)), *[frame[n].to_numpy() for n in names]])
    coef, *_ = np.linalg.lstsq(x, frame["y"].to_numpy(), rcond=None)
    total_mean = float(frame["y"].mean())
    # With an intercept, OLS residuals sum to zero by construction. The intercept is the
    # factor-adjusted mean return ("alpha") and is the quantity this lens must test.
    adjusted_mean = float(coef[0])
    share = adjusted_mean / total_mean if total_mean > 1e-12 else 0.0
    return {"applicable": True, "intercept": float(coef[0]),
            "betas": {n: float(v) for n, v in zip(names, coef[1:])},
            "factor_adjusted_mean": adjusted_mean, "factor_adjusted_share": share,
            "survives": bool(adjusted_mean > 0 and share >= 0.5)}


def crisis_leave_one_out(net, crisis_windows: list[tuple[str, str]] | None = None) -> dict:
    s = pd.Series(net).dropna()
    if not isinstance(s.index, pd.DatetimeIndex) or not crisis_windows:
        return {"applicable": False}
    base = float(s.mean())
    rows = []
    for start, end in crisis_windows:
        kept = s[~((s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end)))]
        rows.append({"window": [start, end], "mean_without": float(kept.mean()) if len(kept) else None})
    dominated = base > 0 and any(x["mean_without"] is not None and x["mean_without"] <= 0 for x in rows)
    return {"applicable": True, "base_mean": base, "windows": rows, "dominated": dominated}


def analyze(net, bars_per_year: float, *, market=None, momentum=None,
            crisis_windows=None, simpler_net=None) -> list[dict]:
    out = []
    exp = exposure_decomposition(net, market, momentum)
    out.append({"explanation": "returns are explained by market or momentum exposure",
                "test": exp,
                "verdict": ("untested" if not exp.get("applicable")
                            else "rejected" if exp.get("survives") else "survives")})
    reg = regime_split(pd.Series(net), bars_per_year)
    out.append({"explanation": "performance is concentrated in a single regime",
                "test": reg, "verdict": "survives" if reg.get("fragile") else "rejected"})
    crisis = crisis_leave_one_out(net, crisis_windows)
    out.append({"explanation": "a crisis window dominates the observation",
                "test": crisis,
                "verdict": ("untested" if not crisis.get("applicable")
                            else "survives" if crisis.get("dominated") else "rejected")})
    if simpler_net is not None:
        a, b = float(pd.Series(net).mean()), float(pd.Series(simpler_net).mean())
        out.append({"explanation": "a simpler delta-neutral or carry variant captures the result",
                    "test": {"submitted_mean": a, "simpler_mean": b},
                    "verdict": "survives" if b >= a else "rejected"})
    return out
