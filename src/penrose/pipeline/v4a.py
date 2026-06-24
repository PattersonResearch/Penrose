"""Calibration tolerance.

V4a asks: does a paper->module-generated backtest reproduce the operator's
working-code ground truth "within tolerance". The spec never
defined "within tolerance", which made the gate unfalsifiable. Now that P7 emits
a bootstrap CI, we can pin it to the data instead of a magic constant:

  1. Same verdict DIRECTION   — the OOS edge has the same sign (a strategy that
     flips long/short between reference and candidate is not "calibrated").
  2. Edge within the REFERENCE bootstrap CI half-width — the candidate's mean OOS
     edge sits inside the band the reference's own resampling says is noise. This
     is "within the harness's own bootstrap CI half-width" made concrete.
  3. Sharpe within the reference Sharpe-CI half-width (same idea, Sharpe scale).
  4. Capacity within one order of magnitude.

Pass = all four. The point is that the tolerance scales with how noisy the
reference itself is: a high-variance reference grants a wider band, a tight one
demands a closer match. No hand-tuned epsilon.
"""
from __future__ import annotations

import math

from .. import config


def _halfwidth(ci) -> float | None:
    if not ci or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return None
    return abs(float(ci[1]) - float(ci[0])) / 2.0


def within_tolerance(reference: dict, candidate: dict, tol: dict | None = None) -> dict:
    """Compare a candidate backtest to the reference (ground-truth) backtest.

    Both are P7 `run_backtest` outputs (must include `bootstrap`). Returns a dict
    with an overall `pass` and a per-dimension breakdown so the gap is legible.
    """
    tol = tol or config.V4A_TOLERANCE
    ref_boot = reference.get("bootstrap") or {}
    checks: dict = {}

    # 1. verdict direction (sign of OOS edge)
    r_edge = reference.get("avg_net_edge")
    c_edge = candidate.get("avg_net_edge")
    if r_edge is None or c_edge is None:
        checks["direction"] = {"pass": None, "note": "missing avg_net_edge"}
    else:
        same = (r_edge >= 0) == (c_edge >= 0)
        checks["direction"] = {"pass": bool(same), "reference": r_edge, "candidate": c_edge}

    # 2. edge within the reference bootstrap CI half-width
    hw_edge = _halfwidth(ref_boot.get("edge_ci"))
    if hw_edge is None or r_edge is None or c_edge is None:
        checks["edge"] = {"pass": None, "note": "no reference bootstrap edge CI",
                          "tolerance_halfwidth": hw_edge}
    else:
        diff = abs(c_edge - r_edge)
        checks["edge"] = {"pass": bool(diff <= hw_edge), "abs_diff": round(diff, 6),
                          "tolerance_halfwidth": round(hw_edge, 6)}

    # 3. Sharpe within the reference Sharpe-CI half-width
    hw_sh = _halfwidth(ref_boot.get("sharpe_ci"))
    r_sh, c_sh = reference.get("oos_sharpe"), candidate.get("oos_sharpe")
    if hw_sh is None or r_sh is None or c_sh is None:
        checks["sharpe"] = {"pass": None, "note": "no reference bootstrap Sharpe CI",
                            "tolerance_halfwidth": hw_sh}
    else:
        diff = abs(c_sh - r_sh)
        checks["sharpe"] = {"pass": bool(diff <= hw_sh), "abs_diff": round(diff, 3),
                            "tolerance_halfwidth": round(hw_sh, 3)}

    # 4. capacity within N orders of magnitude
    r_cap, c_cap = reference.get("capacity_usd"), candidate.get("capacity_usd")
    if not r_cap or not c_cap or r_cap <= 0 or c_cap <= 0:
        checks["capacity"] = {"pass": None, "note": "capacity not estimable for both"}
    else:
        orders = abs(math.log10(c_cap) - math.log10(r_cap))
        checks["capacity"] = {"pass": bool(orders <= tol["capacity_orders_of_magnitude"]),
                              "orders_of_magnitude_apart": round(orders, 2)}

    # overall: every dimension that COULD be evaluated must pass; require at least
    # direction + edge to be evaluable (else the calibration is inconclusive).
    evaluable = {k: v for k, v in checks.items() if v.get("pass") is not None}
    decisive = checks["direction"].get("pass") is not None and checks["edge"].get("pass") is not None
    passed = decisive and all(v["pass"] for v in evaluable.values())
    return {
        "pass": bool(passed),
        "conclusive": bool(decisive),
        "checks": checks,
        "summary": ("calibrated within tolerance" if passed else
                    "inconclusive (insufficient reference stats)" if not decisive else
                    "outside tolerance"),
    }
