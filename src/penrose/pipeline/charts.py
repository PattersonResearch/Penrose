"""Render a backtest result chart (matplotlib) for the Analysis Reports page.

When a claim actually backtests (a module returned a net-return series), we draw the
strategy's cumulative-return equity curve plus a metrics caption, and save a PNG the
dashboard serves. Best-effort: any failure returns None and the pipeline carries on —
a missing chart never blocks a run.
"""
from __future__ import annotations

from pathlib import Path

from .. import config

# dark palette to match the dashboard
_BG = "#0a0e14"
_CARD = "#0f1620"
_GREEN = "#34d399"
_RED = "#f87171"
_AMBER = "#fbbf24"
_TX = "#c9d4e0"
_MUTED = "#6b7a8d"


def _verdict_color(verdict: str) -> str:
    return {"research-supported": _GREEN, "supported": _GREEN,
            "watch": _AMBER, "needs_data": "#22d3ee",
            "pending_module": _MUTED}.get(verdict, _RED)


def render_backtest_chart(claim_id: str, title: str, net, metrics: dict,
                          verdict: str) -> str | None:
    """Draw the equity curve for a backtested claim. Returns the PNG filename
    (relative to reports/charts/) or None if it couldn't be drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        s = pd.Series(net).dropna()
        if len(s) < 2:
            return None
        equity = (1.0 + s.astype(float)).cumprod()
        x = equity.index if isinstance(equity.index, pd.DatetimeIndex) else range(len(equity))

        out_dir = config.REPORTS / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{claim_id}.png"
        path = out_dir / fname

        vcol = _verdict_color(verdict)
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_CARD)
        ax.plot(x, equity.values, color=vcol, linewidth=1.8)
        ax.axhline(1.0, color=_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.fill_between(range(len(equity)) if not isinstance(equity.index, pd.DatetimeIndex) else x,
                        1.0, equity.values, color=vcol, alpha=0.08)

        for spine in ax.spines.values():
            spine.set_color("#1e2a38")
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.set_ylabel("cumulative return (×)", color=_MUTED, fontsize=9)
        ttl = (title or claim_id)[:70]
        ax.set_title(ttl, color=_TX, fontsize=10, loc="left", pad=10)

        # metrics caption
        def f(v, d=2):
            try:
                return f"{float(v):.{d}f}"
            except Exception:  # noqa: BLE001
                return "—"
        tf = metrics.get("three_fold")
        tf_s = "[" + ", ".join(f(z) for z in tf) + "]" if isinstance(tf, (list, tuple)) and tf else "—"
        cap = (f"verdict {verdict}   ·   DSR {f(metrics.get('dsr'))}   ·   "
               f"OOS Sharpe {f(metrics.get('oos_sharpe'))}   ·   "
               f"n={metrics.get('n_trades','—')}   ·   3-fold {tf_s}")
        fig.text(0.012, 0.03, cap, color=_MUTED, fontsize=8, family="monospace")

        # per-regime Sharpe line (weekday/weekend) — the kill-lens, made visible
        reg = (metrics.get("regime") or {})
        wk = (reg.get("schemes") or {}).get("weekend") or {}
        rect_bottom = 0.10
        if wk:
            seg = "   ·   ".join(f"{k} Sh {f(v.get('sharpe'))} (n{v.get('n')})"
                                 for k, v in wk.items())
            flag = "  ⚠ regime-fragile" if reg.get("fragile") else ""
            fig.text(0.012, 0.0, "regime:  " + seg + flag,
                     color=(_RED if reg.get("fragile") else _MUTED), fontsize=8, family="monospace")
            rect_bottom = 0.13

        # fidelity line — does the module faithfully implement the claim?
        fd = metrics.get("fidelity") or {}
        if fd and "faithful" in fd:
            suspect = (not fd.get("faithful")) and fd.get("confidence", 0) >= 0.6
            txt = ("fidelity:  ⚠ SUSPECT (conf %.2f) %s" % (fd.get("confidence", 0), fd.get("note", ""))
                   if suspect else "fidelity:  faithful (conf %.2f)" % fd.get("confidence", 0))
            fig.text(0.012, -0.04 if wk else 0.0, txt[:110],
                     color=(_RED if suspect else _GREEN), fontsize=8, family="monospace")
            rect_bottom += 0.03

        fig.tight_layout(rect=(0, rect_bottom, 1, 1))
        fig.savefig(path, facecolor=_BG)
        plt.close(fig)
        return fname
    except Exception:  # noqa: BLE001 — a chart is never worth breaking a run
        return None
