"""Shared calibration helpers for seeded no-edge controls."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ["PENROSE_HOLDOUT_LOCK"] = os.path.join(
    tempfile.gettempdir(), "penrose_calib_holdout.lock")
if os.path.exists(os.environ["PENROSE_HOLDOUT_LOCK"]):
    os.unlink(os.environ["PENROSE_HOLDOUT_LOCK"])

from penrose import config  # noqa: E402
from penrose.brain import Claim  # noqa: E402
from penrose.data import client  # noqa: E402
from penrose.data.contract import Series  # noqa: E402
from penrose.pipeline import p7_backtest, stages  # noqa: E402

BARS_PER_YEAR = 365.0


def real_btc_returns() -> pd.Series:
    """Return BTC-like daily log returns, preferring real BTC and tagging any fallback."""
    b = client.fetch_bundle()
    for k in ("btc_price", "btc_spot_daily"):
        v = b.series.get(k)
        if isinstance(v, Series) and v.data is not None and len(v.data) > 300:
            r = np.log(v.data.astype(float)).diff().dropna()
            if len(r) > 300 and float(r.std(ddof=1)) > 0:
                r.name = f"ret[{v.provenance}]"
                return r
    # Offline/dev fallback: keep calibration commands executable in restricted environments while
    # making the provenance impossible to confuse with the preferred real-return run.
    rng = np.random.default_rng(20260624)
    n = 1186
    vol = np.zeros(n)
    state = 0.012
    vals = np.zeros(n)
    for i in range(n):
        state = max(0.004, 0.985 * state + 0.015 * abs(rng.normal(0.012, 0.006)))
        vol[i] = state
        vals[i] = rng.normal(0.0, state)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    r = pd.Series(vals, index=idx, name="ret[synthetic-calibration-fallback]")
    return r


def ar1_signal(n: int, rng: np.random.Generator, phi: float = 0.9) -> np.ndarray:
    """Persistent noise, independent of returns."""
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    return s


def calib_claim(claim_id: str, statement: str, strategy_class: str) -> Claim:
    return Claim(claim_id=claim_id, statement=statement, mechanism="", scope="",
                 horizon="", source_id="calib", source_span="", claimed_metric_quote="",
                 applicable_strategy_class=strategy_class)


def verdict_for_signal(sig: np.ndarray, ret: pd.Series, cost_frac: float, *,
                       name: str = "placebo",
                       family: str = "placebo::crypto",
                       claim_id: str = "placebo",
                       statement: str = "no-edge control",
                       strategy_class: str = "placebo") -> tuple[str, dict]:
    """Trade sign(sig_t), realize ret_{t+1}. Returns (verdict, backtest)."""
    if len(sig) != len(ret):
        raise ValueError("signal and return series must have the same length")
    if len(ret) < 40:
        raise ValueError("calibration series is too short for an OOS verdict")
    pos = np.sign(sig[:-1])
    pay = ret.values[1:]
    idx = ret.index[1:]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0
    net = pos * pay - turn.astype(float) * cost_frac
    net_s = pd.Series(net, index=idx)
    bt = p7_backtest.run_backtest(
        name, net_s, pd.Series(np.abs(pos), index=idx), BARS_PER_YEAR,
        payoff=pd.Series(pay, index=idx), position_signed=pd.Series(pos, index=idx),
        cost_frac=cost_frac, family=family, log=False)
    claim = calib_claim(claim_id, statement, strategy_class)
    dec = stages.p8_verdict(claim, bt, {}, synthetic=False)
    if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
        holdout = p7_backtest.final_holdout_eval(name, net_s, BARS_PER_YEAR, force=True)
        dec = stages.p8_verdict(claim, bt, holdout, synthetic=False)
    return dec.verdict, bt
