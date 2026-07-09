import numpy as np
import pandas as pd

from penrose.pipeline import p7_backtest as P7, stages


class _Claim:
    claim_id = "imp"
    statement = ""
    mechanism = ""


def _run(net):
    pos = pd.Series(1.0, index=net.index)
    bt = P7.run_backtest("imp", net, pos, 252.0, log=False)
    return bt, stages.p8_verdict(_Claim(), bt, {}, synthetic=False)


def test_implausible_high_sharpe_routes_to_needs_review():
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    # near-constant-sign tiny-vol series -> implausibly high annualized Sharpe
    net = pd.Series(np.abs(rng.normal(0.001, 0.0002, 400)), index=idx)
    bt, dec = _run(net)
    assert bt["implausible"]["triggered"] is True
    assert dec.verdict == "needs_review"
    assert dec.verdict not in ("watch", "research-supported")


def test_realistic_edge_not_flagged():
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    # daily Sharpe ~0.15 -> annualized ~2.4, a plausible strong edge
    net = pd.Series(rng.normal(0.0012, 0.008, 400), index=idx)
    bt, dec = _run(net)
    assert bt["implausible"]["triggered"] is False
    assert dec.verdict != "needs_review"


def test_funding_carry_shape_flagged():
    """The motivating case: sign(funding[t-1])*funding[t] on a persistent-sign funding series is a
    near-zero-vol modeling artifact (Sharpe ~18+), not a real edge -> must be flagged."""
    idx = pd.date_range("2021-01-01", periods=1000, freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    funding = pd.Series(np.abs(rng.normal(0.0004, 0.0001, 1000)), index=idx)
    net = (np.sign(funding.shift(1).fillna(0.0)) * funding).rename("net")
    bt, dec = _run(net)
    assert bt["implausible"]["triggered"] is True
    assert dec.verdict == "needs_review"
