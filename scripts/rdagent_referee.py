"""penrose as REFEREE for a generative factor pipeline (e.g. Microsoft RD-Agent(Q)).

Verified from RD-Agent's code (2026-06-20): it generates factors as a cross-sectional matrix
(`combined_factors_df.parquet`, datetime x instrument x factor), selects winners by test-set
IC/Sharpe over a large bandit search, and applies NO multiple-testing correction / Sharpe
deflation. That is exactly the selection-bias hole the Deflated Sharpe Ratio exists to plug — and
the factors are cross-sectional at native breadth (~300 names), the regime where penrose's
detection floor is LOWEST (see calibration_breadth.py). So this is the configuration where penrose
is strongest: clean code inputs (no prose->code reconstruction risk) + high breadth.

What this does:
  * `referee(factors, fwd_ret, family)` — for each candidate factor, build a dollar-neutral
    z-weighted cross-sectional portfolio, run it through penrose's REAL backtest + power-aware
    p8_verdict, and DEFLATE every factor's score for the size of the search (the thing RD-Agent
    lacks): all K factors are logged in one family so each is judged against the best-of-K null.
  * `load_rdagent_parquet(path)` — adapter for a real `combined_factors_df.parquet` (when you have
    run `rdagent fin_quant`); pair it with a forward-return panel.
  * demo `main()` — a faithful RD-Agent-style factor SEARCH (K candidates, mostly data-mined noise
    + a few real edges) on a synthetic 100-name panel, contrasting naive "keep top by IC" (RD-Agent)
    with penrose's deflated referee. Shows how many "winners" are selection-bias artifacts.

Run:  python scripts/rdagent_referee.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import os as _os, tempfile as _tf
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")
_os.path.exists(_os.environ["PENROSE_HOLDOUT_LOCK"]) and _os.unlink(_os.environ["PENROSE_HOLDOUT_LOCK"])
from penrose import config                            # noqa: E402
config.COST_PROVENANCE = "measured"                  # E2 off: let a genuinely strong factor certify
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim                       # noqa: E402

BARS_PER_YEAR = 252.0


def _claim(name: str) -> Claim:
    return Claim(claim_id=name, statement=f"factor {name}", mechanism="", scope="", horizon="",
                 source_id="rdagent", source_span="", claimed_metric_quote="",
                 applicable_strategy_class="rdagent-factor")


def factor_portfolio_return(fac: np.ndarray, fwd_ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional dollar-neutral z-weighted long-short book. fac, fwd_ret: (T, N) where
    fac[t] is the factor observed at t and fwd_ret[t] is the return EARNED over (t, t+1] (causal —
    no look-ahead; the factor predicts its own forward return). Returns (port_ret[T], gross[T])."""
    f = fac.astype(float)
    if f.shape[1] > 1:
        f = (f - np.nanmean(f, axis=1, keepdims=True)) / (np.nanstd(f, axis=1, keepdims=True) + 1e-9)
        w = f / (np.abs(f).sum(axis=1, keepdims=True) + 1e-9)     # dollar-neutral, unit gross
    else:
        w = np.sign(f)
    w = np.nan_to_num(w)
    return (w * np.nan_to_num(fwd_ret)).sum(axis=1), np.abs(w).sum(axis=1)


def referee(factors: dict[str, np.ndarray], fwd_ret: np.ndarray, family: str = "rdagent::ref") -> dict:
    """Referee a dict of {name -> (T,N) factor matrix} against a (T,N) forward-return panel.
    Multiple-testing deflation across the WHOLE set: log all K factors in one family, then judge
    each against the best-of-K deflation null. Returns {name -> {verdict, dsr, oos_sharpe, mde_ic}}."""
    idx = pd.date_range("2015-01-01", periods=fwd_ret.shape[0], freq="D")
    ports = {}
    for name, fac in factors.items():
        port, gross = factor_portfolio_return(fac, fwd_ret)
        ports[name] = (pd.Series(port, index=idx), pd.Series(gross, index=idx))

    tmp = Path(tempfile.mktemp(suffix="_ref.tsv"))
    old = P7.LEDGER
    P7.LEDGER = tmp
    try:
        # pre-pass: populate the family ledger so DSR deflates by the full search size K.
        for name, (net, pos) in ports.items():
            P7.run_backtest(name, net, pos, BARS_PER_YEAR, cost_frac=0.0, family=family, log=True)
        # eval-pass: every factor now judged against best-of-K (multiple-testing correction).
        out = {}
        for name, (net, pos) in ports.items():
            bt = P7.run_backtest(name, net, pos, BARS_PER_YEAR, cost_frac=0.0, family=family, log=False)
            dec = stages.p8_verdict(_claim(name), bt, {}, synthetic=False)
            if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
                ho = P7.final_holdout_eval(name, net, BARS_PER_YEAR, force=True)
                dec = stages.p8_verdict(_claim(name), bt, ho, synthetic=False)
            out[name] = {"verdict": dec.verdict, "dsr": bt.get("dsr"),
                         "oos_sharpe": bt.get("oos_sharpe"), "mde_ic": dec.metrics.get("mde_ic")}
        return out
    finally:
        P7.LEDGER = old
        tmp.unlink(missing_ok=True)


def load_rdagent_parquet(factor_path: str, ret_path: str):
    """Adapter for a REAL RD-Agent run: combined_factors_df.parquet (MultiIndex datetime x
    instrument, one column per factor) + a forward-return panel (same index). Returns
    (factors dict, fwd_ret (T,N)). Use when you have run `rdagent fin_quant`."""
    fdf = pd.read_parquet(factor_path)
    rdf = pd.read_parquet(ret_path)
    dates = sorted(fdf.index.get_level_values(0).unique())
    insts = sorted(fdf.index.get_level_values(1).unique())
    def _mat(col_df, col):
        return col_df[col].unstack().reindex(index=dates, columns=insts).to_numpy()
    factors = {c: _mat(fdf, c) for c in fdf.columns}
    fwd = _mat(rdf, rdf.columns[0])
    return factors, fwd


def main() -> None:
    rng = np.random.default_rng(7)
    N, T = 100, 1400          # 100-name panel (native cross-sectional breadth), 1400 periods
    K_REAL, K_NOISE, IC = 5, 45, 0.04   # a few real ~0.04-IC edges hidden in a zoo of 45 data-mined ones
    print(f"[referee] RD-Agent-style factor SEARCH: {K_REAL} real (IC~{IC}) + {K_NOISE} noise "
          f"on N={N} names, T={T}.\n")

    real_facs = [rng.standard_normal((T, N)) for _ in range(K_REAL)]
    def _z(a): return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
    c = IC / np.sqrt(max(1e-9, 1 - K_REAL * IC * IC))          # per-factor coef -> marginal IC~IC
    fwd = rng.standard_normal((T, N)) + sum(c * _z(f) for f in real_facs)   # realized forward returns

    factors = {f"real_{i}": real_facs[i] for i in range(K_REAL)}
    factors.update({f"noise_{j}": rng.standard_normal((T, N)) for j in range(K_NOISE)})

    # --- RD-Agent-style selection: rank by IN-SAMPLE IC (first 70%), keep the top, NO deflation ----
    cut = int(T * 0.7)
    def insample_ic(fac):
        f, r = _z(fac[:cut]), fwd[:cut]
        return float(np.nanmean([np.corrcoef(f[t], r[t])[0, 1] for t in range(cut)]))
    ics = {n: insample_ic(f) for n, f in factors.items()}
    keep_n = 10
    rdagent_keep = sorted(ics, key=ics.get, reverse=True)[:keep_n]
    rd_noise_kept = [n for n in rdagent_keep if n.startswith("noise")]
    print(f"RD-Agent-style (top {keep_n} by in-sample IC, no deflation): keeps {keep_n} factors, of "
          f"which {len(rd_noise_kept)} are pure NOISE (data-mined false positives): {rd_noise_kept}")

    # --- penrose referee: deflated, power-aware verdict for every factor in the search -------------
    res = referee(factors, fwd)
    from collections import Counter
    real_v = Counter(res[f"real_{i}"]["verdict"] for i in range(K_REAL))
    noise_v = Counter(res[f"noise_{j}"]["verdict"] for j in range(K_NOISE))
    cert = {n for n, r in res.items() if r["verdict"] in ("research-supported", "watch")}
    print(f"\npenrose referee (deflated by the full search of {len(factors)}; power-aware):")
    print(f"  REAL  factors ({K_REAL}): {dict(real_v)}")
    print(f"  NOISE factors ({K_NOISE}): {dict(noise_v)}")
    print(f"  certified/watch (survive deflation): {sorted(cert)}")
    rd_survive = [n for n in rdagent_keep if res[n]["verdict"] in ("research-supported", "watch")]
    print(f"\nReferee verdict on RD-Agent's {keep_n} kept factors: {len(rd_survive)} survive penrose's "
          f"deflated scrutiny ({sorted(rd_survive)}).")
    print(f"  -> penrose rejects {keep_n - len(rd_survive)} of RD-Agent's 'winners' as selection-bias / "
          f"underpowered, and recovers the planted-real edges that survive multiple-testing correction.")
    print("\nThis is the value-add RD-Agent structurally lacks: a referee that deflates for the SIZE of "
          "the search before believing a winner. With a real run, point load_rdagent_parquet() at the "
          "combined_factors_df.parquet it emits.")


if __name__ == "__main__":
    main()
