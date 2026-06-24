"""penrose eval suite — planted strategies with KNOWN-correct verdicts.

Ground truth for a falsification engine. We feed strategies whose answer we already
know and assert penrose's verdict. Two jobs:
  1. Runtime-loop proof: penrose actually DISCRIMINATES signal from overfit (not just
     internally consistent — correct on cases where we know the truth).
  2. Dev-loop regression net: every bug we find becomes a new case here, so it can
     never silently come back.

Deterministic: fixed seeds, and the DSR trial count is pinned (no dependence on the
global ledger) so a verdict is a function of the strategy alone.

Run:  make eval     (or: PYTHONPATH=src:. python scripts/eval_suite.py)
Exit code 1 on any FAIL — loop-friendly, same contract as `make verify`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.pipeline import p7_backtest as P7   # noqa: E402 (self-adds harness path)
from penrose.pipeline import stages              # noqa: E402
from penrose.brain import Claim                  # noqa: E402

# --- isolation: pin the trial count so a verdict depends only on the strategy ------- #
_REAL_TRIAL_STATS = P7._trial_stats          # keep the real one for the A-004/C1 tests
P7._trial_stats = lambda *a, **k: (1, 0.0)   # pin (now takes family, strategy)

BPY = 365.0 / 5.0           # bars/year for a 5-day hold (realistic, not daily)


def _claim() -> Claim:
    return Claim(claim_id="eval", statement="eval planted strategy", mechanism="",
                 scope="", horizon="", source_id="eval", source_span="",
                 claimed_metric_quote="", applicable_strategy_class="eval")


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n, freq="D")


def _verdict(net: pd.Series):
    bt = P7.run_backtest("eval", net, pd.Series(1.0, index=net.index), BPY, log=False)
    dec = stages.p8_verdict(_claim(), bt, {}, synthetic=False)
    return dec, bt


# --- planted strategies (deterministic given the seed) ------------------------------ #
def build_overfit(rng):
    """Strong in-sample, dead out-of-sample -> the classic overfit. Must KILL."""
    n = 200
    v = np.empty(n)
    cut = int(n * 0.5)                       # the IS window
    v[:cut] = rng.normal(0.015, 0.006, cut)  # strong IS edge
    v[cut:] = rng.normal(0.0, 0.012, n - cut)  # dead OOS
    return pd.Series(v, index=_idx(n))


def build_robust(rng):
    """Persistent, uniform edge across time AND regimes. Must SURVIVE (watch)."""
    n = 220
    v = rng.normal(0.012, 0.008, n)
    return pd.Series(v, index=_idx(n))


def build_regime_fragile(rng):
    """Edge lives almost entirely on weekends; weekdays barely positive. Strong enough
    to pass 3-fold + OOS (reach a survivor verdict), then the regime lens must KILL it."""
    n = 240
    idx = _idx(n)
    v = rng.normal(0.002, 0.004, n)          # weekdays: barely positive
    wknd = idx.dayofweek >= 5
    v[wknd] = rng.normal(0.060, 0.010, int(wknd.sum()))  # weekends: big edge
    return pd.Series(v, index=idx)


def build_random(rng):
    """Zero-mean noise — no edge at all. Must KILL."""
    n = 200
    return pd.Series(rng.normal(0.0, 0.011, n), index=_idx(n))


def build_thin(rng):
    """Too few trades to trust any verdict. Must be insufficient_data."""
    n = 20
    return pd.Series(rng.normal(0.012, 0.01, n), index=_idx(n))


# (name, builder, expected verdict(s), expected kill_reason substring or None)
CASES = [
    ("overfit (IS-only)",        build_overfit,        ("kill",),                          None),
    ("robust edge",              build_robust,         ("watch", "research-supported"),     None),
    ("regime-fragile (weekend)", build_regime_fragile, ("kill",),                           "regime_fragile"),
    ("random noise",             build_random,         ("kill",),                           None),
    ("thin sample",              build_thin,           ("insufficient_data",),              None),
]


def invariants() -> list[tuple[str, bool]]:
    """Component-level regression guards — each locks in a bug we already fixed so it can
    never silently come back (the DISTILL station of the dev loop, made permanent)."""
    from penrose.data.contract import DataBundle, Series
    from penrose.pipeline import robustness as R
    out = []

    # --- alias normalization in bundle.get (guards the naming-drift fix) ------------- #
    ix = _idx(30)
    mk = lambda nm: Series(nm, pd.Series(np.arange(30, dtype=float), index=ix), "test", "lake")
    b = DataBundle(series={k: mk(k) for k in ("eth_spot_daily", "sol_spot_daily", "funding_eth")})
    out.append(("alias: 'price.eth_usd_spot_daily' -> eth_spot_daily", b.get("price.eth_usd_spot_daily") is not None))
    out.append(("alias: 'eth_funding' -> funding_eth",                 b.get("eth_funding") is not None))
    out.append(("alias: distinct entity stays missing (no false match)", b.get("crypto_market_cap_usd") is None))
    out.append(("alias: exact key still resolves",                      b.get("eth_spot_daily") is not None))

    # --- regime lens flags concentration, passes uniform (guards the kill-lens) ------ #
    rng = np.random.default_rng(7)
    n = 220
    iz = _idx(n)
    frag = pd.Series(rng.normal(0.0, 0.004, n), index=iz)
    wknd = iz.dayofweek >= 5
    frag[wknd] += rng.normal(0.05, 0.01, int(wknd.sum()))
    out.append(("regime: weekend-concentrated -> fragile", R.regime_split(frag, BPY).get("fragile") is True))
    unif = pd.Series(rng.normal(0.010, 0.008, n), index=iz)
    out.append(("regime: uniform edge -> not fragile",      R.regime_split(unif, BPY).get("fragile") is False))

    # --- fidelity refuter errors are inconclusive, never permission to trust a module -------- #
    from penrose.pipeline import fidelity
    f0 = fidelity.assess(_claim(), "")            # no code -> early inconclusive, no LLM call
    out.append(("fidelity: empty code is unverified (never faithful)",
                f0.get("faithful") is False and f0.get("verified") is False))

    # --- A-001: the verdict gates on DEFLATED Sharpe, not max(psr,dsr) ---------------- #
    # With AMPLE power (large n_oos so MDE-IC < the realistic floor), a low DSR is a genuine kill;
    # PSR(0.97) must not sneak it past the DSR(0.80) gate.
    bt = {"psr": 0.97, "dsr": 0.80, "edge_t": 3.0, "n_oos": 1100, "oos_sharpe": 1.5, "bars_per_year": 252,
          "three_fold": {"folds": [1, 1, 1], "consistent": True}, "capacity_usd": 1e6,
          "bootstrap": {}, "permutation": {}, "regime": {}}
    d = stages.p8_verdict(_claim(), bt, {}, False)
    out.append(("A-001: high PSR(0.97) + low DSR(0.80), ample power -> kill (gate uses DSR)", d.verdict == "kill"))

    # --- POWER-aware taxonomy: dead vs underpowered vs structural ---------------------- #
    # Same low DSR but a THIN sample -> `underpowered` (below detection floor), NOT a kill: a
    # non-structural null on data that can't resolve a realistic edge is not "proven dead".
    _bt_up = dict(bt); _bt_up["n_oos"] = 60
    _d_up = stages.p8_verdict(_claim(), _bt_up, {}, False)
    out.append(("POWER: low DSR + thin sample -> underpowered (not kill); MDE reported",
                _d_up.verdict == "underpowered" and _d_up.kill_reason == "below_detection_floor"
                and _d_up.metrics.get("mde_ic") and _d_up.metrics.get("power_sufficient") is False))
    # Structural kills are power-INDEPENDENT: a THIN sample with sign-unstable 3-fold still KILLS.
    _bt_st = {"psr": 0.5, "dsr": 0.5, "edge_t": 0.2, "n_oos": 60, "oos_sharpe": 0.3, "bars_per_year": 252,
              "three_fold": {"folds": [1, -1, 1], "consistent": False}, "capacity_usd": 1e6,
              "bootstrap": {}, "permutation": {}, "regime": {}}
    _d_st = stages.p8_verdict(_claim(), _bt_st, {}, False)
    out.append(("POWER: structural kill (3-fold unstable) stays kill even when thin (power-independent)",
                _d_st.verdict == "kill" and _d_st.kill_reason == "in_sample_only"))

    # --- A-002: the locked holdout tail must not influence the 3-fold/regime gates ---- #
    n = 250
    base = np.concatenate([np.full(80, 0.012), np.zeros(120)])   # seen[:200]: fold1 + flat -> inconsistent
    s_zero = pd.Series(np.concatenate([base, np.zeros(50)]), index=_idx(n))
    s_rich = pd.Series(np.concatenate([base, np.full(50, 0.08)]), index=_idx(n))  # strong holdout tail
    b_zero = P7.run_backtest("a002z", s_zero, pd.Series(1.0, index=s_zero.index), BPY, log=False)
    b_rich = P7.run_backtest("a002r", s_rich, pd.Series(1.0, index=s_rich.index), BPY, log=False)
    out.append(("A-002: holdout tail can't change the 3-fold gate",
                b_zero.get("three_fold") == b_rich.get("three_fold")))

    # --- A-004: the trial ledger dedups re-runs (a trial is a strategy, not a run) ---- #
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "_eval_ledger.tsv"
    pd.DataFrame({"strategy": ["a", "a", "a", "b", "b"],
                  "per_trade_sharpe": [1.0, 1.0, 1.0, 2.0, 2.0],
                  "dsr": [0] * 5, "n": [100] * 5}).to_csv(tmp, sep="\t", index=False)
    old = P7.LEDGER
    P7.LEDGER = tmp
    try:
        n_distinct, _ = _REAL_TRIAL_STATS()                       # no current strategy
        n_rerun, _ = _REAL_TRIAL_STATS(strategy="a")             # re-run of existing 'a'
        n_new, _ = _REAL_TRIAL_STATS(strategy="c")              # a genuinely new strategy
    finally:
        P7.LEDGER = old
    out.append(("A-004: 5 rows / 2 distinct strategies -> 2 trials (deduped)", n_distinct == 2))
    out.append(("B-016: re-running existing 'a' does NOT add a trial (no +1)", n_rerun == 2))
    out.append(("B-016: a new strategy 'c' counts once -> 3", n_new == 3))

    # ===================== Wave 2/3 audit-fix regression guards ===================== #
    import tempfile
    import pathlib
    from penrose.pipeline import impl_gen as ig
    from penrose.pipeline import robustness as R
    from penrose.data.contract import DataBundle, Series, _norm_key
    from penrose.pipeline.extract import span_in_text
    from penrose.llm import _repair_truncated_json

    # A-010: denylisted ops in generated code are rejected pre-exec; clean code passes
    out.append(("A-010: 'import os' in module code is rejected", ig._scan_code_text("import os\ndef run(): pass") is not None))
    out.append(("A-010: 'open(' rejected", ig._scan_code_text("x=open('/etc/passwd')") is not None))
    out.append(("A-010: clean numpy/pandas code passes scan", ig._scan_code_text("import pandas as pd\ndef run(b,c,f): pass") is None))
    # C-009: a harmless `from __future__ import annotations` must NOT cost a repair attempt
    out.append(("C-009: 'from __future__ import annotations' is allowed",
                ig._ast_import_violation("from __future__ import annotations\nimport os") is not None
                and ig._ast_import_violation("from __future__ import annotations\nimport numpy as np") is None))
    # A-012: look-ahead static guard
    out.append(("A-012: '.shift(-1)' (look-ahead) rejected", ig._scan_code_text("s.shift(-1)") is not None))
    out.append(("A-012: honest '.shift(1)' allowed", ig._scan_code_text("s.shift(1)") is None))

    # A-011/A-013/A-014: validator rejects degenerate/absurd modules, passes a good one
    _td = pathlib.Path(tempfile.mkdtemp())
    _H = "__strategy_class__='x'\nimport pandas as pd, numpy as np\n"
    def _mod(body):
        p = _td / "impl.py"; p.write_text(_H + body); return p
    _const = ("def run(b,c,f):\n idx=pd.date_range('2024-01-01',periods=14)\n"
              " return {'ok':True,'net':pd.Series(0.01,index=idx),'positions':pd.Series(1.0,index=idx),'bars_per_year':52,'n_trades':14}\n")
    ok_c, _ = ig._validate_module(_mod(_const), "m", None, None, 0.0)
    out.append(("A-011: near-constant net rejected", ok_c is False))
    _bpy = ("def run(b,c,f):\n idx=pd.date_range('2024-01-01',periods=60)\n"
            " return {'ok':True,'net':pd.Series(np.linspace(-0.02,0.03,60),index=idx),'positions':pd.Series(1.0,index=idx),'bars_per_year':100000,'n_trades':60}\n")
    ok_b, _ = ig._validate_module(_mod(_bpy), "m", None, None, 0.0)
    out.append(("A-013: absurd bars_per_year rejected", ok_b is False))
    _good = ("def run(b,c,f):\n idx=pd.date_range('2024-01-01',periods=20)\n"
             " return {'ok':True,'net':pd.Series(np.linspace(-0.02,0.03,20),index=idx),'positions':pd.Series(1.0,index=idx),'bars_per_year':52,'n_trades':20}\n")
    ok_g, why_g = ig._validate_module(_mod(_good), "m", None, None, 0.0)
    out.append(("A-014: a well-formed module still passes validation", ok_g is True))

    # A-023: _sharpe distinguishes deterministic edge from no-edge
    out.append(("A-023: constant +series -> Sharpe > 0", R._sharpe(np.array([0.01] * 5), 25) > 0))
    out.append(("A-023: constant 0series -> Sharpe == 0", R._sharpe(np.array([0.0] * 5), 25) == 0.0))
    # A-022: a constant-positive weekend bucket (std==0) is still caught as fragile
    _ix = pd.date_range("2024-01-01", periods=160, freq="D")
    _net = pd.Series(np.where(np.asarray(_ix.dayofweek) >= 5, 0.02, -0.001), index=_ix)
    out.append(("A-022: constant-positive weekend regime -> fragile", R.regime_split(_net, 365).get("fragile") is True))
    # A-021 / B-020: the inert cost term was removed — permutation now reports the honest
    # ALIGNMENT statistic (not a cost-laden 'edge'). (The old "p independent of cost" test was
    # vacuous: cost cancels under permutation regardless of the fix.)
    _rng = np.random.default_rng(0); _pos = _rng.choice([-1, 1], 40); _pay = _rng.normal(0, 0.01, 40)
    _pt = R.permutation_test(_pos, _pay, 0.0, n_perm=200)
    out.append(("A-021/B-020: permutation reports observed_alignment, not cost-laden edge",
                "observed_alignment" in _pt and "observed_edge" not in _pt))

    # A-007: empty-normalized-key never produces a false alias match
    _b = DataBundle()
    _b.series["USD Close"] = Series("USD Close", pd.Series([1.0]), "x", "native")
    out.append(("A-007: all-filler query 'rate' does not false-match", _norm_key("USD Close") == "" and _b.get("rate") is None))
    out.append(("A-007: exact all-filler key still resolves", _b.get("USD Close") is not None))
    # A-008: alias index not stale after series added post first get()
    _b2 = DataBundle()
    _b2.series["eth_spot_daily"] = Series("eth_spot_daily", pd.Series([1.0]), "x", "native")
    assert _b2.get("price.eth_usd_spot_daily") is not None      # builds cache
    _b2.series["sol_spot_daily"] = Series("sol_spot_daily", pd.Series([2.0]), "x", "native")
    out.append(("A-008: series added after first get() is found via alias", _b2.get("price.sol_usd_spot_daily") is not None))

    # A-031/A-032: source_span must occur (verbatim/ws-normalized) in the source text
    out.append(("A-031: ws-normalized in-text span accepted", span_in_text("hello   world", "say hello world now") is True))
    out.append(("A-032: fabricated span rejected", span_in_text("fabricated span", "say hello world now") is False))
    # A-034: truncation repair is brace-in-string safe
    _rep = _repair_truncated_json('{"claims": [{"statement": "a}, b", "x": 1}, {"statement": "second incomplete')
    out.append(("A-034: repair keeps a complete claim whose span contains '},'",
                _rep is not None and __import__("json").loads(_rep) == {"claims": [{"statement": "a}, b", "x": 1}]}))

    # ===================== Wave 4 core regression guards ===================== #
    # C1: trial count is scoped to the family (strategy_class + data_domain)
    _ft = Path(tempfile.gettempdir()) / "_eval_family_ledger.tsv"
    pd.DataFrame({"strategy": ["a", "b", "c", "d"], "family": ["X", "X", "Y", "Y"],
                  "per_trade_sharpe": [1, 1, 2, 2], "dsr": [0] * 4, "n": [100] * 4}).to_csv(_ft, sep="\t", index=False)
    _ol = P7.LEDGER; P7.LEDGER = _ft
    try:
        nX, _ = _REAL_TRIAL_STATS("X", "new"); nAll, _ = _REAL_TRIAL_STATS(None, "new")
    finally:
        P7.LEDGER = _ol
    out.append(("C1: family X scoping -> 2 in-family + current = 3", nX == 3))
    out.append(("C1: unscoped would count all 4 + current = 5 (shows scoping bites)", nAll == 5))

    # E2: modeled costs cap a would-be research-supported at watch
    _bt_rs = {"psr": 0.99, "dsr": 0.99, "edge_t": 5, "n_oos": 80, "oos_sharpe": 2, "is_sharpe": 1.5,
              "three_fold": {"folds": [1, 1, 1], "consistent": True}, "capacity_usd": 1e6,
              "bootstrap": {"edge_ci_includes_zero": False}, "permutation": {"p_value": 0.01},
              "regime": {"fragile": False}, "walk_forward": {}, "capacity_ci": {}}
    out.append(("E2: modeled costs cap research-supported -> watch",
                stages.p8_verdict(_claim(), _bt_rs, {"holdout_sharpe": 1.0}, False).verdict == "watch"))

    # B-007: walk-forward drift kills an otherwise-survivor (independent axis)
    _bt_wf = dict(_bt_rs); _bt_wf["walk_forward"] = {"per_window_sharpe": [1.0, -0.5, -0.3], "consistent": False}
    _d_wf = stages.p8_verdict(_claim(), _bt_wf, {}, False)
    out.append(("B-007: walk-forward drift -> kill(walk_forward_drift)",
                _d_wf.verdict == "kill" and _d_wf.kill_reason == "walk_forward_drift"))

    # B2: a kill we never reproduced in-sample is flagged not-replicated (excluded from principles)
    _bt_nr = {"psr": 0.2, "dsr": 0.2, "edge_t": 0.1, "n_oos": 80, "is_sharpe": -0.5,
              "three_fold": {"folds": [-1, -1, -1], "consistent": False}, "capacity_usd": 1e6,
              "bootstrap": {}, "permutation": {}, "regime": {}}
    out.append(("B2: kill with negative IS Sharpe -> replicated_in_sample False",
                stages.p8_verdict(_claim(), _bt_nr, {}, False).metrics.get("replicated_in_sample") is False))

    # F-finding (CZ swarm): calibration/referee harnesses must NOT pollute the PRODUCTION holdout
    # lock. final_holdout_eval honors $PENROSE_HOLDOUT_LOCK -> an isolated lock writes there, never
    # to P7.HOLDOUT_LOCK (the real .holdout_burned the live pipeline depends on).
    import os as _os, tempfile as _tf
    _iso = _os.path.join(_tf.gettempdir(), "_eval_iso_holdout.lock")
    if _os.path.exists(_iso): _os.unlink(_iso)
    _prod_before = P7.HOLDOUT_LOCK.exists()
    _os.environ["PENROSE_HOLDOUT_LOCK"] = _iso
    _ix2 = _idx(120); _hn = pd.Series(np.tile([0.01, -0.005], 60), index=_ix2)
    P7.final_holdout_eval("eval_iso", _hn, BPY, force=True)
    _os.environ.pop("PENROSE_HOLDOUT_LOCK", None)
    out.append(("HOLDOUT-ISO: $PENROSE_HOLDOUT_LOCK writes the isolated lock, not production",
                _os.path.exists(_iso) and P7.HOLDOUT_LOCK.exists() == _prod_before))
    if _os.path.exists(_iso): _os.unlink(_iso)

    # REGIME-SERIES: point-in-time vol_regime labels are look-ahead-free, and feeding them to
    # regime_split catches a VOL-regime-concentrated edge the calendar-only lens misses.
    from penrose import regime as _RG
    from penrose.pipeline import robustness as _RB
    _rng = np.random.default_rng(0)
    _px = pd.Series(100 * np.exp(np.cumsum(_rng.normal(0, 0.01, 600))),
                    index=pd.date_range("2020-01-01", periods=600))
    _vr_full = _RG.vol_regime(_px); _vr_trunc = _RG.vol_regime(_px.iloc[:400])
    _common = _vr_full.index.intersection(_vr_trunc.index)
    _pit = (_vr_full.reindex(_common) == _vr_trunc.reindex(_common)).mean()
    out.append(("REGIME: vol_regime label is point-in-time (unchanged when future data appended)",
                _pit == 1.0))
    _vr = _RG.vol_regime(_px); _ix = _vr.index
    _edge = np.where(_vr.values == "high_vol", _rng.normal(0.004, 0.002, len(_ix)),
                     _rng.normal(0.0, 0.004, len(_ix)))
    _net = pd.Series(_edge, index=_ix)
    _cal = _RB.regime_split(_net, 252.0)
    _vol = _RB.regime_split(_net, 252.0, extra_schemes={"vol_regime": _vr})
    out.append(("REGIME: vol_regime lens catches a vol-concentrated edge the calendar lens misses",
                _cal["fragile"] is False and _vol["fragile"] is True))
    # H-001: a tz-NAIVE trade index vs tz-AWARE UTC labels must still align (regime_split normalizes
    # both to UTC); pre-fix the mismatch reindexed to all-"unknown" and the partition silently vanished.
    _vr_utc = _vr.copy(); _vr_utc.index = pd.DatetimeIndex(_vr_utc.index).tz_localize("UTC")
    _mismatch = _RB.regime_split(_net, 252.0, extra_schemes={"vol_regime": _vr_utc})
    out.append(("REGIME H-001: vol lens fires across a tz-naive/tz-aware index mismatch",
                "vol_regime" in _mismatch.get("schemes", {}) and _mismatch["fragile"] is True))
    # REGIME-WIRING: the PRODUCTION path (run_backtest) threads regime_schemes into the kill-lens,
    # so vol/trend buckets appear in P7's regime output AND inflate the DSR trial count (no free look).
    _pos = pd.Series(1.0, index=_ix)
    _bt_cal = P7.run_backtest("eval_regime_cal", _net, _pos, 252.0, log=False)
    _bt_vol = P7.run_backtest("eval_regime_vol", _net, _pos, 252.0, log=False,
                              regime_schemes={"vol_regime": _vr})
    out.append(("REGIME-WIRING: run_backtest threads regime_schemes into the kill-lens",
                "vol_regime" in _bt_vol["regime"].get("schemes", {})
                and "vol_regime" not in _bt_cal["regime"].get("schemes", {})))
    out.append(("REGIME-WIRING: vol/trend partitions inflate the DSR trial count",
                _bt_vol["n_trials"] > _bt_cal["n_trials"]))
    # PROVENANCE-SHELF: auto-generated (UNTRUSTED) modules live under modules/_auto and are NEVER
    # registered for cross-claim routing; operator-curated trusted modules ARE. (H-iteration-1)
    from penrose.pipeline import run as _RUN
    from penrose import config as _CFG
    _RUN.REGISTRY.clear()
    _RUN._register_known_modules()
    _reg_ids = {getattr(m, "__module_id__", "?") for m in _RUN.REGISTRY.values()}
    _auto_ids = {p.parent.name for p in (_CFG.AUTO_MODULES.glob("*/impl.py"))} if _CFG.AUTO_MODULES.exists() else set()
    out.append(("PROVENANCE-SHELF: no _auto (machine-generated) module is registered for routing",
                len(_reg_ids & _auto_ids) == 0))
    out.append(("PROVENANCE-SHELF: trusted operator module crypto_funding_carry IS registered",
                "crypto_funding_carry" in _reg_ids))

    # ===================== Wave 5 regression guards ===================== #
    # C-004: the walk-forward KILL gate must never read the locked holdout. run_backtest
    # truncates wf_frame to IS+OOS (final 0.20 dropped), so flipping ONLY the holdout tail of
    # wf_frame cannot change walk_forward. Build a frame whose non-holdout portion is identical
    # in both cases and whose holdout tail differs wildly; the walk_forward result must match.
    _n = 200
    _wf_idx = _idx(_n)
    _sig = pd.Series(np.tile([1.0, -1.0], _n // 2), index=_wf_idx)
    _rv = pd.Series(np.tile([0.02, 0.01], _n // 2), index=_wf_idx)
    _iv = pd.Series(0.015, index=_wf_idx)
    _base = pd.DataFrame({"signal": _sig, "fut_rv": _rv, "iv": _iv})
    _o = int(_n * (P7.IS_FRAC + P7.OOS_FRAC))
    _frame_a = _base.copy()
    _frame_b = _base.copy()
    _frame_b.iloc[_o:, _frame_b.columns.get_loc("fut_rv")] = -5.0   # corrupt ONLY the holdout tail
    _net = pd.Series(np.tile([0.001, -0.0005], _n // 2), index=_wf_idx)
    _pos = pd.Series(1.0, index=_wf_idx)
    _wa = P7.run_backtest("eval_wf_a", _net, _pos, BPY, log=False, cost_frac=0.0008, wf_frame=_frame_a)
    _wb = P7.run_backtest("eval_wf_b", _net, _pos, BPY, log=False, cost_frac=0.0008, wf_frame=_frame_b)
    out.append(("C-004: corrupting only the holdout tail of wf_frame leaves walk_forward unchanged",
                _wa.get("walk_forward") == _wb.get("walk_forward")))

    # C-005 (per-verdict synthetic): accessed_synthetic() trips ONLY on a synthetic series the
    # module actually READ, not merely because one EXISTS in the bundle.
    from penrose.data.contract import DataBundle, Series as _DSeries  # noqa: E402
    _si = _idx(10)
    _b = DataBundle(series={
        "real_one": _DSeries("real_one", pd.Series(1.0, index=_si), "binance-live", "frac"),
        "synth_one": _DSeries("synth_one", pd.Series(1.0, index=_si), "synthetic", "frac"),
    })
    _b.reset_access(); _b.get("real_one")
    out.append(("C-005: reading only a real series -> accessed_synthetic False",
                _b.accessed_synthetic() is False))
    _b.reset_access(); _b.get("synth_one")
    out.append(("C-005: reading a synthetic series -> accessed_synthetic True",
                _b.accessed_synthetic() is True))

    # LEDGER-SCHEMA: a ragged trial ledger (legacy 4-col header + a newer 5-col `family` row)
    # must NEVER raise into a backtest -> _trial_stats reads defensively and degrades gracefully.
    _rl = Path(tempfile.gettempdir()) / "_eval_ragged_ledger.tsv"
    _rl.write_text("strategy\tper_trade_sharpe\tdsr\tn\n"
                   "a\t1.0\t0.5\t100\n"
                   "b\tfam::x\t0.9\t0.4\t100\n")     # 5 fields under a 4-field header
    _ol2 = P7.LEDGER; P7.LEDGER = _rl
    try:
        _ok = True
        try:
            _n, _v = _REAL_TRIAL_STATS("fam::x", "cur")
        except Exception:
            _ok = False
    finally:
        P7.LEDGER = _ol2
    out.append(("LEDGER: ragged/schema-drifted ledger does not crash _trial_stats", _ok))

    # Appending to a legacy ledger must rewrite the canonical header rather than add a wider row
    # beneath the old schema.
    _ml = Path(tempfile.gettempdir()) / "_eval_migrate_ledger.tsv"
    _ml.write_text("strategy\tper_trade_sharpe\tdsr\tn\nold\t0.1\t0.2\t50\n")
    _old_ml = P7.LEDGER; P7.LEDGER = _ml
    try:
        P7._append_ledger("new", {"per_trade_sharpe": 0.2, "dsr": 0.3, "n": 60},
                          family="generated::crypto")
        _migrated = pd.read_csv(_ml, sep="\t")
    finally:
        P7.LEDGER = _old_ml
        _ml.unlink(missing_ok=True)
        Path(str(_ml) + ".lock").unlink(missing_ok=True)
    out.append(("LEDGER-MIGRATION: append rewrites the canonical schema",
                list(_migrated.columns) == P7._LEDGER_COLS and len(_migrated) == 2))

    # ===================== Pass-4 swarm fixes (D-findings) ===================== #
    # D-001/D-002: untrusted (auto-generated) module code must NEVER be import-exec'd in penrose's
    # process — the Docker sandbox only confines run(), not the file's top-level code. Registration
    # and the sandboxed validation path read the contract STATICALLY via ast_module_meta. A file
    # whose top-level code would CRASH if executed (1/0) must still yield correct metadata.
    _modtext = ("__strategy_class__ = 'x'\n__module_id__ = 'm'\n__auto_generated__ = True\n"
                "1/0  # would raise if this were executed\n"
                "def run(b, c, f):\n    return {'ok': True}\n")
    _meta = ig.ast_module_meta(_modtext)
    out.append(("D-001/002: ast_module_meta reads contract WITHOUT executing top-level code",
                _meta["has_run"] and _meta["strategy_class"] == "x" and _meta["module_id"] == "m"))
    out.append(("D-002: __auto_generated__ detected statically -> registration skips in-process exec",
                _meta["auto_generated"] is True))
    out.append(("D-002: a trusted operator module is not mis-flagged auto_generated",
                ig.ast_module_meta("__strategy_class__='y'\ndef run(b,c,f): pass\n")["auto_generated"] is False))
    out.append(("D-001: scanner now rejects pd.read_pickle (one-line RCE that \\bpickle\\b missed)",
                ig._scan_code_text("import pandas as pd\npd.read_pickle('/tmp/x')") is not None))

    # D-004: the SANDBOXED validation path (prerun_result provided) must return (True, handle) on a
    # clean module — the D-001 refactor left `return True, module` referencing an unbound name there,
    # silently breaking auto-impl on the production path while eval (which omits prerun_result) stayed
    # green. This guard exercises the real production branch.
    import tempfile as _tf, os as _os
    _md = _tf.mkdtemp(); _ip = _os.path.join(_md, "impl.py")
    open(_ip, "w").write("__strategy_class__='x'\n__module_id__='m'\n__auto_generated__=True\n"
                         "def run(b,c,f):\n return {'ok':True}\n")
    _ix = pd.date_range("2024-01-01", periods=60)
    _nz = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 60), index=_ix)
    _pr = {"ok": True, "net": _nz, "positions": pd.Series(1.0, index=_ix), "bars_per_year": 365.0,
           "n_trades": 60, "payoff": _nz, "position_signed": pd.Series(1.0, index=_ix)}
    _ok4, _h4 = ig._validate_module(_ip, "m", None, None, 0.0008, prerun_result=_pr)
    out.append(("D-004: sandboxed _validate_module returns (True, handle) on a clean module",
                _ok4 is True and getattr(_h4, "__auto_generated__", False) is True))

    # D-005: the REJECTED stub must (a) be marked __auto_generated__ so _register_known_modules skips
    # it (never exec'd), and (b) collapse an LLM-controlled last_err so an embedded newline+code
    # can't break out of the `#` comment into an executable top-level statement.
    _payload = 'm\nimport os\nos.system("x")\n#'
    _safe = repr(str(_payload))[:300].replace("\n", " ")
    _stub = ("__auto_generated__ = True  # REJECTED stub\n"
             f"# auto-impl REJECTED after 3 attempts: {_safe}\n# spec kept.\n")
    _sm = ig.ast_module_meta(_stub)
    _stub_tree = __import__("ast").parse(_stub)
    _injected = any(isinstance(n, (__import__("ast").Import, __import__("ast").Expr,
                                   __import__("ast").Call)) for n in _stub_tree.body)
    out.append(("D-005: rejected stub is __auto_generated__ (registration skips it)",
                _sm["auto_generated"] is True))
    out.append(("D-005: newline-injected last_err cannot become an executable stub statement",
                _injected is False))

    # ===================== DREAM CYCLE invariants ===================== #
    # Define the methodology BEFORE the generator exists. A dream run is a registered search,
    # not a prompt that quietly hands the pipeline only its favorite candidates.
    from penrose import dream as _DR
    _dream_dir = Path(tempfile.mkdtemp()) / "dream-test"
    _manifest = _DR.create_manifest(
        run_id="dream-test", generation_budget=7, model="test-model",
        corpus_snapshot_hash="abc123", root=_dream_dir)
    _raw = [
        {"statement": f"candidate {i}", "mechanism": "m", "scope": "BTC",
         "horizon": "1d", "strategy_class": "dream-eval",
         "candidate_class": "testable_now"}
        for i in range(5)
    ]
    _DR.record_candidates(_manifest, _raw)
    _saved = __import__("json").loads((_dream_dir / "manifest.json").read_text())
    out.append(("DREAM-MANIFEST: candidates_generated equals the actual immutable emitted count",
                _saved["candidates_generated"] == 5
                and _saved["generation_budget"] == 7))

    # An untrusted generator may over-emit. The effective denominator must never be smaller than
    # the actual population that can enter testing.
    _over_dir = Path(tempfile.mkdtemp()) / "dream-over"
    _over_manifest = _DR.create_manifest(
        run_id="dream-over", generation_budget=2, model="test-model",
        corpus_snapshot_hash="abc123", root=_over_dir)
    _over_raw = [
        {"statement": f"over candidate {i}", "strategy_class": f"class-{i}",
         "candidate_class": "testable_now"} for i in range(3)
    ]
    _over_manifest = _DR.record_candidates(_over_manifest, _over_raw)
    _over_claims, _over_norm = _DR.normalize_candidates("dream-over", _over_raw)
    _odl = Path(tempfile.gettempdir()) / "_eval_dream_over.tsv"
    _odl.unlink(missing_ok=True); _old_odl = P7.LEDGER; P7.LEDGER = _odl
    try:
        _over_manifest = _DR.register_search(_over_manifest, _over_claims, _over_norm)
    finally:
        P7.LEDGER = _old_odl
        _odl.unlink(missing_ok=True)
        Path(str(_odl) + ".lock").unlink(missing_ok=True)
    out.append(("DREAM-OVEREMIT: effective denominator >= actual emitted population",
                _over_manifest["effective_search_denominator"] == 3))

    # Pre-registration: even candidates that never reach P7 count in the search denominator.
    _dl = Path(tempfile.gettempdir()) / "_eval_dream_ledger.tsv"
    _dl.unlink(missing_ok=True)
    _old_dl = P7.LEDGER; P7.LEDGER = _dl
    try:
        P7.register_trials([
            {"strategy": f"dream-test-c{i}", "family": "dream-eval::crypto",
             "generation_source": "dream", "search_cohort_id": "dream-test",
             "search_denominator": 7}
            for i in range(5)
        ])
        _dn, _ = _REAL_TRIAL_STATS("dream-eval::crypto", "dream-test-c1")
        _ddf = pd.read_csv(_dl, sep="\t")
    finally:
        P7.LEDGER = _old_dl; _dl.unlink(missing_ok=True)
    out.append(("DREAM-SEARCH: P7 uses the pre-registered budget, not emitted/admitted backtests",
                _dn == 7))
    out.append(("DREAM-LEDGER: generation source + cohort are durable audit columns",
                {"generation_source", "search_cohort_id", "search_denominator"}.issubset(_ddf.columns)
                and set(_ddf["generation_source"]) == {"dream"}))

    # Generator vocabulary cannot reset a family: distinct/renamed strategy classes in one data
    # domain share Penrose's stable generated::<domain> family.
    _fam_raw = [
        {"statement": "BTC carry candidate", "strategy_class": "carry-alpha",
         "candidate_class": "testable_now"},
        {"statement": "Bitcoin carry candidate", "strategy_class": "renamed-carry-system",
         "candidate_class": "testable_now"},
    ]
    _fam_claims, _ = _DR.normalize_candidates("family-test", _fam_raw)
    out.append(("DREAM-FAMILY-CANON: generator class renames cannot reset the family",
                len({_DR._family_for(c) for c in _fam_claims}) == 1))

    # Repeated cohorts in the SAME stable family accumulate.
    _dl2 = Path(tempfile.gettempdir()) / "_eval_dream_accum.tsv"
    _dl2.unlink(missing_ok=True); _old_dl2 = P7.LEDGER; P7.LEDGER = _dl2
    try:
        for cohort in ("r1", "r2"):
            P7.register_trials([
                {"strategy": f"{cohort}-c{i}", "family": "generated::crypto",
                 "generation_source": "dream", "search_cohort_id": cohort,
                 "search_denominator": 3}
                for i in range(3)
            ])
        _accum_n, _ = _REAL_TRIAL_STATS("generated::crypto", "r2-c0")
    finally:
        P7.LEDGER = _old_dl2; _dl2.unlink(missing_ok=True)
    out.append(("DREAM-FAMILY: repeated dream cohorts accumulate in one persistent strategy family",
                _accum_n == 6))

    # Dream triage cannot burn ANY holdout lock, including when force=True is accidentally passed.
    _ro = Path(tempfile.gettempdir()) / "_eval_dream_readonly_holdout.lock"
    _ro.unlink(missing_ok=True)
    _old_mode = _os.environ.get("PENROSE_HOLDOUT_MODE")
    _old_lock_env = _os.environ.get("PENROSE_HOLDOUT_LOCK")
    _os.environ["PENROSE_HOLDOUT_MODE"] = "readonly"
    _os.environ["PENROSE_HOLDOUT_LOCK"] = str(_ro)
    _ro_res = P7.final_holdout_eval("dream", _hn, BPY, force=True)
    if _old_mode is None: _os.environ.pop("PENROSE_HOLDOUT_MODE", None)
    else: _os.environ["PENROSE_HOLDOUT_MODE"] = _old_mode
    if _old_lock_env is None: _os.environ.pop("PENROSE_HOLDOUT_LOCK", None)
    else: _os.environ["PENROSE_HOLDOUT_LOCK"] = _old_lock_env
    out.append(("DREAM-HOLDOUT: read-only mode refuses and never writes a lock (even force=True)",
                _ro_res.get("refused") is True and not _ro.exists()))

    # LLM-originated hypotheses have no external anchor. Even a positive holdout cannot let a
    # self-consistent dream claim claim the strongest external-evidence verdict.
    _old_cost = _CFG.COST_PROVENANCE
    _CFG.COST_PROVENANCE = "measured"  # isolate the source-origin cap from E2
    try:
        _dc = _claim(); _dc.source_type = "generated_hypothesis"
        _ddec = stages.p8_verdict(_dc, _bt_rs, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
        _cc = _claim(); _cc.source_type = "chat"
        _cdec = stages.p8_verdict(_cc, _bt_rs, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
    finally:
        _CFG.COST_PROVENANCE = _old_cost
    out.append(("DREAM-FIDELITY: generated hypotheses are capped below research-supported",
                _ddec.verdict == "watch"
                and _ddec.metrics.get("fidelity_provenance") == "self-authored-unanchored"
                and "external anchor" in _ddec.rationale))
    out.append(("CHAT-FIDELITY: chat hypotheses receive the same unanchored-source cap",
                _cdec.verdict == "watch"
                and _cdec.metrics.get("fidelity_provenance") == "self-authored-unanchored"))

    # ===================== PENROSE EDGE engine guards (WP1-WP6) ===================== #
    from penrose.concepts import extract as _extract_concept
    from penrose.corpus import build as _build_corpus
    from penrose.confirmation import validate_firewall as _validate_firewall
    from penrose.explanations import exposure_decomposition as _exposure
    from penrose.synthesize import normalize as _normalize_synthesis

    _kc = _extract_concept({
        "claim_id": "edge-kill", "statement": "noise edge", "verdict": "kill",
        "competing_explanations": [{"explanation": "carry survived", "verdict": "survives"}],
    }, use_llm=False)
    out.append(("EDGE-WP1: kill cannot emit a supported surviving explanation",
                _kc is not None and _kc.surviving_explanation == ""))

    _base = [{"concept_id": f"e{i}", "statement": "positive observation",
              "created_at": "2026-01-01", "evidence_direction": "positive", "data_provenance":
              {"strategy_family": "carry", "data_domain": "crypto"}} for i in range(3)]
    _cg1 = _build_corpus(_base, min_support=3, current_year=2026)
    _more = _base + [{"concept_id": f"r{i}", "statement": "positive observation",
                      "created_at": "2026-01-01", "evidence_direction": "positive", "data_provenance":
                      {"strategy_family": "yield", "data_domain": "rates"}} for i in range(3)]
    _cg2 = _build_corpus(_more, min_support=3, current_year=2026)
    out.append(("EDGE-WP2: cross-family mechanism requires two distinct domains",
                not any(n["level"] == "cross_family_mechanism" for n in _cg1["nodes"])
                and any(n["level"] == "cross_family_mechanism" for n in _cg2["nodes"])))

    _x = pd.Series(np.linspace(-1, 1, 100)); _y = 0.002 + 2 * _x
    _xd = _exposure(_y, market=_x)
    out.append(("EDGE-WP3: exposure decomposition recovers known beta",
                _xd.get("applicable") and abs(_xd["betas"]["market"] - 2) < 1e-9
                and _xd.get("survives") is True))

    _sg = {"nodes": [{"node_id": "cross-1", "level": "cross_family_mechanism"}]}
    _sc, _sn = _normalize_synthesis("edge-s", [{
        "statement": "candidate hypothesis", "candidate_class": "testable_now",
        "strategy_class": "edge", "inspired_by": ["cross-1"]}], _sg)
    out.append(("EDGE-WP4: synthesized hypotheses are unanchored candidates with grounded lineage",
                len(_sc) == 1 and _sc[0].source_type == "synthesized_hypothesis"
                and _sn[0]["grounded"] and _sn[0]["admitted"]))

    _fw_ok, _fw_reason = _validate_firewall(
        {"data_provenance": {"data_domains": ["reserved"],
                             "periods": [{"start": "2020-01-01", "end": "2021-01-01"}]}},
        {"epoch_id": "r", "start": "2024-01-01", "end": "2025-01-01",
         "data_domains": ["reserved"], "datasets": []})
    out.append(("EDGE-WP5: confirmation refuses reserve-touching provenance",
                not _fw_ok and "intersects" in _fw_reason))

    _noise = [{"concept_id": f"n{i}", "statement": "random observation",
               "created_at": "2026-01-01", "evidence_direction": "unknown", "data_provenance":
               {"strategy_family": f"noise-{i}", "data_domain": f"noise-{i}"}} for i in range(9)]
    _ng = _build_corpus(_noise, min_support=3, current_year=2026)
    out.append(("EDGE-WP6: noise placebo yields zero cross-family synthesis inputs",
                not any(n["level"] == "cross_family_mechanism" for n in _ng["nodes"])))

    # ===================== Falsification-gate battery (G1/G3) ===================== #
    from scripts.calibration_persistence import (
        block_flip as _block_flip,
        dead_persistent_series as _dead_persistent_series,
        per_period_flip as _per_period_flip,
        real_btc_returns as _real_btc_returns,
        verdict_for_dead_persistent as _verdict_for_dead_persistent,
        verdict_for_return_momentum as _verdict_for_return_momentum,
    )

    _ret = _real_btc_returns()
    _pp_ok = True
    _block_ok = True
    for _k in range(5):
        _pp = _per_period_flip(_ret, np.random.default_rng(411_000 + _k))
        _pv, _ = _verdict_for_return_momentum(_pp, 0.0, name="eval_persist_pp")
        _pp_ok = _pp_ok and _pv not in ("watch", "research-supported")
        _bf = _block_flip(_ret, np.random.default_rng(420_000 + _k), 12)
        _bv, _ = _verdict_for_return_momentum(_bf, 0.0, name="eval_persist_b12")
        _block_ok = _block_ok and _bv != "research-supported"
    out.append(("CALIB-PERSIST-1: block-preserving drift-destroyed null never certifies",
                _block_ok))
    out.append(("CALIB-PERSIST-2: per-period sign-flip null never reaches watch+",
                _pp_ok))
    _dead = _verdict_for_dead_persistent(seed=430_000)
    out.append(("CALIB-DEAD-1: dead-but-persistent null never reaches watch+",
                _dead in ("kill", "underpowered", "insufficient_data")))
    # Degenerate dead-state helper fails clearly instead of surfacing a raw numeric traceback.
    _dead_msg = ""
    try:
        _dead_persistent_series(n=10, seed=1)
    except ValueError as e:
        _dead_msg = str(e)
    out.append(("CALIB-DEAD: degenerate series has a clear failure message",
                "at least 40 observations" in _dead_msg))
    return out


def main() -> None:
    rows = []
    for name, build, expect_v, expect_k in CASES:
        rng = np.random.default_rng(42)
        net = build(rng)
        dec, bt = _verdict(net)
        vpass = dec.verdict in expect_v
        kpass = expect_k is None or expect_k in str(dec.kill_reason or "")
        rows.append((name, dec.verdict, dec.kill_reason, expect_v, expect_k, vpass and kpass))

    print("== verdict ground-truth ==")
    print(f"{'case':28s} {'verdict':20s} {'kill_reason':16s} {'expected':30s} result")
    print("-" * 100)
    for name, v, k, ev, ek, ok in rows:
        exp = "|".join(ev) + (f" /{ek}" if ek else "")
        print(f"{name:28s} {v:20s} {str(k or '-'):16s} {exp:30s} {'OK' if ok else 'FAIL'}")

    inv = invariants()
    print("\n== regression invariants ==")
    for name, ok in inv:
        print(f"{name:60s} {'OK' if ok else 'FAIL'}")

    all_rows = [r[-1] for r in rows] + [ok for _, ok in inv]
    npass, total = sum(all_rows), len(all_rows)
    print("-" * 100)
    print(f"{npass}/{total} passed")
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    main()
