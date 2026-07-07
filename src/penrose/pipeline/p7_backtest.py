"""P7 — backtest harness (S4 discipline) for penrose.

Uses the vendored statistics module for its INSTRUMENT-AGNOSTIC statistics
(deflated_sharpe, probabilistic_sharpe, sharpe) — the multiple-testing-aware DSR
is the whole point. penrose supplies the strategy P&L because the paper's
instrument is volatility, not a crypto perp, so harness.evaluate()'s crypto-panel
loader does not apply.

What this adds on top of the raw statistics, per S4:
  * 50/30/20 IS / OOS / holdout time split (mirrors harness IS_FRAC/OOS_FRAC)
  * 3-fold consistency check on IS+OOS (sign-stable per-fold Sharpe)
  * single-use locked holdout (penrose-local lockfile, like harness BURN)
  * capacity_usd via the harness linear-impact model
  * multiple-testing penalty: n_trials/sr_variance grow via a penrose ledger
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd

from .. import config
from . import robustness as R
from .. import stats as H        # vendored DSR/Sharpe/capacity — penrose is self-contained

LEDGER = config.ROOT / "backtest_ledger.tsv"
HOLDOUT_LOCK = config.ROOT / ".holdout_burned"
IS_FRAC, OOS_FRAC = 0.50, 0.30                 # holdout = final 0.20


def _claim_holdout_lock(name: str) -> Path:
    """Return the atomic holdout lock for this claim identity."""
    if os.environ.get("PENROSE_HOLDOUT_LOCK"):
        return Path(os.environ["PENROSE_HOLDOUT_LOCK"])
    raw = str(name or "unknown-claim")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")[:80] or "claim"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return HOLDOUT_LOCK.parent / f"{HOLDOUT_LOCK.name}.{slug}.{digest}.lock"


def _trial_stats(family: str | None = None, strategy: str | None = None,
                 registered_trials: int | None = None, *,
                 generation_source: str = "paper",
                 search_cohort_id: str | None = None,
                 preregistered_single_cohort: bool = False) -> tuple[int, float]:
    """n_trials + cross-trial Sharpe variance for DSR deflation.

    C1: trials are scoped to the FAMILY (strategy_class + data_domain), so testing 500 crypto
    ideas doesn't make an unrelated weather strategy unbeatable. A trial is a DISTINCT strategy,
    not a run (dedup), and the CURRENT strategy is counted exactly once (B-016: no +1 double
    count on a re-run)."""
    prior = getattr(config, "DEFLATION_PRIOR", {})
    if generation_source == "provided_series_statistic" and preregistered_single_cohort:
        floor = 1
    else:
        floor = int(prior.get(
            "generated_min_trials" if generation_source == "generated" else "external_min_trials",
            1,
        ))
    sr_var_prior = float(prior.get("sr_var_prior", 0.0))
    min_scored = int(prior.get("min_scored_for_empirical_var", 0))

    def _with_prior(n_raw: int, var_raw: float, n_scored: int) -> tuple[int, float]:
        n = max(1, n_raw, floor)
        var = float(var_raw)
        if min_scored > 0 and n_scored < min_scored:
            var = max(var, sr_var_prior)
        elif min_scored > 0 and sr_var_prior > 0:
            var = ((n_scored * var) + (min_scored * sr_var_prior)) / (n_scored + min_scored)
        return n, var

    if not LEDGER.exists():
        return _with_prior(1 if strategy else 1, 0.0, 0)
    try:
        # Tolerate a ragged ledger (schema drift from the C1 `family` column, or a half-written
        # row): a corrupt trial ledger must NEVER raise into a real backtest. Skip bad lines and
        # fall back to "current strategy only" if the file is unreadable.
        df = pd.read_csv(LEDGER, sep="\t", engine="python", on_bad_lines="skip")
    except Exception:  # noqa: BLE001
        return _with_prior(1 if strategy else 1, 0.0, 0)
    if family is not None and "family" in df.columns:
        df = df[df["family"].astype(str) == family]      # deflate within the family only
    if "strategy" in df.columns:
        df = df.drop_duplicates(subset="strategy", keep="last")
    strategies = set(df["strategy"].astype(str)) if "strategy" in df.columns else set()
    if strategy:
        strategies.add(strategy)                          # current strategy counted once (B-016)
    # Registered generator searches may contain candidates that never reached P7 (duplicate,
    # conceptual-only, needs-data, failed auto-impl). They still belong to the search and must
    # count in the DSR denominator. Each cohort stores its declared denominator on every row;
    # sum one max value per cohort, then compare with the concrete distinct-strategy count.
    registered = int(registered_trials or 0)
    if {"search_cohort_id", "search_denominator"}.issubset(df.columns):
        cohorts = df.dropna(subset=["search_cohort_id"]).copy()
        if len(cohorts):
            cohorts["search_cohort_id"] = cohorts["search_cohort_id"].astype(str)
            cohorts["search_denominator"] = pd.to_numeric(
                cohorts["search_denominator"], errors="coerce").fillna(0)
            registered = max(
                registered,
                int(cohorts.groupby("search_cohort_id")["search_denominator"].max().sum()),
            )
    n = max(1, len(strategies), registered)
    scored = df
    if search_cohort_id and "search_cohort_id" in scored.columns:
        # PEN-06: same-run cohort members must see the same empirical variance. Use only prior
        # cohorts/outside rows; the current cohort's scoring order cannot move DSR.
        scored = scored[scored["search_cohort_id"].astype(str) != str(search_cohort_id)]
    sr = pd.to_numeric(scored.get("per_trade_sharpe", pd.Series(dtype=float)), errors="coerce").dropna()
    var = float(sr.var(ddof=1)) if len(sr) > 1 else 0.0
    return _with_prior(n, var, len(sr))


_LEDGER_COLS = [
    "strategy", "family", "generation_source", "search_cohort_id", "search_denominator",
    "per_trade_sharpe", "dsr", "n",
]


def _canonicalize_ledger() -> pd.DataFrame:
    """Read + migrate the small append ledger to the current schema."""
    if not LEDGER.exists():
        return pd.DataFrame(columns=_LEDGER_COLS)
    old = pd.read_csv(LEDGER, sep="\t", engine="python", on_bad_lines="skip")
    for col in _LEDGER_COLS:
        if col not in old.columns:
            old[col] = ""
    return old.reindex(columns=_LEDGER_COLS)


@contextmanager
def _ledger_guard():
    lock_path = Path(str(LEDGER) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_ledger(df: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_name(f"{LEDGER.name}.{uuid.uuid4().hex}.tmp")
    df.reindex(columns=_LEDGER_COLS).to_csv(tmp, sep="\t", index=False)
    tmp.replace(LEDGER)


def register_trials(rows: list[dict]) -> None:
    """Pre-register every candidate in a disclosed search BEFORE any candidate is tested.

    This is the generator-firehose defense: duplicates, blockers, and candidates that fail before
    P7 still inflate the denominator. Re-registering the same strategy is idempotent.
    """
    if not rows:
        return
    with _ledger_guard():
        try:
            old = _canonicalize_ledger()
        except Exception:  # noqa: BLE001
            old = pd.DataFrame(columns=_LEDGER_COLS)
        incoming = pd.DataFrame(rows).reindex(columns=_LEDGER_COLS)
        if not old.empty and not incoming.empty:
            old_last = old.drop_duplicates(subset=["strategy"], keep="last").set_index("strategy")
            kept = []
            for row in incoming.to_dict("records"):
                strategy = row.get("strategy")
                if strategy in old_last.index:
                    existing = old_last.loc[strategy].to_dict()
                    if all(str(existing.get(col, "")) == str(row.get(col, "")) for col in _LEDGER_COLS):
                        kept.append(row)
                        continue
                    scored = not pd.isna(pd.to_numeric(existing.get("per_trade_sharpe"), errors="coerce"))
                    if scored:
                        print(
                            f"penrose: warning: registration for scored strategy '{strategy}' ignored",
                            file=sys.stderr,
                        )
                        continue
                    old_den = pd.to_numeric(existing.get("search_denominator"), errors="coerce")
                    new_den = pd.to_numeric(row.get("search_denominator"), errors="coerce")
                    old_den = 0 if pd.isna(old_den) else int(old_den)
                    new_den = 0 if pd.isna(new_den) else int(new_den)
                    row["search_denominator"] = max(old_den, new_den)
                kept.append(row)
            incoming = pd.DataFrame(kept).reindex(columns=_LEDGER_COLS)
        merged = pd.concat([old, incoming], ignore_index=True)
        merged = merged.drop_duplicates(subset=["strategy"], keep="last")
        _write_ledger(merged)


def cleanup_unscored_paper_cohorts(cohort_ids: set[str]) -> int:
    """Remove paper-path cohort registrations that never became scored trials.

    Dream/generator cohorts intentionally count blocker candidates. This cleanup is only for
    paper/external runs, where pre-registration exists to make same-run scoring order-independent
    and should not leave unscored phantoms that over-deflate later papers in the same family.
    """
    cohort_ids = {str(c or "") for c in cohort_ids if str(c or "")}
    if not cohort_ids:
        return 0
    with _ledger_guard():
        old = _canonicalize_ledger()
        if old.empty:
            return 0
        cohort = old["search_cohort_id"].astype(str).isin(cohort_ids)
        paper = old["generation_source"].astype(str) == "paper"
        unscored = pd.to_numeric(old["per_trade_sharpe"], errors="coerce").isna()
        remove = cohort & paper & unscored
        removed = int(remove.sum())
        if removed:
            _write_ledger(old.loc[~remove].copy())
        return removed


def _append_ledger(name: str, stats: dict, family: str | None = None, *,
                   generation_source: str = "paper",
                   search_cohort_id: str | None = None,
                   search_denominator: int | None = None) -> None:
    row = {
        "strategy": name, "family": family or "",
        "generation_source": generation_source,
        "search_cohort_id": search_cohort_id or "",
        "search_denominator": search_denominator or "",
        "per_trade_sharpe": stats.get("per_trade_sharpe"),
        "dsr": stats.get("dsr"), "n": stats.get("n"),
    }
    # Always read/migrate/rewrite the small ledger under one lock. Appending beneath a legacy
    # header can create a ragged TSV, and concurrent whole-file updates can lose cohorts.
    with _ledger_guard():
        try:
            old = _canonicalize_ledger()
        except Exception:  # noqa: BLE001 — a corrupt ledger must not block logging; start fresh
            old = pd.DataFrame(columns=_LEDGER_COLS)
        merged = pd.concat(
            [old, pd.DataFrame([row]).reindex(columns=_LEDGER_COLS)], ignore_index=True)
        merged = merged.drop_duplicates(subset=["strategy"], keep="last")
        _write_ledger(merged)


def _three_fold(net: np.ndarray) -> dict:
    """Sign-stable Sharpe across 3 contiguous folds (overfit smell test)."""
    if len(net) < 30:
        return {"folds": [], "consistent": False, "note": "too few trades for 3-fold",
                "fold_n": [], "fold_t": [], "min_fold_t": None}
    folds = np.array_split(net, 3)
    sh = []
    fold_t = []
    for f in folds:
        if len(f) >= 5 and f.std(ddof=1) > 0:
            t = round(float(f.mean() / (f.std(ddof=1) / math.sqrt(len(f)))), 3)
            sh.append(t)
            fold_t.append(t)
        else:
            sh.append(None)
            fold_t.append(None)
    valid = [s for s in sh if s is not None]
    valid_t = [t for t in fold_t if t is not None]
    consistent = len(valid) == 3 and all(s > 0 for s in valid)
    return {"folds": sh, "consistent": consistent,
            "fold_n": [len(f) for f in folds],
            "fold_t": fold_t,
            "min_fold_t": min(valid_t) if valid_t else None}


def _turnover_from_position(pos: pd.Series) -> pd.Series:
    p = pd.Series(pos).fillna(0.0).astype(float)
    return p.diff().abs().fillna(p.abs())


def _align_to_index(series: pd.Series, index: pd.Index) -> pd.Series:
    """Align a contract series while preserving duplicate timestamps when already aligned."""
    if isinstance(series, pd.Series) and series.index.equals(index):
        return series
    return series.reindex(index)


def _cost_sensitivity(net: pd.Series, positions: pd.Series, bars_per_year: float,
                      configured_cost_frac: float | None, n_trials: int, sr_var: float,
                      *, payoff: pd.Series | None = None,
                      position_signed: pd.Series | None = None) -> dict:
    """Find the first higher round-trip cost where the OOS edge would fail.

    This is advisory unless config.COST_SENSITIVITY_GATE is explicitly enabled. It never mutates
    the supplied net series and uses a coarse deterministic grid with no new randomness.
    """
    cfg = float(configured_cost_frac or 0.0)
    if len(net) < config.DSR_DECISION["min_oos_bars"]:
        return {"breakeven_cost_frac": None, "configured_cost_frac": cfg, "margin": None,
                "max_tested_cost_frac": cfg, "note": "insufficient trades for cost sensitivity"}
    try:
        base_idx = net.index
        if payoff is not None and position_signed is not None:
            pay = _align_to_index(payoff, base_idx).astype(float)
            signed = _align_to_index(position_signed, base_idx).fillna(0.0).astype(float)
            turn = _turnover_from_position(signed)

            def at_cost(c: float) -> pd.Series:
                return signed * pay - turn * c
        else:
            pos = _align_to_index(positions, base_idx).fillna(0.0).astype(float)
            turn = _turnover_from_position(pos)

            def at_cost(c: float) -> pd.Series:
                return net - turn * max(0.0, c - cfg)

        step = cfg if cfg > 0 else 0.0005
        max_cost = max(step * 10, cfg)
        grid = [round(step * i, 10) for i in range(int(round(max_cost / step)) + 1)]
        if cfg not in grid:
            grid.append(cfg)
        grid = sorted(c for c in grid if c >= cfg - 1e-12)
        i = int(len(net) * IS_FRAC)
        o = int(len(net) * (IS_FRAC + OOS_FRAC))
        breakeven = None
        for c in grid:
            trial = at_cost(c).dropna()
            oos = trial.iloc[i:o].values
            if len(oos) < 20:
                continue
            dsr = H.deflated_sharpe(oos, n_trials, sr_var)
            if dsr < config.DSR_DECISION["kill_below_psr"]:
                breakeven = float(c)
                break
            boot_cfg = dict(config.BOOTSTRAP)
            boot_cfg["n_boot"] = min(int(boot_cfg.get("n_boot", 2000)), 200)
            boot = R.block_bootstrap(oos, bars_per_year, **boot_cfg) if len(oos) >= 10 else {}
            if boot.get("edge_ci_includes_zero"):
                breakeven = float(c)
                break
        margin = (breakeven / cfg) if (breakeven is not None and cfg > 0) else None
        return {"breakeven_cost_frac": breakeven, "configured_cost_frac": cfg,
                "margin": round(float(margin), 4) if margin is not None else None,
                "max_tested_cost_frac": float(max(grid))}
    except Exception as e:  # noqa: BLE001
        return {"breakeven_cost_frac": None, "configured_cost_frac": cfg, "margin": None,
                "max_tested_cost_frac": cfg, "note": f"cost sensitivity unavailable: {e}"}


def run_backtest(name: str, net_per_trade: pd.Series, positions: pd.Series,
                 bars_per_year: float, log: bool = True, *,
                 payoff: pd.Series | None = None,
                 position_signed: pd.Series | None = None,
                 cost_frac: float | None = None,
                 wf_frame: pd.DataFrame | None = None,
                 family: str | None = None,
                 generation_source: str = "paper",
                 search_cohort_id: str | None = None,
                 search_denominator: int | None = None,
                 preregistered_single_cohort: bool = False,
                 regime_schemes: dict | None = None,
                 declared_regime: dict | None = None) -> dict:
    """Score a strategy's per-trade net returns under full S4 discipline, plus the
    empirical robustness layer (bootstrap CI, permutation, walk-forward, capacity CI).

    net_per_trade   : per-trade net return (fraction of vega notional), time-indexed
    positions       : per-trade position SIZE (turnover/capacity)
    payoff          : per-trade raw payoff per unit (e.g. realized_vol - implied_vol)
    position_signed : per-trade SIGNED position (the directional bet) for permutation
    cost_frac       : round-trip cost fraction (needed to reconstruct net in permutation)
    wf_frame        : DataFrame[signal, fut_rv, iv] for walk-forward re-fit
    regime_schemes  : dict[name -> date->label pd.Series] of PRE-REGISTERED, point-in-time
                      MARKET-regime labels (vol/trend from penrose.regime). Passed to the
                      kill-lens so it also catches edges concentrated in one vol/trend regime.
    declared_regime : optional pre-registered claim scope, e.g.
                      {"scheme": "vol_regime", "label": "high_vol"}.

    These optional inputs are keyword-only: existing callers are unaffected; permutation,
    walk-forward, and the market-regime lens simply activate when the module supplies them.
    """
    net = net_per_trade.dropna()
    n = len(net)
    if n < config.DSR_DECISION["min_oos_bars"]:        # need enough total trades to carve an OOS (A-005)
        return {"n": n, "note": "insufficient_trades", "dsr": 0.0, "tail": R.tail_metrics(net)}

    # time split FIRST — the final 0.20 is the LOCKED, single-use HOLDOUT. It must NEVER
    # reach a kill gate (3-fold, regime); only final_holdout_eval may read net[o:]. (A-002)
    i = int(n * IS_FRAC)
    o = int(n * (IS_FRAC + OOS_FRAC))
    oos = net.iloc[i:o].values
    seen = net.iloc[:o]               # IS+OOS only — the only window any gate may see
    full = net.values                 # descriptive metrics ONLY (full_sharpe/total_net), never a gate

    n_trials, sr_var = _trial_stats(
        family, name, registered_trials=search_denominator,
        generation_source=generation_source,
        search_cohort_id=search_cohort_id,
        preregistered_single_cohort=preregistered_single_cohort)  # current + registered search
    # Regime kill-lens (Punisher) on the NON-HOLDOUT window. Calendar buckets always; the
    # pre-registered point-in-time vol/trend labels (extra_schemes) join when supplied so the
    # lens also catches edges concentrated in one MARKET regime. Each populated partition is an
    # extra "look", so it inflates the DSR trial count before deflation (incl. the vol/trend buckets).
    # A provided-series statistic earns the regime-partition break only when the source explicitly
    # declares a single pre-registered pooled cohort; classifier routing alone is not enough.
    if generation_source == "provided_series_statistic" and preregistered_single_cohort:
        regime = {"n_partitions": 0, "fragile": False, "provided_series_statistic": True}
    else:
        regime = R.regime_split(
            seen, bars_per_year, extra_schemes=regime_schemes, declared=declared_regime)
    tail = R.tail_metrics(seen)
    n_trials += int(regime.get("n_partitions", 0))

    mean, sd = float(oos.mean()), float(oos.std(ddof=1)) if len(oos) > 1 else 0.0

    out = {
        "n": n,
        "n_oos": len(oos),
        "bars_per_year": bars_per_year,        # needed for power/MDE in the verdict (power-aware labels)
        "n_trials": n_trials,
        "regime": regime,
        "tail": tail,
        "dsr": round(H.deflated_sharpe(oos, n_trials, sr_var), 4) if len(oos) >= 20 else 0.0,
        "psr": round(H.probabilistic_sharpe(oos), 4) if len(oos) >= 20 else 0.0,
        "oos_sharpe": round(H.sharpe(oos, bars_per_year), 3) if len(oos) >= 20 else None,
        "is_sharpe": round(H.sharpe(net.iloc[:i].values, bars_per_year), 3) if i >= 20 else None,  # B2
        "full_sharpe": round(H.sharpe(full, bars_per_year), 3) if n >= 20 else None,
        "per_trade_sharpe": round(mean / sd, 4) if sd else None,
        "edge_t": round(mean / (sd / math.sqrt(len(oos))), 3) if sd and len(oos) else None,
        "avg_net_edge": round(mean, 5),
        "total_net": round(float(full.sum()), 4),
        "three_fold": _three_fold(seen.values),   # non-holdout only (A-002)
    }

    # capacity: reuse harness linear-impact model — on the NON-HOLDOUT window only, so the
    # locked holdout never leaks into capacity / ann_ret either (B-001; A-002 made absolute).
    if positions.index.equals(net.index):
        pos_seen = positions.iloc[:len(seen)].copy()
        pos_seen.index = seen.index
    else:
        pos_seen = positions.reindex(seen.index)
    pos_df = pd.DataFrame({"VOL:BTC": pos_seen.fillna(0.0)})
    ann_ret = float(seen.values.mean()) * bars_per_year
    out["ann_ret"] = round(ann_ret, 4)
    out["capacity_usd"] = H._capacity_usd(pos_df, ann_ret, bars_per_year,
                                          config.IMPACT_COEF_BPS_PER_1M)
    out["cost_sensitivity"] = _cost_sensitivity(
        net, positions, bars_per_year, cost_frac, n_trials, sr_var,
        payoff=payoff, position_signed=position_signed)

    # --- empirical robustness layer (Monte-Carlo + walk-forward) ----------- #
    # Bootstrap the OOS edge: this is what hardens borderline verdicts at the
    # small-sample / fat-tail regime where DSR/PSR are least reliable.
    if len(oos) >= 10:
        out["bootstrap"] = R.block_bootstrap(oos, bars_per_year, **config.BOOTSTRAP)
    out["capacity_ci"] = R.capacity_ci(seen.values, pos_df, bars_per_year,
                                       config.IMPACT_COEF_BPS_PER_1M)
    # Permutation (data-snooping) needs the signal->payoff pairing from the module.
    if payoff is not None and position_signed is not None and cost_frac is not None:
        pay = payoff.reindex(net.index) if hasattr(payoff, "reindex") else pd.Series(payoff)
        ps = position_signed.reindex(net.index) if hasattr(position_signed, "reindex") else pd.Series(position_signed)
        i, o = int(n * IS_FRAC), int(n * (IS_FRAC + OOS_FRAC))
        out["permutation"] = R.permutation_test(ps.values[i:o], pay.values[i:o],
                                                cost_frac, **config.PERMUTATION)
    # Walk-forward re-fit (vs the single static split) when the module hands us the frame.
    # C-004: TRUNCATE the frame to the non-holdout window first — the B-007 walk-forward KILL
    # gate must never read net[o:] (the locked holdout), same invariant as 3-fold/regime (A-002).
    if wf_frame is not None:
        wf_seen = wf_frame.iloc[:int(len(wf_frame) * (IS_FRAC + OOS_FRAC))]
        out["walk_forward"] = R.walk_forward_vol(
            wf_seen, cost_frac=cost_frac or 0.0008, **config.WALK_FORWARD)
    cpcv_keys = {"n_groups", "k_test", "embargo_frac", "max_combos", "seed"}
    out["cpcv"] = R.cpcv(
        seen, bars_per_year,
        **{k: v for k, v in config.CPCV.items() if k in cpcv_keys},
    )

    if log:
        _append_ledger(
            name, out, family,
            generation_source=generation_source,
            search_cohort_id=search_cohort_id,
            search_denominator=search_denominator,
        )
    return out


def final_holdout_eval(name: str, net_per_trade: pd.Series, bars_per_year: float,
                       force: bool = False) -> dict:
    """Single-use holdout (S4). First call burns the lock; later calls refuse —
    a holdout peeked at twice is just a slow second OOS set.

    The lock path honors $PENROSE_HOLDOUT_LOCK if set, so CALIBRATION/REFEREE harnesses (which
    force-consult hundreds of synthetic holdouts) point it at an isolated temp file and never
    pollute penrose's PRODUCTION per-claim `.holdout_burned.*.lock` state (state-safety fix;
    the prod default is used by the real pipeline run.py)."""
    # Dream triage is categorically forbidden from peeking at a shared holdout. Check this before
    # `force`: even an accidental force=True call cannot burn or read the lock in read-only mode.
    if os.environ.get("PENROSE_HOLDOUT_MODE", "").lower() == "readonly":
        return {"refused": True, "reason": "holdout disabled: read-only dream triage"}
    lock = _claim_holdout_lock(name)
    claimed = False
    if not force:
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return {"refused": True, "reason": f"holdout already burned for this claim (lock {lock.name})"}
        else:
            with os.fdopen(fd, "w") as f:
                f.write(f"strategy={name} evaluation=in-progress")
            claimed = True
    net = net_per_trade.dropna()
    n = len(net)
    o = int(n * (IS_FRAC + OOS_FRAC))
    hold = net.iloc[o:].values
    if len(hold) < 20:
        if claimed:
            lock.unlink(missing_ok=True)
        return {"refused": False, "note": "holdout too small", "nbars": len(hold)}
    try:
        res = {"holdout_sharpe": round(H.sharpe(hold, bars_per_year), 3),
               "holdout_psr": round(H.probabilistic_sharpe(hold), 4), "nbars": len(hold)}
        digest = hashlib.sha256(repr(res).encode()).hexdigest()[:16]
        burned_at = pd.Timestamp.now(tz="UTC").isoformat()
        lock.write_text(f"strategy={name} bars={len(hold)} digest={digest} burned_at={burned_at}")
    except Exception as e:  # noqa: BLE001
        if claimed:
            lock.unlink(missing_ok=True)
        return {"refused": True, "reason": f"holdout evaluation failed: {type(e).__name__}: {e}"}
    return res
