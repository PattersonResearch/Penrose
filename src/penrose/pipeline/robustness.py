"""Empirical robustness tests for P7 (the Monte-Carlo + walk-forward layer).

penrose's analytic guards (DSR, PSR) are asymptotic approximations that are least
reliable in exactly the regime we now operate in: small samples (tens of trades)
and fat tails (vol / PM / crypto returns). These functions add the empirical,
assumption-light complement:

  * block_bootstrap   — stationary-block resample of the OOS per-trade returns.
                        Gives a Sharpe / edge confidence interval and a drawdown
                        distribution without assuming normality. The verdict can
                        then KILL when the edge CI includes zero.
  * permutation_test  — shuffles the signal-vs-payoff alignment to build the null
                        of "no predictive relationship" (honest data-snooping
                        guard; complements DSR's crude trial-count deflation).
  * walk_forward_vol  — rolling / anchored re-fit of the z-score vol strategy,
                        instead of one static IS/OOS cut. Catches parameter drift
                        the single split misses.
  * capacity_ci       — bootstrap CI on capacity, so it is a range not a point.

Pure numpy / pandas; no harness coupling. Deterministic given a seed.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .. import config


# --------------------------------------------------------------------------- #
def _stationary_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths,
    wrap-around). Preserves serial dependence; reduces to iid resampling at block=1."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / max(1, block)
    cur = int(rng.integers(n))
    for t in range(n):
        idx[t] = cur
        if rng.random() < p:
            cur = int(rng.integers(n))
        else:
            cur = (cur + 1) % n
    return idx


def _max_drawdown(r: np.ndarray) -> float:
    """Max peak-to-trough of the additive equity curve (positive magnitude)."""
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def tail_metrics(net) -> dict:
    """Left-tail diagnostics for bounded-up / unbounded-down payoff shapes.

    The verdict gate reads these only when config.TAIL_RISK_GATE["enabled"] is
    true. This function is intentionally fail-open and never raises into P7.
    """
    out = {
        "skew": None,
        "cvar_5": None,
        "cvar_95": None,
        "tail_ratio": None,
        "max_loss": None,
        "max_gain": None,
        "worst_vs_typical": None,
        "asymmetric": False,
    }
    try:
        x = np.asarray(pd.Series(net).dropna(), dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n == 0:
            return out

        out["max_loss"] = float(np.min(x))
        out["max_gain"] = float(np.max(x))

        k = max(1, int(math.ceil(0.05 * n)))
        xs = np.sort(x)
        cvar_5 = float(np.mean(xs[:k]))
        cvar_95 = float(np.mean(xs[-k:]))
        out["cvar_5"] = cvar_5
        out["cvar_95"] = cvar_95
        if cvar_95 > 0:
            out["tail_ratio"] = float(abs(cvar_5) / cvar_95)

        gains = x[x > 0]
        if len(gains):
            p95_gain = float(np.percentile(gains, 95))
            if p95_gain > 0:
                out["worst_vs_typical"] = float(abs(out["max_loss"]) / p95_gain)

        if n < 20:
            return out
        mu = float(np.mean(x))
        sd = float(np.std(x))
        # Near-constant series (sd ~ float noise relative to scale) have an UNDEFINED skew; computing
        # ((x-mu)/sd)**3 there yields garbage (and a divide overflow). Treat as no-skew, like sd==0.
        scale = max(abs(mu), float(np.max(np.abs(x))), 1e-12)
        if not np.isfinite(sd) or sd <= 0.0 or sd < 1e-9 * scale:
            return out
        skew = float(np.mean(((x - mu) / sd) ** 3))
        out["skew"] = skew

        trg = getattr(
            config,
            "TAIL_RISK_GATE",
            {"max_skew": -0.5, "severe_skew": -3.0, "min_tail_ratio": 3.0},
        )
        tail_ratio = out["tail_ratio"]
        severe_skew = float(trg.get("severe_skew", -3.0))
        max_skew = float(trg.get("max_skew", -0.5))
        min_tail_ratio = float(trg.get("min_tail_ratio", 3.0))
        severe_min_n = int(trg.get("severe_min_n", 40))
        # J-3: the severe-skew arm flags on skew ALONE (no tail_ratio corroboration), so it needs
        # an adequate sample -- sample skew is unstable at small n (SE ~ sqrt(6/n) ~ 0.55 at n=20).
        # Require severe_min_n before trusting a skew-only widow-maker flag; below it, only the
        # corroborated (skew AND tail_ratio) arm applies.
        out["asymmetric"] = bool(
            (skew <= severe_skew and n >= severe_min_n)
            or (
                skew <= max_skew
                and tail_ratio is not None
                and tail_ratio >= min_tail_ratio
            )
        )
        return out
    except Exception:  # noqa: BLE001
        return out


# Capped sentinel for a deterministic (zero-variance) edge. A real annualized
# Sharpe never legitimately reaches this, so it reads as "deterministic edge"
# downstream while staying finite (no inf/NaN to crash percentiles / JSON).
_SHARPE_DETERMINISTIC = 1.0e6
_CPCV_MIN_PER_GROUP = 5


def _sharpe(r: np.ndarray, bpy: float) -> float:
    """Annualized Sharpe of a per-bar return series.

    std==0 is NOT "no edge": a deterministic positive series has a real (infinite)
    Sharpe. We distinguish the cases so a constant-positive regime is not silently
    treated as zero-edge by the fragility / bootstrap lenses:
      * len<2            -> 0.0 (undefined)
      * std==0, mean>0   -> +capped sentinel  (deterministic positive edge)
      * std==0, mean<0   -> -capped sentinel  (deterministic loss)
      * std==0, mean==0  -> 0.0 (genuinely no edge)
    """
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd > 0:
        return float(r.mean() / sd * math.sqrt(bpy))
    mean = float(r.mean())
    if mean > 0:
        return _SHARPE_DETERMINISTIC
    if mean < 0:
        return -_SHARPE_DETERMINISTIC
    return 0.0


# --------------------------------------------------------------------------- #
def cpcv(net, bars_per_year: float, *, n_groups: int = 8, k_test: int = 2,
         embargo_frac: float = 0.01, max_combos: int = 200, seed: int = 0) -> dict:
    """Combinatorial purged CV on the already non-holdout return window.

    This is a single-strategy overfitting lens: it reports the distribution of
    purged combinatorial OOS Sharpes and the share of paths with non-positive
    OOS Sharpe. It never fabricates a CSCV/PBO value; `pbo` is reserved for a
    real >=2-config family path.
    """
    try:
        s = pd.Series(net).dropna().astype(float)
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "reason": f"invalid input: {type(e).__name__}: {e}"}
    n = len(s)
    try:
        n_groups = int(n_groups)
        k_test = int(k_test)
        max_combos = int(max_combos)
        embargo_frac = float(embargo_frac)
    except (TypeError, ValueError):
        return {"ran": False, "reason": "invalid CPCV configuration"}
    if n_groups < 2 or k_test < 1 or k_test >= n_groups:
        return {"ran": False, "reason": "invalid CPCV group/test configuration"}
    if embargo_frac <= 0.0:
        return {"ran": False, "reason": "CPCV requires a positive embargo fraction"}
    if max_combos < 1:
        return {"ran": False, "reason": "CPCV max_combos must be positive"}
    if n < n_groups * _CPCV_MIN_PER_GROUP:
        return {"ran": False, "reason": f"too few observations for {n_groups} CPCV groups"}
    vals = s.to_numpy(dtype=float)
    if float(np.nanstd(vals, ddof=1)) == 0.0:
        return {"ran": False, "reason": "zero-variance series cannot support CPCV"}

    groups = [g.astype(int) for g in np.array_split(np.arange(n), n_groups)]
    if any(len(g) < _CPCV_MIN_PER_GROUP for g in groups):
        return {"ran": False, "reason": "CPCV groups too small after partitioning"}
    all_combos = list(itertools.combinations(range(n_groups), k_test))
    combos_total = len(all_combos)
    min_paths = int(getattr(config, "CPCV", {}).get("min_paths", 1))
    if combos_total < min_paths:
        return {"ran": False, "reason": f"too few CPCV paths ({combos_total} < {min_paths})"}
    if combos_total > max_combos:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(combos_total, size=max_combos, replace=False))
        combos = [all_combos[int(i)] for i in chosen]
        subsampled = True
    else:
        combos = all_combos
        subsampled = False

    embargo_n = max(1, int(math.ceil(embargo_frac * n)))
    sharpes: list[float] = []
    split_meta: list[dict] = []
    for combo in combos:
        test_idx = np.concatenate([groups[g] for g in combo])
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        purged = set()
        embargoed = set()
        for g in combo:
            block = groups[g]
            start, end = int(block[0]), int(block[-1])
            for p in (start - 1, end + 1):
                if 0 <= p < n and train_mask[p]:
                    purged.add(p)
            emb_start = end + 1
            emb_end = min(n, end + 1 + embargo_n)
            for p in range(emb_start, emb_end):
                if train_mask[p]:
                    embargoed.add(p)
        if purged:
            train_mask[list(purged)] = False
        if embargoed:
            train_mask[list(embargoed)] = False
        if not np.any(train_mask):
            return {"ran": False, "reason": "CPCV purge/embargo removed all training observations"}
        sh = _sharpe(vals[np.sort(test_idx)], bars_per_year)
        sharpes.append(float(sh))
        split_meta.append({
            "test_groups": [int(g) for g in combo],
            "test_n": int(len(test_idx)),
            "train_n_after_purge_embargo": int(train_mask.sum()),
            "purged_n": int(len(purged)),
            "embargoed_n": int(len(embargoed)),
        })

    arr = np.asarray(sharpes, dtype=float)
    q25, q75 = np.percentile(arr, [25, 75])
    is_sharpe = _sharpe(vals, bars_per_year)
    prob_oos_loss = float((arr <= 0.0).mean())
    threshold = float(getattr(config, "CPCV", {}).get("overfit_prob_kill", 0.50))
    return {
        "ran": True,
        "n_paths": int(len(combos)),
        "combos_used": int(len(combos)),
        "combos_total": int(combos_total),
        "subsampled": bool(subsampled),
        "seed": int(seed),
        "n_groups": int(n_groups),
        "k_test": int(k_test),
        "embargo_frac": float(embargo_frac),
        "embargo_n": int(embargo_n),
        "purge": "one_observation_adjacent_to_each_test_block",
        "oos_sharpes": [round(float(x), 3) for x in arr],
        "median_oos_sharpe": round(float(np.median(arr)), 3),
        "iqr_oos_sharpe": round(float(q75 - q25), 3),
        "prob_oos_loss": round(prob_oos_loss, 4),
        "is_sharpe": round(float(is_sharpe), 3),
        "haircut": round(float(np.median(arr) / is_sharpe), 4) if is_sharpe > 0 else None,
        "pbo": None,
        "pbo_note": "single-strategy CPCV; PBO requires >=2 candidate configs",
        "overfit": bool(prob_oos_loss >= threshold),
        "splits": split_meta,
    }


# --------------------------------------------------------------------------- #
def block_bootstrap(net, bars_per_year: float, n_boot: int = 2000,
                    ci: float = 0.90, block: int | None = None,
                    seed: int = 0) -> dict:
    """Stationary-block bootstrap of per-trade net returns.

    Returns the edge (mean) and Sharpe CIs, P(edge>0), and a drawdown
    distribution. `edge_ci_includes_zero` is the verdict-hardening flag: if the
    CI straddles zero, the OOS edge is not distinguishable from luck.
    """
    net = np.asarray(net, dtype=float)
    net = net[~np.isnan(net)]
    n = len(net)
    if n < 10:
        return {"note": "too few trades for bootstrap", "n": n,
                "edge_ci_includes_zero": None}
    b = block or max(1, round(math.sqrt(n)))
    rng = np.random.default_rng(seed)
    sh = np.empty(n_boot)
    ed = np.empty(n_boot)
    dd = np.empty(n_boot)
    for k in range(n_boot):
        s = net[_stationary_indices(n, b, rng)]
        sh[k] = _sharpe(s, bars_per_year)
        ed[k] = s.mean()
        dd[k] = _max_drawdown(s)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    edge_lo, edge_hi = np.percentile(ed, [lo_q, hi_q])
    return {
        "n_boot": n_boot, "block": b, "ci": ci,
        "edge_ci": [round(float(edge_lo), 5), round(float(edge_hi), 5)],
        "edge_ci_includes_zero": bool(edge_lo <= 0.0 <= edge_hi),
        "p_edge_gt0": round(float((ed > 0).mean()), 4),
        "sharpe_ci": [round(float(np.percentile(sh, lo_q)), 3),
                      round(float(np.percentile(sh, hi_q)), 3)],
        "p_sharpe_gt0": round(float((sh > 0).mean()), 4),
        "boot_psr": round(float((sh > 0).mean()), 4),     # empirical analog of PSR
        "drawdown_median": round(float(np.median(dd)), 5),
        "drawdown_p95": round(float(np.percentile(dd, 95)), 5),
    }


# --------------------------------------------------------------------------- #
def permutation_test(position_signed, payoff, cost_frac: float,
                     n_perm: int = 2000, seed: int = 0) -> dict:
    """Data-snooping null: is the signal->payoff ALIGNMENT better than chance?

    This is a test of SIGNAL-PAYOFF ALIGNMENT SIGNIFICANCE, NOT of post-cost
    profitability. Profitability (after costs) is the job of DSR / edge / the
    bootstrap edge CI — not of this test.

    position_signed : per-trade signed position (the signal's directional bet)
    payoff          : per-trade raw payoff per unit (e.g. realized_vol - implied_vol)

    We measure alignment = mean(position * payoff). The per-trade cost term
    (|position|*cost_frac) is IDENTICAL under every payoff permutation, so it
    cancels exactly and cannot move the p-value — including it only created the
    false impression that this guarded profitability. We therefore omit it.
    `cost_frac` is accepted for call-site compatibility but is intentionally
    unused here. We shuffle payoff against position to break any real
    relationship and recompute the mean alignment; p = fraction of shuffles that
    match or beat the observed alignment. High p => the signal's direction is
    indistinguishable from betting randomly relative to outcomes.
    """
    del cost_frac  # intentionally unused: cost is inert under permutation (see docstring)
    pos = np.asarray(position_signed, dtype=float)
    pay = np.asarray(payoff, dtype=float)
    m = ~(np.isnan(pos) | np.isnan(pay))
    pos, pay = pos[m], pay[m]
    n = len(pos)
    if n < 10:
        return {"note": "too few trades for permutation", "n": n, "p_value": None}
    observed = float((pos * pay).mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        align = float((pos * pay[rng.permutation(n)]).mean())
        if align >= observed:
            ge += 1
    return {"n_perm": n_perm,
            "tests": "signal-payoff alignment significance (NOT post-cost "
                     "profitability; DSR/edge/bootstrap cover that)",
            "observed_alignment": round(observed, 6),
            "p_value": round((ge + 1) / (n_perm + 1), 4)}   # +1 smoothing


# --------------------------------------------------------------------------- #
def capacity_ci(net, positions: pd.DataFrame, bars_per_year: float,
                impact_bps_per_1m: float, n_boot: int = 500, ci: float = 0.90,
                seed: int = 1) -> dict | None:
    """Bootstrap CI on capacity (notional where linear impact erases the edge),
    so the principle records a range, not a single fragile point."""
    net = np.asarray(net, dtype=float)
    net = net[~np.isnan(net)]
    n = len(net)
    if n < 10 or impact_bps_per_1m <= 0:
        return None
    turn_bar = positions.diff().abs().sum(axis=1).dropna().mean()
    if not turn_bar or turn_bar <= 0:
        return None
    impact_per_dollar = (impact_bps_per_1m / 1e4) / 1e6
    turn_ann = turn_bar * bars_per_year
    b = max(1, round(math.sqrt(n)))
    rng = np.random.default_rng(seed)
    caps = []
    for _ in range(n_boot):
        ann = net[_stationary_indices(n, b, rng)].mean() * bars_per_year
        if ann > 0:
            caps.append(ann / (turn_ann * impact_per_dollar))
    p_positive_edge = round(len(caps) / n_boot, 3)
    if not caps:
        return {"note": "edge non-positive in ALL resamples; capacity undefined",
                "p_positive_edge": p_positive_edge,
                "conditional_on_positive_edge": True}
    # Negligible turnover makes modeled linear-impact capacity diverge to +inf. Drop the
    # non-finite resamples and, if none remain, report capacity as undefined rather than
    # crashing on int(inf) — a low-turnover strategy must still get a graceful verdict.
    caps = [c for c in caps if math.isfinite(c)]
    if not caps:
        return {"note": "capacity undefined: turnover negligible, modeled capacity unbounded",
                "p_positive_edge": p_positive_edge,
                "conditional_on_positive_edge": True}
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {"ci": ci, "p_positive_edge": p_positive_edge,
           "capacity_lo": int(round(np.percentile(caps, lo_q), -3)),
           "capacity_median": int(round(np.median(caps), -3)),
           "capacity_hi": int(round(np.percentile(caps, hi_q), -3))}
    # The CI is computed ONLY over positive-edge resamples. When a meaningful
    # fraction of resamples had no capacity at all (negative/zero edge), the
    # reported range is CONDITIONAL on the edge being positive and overstates
    # robustness. Flag it rather than fabricating capacity for those resamples.
    if p_positive_edge < 0.90:
        out["conditional_on_positive_edge"] = True
        out["note"] = (f"capacity CI is CONDITIONAL on a positive edge: only "
                       f"{p_positive_edge:.0%} of resamples had any capacity "
                       f"(the rest had non-positive edge and zero capacity)")
    else:
        out["conditional_on_positive_edge"] = False
    return out


# --------------------------------------------------------------------------- #
def walk_forward_vol(frame: pd.DataFrame, hold_days: int = 5, cost_frac: float = 0.0008,
                     n_windows: int = 4, scheme: str = "anchored",
                     is_min: float = 0.30) -> dict:
    """Rolling / anchored walk-forward of the z-score vol strategy.

    Mirrors the macro_vol module's logic (standardize signal on the TRAIN window,
    trade non-overlapping `hold_days` entries on the TEST window), but re-fits the
    (mu, sigma) standardization as the window rolls instead of fitting once. Frame
    columns: `signal`, `fut_rv` (realized vol over the next hold_days), `iv`
    (implied vol at entry). Returns per-window and aggregate OOS Sharpe.
    """
    cols = {"signal", "fut_rv", "iv"}
    if not cols.issubset(frame.columns):
        return {"note": f"frame needs columns {cols}", "consistent": False}
    df = frame.dropna(subset=list(cols)).reset_index(drop=True)
    n = len(df)
    entries = np.arange(0, n - hold_days, hold_days)
    cut0 = int(n * is_min)
    test_entries = [int(e) for e in entries if e >= cut0]
    if len(test_entries) < n_windows * 3:
        return {"note": "too few non-overlapping trades for walk-forward",
                "n_trades": len(test_entries), "consistent": False}
    sig = df["signal"].to_numpy()
    rv = df["fut_rv"].to_numpy()
    iv = df["iv"].to_numpy()
    segs = [s for s in np.array_split(test_entries, n_windows) if len(s)]
    oos, per = [], []
    for seg in segs:
        train_end = int(seg[0])
        if scheme == "rolling":
            train_start = max(0, train_end - cut0)
        else:                                   # anchored / expanding
            train_start = 0
        tr = sig[train_start:train_end]
        if len(tr) < 5 or tr.std(ddof=1) == 0:
            per.append(None); continue
        mu, sd = tr.mean(), tr.std(ddof=1)
        seg_net = []
        for e in seg:
            z = float(np.clip((sig[e] - mu) / sd, -1, 1))
            seg_net.append(z * (rv[e] - iv[e]) - abs(z) * cost_frac)
        seg_net = np.asarray(seg_net)
        oos.extend(seg_net.tolist())
        per.append(round(_sharpe(seg_net, 365 / hold_days), 3) if len(seg_net) > 1 else None)
    oos = np.asarray(oos)
    valid = [p for p in per if p is not None]
    return {
        "scheme": scheme, "n_windows": len(segs), "hold_days": hold_days,
        "per_window_sharpe": per,
        "oos_sharpe": round(_sharpe(oos, 365 / hold_days), 3) if len(oos) > 1 else None,
        "consistent": bool(valid) and all(p > 0 for p in valid),
        "n_trades": int(len(oos)),
    }


# --------------------------------------------------------------------------- #
def _normalize_declared_regime(declared: dict | None) -> dict | None:
    if declared is None:
        return None
    try:
        scheme = str(declared.get("scheme", "")).strip().lower()
        label = str(declared.get("label", "")).strip().lower()
    except AttributeError:
        return None
    if not scheme or not label:
        return None
    return {"scheme": scheme, "label": label}


# --------------------------------------------------------------------------- #
def regime_split(net, bars_per_year: float, min_bucket: int = 8, extra_schemes=None,
                 declared: dict | None = None) -> dict:
    """STRICT regime kill-lens (Punisher's lesson: most bots die to regime, not signal).

    Partition the per-trade net series by EXOGENOUS calendar regime (weekday/weekend,
    day-of-week — derived from the trade timestamps, never from the returns) and ask
    one question: does the edge survive when its single best regime is removed? If a
    positive overall edge collapses to <= 0 without one bucket, the edge is concentrated
    in a regime and is fragile -> a KILL.

    `extra_schemes` (optional): dict[name -> date->label pd.Series] of PRE-REGISTERED, point-in-time
    MARKET-regime labels (e.g. vol_regime / trend_regime from penrose.regime). Aligned to the trade
    dates and added as partition schemes — so the lens also catches edges concentrated in one VOL or
    TREND regime, a fragility the calendar-only partition is blind to. These labels are trailing-only
    (no look-ahead) and exogenous, so partitioning by them is not data-snooping; each populated bucket
    still inflates n_partitions (the DSR trial count), so surfacing them is never a free pass.

    This is deliberately falsification-only: it never SELECTS a regime to trade (that
    would be data snooping — exactly how 500 backtested bots died live). `n_partitions`
    is the count of populated buckets examined; P7 folds it into the DSR trial count so
    surfacing per-regime Sharpes can never become a free pass.

    Granularity scales with the data: daily series -> weekday/weekend + day-of-week.
    Intraday session regimes activate automatically once the index carries intraday
    timestamps (more buckets, same logic).
    """
    declared_norm = _normalize_declared_regime(declared)
    s = pd.Series(net).dropna()
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < max(20, 3 * min_bucket):
        out = {"applicable": False, "n_partitions": 0, "fragile": False}
        if declared_norm is not None:
            out.update({"declared_regime": declared_norm, "declared_present": False,
                        "adheres": False,
                        "adherence": {"trade_share": 0.0, "edge_share": 0.0,
                                      "top_edge_label": None},
                        "declared_note": "regime adherence unavailable: insufficient dated trades"})
        return out
    vals = s.values.astype(float)
    overall = float(vals.mean())
    dow = np.asarray(s.index.dayofweek)
    has_intraday = bool((np.asarray(s.index.hour) != 0).any())
    schemes_def = {
        "weekend": np.where(dow >= 5, "weekend", "weekday"),
        "day_of_week": np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[dow],
    }
    if has_intraday:                              # only meaningful with intraday stamps
        hr = np.asarray(s.index.hour)
        schemes_def["session"] = np.select(
            [hr < 8, hr < 16], ["asia", "europe"], default="us")
    # Pre-registered point-in-time MARKET-regime labels (vol/trend), aligned to the trade dates.
    # Trades whose date has no label become "unknown" (its own bucket; min_bucket gates it out if thin).
    # H-001: normalize BOTH the trade index and each label index to tz-aware UTC before reindex. A
    # tz-naive trade index vs tz-aware UTC labels (or vice-versa) reindexes to ALL-NaN -> every trade
    # falls to "unknown" -> the vol/trend partition is silently dropped (degraded falsification, no
    # error). Normalizing both first makes the lens fire whenever the dates actually overlap.
    def _utc(idx):
        idx = pd.DatetimeIndex(idx)
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    trade_idx = _utc(s.index)
    for name, lab_series in (extra_schemes or {}).items():
        try:
            lab = pd.Series(lab_series)
            lab.index = _utc(lab.index)
            aligned = lab.reindex(trade_idx)
            schemes_def[name] = aligned.where(aligned.notna(), "unknown").astype(str).values
        except Exception:  # noqa: BLE001
            continue

    # Fragile if removing the single best regime leaves < SURVIVAL_FRAC of the per-trade
    # edge (i.e. the edge was carried by one regime). Strict, but trial-count deflation
    # is a second independent layer, so this only needs to catch clear concentration.
    SURVIVAL_FRAC = 0.25
    schemes, n_partitions, fragile, reason = {}, 0, False, None
    fragility_p = {}
    declared_present = False
    declared_exempted = False
    declared_note = None
    adherence = None
    for name, labels in schemes_def.items():
        buckets = {}
        for lab in pd.unique(labels):
            r = vals[labels == lab]
            # NOTE: do NOT require std>0. A zero-variance bucket (e.g. a weekend
            # bucket of constant positive returns) is EXACTLY the concentration
            # this fragility lens exists to catch; dropping it made that
            # concentration invisible. _sharpe handles std==0 via a capped
            # sentinel, so no div-by-zero arises.
            if len(r) >= min_bucket:
                buckets[str(lab)] = {"n": int(len(r)),
                                     "sharpe": round(_sharpe(r, bars_per_year), 3),
                                     "edge": round(float(r.mean()), 6)}
        declared_scheme = declared_norm is not None and name.lower() == declared_norm["scheme"]
        if declared_scheme:
            labels_norm = np.asarray([str(x).strip().lower() for x in labels])
            declared_mask = labels_norm == declared_norm["label"]
            declared_n = int(declared_mask.sum())
            declared_present = declared_n >= min_bucket
            contrib_all = {str(lab).strip().lower(): float(vals[labels_norm == str(lab).strip().lower()].sum())
                           for lab in pd.unique(labels_norm)}
            top_edge_label = max(contrib_all, key=contrib_all.get) if contrib_all else None
            total_abs_edge = float(np.abs(vals).sum())
            declared_edge = float(vals[declared_mask].sum()) if declared_n else 0.0
            adherence = {
                "trade_share": round(float(declared_n / len(vals)), 4) if len(vals) else 0.0,
                "edge_share": round(float(declared_edge / total_abs_edge), 4) if total_abs_edge else 0.0,
                "declared_edge": round(declared_edge, 6),
                "top_edge_label": top_edge_label,
                "min_trade_share": float(getattr(config, "REGIME_ADHERENCE_MIN", 0.60)),
            }
        if len(buckets) >= 2:
            schemes[name] = buckets
            n_partitions += len(buckets)
            if overall > 0 and not fragile:       # drop-the-best fragility test
                contrib = {lab: float(vals[labels == lab].sum()) for lab in buckets}
                best = max(contrib, key=contrib.get)
                rest = vals[labels != best]
                rest_edge = float(rest.mean()) if len(rest) >= min_bucket else 0.0
                if rest_edge <= SURVIVAL_FRAC * overall:
                    if (declared_scheme
                            and str(best).strip().lower() == declared_norm["label"]):
                        declared_exempted = True
                        declared_note = (f"declared regime concentration exempted for "
                                         f"{name}={best}; other schemes still tested")
                        continue
                    rfg = getattr(config, "REGIME_FRAGILITY", {"n_perm": 0, "p_kill": 1.0, "seed": 0})
                    n_perm = int(rfg.get("n_perm", 0) or 0)
                    p_kill = float(rfg.get("p_kill", 0.05))
                    p = 0.0
                    if len(vals) >= 20 and n_perm > 0:
                        rng = np.random.default_rng(int(rfg.get("seed", 0)))
                        hits = 0
                        for _ in range(n_perm):
                            perm_labels = rng.permutation(labels)
                            perm_buckets = [lab for lab in pd.unique(perm_labels)
                                            if int((perm_labels == lab).sum()) >= min_bucket]
                            if len(perm_buckets) < 2:
                                continue
                            perm_contrib = {lab: float(vals[perm_labels == lab].sum())
                                            for lab in perm_buckets}
                            perm_best = max(perm_contrib, key=perm_contrib.get)
                            perm_rest = vals[perm_labels != perm_best]
                            perm_rest_edge = (
                                float(perm_rest.mean()) if len(perm_rest) >= min_bucket else 0.0
                            )
                            if perm_rest_edge <= rest_edge:
                                hits += 1
                        p = hits / n_perm
                        fragility_p[name] = round(float(p), 4)
                    if len(vals) >= 20 and n_perm > 0 and p >= p_kill:
                        continue
                    fragile = True
                    reason = (f"edge concentrated in '{best}' ({name}): only "
                              f"{rest_edge:.5f}/trade survives without it "
                              f"({rest_edge / overall * 100:.0f}% of {overall:.5f})"
                              + (f" (perm p={p:.4f} < {p_kill})"
                                 if len(vals) >= 20 and n_perm > 0 else ""))
    out = {"applicable": True, "overall_edge": round(overall, 6),
           "schemes": schemes, "n_partitions": int(n_partitions),
           "fragile": bool(fragile), "fragile_reason": reason}
    if fragility_p:
        out["fragility_p"] = fragility_p
    if declared_norm is not None:
        if adherence is None:
            adherence = {"trade_share": 0.0, "edge_share": 0.0, "declared_edge": 0.0,
                         "top_edge_label": None,
                         "min_trade_share": float(getattr(config, "REGIME_ADHERENCE_MIN", 0.60))}
            declared_note = (declared_note or
                             f"declared regime scheme '{declared_norm['scheme']}' was not available")
        adheres = bool(
            declared_present
            and adherence["trade_share"] >= adherence["min_trade_share"]
            and adherence["top_edge_label"] == declared_norm["label"]
        )
        out.update({
            "declared_regime": declared_norm,
            "declared_present": bool(declared_present),
            "declared_exempted": bool(declared_exempted),
            "adheres": adheres,
            "adherence": adherence,
        })
        if declared_note:
            out["declared_note"] = declared_note
    return out
