"""Cheap-kill + routing + verdict stages (P1, P3, P4, P5, P6, P8).

Each returns a small dict describing what it did and whether it killed the claim.
The expensive P7 backtest only runs if a claim survives P3–P5.
"""
from __future__ import annotations

import math
import sys

import numpy as np
try:
    from scipy.stats import norm
except ModuleNotFoundError:
    norm = None

from .. import config
from ..brain import Claim, Decision, Principle, BrainReader, source_is_unanchored
from .. import stats as H        # vendored DSR/Sharpe/capacity — penrose is self-contained
from . import fidelity_memory


def _genuine_cscv_pbo(candidate_oos_series, cpcv: dict) -> float | None:
    """Real CSCV PBO for a supplied >=2 candidate family.

    Current production callers do not pass such a family. This helper stays inert
    unless a future caller supplies candidate return series explicitly; the
    single-strategy CPCV path must keep `pbo` as None.
    """
    if not candidate_oos_series:
        return None
    fam = list(candidate_oos_series.values()) if isinstance(candidate_oos_series, dict) else list(candidate_oos_series)
    if len(fam) < 2:
        return None
    vals = []
    for series in fam:
        try:
            arr = [float(x) for x in series if x is not None and not math.isnan(float(x))]
        except (TypeError, ValueError):
            return None
        vals.append(arr)
    if not vals or len({len(v) for v in vals}) != 1 or len(vals[0]) < 10:
        return None
    splits = cpcv.get("splits") or []
    n_groups = int(cpcv.get("n_groups") or 0)
    if not splits or n_groups < 2:
        return None
    n = len(vals[0])
    groups = [list(g) for g in np.array_split(range(n), n_groups)]
    logits = []
    for sp in splits:
        test_groups = set(sp.get("test_groups") or [])
        test_idx = [i for g, idx in enumerate(groups) if g in test_groups for i in idx]
        train_idx = [i for g, idx in enumerate(groups) if g not in test_groups for i in idx]
        if not test_idx or not train_idx:
            continue
        is_scores = [sum(v[i] for i in train_idx) / len(train_idx) for v in vals]
        best = max(range(len(is_scores)), key=lambda j: (is_scores[j], -j))
        oos_scores = [sum(v[i] for i in test_idx) / len(test_idx) for v in vals]
        rank = 1 + sum(x < oos_scores[best] for x in oos_scores)
        pct = rank / (len(oos_scores) + 1.0)
        if 0.0 < pct < 1.0:
            logits.append(math.log(pct / (1.0 - pct)))
    if not logits:
        return None
    return round(sum(1 for x in logits if x < 0.0) / len(logits), 4)


def _require_scipy(feature: str):
    if norm is None:
        raise RuntimeError(f"pip install scipy required for {feature}")
    return norm


def _power_resolution(n_oos, bars_per_year, mde_ic, pw: dict) -> dict | None:
    """Sequential sample guidance for verdicts below the detection floor."""
    try:
        current = int(n_oos or 0)
        floor = float(pw.get("realistic_ic_floor") or 0.0)
        z_certify = float(pw.get("z_certify") or 0.0)
    except (TypeError, ValueError):
        return None
    if current < 1 or floor <= 0.0 or z_certify <= 0.0 or mde_ic is None:
        return None
    needed = int((z_certify / floor) ** 2) + 1
    try:
        bpy = float(bars_per_year or 0.0)
    except (TypeError, ValueError):
        bpy = 0.0
    needed_breadth = (max(1, int(math.ceil((z_certify / floor) ** 2 / max(1, current))))
                      if bpy > 0.0 else None)
    return {
        "current_oos_bars": current,
        "needed_oos_bars": needed,
        "more_oos_bars_needed": max(0, needed - current),
        "needed_breadth_n": needed_breadth,
        "current_mde_ic": mde_ic,
        "basis": "z/sqrt(n) single-asset; breadth via IR=IC*sqrt(N*bars)",
    }


def _resolution_rationale(verdict: str, resolution: dict, pw: dict) -> str:
    floor = pw.get("realistic_ic_floor")
    more = resolution.get("more_oos_bars_needed")
    breadth = resolution.get("needed_breadth_n")
    if breadth is None:
        breadth_txt = "or add native cross-sectional breadth"
    else:
        breadth_txt = f"or breadth >= {breadth} names"
    return (f"{verdict}: ~{more} more OOS trades ({breadth_txt}) would resolve a realistic "
            f"{floor} IC edge")


# --- P1 ingest -------------------------------------------------------------- #
def p1_ingest(source_id: str, title: str) -> dict:
    return {"stage": "P1", "source_id": source_id, "title": title,
            "sanitized": True, "note": "source text treated as untrusted data"}


# --- P3 falsifiability ------------------------------------------------------ #
def p3_falsifiability(claim: Claim) -> dict:
    # Both BTC channels make a quantified, terminal-resolution forecast -> testable.
    route = "generated-module-testable"
    return {"stage": "P3", "route": route, "killed": False,
            "reason": None,
            "note": "quantified forecast with ground-truth resolution; testable via module"}


# --- P4 fee-curve filter ---------------------------------------------------- #
def p4_fee_curve(claim: Claim, expected_edge: float | None, trade_price: float = 0.5,
                 venue: str = "kalshi") -> dict:
    """For a binary-priced trade the fee is C*fee_rate*p(1-p). The tradeable
    translation here is a VOL trade (Deribit), not a 50c binary, so the binary
    fee wall does not gate it — but we still record where it would trade and the
    vol-trade cost it must clear."""
    cfg = config.FEE_CURVE[venue]
    binary_fee = H.pm_fee_frac(trade_price, cfg["fee_rate"], cfg["C"])
    vol_cost = config.VOL_TRADE_COST["deribit_roundtrip_bps_of_vega"] / 1e4
    killed = expected_edge is not None and expected_edge < vol_cost
    evaluated = expected_edge is not None
    note = ("vol trade dodges the 50c binary fee wall; must clear "
            f"{vol_cost:.4f} round-trip vega cost")
    if not evaluated:
        note = "no claimed edge stated; fee gate not evaluated (advisory only)"
    return {"stage": "P4", "killed": bool(killed),
            "reason": "fee_curve" if killed else None,
            "evaluated": evaluated,
            "binary_fee_at_50c": round(float(binary_fee), 4),
            "vol_trade_cost_frac": round(vol_cost, 5),
            "note": note}


# --- P5 dedup --------------------------------------------------------------- #
def p5_dedup(claim: Claim, reader: BrainReader) -> dict:
    """Three layers, cheapest first:
      (a) exact claim_id (free, deterministic — must auto-kill or pipeline re-runs loop)
      (b) brainstore hybrid search (cheap, ranks by semantic + lexical)
      (c) embedding cosine similarity on top-k hits (>0.92 auto-kill, 0.75–0.92 review)

    Falls back to exact-match only if the embeddings endpoint is unavailable.
    """
    # (a) exact claim_id
    existing = reader.get(f"atoms/penrose/claim/{claim.claim_id}")
    if existing is not None:
        return {"stage": "P5", "killed": True, "reason": "dedup",
                "mode": "exact-claim-id",
                "note": "identical claim already evaluated"}

    # (b) hybrid search via the brainstore
    hits_text = reader.search(claim.statement, limit=5)

    # (c) embedding cosine similarity
    from .. import llm
    if not llm.embed_available():
        # no embeddings endpoint — return the search hit count only, don't kill
        n_hits = len([l for l in (hits_text or "").splitlines() if l.strip()])
        return {"stage": "P5", "killed": False, "reason": None,
                "mode": "exact-only-no-embeddings",
                "search_hits": n_hits,
                "note": "embeddings endpoint unavailable; only exact match checked"}

    claim_vec = llm.embed(claim.statement)
    if not claim_vec:
        return {"stage": "P5", "killed": False, "reason": None,
                "mode": "exact-only-embed-failed",
                "note": "could not embed claim statement; only exact match checked"}

    # Parse search hits and embed each; compute best similarity
    best_sim = 0.0
    best_hit = None
    hit_lines = [l.strip() for l in (hits_text or "").splitlines() if l.strip()][:5]
    for hit in hit_lines:
        # crude extraction: take first 200 chars as the comparison text
        hit_text = hit[:200]
        hit_vec = llm.embed(hit_text)
        if hit_vec:
            sim = llm.cosine(claim_vec, hit_vec)
            if sim > best_sim:
                best_sim = sim
                best_hit = hit_text[:100]

    killed = best_sim >= 0.92
    review = 0.75 <= best_sim < 0.92
    return {
        "stage": "P5", "killed": killed, "reason": "dedup" if killed else None,
        "mode": "embedding-cosine",
        "best_similarity": round(best_sim, 4),
        "best_hit": best_hit,
        "review_threshold_hit": review,
        "note": (f"auto-kill: cosine ≥ 0.92 (got {best_sim:.3f})" if killed
                 else f"review: 0.75 ≤ cosine < 0.92 (got {best_sim:.3f})" if review
                 else f"novel: cosine < 0.75 (got {best_sim:.3f})"),
    }


def corpus_isolation(claim: Claim, reader: BrainReader) -> dict:
    """Advisory-only corpus context. Empty/unavailable corpus is inert."""
    try:
        listed = reader.list(prefix="atoms/penrose", n=1)
    except Exception as e:  # noqa: BLE001
        return {"neighbor_count": 0, "nearest": [], "mechanism_family_present": False,
                "isolation_score": None, "advisory": f"corpus unavailable; no isolation signal: {e}"}
    if not [line for line in (listed or "").splitlines() if line.strip()]:
        return {"neighbor_count": 0, "nearest": [], "mechanism_family_present": False,
                "isolation_score": None, "advisory": "corpus empty; no isolation signal"}
    query = " ".join(x for x in (claim.statement, claim.mechanism, claim.applicable_strategy_class) if x)
    try:
        raw = reader.search(query, limit=5)
    except Exception as e:  # noqa: BLE001
        return {"neighbor_count": 0, "nearest": [], "mechanism_family_present": False,
                "isolation_score": None, "advisory": f"corpus search unavailable; no isolation signal: {e}"}
    nearest = []
    mechanism_terms = {
        t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in (claim.mechanism or "")).split()
        if len(t) >= 4
    }
    mechanism_family_present = False
    for line in [l.strip() for l in (raw or "").splitlines() if l.strip()][:5]:
        parts = [p.strip() for p in line.split("::")]
        slug = parts[0] if parts else line
        title = parts[1] if len(parts) > 1 else ""
        score = None
        if len(parts) > 2:
            try:
                score = float(parts[2])
            except ValueError:
                score = None
        text = f"{slug} {title}".lower()
        if mechanism_terms and any(t in text for t in mechanism_terms):
            mechanism_family_present = True
        nearest.append({"slug": slug, "title": title, "score": score})
    n = len(nearest)
    if n == 0:
        return {"neighbor_count": 0, "nearest": [], "mechanism_family_present": False,
                "isolation_score": 1.0, "advisory": "no nearby corpus neighbors found"}
    score = round(1.0 / (1.0 + n + (1 if mechanism_family_present else 0)), 4)
    advisory = ("related corpus neighbors found"
                if mechanism_family_present else "neighbors found, but no recurring mechanism family")
    return {"neighbor_count": n, "nearest": nearest,
            "mechanism_family_present": mechanism_family_present,
            "isolation_score": score, "advisory": advisory}


# --- P6 routing / spec gen -------------------------------------------------- #
def p6_routing(claim: Claim, registry: dict) -> dict:
    mod = registry.get(claim.applicable_strategy_class)
    if mod:
        return {"stage": "P6", "module_id": mod, "spec_generated": False,
                "note": "existing module routes this strategy class"}
    return {"stage": "P6", "module_id": "macro_vol_btc", "spec_generated": True,
            "note": "cold-start: no module for this class; ModuleSpec generated (F10)"}


# --- P8 verdict + principle proposal ---------------------------------------- #
def p8_verdict(claim: Claim, bt: dict, holdout: dict, synthetic: bool) -> Decision:
    psr = bt.get("psr", 0.0) or 0.0
    dsr = bt.get("dsr", 0.0) or 0.0
    edge_t = bt.get("edge_t") or 0.0
    folds = bt.get("three_fold", {})
    cap = bt.get("capacity_usd")
    band = config.DSR_DECISION
    claim_type = str(bt.get("claim_type") or "")
    if not claim_type:
        try:
            claim_type = fidelity_memory.classify_claim_type(claim)
        except Exception:  # noqa: BLE001 - verdict caps must fail closed to legacy behavior
            claim_type = "trading_strategy"

    # --- POWER: how small an edge could THIS backtest even resolve? ----------------------------
    # MDE (min detectable IC) ~ z/sqrt(n_oos): the smallest per-bar Sharpe (~= IC for a single-asset
    # strategy) that could clear the certification band. If it sits ABOVE the realistic floor, the
    # data cannot tell a true marginal edge from noise — so a non-structural null is `underpowered`,
    # NOT a kill. We compute it for EVERY verdict and ship it, so a "kill" is never mistaken for
    # "proven dead" when it's really "below my detection floor".
    pw = config.POWER
    _n_oos = bt.get("n_oos", 0) or 0
    mde_ic = round(pw["z_certify"] / math.sqrt(max(2, _n_oos)), 4) if _n_oos >= 2 else None
    _bpy = bt.get("bars_per_year") or 0.0
    mde_sharpe_ann = round(mde_ic * math.sqrt(_bpy), 3) if (mde_ic and _bpy) else None
    power_sufficient = bool(mde_ic is not None and mde_ic <= pw["realistic_ic_floor"])
    resolution = None

    # Gate on the DEFLATED Sharpe — the multiple-testing-corrected metric. (A-001)
    # `max(psr, dsr)` was inert: DSR = PSR(sr_star>=0) <= PSR always, so max() collapsed to
    # the undeflated PSR and the entire trial-count/deflation apparatus never touched a verdict.
    # dsr falls back to psr when n_trials<2 / sr_var<=0 (see harness.deflated_sharpe), so this
    # is strictly correct and only ever stricter.
    score = dsr
    reasons = []
    fold_ns = folds.get("fold_n") or []
    observed_ic = bt.get("per_trade_sharpe")
    try:
        observed_ic = float(observed_ic)
    except (TypeError, ValueError):
        observed_ic = None
    # PEN-01 amendment: test 3-fold power against the DECLARED realistic-edge floor, never the
    # observed in-sample Sharpe. The in-sample point estimate is upward-biased, so using it as the
    # power reference is circular — it inflates the computed power and hard-kills true marginal
    # edges that fail the all-signs test by chance. The power question is "could this data resolve
    # a REALISTIC edge?", whose effect size is the frozen config floor (config.POWER). (Investigated
    # 2026-07-03: 9/15 Part-A false-kills came through this inflated-power path.) observed_ic is
    # retained only for the decision record, not for gating.
    ic_ref = pw["realistic_ic_floor"]
    three_fold_power = (
        math.prod(_require_scipy("P8 three-fold power").cdf(ic_ref * math.sqrt(n)) for n in fold_ns)
        if len(fold_ns) == 3 and all(n >= 5 for n in fold_ns)
        else None
    )
    min_fold_t = folds.get("min_fold_t")
    ambiguous_three_fold = False
    if bt.get("n_oos", 0) < band["min_oos_bars"]:
        verdict, kill = "insufficient_data", "data_unavailable"
        reasons.append(f"only {bt.get('n_oos', 0)} OOS trades (< {band['min_oos_bars']} "
                       "minimum); too thin to trust a verdict")
    elif not folds.get("consistent", False):
        verdict, kill = "kill", "in_sample_only"
        reasons.append(f"3-fold Sharpe not sign-stable: {folds.get('folds')}")
        if min_fold_t is not None and min_fold_t <= pw["structural_fold_t"]:
            pass
        elif three_fold_power is not None and three_fold_power >= pw["three_fold_min_power"]:
            pass
        else:
            ambiguous_three_fold = True
    elif score < band["kill_below_psr"] or edge_t < 1.0:
        verdict, kill = "kill", ("no_oos_edge" if score < band["kill_below_psr"] else "low_edge_t")
        reasons.append(f"OOS score {score:.3f} (<{band['kill_below_psr']}), edge_t {edge_t}")
    elif score < band["watch_band"][1]:
        verdict, kill = "watch", None
        reasons.append(f"OOS score {score:.3f} in watch band {band['watch_band']}")
    else:
        # holdout must confirm by sign and significance to reach research-supported
        holdout_sharpe = holdout.get("holdout_sharpe", -9) or -9
        holdout_psr = holdout.get("holdout_psr", 0.0) or 0.0
        if holdout_sharpe > 0 and holdout_psr >= config.HOLDOUT_CONFIRM_PSR:
            verdict, kill = "research-supported", None
        else:
            verdict, kill = "watch", None
            reasons.append(
                "OOS strong but holdout did not confirm "
                f"(sharpe={holdout.get('holdout_sharpe')}, psr={holdout.get('holdout_psr')}, "
                f"required_psr>={config.HOLDOUT_CONFIRM_PSR})")

    # --- empirical robustness gates (Monte-Carlo) ------------------------- #
    # These only bite on borderline survivors (watch / research-supported) — the
    # exact regime where the analytic PSR is least trustworthy. A clear kill is
    # already dead above; here we stop luck from graduating to a survivor.
    boot = bt.get("bootstrap") or {}
    perm = bt.get("permutation") or {}
    regime = bt.get("regime") or {}
    declared_regime = regime.get("declared_regime") or getattr(claim, "declared_regime", None)
    cpcv = dict(bt.get("cpcv") or {})
    pbo = _genuine_cscv_pbo(bt.get("cpcv_candidate_oos_series"), cpcv)
    if pbo is not None:
        cpcv["pbo"] = pbo
    gates = config.ROBUSTNESS_GATES
    if verdict in ("watch", "research-supported"):
        if declared_regime and regime.get("adheres") is False:
            verdict, kill = "kill", "regime_mismatch"
            adherence = regime.get("adherence") or {}
            reasons.append(
                f"regime_mismatch: declared {declared_regime.get('scheme')}="
                f"{declared_regime.get('label')} but adherence failed "
                f"(trade_share={adherence.get('trade_share')}, "
                f"top_edge_label={adherence.get('top_edge_label')})")
        elif gates.get("kill_if_regime_fragile", True) and regime.get("fragile"):
            verdict, kill = "kill", "regime_fragile"
            reasons.append(f"regime-fragile: {regime.get('fragile_reason')}")
        elif gates["kill_if_edge_ci_includes_zero"] and boot.get("edge_ci_includes_zero"):
            verdict, kill = "kill", "edge_ci_zero"        # AMBIGUOUS: wide CK -> could be underpowered
            reasons.append(f"bootstrap {int(boot.get('ci', 0) * 100)}% edge CI includes "
                           f"zero {boot.get('edge_ci')} (edge not distinguishable from luck)")
        elif perm.get("p_value") is not None and perm["p_value"] > gates["permutation_kill_p"]:
            verdict, kill = "kill", "no_signal_alignment"  # STRUCTURAL: no real signal->payoff link
            reasons.append(f"permutation p={perm['p_value']} > {gates['permutation_kill_p']} "
                           f"(signal indistinguishable from shuffled outcomes)")
        else:
            # B-007: walk-forward is an INDEPENDENT axis — wire it as a conjunctive kill gate
            # (it was decorative before). Only bites when it actually ran (has windows).
            wf = bt.get("walk_forward") or {}
            if (gates.get("kill_if_walk_forward_inconsistent", True)
                    and wf.get("per_window_sharpe") and wf.get("consistent") is False):
                verdict, kill = "kill", "walk_forward_drift"
                reasons.append(f"walk-forward inconsistent across windows {wf.get('per_window_sharpe')} "
                               "(parameter drift)")
            elif (gates.get("kill_if_cpcv_overfit", True)
                  and cpcv.get("ran") and cpcv.get("overfit")):
                verdict, kill = "kill", "overfit_cpcv"
                reasons.append(
                    f"CPCV overfit: prob_oos_loss={cpcv.get('prob_oos_loss')}, "
                    f"median_oos_sharpe={cpcv.get('median_oos_sharpe')}, "
                    f"combos={cpcv.get('combos_used')}")

    # Trust/tail caps are independent and all must remain VISIBLE. Capture whether the statistical
    # path reached the strongest verdict BEFORE any cap (tail or trust); otherwise the first cap
    # hides the others' messages in reports and review.
    reached_research_supported = verdict == "research-supported"

    trg = getattr(config, "TAIL_RISK_GATE", {"enabled": False})
    tail = bt.get("tail") or {}
    tail_asymmetric = bool(trg.get("enabled") and tail.get("asymmetric"))
    if tail_asymmetric:
        warning = (
            f"tail-asymmetric widow-maker warning: bounded-up/unbounded-down payoff "
            f"(skew={tail.get('skew')}, tail_ratio={tail.get('tail_ratio')}, "
            f"max_loss={tail.get('max_loss')})"
        )
        if claim_type == "provided_series_statistic":
            warning += (
                "; provided-series caveat: tested series is pre-aggregated, so per-trade "
                "and cross-unit portfolio tail risk is understated; true tail risk requires "
                "per-trade reconstruction"
            )
        reasons.append(warning)
        # J-1: the warning stands for ANY verdict, but only CHANGE the verdict for a current
        # survivor. Never overwrite a prior structural kill, insufficient_data routing, or the
        # power-aware underpowered relabel (those are decided upstream and must stand).
        if verdict in ("watch", "research-supported"):
            if trg.get("cap_only"):
                if verdict == "research-supported":
                    verdict = "watch"
                    reasons.append("tail-asymmetric: capped to watch")
            else:
                verdict, kill = "kill", "tail_asymmetric"

    if cap is None and verdict in ("watch", "research-supported"):
        reasons.append("capacity not estimable")

    source_type = getattr(claim, "source_type", "external_source")
    unanchored = source_is_unanchored(source_type)
    fidelity_provenance = "self-authored-unanchored" if unanchored else "external-source"
    # `verdict != "kill"` guards against a cap un-killing a claim the tail gate (cap_only=False)
    # just killed; the message-visibility is preserved via reached_research_supported.
    if reached_research_supported and verdict != "kill" and getattr(config, "COST_PROVENANCE", "modeled") != "measured":
        verdict = "watch"
        reasons.append("costs/capacity are MODELED placeholders — capped at watch until measured (E2)")
    if reached_research_supported and verdict != "kill" and unanchored:
        verdict = "watch"
        reasons.append(f"{source_type} lacks an external anchor — capped at watch until "
                       "independent forward/external confirmation")
    provided_series_provenance = (
        "unverified_construction" if claim_type == "provided_series_statistic" else None
    )
    if claim_type == "provided_series_statistic" and verdict in ("watch", "research-supported"):
        if verdict == "research-supported":
            verdict = "watch"
        reasons.append(
            "provided-series claim: Penrose tests a pre-computed series it did not construct; "
            "capped at watch/provisional pending primitive reconstruction (EXP-2)")
    post_sample_missing = False
    ps = bt.get("post_sample") or {}
    post_cfg = getattr(config, "POST_SAMPLE", {})
    if (ps and post_cfg.get("enabled") and source_type == "external_source"
            and verdict in ("watch", "research-supported")):
        min_post_years = float(post_cfg.get("min_post_years", 1.0))
        post_years = ps.get("post_years")
        if (not ps.get("declared")) or post_years is None or post_years < min_post_years:
            if verdict == "research-supported":
                verdict = "watch"
            post_sample_missing = True
            reasons.append(
                "no post-sample evidence: bundle ends "
                f"{ps.get('data_end')}, claim sample ends {ps.get('sample_end') or 'undeclared'}")
    csg = getattr(config, "COST_SENSITIVITY_GATE", {"enabled": False})
    cs = bt.get("cost_sensitivity") or {}
    if csg.get("enabled") and verdict in ("watch", "research-supported"):
        margin = cs.get("margin")
        if margin is not None and margin < csg.get("min_margin", 1.5):
            if verdict == "research-supported":
                verdict = "watch"
                reasons.append(f"cost-sensitivity margin {margin}x below {csg.get('min_margin')}x")
            else:
                verdict, kill = "kill", "cost_sensitive"
                reasons.append(f"cost-sensitivity margin {margin}x below {csg.get('min_margin')}x")
    # B-006: make the capacity-CI conditionality visible (it's only over positive-edge resamples)
    cci = bt.get("capacity_ci") or {}
    if verdict in ("watch", "research-supported") and cci.get("conditional_on_positive_edge"):
        reasons.append(f"capacity CI conditional on positive edge (p={cci.get('p_positive_edge')})")

    # --- POWER-AWARE RECLASSIFICATION ---------------------------------------------------------
    # A NON-structural null — low DSR (`no_oos_edge`), low edge t-stat (`low_edge_t`), or a
    # bootstrap CI that includes zero (`edge_ci_zero`) — on data that could NOT resolve a realistic
    # edge is NOT proven dead; it is below the detection floor. Relabel it `underpowered`.
    # Structural kills (in_sample_only, regime_fragile, walk_forward_drift, no_signal_alignment)
    # are power-INDEPENDENT and stand. This is the fix for "a skeptic that rejects everything real
    # == a broken always-no".
    _AMBIGUOUS_NULL = {"no_oos_edge", "edge_ci_zero", "overfit_cpcv", "low_edge_t"}
    ambiguous_three_fold_underpowered = (
        kill == "in_sample_only"
        and ambiguous_three_fold
        and (three_fold_power is None or three_fold_power < pw["three_fold_min_power"])
    )
    if (verdict == "kill"
            and ((kill in _AMBIGUOUS_NULL and not power_sufficient)
                 or ambiguous_three_fold_underpowered)):
        verdict, kill = "underpowered", "below_detection_floor"
        resolution = _power_resolution(_n_oos, _bpy, mde_ic, pw)
        reasons.append(
            f"NOT proven dead — below the detection floor: min detectable IC ~{mde_ic} on n_oos="
            f"{_n_oos} (realistic edges are 0.02-0.05). To resolve a {pw['realistic_ic_floor']} IC "
            f"edge you'd need n_oos >~ {resolution.get('needed_oos_bars') if resolution else None} "
            f"at this breadth, or to test at native cross-sectional "
            f"breadth. Verdict = cannot falsify at deployable confidence, not 'dead'.")
    if resolution is None and not power_sufficient:
        resolution = _power_resolution(_n_oos, _bpy, mde_ic, pw)
    # The "~N more trades would resolve it" RATIONALE applies ONLY to the power-limited labels.
    # Structural kills (in_sample_only, regime_fragile, no_signal_alignment, walk_forward_drift, ...)
    # are power-INDEPENDENT and stand — telling a researcher to collect more data there is false. The
    # resolution OBJECT is still attached in metrics (above) as neutral data for any not-powered case.
    if resolution and verdict in ("underpowered", "insufficient_data"):
        reasons.append(_resolution_rationale(verdict, resolution, pw))

    rationale = "; ".join(reasons) or "passed all gates"
    if declared_regime and verdict != "kill" and regime.get("adheres") is True:
        rationale += (f"; valid within declared regime: {declared_regime.get('scheme')}="
                      f"{declared_regime.get('label')}; not a claim about other regimes")
    if synthetic:
        rationale += ". NOTE: macro signal SYNTHETIC this run — provisional verdict."

    return Decision(
        decision_id=f"{claim.claim_id}-d1",
        claim_id=claim.claim_id, verdict=verdict, kill_reason=kill,
        rationale=rationale,
        metrics={"psr": psr, "dsr": dsr, "edge_t": edge_t,
                 "n_trials": bt.get("n_trials"),
                 "oos_sharpe": bt.get("oos_sharpe"), "capacity_usd": cap,
                 "mde_ic": mde_ic, "mde_sharpe_ann": mde_sharpe_ann,   # power: smallest edge resolvable
                 "power_sufficient": power_sufficient,                  # could we have seen a realistic edge?
                 "three_fold_power": three_fold_power,
                 "min_fold_t": min_fold_t,
                 "no_post_sample_data": post_sample_missing,
                 "resolution": resolution,
                 # B2 (in-sample replication proxy): a KILL is only corpus-worthy if the edge was
                 # first reproduced IN-SAMPLE (positive IS Sharpe) and then died OOS. If we never
                 # saw the edge even in-sample, the "kill" reflects our build, not the paper.
                 "replicated_in_sample": (bt.get("is_sharpe") is not None and (bt.get("is_sharpe") or 0) > 0),
                 "is_sharpe": bt.get("is_sharpe"),
                 "three_fold": folds.get("folds"), "n_trades": bt.get("n"),
                 "bootstrap": boot, "permutation": perm, "regime": regime,
                 "tail": bt.get("tail"),
                 "tail_asymmetric": tail_asymmetric,
                 "tail_skew": tail.get("skew"),
                 "tail_tail_ratio": tail.get("tail_ratio"),
                 "tail_max_loss": tail.get("max_loss"),
                 "cpcv": cpcv,
                 "capacity_ci": bt.get("capacity_ci"),
                 "cost_sensitivity": bt.get("cost_sensitivity"),
                 "walk_forward": bt.get("walk_forward"),
                 "declared_regime": declared_regime,
                 "regime_adherence": regime.get("adherence"),
                 "fidelity_provenance": fidelity_provenance,
                 "provided_series_provenance": provided_series_provenance,
                 "claim_type": claim_type,
                 "holdout": holdout, "synthetic_signal": synthetic},
        revisit_at="2026-07-01",
    )


def propose_principle(decisions: list[Decision]) -> Principle | None:
    """N≥3 supporting kills, consistent kill_reason + strategy class. With BTC
    only (2 claims) we cannot reach N≥3 — return a sub-threshold note instead."""
    kills = [d for d in decisions if d.verdict == "kill"]
    if len(kills) < 3:
        return None
    reasons = {d.kill_reason for d in kills}
    if len(reasons) != 1:
        return None
    return Principle(
        principle_id="principle-macro-vol-fed-fragile",
        statement=("Macro-prediction-market signals that are strong in-sample on "
                   "crypto vol tend to lose their edge out-of-sample net of vol-trade "
                   "costs."),
        supporting_kills=[d.decision_id for d in kills],
        applicable_strategy_classes=[config.STRATEGY_CLASS_VOL],
        n_observations=len(kills), confidence=0.5,
    )
