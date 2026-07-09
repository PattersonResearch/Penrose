"""LLM-driven ModuleSpec generation.

When P6 routing finds no registered module for a claim's strategy class (cold-
start registry, or a genuinely novel strategy class), this generates a
ModuleSpec YAML the operator can review and implement (or hand to an agent
swarm). One-shot insertion: the spec lands in modules/_specs/, the
pipeline continues, and the operator reviews it asynchronously.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import config, llm
from ..brain import Claim
from ..data.contract import load_catalog_loader
from ..strategy_family import declared_strategy_family
from . import fidelity_memory
from .p1_ingest import IngestedSource


SPEC_SYSTEM = (
    "You are a research-engine module spec generator. Given a falsifiable claim "
    "extracted from a paper, you produce a ModuleSpec — a precise, implementable "
    "description of how to test the exact claim. If claim_type is "
    "predictive_regression, specify the exact predictor, target, horizon, estimator, "
    "declared statistic, and search grid; do NOT translate it into a trading strategy. "
    "If claim_type is factor_spanning, specify the declared candidate factor, benchmark "
    "factor set, alpha/intercept statistic, and declared search grid; do NOT translate "
    "it into a trading strategy. "
    "If claim_type is cross_sectional_sort, specify the declared returns panel, "
    "characteristic panel, characteristic, bucket count, rebalance cadence, hold, and "
    "declared search grid; do NOT translate it into a trading overlay. "
    "If claim_type is event_study, specify the declared return series, event calendar, "
    "event window, estimation window, baseline model, optional market series, and "
    "declared search grid; do NOT translate it into a trading strategy. "
    "If claim_type is forecast_skill, specify the declared model forecast series, "
    "realized target series, explicit benchmark forecast series OR declared implied "
    "benchmark (random_walk/historical_mean), squared-loss differential, and declared "
    "search grid; do NOT translate it into a trading strategy. "
    "If claim_type is formulaic_signal, preserve the declared DSL signal string exactly, "
    "declare the traded price series, optional funding_pnl_series, position_map, and "
    "every required input series; do NOT emit Python code. "
    "If claim_type is descriptive_statistical, specify the statistic to compute and its uncertainty; "
    "do NOT translate it into a trading strategy. If claim_type is trading_strategy, "
    "act as a translator, not a co-author: preserve the claim's signal exactly, do "
    "not add unclaimed gates or kill conditions, and choose one explicit conventional "
    "default for any unspecified required parameter. Describe the signal, positions, "
    "PnL, DSR / 3-fold / locked-holdout / fee + slippage + capacity discipline. If "
    "claim_type is structural_proposition, be "
    "honest about what cannot be operationalized. Be concrete: specify the inputs "
    "required (use the data contract vocabulary), the kill criterion, and the "
    "unknowns. If a claim requires data that is not in the available data vocabulary, "
    "declare the true required input using a clear catalog-style series name and list "
    "it in unknowns as unavailable; never fabricate, synthesize, derive, or proxy a "
    "missing input from other series to make the spec runnable. The operator (or an "
    "agent swarm) will implement your spec; vagueness "
    "costs them time. Respond strictly in JSON."
)

TRADING_SPEC_USER_TMPL = """Generate a ModuleSpec for this claim:

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Translator-not-co-author constraints for trading_strategy claims:
- Do NOT add any significance test, data-quality filter, minimum-observation gate, or
  rejection/kill condition that the claim itself does not state. Required cost/fee
  modeling is fine; invented pass/fail gates are not.
- Preserve the claim's exact signal definition. If the claim states an absolute
  directional rule, do not convert it to a cross-sectional, relative-value, ranked,
  demeaned, normalized, or otherwise transformed signal unless the claim says to.
  Likewise, do not convert a relative/ranked claim into an absolute directional rule.
- For any required parameter the claim leaves unspecified (lookback, exit rule, sign
  convention, venue), commit to one conventional default and state it explicitly. Never
  emit a menu of alternatives in one field, and never leave a required field unresolved.
- Data fidelity is part of faithful translation. If the exact input the claim requires is
  not present in AVAILABLE DATA SERIES, do NOT synthesize, fabricate, derive, blend, or
  proxy it from other series. Declare the true required input in `inputs` with a clear
  catalog-style name (for example `bnb_spot_daily`) and include an `unknowns` entry saying
  that series is unavailable. A faithful-but-data-blocked spec is correct.

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "strategy_family": {{"components": [str], "method": "single|regime_blend|ensemble|overlay"}},
  "claim_translation": str (how the academic claim becomes a tradeable, falsifiable test),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "signal_logic": str (pseudocode or precise description of how the module computes positions),
  "kill_criterion": str (concrete: which metric thresholds kill the strategy),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""

DESCRIPTIVE_SPEC_USER_TMPL = """Generate a ModuleSpec for this descriptive statistical claim.
The implementation must compute and test the stated statistic directly. Do not invent
entry rules, positions, PnL, Sharpe, or a conditional trading strategy unless the claim
itself explicitly says to trade.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "descriptive_statistical",
  "claim_translation": str (the exact statistic/hypothesis to test, e.g. unconditional mean + CI),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "statistic_logic": str (precise description of the statistic, sample, uncertainty, and test),
  "signal_logic": str ("descriptive_statistical: compute the statistic directly; no positions/PnL"),
  "kill_criterion": str (concrete: what statistical result fails to replicate the stated claim),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""

STRUCTURAL_SPEC_USER_TMPL = """Generate a ModuleSpec for this structural proposition.
If it cannot be operationalized against available data without inventing a trading
strategy, say so plainly in claim_translation and unknowns.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": str (echo or refine the strategy class),
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "structural_proposition",
  "claim_translation": str (what can and cannot be falsified),
  "inputs": [str] (list of F7a data contract series names the module needs),
  "signal_logic": str ("structural_proposition: cannot create positions unless explicitly claimed"),
  "kill_criterion": str (concrete falsification criterion, or "cannot_operationalize"),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (any non-obvious gotchas the implementer should know)
}}
"""

PREDICTIVE_REGRESSION_SPEC_USER_TMPL = """Generate a ModuleSpec for this predictive-regression claim.
The implementation must test the declared predictor -> target -> horizon relationship directly. Do not
invent entry rules, positions, PnL, costs, capacity, or a trading strategy.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "predictive_regression",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "predictive_regression",
  "claim_translation": str (the exact predictor -> target -> horizon relationship to test),
  "inputs": [str, str] (predictor series first, target series second),
  "predictor": str (same as inputs[0]),
  "target": str (same as inputs[1]),
  "horizon": int or str (declared h-ahead target horizon),
  "estimator": str (for v1, single-predictor OLS / covariance-sign test),
  "statistic": str (declared sign, coefficient, t-statistic, R2, MSFE, or equivalent),
  "param_grid": dict (declared search grid; include coins/specs/horizons if the claim declares a search),
  "signal_logic": str ("predictive_regression: align X_t with Y_t+h; no trading overlay"),
  "kill_criterion": str (concrete: OOS/holdout predictive direction or deflated statistic fails),
  "expected_data_needs": str (window, frequency, granularity),
  "unknowns": [str] (assumptions the operator must verify; data that may not exist; etc.),
  "implementation_notes": str (no look-ahead: emit non-overlapping h-sampled rows; fit sign and z-score moments on IS rows whose target timestamps remain in-sample)
}}
"""

FACTOR_SPANNING_SPEC_USER_TMPL = """Generate a ModuleSpec for this factor-spanning claim.
The implementation must test whether the declared candidate factor earns alpha after controlling
for the declared benchmark factor set. Do not invent entry rules, positions, PnL, costs, capacity,
or a trading strategy.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "factor_spanning",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "factor_spanning",
  "claim_translation": str (candidate factor F regressed on the declared benchmark factors),
  "inputs": [str, ...] (candidate factor first, then benchmark Series),
  "candidate_factor": str (same as inputs[0]),
  "benchmark_set": "capm|ff3|ff5|carhart",
  "benchmark_factors": [str, ...] (same as inputs[1:]),
  "estimator": "multivariate_ols_is_frozen_betas",
  "statistic": "alpha_t_stat",
  "param_grid": dict (declared benchmark sets/candidates tried for deflation),
  "signal_logic": str ("factor_spanning: fit F on benchmarks in-sample only; emit residual alpha series; no trading overlay"),
  "kill_criterion": str (concrete: OOS/holdout residual alpha or deflated statistic fails),
  "expected_data_needs": str (candidate and benchmark factor return Series on a common DatetimeIndex),
  "unknowns": [str] (candidate/benchmark bindings an operator must verify),
  "implementation_notes": str (freeze betas on the in-sample prefix only; never refit on OOS/holdout)
}}
"""

CROSS_SECTIONAL_SORT_SPEC_USER_TMPL = """Generate a ModuleSpec for this cross-sectional-sort claim.
The implementation must test the declared characteristic sort directly: sort entities by the
characteristic known at the rebalance date and test the top-minus-bottom spread. Do not invent
entry rules, timing overlays, costs, capacity, or a separate trading strategy.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "cross_sectional_sort",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "cross_sectional_sort",
  "claim_translation": str (characteristic-sorted top-minus-bottom spread to test),
  "inputs": [] (Series inputs are not used for this claim type),
  "panel_inputs": {{"returns": str or object, "characteristic": str or object}},
  "characteristic": str (declared characteristic C),
  "n_buckets": int (e.g. 10 for deciles, 5 for quintiles),
  "rebalance": str (declared cadence, e.g. "ME"),
  "hold": str (declared holding window, e.g. "1M"),
  "statistic": str (declared mean spread/t-stat/Sharpe or equivalent),
  "param_grid": dict (declared bucket counts / characteristics tried for deflation),
  "signal_logic": str ("cross_sectional_sort: form_factor(returns_panel, characteristic_panel, ...); no trading overlay"),
  "kill_criterion": str (concrete: OOS/holdout spread or deflated statistic fails),
  "expected_data_needs": str (point-in-time characteristic Panel plus survivorship-corrected returns Panel),
  "unknowns": [str] (unconfirmed characteristic/universe/panel bindings),
  "implementation_notes": str (ranking uses only characteristics known at/before rebalance; returns panel must retain delisted entities)
}}
"""

EVENT_STUDY_SPEC_USER_TMPL = """Generate a ModuleSpec for this event-study claim.
The implementation must test cumulative abnormal returns around the declared event dates directly.
Do not invent entry rules, positions, PnL, costs, capacity, or a trading strategy.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "event_study",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "event_study",
  "claim_translation": str (average CAR around the declared event calendar to test),
  "inputs": [str, ...] (return Series first; market Series second only for market_model baseline),
  "return_series": str (same as inputs[0]),
  "event_calendar": str or object (declared table/path with date_col),
  "window": [int, int] (event-window offsets in trading rows, e.g. [0, 5]),
  "estimation_window": int (pre-event bars used for the baseline),
  "baseline": "mean_adjusted|market_model",
  "market_series": str (required only when baseline is market_model),
  "statistic": "average_car",
  "param_grid": dict (declared windows/baselines tried for deflation),
  "signal_logic": str ("event_study: estimate baseline strictly before each event; emit per-event CAR; no trading overlay"),
  "kill_criterion": str (concrete: OOS/holdout average CAR or deflated statistic fails),
  "expected_data_needs": str (return Series plus declared event calendar table with one date column),
  "unknowns": [str] (unconfirmed event calendar/return-series bindings),
  "implementation_notes": str (no look-ahead: baseline estimation window ends strictly before event date)
}}
"""

FORECAST_SKILL_SPEC_USER_TMPL = """Generate a ModuleSpec for this forecast-skill claim.
The implementation must test whether the declared model forecast beats the declared
benchmark forecast on the declared realized target out-of-sample. Do not invent
entry rules, positions, PnL, costs, capacity, or a trading strategy.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "forecast_skill",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "forecast_skill",
  "claim_translation": str (model forecast F compared with benchmark B on realized target Y),
  "inputs": [str, str, ...] (model forecast first, target second, explicit benchmark third only when declared),
  "model_forecast": str (same as inputs[0]),
  "target": str (same as inputs[1]),
  "benchmark": str or object (explicit Series OR {{"kind": "implied", "method": "random_walk|historical_mean"}}),
  "loss": "squared_error",
  "statistic": "loss_differential_mean",
  "param_grid": dict (declared models/benchmarks/specs tried for deflation),
  "signal_logic": str ("forecast_skill: emit (B_t-Y_t)^2 - (F_t-Y_t)^2; no trading overlay"),
  "kill_criterion": str (concrete: OOS/holdout loss differential or deflated statistic fails),
  "expected_data_needs": str (forecast and target Series on a common DatetimeIndex),
  "unknowns": [str] (unconfirmed forecast/target/benchmark bindings),
  "implementation_notes": str (constructed benchmarks must be strictly causal: random_walk=Y.shift(1), historical_mean=expanding_mean(Y).shift(1))
}}
"""

FORMULAIC_SIGNAL_SPEC_USER_TMPL = """Generate a ModuleSpec for this formulaic-signal claim.
The implementation must preserve the declared DSL formula exactly and run it through the
trusted formulaic_signal executor. Do not emit Python code or invent entry/exit rules.

statement: {statement}
mechanism:  {mechanism}
scope:      {scope}
horizon:    {horizon}
strategy_class: {strategy_class}
claim_type: {claim_type}

source_span (verbatim from paper): "{source_span}"
claimed_metric: "{claimed_metric}"

Output JSON with this exact shape:
{{
  "module_id": str (snake_case identifier for the module),
  "strategy_class": "formulaic_signal",
  "strategy_family": {{"components": [str], "method": "single"}},
  "claim_type": "formulaic_signal",
  "claim_translation": str (formula S_t -> P_t -> one-bar-lagged trade_series return),
  "inputs": [str, ...] (exactly Names(signal) plus trade_series plus funding_pnl_series if declared),
  "trade_series": str (price series traded),
  "signal": str (DSL formula string; preserve exactly),
  "position_map": "sign|zscore_clip",
  "funding_pnl_series": str or null,
  "param_grid": dict (declared formula/grid parameters tried for deflation),
  "signal_logic": str ("formulaic_signal: parse DSL; P_t=position_map(S_t); net_t=P_(t-1)*ret_t - costs; optional funding cash-flow -P_(t-1)*funding_t"),
  "kill_criterion": str (concrete: OOS/holdout net return fails, 3-fold sign stability fails, or deflated statistic fails),
  "expected_data_needs": str (DatetimeIndex Series for each input),
  "unknowns": [str] (unconfirmed data bindings or formula fields),
  "implementation_notes": str (no look-ahead: executor applies one-bar delay exactly once; parser allows only the audited operator table)
}}
"""


def classify_claim_type(claim: Claim, source: IngestedSource | None = None) -> str:
    return fidelity_memory.classify_claim_type(claim, source)


def _embedded_formulaic_signal_spec(claim: Claim) -> dict | None:
    try:
        raw = (getattr(claim, "data_provenance", {}) or {}).get("formulaic_signal_spec")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict) or raw.get("claim_type") != "formulaic_signal":
        return None
    try:
        from . import formulaic_signal

        signal = str(raw.get("signal") or "").strip()
        trade_series = str(raw.get("trade_series") or "").strip()
        formulaic_signal.validate_signal(signal)
        names = formulaic_signal.referenced_names(signal)
    except Exception:  # noqa: BLE001
        return None
    if not trade_series:
        return None
    if not names:
        return None  # S72-1: a constant-only signal references no series -> not executable; refuse the spec
    position_map = str(raw.get("position_map") or "sign").strip().lower()
    if position_map not in {"sign", "zscore_clip"}:
        return None
    out = dict(raw)
    out["claim_type"] = "formulaic_signal"
    out["signal"] = signal
    out["trade_series"] = trade_series
    out["position_map"] = position_map
    out["inputs"] = sorted(names | {trade_series})
    out["param_grid"] = dict(out.get("param_grid") or out.get("grid") or {})
    out["grid"] = dict(out.get("grid") or out["param_grid"])
    out.setdefault("strategy_class", "formulaic_signal")
    out.setdefault("claim_statement", claim.statement)
    out.setdefault("claim_source_span", claim.source_span)
    out.setdefault("claim_mechanism", claim.mechanism)
    out.setdefault("_llm_mode", "embedded-formulaic-signal")
    return out


def _catalog_vocab() -> str:
    """Newline list of catalog series. Fail-open to an omitted prompt block."""
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        if not hasattr(catalog, "available"):
            return ""
        if hasattr(catalog, "describe_brief"):
            lines = [catalog.describe_brief(n) for n in catalog.available()]
        else:
            lines = list(catalog.available())
        return "\n".join(f"  - {line}" for line in lines)
    except Exception:  # noqa: BLE001
        return ""


def generate_spec(claim: Claim, source: IngestedSource,
                  *, use_llm: bool = True,
                  prior_divergences: list[str] | None = None) -> dict:
    """Generate a ModuleSpec for a claim. Returns the spec dict + writes YAML to disk.

    The YAML lands at modules/_specs/<claim_id>.yaml. The operator reviews it
    asynchronously and (if accepted) moves it under modules/<module_id>/ and
    implements impl.py — at which point the registry picks it up on the next run.
    """
    specs_dir = config.MODULES / "_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    out_path = specs_dir / f"{claim.claim_id}.yaml"

    embedded_formulaic = _embedded_formulaic_signal_spec(claim)
    claim_type = "formulaic_signal" if embedded_formulaic is not None else classify_claim_type(claim, source)
    if embedded_formulaic is not None:
        spec = embedded_formulaic
    elif claim_type == "provided_series_statistic":
        # 6g: a mechanical translation, not a creative one -- never touches the LLM, so it
        # can neither invent gates (over-specification, EXP-1) nor fall back to an empty
        # stub (under-specification, EXP-1b). See _provided_series_stat_spec.
        spec = _provided_series_stat_spec(claim, source)
    elif claim_type == "predictive_regression":
        spec = _predictive_regression_spec(claim, source)
    elif claim_type == "factor_spanning":
        spec = _factor_spanning_spec(claim, source)
    elif claim_type == "cross_sectional_sort":
        spec = _cross_sectional_sort_spec(claim, source)
    elif claim_type == "event_study":
        spec = _event_study_spec(claim, source)
    elif claim_type == "forecast_skill":
        spec = _forecast_skill_spec(claim, source)
    elif claim_type == "formulaic_signal":
        spec = _formulaic_signal_spec(claim, source)
    elif not use_llm:
        spec = _stub_spec(claim, source)
    else:
        try:
            spec = _llm_spec(claim, source, claim_type=claim_type,
                             prior_divergences=prior_divergences)
        except Exception as e:  # noqa: BLE001
            # never let spec generation failure block the pipeline — fall back to stub
            spec = _stub_spec(claim, source)
            spec["_llm_error"] = str(e)
    spec.setdefault("claim_type", claim_type)
    spec["strategy_family"] = declared_strategy_family(
        claim, source, raw=spec.get("strategy_family"))

    spec["_generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec["_claim_id"] = claim.claim_id
    spec["_source_id"] = source.source_id
    spec["_path"] = str(out_path)

    _write_yaml(out_path, spec)
    return spec


def _immediate_correction_guidance(prior_divergences: list[str] | None) -> str:
    items = [str(d).strip() for d in (prior_divergences or []) if str(d).strip()]
    if not items:
        return ""
    bullets = "\n".join(f"- {item[:500]}" for item in items[:8])
    return (
        "YOUR PREVIOUS SPEC FOR THIS CLAIM WAS REJECTED AS UNFAITHFUL for these "
        "specific reasons:\n"
        f"{bullets}\n"
        "Produce a corrected spec that FIXES exactly these, WITHOUT introducing any new "
        "divergence. Do not over-correct into a different claim.\n"
    )


def _base_prompt(claim: Claim, source: IngestedSource, claim_type: str,
                 prior_divergences: list[str] | None = None) -> str:
    tmpl = {
        "descriptive_statistical": DESCRIPTIVE_SPEC_USER_TMPL,
        "structural_proposition": STRUCTURAL_SPEC_USER_TMPL,
        "predictive_regression": PREDICTIVE_REGRESSION_SPEC_USER_TMPL,
        "factor_spanning": FACTOR_SPANNING_SPEC_USER_TMPL,
        "cross_sectional_sort": CROSS_SECTIONAL_SORT_SPEC_USER_TMPL,
        "event_study": EVENT_STUDY_SPEC_USER_TMPL,
        "forecast_skill": FORECAST_SKILL_SPEC_USER_TMPL,
        "formulaic_signal": FORMULAIC_SIGNAL_SPEC_USER_TMPL,
    }.get(claim_type, TRADING_SPEC_USER_TMPL)
    user = tmpl.format(
        statement=claim.statement[:600],
        mechanism=(claim.mechanism or "")[:300],
        scope=(claim.scope or "")[:200],
        horizon=(claim.horizon or "")[:80],
        strategy_class=claim.applicable_strategy_class or "unspecified",
        claim_type=claim_type,
        source_span=claim.source_span[:400],
        claimed_metric=(claim.claimed_metric_quote or "")[:200],
    )
    guidance = fidelity_memory.rejection_guidance(
        claim.applicable_strategy_class or "unspecified", claim_type)
    if guidance:
        user = guidance + "\n" + user
    correction = _immediate_correction_guidance(prior_divergences)
    if correction:
        user = correction + "\n" + user
    vocab = _catalog_vocab()
    if vocab:
        user = (
            "AVAILABLE DATA SERIES (use these EXACT names in `inputs` when they fit the claim):\n"
            + vocab + "\n\n" + user
            + "\n\nUse inputs drawn from the AVAILABLE DATA SERIES list when one exactly satisfies "
            "the claim. If the claim requires an input that is absent from this list, do NOT "
            "fabricate, synthesize, derive, or proxy it from listed series. Instead, declare the "
            "true required input in `inputs` using a clear catalog-style name and list it in "
            "`unknowns` as unavailable."
        )
    return user


def _llm_spec(claim: Claim, source: IngestedSource, *, claim_type: str | None = None,
              prior_divergences: list[str] | None = None) -> dict:
    claim_type = claim_type or classify_claim_type(claim)
    user = _base_prompt(claim, source, claim_type, prior_divergences=prior_divergences)
    parsed, _ = llm.call_json(
        "module_spec_generator",
        [{"role": "system", "content": SPEC_SYSTEM},
         {"role": "user",   "content": user}],
        temperature=0.1,
    )
    # ensure required keys exist
    required = ["module_id", "strategy_class", "strategy_family", "claim_translation", "inputs",
                "signal_logic", "kill_criterion", "unknowns"]
    for k in required:
        if k not in parsed:
            parsed[k] = "" if k != "inputs" and k != "unknowns" else []
    # Audit D-4 (2026-07-05): claim_type is a GATE-SELECTION switch (fidelity override +
    # variable-coverage skip). It must come ONLY from the deterministic classifier —
    # never from the LLM's spec JSON, which an over-eager model could set to
    # provided_series_statistic and loosen two gates at once. Log any disagreement.
    llm_claim_type = parsed.get("claim_type")
    if llm_claim_type and llm_claim_type != claim_type:
        print(f"[penrose] spec LLM proposed claim_type={llm_claim_type!r}; "
              f"keeping deterministic {claim_type!r}")
    parsed["claim_type"] = claim_type
    parsed["version"] = 0
    parsed["status"] = "spec-only"
    parsed["source"] = source.source_id
    return parsed


_SERIES_DECISION_EXCLUSION_CUES = (
    "not part of the decision",
    "reported, not",
    "for reference",
    "benchmark",
    "control series",
    "control:",
    "comparison",
    "compared to",
    "baseline",
    "reference series",
    "context (",
    "descriptive only",
    "not a gate",
    "reported only",
    "informational",
    "for context",
    "as context",
    "contextual",
    "as reference",
    "we also report",
    "reported alongside",
    "illustrative",
)

_SERIES_RESOLVER_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "daily",
    "for",
    "from",
    "in",
    "index",
    "of",
    "on",
    "signal",
    "the",
    "to",
    "with",
}

_SINGLE_TOKEN_SERIES_GENERIC = {
    "abs",
    "change",
    "delta",
    "funding",
    "native",
    "perp",
    "price",
    "prices",
    "prob",
    "probability",
    "ret",
    "return",
    "returns",
    "rv",
    "signal",
    "spot",
    "vol",
    "volume",
}

_RELATION_VERBS_RE = re.compile(
    r"\b(?:predicts?|forecasts?|forecasting|leads?|explains?)\b",
    flags=re.IGNORECASE,
)
_RELATION_TRAILING_CUES_RE = re.compile(
    r"\b(?:in[- ]sample|out[- ]of[- ]sample|oos|is|with|regression|beta|coefficient|"
    r"t[- ]?stat(?:istic)?|p\s*=|r\s*(?:\^2|2)|msfe|sharpe|information ratio)\b.*$",
    flags=re.IGNORECASE,
)
_HORIZON_PREFIX_RE = re.compile(
    r"^\s*(?:next\s+)?(?:\d+\s*[- ]?)?"
    r"(?:d|day|days|period|periods|week|weeks|month|months|quarter|quarters|year|years)?"
    r"\s*(?:ahead|forward|future)?\s*",
    flags=re.IGNORECASE,
)


# Full crypto asset names as written in claims -> the ticker the catalog uses. Applied only as an
# ADDITIVE expansion (the full word is kept too), so a claim saying "Chainlink"/"Bitcoin" can bind to
# link_/btc_ series whose ticker is not a prefix of the full name. Deterministic; no model.
_ASSET_SYNONYMS = {
    "bitcoin": "btc", "ethereum": "eth", "solana": "sol", "ripple": "xrp",
    "dogecoin": "doge", "cardano": "ada", "chainlink": "link", "binance": "bnb",
    "polkadot": "dot", "avalanche": "avax", "litecoin": "ltc", "polygon": "matic",
}


def _series_resolver_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.split(r"[^a-z0-9]+", str(text or "").lower()):
        if not tok or tok in _SERIES_RESOLVER_STOPWORDS:
            continue
        out.append(tok)
        syn = _ASSET_SYNONYMS.get(tok)
        if syn and syn not in out:
            out.append(syn)
    return out


def _prefix_token_match(a: str, b: str) -> bool:
    return len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a))


def _single_content_token_binding_allowed(name_tokens: list[str], score: float) -> bool:
    if len(name_tokens) != 1 or score != 1.0:
        return False
    token = name_tokens[0]
    return len(token) >= 3 and token not in _SINGLE_TOKEN_SERIES_GENERIC


def resolve_series_from_prose(description: str, catalog_names: list[str]) -> dict | None:
    """Resolve prose to a catalog series by deterministic prefix-token coverage."""
    desc_tokens = _series_resolver_tokens(description)
    if not desc_tokens:
        return None
    best: tuple[float, int, str, list[tuple[str, str]], list[str]] | None = None
    for raw_name in sorted(str(n) for n in catalog_names if str(n or "").strip()):
        name_tokens = _series_resolver_tokens(raw_name)
        if not name_tokens:
            continue
        matches: list[tuple[str, str]] = []
        used_desc: set[int] = set()
        for name_tok in name_tokens:
            for idx, desc_tok in enumerate(desc_tokens):
                if idx in used_desc:
                    continue
                if _prefix_token_match(name_tok, desc_tok):
                    used_desc.add(idx)
                    matches.append((name_tok, desc_tok))
                    break
        score = len(matches) / float(len(name_tokens))
        matched_name = {name_tok for name_tok, _ in matches}
        unmatched_name = [tok for tok in name_tokens if tok not in matched_name]
        candidate = (score, len({d for _, d in matches}), raw_name, matches, unmatched_name)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    score, matched_desc_count, series, matches, unmatched_name = best
    name_tokens = _series_resolver_tokens(series)
    if score < 0.5 or (
        matched_desc_count < 2
        and not _single_content_token_binding_allowed(name_tokens, score)
    ):
        return None
    matched_tokens = [
        name_tok if name_tok == desc_tok else f"{name_tok}~{desc_tok}"
        for name_tok, desc_tok in matches
    ]
    full_coverage = not unmatched_name
    return {
        "series": series,
        "score": float(score),
        "matched_tokens": matched_tokens,
        "full_coverage": full_coverage,
        "unmatched_name_tokens": unmatched_name,
        "why": (
            f"matched {series} by prefix-token coverage "
            f"({len(matches)}/{len(name_tokens)}): "
            + ", ".join(matched_tokens)
            + ("" if full_coverage else "; unmatched name tokens: " + ", ".join(unmatched_name))
        ),
    }


def _resolve_signal_alias_from_prose(description: str, catalog_names: list[str]) -> dict | None:
    """Resolve terse '<entity> signal' prose to a unique abs-prob-change catalog signal."""
    raw = str(description or "").lower()
    if "signal" not in raw:
        return None
    desc_tokens = _series_resolver_tokens(description)
    if not desc_tokens:
        return None
    candidates = []
    for name in sorted(str(n) for n in catalog_names if str(n or "").strip()):
        name_tokens = _series_resolver_tokens(name)
        if not {"abs", "prob", "change"}.issubset(set(name_tokens)):
            continue
        entity_hits = [
            tok for tok in desc_tokens
            if any(_prefix_token_match(tok, name_tok) for name_tok in name_tokens)
        ]
        if entity_hits:
            candidates.append((name, entity_hits[0]))
    if len(candidates) != 1:
        return None
    series, entity = candidates[0]
    matched_tokens = [entity, "abs~signal", "prob~signal", "change~signal"]
    return {
        "series": series,
        "score": 1.0,
        "matched_tokens": matched_tokens,
        "full_coverage": True,
        "unmatched_name_tokens": [],
        "why": (
            f"matched unique abs_prob_change catalog signal {series} from terse "
            f"signal phrase: " + ", ".join(matched_tokens)
        ),
    }


def _resolve_spot_alias_from_prose(description: str, catalog_names: list[str]) -> dict | None:
    """Resolve terse asset prose such as 'Solana' to a unique '<asset>_spot_daily' series."""
    desc_tokens = _series_resolver_tokens(description)
    if not desc_tokens:
        return None
    candidates = []
    for name in sorted(str(n) for n in catalog_names if str(n or "").strip()):
        name_tokens = _series_resolver_tokens(name)
        if "spot" not in name_tokens:
            continue
        entity_hits = [
            tok for tok in desc_tokens
            if any(
                name_tok != "spot" and _prefix_token_match(tok, name_tok)
                for name_tok in name_tokens
            )
        ]
        if entity_hits:
            candidates.append((name, entity_hits[0]))
    if not candidates:
        return None
    # Disambiguate multiple spot variants for the SAME asset (e.g. sol_spot_daily / sol_okx_spot_daily /
    # price.sol_usd_spot_daily). Bail only when candidates span DIFFERENT asset tickers (genuinely
    # ambiguous); otherwise prefer the canonical bare "<ticker>_spot_daily" (no dotted namespace, no
    # exchange qualifier, shortest).
    def _spot_ticker(name: str) -> str | None:
        toks = [t for t in _series_resolver_tokens(name)
                if t not in ("spot", "daily", "usd", "price")]
        return toks[0] if toks else None
    tickers = {_spot_ticker(n) for n, _ in candidates}
    tickers.discard(None)
    if len(tickers) != 1:
        return None
    def _canonical_rank(item: tuple[str, str]) -> tuple[int, int]:
        name = item[0]
        canonical = ("." not in name) and name.endswith("_spot_daily")
        return (0 if canonical else 1, len(name))
    series, entity = min(candidates, key=_canonical_rank)
    name_entity = next(
        (tok for tok in _series_resolver_tokens(series) if tok != "spot"),
        entity,
    )
    matched_tokens = [
        name_entity if name_entity == entity else f"{name_entity}~{entity}",
        "spot~asset",
    ]
    return {
        "series": series,
        "score": 1.0,
        "matched_tokens": matched_tokens,
        "full_coverage": True,
        "unmatched_name_tokens": [],
        "why": (
            f"matched unique spot catalog series {series} from asset phrase: "
            + ", ".join(matched_tokens)
        ),
    }


def _horizon_to_int_for_spec(value: int | str) -> int:
    if isinstance(value, int):
        return max(1, value)
    m = re.search(r"\d+", str(value or ""))
    return max(1, int(m.group(0))) if m else 1


def _trim_series_phrase(text: str) -> str:
    # Keep commas: a series is often named in an appositive clause ("the inflation channel, measured by
    # CPI repricing on KXCPI contracts") whose identifying tokens sit AFTER the comma. Truncating at the
    # comma dropped them and left an unresolvable head. The resolver scores by token coverage (extra words
    # are harmless) and BIND-1's measurement-conflict gate guards against a wrong bind, so keeping the full
    # clause up to sentence-ending punctuation is safe and recovers paraphrase predictors.
    phrase = re.split(r"[.;:()\[\]\n]", str(text or ""), maxsplit=1)[0]
    phrase = _RELATION_TRAILING_CUES_RE.sub("", phrase)
    phrase = re.sub(
        r"\b(?:for|over|at)\s+(?:the\s+)?(?:next\s+)?\d+\s*[- ]?"
        r"(?:d|day|days|period|periods|week|weeks|month|months|quarter|quarters|year|years)"
        r"(?:\s+(?:horizon|ahead|forward|future))?\b",
        "",
        phrase,
        flags=re.IGNORECASE,
    )
    phrase = _HORIZON_PREFIX_RE.sub("", phrase)
    return " ".join(phrase.split()).strip(" -")


def _derived_base_phrase(description: str, match: re.Match) -> str:
    before = _trim_series_phrase(description[:match.start()])
    after = _trim_series_phrase(description[match.end():])
    m = re.match(r"^\s*(?:of|for|on|in)\s+(.+)$", after, flags=re.IGNORECASE)
    if m:
        return _trim_series_phrase(m.group(1))
    if before:
        return before
    return after


def resolve_derived_series(description: str, catalog_names: list[str],
                           horizon: int | str) -> dict | None:
    """Resolve prose describing a deterministic derived series."""
    text = str(description or "")
    horizon_int = _horizon_to_int_for_spec(horizon)
    for transform, pattern in (
        ("realized_vol", r"\b(?:realized\s+vol(?:atility)?|rv)\b"),
        ("log_returns", r"\blog\s+returns?\b"),
        ("returns", r"\breturns?\b"),
    ):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        base_phrase = _derived_base_phrase(text, m)
        base = (
            resolve_series_from_prose(base_phrase, catalog_names)
            or _resolve_spot_alias_from_prose(base_phrase, catalog_names)
        )
        if base is None:
            continue
        out = {
            "kind": "derived_series",
            "transform": transform,
            "base_series": base["series"],
            "why": f"{transform} derived from {base['series']}; {base['why']}",
            "base_resolution": base,
            "full_coverage": bool(base.get("full_coverage")),
            "unmatched_name_tokens": list(base.get("unmatched_name_tokens") or []),
        }
        if transform == "realized_vol":
            out["window"] = horizon_int
            out["horizon_encoded"] = True
        return out
    return None


def _series_text_units(text: str) -> list[str]:
    """Text units for the decision-exclusion scan. Split on BLANK-line paragraph
    boundaries and JOIN soft-wrapped lines within each paragraph first, so a cue like
    "Context (reported, not part of the decision):" stays attached to the series names
    that wrap onto the following physical line(s). Emit each joined paragraph AND its
    sentences, so a series named in a context paragraph/sentence is correctly excluded."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        joined = " ".join(para.split())     # collapse soft wraps + whitespace
        if not joined:
            continue
        units.append(joined)
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip())
    return units


def _declared_series_from_text(claim: Claim, source: IngestedSource | None = None) -> list[str]:
    """Deterministically pull any catalog series names that literally appear in the
    claim/source text, excluding names that appear only in context/reference units.
    No invention, no LLM: a name is included only if it is BOTH a real catalog series
    AND already named by the claim/source -- the same discipline as the #5a
    catalog-vocabulary guard, applied without a model in the loop. Fail-open to [] on
    any catalog error."""
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        names = list(catalog.available()) if hasattr(catalog, "available") else []
    except Exception:  # noqa: BLE001
        return []
    text = "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])
    matched = set()
    for unit in _series_text_units(text):
        lowered = unit.lower()
        if any(cue in lowered for cue in _SERIES_DECISION_EXCLUSION_CUES):
            continue
        matched.update(
            n for n in names
            if n and re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", unit)
        )
    return sorted(matched)


def _literal_series_in_text(text: str, catalog_names: list[str]) -> str:
    for name in sorted(catalog_names):
        if name and re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            return name
    return ""


def _predictive_relation_phrases(claim: Claim, source: IngestedSource | None = None) -> tuple[str, str, str]:
    texts = [
        getattr(claim, "statement", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ]
    combined = " ".join(t for t in texts if t)
    for raw in texts + [combined]:
        if not raw:
            continue
        m = _RELATION_VERBS_RE.search(raw)
        if not m:
            continue
        left = raw[:m.start()]
        right = raw[m.end():]
        left_units = re.split(r"[.;:\n]", left)
        predictor = _trim_series_phrase(left_units[-1] if left_units else left)
        target = _trim_series_phrase(right)
        if predictor and target:
            return predictor, target, raw
    return "", "", combined


def _catalog_names() -> list[str]:
    try:
        catalog = load_catalog_loader(config.DATA_DIR)
        return list(catalog.available()) if hasattr(catalog, "available") else []
    except Exception:  # noqa: BLE001
        return []


def _provided_series_stat_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic (non-LLM) ModuleSpec for a `provided_series_statistic` claim (6g):
    "test the statistic of a provided/pre-computed series." This claim class needs no
    creative translation -- pool the claim's own declared series and apply exactly the
    claim's own stated decision rule. Building it as a template rather than an LLM
    generation is what makes both failure modes structurally impossible: there is no
    generation step to invent a p<=0.05 gate, a data-quality kill, or a menu of deflation
    methods (over-specification, EXP-1), and no generation step to fall back to an empty
    stub (under-specification, EXP-1b).
    """
    inputs = _declared_series_from_text(claim, source)
    decision_rule = (claim.claimed_metric_quote or claim.statement or "").strip()
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": claim.applicable_strategy_class or "unspecified",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "provided_series_statistic",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic one-sample test on the claim's own declared/provided series: pool "
            "ALL declared series into one sample (no re-grouping, no sub-bucketing), compute "
            "the sample mean and its one-sample test/CI, and apply EXACTLY the claim's stated "
            "decision rule. No positions, no PnL simulation, and no extra pass/fail filters "
            "or deflation method beyond what the claim itself declares."
        ),
        "inputs": inputs,
        "statistic_logic": (
            "one_sample_mean_test: pool the declared series into one sample; compute the mean, "
            "its standard error, and a one-sample test against the claim's own stated decision "
            "rule. If the claim declares a single cohort/deflation family, use exactly that one "
            "method; never offer a menu of methods, and never add extra pass/fail filters the "
            "claim did not state."
        ),
        "signal_logic": "provided_series_statistic: compute the statistic directly; no positions/PnL.",
        "kill_criterion": (
            f"the claim's own stated decision rule is not met: {decision_rule}"[:400]
            if decision_rule else
            "the claim's own stated decision rule is not met (see claim_statement)"
        ),
        "expected_data_needs": "as declared by the claim; no additional window/frequency invented",
        "unknowns": (
            [] if inputs else
            ["no catalog series literally matched the claim text; operator must confirm the "
             "series names before implementing"]
        ),
        "implementation_notes": (
            "Deterministically generated (no LLM) for a provided-series-statistics claim: this "
            "claim type is a mechanical translation, not a creative one -- preserve the stated "
            "decision rule exactly."
        ),
        "_llm_mode": "deterministic-template",
    }


def _predictive_horizon_from_text(claim: Claim) -> int | str:
    text = " ".join([
        getattr(claim, "horizon", "") or "",
        getattr(claim, "statement", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(claim, "source_span", "") or "",
    ]).lower()
    for pat in (
        r"\b(\d+)\s*(?:d|day|days|period|periods)[- ]ahead\b",
        r"\bh\s*=\s*(\d+)\b",
        r"\b(\d+)\s*(?:d|day|days)\s+horizon\b",
    ):
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return (getattr(claim, "horizon", "") or "1").strip() or 1


def _declared_statistic_from_text(claim: Claim) -> str:
    text = " ".join([
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(claim, "statement", "") or "",
        getattr(claim, "source_span", "") or "",
    ])
    cues = []
    for pat, label in (
        (r"\bt\s*[-=]\s*-?\d+(?:\.\d+)?\b|\bt[- ]stat(?:istic)?\b", "t_statistic"),
        (r"\br\s*(?:\^2|2)\b|\br-squared\b", "r2"),
        (r"\bmsfe\b", "msfe"),
        (r"\bcoefficient\b|\bbeta\b", "coefficient"),
        (r"\bsign\b|negative|positive", "directional_sign"),
    ):
        if re.search(pat, text, flags=re.IGNORECASE):
            cues.append(label)
    return ", ".join(cues) if cues else "directional predictive relationship"


def _predictive_declared_search_grid(claim: Claim) -> dict:
    text = " ".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(claim, "source_span", "") or "",
    ]).lower()
    widths = []
    for pat in (
        r"\b(\d+)\s*[- ]?(?:coin|coins|asset|assets|market|markets)\b",
        r"\b(\d+)\s*[- ]?(?:spec|specification|specifications|model|models)\b",
        r"\b(?:search|grid|tested across|across)\s+(\d+)\b",
    ):
        for m in re.finditer(pat, text):
            try:
                widths.append(int(m.group(1)))
            except ValueError:
                pass
    total = 1
    for width in widths:
        if 1 < width <= 10000:
            total *= width
    if total <= 1:
        return {}
    return {"declared_regression_search": list(range(min(total, 10000)))}


def _factor_spanning_text(claim: Claim, source: IngestedSource | None = None) -> str:
    return "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])


def _factor_benchmark_set_from_text(text: str) -> str:
    low = str(text or "").lower()
    if re.search(r"\bff\s*5\b|\bfama[- ]french\s+5\b|\bfive[- ]factor\b|\brmw\b|\bcma\b", low):
        return "ff5"
    if re.search(r"\bcarhart\b", low):
        return "carhart"
    if re.search(r"\bff\s*3\b|\bfama[- ]french\s+3\b|\bthree[- ]factor\b|\bsmb\b|\bhml\b", low):
        return "ff3"
    if re.search(r"\bcapm\b|\bmarket\s+factor\b|\bmkt[- ]rf\b", low):
        return "capm"
    return "ff3"


def _factor_benchmark_series(benchmark_set: str) -> list[str]:
    return {
        "capm": ["us_equity_ff3_mkt_rf"],
        "ff3": ["us_equity_ff3_mkt_rf", "us_equity_ff3_smb", "us_equity_ff3_hml"],
        "ff5": [
            "us_equity_ff5_mkt_rf",
            "us_equity_ff5_smb",
            "us_equity_ff5_hml",
            "us_equity_ff5_rmw",
            "us_equity_ff5_cma",
        ],
        "carhart": [
            "us_equity_ff3_mkt_rf",
            "us_equity_ff3_smb",
            "us_equity_ff3_hml",
            "us_equity_momentum_wml",
        ],
    }.get(benchmark_set, [])


def _factor_candidate_phrase(claim: Claim, source: IngestedSource | None = None) -> str:
    texts = [
        getattr(claim, "statement", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ]
    combined = " ".join(t for t in texts if t)
    patterns = (
        r"(?P<candidate>.+?)\s+(?:earns?|has|delivers?|shows?)\s+(?:a\s+)?(?:positive\s+)?alpha\b",
        r"(?:alpha|intercept)\s+(?:of|for)\s+(?P<candidate>.+?)\s+(?:after|controlling|net\b)",
        r"regress(?:ing|ion)?\s+(?P<candidate>.+?)\s+on\s+.+?\bfactors?\b",
    )
    for raw in texts + [combined]:
        if not raw:
            continue
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                return _trim_series_phrase(m.group("candidate"))
    return _trim_series_phrase(combined)


def _factor_declared_statistic_from_text(claim: Claim) -> str:
    text = _factor_spanning_text(claim)
    cues = []
    for pat, label in (
        (r"\bt\s*[-=]\s*-?\d+(?:\.\d+)?\b|\bt[- ]stat(?:istic)?\b", "alpha_t_stat"),
        (r"\bintercept\b", "intercept"),
        (r"\balpha\b", "alpha"),
        (r"\br\s*(?:\^2|2)\b|\br-squared\b", "r2"),
    ):
        if re.search(pat, text, flags=re.IGNORECASE):
            cues.append(label)
    return ", ".join(cues) if cues else "alpha_t_stat"


def _factor_declared_search_grid(claim: Claim, benchmark_set: str) -> dict:
    text = _factor_spanning_text(claim).lower()
    grid: dict[str, list] = {}
    candidate_counts = []
    for pat in (
        r"\b(\d+)\s*[- ]?(?:candidate\s+)?factors?\b",
        r"\b(\d+)\s*[- ]?(?:anomal(?:y|ies)|signals?)\b",
        r"\b(?:search|grid|tested across|across)\s+(\d+)\b",
    ):
        for m in re.finditer(pat, text):
            try:
                candidate_counts.append(int(m.group(1)))
            except ValueError:
                pass
    if candidate_counts:
        n = max(1, min(max(candidate_counts), 10000))
        if n > 1:
            grid["candidate_factors"] = list(range(n))
    declared_sets = []
    for name, pat in (
        ("capm", r"\bcapm\b"),
        ("ff3", r"\bff\s*3\b|\bfama[- ]french\s+3\b|\bthree[- ]factor\b"),
        ("ff5", r"\bff\s*5\b|\bfama[- ]french\s+5\b|\bfive[- ]factor\b"),
        ("carhart", r"\bcarhart\b"),
    ):
        if re.search(pat, text):
            declared_sets.append(name)
    if not declared_sets:
        declared_sets = [benchmark_set]
    if len(declared_sets) > 1:
        grid["benchmark_sets"] = declared_sets
    return grid


def _factor_spanning_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for factor-spanning claims."""
    full_text = _factor_spanning_text(claim, source)
    catalog_names = _catalog_names()
    benchmark_set = _factor_benchmark_set_from_text(full_text)
    benchmark_factors = _factor_benchmark_series(benchmark_set)
    candidate_phrase = _factor_candidate_phrase(claim, source)
    binding_provenance: dict[str, dict] = {}
    benchmark_set_prov = {
        "kind": "benchmark_set",
        "benchmark_set": benchmark_set,
        "series": list(benchmark_factors),
        "description": benchmark_set,
        "score": 1.0,
        "confirmed": True,
        "why": f"declared benchmark family resolved to {benchmark_set}",
    }
    binding_provenance["benchmark_set"] = benchmark_set_prov
    for name in benchmark_factors:
        binding_provenance[name] = {
            "kind": "default_benchmark",
            "series": name,
            "score": 1.0,
            "confirmed": True,
            "why": f"default {benchmark_set} benchmark factor",
        }

    benchmark_set_names = set(benchmark_factors)
    candidate = _literal_series_in_text(candidate_phrase, catalog_names)
    if candidate in benchmark_set_names:
        candidate = ""
    if candidate:
        binding_provenance["candidate_factor"] = {
            "kind": "literal",
            "series": candidate,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(candidate),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": candidate_phrase,
            "why": f"literal catalog series {candidate} appeared in candidate phrase",
        }
    else:
        candidate_resolution = (
            resolve_series_from_prose(candidate_phrase, catalog_names)
            or resolve_series_from_prose(full_text, catalog_names)
        )
        if candidate_resolution and candidate_resolution.get("series") not in benchmark_set_names:
            candidate = candidate_resolution["series"]
            binding_provenance["candidate_factor"] = {
                "kind": "prose",
                "description": candidate_phrase,
                **candidate_resolution,
            }
    if not candidate:
        binding_provenance["candidate_factor"] = {
            "kind": "unresolved",
            "description": candidate_phrase,
            "confirmed": False,
            "why": "candidate factor prose did not resolve to a catalog series",
        }

    inputs = ([candidate] if candidate else []) + benchmark_factors
    statistic = _factor_declared_statistic_from_text(claim)
    param_grid = _factor_declared_search_grid(claim, benchmark_set)
    unknowns = []
    if not candidate:
        unknowns.append(
            "candidate factor series was not resolved from literal/prose bindings; "
            "operator must confirm candidate_factor"
        )
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "factor_spanning",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "factor_spanning",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            f"Deterministic factor-spanning test: regress candidate factor F on the declared "
            f"{benchmark_set.upper()} benchmark factors using the in-sample prefix only, freeze "
            "the beta vector, and test whether the benchmark-hedged residual alpha survives "
            "OOS and holdout through the normal P7/P8 stack."
        ),
        "inputs": inputs,
        "candidate_factor": candidate,
        "benchmark_set": benchmark_set,
        "benchmark_factors": benchmark_factors,
        "binding_provenance": binding_provenance,
        "estimator": "multivariate_ols_is_frozen_betas",
        "statistic": statistic,
        "param_grid": param_grid,
        "signal_logic": (
            "factor_spanning: fit F_t = alpha + beta'B_t on IS only; emit "
            "F_t - beta_IS'B_t as net residual alpha; no trading overlay"
        ),
        "kill_criterion": (
            "OOS/holdout residual alpha fails, 3-fold sign stability fails, or the deflated "
            "alpha statistic fails the verdict band"
        ),
        "expected_data_needs": "candidate and benchmark factor return Series on a common DatetimeIndex",
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Use plain Series only. "
            "Freeze all benchmark betas on the in-sample prefix and never refit on OOS "
            "or holdout rows. bars_per_year is the emitted factor-return observation rate."
        ),
        "_llm_mode": "deterministic-template",
    }


def _cross_sectional_text(claim: Claim, source: IngestedSource | None = None) -> str:
    return "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])


def _cross_sectional_characteristic_from_text(text: str) -> str:
    patterns = (
        r"\bsort(?:ed|s|ing)?\s+(?:stocks?|assets?|firms?|entities|securities)?\s*(?:by|on)\s+(?P<c>[^.;,()]+)",
        r"\brank(?:ed|s|ing)?\s+(?:stocks?|assets?|firms?|entities|securities)?\s*(?:by|on)\s+(?P<c>[^.;,()]+)",
        r"\b(?P<c>book[- ]to[- ]market|momentum|size|market\s+cap(?:italization)?|quality|profitability|investment|accruals?|asset\s+growth|leverage)\b",
    )
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            phrase = _trim_series_phrase(m.group("c"))
            phrase = re.split(
                r"\b(?:into|to\s+form|earns?|earn|has|shows?|delivers?|produces?|with)\b",
                phrase,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            return phrase.strip(" -")
    return ""


def _safe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_") or "characteristic"


_BUCKET_WORDS = [
    (r"\bdecile(?:s)?\b|\btop\s+10%\b|\bbottom\s+10%\b", 10),
    (r"\bvigintile(?:s)?\b|\btop\s+5%\b|\bbottom\s+5%\b", 20),
    (r"\boctile(?:s)?\b", 8),
    (r"\bsextile(?:s)?\b", 6),
    (r"\bquintile(?:s)?\b|\btop\s+20%\b|\bbottom\s+20%\b", 5),
    (r"\bquartile(?:s)?\b|\btop\s+25%\b|\bbottom\s+25%\b", 4),
    (r"\btercile(?:s)?\b|\btertile(?:s)?\b|\btop\s+third\b|\bbottom\s+third\b", 3),
]


def _claimed_bucket_count(text: str) -> int | None:
    """The bucket scheme the claim EXPLICITLY names, or None if it names none (so silence never triggers a
    substitution rejection in the fidelity correspondence check)."""
    low = str(text or "").lower()
    for pat, n in _BUCKET_WORDS:
        if re.search(pat, low):
            return n
    m = re.search(r"\b(\d+)\s+(?:bucket|portfolio|group|fractile)s?\b", low)
    if m:
        try:
            return max(2, int(m.group(1)))
        except ValueError:
            pass
    return None


def _claimed_rebalance(text: str) -> str | None:
    """The rebalance cadence the claim EXPLICITLY names, or None if silent."""
    low = str(text or "").lower()
    if re.search(r"\bmonthly\b|\beach\s+month\b", low):
        return "ME"
    if re.search(r"\bquarterly\b|\beach\s+quarter\b", low):
        return "QE"
    if re.search(r"\bannually\b|\byearly\b|\beach\s+year\b|\bannual\s+rebalanc", low):
        return "YE"
    if re.search(r"\bweekly\b|\beach\s+week\b", low):
        return "W"
    if re.search(r"\bdaily\b|\beach\s+day\b", low):
        return "D"
    return None


def _cross_sectional_n_buckets(text: str) -> int:
    return _claimed_bucket_count(text) or 10


def _cross_sectional_rebalance(text: str) -> str:
    return _claimed_rebalance(text) or "ME"


def _cross_sectional_declared_search_grid(claim: Claim, n_buckets: int) -> dict:
    text = _cross_sectional_text(claim).lower()
    grid: dict[str, list] = {}
    bucket_counts = []
    if re.search(r"\bdecile", text):
        bucket_counts.append(10)
    if re.search(r"\bquintile", text):
        bucket_counts.append(5)
    for m in re.finditer(r"\b(\d+)\s+(?:bucket|portfolio|group|fractile)s?\b", text):
        try:
            bucket_counts.append(max(2, int(m.group(1))))
        except ValueError:
            pass
    bucket_counts = sorted(set(bucket_counts))
    if len(bucket_counts) > 1:
        grid["n_buckets"] = bucket_counts
    elif bucket_counts and bucket_counts[0] != n_buckets:
        grid["n_buckets"] = [n_buckets, bucket_counts[0]]

    characteristic_counts = []
    for pat in (
        r"\b(\d+)\s*(?:candidate\s+)?characteristics?\b",
        r"\b(\d+)\s*(?:anomal(?:y|ies)|signals?)\b",
        r"\b(?:search|grid|tested across|across)\s+(\d+)\b",
    ):
        for m in re.finditer(pat, text):
            try:
                characteristic_counts.append(int(m.group(1)))
            except ValueError:
                pass
    if characteristic_counts:
        n = max(1, min(max(characteristic_counts), 10000))
        if n > 1:
            grid["characteristics"] = list(range(n))
    return grid


def _cross_sectional_sort_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for cross-sectional sort claims."""
    full_text = _cross_sectional_text(claim, source)
    characteristic = _cross_sectional_characteristic_from_text(full_text)
    n_buckets = _cross_sectional_n_buckets(full_text)
    rebalance = _cross_sectional_rebalance(full_text)
    hold = "1M" if rebalance in {"ME", "M"} else "1Q" if rebalance in {"QE", "Q"} else "1Y" if rebalance in {"YE", "Y"} else rebalance
    returns_table = "returns_panel"
    characteristic_key = _safe_key(characteristic) if characteristic else "characteristic"
    characteristic_table = f"{characteristic_key}_panel"
    binding_provenance = {
        "returns_panel": {
            "kind": "declared_panel",
            "panel": returns_table,
            "confirmed": False,
            "why": "returns panel table must be supplied by the operator/catalog",
        },
        "characteristic": {
            "kind": "prose" if characteristic else "unresolved",
            "description": characteristic or "unresolved characteristic",
            "characteristic": characteristic,
            "confirmed": bool(characteristic),
            "why": (
                "characteristic phrase extracted from the structural sort claim"
                if characteristic else
                "characteristic prose did not resolve"
            ),
        },
        "characteristic_panel": {
            "kind": "declared_panel",
            "panel": characteristic_table,
            "confirmed": False,
            "why": "point-in-time characteristic panel table must be supplied by the operator/catalog",
        },
    }
    unknowns = [
        "returns and characteristic panel tables must be confirmed; returns must declare survivorship=corrected"
    ]
    if not characteristic:
        unknowns.append("characteristic was not resolved from the sort claim")
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "cross_sectional_sort",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "cross_sectional_sort",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic cross-sectional sort test: sort the declared entity universe by "
            f"{characteristic or 'the declared characteristic'} at each {rebalance} rebalance, "
            f"form the top-minus-bottom {n_buckets}-bucket spread using only characteristics "
            "known as of the rebalance date, and send that spread through P7/P8."
        ),
        "inputs": [],
        "panel_inputs": {
            "returns": {"table": returns_table, "survivorship": "corrected"},
            "characteristic": {"table": characteristic_table},
        },
        "characteristic": characteristic,
        "n_buckets": n_buckets,
        "rebalance": rebalance,
        "hold": hold,
        "binding_provenance": binding_provenance,
        "estimator": "characteristic_sorted_top_minus_bottom",
        "statistic": "mean_spread",
        "param_grid": _cross_sectional_declared_search_grid(claim, n_buckets),
        "signal_logic": (
            "cross_sectional_sort: form_factor(returns_panel, characteristic_panel, "
            "n_buckets, rebalance, hold); no trading overlay"
        ),
        "kill_criterion": (
            "OOS/holdout top-minus-bottom spread fails, 3-fold sign stability fails, "
            "or the deflated spread statistic fails the verdict band"
        ),
        "expected_data_needs": (
            "survivorship-corrected returns Panel and point-in-time characteristic Panel "
            "with dates x entity columns"
        ),
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Reuse data.xsection.form_factor; "
            "it ranks using sig.loc[:rebalance] only. The returns panel must retain delisted "
            "entities during their live windows."
        ),
        "_llm_mode": "deterministic-template",
    }


def _event_study_text(claim: Claim, source: IngestedSource | None = None) -> str:
    return "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])


def _claimed_event_window(text: str) -> tuple[int, int] | None:
    """Return the event window explicitly named by the claim, or None if silent."""
    low = str(text or "").lower()
    for pat in (
        r"\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]\s*(?:day|days)?\s*window",
        r"window\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]",
        r"\bevent\s+window\s*(?:of|=|:)?\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]",
    ):
        m = re.search(pat, low)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start <= end:
                return start, end
    m = re.search(r"\b(?:over|during)\s+(?:the\s+)?(\d+)\s*(?:day|trading[- ]day)s?\s+after\b", low)
    if m:
        return 0, max(0, int(m.group(1)) - 1)
    return None


def _claimed_baseline(text: str) -> str | None:
    """Return the baseline explicitly named by the claim, or None if silent."""
    low = str(text or "").lower()
    if re.search(r"\bmarket[- ]model\b|\bmarket\s+adjusted\b|\bcapm[- ]adjusted\b", low):
        return "market_model"
    if re.search(r"\bmean[- ]adjusted\b|\bconstant[- ]mean\b|\bhistorical\s+mean\b", low):
        return "mean_adjusted"
    return None


def _event_estimation_window(text: str) -> int:
    low = str(text or "").lower()
    for pat in (
        r"\bestimation\s+window\s*(?:of|=|:)?\s*(\d+)",
        r"\b(\d+)\s*(?:day|trading[- ]day)s?\s+estimation\s+window\b",
        r"\busing\s+(\d+)\s*(?:pre[- ]event|prior)\s*(?:day|trading[- ]day)s?\b",
    ):
        m = re.search(pat, low)
        if m:
            return max(5, int(m.group(1)))
    return 60


def _event_return_phrase(claim: Claim, source: IngestedSource | None = None) -> str:
    texts = [
        getattr(claim, "statement", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ]
    for raw in texts:
        if not raw:
            continue
        m = re.search(
            r"\babnormal\s+returns?\s+(?:of|for|on)\s+(?P<asset>[^.;,()\[\]]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            return _trim_series_phrase(m.group("asset"))
    return _trim_series_phrase(" ".join(t for t in texts if t))


def _event_calendar_phrase(text: str) -> str:
    patterns = (
        r"\b(?P<event>earnings\s+announcements?|fomc\s+announcements?|index\s+additions?|halvings?|listings?)\b",
        r"\b(?:around|following|after)\s+(?P<event>[^.;,()\[\]]+?)\s+(?:events?|announcements?)\b",
        r"\b(?P<event>[^.;,()\[\]]+?)\s+(?:event\s+calendar|events?|announcements?)\b",
    )
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _trim_series_phrase(m.group("event"))
    return "event_calendar"


def _event_declared_search_grid(claim: Claim, window: tuple[int, int], baseline: str) -> dict:
    text = _event_study_text(claim).lower()
    grid: dict[str, list] = {}
    windows = []
    for m in re.finditer(r"\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]", text):
        start, end = int(m.group(1)), int(m.group(2))
        if start <= end:
            windows.append([start, end])
    windows = [w for i, w in enumerate(windows) if w not in windows[:i]]
    if len(windows) > 1:
        grid["windows"] = windows
    baselines = []
    if re.search(r"\bmean[- ]adjusted\b|\bconstant[- ]mean\b|\bhistorical\s+mean\b", text):
        baselines.append("mean_adjusted")
    if re.search(r"\bmarket[- ]model\b|\bmarket\s+adjusted\b|\bcapm[- ]adjusted\b", text):
        baselines.append("market_model")
    baselines = [b for i, b in enumerate(baselines) if b not in baselines[:i]]
    if len(baselines) > 1:
        grid["baselines"] = baselines
    event_counts = []
    for pat in (
        r"\b(\d+)\s*(?:event\s+)?windows?\b",
        r"\b(\d+)\s*baselines?\b",
        r"\b(?:search|grid|tested across|across)\s+(\d+)\b",
    ):
        for m in re.finditer(pat, text):
            event_counts.append(max(1, min(int(m.group(1)), 10000)))
    if event_counts and max(event_counts) > 1:
        grid["declared_event_study_specs"] = list(range(max(event_counts)))
    if not grid and (list(window) != [0, 5] or baseline != "mean_adjusted"):
        return {}
    return grid


def _event_study_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for event-study claims."""
    full_text = _event_study_text(claim, source)
    catalog_names = _catalog_names()
    return_phrase = _event_return_phrase(claim, source)
    binding_provenance: dict[str, dict] = {}
    return_series = _literal_series_in_text(return_phrase, catalog_names)
    if return_series:
        binding_provenance["return_series"] = {
            "kind": "literal",
            "series": return_series,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(return_series),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": return_phrase,
            "why": f"literal catalog series {return_series} appeared in return-series phrase",
        }
    else:
        resolution = (
            resolve_series_from_prose(return_phrase, catalog_names)
            or resolve_series_from_prose(full_text, catalog_names)
        )
        if resolution:
            return_series = resolution["series"]
            binding_provenance["return_series"] = {
                "kind": "prose",
                "description": return_phrase,
                **resolution,
            }
    if not return_series:
        binding_provenance["return_series"] = {
            "kind": "unresolved",
            "description": return_phrase,
            "confirmed": False,
            "why": "return-series prose did not resolve to a catalog series",
        }

    window = _claimed_event_window(full_text) or (0, 5)
    baseline = _claimed_baseline(full_text) or "mean_adjusted"
    estimation_window = _event_estimation_window(full_text)
    calendar_phrase = _event_calendar_phrase(full_text)
    calendar_table = f"{_safe_key(calendar_phrase)}_event_calendar"
    binding_provenance["event_calendar"] = {
        "kind": "declared_table",
        "table": calendar_table,
        "description": calendar_phrase,
        "confirmed": False,
        "why": "event calendar table must be supplied by the operator/catalog",
    }
    inputs = [return_series] if return_series else []
    market_series = ""
    if baseline == "market_model":
        market_series = _literal_series_in_text(full_text, catalog_names)
        if market_series and market_series == return_series:
            market_series = ""
        if market_series:
            inputs.append(market_series)
    unknowns = ["event calendar table must be confirmed and supplied with a date column"]
    if not return_series:
        unknowns.append("return series was not resolved from literal/prose bindings")
    if baseline == "market_model" and not market_series:
        unknowns.append("market_model baseline requires a market_series binding")
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "event_study",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "event_study",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic event-study test: estimate the declared baseline strictly "
            "before each event, compute abnormal returns over the declared event window, "
            "emit one CAR observation per event, and test average CAR through P7/P8."
        ),
        "inputs": inputs,
        "return_series": return_series,
        "event_calendar": {"table": calendar_table, "date_col": "date"},
        "window": list(window),
        "estimation_window": int(estimation_window),
        "baseline": baseline,
        "market_series": market_series,
        "binding_provenance": binding_provenance,
        "statistic": "average_car",
        "param_grid": _event_declared_search_grid(claim, window, baseline),
        "signal_logic": (
            "event_study: fit mean_adjusted or market_model baseline on rows strictly "
            "before event date; emit CAR_e over window; no trading overlay"
        ),
        "kill_criterion": (
            "OOS/holdout average CAR fails, 3-fold sign stability fails, or the deflated "
            "CAR statistic fails the verdict band"
        ),
        "expected_data_needs": "return Series plus declared event-calendar table with a date column",
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Baseline estimation uses "
            "only the fixed pre-event window ending strictly before the event date. "
            "bars_per_year is events per year, not 252 calendar bars."
        ),
        "_llm_mode": "deterministic-template",
    }


def _forecast_skill_text(claim: Claim, source: IngestedSource | None = None) -> str:
    return "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])


def _forecast_skill_phrases(claim: Claim, source: IngestedSource | None = None) -> tuple[str, str, str, str]:
    texts = [
        getattr(claim, "statement", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ]
    combined = " ".join(t for t in texts if t)
    model_phrase = ""
    target_phrase = ""
    benchmark_phrase = ""
    for raw in texts + [combined]:
        if not raw:
            continue
        m = re.search(
            r"(?P<model>[^.;:\n]+?)\s+(?:forecasts?|predicts?|prediction\s+of)\s+"
            r"(?P<target>[^.;:\n]+?)\s+(?:better\s+than|beats?|outperforms?|relative\s+to|against)\s+"
            r"(?P<bench>[^.;:\n]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            return (
                _trim_series_phrase(m.group("model")),
                _trim_series_phrase(m.group("target")),
                _trim_series_phrase(m.group("bench")),
                raw,
            )
        m = re.search(
            r"(?P<model>[^.;:\n]+?)\s+(?:beats?|outperforms?)\s+(?:the\s+)?"
            r"(?P<bench>random[- ]walk|benchmark|naive|historical\s+mean)[^.;:\n]*?"
            r"(?:forecast(?:s|ing)?|for|on|of)\s+(?P<target>[^.;:\n]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            return (
                _trim_series_phrase(m.group("model")),
                _trim_series_phrase(m.group("target")),
                _trim_series_phrase(m.group("bench")),
                raw,
            )
        if not target_phrase:
            m = re.search(
                r"(?:forecast(?:s|ing)?|predict(?:s|ing)?)\s+(?P<target>[^.;:\n]+?)\s+"
                r"(?:out[- ]of[- ]sample|oos|with|using|by)\b",
                raw,
                flags=re.IGNORECASE,
            )
            if m:
                target_phrase = _trim_series_phrase(m.group("target"))
    if not model_phrase:
        model_phrase = _trim_series_phrase(combined)
    if not target_phrase:
        target_phrase = _trim_series_phrase(combined)
    return model_phrase, target_phrase, benchmark_phrase or combined, combined


def _forecast_benchmark_method(text: str) -> str:
    low = str(text or "").lower()
    if re.search(r"\bhistorical\s+mean\b|\bexpanding\s+mean\b", low):
        return "historical_mean"
    if re.search(r"\brandom[- ]walk\b|\bnaive\b|\bpersistence\b|\blast\s+value\b", low):
        return "random_walk"
    return ""


def _forecast_declared_statistic_from_text(claim: Claim) -> str:
    text = _forecast_skill_text(claim)
    cues = []
    for pat, label in (
        (r"\bdiebold[- ]mariano\b", "diebold_mariano"),
        (r"\bclark[- ]west\b", "clark_west"),
        (r"\bmsfe\b", "msfe"),
        (r"\bmspe\b", "mspe"),
        (r"\brmse\b", "rmse"),
        (r"\bout[- ]of[- ]sample\s+r\s*(?:\^2|2)\b|\boos\s+r\s*(?:\^2|2)\b", "oos_r2"),
    ):
        if re.search(pat, text, flags=re.IGNORECASE):
            cues.append(label)
    return ", ".join(cues) if cues else "loss_differential_mean"


def _forecast_declared_search_grid(claim: Claim, benchmark_method: str, explicit_benchmark: str) -> dict:
    text = _forecast_skill_text(claim).lower()
    grid: dict[str, list] = {}
    counts = []
    for pat in (
        r"\b(\d+)\s*[- ]?(?:forecasting\s+)?models?\b",
        r"\b(\d+)\s*[- ]?(?:spec|specification|specifications)\b",
        r"\b(?:search|grid|tested across|across)\s+(\d+)\b",
    ):
        for m in re.finditer(pat, text):
            counts.append(max(1, min(int(m.group(1)), 10000)))
    if counts and max(counts) > 1:
        grid["declared_forecast_models"] = list(range(max(counts)))
    benchmarks = []
    if explicit_benchmark:
        benchmarks.append(explicit_benchmark)
    if benchmark_method:
        benchmarks.append(benchmark_method)
    if re.search(r"\brandom[- ]walk\b|\bnaive\b|\bpersistence\b", text) and "random_walk" not in benchmarks:
        benchmarks.append("random_walk")
    if re.search(r"\bhistorical\s+mean\b|\bexpanding\s+mean\b", text) and "historical_mean" not in benchmarks:
        benchmarks.append("historical_mean")
    if len(benchmarks) > 1:
        grid["benchmarks"] = benchmarks
    return grid


def _forecast_skill_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for forecast-skill claims."""
    full_text = _forecast_skill_text(claim, source)
    catalog_names = _catalog_names()
    model_phrase, target_phrase, benchmark_phrase, relation_text = _forecast_skill_phrases(claim, source)
    binding_provenance: dict[str, dict] = {}

    model_forecast = _literal_series_in_text(model_phrase, catalog_names)
    if model_forecast:
        binding_provenance["model_forecast"] = {
            "kind": "literal",
            "series": model_forecast,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(model_forecast),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": model_phrase,
            "why": f"literal catalog series {model_forecast} appeared in model-forecast phrase",
        }
    else:
        model_resolution = (
            resolve_series_from_prose(model_phrase, catalog_names)
            or _resolve_signal_alias_from_prose(model_phrase, catalog_names)
            or resolve_series_from_prose(relation_text, catalog_names)
            or _resolve_signal_alias_from_prose(relation_text, catalog_names)
        )
        if model_resolution:
            model_forecast = model_resolution["series"]
            binding_provenance["model_forecast"] = {
                "kind": "prose",
                "description": model_phrase,
                **model_resolution,
            }
    if not model_forecast:
        binding_provenance["model_forecast"] = {
            "kind": "unresolved",
            "description": model_phrase,
            "confirmed": False,
            "why": "model-forecast prose did not resolve to a catalog series",
        }

    target = _literal_series_in_text(target_phrase, catalog_names)
    if target:
        binding_provenance["target"] = {
            "kind": "literal",
            "series": target,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(target),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": target_phrase,
            "why": f"literal catalog series {target} appeared in target phrase",
        }
    else:
        target_resolution = (
            resolve_series_from_prose(target_phrase, catalog_names)
            or resolve_series_from_prose(relation_text, catalog_names)
            or resolve_series_from_prose(full_text, catalog_names)
        )
        if target_resolution:
            target = target_resolution["series"]
            binding_provenance["target"] = {
                "kind": "prose",
                "description": target_phrase,
                **target_resolution,
            }
    if not target:
        binding_provenance["target"] = {
            "kind": "unresolved",
            "description": target_phrase,
            "confirmed": False,
            "why": "target prose did not resolve to a catalog series",
        }

    benchmark_method = _forecast_benchmark_method(benchmark_phrase or full_text)
    explicit_benchmark = ""
    benchmark = {}
    if not benchmark_method:
        benchmark_resolution = resolve_series_from_prose(benchmark_phrase, catalog_names)
        if benchmark_resolution and benchmark_resolution.get("series") not in {model_forecast, target}:
            explicit_benchmark = benchmark_resolution["series"]
            binding_provenance["benchmark"] = {
                "kind": "prose",
                "description": benchmark_phrase,
                **benchmark_resolution,
            }
    if explicit_benchmark:
        benchmark = explicit_benchmark
    elif benchmark_method:
        benchmark = {"kind": "implied", "method": benchmark_method}
        binding_provenance["benchmark"] = {
            "kind": "implied",
            "method": benchmark_method,
            "confirmed": True,
            "why": f"declared implied benchmark resolved to {benchmark_method}",
        }
    else:
        binding_provenance["benchmark"] = {
            "kind": "unresolved",
            "description": benchmark_phrase,
            "confirmed": False,
            "why": "benchmark must be an explicit series or a declared implied random_walk/historical_mean",
        }

    inputs = []
    if model_forecast:
        inputs.append(model_forecast)
    if target:
        inputs.append(target)
    if explicit_benchmark:
        inputs.append(explicit_benchmark)
    unknowns = []
    if not model_forecast or not target:
        unknowns.append(
            "model forecast and target series were not both resolved from literal/prose bindings"
        )
    if not benchmark:
        unknowns.append(
            "benchmark forecast was not resolved; declare an explicit benchmark series or "
            "an implied random_walk/historical_mean benchmark"
        )
    statistic = _forecast_declared_statistic_from_text(claim)
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "forecast_skill",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "forecast_skill",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic forecast-skill test: compare the declared model forecast F_t "
            "with the declared benchmark forecast B_t on the declared realized target Y_t, "
            "emit the squared-loss differential (B_t-Y_t)^2 - (F_t-Y_t)^2, and test "
            "whether positive forecast skill survives OOS and holdout through P7/P8."
        ),
        "inputs": inputs,
        "model_forecast": model_forecast,
        "target": target,
        "benchmark": benchmark,
        "binding_provenance": binding_provenance,
        "loss": "squared_error",
        "statistic": statistic,
        "param_grid": _forecast_declared_search_grid(claim, benchmark_method, explicit_benchmark),
        "signal_logic": (
            "forecast_skill: emit (B_t - Y_t)^2 - (F_t - Y_t)^2 as the loss "
            "differential; positive means model beats benchmark; no trading overlay"
        ),
        "kill_criterion": (
            "OOS/holdout loss differential fails, 3-fold sign stability fails, or the "
            "deflated Diebold-Mariano-style statistic fails the verdict band"
        ),
        "expected_data_needs": (
            "model forecast and realized target Series on a common DatetimeIndex; explicit "
            "benchmark Series only when the claim declares one"
        ),
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Constructed benchmarks are "
            "strictly causal: random_walk uses Y.shift(1), historical_mean uses "
            "expanding_mean(Y).shift(1). v1 uses the DM loss differential; Clark-West for "
            "nested forecasts is a future refinement."
        ),
        "_llm_mode": "deterministic-template",
    }


def _formulaic_text(claim: Claim, source: IngestedSource | None = None) -> str:
    return "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])


def _formulaic_clean_value(value: str) -> str:
    text = str(value or "").strip().strip(",").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        return text[1:-1].strip()
    return text


def _formulaic_field(text: str, field: str) -> str:
    pat = rf"\b{re.escape(field)}\s*[:=]\s*(?P<value>[^\n;]+)"
    m = re.search(pat, text, flags=re.IGNORECASE)
    return _formulaic_clean_value(m.group("value")) if m else ""


def _formulaic_signal_expr(text: str) -> str:
    m = re.search(
        r"\bsignal\s*(?:formula|dsl)?\s*[:=]\s*(?P<value>[^\n;]+)",
        text,
        flags=re.IGNORECASE,
    )
    return _formulaic_clean_value(m.group("value")) if m else ""


def _formulaic_position_map(text: str) -> str:
    raw = _formulaic_field(text, "position_map").lower().replace("-", "_")
    if raw.startswith("zscore_clip"):
        return "zscore_clip"
    return "sign"


def _formulaic_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for explicitly declared formulaic-signal claims."""
    full_text = _formulaic_text(claim, source)
    signal = _formulaic_signal_expr(full_text)
    trade_series = _formulaic_field(full_text, "trade_series")
    funding_pnl_series = _formulaic_field(full_text, "funding_pnl_series")
    position_map = _formulaic_position_map(full_text)
    formula_names: set[str] = set()
    unknowns: list[str] = []
    try:
        from . import formulaic_signal

        formula_names = formulaic_signal.referenced_names(signal)
    except Exception as exc:  # noqa: BLE001
        unknowns.append(f"signal formula did not parse under formulaic_signal DSL: {exc}")
    inputs = {str(x or "").strip() for x in formula_names if str(x or "").strip()}
    if trade_series:
        inputs.add(trade_series)
    else:
        unknowns.append("trade_series was not declared")
    if funding_pnl_series:
        inputs.add(funding_pnl_series)
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "formulaic_signal",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "formulaic_signal",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic formulaic signal test: parse the declared DSL formula into S_t, "
            "map to P_t, apply one-bar-lagged exposure to trade_series returns, subtract "
            "turnover costs, and include optional funding cash-flow in the executor."
        ),
        "inputs": sorted(inputs),
        "trade_series": trade_series,
        "signal": signal,
        "position_map": position_map,
        "funding_pnl_series": funding_pnl_series or None,
        "binding_provenance": {
            "signal_names": sorted(formula_names),
            "trade_series": {"series": trade_series, "confirmed": bool(trade_series)},
            "funding_pnl_series": {
                "series": funding_pnl_series or None,
                "confirmed": bool(funding_pnl_series),
            },
        },
        "param_grid": {},
        "signal_logic": (
            "formulaic_signal: parse DSL; P_t=position_map(S_t); "
            "net_t=P_(t-1)*ret_t - turnover_costs; if funding_pnl_series is declared, "
            "net_t += -P_(t-1)*funding_t"
        ),
        "kill_criterion": (
            "OOS/holdout net return fails, 3-fold sign stability fails, or the deflated "
            "net-return statistic fails the verdict band"
        ),
        "expected_data_needs": "DatetimeIndex Series for every input, with trade_series as prices",
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Parser whitelist is ast "
            "Expression/BinOp/UnaryOp/Call/Name/Constant with the audited operator table. "
            "No look-ahead: executor applies the one-bar position delay exactly once."
        ),
        "_llm_mode": "deterministic-template",
    }


def _predictive_regression_spec(claim: Claim, source: IngestedSource) -> dict:
    """Deterministic ModuleSpec for predictive-regression claims.

    The template binds claim prose to catalog series only through deterministic,
    recorded literal/prose/derived-series proposals. The operator can review
    predictor/target ordering before the trusted deterministic executor runs; no
    trading overlay or LLM-generated code is introduced.
    """
    horizon = _predictive_horizon_from_text(claim)
    catalog_names = _catalog_names()
    predictor_phrase, target_phrase, relation_text = _predictive_relation_phrases(claim, source)
    full_text = "\n".join([
        getattr(claim, "statement", "") or "",
        getattr(claim, "mechanism", "") or "",
        getattr(claim, "source_span", "") or "",
        getattr(claim, "claimed_metric_quote", "") or "",
        getattr(source, "text", "") if source is not None else "",
    ])
    binding_provenance: dict[str, dict] = {}

    predictor = _literal_series_in_text(predictor_phrase, catalog_names)
    if predictor:
        binding_provenance["predictor"] = {
            "kind": "literal",
            "series": predictor,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(predictor),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": predictor_phrase,
            "why": f"literal catalog series {predictor} appeared in predictor phrase",
        }
    else:
        predictor_resolution = (
            resolve_series_from_prose(predictor_phrase, catalog_names)
            or _resolve_signal_alias_from_prose(predictor_phrase, catalog_names)
            or resolve_series_from_prose(relation_text, catalog_names)
            or _resolve_signal_alias_from_prose(relation_text, catalog_names)
            or resolve_series_from_prose(full_text, catalog_names)
            or _resolve_signal_alias_from_prose(full_text, catalog_names)
        )
        if predictor_resolution:
            predictor = predictor_resolution["series"]
            binding_provenance["predictor"] = {
                "kind": "prose",
                "description": predictor_phrase,
                **predictor_resolution,
            }
    if not predictor:
        binding_provenance["predictor"] = {
            "kind": "unresolved",
            "description": predictor_phrase,
            "confirmed": False,
            "why": "predictor prose did not resolve to a catalog series",
        }

    target = _literal_series_in_text(target_phrase, catalog_names)
    if target:
        binding_provenance["target"] = {
            "kind": "literal",
            "series": target,
            "score": 1.0,
            "matched_tokens": _series_resolver_tokens(target),
            "full_coverage": True,
            "unmatched_name_tokens": [],
            "description": target_phrase,
            "why": f"literal catalog series {target} appeared in target phrase",
        }
    else:
        target_resolution = (
            resolve_derived_series(target_phrase, catalog_names, horizon)
            or resolve_derived_series(relation_text, catalog_names, horizon)
            or resolve_derived_series(full_text, catalog_names, horizon)
        )
        if target_resolution:
            target = target_resolution
            binding_provenance["target"] = {
                **target_resolution,
                "kind": "derived",
                "description": target_phrase,
            }
        else:
            prose_target = (
                resolve_series_from_prose(target_phrase, catalog_names)
                or resolve_series_from_prose(relation_text, catalog_names)
            )
            if prose_target:
                target = prose_target["series"]
                binding_provenance["target"] = {
                    "kind": "prose",
                    "description": target_phrase,
                    **prose_target,
                }
    if not target:
        binding_provenance["target"] = {
            "kind": "unresolved",
            "description": target_phrase,
            "confirmed": False,
            "why": "target prose did not resolve to a catalog series",
        }

    inputs = []
    if predictor:
        inputs.append(predictor)
    if target:
        inputs.append(target)
    statistic = _declared_statistic_from_text(claim)
    param_grid = _predictive_declared_search_grid(claim)
    unknowns = []
    if not predictor or not target:
        unknowns.append(
            "predictor and target series were not both resolved from literal/prose bindings; "
            "operator must confirm inputs[0]=predictor and inputs[1]=target"
        )
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": "predictive_regression",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": "predictive_regression",
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": (
            "Deterministic predictive-regression test: align the declared predictor X_t "
            f"with declared target Y_t+h at horizon {horizon}, fit the relationship sign "
            "and standardization moments on the in-sample prefix only, and test whether "
            "the predictive direction survives OOS and holdout through the normal P7/P8 stack."
        ),
        "inputs": inputs[:2],
        "predictor": predictor,
        "target": target,
        "binding_provenance": binding_provenance,
        "horizon": horizon,
        "estimator": "single_predictor_ols_covariance_sign",
        "statistic": statistic,
        "param_grid": param_grid,
        "signal_logic": (
            "predictive_regression: emit s*zscore_IS(X_t)*zscore_IS(Y_t+h), with "
            "positions=s*zscore_IS(X_t); no trading overlay"
        ),
        "kill_criterion": (
            "OOS/holdout predictive direction fails, 3-fold sign stability fails, or "
            "the deflated predictive-return statistic fails the verdict band"
        ),
        "expected_data_needs": "predictor and target Series on a common DatetimeIndex",
        "unknowns": unknowns,
        "implementation_notes": (
            "Deterministic trusted module; no Docker/impl_gen. Emit non-overlapping "
            "h-sampled observations at aligned positions 0, h, 2h, ...; freeze sign "
            "and z-score moments on in-sample rows whose target timestamps remain "
            "in-sample. bars_per_year is the emitted observation rate, with no second "
            "division by horizon."
        ),
        "_llm_mode": "deterministic-template",
    }


def _stub_spec(claim: Claim, source: IngestedSource) -> dict:
    claim_type = classify_claim_type(claim)
    if claim_type == "descriptive_statistical":
        translation = (
            "OFFLINE STUB — operator must compute and test the descriptive statistic "
            "stated by the claim directly. Re-run with PENROSE_LLM_API_KEY set to get "
            "an LLM-generated translation."
        )
        signal_logic = "descriptive_statistical: compute the stated statistic directly; no positions/PnL."
        kill_criterion = "The stated statistic does not replicate with appropriate uncertainty bounds."
    elif claim_type == "structural_proposition":
        translation = "OFFLINE STUB — operator must determine whether this structural proposition is operationalizable."
        signal_logic = "structural_proposition: cannot create positions unless explicitly claimed."
        kill_criterion = "cannot_operationalize unless a direct falsification test is specified"
    else:
        translation = (
            "OFFLINE STUB — operator must translate this claim to a tradeable test. "
            "Re-run with PENROSE_LLM_API_KEY set to get an LLM-generated translation."
        )
        signal_logic = "OFFLINE STUB — not generated."
        kill_criterion = "OOS DSR/PSR < 0.90 OR 3-fold sign-unstable OR edge_t < 1.0"
    return {
        "module_id": f"auto_{claim.claim_id}",
        "version": 0,
        "status": "spec-only",
        "strategy_class": claim.applicable_strategy_class or "unspecified",
        "strategy_family": declared_strategy_family(claim, source),
        "claim_type": claim_type,
        "source": source.source_id,
        "claim_statement": claim.statement,
        "claim_source_span": claim.source_span,
        "claim_mechanism": claim.mechanism,
        "claim_translation": translation,
        "inputs": [],
        "signal_logic": signal_logic,
        "kill_criterion": kill_criterion,
        "unknowns": ["auto-generated stub spec; operator review required"],
        "_llm_mode": "stub",
    }


def _write_yaml(path: Path, spec: dict) -> None:
    # strip internal keys starting with _ for the YAML body; keep _path separate
    body = {k: v for k, v in spec.items() if not k.startswith("_")}
    header = (
        f"# auto-generated ModuleSpec (F10) — review and implement\n"
        f"# claim: {spec.get('_claim_id', '?')} · source: {spec.get('_source_id', '?')}\n"
        f"# generated: {spec.get('_generated_at', '?')}\n\n"
    )
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False))
