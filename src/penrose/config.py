"""Central paths + tunable constants for penrose v1.

Fee-curve constants live here with provenance + effective date. The harness
is reused for its instrument-agnostic statistics (DSR / PSR /
Sharpe / capacity); penrose's P7 supplies the strategy P&L because the paper's
instrument is volatility, not a crypto perp.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # repository root (src/penrose/config.py -> ../..)


def _load_dotenv() -> None:
    """Auto-load ROOT/.env so a fresh clone 'just works' after pasting a key —
    no manual `source` needed. Existing env vars win (never clobber a real export).
    Tiny KEY=VALUE parser; no python-dotenv dependency."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
# PEN-17: generative layer master switch. OFF by default until the M1 verdict recalibration
# lands and the decision corpus is re-scored. Enable explicitly: PENROSE_GENERATIVE_LAYER=1
GENERATIVE_LAYER_ENABLED = os.environ.get("PENROSE_GENERATIVE_LAYER", "0").lower() in (
    "1", "true", "yes")
BRAIN_HOME = ROOT / ".brain"
BRAINSTORE_DB = ROOT / ".brainstore" / "atoms.db"
ARCHIVES = ROOT / "archives"
DREAM_ARCHIVES = ARCHIVES / "dreams"
INBOX = ROOT / "inbox"
REPORTS = ROOT / "reports"
MODULES = ROOT / "modules"
# Provenance shelf: machine-generated (auto-implemented, UNTRUSTED) modules live HERE, separate
# from operator-curated trusted modules at MODULES/. The leading underscore makes
# _register_known_modules skip the whole shelf (it never cross-routes auto modules anyway), so
# operator-curated and machine-generated corpora can't be confused. (ROADMAP: partition by provenance.)
AUTO_MODULES = MODULES / "_auto"
DATA_CACHE = ROOT / ".data_cache"
LLM_CACHE_DIR = ROOT / ".llm_cache"
DECISIONS_LOG = ROOT / "decisions.jsonl"               # decisions log (atoms also go to brain)
PRINCIPLES_LOG = ROOT / "principles.jsonl"
PROPOSALS_LOG = REPORTS / "proposals.jsonl"            # propose-only, never P9-approved knowledge
REVIEW_QUEUE = ROOT / "review_queue.jsonl"             # P9 Action Required queue
DATA_REQUESTS = ROOT / "data_requests.jsonl"          # backlog: data a claim needs but the catalog lacks
PROCESSED_PAPERS = ROOT / "processed_papers.json"     # filenames already run, so the loop advances through inbox/
DREAM_RUNS = ROOT / "dream_runs.jsonl"                # registered generator searches + lifecycle summaries
PROGRESS_JSON = ROOT / "dashboard" / "progress.json"  # live per-stage progress for the dashboard activity panel
ANALYSIS_INDEX = ROOT / "reports" / "analysis_index.jsonl"  # backtested outcomes + chart paths for the Reports page
FIDELITY_REJECTIONS = ROOT / "reports" / "fidelity_rejections.jsonl"
CONCEPTS = REPORTS / "concepts.jsonl"
CORPUS_GRAPH = REPORTS / "corpus_graph.jsonl"
CORPUS_JSON = ROOT / "dashboard" / "corpus.json"
SYNTHESIS_ARCHIVES = ARCHIVES / "syntheses"
SYNTHESIS_RUNS = ROOT / "synthesis_runs.jsonl"
LIVE_JSON = ROOT / "dashboard" / "live.json"           # what the read-only dash server injects


def ensure_output_dirs() -> None:
    """Create first-run output directories. Safe to call repeatedly."""
    for path in (
        REPORTS,
        REPORTS / "charts",
        LIVE_JSON.parent,
        LLM_CACHE_DIR,
        ARCHIVES,
    ):
        path.mkdir(parents=True, exist_ok=True)

# --- LLM provider config --------------------------------------------------- #
# All roles default to GLM 5x. Swap by editing PENROSE_LLM_* env vars or
# per-role overrides below. Single OpenAI-compatible adapter serves every
# provider via PENROSE_LLM_BASE_URL (Artificial Analysis, OpenAI, z.ai compat,
# Ollama, LiteLLM proxy, etc.).
DEFAULT_LLM_MODEL = os.environ.get("PENROSE_LLM_DEFAULT_MODEL", "glm-5.2")
VERIFIER_LLM_MODEL = os.environ.get("PENROSE_LLM_VERIFIER_MODEL", DEFAULT_LLM_MODEL)
VERIFIER_LLM_BASE_URL = os.environ.get("PENROSE_LLM_VERIFIER_BASE_URL", "")
VERIFIER_LLM_API_KEY = os.environ.get("PENROSE_LLM_VERIFIER_API_KEY", "")

LLM_ROLES = {
    # NOTE: glm-5.2 is a THINKING model — reasoning tokens count against max_tokens. A cap
    # that's fine for the JSON content alone returns an EMPTY body once reasoning eats it
    # (seen on large papers + a big controlled-vocab prompt). Caps below leave room for
    # reasoning + content. See pipeline/extract.py and llm.call_json.
    # bulk / cheap roles — small context, low cost
    "claim_extractor":          {"model": DEFAULT_LLM_MODEL, "max_tokens": 16000,
                                 "max_cost_per_call": 0.50},
    "falsifiability_classifier":{"model": DEFAULT_LLM_MODEL, "max_tokens": 6000,
                                 "max_cost_per_call": 0.10},
    "module_spec_generator":    {"model": DEFAULT_LLM_MODEL, "max_tokens": 12000,
                                 "max_cost_per_call": 0.50},
    "module_implementer":       {"model": DEFAULT_LLM_MODEL, "max_tokens": 16000,
                                 "max_cost_per_call": 0.60},
    # adversarial VERIFY gate — checks a module faithfully implements its claim. Configure an
    # independent judge with PENROSE_LLM_VERIFIER_BASE_URL/API_KEY/MODEL; unset preserves the
    # default provider/model path.
    "fidelity_refuter":         {"model": VERIFIER_LLM_MODEL, "max_tokens": 4000,
                                 "max_cost_per_call": 0.30},
    "concept_extractor":        {"model": DEFAULT_LLM_MODEL, "max_tokens": 5000,
                                 "max_cost_per_call": 0.30},
    "concept_grounding_refuter":{"model": DEFAULT_LLM_MODEL, "max_tokens": 4000,
                                 "max_cost_per_call": 0.30},
    # frontier-tier roles — bigger context, more expensive
    "deep_reader":              {"model": DEFAULT_LLM_MODEL, "max_tokens": 10000,
                                 "max_cost_per_call": 0.80},
    "dreamer":                  {"model": DEFAULT_LLM_MODEL, "max_tokens": 8000,
                                 "max_cost_per_call": 0.50},
    "synthesizer":              {"model": DEFAULT_LLM_MODEL, "max_tokens": 5000,
                                 "max_cost_per_call": 0.40},
    "qual_lens_default":        {"model": DEFAULT_LLM_MODEL, "max_tokens": 3000,
                                 "max_cost_per_call": 0.20},
    "chat_preflight":           {"model": DEFAULT_LLM_MODEL, "max_tokens": 1500,
                                 "max_cost_per_call": 0.05},
    "chat_assistant":           {"model": DEFAULT_LLM_MODEL, "max_tokens": 2500,
                                 "max_cost_per_call": 0.10},
}

# Rough USD per million tokens (used for budget enforcement only; not invoiced).
# Replace with live pricing on first swap.
LLM_PRICING = {
    "glm-5.2":          {"in_per_m": 0.50, "out_per_m": 1.50},
    "glm-4.6":          {"in_per_m": 0.30, "out_per_m": 0.90},
    "gpt-4o":           {"in_per_m": 2.50, "out_per_m": 10.00},
    "gpt-4o-mini":      {"in_per_m": 0.15, "out_per_m": 0.60},
    "claude-3-5-haiku": {"in_per_m": 0.80, "out_per_m": 4.00},
    "claude-sonnet-4":  {"in_per_m": 3.00, "out_per_m": 15.00},
    "__default__":      {"in_per_m": 1.00, "out_per_m": 3.00},
}

LLM_BUDGET = {
    "max_usd_per_day": float(os.environ.get("PENROSE_LLM_MAX_USD_DAY", "40.0")),
    "_note": "hard daily cap (F6-lite); raises RuntimeError if exceeded",
}

LLM_TIMEOUTS = {
    "default": int(os.environ.get("PENROSE_LLM_TIMEOUT_DEFAULT", "90")),
    "fidelity_refuter": int(os.environ.get("PENROSE_LLM_TIMEOUT_FIDELITY_REFUTER", "150")),
}

CLAIM_TIME_BUDGET_SECONDS = float(os.environ.get("PENROSE_CLAIM_TIME_BUDGET_SECONDS", "0") or 0)

# Optional local data catalog for PRE-COLLECTED series that have no free live API
# (e.g. historical Kalshi macro signals). Bring-your-own: point PENROSE_DATA_DIR at a
# directory exposing a loader for the data contract; when it is absent, modules fall
# back to the clearly-tagged synthetic generator. Live venues (Coinbase/Kraken/Deribit)
# are keyless and need no catalog; vendor adapters (e.g. Databento) use their own API key.
DATA_DIR = Path(os.environ.get("PENROSE_DATA_DIR", "") or (ROOT.parent / "penrose-data"))

# Optional external analysis venv for vendor adapters; empty unless explicitly configured.
PMA_VENV_PY = Path(os.environ.get("PENROSE_PMA_VENV_PY", "") or (ROOT / ".nonexistent"))

SCOPE = "penrose_pm"                                     # knowledge-store source_id / scope

# --- Fee curve: Polymarket-style dynamic fee, peaks at 50c, ~0 at tails ---
# Kalshi's own fee on these macro contracts is quadratic_with_maker_fees (observed
# fee_type on the KXFED series). We model the worst-case round-trip taker fee with
# the same p(1-p) shape the harness uses, calibrated per-venue below.
FEE_CURVE = {
    "polymarket": {"fee_rate": 0.07, "C": 1.0, "_note": "crypto category ~1.75% peak"},
    "kalshi":     {"fee_rate": 0.07, "C": 1.0, "_note": "quadratic_with_maker_fees; modeled p(1-p)"},
    "_effective_date": "2026-06-18",
    "_provenance": "modeled fee curve (p(1-p) shape), calibrated per venue",
}

# Deribit BTC options round-trip cost for the tradeable-vol translation (taker
# fee capped at 12.5% of option premium + ~exchange/settlement). Modeled as a
# flat fraction of vega notional per round trip; refined with real fills later.
VOL_TRADE_COST = {
    "deribit_roundtrip_bps_of_vega": 8.0,   # bps of vega-notional per round trip
    "_note": "placeholder; replace with paper-traded fills (S8) before any capital",
}

# Capacity: linear market-impact, bps of extra slippage per $1M traded (reuses
# the harness _capacity_usd model). BTC vol (Deribit) is thinner than spot.
IMPACT_COEF_BPS_PER_1M = 25.0

DSR_DECISION = {
    "kill_below_psr": 0.90,        # PSR/DSR below this on OOS -> kill
    "watch_band": (0.90, 0.95),    # ambiguous -> watch / Action Required
    "min_oos_bars": 30,            # OOS trades below this -> insufficient_data (used directly, no //2; A-005)
}
HOLDOUT_CONFIRM_PSR = DSR_DECISION["kill_below_psr"]

# Power-aware verdict labeling. A null result on data too thin to
# RESOLVE a realistic edge is "below the detection floor", NOT "proven dead" — emitting one token
# ("kill") for dead / underpowered / mis-built makes a rigorous skeptic indistinguishable from a
# broken always-no machine. MDE (minimum detectable IC) ~ z / sqrt(n_oos) for a single-asset
# strategy (per-bar Sharpe ~= IC). A non-structural null with MDE above the realistic floor is
# relabeled `underpowered`, not `kill`. Structural kills (look-ahead, regime-fragile, no
# signal-alignment, walk-forward drift) are power-INDEPENDENT and stand.
POWER = {"realistic_ic_floor": 0.05,   # the effect size we want to be able to resolve (real daily IC 0.02-0.05)
         "z_certify": 1.645,           # one-sided 95% — the band a verdict must clear to certify
         # PEN-01: the all-signs-positive 3-fold test is only a STRUCTURAL kill when it had
         # adequate power against a realistic edge. Below this power, a failed 3-fold is an
         # ambiguous null (-> underpowered), unless a fold is SIGNIFICANTLY negative.
         "three_fold_min_power": 0.60,
         "structural_fold_t": -1.0}

# PEN-05: external claims must demonstrate post-sample evidence before a survivor verdict is
# trusted. If the bundle does not extend at least `min_post_years` beyond the claim's own
# sample_period end (or the claim did not declare one), a would-be survivor is capped at watch.
POST_SAMPLE = {"enabled": True, "min_post_years": 1.0}

# PEN-06: Harvey-Liu-style effective-trials prior. An external claimant's search is invisible;
# assume a conservative prior search size rather than n=1. sr_var_prior is the assumed
# cross-trial variance of PER-TRADE Sharpe (units match per_trade_sharpe in the ledger);
# it is blended with the empirical variance until >= min_scored trials exist.
DEFLATION_PRIOR = {
    "external_min_trials": 10,
    "generated_min_trials": 1,
    "sr_var_prior": 0.01,
    "min_scored_for_empirical_var": 3,
}

# Empirical robustness layer (Monte-Carlo + walk-forward). Analytic DSR/PSR are
# asymptotic; these add small-sample / fat-tail honesty (see pipeline/robustness.py).
BOOTSTRAP = {"n_boot": 2000, "ci": 0.90, "block": None, "seed": 0}
PERMUTATION = {"n_perm": 2000, "seed": 0}
# PEN-02: regime fragility must be statistically distinguishable from noise-selection.
# The drop-the-best statistic is compared against a permutation null (bucket labels shuffled).
REGIME_FRAGILITY = {"n_perm": 500, "p_kill": 0.05, "seed": 0}
WALK_FORWARD = {"n_windows": 4, "scheme": "anchored", "is_min": 0.30}
CPCV = {"n_groups": 8, "k_test": 2, "embargo_frac": 0.01,
        "max_combos": 200, "seed": 0,
        "overfit_prob_kill": 0.50,
        "min_paths": 6}
ROBUSTNESS_GATES = {
    "kill_if_edge_ci_includes_zero": True,   # bootstrap edge CI straddles 0 -> kill
    "permutation_kill_p": 0.10,              # data-snooping p above this -> kill
    "kill_if_regime_fragile": True,          # edge concentrated in one calendar regime -> kill
    "kill_if_walk_forward_inconsistent": True,  # B-007: walk-forward drift -> kill (independent axis)
    "kill_if_cpcv_overfit": True,            # CPCV loss distribution kills only borderline survivors
}
REGIME_ADHERENCE_MIN = 0.60

# Advisory by default: report how much higher modeled costs can go before a survivor flips.
# Enabling this turns the advisory into a stricter verdict gate for isolated tests/experiments.
COST_SENSITIVITY_GATE = {"enabled": False, "min_margin": 1.5}

# Tail-risk / widow-maker gate: kills (or caps) a stable, well-deflated strategy whose payoff is
# bounded-up / unbounded-down. DEFAULT-OFF so it never silently moves an existing verdict.
TAIL_RISK_GATE = {"enabled": False, "max_skew": -0.5, "min_tail_ratio": 3.0, "cap_only": False}

# E2: all penrose economics (fee curve, impact coef, vol-trade cost) are MODELED placeholders,
# not measured fills. While this is "modeled", a survivor is structurally capped at `watch`
# (never research-supported). Flip to "measured" only when real paper-traded costs exist.
COST_PROVENANCE = "modeled"

# Auto-implement a generated ModuleSpec into a working module (close the
# cold-start loop). The generated code is validated by running on the live bundle
# before it's registered; on any failure the claim stays pending_module.
AUTO_IMPLEMENT_MODULES = True

# Relevance gate (pre-P2): cheap LLM screen of the abstract vs penrose's data domains;
# off-domain papers are skipped before the expensive stages. Fails open. (pipeline/relevance.py)
RELEVANCE_GATE = True

# Fidelity refuter (post-backtest VERIFY): adversarial check that a module faithfully
# implements its claim. An unfaithful module's verdict is untrustworthy — flag it, and
# never let an unfaithful module reach research-supported. Fails open. (pipeline/fidelity.py)
FIDELITY_CHECK = True
FIDELITY_KILL_CONFIDENCE = 0.6     # unfaithful at >= this confidence -> flag + cap verdict trust

# Advisory corpus promotion. These thresholds never participate in P8.
CORPUS_MIN_SUPPORT = int(os.environ.get("PENROSE_CORPUS_MIN_SUPPORT", "3"))
CORPUS_HALF_LIFE_YEARS = float(os.environ.get("PENROSE_CORPUS_HALF_LIFE_YEARS", "4"))

# The generator is never shown the confirmation reserve. Confirmation requires one distinct,
# held-aside epoch per candidate; no default/sentinel epoch is treated as real data.
try:
    CONFIRMATION_RESERVE = __import__("json").loads(os.environ.get(
        "PENROSE_CONFIRMATION_RESERVE",
        '{"reserve_id":"unconfigured","epochs":[]}'))
except Exception:
    CONFIRMATION_RESERVE = {"reserve_id": "invalid-reserve", "epochs": []}

# Databento BYO market data (data/databento.py). Point-in-time, survivorship-aware data —
# upgrades backtest fidelity (attacks look-ahead). BYO: set DATABENTO_API_KEY + `pip install
# databento`, then map a logical bundle key -> a Databento request below. Empty by default;
# fill in only series your account is entitled to. Each becomes a bundle series modules can
# request; absent without a key/entitlement -> honest needs_data, never a crash. Example:
#   "spx_es_front_daily": {"dataset": "GLBX.MDP3", "symbol": "ES.c.0", "schema": "ohlcv-1d",
#                          "field": "close", "stype_in": "continuous",
#                          "start": "2018-01-01", "end": "2025-12-31"},
DATABENTO_SERIES: dict = {}

# Calibration tolerance. The edge /
# Sharpe bands ARE the reference's own bootstrap CI half-width (see pipeline/v4a.py),
# so the only free constant is the capacity order-of-magnitude allowance.
V4A_TOLERANCE = {"capacity_orders_of_magnitude": 1.0}

# Strategy classes (for principle extraction grouping)
STRATEGY_CLASS_VOL = "macro-signal-volatility-forecast"


def env() -> dict:
    """Environment for shelling out to optional external helpers (knowledge store home set)."""
    e = dict(os.environ)
    e["PENROSE_BRAIN_HOME"] = str(BRAIN_HOME)
    return e
