"""Auto-implement a module from its ModuleSpec (close the cold-start loop).

The spec is implemented by the operator or an agent swarm. This is the
agent-swarm path, automated: take the ModuleSpec and have the LLM write a
working `impl.py` that conforms to the module contract, then VALIDATE it by
actually running it on the real data bundle before registering. If the generated
code crashes or returns the wrong shape, we reject it and the claim stays
`pending_module` (operator implements) — auto-implementation can never break the
pipeline or register broken code.

Module contract (what the generated impl.py must expose):
  __module_id__, __strategy_class__, __strategy_class_aliases__, __description__
  run(bundle, claim, cost_frac) -> dict, one of:
    {"ok": True, "net": pd.Series, "positions": pd.Series, "bars_per_year": float,
     "n_trades": int, optional: "payoff": pd.Series, "position_signed": pd.Series,
     "wf_frame": pd.DataFrame[signal,fut_rv,iv]}
    {"ok": False, "reason": "data_unavailable: <what is missing>"}   # honest blocker
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from pathlib import Path

from .. import config, llm
from ..brain import Claim
from ..trace import normalize_failure_reason

# Human-readable description of every series the data bundle currently provides, so
# the LLM maps spec inputs to what actually exists (and returns a data blocker otherwise).
_STATIC_BUNDLE_KEYS = {
    "btc_price": "daily BTC close price (USD), ~2023-01..2026-03",
    "btc_realized_vol_5d": "BTC 5-day realized volatility, annualized fraction",
    "btc_implied_vol": "BTC implied volatility (Deribit DVOL), annualized fraction",
    "kxfed_signal": "Kalshi KXFED |daily prob change| — monetary-policy macro signal",
    "kxrecssnber_signal": "Kalshi recession-risk |daily prob change| — macro signal",
    "btc_vol_regime": ("PRE-REGISTERED point-in-time volatility-regime LABEL (string series, "
                       "values 'low_vol'/'mid_vol'/'high_vol') — trailing-vol terciles, fixed "
                       "boundary, no look-ahead. Use to CONDITION a regime-dependent strategy."),
    "btc_trend_regime": ("PRE-REGISTERED point-in-time trend-regime LABEL (string series, values "
                         "'uptrend'/'downtrend') — vs a trailing moving average, fixed boundary, "
                         "no look-ahead. Use to CONDITION a regime-dependent strategy."),
}

# Real series provided by the optional local data catalog (config.DATA_DIR, set via
# PENROSE_DATA_DIR). Loaded from the same catalog the data client reads, so this list
# never drifts from what the bundle holds; absent the catalog, only the keyless live
# venues and the synthetic generator are offered.
_CATALOG_DESCRIPTIONS = {
    "btc_spot_daily": "daily BTC spot close (USD), Hyperliquid-derived",
    "eth_spot_daily": "daily ETH spot close (USD), Hyperliquid-derived",
    "sol_spot_daily": "daily SOL spot close (USD), Hyperliquid-derived",
    "link_spot_daily": "daily LINK spot close (USD), Hyperliquid-derived",
    "avax_spot_daily": "daily AVAX spot close (USD), Hyperliquid-derived",
    "ada_spot_daily": "daily ADA spot close (USD), Hyperliquid-derived",
    "funding_btc": "BTC perp funding rate (Binance), per-interval fraction",
    "funding_eth": "ETH perp funding rate (Binance), per-interval fraction",
    "funding_sol": "SOL perp funding rate (Binance), per-interval fraction",
    "weather_temp_ny": "NY daily actual temperature (NOAA), degrees",
    "weather_temp_chi": "Chicago daily actual temperature (NOAA), degrees",
    "weather_temp_mia": "Miami daily actual temperature (NOAA), degrees",
    "weather_temp_lax": "LAX daily actual temperature (NOAA), degrees",
    "btc_close_daily": "BTC hourly close resampled to daily (exchange OHLCV)",
}


def _normalize_failure_reason(reason: object) -> str:
    return normalize_failure_reason(reason)


def _failure_signature(spec: dict, reason: object) -> dict:
    """Stable signature for no-progress detection; fail-open at call sites."""
    strategy_class = str(spec.get("strategy_class") or "").strip().lower() or "unspecified"
    claim_type = str(spec.get("claim_type") or "").strip().lower() or "unspecified"
    category = _normalize_failure_reason(reason)
    payload = f"{strategy_class}\0{claim_type}\0{category}"
    return {
        "signature": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "category": category[:160],
        "strategy_class": strategy_class,
        "claim_type": claim_type,
    }


def _catalog_keys() -> dict:
    """Return only the catalog series whose referenced file actually resolves right now,
    so the LLM is never told about a series it cannot load."""
    import sys
    dd = str(config.DATA_DIR)
    try:
        if dd not in sys.path:
            sys.path.insert(0, dd)
        import loader as catalog  # <PENROSE_DATA_DIR>/loader.py
        return {k: _CATALOG_DESCRIPTIONS.get(k, f"catalog series {k}")
                for k in catalog.available()}
    except Exception:  # noqa: BLE001
        return {}


def _databento_keys() -> dict:
    """Advertise the user's configured Databento BYO series so auto-implemented modules can
    request them by their logical bundle key (only what the operator opted into)."""
    specs = getattr(config, "DATABENTO_SERIES", {}) or {}
    return {k: (f"Databento {v.get('symbol', '?')} {v.get('field', 'close')} "
                f"({v.get('schema', 'ohlcv-1d')}, point-in-time)")
            for k, v in specs.items() if isinstance(v, dict)}


BUNDLE_KEYS = {**_STATIC_BUNDLE_KEYS, **_catalog_keys(), **_databento_keys()}

IMPL_SYSTEM = (
    "You write ONE Python module for the penrose backtest engine. Output ONLY raw Python "
    "code — no markdown fences, no prose. The module MUST define module-level "
    "__module_id__, __strategy_class__ (a string), __strategy_class_aliases__ (list of "
    "strings including the strategy_class), __description__ (one sentence), and a function "
    "run(bundle, claim, cost_frac). "
    "Read series with: s = bundle.get(KEY); data = s.data if (s is not None and "
    "getattr(s,'available',False)) else None  (data is a pandas Series indexed by UTC "
    "date). You may ONLY use the bundle keys listed by the user. If a series the strategy "
    "needs is not in that list or is None, run MUST return "
    "{'ok': False, 'reason': 'data_unavailable: <what is missing>'} — never invent data, "
    "never crash. On success return {'ok': True, 'net': net, 'positions': positions, "
    "'bars_per_year': bpy, 'n_trades': len(net)} where net is a per-trade net-return "
    "pandas Series (fraction), positions is the per-trade absolute position-size Series "
    "(same index), and bpy is bars-per-year (e.g. 365/hold_days). Use non-overlapping "
    "trades to keep returns ~iid. Import numpy as np and pandas as pd. Keep it correct and "
    "self-contained; no network, no file IO, no other imports. "
    "REGIME-CONDITIONAL claims (e.g. 'works only in high volatility', 'in uptrends'): do NOT "
    "build your own regime detector or tune thresholds — that is look-ahead/overfitting. Instead "
    "CONDITION on the PRE-REGISTERED LABEL series btc_vol_regime ('low_vol'/'mid_vol'/'high_vol') "
    "or btc_trend_regime ('uptrend'/'downtrend'), read like any other bundle series. These are "
    "STRING series with a fixed point-in-time boundary. Gate each trade by the label KNOWN AT "
    "DECISION TIME (align the label to t-1, never t — no look-ahead): take the position only when "
    "the label matches the claimed regime, otherwise stay flat (drop that period from net). The "
    "engine's kill-lens partitions by these same labels and the regime degree-of-freedom is "
    "charged in the deflation trial count, so conditioning buys no free pass."
)

IMPL_USER_TMPL = """Implement this ModuleSpec as the module described above.

module_id: {module_id}
strategy_class: {strategy_class}
claim_type: {claim_type}
claim_translation: {claim_translation}
signal_logic: {signal_logic}
inputs requested by the spec: {inputs}
kill_criterion (for context; the engine applies it, you do not): {kill_criterion}
template guidance: {template_guidance}

The ONLY data series available in `bundle` (use these exact keys; anything else -> data_unavailable):
{keys}

Write run(bundle, claim, cost_frac). Charge cost_frac per unit of |position| as a
round-trip cost. Return the contract dict. Output only Python code.
"""


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "auto")).strip("_").lower()
    return s or "auto_module"


def _extract_code(text: str) -> str:
    """Pull runnable Python out of an LLM reply that may wrap it in markdown or prose.
    1) if there are ```fenced``` blocks, take the largest (the actual module);
    2) else drop any leading prose until the first code-like line (import/from/def/
       class/@/__assignment/#!). Defensive against the 'invalid syntax line 1' reject."""
    t = (text or "").strip()
    blocks = re.findall(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", t, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    if t.startswith("```"):                     # unterminated fence
        t = re.sub(r"^```[a-zA-Z0-9_+\-]*\n?", "", t).strip()
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("import ", "from ", "def ", "class ", "@", "#!", "__")):
            return "\n".join(lines[i:]).strip()
    return t


# back-compat alias (older callers/tests referenced _strip_fences)
_strip_fences = _extract_code


def _template_guidance(spec: dict) -> str:
    claim_type = str(spec.get("claim_type") or "trading_strategy")
    if claim_type == "provided_series_statistic":
        return (
            "PROVIDED_SERIES_STATISTIC (6g): pool the declared input series into ONE sample and "
            "compute EXACTLY the one-sample statistic/test the spec's statistic_logic describes, "
            "then apply the claim's own stated decision rule. Do NOT add a significance threshold "
            "(e.g. p<=0.05), a data-quality/minimum-observation kill, or an extra deflation/"
            "multiplicity method unless the claim itself states it. Do NOT build entry/exit rules, "
            "positions, or a trading backtest -- there is no signal to trade, only a statistic to "
            "test. Expose a minimal contract-valid result for the engine (as with "
            "descriptive_statistical)."
        )
    if claim_type == "predictive_regression":
        return (
            "PREDICTIVE_REGRESSION: this claim type is normally handled by Penrose's trusted "
            "deterministic executor, not by generated code. If implementing manually, test the "
            "declared predictor -> target relationship directly: align X_t with Y_t+h, fit only "
            "the in-sample sign and z-score moments, emit net=s*zscore_IS(X_t)*zscore_IS(Y_t+h) "
            "and positions=s*zscore_IS(X_t), and annualize overlapping h-ahead observations as "
            "observations_per_year / h. Do NOT add entry/exit rules, costs, capacity, or a "
            "trading overlay."
        )
    if claim_type == "factor_spanning":
        return (
            "FACTOR_SPANNING: this claim type is normally handled by Penrose's trusted "
            "deterministic executor, not by generated code. If implementing manually, test "
            "the declared candidate factor directly against the declared benchmark factors: "
            "fit multivariate OLS F_t = alpha + beta'B_t on the in-sample prefix only, freeze "
            "the beta vector, emit net=F_t - beta_IS'B_t and positions as the benchmark-hedged "
            "factor exposure, and compute bars_per_year from the emitted factor-return cadence. "
            "Do NOT add entry/exit rules, costs, capacity, unclaimed controls, or a trading overlay."
        )
    if claim_type == "cross_sectional_sort":
        return (
            "CROSS_SECTIONAL_SORT: this claim type is normally handled by Penrose's trusted "
            "deterministic executor, not by generated code. If implementing manually, load the "
            "declared returns and characteristic Panels from spec.panel_inputs, require the "
            "returns panel to be survivorship-corrected, call data.xsection.form_factor exactly "
            "with the declared n_buckets/rebalance/hold, synthesize positions from bucket "
            "membership, and compute bars_per_year from the rebalance cadence. Do NOT add "
            "entry/exit rules, timing overlays, proxy characteristics, or unclaimed universe filters."
        )
    if claim_type == "event_study":
        return (
            "EVENT_STUDY: this claim type is normally handled by Penrose's trusted deterministic "
            "executor, not by generated code. If implementing manually, load the declared return "
            "Series and event-calendar table, estimate the baseline for each event using only the "
            "pre-event estimation window ending strictly before the event date, compute abnormal "
            "returns over the declared event window, emit one CAR observation per event as net, "
            "positions=1, and bars_per_year=events_per_year. Do NOT add entry/exit rules, costs, "
            "capacity, unclaimed windows/baselines, or a trading overlay."
        )
    if claim_type == "forecast_skill":
        return (
            "FORECAST_SKILL: this claim type is normally handled by Penrose's trusted deterministic "
            "executor, not by generated code. If implementing manually, compare the declared "
            "model forecast F_t with the declared benchmark B_t on realized target Y_t, emit "
            "net=(B_t-Y_t)^2-(F_t-Y_t)^2 with positions=1 and bars_per_year from the emitted "
            "forecast cadence. Construct implied random_walk/historical_mean benchmarks "
            "strictly causally from Y through t-1 only. Do NOT add entry/exit rules, costs, "
            "capacity, target substitution, benchmark substitution, or a trading overlay."
        )
    if claim_type == "descriptive_statistical":
        return (
            "DESCRIPTIVE_STATISTICAL: compute the stated statistic directly over the requested "
            "sample (for example an unconditional mean, frequency, or correlation) and expose a "
            "minimal contract-valid result for the engine. Do not create entry thresholds, rolling "
            "signals, conditional positions, or a proxy trading strategy unless the claim explicitly "
            "requires trading."
        )
    if claim_type == "structural_proposition":
        return (
            "STRUCTURAL_PROPOSITION: if the spec cannot be operationalized against the available "
            "bundle keys, return {'ok': False, 'reason': 'data_unavailable: cannot_operationalize: ...'} "
            "rather than inventing a trading strategy."
        )
    return (
        "TRADING_STRATEGY: implement the stated signal -> position -> net-return test with "
        "non-overlapping trades and no look-ahead."
    )


# --- Static safety scan of LLM-generated code TEXT, run BEFORE exec ---------------
# A-010 denylist: dangerous operations that LLM code must never contain. This is a
# DENYLIST MITIGATION, not a true sandbox — a determined adversary can bypass it
# (e.g. getattr(__builtins__, ...), string-built imports). It is now ONLY cheap
# defense-in-depth: as of D-001/D-002, untrusted (auto-generated) module code is NEVER
# exec'd in penrose's process — it runs solely inside the Docker container (sandbox.py),
# and its contract is read statically via `ast_module_meta` (parse, never exec). The
# in-process `exec_module` path remains only for TRUSTED operator modules and unit tests
# (`prerun_result is None`). So the real sandbox boundary is the container, not this scan.
#
# B-002 FIX: imports are now gated by an AST-based ALLOWLIST (_ast_import_violation),
# NOT a regex denylist. A regex import-denylist is trivially bypassed
# (`import importlib; importlib.import_module("os")`, `import pathlib`, `shutil`,
# `ctypes`, `inspect`, `builtins`, `multiprocessing`, `runpy`, ...), so we parse the
# code and reject ANY import whose top-level module name is not in _IMPORT_ALLOWLIST.
# The text patterns below remain as a belt-and-suspenders scan for non-import dangerous
# calls (open(/eval(/exec(/__import__/compile/pickle) that AST import-checking won't see.
# `__future__` is compile-time only (no runtime code, no attack surface) and LLMs emit
# `from __future__ import annotations` reflexively — allowing it stops a needless self-repair
# attempt from being burned on a harmless line (C-009).
_IMPORT_ALLOWLIST = {"numpy", "np", "pandas", "pd", "math", "warnings", "__future__"}

# Belt-and-suspenders TEXT scan for dangerous non-import operations. Import safety is
# enforced by AST (above); these catch dynamic-exec / IO calls expressed as plain calls.
_DENY_PATTERNS = [
    (r"\bopen\s*\(", "open("),
    (r"\beval\s*\(", "eval("),
    (r"\bexec\s*\(", "exec("),
    (r"\b__import__\b", "__import__"),
    (r"\bcompile\s*\(", "compile("),
    # `\bpickle\b` missed read_pickle/to_pickle ('_' is a word char, so no boundary before
    # 'pickle'); pd.read_pickle is a one-line RCE on unpickle (D-001). Match any 'pickle'.
    (r"pickle", "pickle (read_pickle/to_pickle/loads -> RCE)"),
]


def _ast_import_violation(code: str) -> str | None:
    """B-002: AST-based import ALLOWLIST. Parse `code`, walk every node, and reject if
    there is ANY `import x` / `from x import ...` whose TOP-LEVEL module name is not in
    _IMPORT_ALLOWLIST. Returns a rejection reason string on the first violation, else
    None. If the code cannot be parsed, reject (conservative — we never exec unparseable
    code). This closes the regex bypass: `import importlib; importlib.import_module("os")`
    is rejected because `importlib` is not on the allowlist."""
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError) as e:
        return f"could not AST-parse generated code (rejecting) (B-002): {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split(".")[0]
                if top not in _IMPORT_ALLOWLIST:
                    return f"disallowed import '{alias.name}' (allowlist only) (B-002)"
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a relative import (from . import x) — never allowed here.
            if node.level and node.level > 0:
                return "disallowed relative import (allowlist only) (B-002)"
            top = (node.module or "").split(".")[0]
            if top not in _IMPORT_ALLOWLIST:
                return f"disallowed import 'from {node.module} import ...' (allowlist only) (B-002)"
    return None


def ast_module_meta(file_text: str) -> dict:
    """Extract a module's contract metadata WITHOUT executing it (D-001/D-002).

    Untrusted (auto-generated) module code must NEVER be import-exec'd in penrose's process —
    `exec_module` runs the file's TOP-LEVEL code with full host privileges, which the Docker
    sandbox (that only confines `run()`) does not contain. So registration and the sandboxed
    validation path read the contract statically from the AST: the presence of `def run`, and the
    top-level literal assignments `__strategy_class__`, `__module_id__`, `__strategy_class_aliases__`,
    `__auto_generated__`. Last assignment wins (penrose appends `__auto_generated__ = True` last)."""
    meta = {"has_run": False, "strategy_class": None, "module_id": None,
            "aliases": [], "auto_generated": False}
    try:
        tree = ast.parse(file_text or "")
    except (SyntaxError, ValueError):
        return meta
    for node in tree.body:                       # top-level only — never descend into code
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            meta["has_run"] = True
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                v = node.value
                if tgt.id == "__strategy_class__" and isinstance(v, ast.Constant):
                    meta["strategy_class"] = v.value
                elif tgt.id == "__module_id__" and isinstance(v, ast.Constant):
                    meta["module_id"] = v.value
                elif tgt.id == "__auto_generated__":
                    meta["auto_generated"] = bool(getattr(v, "value", False))
                elif tgt.id == "__strategy_class_aliases__" and isinstance(v, (ast.List, ast.Tuple)):
                    meta["aliases"] = [e.value for e in v.elts if isinstance(e, ast.Constant)]
    return meta

# A-012 look-ahead heuristics: future-peeking patterns. This is a HEURISTIC, not a
# proof of no look-ahead — it catches the obvious cases (negative shift, future
# indexing) but cannot detect look-ahead expressed in other ways.
_LOOKAHEAD_PATTERNS = [
    # .shift(-1), .shift(- 1), .shift(periods=-1) — pulling future rows back
    (r"\.shift\s*\(\s*(?:periods\s*=\s*)?-\s*\d", "negative .shift(-N) (future leak)"),
    (r"\.shift\s*\([^)]*periods\s*=\s*-\s*\d", "negative .shift(periods=-N) (future leak)"),
    # .iloc[i+1] / .iloc[i + 1] style forward indexing into the future
    (r"\.iloc\s*\[[^\]]*\+\s*\d+[^\]]*\]", "forward .iloc[...+N] indexing (possible future leak)"),
    (r"np\.roll\s*\([^)]*,\s*-\s*\d", "np.roll(..., -N) future rotation"),
]


def _scan_code_text(code: str) -> str | None:
    """Static pre-exec scan of LLM code TEXT. Returns a rejection reason string if the
    code trips a denylist (A-010) or look-ahead (A-012) pattern, else None. Heuristic /
    denylist only — see notes on _DENY_PATTERNS / _LOOKAHEAD_PATTERNS.

    B-002: import safety is enforced FIRST via the AST allowlist (rejects any import not
    in {numpy/np, pandas/pd, math}, and rejects unparseable code), then the text
    denylist catches dangerous non-import calls (open/eval/exec/...)."""
    text = code or ""
    ast_err = _ast_import_violation(text)        # B-002 AST import allowlist (real fix)
    if ast_err is not None:
        return ast_err
    for pat, label in _DENY_PATTERNS:
        if re.search(pat, text):
            return f"denylisted operation in generated code (A-010): {label}"
    for pat, label in _LOOKAHEAD_PATTERNS:
        if re.search(pat, text):
            return f"look-ahead pattern in generated code (A-012): {label}"
    return None


def _generate_code(spec: dict, available: dict,
                   feedback: tuple[str, str] | None = None) -> str:
    keys = "\n".join(f"  - {k}: {v}" for k, v in available.items())
    user = IMPL_USER_TMPL.format(
        module_id=spec.get("module_id", "auto_module"),
        strategy_class=spec.get("strategy_class", "unspecified"),
        claim_type=spec.get("claim_type", "trading_strategy"),
        claim_translation=str(spec.get("claim_translation", ""))[:600],
        signal_logic=str(spec.get("signal_logic", ""))[:800],
        inputs=", ".join(spec.get("inputs", []) or []) or "(unspecified)",
        kill_criterion=str(spec.get("kill_criterion", ""))[:200],
        template_guidance=_template_guidance(spec),
        keys=keys,
    )
    if feedback:                                  # self-repair: prior code + the error it hit
        prev_code, err = feedback
        if str(err).startswith("FIDELITY:"):
            user += (
                "\n\n--- YOUR PREVIOUS ATTEMPT VALIDATED BUT WAS UNFAITHFUL ---\n"
                f"{str(err)[len('FIDELITY:'):].strip()}\n\n"
                f"the code you wrote:\n{prev_code[:3500]}\n\n"
                "Regenerate the implementation to fix exactly these fidelity divergences, "
                "without breaking validation. Output ONLY the corrected full Python module."
            )
        else:
            user += (
                "\n\n--- YOUR PREVIOUS ATTEMPT FAILED VALIDATION ---\n"
                f"error: {err}\n\nthe code you wrote:\n{prev_code[:3500]}\n\n"
                "Fix the specific failure above. Common fixes: only read bundle keys from the "
                "allowed list (anything else -> return data_unavailable); guard None/empty "
                "series; use NON-OVERLAPPING trades so len(net) >= 10; return the exact contract "
                "dict. Output ONLY the corrected full Python module."
            )
    last_err = None
    for attempt in range(3):                     # retry transient LLM errors (timeouts)
        try:
            resp = llm.call("module_implementer",
                            [{"role": "system", "content": IMPL_SYSTEM},
                             {"role": "user", "content": user}],
                            temperature=0.1, timeout=300)   # GLM-5.2 reasons; long regime specs
                                                            # exceed the 90s default (read timeout)
            code = _extract_code(resp.text)
            if "def run" in code and "__strategy_class__" in code:
                return code
            last_err = "reply missing run()/__strategy_class__"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    raise RuntimeError(f"code generation failed after 3 attempts: {last_err}")


# --- B-003: runtime look-ahead detector ------------------------------------------
def _truncate_bundle(bundle, frac: float = 0.70):
    """Shallow-copy `bundle` and replace each available Series' data with its leading
    `frac` of the date range (head), preserving the Series wrapper/contract so the module
    reads it exactly the same way (same .get / .data / .available semantics). Unavailable
    entries are passed through unchanged. Returns the truncated bundle, or None if it can't
    be built (caller then SKIPS the check rather than falsely rejecting)."""
    try:
        import copy

        series_map = getattr(bundle, "series", None)
        if not isinstance(series_map, dict):
            return None
        new_series = {}
        truncated_any = False
        for name, s in series_map.items():
            data = getattr(s, "data", None)
            if getattr(s, "available", False) and data is not None and len(data) >= 4:
                cut = max(2, int(len(data) * frac))
                if cut < len(data):
                    truncated_any = True
                # copy the Series wrapper so we don't mutate the live bundle's object
                s2 = copy.copy(s)
                try:
                    s2.data = data.iloc[:cut]
                except Exception:  # noqa: BLE001
                    s2.data = data[:cut]
                new_series[name] = s2
            else:
                new_series[name] = s            # Unavailable / too-short: pass through
        if not truncated_any:
            return None                          # nothing to cut -> no meaningful 2nd run
        nb = copy.copy(bundle)
        nb.series = new_series
        # drop any cached normalized-key index so get() rebuilds against the new dict
        for attr in ("_norm_index", "_norm_index_len"):
            if hasattr(nb, attr):
                try:
                    object.__setattr__(nb, attr, None)
                except Exception:  # noqa: BLE001
                    pass
        return nb
    except Exception:  # noqa: BLE001
        return None


def _runtime_lookahead_check(module, bundle, claim, cost_frac, net):
    """B-003: detect look-ahead by re-running on a truncated bundle and comparing net on the
    overlapping early dates. Returns a rejection reason string if look-ahead is detected,
    else None. Conservative: returns None (SKIP) whenever bundle is None or the truncated
    run errors / returns a non-ok / data_unavailable result — we never falsely reject."""
    if bundle is None:
        return None
    try:
        import pandas as pd

        tb = _truncate_bundle(bundle)
        if tb is None:
            return None
        res2 = module.run(tb, claim, cost_frac)
        if not isinstance(res2, dict) or not res2.get("ok"):
            return None                          # data_unavailable / blocker on short data -> skip
        net2 = res2.get("net")
        if not isinstance(net2, pd.Series) or not isinstance(net2.index, pd.DatetimeIndex):
            return None
        return _lookahead_diff_reason(net, net2)
    except Exception:  # noqa: BLE001
        return None                              # conservative: never falsely reject on harness error


def _lookahead_diff_reason(net, net2):
    """Shared overlap comparison for in-process and sandbox truncated-bundle reruns."""
    try:
        import numpy as np
        import pandas as pd

        if not isinstance(net, pd.Series) or not isinstance(net.index, pd.DatetimeIndex):
            return None
        if not isinstance(net2, pd.Series) or not isinstance(net2.index, pd.DatetimeIndex):
            return None
        if len(net2) == 0:
            return None
        # Compare the two runs on the dates they SHARE (the early, overlapping dates). If the
        # module is causal, those values are identical regardless of the later data we removed.
        common = net.index.intersection(net2.index)
        if len(common) < 3:
            return None
        a = net.reindex(common).to_numpy(dtype="float64", na_value=np.nan)
        b = net2.reindex(common).to_numpy(dtype="float64", na_value=np.nan)
        # Conservative skip: a leak that creates a boundary NaN can evade this dynamic check.
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return None
        diff = ~np.isclose(a, b, rtol=1e-6, atol=1e-9)
        if diff.any():
            n_diff = int(diff.sum())
            return (f"run() ok but net on {n_diff}/{len(common)} overlapping early dates changed "
                    "when later data was removed - module uses future data (look-ahead) (B-003)")
        return None
    except Exception:  # noqa: BLE001
        return None


def _lookahead_skip(meta, reason: str):
    if isinstance(meta, dict):
        meta["dynamic_lookahead_check"] = "skipped"
        meta["dynamic_lookahead_skip_reason"] = reason
    try:
        import sys

        print(f"penrose: warning: dynamic sandbox look-ahead check did not run: {reason}",
              file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
    return None


def _sandbox_lookahead_check(impl_path, bundle, claim, cost_frac, net, validation_meta=None):
    """Run the truncated-bundle look-ahead check through the sandbox production path."""
    if bundle is None:
        return _lookahead_skip(validation_meta, "missing data bundle")
    try:
        import pandas as pd
        from . import sandbox

        tb = _truncate_bundle(bundle)
        if tb is None:
            return _lookahead_skip(validation_meta, "bundle could not be truncated")
        res2 = sandbox.run_in_container(str(impl_path), tb, claim, cost_frac)
        if not isinstance(res2, dict) or not res2.get("ok"):
            reason = res2.get("reason") if isinstance(res2, dict) else type(res2).__name__
            return _lookahead_skip(validation_meta, f"truncated sandbox run failed: {reason}")
        net2 = res2.get("net")
        if not isinstance(net2, pd.Series) or not isinstance(net2.index, pd.DatetimeIndex):
            return _lookahead_skip(validation_meta, "truncated sandbox run returned non-Series net")
        if isinstance(validation_meta, dict):
            validation_meta["dynamic_lookahead_check"] = "ran"
        return _lookahead_diff_reason(net, net2)
    except Exception as e:  # noqa: BLE001
        return _lookahead_skip(validation_meta, f"sandbox error: {type(e).__name__}: {e}")


def _validate_module(impl_path, mid: str, bundle, claim, cost_frac: float, prerun_result=None,
                     validation_meta=None):
    """Validate a generated module against the contract. Returns (True, module) or (False, error).

    If `prerun_result` is provided, the module was already RUN in the sandbox (untrusted code
    never execs in this process) and we validate that result; we still import the module to get
    the object + check module-level contract (benign: module-level is just assignments + def run,
    and run() is never called in-process). If prerun_result is None (unit tests), it runs in-process."""
    try:
        # A-010 / A-012: re-run the static safety scan on the FILE TEXT before exec, so a
        # bad module is rejected even if _validate_module is reached via a path that skipped
        # the try_implement pre-write scan. Denylist/heuristic only (see _scan_code_text).
        try:
            file_text = Path(impl_path).read_text()
        except Exception as e:  # noqa: BLE001
            return False, f"could not read module for safety scan: {e}"
        scan_err = _scan_code_text(file_text)
        if scan_err is not None:
            return False, scan_err
        if prerun_result is not None:
            # SANDBOXED (untrusted) path: the container already exec'd the file and ran run().
            # We must NOT import-exec it here — that would run its TOP-LEVEL code in penrose's
            # process, outside the sandbox (D-001). Inspect the contract STATICALLY instead.
            meta = ast_module_meta(file_text)
            if not (meta["has_run"] and meta["strategy_class"]):
                return False, "missing run() or __strategy_class__ (static check; no in-process exec)"
            declared_mid = meta["module_id"]
            res = prerun_result
            # D-004: the terminal `return True, module` needs a handle, but we must NOT import the
            # file. Build a no-exec stand-in carrying only the attributes the caller reads
            # (__file__/__auto_generated__/__module_id__/__strategy_class__); the auto path runs
            # the module in the SANDBOX via __file__, never via this object's (absent) run().
            import types
            module = types.SimpleNamespace(
                __file__=str(impl_path),
                __auto_generated__=True,
                __module_id__=meta["module_id"],
                __strategy_class__=meta["strategy_class"],
                __strategy_class_aliases__=meta.get("aliases", []),
            )
        else:
            # TRUSTED path (unit tests / operator, no sandbox result): in-process import is fine.
            il = importlib.util.spec_from_file_location(f"modules.{mid}.impl", impl_path)
            module = importlib.util.module_from_spec(il)
            il.loader.exec_module(module)
            if not (hasattr(module, "run") and getattr(module, "__strategy_class__", None)):
                return False, "missing run() or __strategy_class__ after import"
            declared_mid = getattr(module, "__module_id__", None)
            res = module.run(bundle, claim, cost_frac)
        # B-011: the module's __module_id__ MUST equal the on-disk slug `mid`, otherwise the
        # fidelity path that reads modules/<__module_id__>/impl.py looks in the wrong dir and
        # silently misses the file. Reject on mismatch so the claim stays pending_module.
        if declared_mid is not None and declared_mid != mid:
            return False, (f"module __module_id__={declared_mid!r} != on-disk slug "
                           f"{mid!r} — fidelity path would miss modules/{declared_mid}/impl.py (B-011)")
        if not isinstance(res, dict):
            return False, f"run() returned {type(res).__name__}, expected dict"
        if res.get("ok"):
            import math
            import numpy as np
            import pandas as pd

            net = res.get("net")
            positions = res.get("positions")

            # --- A-014: contract enforcement on the ok-branch -----------------------
            # net must be a pandas Series with a DatetimeIndex, all-finite, len >= 10.
            if not isinstance(net, pd.Series):
                return False, f"run() ok but net is {type(net).__name__}, expected pd.Series (A-014)"
            if len(net) < 10:
                return False, "run() ok but net series too short (<10 trades) (A-014)"
            if not isinstance(net.index, pd.DatetimeIndex):
                return False, "run() ok but net index is not a DatetimeIndex (A-014)"
            net_vals = net.to_numpy(dtype="float64", na_value=np.nan)
            if not np.isfinite(net_vals).all():
                return False, "run() ok but net contains NaN/inf (A-014)"
            # positions must be a Series of the SAME length as net.
            if not isinstance(positions, pd.Series):
                return False, (f"run() ok but positions is {type(positions).__name__}, "
                               "expected pd.Series (A-014)")
            if len(positions) != len(net):
                return False, (f"run() ok but positions length {len(positions)} != net length "
                               f"{len(net)} (A-014)")
            pos_vals = positions.to_numpy(dtype="float64", na_value=np.nan)
            if not np.isfinite(pos_vals).all():
                return False, "run() ok but positions contains NaN/inf (A-014)"

            # --- B-009: index alignment + optional-series shape -------------------
            # len(positions)==len(net) is not enough: a RangeIndex on positions would be
            # silently reindexed-to-zeros downstream, so require IDENTICAL indexes.
            if not positions.index.equals(net.index):
                return False, ("run() ok but positions.index != net.index — misaligned "
                               "(would silently reindex-to-zeros downstream) (B-009)")
            # Optional result series, if present, must be finite Series aligned to net.index;
            # wf_frame, if present, must be a DataFrame. Reject on any malformed carrier.
            for _opt in ("payoff", "position_signed"):
                if _opt in res and res.get(_opt) is not None:
                    _s = res.get(_opt)
                    if not isinstance(_s, pd.Series):
                        return False, f"run() ok but '{_opt}' is {type(_s).__name__}, expected pd.Series (B-009)"
                    if not _s.index.equals(net.index):
                        return False, f"run() ok but '{_opt}'.index != net.index — misaligned (B-009)"
                    if not np.isfinite(_s.to_numpy(dtype='float64', na_value=np.nan)).all():
                        return False, f"run() ok but '{_opt}' contains NaN/inf (B-009)"
            if "wf_frame" in res and res.get("wf_frame") is not None:
                _wf = res.get("wf_frame")
                if not isinstance(_wf, pd.DataFrame):
                    return False, ("run() ok but 'wf_frame' is "
                                   f"{type(_wf).__name__}, expected pd.DataFrame (B-009)")
                # C-010: walk_forward_vol re-fits signal->fut_rv vs iv, so the carrier must
                # actually carry those columns with finite, sane values — a shape-only check
                # let a malformed frame reach the kill gate and silently no-op or mis-fit.
                _need = {"signal", "fut_rv", "iv"}
                _missing = _need - set(_wf.columns)
                if _missing:
                    return False, f"run() ok but 'wf_frame' missing columns {sorted(_missing)} (C-010)"
                if len(_wf) < 1:
                    return False, "run() ok but 'wf_frame' is empty (C-010)"
                for _col in _need:
                    if not np.isfinite(_wf[_col].to_numpy(dtype='float64', na_value=np.nan)).all():
                        return False, f"run() ok but 'wf_frame[{_col}]' contains NaN/inf (C-010)"

            # --- A-011: degenerate / constant cheat -------------------------------
            # A near-constant net with non-zero mean fakes an enormous Sharpe (mean/~0 std).
            mean = float(np.mean(net_vals))
            std = float(np.std(net_vals, ddof=1))
            if std < 1e-9 and abs(mean) > 1e-12:
                return False, ("run() ok but net is near-constant with non-zero mean "
                               "(fake infinite Sharpe) (A-011)")

            # --- A-013: bars_per_year inflation knob ------------------------------
            # bpy scales the annualized Sharpe by sqrt(bpy); an absurd value (e.g. 100000)
            # is an inflation knob. Sane range: 1 trade/year .. minutely (366*24*60).
            bpy = res.get("bars_per_year")
            if not isinstance(bpy, (int, float)) or isinstance(bpy, bool) or not math.isfinite(bpy):
                return False, f"run() ok but bars_per_year is not a finite number: {bpy!r} (A-013)"
            if not (1 <= bpy <= 366 * 24 * 60):
                return False, f"run() ok but bars_per_year={bpy} outside sane range [1, 526560] (A-013)"
            # Stronger: bpy must be consistent with the net's OWN frequency. The index spans
            # a real calendar duration, so trades/year is implied by it; a value wildly above
            # that is a Sharpe-inflation knob (e.g. a daily series claiming per-minute bpy).
            # 50x margin so genuine frequency mismatches (entry-date vs calendar) never trip.
            # B-008: gate lowered — run the frequency-consistency check whenever there are
            # enough trades over a non-trivial span (>=10 trades AND span_days>=5), not just
            # span_days>30, so short high-frequency series can't dodge the bpy-inflation gate.
            span_days = (net.index[-1] - net.index[0]).days
            if len(net) >= 10 and span_days >= 5:
                implied = len(net) / (span_days / 365.25)
                if bpy > 50 * implied:
                    return False, (f"run() ok but bars_per_year={bpy:.0f} is {bpy/implied:.0f}x the "
                                   f"frequency implied by the net index (~{implied:.0f}/yr) — "
                                   "Sharpe inflation (A-013)")

            # --- B-010: ANNUALIZED Sharpe cap (replaces the per-trade |mean|/std>5 cap) ---
            # The old per-trade cap of 5 let annualized Sharpe = 5*sqrt(bpy) ~= 95 pass.
            # Cap the ANNUALIZED Sharpe directly: |mean|/std * sqrt(bpy). 20 is already
            # implausibly high for any real strategy, so above that is almost certainly a
            # degenerate/cheat module, not alpha.
            if std > 0:
                ann_sharpe = abs(mean) / std * math.sqrt(bpy)
                if ann_sharpe > 20.0:
                    return False, (f"run() ok but annualized Sharpe |mean|/std*sqrt(bpy)="
                                   f"{ann_sharpe:.1f} > 20 (implausibly high, likely degenerate) (B-010)")

            # --- B-003: RUNTIME look-ahead detector --------------------------------
            # Static regex can't catch the natural look-ahead idioms (s.iloc[1:], s[:-1],
            # np.roll(s,-1), ...). Real fix: re-run the module on a TRUNCATED bundle (each
            # series cut to its first ~70%) and verify net on the OVERLAPPING early dates is
            # identical. If the early-date net changed when later data was removed, the module
            # consumed future data -> reject. Conservative: skip (don't falsely reject) when
            # there's no real bundle or the truncated run errors / returns a data blocker.
            if prerun_result is None:
                la_err = _runtime_lookahead_check(module, bundle, claim, cost_frac, net)
            else:
                la_err = _sandbox_lookahead_check(
                    impl_path, bundle, claim, cost_frac, net, validation_meta)
            if la_err is not None:
                return False, la_err
        elif "data_unavailable" not in str(res.get("reason", "")).lower():
            return False, f"run() not ok and not a clean data blocker: {res.get('reason')}"
        return True, module
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def try_implement(spec: dict, claim: Claim, bundle, cost_frac: float,
                  *, use_llm: bool = True, max_attempts: int = 3) -> dict:
    """Generate -> write -> VALIDATE (run on the real bundle) -> register.

    Returns {"ok": True, "module": <module obj>, "module_id": str} if the generated
    module imports, runs without crashing, and returns a contract-valid result (either a
    real net series or a clean data_unavailable). Otherwise {"ok": False, "reason": ...}
    and nothing is registered (claim stays pending_module).
    """
    if not use_llm:
        return {"ok": False, "reason": "auto-impl disabled (no LLM)"}
    # Auto modules are claim-scoped. An LLM-supplied module_id is descriptive, not a unique storage
    # key; binding the claim id prevents two parallel claims from overwriting the same impl.py.
    mid = _slug(f"{spec.get('module_id') or 'auto'}-{claim.claim_id}")
    spec = dict(spec)
    spec["module_id"] = mid
    mod_dir = config.AUTO_MODULES / mid          # machine-generated provenance shelf (UNTRUSTED)
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "__init__.py").write_text("")
    impl_path = mod_dir / "impl.py"

    # SELF-REPAIR LOOP: generate -> validate on the live bundle -> if it fails, feed the
    # error + the code back and regenerate. Keep going until it works or attempts run out.
    # The point (operator's ask): if a paper needs a module, BUILD it and TEST it — don't
    # bail to pending_module on the first rough draft.
    feedback = None
    last_err = None
    last_validated = None
    fidelity_attempts = []
    last_failure_sig = None
    repeated_failure_count = 0
    no_progress_audit = []
    try:
        no_progress_limit = int(getattr(config, "IMPL_NO_PROGRESS_LIMIT", 2) or 0)
    except Exception:  # noqa: BLE001
        no_progress_limit = 2
    # LD-3: the guard needs >=2 attempts to have a "repeat" (the counter starts at 1 on the first
    # rejection). <=0 disables the guard; 1 is a degenerate footgun (would fire on the FIRST failure,
    # disabling the self-repair loop entirely) so it is clamped up to the minimum meaningful value 2.
    no_progress_limit = 0 if no_progress_limit <= 0 else max(2, no_progress_limit)

    def _write_rejected_stub(reason, attempts_tried: int) -> None:
        try:
            _safe_err = repr(str(reason))[:300].replace("\n", " ")
            impl_path.write_text(
                "__auto_generated__ = True  # REJECTED stub — never exec'd in-process (D-002/D-005)\n"
                f"# auto-impl REJECTED after {attempts_tried} attempts: {_safe_err}\n"
                "# spec kept; operator may implement.\n")
        except Exception:  # noqa: BLE001
            pass

    def _track_rejected_attempt(reason, attempt: int):
        nonlocal last_failure_sig, repeated_failure_count
        try:
            sig = _failure_signature(spec, reason)
            if sig["signature"] == last_failure_sig:
                repeated_failure_count += 1
            else:
                last_failure_sig = sig["signature"]
                repeated_failure_count = 1
            row = {
                "attempt": attempt,
                "signature": sig["signature"],
                "category": sig["category"],
                "consecutive": repeated_failure_count,
            }
            no_progress_audit.append(row)
            if no_progress_limit and repeated_failure_count >= no_progress_limit:
                review_reason = (
                    "auto-impl made no progress: "
                    f"{repeated_failure_count} attempts, same failure signature ({sig['category']})"
                )
                audit = {
                    "attempts_tried": attempt,
                    "limit": no_progress_limit,
                    "consecutive_attempts": repeated_failure_count,
                    "signature": sig["signature"],
                    "category": sig["category"],
                    "attempts": list(no_progress_audit),
                }
                _write_rejected_stub(review_reason, attempt)
                return {
                    "ok": False,
                    "needs_review": True,
                    "reason": review_reason,
                    "attempts": attempt,
                    "no_progress": audit,
                    **({"fidelity_attempts": fidelity_attempts} if fidelity_attempts else {}),
                }
        except Exception:  # noqa: BLE001
            return None
        return None

    for attempt in range(1, max_attempts + 1):
        try:
            code = _generate_code(spec, BUNDLE_KEYS, feedback=feedback)
        except Exception as e:  # noqa: BLE001
            last_err = f"generation failed: {e}"
            short = _track_rejected_attempt(last_err, attempt)
            if short is not None:
                return short
            continue
        if "def run" not in code or "__strategy_class__" not in code:
            last_err = "generated code missing contract (run/__strategy_class__)"
            feedback = (code, last_err)
            short = _track_rejected_attempt(last_err, attempt)
            if short is not None:
                return short
            continue
        # A-010 / A-012: static safety scan of the code TEXT BEFORE we ever write or
        # exec it. Denylisted ops or look-ahead patterns -> reject (feed back for repair),
        # never exec. Defense-in-depth: the scan re-runs on the file inside _validate_module.
        scan_err = _scan_code_text(code)
        if scan_err is not None:
            last_err = scan_err
            feedback = (code, scan_err)
            short = _track_rejected_attempt(scan_err, attempt)
            if short is not None:
                return short
            continue
        # Mark auto-generated. This module is UNTRUSTED: it lives on the _auto provenance shelf,
        # is NEVER registered for cross-claim routing (_register_known_modules skips it), and is
        # used only in-run for THIS claim via the validated object returned below. The flag also
        # excludes it from the P2 controlled-vocab prompt (auto modules proliferate; left in, they
        # exhaust glm) and guarantees _register_known_modules never import-execs it in-process.
        impl_path.write_text(code.rstrip() + "\n\n__auto_generated__ = True\n")
        # SECURITY: run the candidate in the Docker sandbox, NOT in this process. Validate the
        # sandbox result against the contract. (run.py only calls try_implement when Docker is up.)
        from . import sandbox
        sbx = sandbox.run_in_container(str(impl_path), bundle, claim, cost_frac)
        if not isinstance(sbx, dict) or "sandbox" in str(sbx.get("reason", "")).lower():
            last_err = f"sandbox unavailable/failed: {sbx.get('reason') if isinstance(sbx, dict) else sbx}"
            feedback = (code, last_err)
            short = _track_rejected_attempt(last_err, attempt)
            if short is not None:
                return short
            continue
        validation_meta = {}
        ok, result = _validate_module(
            impl_path, mid, bundle, claim, cost_frac, prerun_result=sbx,
            validation_meta=validation_meta)
        if ok:
            last_validated = (result, attempt, validation_meta)
            if use_llm and getattr(config, "FIDELITY_CHECK", False):
                try:
                    from . import fidelity

                    module_code = impl_path.read_text()
                    fid = fidelity.assess(claim, module_code, spec=spec)
                    if not isinstance(fid, dict):
                        fid = {"faithful": False, "confidence": 0.0, "divergences": [],
                               "note": "fidelity assessor returned non-dict"}
                except Exception as e:  # noqa: BLE001
                    fid = {"faithful": False, "confidence": 0.0, "divergences": [],
                           "note": f"fidelity check errored: {e}"}

                divergences = fid.get("divergences") or []
                if not divergences and fid.get("note"):
                    divergences = [str(fid.get("note"))]
                fid_attempt = {
                    "attempt": attempt,
                    "faithful": bool(fid.get("faithful", False)),
                    "confidence": float(fid.get("confidence", 0.0) or 0.0),
                    "divergences": [str(d) for d in divergences],
                }
                fidelity_attempts.append(fid_attempt)
                confidently_unfaithful = (
                    not fid_attempt["faithful"]
                    and fid_attempt["confidence"] >= config.FIDELITY_KILL_CONFIDENCE
                )
                if confidently_unfaithful and attempt < max_attempts:
                    fidelity_reason = (
                        "fidelity: "
                        f"{'; '.join(fid_attempt['divergences']) or str(fid.get('note') or 'unspecified divergence')}"
                    )
                    short = _track_rejected_attempt(fidelity_reason, attempt)
                    if short is not None:
                        return short
                    guidance = (
                        "The previous module was VALIDATED but judged UNFAITHFUL to the claim "
                        "for these reasons: "
                        f"{'; '.join(fid_attempt['divergences']) or 'unspecified divergence'}. "
                        "Regenerate the implementation to fix exactly these, without breaking "
                        "validation."
                    )
                    last_err = guidance
                    feedback = (module_code, f"FIDELITY: {guidance}")
                    continue
                if confidently_unfaithful:
                    fidelity_reason = (
                        "fidelity: "
                        f"{'; '.join(fid_attempt['divergences']) or str(fid.get('note') or 'unspecified divergence')}"
                    )
                    short = _track_rejected_attempt(fidelity_reason, attempt)
                    if short is not None:
                        return short

            return {"ok": True, "module": result, "module_id": mid, "attempts": attempt,
                    "validation": validation_meta,
                    **({"fidelity_attempts": fidelity_attempts} if fidelity_attempts else {}),
                    "note": f"auto-implemented + validated on the live bundle (attempt {attempt})"}
        last_err = result
        short = _track_rejected_attempt(result, attempt)
        if short is not None:
            return short
        feedback = (code, result)              # repair on the next pass

    # If fidelity self-correction spent the remaining attempts without finding a faithful
    # replacement, return the last validated module and let the downstream gate render the
    # authoritative unfaithful_module outcome with full metrics.
    if last_validated is not None:
        result, attempt, validation_meta = last_validated
        return {"ok": True, "module": result, "module_id": mid, "attempts": attempt,
                "validation": validation_meta,
                **({"fidelity_attempts": fidelity_attempts} if fidelity_attempts else {}),
                "note": f"auto-implemented + validated on the live bundle (attempt {attempt})"}

    # all attempts failed validation -> leave a REJECTED stub so nothing broken is ever registered.
    # D-005: last_err carries LLM-controlled substrings (a crafted __module_id__ / run() reason
    # with an embedded newline) — interpolating it RAW could break out of the `#` comment into
    # executable code. Two defenses: (1) the stub is marked __auto_generated__=True, so
    # _register_known_modules skips it via the static check and NEVER imports it; (2) last_err is
    # collapsed to a single safe line (repr drops real newlines) before it touches the file.
    _write_rejected_stub(last_err, max_attempts)
    return {"ok": False, "reason": f"validation failed after {max_attempts} attempts: {last_err}",
            **({"no_progress": {"attempts": no_progress_audit}} if no_progress_audit else {})}
