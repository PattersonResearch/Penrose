# AGENTS.md

Guidance for AI coding agents working in this repository. Humans:
see [README.md](README.md) for the project overview and [CONTRIBUTING.md](CONTRIBUTING.md) for the
contribution rules. This file is the short, operational version for agents.

## What this project is

Penrose is a **falsification referee for quantitative trading claims**. It ingests a claim (a paper, a
thesis, a code-complete strategy, or a machine-generated hypothesis), reconstructs it in a sandbox,
runs it through a robustness and power stack, validates its own detector, and returns a calibrated
verdict. It is **not** a backtester and **not** an alpha generator, and it makes **no claim that any
strategy is profitable**. DSR deflation scales with the search Penrose has actually seen, floored by
a conservative effective-trials prior for external claims (config.DEFLATION_PRIOR). See
[docs/GATES.md](docs/GATES.md) for every gate in plain language.

## Operating Penrose (running a claim)

To *run* Penrose on a claim (rather than work on its code), the loop is: shape a falsifiable claim,
submit it, read the verdict, stop at the human gate. Full walkthrough: [docs/OPERATING.md](docs/OPERATING.md).

The minimum:

1. **Install** with `pip install -e .` (the core runs keyless; ingesting a paper needs a model). The
   model seam is one OpenAI-compatible endpoint configured by env: `PENROSE_LLM_API_KEY`,
   `PENROSE_LLM_BASE_URL` (any compatible provider), and `PENROSE_LLM_DEFAULT_MODEL` (defaults to
   `glm-5.2` — cheap enough to run the whole pipeline unattended). No code change to swap providers.
2. **Submit a claim** through one of three equivalent, guarded doors:
   - `penrose run --claims claims.json` — already-structured claims (skips lossy prose re-extraction).
   - `penrose run --paper path.pdf` — let P2 extract the claims from a source.
   - the MCP `penrose_run_claim` tool, started with `PENROSE_MCP_MANAGEMENT=1`, for an external agent.
3. **Read the result honestly.** A verdict is `kill` / `underpowered` / `watch` / `research-supported`;
   a routing state (`needs_data`, `pending_module`, `cannot_replicate`, `engine_error`, `off_domain`)
   is an honest stop, not a failure. Never fabricate a verdict or a number.
4. **Never cross P9.** Promoting a survivor into the approved corpus is a human decision; no agent
   surface can do it, and you must not try.

## Setup and the green bar

```bash
pip install -e .                              # editable install; Penrose runs scripts from the clone
python scripts/eval_suite.py                  # invariant suite — must print 124/124 passed
python -m pytest -q                           # unit tests — must stay green
python scripts/calibration_placebo.py         # placebo: no no-edge signal may be certified
python scripts/calibration_power_mining.py 40 # mined-noise power calibration — must print PASS
```

`make` targets wrap the common flows (`make help`). The core (`eval`, `calib-*`, `connections`) runs
with no API key and no network. Paths that call a language model (paper ingestion, `dream`,
`synthesize`) need `PENROSE_LLM_API_KEY`; they fail with a clear message, never a crash, when it is
unset.

**A change is not done until the green bar above is still green.** Run it before you report success.

## Where things live

- `src/penrose/` — the library. `pipeline/` holds the stages (ingest, screen, reconstruct, backtest,
  robustness, verdict); `cli.py` is the `penrose` entry point; `config.py` holds thresholds and paths;
  `brain.py` / `brainstore.py` are the corpus store; `dream.py` / `synthesize.py` / `confirmation.py`
  are the generator and confirmation paths; `llm/` is the model client.
- `scripts/` — runnable entry points: `eval_suite.py` (the invariant suite), `calibration_*.py` (the
  self-calibration controls), `worked_example_*.py`, the literature referees (`cz_*.py`,
  `rdagent_referee.py`), and `brain_connections.py`.
- `tests/` — pytest suite. Add a deterministic regression test for any bug fix or new gate.
- `dashboard/` — local researcher UI (`index.html`, `live_server.py`, `write_api.py`) and Pennie, the
  chat assistant.
- `modules/` — reviewed, deterministic strategy modules the pipeline can route claims to.
- `docs/` — `GATES.md` (plain-language gate reference) and assets.

## Building data adapters

Penrose reads data through provenance-carrying contracts in `src/penrose/data/`: `Series` (daily scalar
time series), `Panel` (dates × entity columns, for cross-sectional claims), `EventCalendar` (sorted,
de-duplicated event timestamps, for event-study claims), and `EventMarketPanel` (per-event bracket
markets, for prediction-market/bracket claims). Vendor adapters live in
`src/penrose/data/vendors/` and `src/penrose/data/*.py` (stooq, polygon, tiingo, fred, sec_edgar, …).
The `predictive_regression` claim type consumes two existing `Series` inputs (predictor + target) through
`src/penrose/pipeline/predictive_regression.py`; do not add a new data shape or trading overlay for it.
The `factor_spanning` claim type also consumes only existing `Series` inputs (candidate factor +
benchmark factors) through `src/penrose/pipeline/factor_spanning.py`; do not add a Panel shape or trading
overlay for it.
The `cross_sectional_sort` claim type consumes declared returns + characteristic `Panel` tables through
`src/penrose/data/panel_load.py` and `src/penrose/pipeline/cross_sectional_sort.py`; reuse
`data.xsection.form_factor`, require survivorship-corrected returns panels, and do not add a trading
overlay.
The `event_study` claim type consumes a return `Series` plus a declared event-calendar table through
`src/penrose/data/event_calendar_load.py` and `src/penrose/pipeline/event_study.py`; estimate the
baseline strictly before each event, emit per-event CAR, annualize by events/year, and do not add a
trading overlay.
The `forecast_skill` claim type consumes plain `Series` inputs (model forecast + realized target +
optional explicit benchmark forecast) through `src/penrose/pipeline/forecast_skill.py`; if the benchmark
is declared-implied, construct only `random_walk=Y.shift(1)` or `historical_mean=expanding_mean(Y).shift(1)`,
emit the per-period squared-loss differential, and do not add a trading overlay.

When you add or extend an adapter:

- **Satisfy a contract, don't invent a shape.** Return one of the contracts above (extend a contract only
  with a spec + tests). Carry `provenance` honestly; return `Unavailable`, never a fabricated value, when
  the contract can't be met.
- **No lookahead, checked at the boundary.** Only data known at/before the decision time may enter a row;
  frequency is checked so intraday can't be silently treated as daily. A settlement/outcome field must
  not leak information available only after close.
  Predictive-regression modules must align `X_t` with `Y_{t+h}` and freeze sign/z-score moments on the
  in-sample prefix only.
  Forecast-skill constructed benchmarks must be strictly causal: benchmark value at `t` may use only
  realized targets before `t`.
- **Deterministic + seeded.** No wall-clock dependence, no unseeded RNG, stable ordering.
- **Fail gracefully.** Missing keys, empty pulls, and degenerate inputs produce a clear message, never a
  traceback (invariant #5).
- **Public vs internal is enforced by tooling, not vigilance.** Build in the open; `scripts/build_public.sh`
  (allowlist + leak-check) decides what syncs to the public repo and aborts on a leak. Public-venue
  adapters (e.g. Kalshi, SEC EDGAR) can ship public; adapters that reference our private catalog contents
  or paths stay internal — keep those under an internal-only path/name so the allowlist excludes them.
- **Test it.** A new adapter/contract ships with deterministic tests (including a lookahead-rejection
  test) and keeps `eval_suite.py` green.

## Non-negotiable invariants (do not violate these)

1. **Never weaken a gate or a test to make a result pass.** If a change breaks a calibration or
   evaluation invariant, the change is wrong, not the gate.
2. **Keep discovery and confirmation separated.** Nothing on the discovery side may read the
   reserved/confirmation data. The single-use locked holdout confirms a distinct claim exactly once
   and then burns for that claim. Do not add a path that reads it during discovery or triage.
3. **Determinism.** Seed all randomness explicitly. No wall-clock dependence, no unseeded RNG, no
   nondeterministic ordering in evaluation paths.
4. **No alpha claims.** The verdict vocabulary is `kill` / `underpowered` / `watch` /
   `research-supported` (plus routing states). Never emit "profitable", "tradeable", or "alpha" as a
   verdict, and never describe a result that way.
5. **Fail gracefully.** Every path that can hit missing data, a missing key, an empty corpus, or a
   degenerate input must produce a clear, user-facing message, never a raw traceback.

## Conventions

- Match the surrounding code's style, naming, and comment density.
- New gates and calibration controls must ship with a deterministic test and keep `eval_suite.py`
  green. Verdict-logic changes get the most scrutiny; call them out explicitly.
- The corpus of invalidations may **inform** a human; it never **gates** a verdict automatically.
- This is a `0.x` research prototype: interfaces may change. Prefer additive, reversible changes.
