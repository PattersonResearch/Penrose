# Changelog

All notable changes to Penrose are documented here. This project follows a 0.x pre-1.0 line:
interfaces may change, and each minor release is a coherent batch of audited work.

## [0.5.0] — 2026-07-07

A capability release on one theme: the referee learns to adjudicate a new *kind* of claim, to judge a
strategy on a **neighborhood** of its parameters rather than one lucky configuration, and to evaluate a
batch of claims in parallel without loosening any statistical guarantee. Green bar: eval 106/106,
pytest 386 passed.

### Added
- **Event-market (bracket) adapter — a new data contract and strategy class.** Penrose can now referee
  strategies on resolution-outcome markets (Kalshi/Polymarket-style brackets), where each event is a
  strike range that settles win/lose. `EventMarketPanel` carries per-event decision-time prices, strikes,
  and settled outcomes; a deterministic `normal_bracket` executor prices each bracket and runs the standard
  backtest, so a bracket claim reaches a metric-bearing verdict through the same gates as any other
  strategy — no generated code between the claim and its verdict. See `docs/ADAPTERS.md`.
- **Parameter-robustness treatment (two parts).** (1) A claim's declared parameter grid is now *charged*
  to the multiple-testing denominator: declare a K-configuration search, and the deflation reflects a
  K-wide search — tuning is no longer free. (2) A new **parameter-fragility gate** re-runs a strategy
  across its declared grid and *kills* an edge that survives only at isolated parameter points; a genuine
  edge is robust to reasonable perturbation, a spurious one needs a magic point. The gate only ever
  downgrades a survivor, never rescues a kill.
- **Reconstructable synthesis candidates.** The candidate-generation layer now emits structured, buildable
  specifications (signal as a function of named series and parameters, with a declared prior grid) and
  admits a candidate only if it can actually be reconstructed — closing the gap where a self-generated
  hypothesis could be proposed but not tested.
- **Parallel claim execution.** Claims in a run can be evaluated concurrently (`--workers auto|N`, default
  `min(4, auto)` — hardware-aware and auto-reduced on small machines so a laptop is never swamped).
  Verdicts are **byte-identical** to a serial run (the multiple-testing denominator is pre-registered, not
  raced), so parallelism changes only speed, never the answer; near-linear speedup on the I/O-bound path.
  Live concurrency self-throttles when the model provider rate-limits.

### Changed
- The widow-maker / tail-risk gate (introduced in 0.4.2) is confirmed default-on in cap-and-warn mode.
- New module re-evaluation contract (`param_override`) lets deterministic modules be re-run at alternate
  parameter settings — the infrastructure powering the fragility gate.

## [0.4.2] — 2026-07-07

A capability-and-calibration release built around one theme: the referee should adjudicate a claim on
exactly what the claim asserts, and it should refuse to be fooled by a payoff that looks good on average
but is catastrophic in the tail. It completes the provided-series claim type introduced in 0.4.1, adds a
widow-maker gate, and closes several routing and self-correction gaps. Green bar: eval 101/101, pytest 340 passed.

### Added
- **Provided-series statistic path — complete build-out.** A claim that supplies its own pre-computed
  return/statistic series is now executed by a deterministic (non-LLM) module that pools the declared
  series and tests the stated statistic — no generated code stands between the claim and its verdict.
  A deterministic spec builder extracts the *declared decision inputs* from the claim's source text
  (and excludes context/benchmark/control series that are mentioned but not part of the decision), and a
  fidelity check judges the module against only the claim's stated contract. Because the construction is
  supplied rather than reproduced, these verdicts are provenance-capped: a provided series can be killed
  or held at watch, but never certified `research-supported` until it is reconstructed from primitives.
- **Tail-risk / widow-maker gate (default on).** The referee now flags bounded-up / unbounded-down
  (short-vol) payoffs — the canonical "looks great, then blows up" trade — using skew and a downside/upside
  tail ratio, with a severe-skew arm that fires on extreme skew alone (guarded by a minimum sample size).
  By default it caps an otherwise-supported survivor at `watch` (it never *certifies* a widow-maker) and
  always emits a visible `tail_asymmetric` warning, even when the verdict was already below the top tier;
  operators may opt into a hard kill. For provided-series claims it adds a caveat that a pre-aggregated
  series understates per-trade and cross-unit tail risk.

### Changed
- **Bounded self-correction across generation.** Both the spec generator and the implementation generator
  now self-correct when a fidelity check rejects their output: the specific divergence is fed back and the
  step retried within a bounded budget, instead of failing the run. This closes the last link in the
  generate → check → repair loop.
- **Deflation break decoupled from classification.** The reduced-deflation break for a single
  pre-registered cohort is now gated on an explicit pre-registration assertion in the claim, not on the
  (inherently spoofable) claim-type classifier. A data-derived hypothesis that merely *looks* like a
  provided series therefore receives normal, conservative deflation.
- **One authoritative claim type.** The resolved claim type is computed once and threaded through routing,
  fidelity, and deflation, removing an inconsistency in which different stages could classify the same
  claim differently and reach contradictory verdicts.

### Fixed
- **Missing data is `needs_data`, not `cannot_replicate`.** A run blocked purely by absent inputs now
  routes to `needs_data` (an honest, recoverable stop) instead of the stronger `cannot_replicate`.
- **Verbatim source-span gate tolerates real-world text.** The gate that requires a claim's quoted span to
  appear verbatim in its source now canonicalizes markdown and unicode punctuation and recovers
  non-contiguous verbatim spans via a guarded sentence-level fallback — ending a class of false
  zero-extraction `engine_error`s on otherwise-valid papers.

## [0.4.1] — 2026-07-04

Building on 0.4.0 (an internal-only milestone, documented below), this is the first public release
since 0.3.0. It is a correctness fix cycle: it hardens the append-only decisions log against a
data-loss path, makes a silent empty run a loud error, and adds a first-class claim type for claims
that supply their own statistics. Green bar: eval 97/97, pytest 267 passed.

### Added
- **First-class claim type for provided-series statistics.** A claim that supplies its own return or
  statistic series is now routed distinctly, checked before the descriptive and trading tally, so its
  verdict reflects what it actually asserts instead of being coerced into a descriptive or trading path.

### Fixed
- **Non-destructive, append-only supersession of decisions.** A re-run that superseded prior rows could
  truncate and rewrite the append-only `decisions.jsonl`, risking loss of earlier decisions when the new
  run then wrote nothing. Supersession is now strictly append-only (old rows are marked superseded, never
  deleted), and the recovery utility (`scripts/restore_decisions.py`) re-appends recovered rows under the
  engine's exclusive lock so it cannot interleave with a live run.
- **Loud failure on an empty run.** A run that extracts zero claims, or writes zero decisions, now routes
  to `engine_error` (a visible failure that needs attention) rather than completing silently. A silent
  empty run is treated as an engine fault, not a success.

## [0.4.0] — 2026-07-03 (internal-only; never published, superseded by 0.4.1)

A verdict-calibration release: the referee is now honest in both directions, and it enforces that
honesty as a permanent control. Monte-Carlo evidence (surfaced by refereeing an external code-complete
framework) showed the prior taxonomy inverted
at realistic effect sizes: it hard-killed true marginal edges as structurally dead while letting
best-of-K mined noise reach a survivor verdict. This release fixes both, adds honest error routing and
an edge-free offline fallback, and freezes the generative layer pending a corpus re-score. The two
calibration failure modes are now CI controls.

### Added
- **Enforcing power/mining calibration.** A Monte-Carlo control (`scripts/calibration_power_mining.py`)
  now gates both failure modes on every run: a true marginal edge (per-trade IC 0.05) must not
  structurally kill above 20 percent of seeds, and best-of-K mined noise submitted as a single claim
  must never reach a survivor verdict. Measured at the frozen realistic-edge floor, false-kill is
  10 percent and mined-noise-to-survivor is 0 percent. The config floor is frozen, so a breach means
  fix the gate, not the floor.
- **First-in-family deflation.** An external single claim is now deflated against a conservative
  Harvey-Liu-style effective-trials floor plus a variance prior, and paper-path multiple-testing
  families are canonicalized by domain, so a novel self-declared strategy class can no longer reset the
  deflation denominator. Best-of-500 mined noise routes to `underpowered`, not `watch`.
- **Post-sample survivor cap.** An external claim that declares its own evaluation window is capped at
  `watch` unless the bundle extends past that window, so a claim re-scored on its own mined sample
  cannot be certified.
- **`engine_error` routing state.** An internal engine or sandbox failure routes to a review queue as
  `engine_error` instead of masquerading as a data blocker; engine bugs no longer write phantom entries
  to the data shopping list, and a genuine `data_unavailable` reason is matched by prefix so an exception
  whose text merely contains those words is not misrouted.
- **BYO Tiingo IEX intraday adapter** for price-only OHLC bars, with explicit single-venue tagging when
  callers opt into IEX volume and no `DEFAULT_SERIES` wiring.
- **Cross-sectional reconstruction primitives:** provenance-carrying `Panel`, point-in-time `xsection`
  transforms, liquidity screens, and factor formation helpers, with no new dependencies.
- **Keyless SEC EDGAR fundamentals adapter** for point-in-time company filings, with disk caching and an
  optional `SEC_EDGAR_UA` contact override.
- **Opt-in MCP management surface** (`penrose-mcp --management` / `PENROSE_MCP_MANAGEMENT=1`) with guarded
  verdict fetch, deflation cohort registration, and claim/paper runs that return proposals only; the hard
  invariant holds that MCP cannot cross P9, write approved knowledge, bypass the Docker sandbox, or touch
  the holdout outside the guarded pipeline.

### Changed
- **Power-aware verdict taxonomy tested against a frozen floor.** The 3-fold consistency gate now asks
  whether the data could resolve a realistic edge (per-trade IC 0.05, the declared config floor), not
  whether it resolved the observed in-sample estimate. Using the upward-biased in-sample Sharpe as the
  power reference was circular and hard-killed true marginal edges that failed the consistency test by
  chance; a genuinely underpowered result is now labeled `underpowered` rather than a false `kill`. On a
  true annualized-0.8-Sharpe edge the structural false-kill rate falls from roughly 55 percent to
  roughly 10 percent (Monte-Carlo verified). The regime-fragility kill now requires a permutation-null
  confirmation before it fires, and the mislabeled `negative_dsr` reason is renamed `low_edge_t` and
  treated as an ambiguous, power-reclassifiable null.
- **Generative layer frozen by default.** dream, synthesize, and principle distillation are gated behind
  `PENROSE_GENERATIVE_LAYER` (default off) until the decision corpus is re-scored under the new taxonomy;
  the read-only surfaces and per-run record-keeping are unaffected.

### Fixed
- **Edge-free offline fallback.** The synthetic fallback used when a real catalog series is unavailable
  no longer contains a planted predictive signal (any edge found on it was a bug), and substitution warns
  loudly so a degraded run is visible.
- **Holdout no longer leaks its statistics.** The single-use holdout lock stores a digest and burn
  timestamp, not the burned Sharpe, so a later reader of the decisions log cannot recover the held-out
  result.
- **Monotonic trial ledger.** A re-registration can no longer shrink a strategy's declared search
  denominator or overwrite an already-scored row.
- **Fee gate uses the claimed edge.** The P4 fee-curve gate reads the claim's stated expected edge
  instead of a hardcoded constant, so it can actually fire, and records "not evaluated" when no numeric
  edge is stated.

## [0.3.0] — 2026-06-27

A robustness, agent-surface, and data release. Post-0.2.0 work, much of it surfaced by a fresh-clone
audit and by refereeing an external code-complete framework.

### Added
- **Tail-risk / widow-maker gate (default-off).** Every backtest now reports tail diagnostics (skew,
  CVaR-5/95, tail ratio, max loss vs gain, worst-vs-typical). An opt-in `TAIL_RISK_GATE` kills (or caps
  at `watch`) a stable, well-deflated strategy whose payoff is bounded-up / unbounded-down (negative
  skew, fat left tail) — the short-vol / positive-carry blind spot the other gates miss. Default-off, so
  no existing verdict moves; `tail_asymmetric` is a structural kill for principle formation.
- **Contrastive principles.** A second distiller learns from the survivor-vs-kill boundary: when a
  structural failure mode recurs in one domain but other domains yield survivors, it proposes an advisory
  contrastive principle (e.g. "regime_fragile is specific to trend-following; carry survives it").
  Additive (recurrence principles unchanged); surfaced via `views.principles()` and the read-only MCP.
- **Point-in-time futures data adapter** (`pysystemtrade`). A fail-open BYO local vendor that reads
  pysystemtrade adjusted-price CSVs, always resamples intraday→daily through the granularity gate before
  the data can reach verdict logic, and tags provenance back-adjusted + resampled. Instrument names are
  restricted to safe characters (no path traversal). Inactive/harmless when no futures dir is configured.
- **Agent-readable principle surface.** `views.principles()` and `views.proposals()` expose the
  distilled principle candidates and the propose-only store as structured read-only data, so an agent
  can pull and discuss "what candidates exist" without the dashboard. The read-only MCP routes its
  `penrose_principles` / `penrose_proposals` tools through these accessors (one read path, no drift);
  promotion to the approved brain still requires human P9.
- **`trend-following` domain** in cross-run principle inference, so trend / EWMAC claims cluster as
  trend-following instead of falling through to `other`.
- **Data-granularity verification** (`penrose.data.granularity`). Infers a series' empirical sampling
  frequency from its index and flags a mismatch with the expected frequency (e.g. intraday bars where a
  rule assumes daily, which silently corrupts every downstream statistic). The input-side analogue of
  the existing output `bars_per_year`-vs-span check. `DataBundle.granularity_warnings()` surfaces it;
  advisory and fail-open by default (no verdict change).

### Fixed
- **Trusted operator modules now ship in every clone.** The public `.gitignore` (`modules/*`) was
  dropping the reviewed `crypto_funding_carry` and `macro_vol_btc` modules from published clones, so a
  fresh clone failed the `PROVENANCE-SHELF` eval invariant (92/93) even though the README documents both.
  The two trusted modules now ship (generated `_auto` modules stay ignored); a cold clone passes 93/93.
- **Graceful capacity on low-turnover strategies.** `capacity_ci` raised `OverflowError` converting an
  infinite modeled capacity (a strategy that barely trades drives turnover toward zero) into an integer,
  crashing the entire backtest. It now drops non-finite resamples and reports capacity as undefined,
  consistent with the fail-visibly contract. Regression-tested.
- **Public test bar.** A `test_cli` check read `Makefile.public`, which the public build renames to
  `Makefile`, so the shipped test failed in the distribution it ships to. It now reads whichever exists;
  the public pytest bar is a clean 137 passed / 2 skipped.

### Docs
- Quickstart uses the real clone URL and surfaces the process-conditional worked example as the
  recommended first reproduction; eval count corrected to 93/93 in AGENTS.md / CLAUDE.md.
- Companion-paper bibliographies verified against publisher / arXiv records.

## [0.2.0] — 2026-06-25

A correctness-and-coverage release. The evaluation-invariant suite, the calibration battery
(null + placebo + injection), and the unit tests are green. The headline is two verdict-lane
correctness fixes plus a real data unblock.

### Verdict integrity
- **Order-independent deflation denominator (5c).** The Deflated Sharpe multiple-testing count is now
  pre-registered as a per-family cohort before evaluation, instead of a running tally read at backtest
  time. Previously the same strategy could get a different verdict depending on whether it ran 1st or
  8th in its family (early members were under-deflated). Now every member deflates by the full family
  size, uniformly and race-free. This can only tighten verdicts (it closes a selection-bias hole); no
  existing eval outcome moved.
- **Module generation learns to be faithful (6c).** Claims are routed by type
  (descriptive-statistical / trading-strategy / structural-proposition) so a descriptive claim (e.g. an
  unconditional mean) is implemented as a statistic test, not a trading backtest. A pre-backtest
  fidelity gate flags unfaithful specs before the expensive run, and a fidelity-rejection memory feeds
  past divergences back into generation. Fidelity only ever demotes or blocks, never promotes.
- **Regime-scope declaration.** A claim can pre-register a declared regime and be tested fairly within
  it (adherence-gated), instead of being falsely killed as regime-fragile for concentrating where it
  intends to trade.
- **CPCV / overfitting kill-lens.** Combinatorial purged cross-validation (Lopez de Prado) added as an
  independent robustness axis next to the bootstrap, permutation, and walk-forward gates.
- **Actionable `underpowered` verdicts.** A verdict that can't resolve a realistic edge now reports how
  much more would resolve it, the marginal OOS trades still needed (or the cross-sectional breadth
  alternative), turning a dead-end label into a sequential next step.
- **Independent fidelity verifier (optional).** The fidelity refuter can route to a genuinely
  independent second LLM provider (configurable via `PENROSE_LLM_VERIFIER_*`), reducing the correlated
  blind spots of a model checking its own work; it falls back to the same provider by default, and each
  result records whether the check was independent.

### Data ("works out of the box" for more than crypto)
- **Catalog-derived domain awareness.** The relevance gate and spec generator read the data catalog at
  runtime, so adding a new-domain series (equities, rates, inflation, commodities) makes those theses
  testable and lets the generator request real series names instead of inventing them. Fail-open to the
  built-in behavior when no catalog is present.
- **Keyless long-history adapter (Stooq).** A 6th out-of-the-box data adapter: decades of daily
  equity/index data with no API key, filling the gap where the free Alpha Vantage tier (~100 bars)
  flipped equity theses to `insufficient_data`.
- **Conservative name-resolution.** Near-miss series names resolve only on a unique high-confidence
  match; ambiguous names miss (never a wrong-series resolution).
- **Auto-fetch the `needs_data` loop.** When a claim needs a series an enabled vendor can supply
  unambiguously, Penrose fetches it once and re-tests, instead of only logging the request. Bounded and
  conservative (never supplies a wrong/ambiguous series).
- **Panel adapter.** A `panel` catalog adapter type for resolution-outcome / microstructure data
  (daily event-date aggregation), the framework for the largest class of data-blocked theses.

### Learning surface (P9 firewall intact)
- **Cross-run principle distillation.** Structural-kill principles are now distilled across the whole
  decision corpus, not just within a single run, so recurring failure modes actually surface.
- **Propose-only read store.** An agent-readable record of "what Penrose has learned" (`status:
  proposed`), strictly separate from the approved brain, promotion still requires human P9 approval.

### Robustness & honesty
- Output directories are created on startup (fresh/CI/sandboxed clones no longer fail on a missing
  `reports/`).
- A fidelity-refuter network timeout degrades to "fidelity unknown" and continues, instead of killing
  the run; LLM timeouts are configurable per role; optional `--max-claims`.
- Re-running an unchanged source is idempotent (atomic supersede by source identity), instead of
  appending duplicate decisions.
- Strategy-class alias collisions no longer log spurious warnings.

### Agentic surface & tooling
- **Read-only MCP server** (`pip install penrose[mcp]`, `penrose-mcp`). Five read-only tools let an
  agent query verdicts, proposed principles, open data-requests, and pipeline status over the Model
  Context Protocol. It exposes operations, not escape hatches: nothing over MCP can approve a verdict
  (P9 stays human), write the corpus, or run anything. `mcp` is optional; the core never requires it.
- `penrose run --json` emits a single machine-readable result object (verdicts + principle), so callers
  no longer have to tail a log; `penrose run --claims <claims.json>` injects pre-built structured claims
  and bypasses the lossy P2 re-extraction round-trip.
- Make targets honor a `PY` override (e.g. `make eval PY=./.venv/bin/python`); `.PHONY` completed;
  stress-testing docs linked.
- The public-build pipeline is hardened: tracked-files-only staging (gitignored operator artifacts can
  never ship), fund-specific leak markers, a symlink guard, and a dry-run-by-default sync.

## [0.1.0] — 2026-06-24

- Initial public release of Penrose: an independent, power-aware falsification referee for
  quantitative trading claims (the full pipeline, the Deflated-Sharpe statistical core, a
  self-calibrated detector, data adapters over a bring-your-own contract, and two companion papers).
