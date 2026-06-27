# Changelog

All notable changes to Penrose are documented here. This project follows a 0.x pre-1.0 line:
interfaces may change, and each minor release is a coherent batch of audited work.

## [0.3.0] — 2026-06-27

A robustness, agent-surface, and data release. Post-0.2.0 work, much of it surfaced by a fresh-clone
audit and by refereeing an external code-complete framework, then adversarially swarm-audited.

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

A correctness-and-coverage release. Every change was implemented and adversarially swarm-audited;
the evaluation-invariant suite, the calibration battery (null + placebo + injection), and the unit
tests are green. The headline is two verdict-lane correctness fixes plus a real data unblock.

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
