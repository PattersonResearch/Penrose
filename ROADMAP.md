# Roadmap

Penrose is an early (`0.x`) research prototype. This roadmap shares the direction and the open
problems; it is not a schedule, and nothing here is a commitment to a date. Priorities will shift with
what the community finds useful and where the science leads. If something here matters to you, or you
disagree with a priority, open a GitHub Discussion.

## The north star

A transparent, calibrated, reproducible standard for evaluating quantitative performance claims that
others can adopt, audit, and extend, plus a growing, shared **corpus of invalidations** that makes each
new claim cheaper to judge than the last. Penrose is the referee layer; it does not generate alpha and
makes no profitability claims, and that will not change.

## Where v0.6.0 stands

The falsification pipeline, the power-aware verdict taxonomy (now tested against a frozen
realistic-edge floor and enforced by a Monte-Carlo control), the anti-mining deflation and
post-sample caps, the self-calibration battery, the discovery/confirmation firewall with a single-use
locked holdout, the corpus with recurrence and contrastive principles, an opt-in tail-risk gate, an
input-side data-granularity check, honest `engine_error` routing, an edge-free offline fallback, an
agent-readable principle surface with an opt-in human-gated management MCP, and Pennie (the
corpus-grounded chat assistant) all ship and run today. Two failure modes that a fresh audit surfaced,
false-killing true marginal edges and passing best-of-K mined noise, are now closed and guarded as CI
controls at the frozen floor. Costs and capacity are still modeled rather than measured, several data
domains lack production adapters, independent replication is not yet automated, and the generative
layer (dream, synthesize, distill) is frozen behind a default-off flag pending a corpus re-score under
the new taxonomy. **0.5.0** adds an event-market (bracket) adapter — a new data contract and strategy
class for resolution-outcome markets — a parameter-robustness treatment (the declared grid is charged to
the multiple-testing denominator, and a fragility gate kills edges that survive only at isolated parameter
points), reconstructable synthesis candidates, and opt-in parallel claim execution whose verdicts are
byte-identical to a serial run. **0.6.0** adds cross-run principle distillation into a propose-only review store, composite strategy-family clustering, structured per-run traces with a `penrose triage` command, a bring-your-own-data contract with a reference loader, an auto-implementation loop-detection guard, and an expanded agent/MCP surface. See the [systems paper](docs/PENROSE_SYSTEMS_PAPER.md) for the full status.

## Directions we are pursuing

- **Reconstruction fidelity.** The central risk for prose inputs is testing a broken approximation of a
  strategy. Stronger reconstruction and a first-class path for code-complete candidates (where this risk
  disappears) is the highest-leverage area.
- **Point-in-time data adapters.** Leakage-safe adapters for more domains (equities, futures, FX,
  macro). Several ship today (FRED, Stooq, Databento, and a BYO-local `pysystemtrade` futures adapter
  that resamples intraday→daily through the granularity gate); the contract is in `src/penrose/data/`.
  More sources (and a fuller futures-roll/point-in-time treatment) are the most valuable place to contribute.
- **Independent replication and fresh-data confirmation.** A workflow that lets a generated or
  borderline claim graduate beyond `watch` only after confirmation on data it never touched during
  discovery.
- **Measured costs and capacity.** Replacing modeled fee/impact curves with paper-traded or observed
  fills, so survivors can be certified without the `watch` cap that modeled costs currently impose.
- **Sequential and power-aware evaluation.** The `underpowered` verdict is now a calibrated decision
  rather than a label: a marginal edge the data cannot resolve is separated from a structurally dead
  one, the power question is posed against a frozen realistic-edge floor rather than the observed
  in-sample estimate (using the upward-biased in-sample estimate was circular), and the false-kill and
  mined-noise-pass rates are enforced by a Monte-Carlo control. What remains is turning the resolution
  guidance (how much more data or cross-sectional breadth would resolve a marginal edge) into a full
  sequential design, drawing on the optimal-stopping literature (see references [8] and [9] in the
  README).
- **Agent-first operation.** The most powerful way to drive Penrose is to point an agent at it: ingest a
  paper or repo, reconstruct claims, register an honest cohort, run the grid, and read what survives, at
  a scale a human clicking a UI cannot match. The read-only MCP server and an opt-in, human-gated
  **management surface** (register a cohort, run a claim, fetch a verdict) both ship today, so an agent
  can operate the referee end to end while never crossing the P9 authorization gate: the orchestrator
  still cannot write the approved corpus, only a human can. In that model the **dashboard
  becomes mostly an overview and authorization surface** rather than the primary control panel; Pennie is
  wired to the MCP so you can manage and run the pipeline directly from the dashboard; and running Penrose
  inside a proper external agent harness is a first-class, recommended path alongside it. The tool surface
  ships as an extra (`pip install penrose[mcp]`), never in the core, and the management tools land only
  once the surface stabilizes and the human gate is provably un-crossable.
- **A public corpus commons (opt-in `penrose share`).** A way to pool anonymized invalidations so the
  corpus compounds across users rather than per-clone, and so the gates can be calibrated against real
  field usage. The trust posture is the whole point and is non-negotiable: it is an **explicit,
  preview-before-send command** a user runs deliberately (never silent telemetry, nothing leaves the
  machine behind your back), it reuses the publish-path leak-check to **strip anything sensitive** (raw
  claim text, strategy code, keys, machine paths), and it sends only **anonymized structured records**
  (verdict distributions, kill reasons, domains, versions, gate-fire counts) to a small serverless
  collector. Default off; opt-in; auditable. Infrastructure is not built yet.

## Non-goals (and they will stay non-goals)

- Penrose is not a backtester and not an alpha generator. Use it alongside those tools, not instead of
  them.
- It will never assert that a strategy is profitable or tradeable. It reports whether evidence survives
  honest testing.
- It is not a publication authority or an oracle. A verdict informs a human; it does not decide truth or
  investment suitability.

## How to influence the roadmap

Open a GitHub Issue for bugs and concrete requests, or a Discussion for direction. The two highest-value
contribution surfaces are **data adapters** and **reviewed strategy modules**; see
[CONTRIBUTING.md](CONTRIBUTING.md).
