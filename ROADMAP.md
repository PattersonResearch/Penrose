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

## Where v0.3.0 stands

The falsification pipeline, the power-aware verdict taxonomy, the self-calibration battery, the
discovery/confirmation firewall with a single-use locked holdout, the corpus (now with contrastive
principles), an opt-in tail-risk gate, an input-side data-granularity check, an agent-readable
principle surface, and Pennie (the corpus-grounded chat assistant) all ship and run today. Costs and
capacity are still modeled rather than measured, several data domains lack production adapters, and
independent replication is not yet automated. See the [systems paper](docs/PENROSE_SYSTEMS_PAPER.md)
for the full status.

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
- **Sequential and power-aware evaluation.** Turning the `underpowered` verdict from a label into a
  decision: how much more data or cross-sectional breadth would resolve a marginal edge, drawing on the
  optimal-stopping literature (see references [8] and [9] in the README).
- **Agent-first operation.** The most powerful way to drive Penrose is to point an agent at it: ingest a
  paper or repo, reconstruct claims, register an honest cohort, run the grid, and read what survives, at
  a scale a human clicking a UI cannot match. The read-only MCP server ships today; the direction is to
  add **human-gated management and run tools** to it (register a cohort, run a claim, fetch a verdict) so
  an agent can operate the referee end to end, while never crossing the P9 authorization gate, the
  orchestrator still cannot write the approved corpus, only a human can. In that model the **dashboard
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
