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

## Where v0.1.0 stands

The falsification pipeline, the power-aware verdict taxonomy, the self-calibration battery, the
discovery/confirmation firewall with a single-use locked holdout, the corpus, and Pennie (the
corpus-grounded chat assistant) all ship and run today. Costs and capacity are modeled rather than
measured, several data domains lack production adapters, and independent replication is not yet
automated. See the [systems paper](docs/PENROSE_SYSTEMS_PAPER.md) for the full status.

## Directions we are pursuing

- **Reconstruction fidelity.** The central risk for prose inputs is testing a broken approximation of a
  strategy. Stronger reconstruction and a first-class path for code-complete candidates (where this risk
  disappears) is the highest-leverage area.
- **Point-in-time data adapters.** Leakage-safe adapters for more domains (equities, futures, FX,
  macro). One reference adapter ships; the contract is in `src/penrose/data/`. This is the most
  valuable place to contribute.
- **Independent replication and fresh-data confirmation.** A workflow that lets a generated or
  borderline claim graduate beyond `watch` only after confirmation on data it never touched during
  discovery.
- **Measured costs and capacity.** Replacing modeled fee/impact curves with paper-traded or observed
  fills, so survivors can be certified without the `watch` cap that modeled costs currently impose.
- **Sequential and power-aware evaluation.** Turning the `underpowered` verdict from a label into a
  decision: how much more data or cross-sectional breadth would resolve a marginal edge, drawing on the
  optimal-stopping literature (see references [8] and [9] in the README).
- **Agent integration.** An optional MCP server so coding agents can run the referee directly. It would
  ship as an extra (`pip install penrose[mcp]`), not in the core, once the tool surface stabilizes.
- **A public corpus commons.** An opt-in, anonymized way to share invalidations so the corpus compounds
  across users rather than per-clone. Opt-in and privacy-preserving by design; infrastructure is not
  built yet.

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
