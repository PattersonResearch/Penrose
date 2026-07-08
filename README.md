<div align="center">

<img src="docs/assets/penrose-wordmark.png" alt="Penrose" width="100%"/>

**Penrose is independent peer review for trading strategies.**

_Anyone can produce a backtest that looks spectacular. Most are statistical mirages. Penrose rebuilds the
strategy, stress-tests its evidence the way a skeptical reviewer would, and tells you whether the edge is
real or just an artifact of how it was found. It never tells you what to trade; it tells you what not to
believe._

[![Version](https://img.shields.io/badge/version-0.6.0-7c5cff.svg)](https://github.com/PattersonResearch/Penrose/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-7c5cff.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-2ee6ff.svg)](pyproject.toml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-3fb950.svg)](CONTRIBUTING.md)
[![Status: research preview](https://img.shields.io/badge/status-research_preview-9fb0c4.svg)](#)

</div>

---

## What Penrose is

Penrose is a **falsification referee** for quantitative trading claims. Give it a claim — from a paper, a
strategy generator, or your own research — and it rebuilds the strategy in a sandbox, stress-tests the
evidence through a stack of statistical gates, and returns a calibrated verdict on how much the result
deserves to be believed, given how it was found.

Its core skill is accounting for the **search behind a result**. If you test 500 ideas and report only
the best, its spectacular numbers are mostly luck. Penrose counts the whole search, not just the winner
you present, and deflates the evidence accordingly — grouping related attempts into **strategy families**
and charging every generated candidate to the shared search, so a machine that tries thousands of factors
cannot launder luck into a "discovery."

And it keeps what it learns. Every result becomes a durable **invalidation** — a record of what was
tested, why it failed, and under what conditions — and these accumulate into a **corpus of
invalidations** that compounds across every claim it referees. The rare claim that survives is flagged
for human review. Penrose finds **no new alpha** and promises none; its value is an honest, compounding
account of what does not survive proper testing, and the discipline to occasionally certify what does.

## What a verdict means

Every claim resolves to one of four verdicts. The point of the taxonomy is to separate *tested and
rejected* from *could not be resolved* — two very different findings that usually get collapsed into one.

| Verdict | What it means |
|---|---|
| `research-supported` | Cleared the full stack and was confirmed on a single-use locked holdout. Means "survived falsification," **not** "will be profitable" — and still requires human review. Rare, and the point. |
| `watch` | Survived the kill gates but not certified: a borderline result, or one capped because costs are modeled rather than measured, or a generated hypothesis not yet independently confirmed. A provisional survivor worth tracking. |
| `underpowered` | The data was too thin to resolve an effect of the claimed size. An *inconclusive*, not a rejection. Ships the minimum detectable effect and a resolution estimate (how much more data or breadth would settle it), so a careful skeptic is never mistaken for a machine that just says no. |
| `kill` | Tested with adequate power and rejected: the sample could resolve the claimed effect, and it does not survive deflation and the robustness stack. |

Claims that can't be evaluated yet get an honest **routing state** instead of a verdict — `needs_data`,
`cannot_replicate`, `pending_module`, `insufficient_data`, `engine_error`, `off_domain` — distinct from a judgment on the
evidence. [docs/GATES.md](docs/GATES.md) explains every test behind these verdicts in plain language.

## How it validates itself

A single corrected statistic doesn't catch all the ways a backtest lies. Penrose is a *suite*, and it
**validates its own detector** before it certifies anything:

- **Deflation for the whole search.** PSR/DSR evidence is scaled to the observed search size, scoped to
  the strategy *family* so unrelated families don't penalize each other. Selection bias is paid for, not
  ignored.
- **A confirmation you cannot game.** The final check is a sealed slice of data used exactly once — it
  confirms a survivor and then burns, so nothing, human or machine, can be quietly tuned until it slips
  through. Discovery and confirmation are firewalled.
- **Layered no-look-ahead.** Backtest structure, a Docker sandbox with static-leak scanning, a
  smoothed-vs-filtered eval, and a data-frequency gate at the boundary (intraday can never be silently
  treated as daily).
- **It checks itself first.** Before certifying, Penrose runs known-fake and known-real strategies
  through the same gates: noise must not pass (**placebo: 0/100 certified**), a planted edge must be
  caught, plus a multi-null battery (**0/300**) and a Monte-Carlo control on the power-aware verdict — a
  true marginal edge must not be over-killed, and best-of-K mined noise must never reach a survivor
  verdict. See [docs/STRESS_TESTING.md](docs/STRESS_TESTING.md).

## How it works

<img src="docs/assets/system-diagram.svg" width="100%" alt="How a claim moves through Penrose: ingestion and grounded extraction, screening, routing to a trusted module or a sandboxed reconstruction, the robustness and power gates, a power-aware verdict, and the corpus of invalidations that feeds back as a prior on new claims."/>

```
claim -> sandboxed reconstruction -> robustness stack -> power-aware verdict -> corpus
```

You give Penrose a claim — from a paper, a strategy generator, or yourself. It reconstructs the claim
(untrusted generated code **only ever runs inside a Docker sandbox**), runs it through the robustness
stack, and returns a calibrated verdict. The stack: PSR/DSR scoped to the search, a single-use locked
holdout, walk-forward, a regime kill-lens (edges concentrated in one calendar/vol/trend regime), purged
cross-validation, a bootstrap edge CI, a permutation test, a default-on **widow-maker gate** (a
bounded-up/unbounded-down payoff — small steady gains, rare large losses — is capped at `watch` and
warned), a **parameter-robustness** treatment (the declared parameter grid is charged to the
multiple-testing denominator, and a fragility gate kills an edge that survives only at one lucky
configuration), realistic cost and capacity, and a fidelity check that the code faithfully tests the
claim. For a plain-language tour of every gate, see [docs/GATES.md](docs/GATES.md).

For generated research, a native `penrose dream` adapter freezes an immutable candidate set and
preregisters the full generation budget *before* testing, so discarded candidates count toward the
denominator instead of letting the generator report only its favorites.

The **brain** accumulates verdicts and finds structure across them — shared failure modes, cross-domain
links, distilled **principles**. Hard rule: these connections **inform, they never gate.** Every new claim
is tested independently on its own data.

## Data: three contracts, bring your own

Penrose reads data through three contracts — `Series` (daily time series), `Panel` (dates × entities, for
cross-sectional claims), and `EventMarketPanel` (per-event bracket markets, for prediction-market claims).
Any source plugs in behind them via a small **adapter**. Keyless crypto and equity venues ship by default;
add your own with the [adapter guide](docs/ADAPTERS.md), or point `PENROSE_DATA_DIR` at your own catalog
using the documented [bring-your-own-data contract](docs/DATA_CONTRACT.md) and its runnable reference
loader (`examples/reference_loader/`).

## Quickstart

```bash
git clone https://github.com/PattersonResearch/Penrose && cd Penrose
pip install -e .

# no key, no external data — the keyless core:
penrose eval                 # planted strategies with known verdicts (106/106)
python scripts/worked_example_process_conditional.py  # identical returns, opposite verdicts by search scope
make calib-nulls             # the 5-null specificity battery (0/300)

# these need a model key and/or a data download (the notebook walks through both):
jupyter notebook notebooks/penrose_demo.ipynb
make cz-referee              # referee the published factor literature
PENROSE_GENERATIVE_LAYER=1 penrose dream -n 10   # generate + referee candidates (opt-in; see Limitations)
```

The core (`eval`, `calib-*`, the worked example) runs on `pip install -e .` alone — no API key, no
network — because it uses planted strategies with known code and has nothing to reconstruct. Refereeing a
real claim uses a model for the reconstruction stages. Commands that need a key fail with a clear message,
never a crash.

Runs parallelize across claims by default (`--workers auto`, or a fixed `N`); verdicts are **byte-identical
to a serial run** because the multiple-testing denominator is pre-registered rather than raced, so
parallelism only changes speed, never the answer.

### Model configuration

Every model-backed stage talks to one **OpenAI-compatible** endpoint through a single seam, so any
compatible provider works with no code change. It defaults to **GLM (`glm-5.2`)** — inexpensive and
capable enough to run the whole pipeline unattended.

```bash
export PENROSE_LLM_API_KEY=...                     # your provider key
export PENROSE_LLM_BASE_URL=https://api.z.ai/...   # any OpenAI-compatible endpoint (z.ai, OpenAI, Ollama, LiteLLM, ...)
export PENROSE_LLM_DEFAULT_MODEL=glm-5.2           # optional; defaults to glm-5.2
```

For a genuinely independent fidelity check (a different model grading the reconstruction), also set
`PENROSE_LLM_VERIFIER_MODEL` / `_BASE_URL` / `_API_KEY`; unset, the check falls back to the default
provider and each result records whether it was independent.

## Results

- **Referee a generator.** Of 16 factors [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)
  produced on real data: **14/16 killed per-factor, 0/16 survive deflation across the full search** —
  including positive-Sharpe factors a naive backtester certifies, which Penrose kills as regime-fragile.
- **Referee the published literature.** Across all 212 Chen-Zimmermann anomalies, survival is a *range*:
  **~48% per-anomaly down to ~3%** once deflated by the whole 212-anomaly search. That the number *depends
  on deflation scope* is itself the finding.
- **Post-publication decay.** Decay is universal (**~52%**, reproducing McLean-Pontiff). Penrose doesn't
  beat it, but its survivors retain **~4× the post-publication return** of its kills — the verdict sorts
  anomalies by post-decay value.
- **Calibration.** Placebo 0/100, 5-null battery 0/300. The detection floor is a sample-power artifact
  (it falls with more history and breadth toward the realistic 0.02–0.05 IC range), not a fixed limit.

Everything here is reproducible from this repository. See
[docs/PENROSE_SYSTEMS_PAPER.md](docs/PENROSE_SYSTEMS_PAPER.md) for the full write-up,
[docs/FPES_STANDARD_PAPER.md](docs/FPES_STANDARD_PAPER.md) for the evidence standard,
[docs/THE_PENROSE_EDGE.md](docs/THE_PENROSE_EDGE.md) for a narrative tour, and
[docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md) for the process-conditional example step by step.

## Where Penrose fits

Use Penrose *alongside* your existing stack, not instead of it. Backtesting frameworks
([zipline](https://github.com/quantopian/zipline), [backtrader](https://github.com/mementum/backtrader),
[vectorbt](https://github.com/polakowo/vectorbt),
[nautilus_trader](https://github.com/nautechsystems/nautilus_trader)) simulate a strategy's returns
faithfully. ML-for-finance platforms ([Qlib](https://github.com/microsoft/qlib),
[RD-Agent](https://github.com/microsoft/RD-Agent)) generate and rank candidates, and are strong at
discovery. Penrose adds the inference layer both leave open: it takes a strategy or a generated
candidate, deflates the evidence by the real search that produced it, holds a single-use confirmation
set, and validates its own detector.

The replication literature sets the prior — most published anomalies don't survive proper testing
(Hou-Xue-Zhang 2020) and decay after publication (McLean-Pontiff 2016) — so the honest starting point on
any edge is *"probably doesn't survive,"* which is exactly why an independent, calibrated referee is
worth having. **The methodology is established and cited, not invented here** — the Deflated Sharpe
Ratio (Bailey & López de Prado), the multiple-testing critique (Harvey-Liu-Zhu), the replication record.
The contribution is integration and self-calibration.

## Driving Penrose: CLI, agents, and MCP

Penrose is built to be operated by an agent as much as by a human. Locally, the CLI and filesystem *are*
the interface — `penrose run`, `penrose verdicts`, `penrose triage` (where claims die and why),
`penrose distill` (mine the corpus for proposed principles). See [AGENTS.md](AGENTS.md) if you are pointing
a coding agent at the repo.

For a **deployed or sandboxed** agent that shouldn't have a shell, an optional
[Model Context Protocol](https://modelcontextprotocol.io) server exposes the same operations safely:

```
pip install -e ".[mcp]"
penrose-mcp                 # read-only tools
penrose-mcp --management    # + guarded management tools
```

By design it **exposes operations, not escape hatches**. Read-only tools (`penrose_verdicts`,
`penrose_proposals`, `penrose_principles`, `penrose_data_requests`, `penrose_status`, `penrose_triage`)
read what Penrose already produced; management tools (`penrose_fetch_verdict`, `penrose_register_cohort`,
`penrose_run_claim`, `penrose_mine_principles`) drive the pipeline through its guardrails. **Nothing over
MCP** can approve a verdict (the P9 sign-off stays human), write the approved corpus, run a model-written
module outside the Docker sandbox, or touch the single-use holdout — so an agent can operate Penrose
without being able to make it fool itself.

**Pennie**, the corpus-grounded chat assistant in the dashboard, is the human-friendly front of this: a
deliberately skeptical collaborator whose one job is to turn a rough idea into a single testable
hypothesis, grounded (discovery-safe) in what Penrose has already killed. She shapes the *input*; the
pipeline does the judging, and nothing is submitted without your confirmation.

## Limitations

This is a **research prototype**. The contribution is methodology and measurement, not a product, and not
trading advice.

- **Reconstruction fidelity** is the central risk for prose inputs ("you tested a broken approximation of
  my strategy") — and a depreciating moat as foundation models improve. The strongest use is refereeing
  *code-complete* candidates, where the risk disappears.
- **Deflation isn't magic on the first look.** On the first single-claim run in a family, DSR is
  effectively PSR; deflation engages as Penrose sees more trials in the same family.
- **Data is the binding constraint,** and most series are static snapshots rather than point-in-time —
  which is why marginal single-asset edges on short samples land at `underpowered`, not a false kill.
- **The generative layer (dream/synthesize) is frozen** behind a default-off flag
  (`PENROSE_GENERATIVE_LAYER=1` to opt in) pending a corpus re-score; generated hypotheses have no
  external evidentiary anchor and are capped at `watch`.
- **Claim-type routing is English-keyword-based** and fails open (a mis-routed claim is tested as a
  strategy, never crashed).

See [ROADMAP.md](ROADMAP.md) for where this is going.

## FAQ

**Does Penrose need an LLM to run?** Only to referee a real claim. The keyless core (`penrose eval`, the
calibration batteries, the worked example) runs with no key and no network, because it uses planted
strategies with known code. Refereeing a real claim uses a model for the reconstruction stages.

**Is the pipeline's model the same as an agent driving Penrose?** No — separate roles. The pipeline's
*internal* model (default `glm-5.2`) reconstructs a claim inside the stages. An *external* agent (CLI,
runner, or the MCP `penrose_run_claim` tool) operates Penrose from the outside and never touches the
reconstruction. One provider runs everything; a second, independent one is optional and affects only the
fidelity gate.

**Do I need a GPU or special hardware?** No. Penrose orchestrates and evaluates; the model runs wherever
your endpoint is. Docker is required for the reconstruction sandbox; no GPU is.

## Why is it called Penrose?

Named for the [Penrose process](https://en.wikipedia.org/wiki/Penrose_process), Roger Penrose's mechanism
for extracting energy from a rotating black hole. Almost everything that enters is swallowed, but a rare
fraction escapes the ergosphere carrying away *more* energy than it arrived with. The gauntlet of
deflation, costs, regimes, and a single-use holdout is the black hole; nearly every edge that enters is
destroyed, and what falls in is not wasted — it becomes the corpus that Penrose mines for new candidates.
The rare claim that escapes is the fraction that gets out enriched: never guaranteed profit, only the
candidate worth confirming.

## References

1. Bailey, D. H., and López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5), 94-107. <https://ssrn.com/abstract=2460551>
2. Harvey, C. R., Liu, Y., and Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68. <https://doi.org/10.1093/rfs/hhv059>
3. Hou, K., Xue, C., and Zhang, L. (2020). Replicating Anomalies. *Review of Financial Studies*, 33(5), 2019-2133. <https://doi.org/10.1093/rfs/hhy131>
4. Chen, A. Y., and Zimmermann, T. (2022). Open Source Cross-Sectional Asset Pricing. *Critical Finance Review*, 11(2), 207-264. <https://doi.org/10.1561/104.00000112> Code/data: <https://github.com/OpenSourceAP/CrossSection>
5. McLean, R. D., and Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1), 5-32. <https://doi.org/10.1111/jofi.12365>
6. Politis, D. N., and Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303-1313. <https://doi.org/10.1080/01621459.1994.10476870>
7. Li, Y., et al. (2025). R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. <https://arxiv.org/abs/2505.15155> Code: <https://github.com/microsoft/RD-Agent>

## License & contact

Apache-2.0 (see `LICENSE`). The framework is open; if you build on it, a citation is appreciated. Bugs,
questions, and ideas: **GitHub Issues / Discussions**. To follow the project, **Star/Watch** the repo. For
anything else — including interest in a hosted referee down the line — email **hello@penrose.systems**. No
product is for sale today; this is a research release.

## A note from the author

Thank you for taking the time to look at Penrose. I built it because I wanted an honest, shared reference
for the quant community — a place where a claim has to survive real scrutiny before anyone believes it,
and where what fails is kept and learned from instead of quietly discarded. It is a research prototype,
not a finished product, and I would genuinely rather you try to break it than take it on faith. If it
helps you, or if you find where it is wrong, I would love to hear from you at <hello@penrose.systems>.

Good luck,

Chuck