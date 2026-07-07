<div align="center">

<img src="docs/assets/penrose-wordmark.png" alt="Penrose" width="100%"/>

**Penrose is independent peer review for trading strategies.**

_Anyone can produce a backtest that looks spectacular. Most are statistical mirages. Penrose rebuilds
the strategy itself, stress-tests its evidence the way a skeptical reviewer would, and tells you
whether the edge is real or just an artifact of how it was found. The same returns can be a genuine
edge or a lucky fluke, depending on how many things you tried to get there. It never tells you what to
trade; it tells you what not to believe._

[![Version](https://img.shields.io/badge/version-0.5.0-7c5cff.svg)](https://github.com/PattersonResearch/Penrose/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-7c5cff.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-2ee6ff.svg)](pyproject.toml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-3fb950.svg)](CONTRIBUTING.md)
[![Status: research preview](https://img.shields.io/badge/status-research_preview-9fb0c4.svg)](#)

</div>

---

## The one thing most backtests ignore: how many strategies you tried

If you test 500 ideas and show me only the best one, its spectacular numbers are mostly luck, not
edge. Penrose counts the whole search behind a claim, not just the winner you present, and deflates
the result accordingly. That single correction, the multiple-testing problem, is why most "amazing"
backtests do not survive it. The same idea lets Penrose group related attempts into **strategy
families** and charge every generated candidate to the shared search, so a machine that tries
thousands of factors cannot launder luck into a "discovery."

## For quants: what it actually does

Penrose is an independent, power-aware falsification **referee for quantitative trading claims**. It
is not a strategy generator and not a backtester. A backtester measures one strategy's returns and
moves on; Penrose asks whether a claim's evidence can be believed at all, given how it was discovered,
and it keeps what it learns. Every result becomes a durable, reusable **invalidation** that records
what was tested, why it did not survive, and under what conditions. These accumulate into a growing
**corpus of invalidations**, a compounding map of what does not work that no single backtest can
build. The rare claim that survives falsification is flagged for human review; the far more common
invalidation is kept and mined for **principles** (patterns that are valuable despite the broader
failure). An experimental, opt-in synthesis process can mine the corpus for connections between those
principles to form unconfirmed hypotheses to test, a speculative frontier sometimes called
[candidate alpha](#generating-candidate-alpha)<sup>[[8]](#references)[[9]](#references)</sup>.

In use, you give Penrose a claim, whether from a paper, a strategy generator, or yourself. It
reconstructs the claim in a sandbox, tests it under a rigorous robustness stack, **validates its own
detector**, and returns a calibrated [verdict](#verdicts). Penrose finds **no new alpha** and makes no
promise to; verdicts are not financial advice or a strategy endorsement. Its value is an honest,
accumulating account of what does not survive proper testing, and the discipline to occasionally
certify what does.

---

## Features

- **A full falsification stack, not a single statistic.** Each gate targets a different way a backtest
  can fool you, and a survivor has to clear all of them: evidence deflated for the whole search behind
  the claim (PSR/DSR), sign stability across a three-way time split, robustness when its single best
  market regime is dropped, purged cross-validation, a bootstrap confidence band on the edge, a shuffle
  (permutation) test, walk-forward consistency, a default-on widow-maker check for bounded-up /
  unbounded-down payoffs (small steady gains, rare large losses) that caps such a survivor at `watch`
  and warns, realistic cost and capacity, and a one-time locked holdout. The deflated
  Sharpe is the spine that accounts for the search, but it is one gate among many, never the verdict on
  its own. See [docs/GATES.md](docs/GATES.md) for every gate in plain language.
- **It separates "doesn't work" from "not enough data."** Every verdict reports the smallest edge the
  data could have detected. When the claimed edge sits below that floor, the result is `underpowered`,
  not `kill`, so a careful skeptic is never mistaken for a machine that just says no to everything.
  "We couldn't resolve it" never becomes "it's dead."
- **It proves it works before you trust it.** Before certifying anything, Penrose runs known-fake and
  known-real strategies through the very same gates and checks it gets them right: noise must not pass
  (placebo), a planted edge must be caught (injection), plus dead-state, native-breadth,
  persistence-matched, and tail-risk controls and a multi-null battery. Almost nothing else validates
  its own instrument. See [docs/STRESS_TESTING.md](docs/STRESS_TESTING.md) for the runnable controls.
- **A one-time confirmation you cannot game.** The final check is a sealed slice of data used exactly
  once. It confirms a survivor and then burns, so nothing, human or machine, can be quietly tuned until
  it slips through.
- **A compounding memory of what fails.** Every verdict is kept as a durable, reusable record of what
  was tried and why it did not survive, so the map of dead ends grows in a way no single backtest can
  build. Penrose mines it for recurring lessons and contrasting ones (where a failure mode that kills
  one kind of strategy spares another), all advisory, never gating a new test.
- **Referees any source, and charges the whole search.** Papers, your own theses, code-complete
  strategies, or machine-generated hypotheses all run through the same pipeline, and a generator that
  produces thousands of candidates is charged for the entire search, not just its best-looking one.
- **Sandboxed and reproducible.** Untrusted generated code only ever runs inside a Docker sandbox, and
  evaluation paths are deterministic and seeded.
- **Pennie, a research assistant grounded in the corpus** that helps you shape a rough idea into a
  single testable hypothesis ([see below](#pennie-your-research-assistant)).

## Why Penrose

Automated quant-research systems now propose, code, and backtest factors with little human input.
They are improving fast, but they select winners by test-set performance over a large search,
with no penalty for the size of that search. Their "winners" are inflated by selection bias, and
selection bias is only one of the ways a backtest lies: low statistical power, look-ahead,
regime-specific luck, ignored costs, and post-publication decay all do the same. A single corrected
statistic does not catch all of them; a referee needs a suite of tools to provide proper verdicts.

Academic record sets the prior: most published anomalies don't survive proper testing
(Hou-Xue-Zhang 2020), they decay after publication (McLean-Pontiff 2016), and the "factor zoo" came
from an enormous undisclosed search (Harvey-Liu-Zhu 2016). So the honest prior on any published or
generated edge is **"probably doesn't survive,"** which is exactly why an *independent, calibrated*
referee has value.

Penrose is that referee. It sits between the **generators** (which don't deflate) and the
**self-audit tools** (which only test your *own* pipeline): it ingests a third party's claim, rebuilds
it, runs it through the full stack (PSR/DSR evidence scaled to the observed search, power accounting,
robustness gates, costs, and a locked holdout), and records a calibrated, power-aware,
provenance-tracked verdict.

## How it works

<img src="docs/assets/system-diagram.svg" width="100%" alt="How a claim moves through Penrose: ingestion and grounded extraction, screening, routing to a trusted module or a sandboxed reconstruction, the robustness and power gates, a power-aware verdict, and the corpus of invalidations that feeds back as a prior on new claims."/>

```
claim -> sandboxed reconstruction -> robustness stack -> power-aware verdict -> corpus
```

For a plain-language tour of every gate in that diagram, with what each one catches and why you
want it, see [docs/GATES.md](docs/GATES.md).

For generated research, a native `penrose dream` source adapter first freezes an immutable candidate
set and preregisters the full generation budget. Eligible hypotheses then enter the same P3-P8
falsification path as extracted paper claims. This makes discarded candidates count toward the
multiple-testing denominator instead of letting the generator report only its favorites.

The robustness stack: PSR/DSR scoped to the observed search, a single-use locked holdout,
walk-forward, a regime kill-lens (it catches edges concentrated in one calendar/vol/trend regime)
with an adherence-gated declared-regime scope, combinatorial purged cross-validation, bootstrap edge
CI, a permutation test, a default-on tail-risk (widow-maker) gate that caps a bounded-up/unbounded-down
survivor at `watch` and warns (hard-kill is opt-in), capacity/impact, a fee curve, and a fidelity check
that the code faithfully tests the claim. Input series are also frequency-checked at the data boundary
(so intraday data can never be silently treated as daily); data adapters include keyless crypto and
equity venues plus bring-your-own futures and Tiingo IEX intraday adapters. Tiingo IEX is price-only
by default: its volume is single-venue, not consolidated tape volume, so volume-gated intraday claims
route to `needs_data` unless a consolidated intraday feed such as paid Polygon is supplied. Untrusted
auto-generated code **only ever runs inside a Docker sandbox,** never in Penrose's own process.

Penrose reads data through three contracts — `Series` (daily time series), `Panel` (dates × entities,
for cross-sectional claims), and `EventMarketPanel` (per-event bracket markets, for prediction-market
claims) — and any source plugs in behind them via a small **adapter**. See
[docs/ADAPTERS.md](docs/ADAPTERS.md) to add one (the contracts, the no-look-ahead/deterministic
invariants, and examples to copy). For cross-sectional claims the data layer also has pure `xsection`
transforms for point-in-time reconstruction: as-of panel assembly, deterministic cross-sectional
ranks/z-scores, liquidity screens, and long-short factor formation. These are referee primitives for
rebuilding what a claim describes; they are not signal-generation tools and make no claim that any
factor is profitable. SEC EDGAR is available as a keyless, point-in-time fundamentals
panel source for public company filings; set `SEC_EDGAR_UA` to your contact string for SEC fair-access
requests, or Penrose uses a generic project contact.

The **brain** accumulates verdicts and finds structure across them (shared failure modes,
cross-domain links, principles). Hard rule: these connections **inform, they never gate.** The
corpus contextualizes a result for a human; it never auto-rejects a new idea. Every new claim is
tested independently on its own data.

### Generating candidate alpha

Penrose can also produce its own **candidate alpha** from the user's corpus of past results, not only
referee external ones (the `dream` and synthesize paths). These candidates are fundamentally **untested hypotheses**, never
predictions of profit: each one re-enters the same falsification path above, is capped at `watch`
until independently confirmed, and the quality of corpus-generated candidates is an open research
question rather than a promised standard.

## Verdicts

Every claim resolves to one of four verdicts. The point of the taxonomy is to separate *tested and
rejected* from *could not be resolved*, which most backtesters conflate.

| Verdict | What it means |
|---|---|
| `research-supported` | Cleared the full stack: PSR/DSR evidence above threshold for the observed search, three-fold sign-stable, regime-robust, bootstrap CI excludes zero, permutation-clean, walk-forward-consistent, and confirmed on a single-use locked holdout. It means "survived falsification," not "will be profitable," and still requires human review. |
| `watch` | Survived the kill gates but not certified: either it sits in the borderline band, or it is capped (for example, costs are modeled rather than measured, or it is a generated hypothesis not yet independently confirmed). A provisional survivor worth tracking. |
| `underpowered` | The data was too thin to resolve an effect of the claimed size. Not a rejection, an inconclusive. Every verdict ships a minimum detectable effect; when the claimed edge is below it, the result is `underpowered` rather than a false `kill`. It also ships a *resolution estimate*: roughly how many more out-of-sample trades, or how much cross-sectional breadth, would settle it. |
| `kill` | Tested with adequate power and rejected. The sample could resolve an effect of the claimed size, and the claim does not survive deflation and the robustness stack. |

Claims that cannot be evaluated yet receive a routing state instead (`needs_data`, `pending_module`,
`cannot_replicate`, `insufficient_data`, `off_domain`), which is distinct from a verdict on the evidence.

New to this? [docs/GATES.md](docs/GATES.md) explains every test behind these verdicts in plain
language, written for a motivated aspiring quant rather than only for a seasoned academic.

## Where Penrose fits

Penrose is not a backtester and not an alpha generator, and it does not compete with either. It is
the layer they leave out.

- **Backtesting frameworks** ([zipline](https://github.com/quantopian/zipline),
  [backtrader](https://github.com/mementum/backtrader),
  [vectorbt](https://github.com/polakowo/vectorbt),
  [nautilus_trader](https://github.com/nautechsystems/nautilus_trader)) simulate a strategy's
  returns. They measure faithfully, but they do not account for how many strategies you tried before
  keeping the one you report.
- **ML-for-finance platforms** ([Qlib](https://github.com/microsoft/qlib),
  [RD-Agent](https://github.com/microsoft/RD-Agent)) generate and rank factors and models. They are
  strong at discovery, and Penrose treats them as exactly that: generators whose search size has to be
  paid for. Neither deflates a candidate by the size of the search that produced it, holds a single-use
  confirmation set, or validates its own detector.
- **The methodology is established and cited, not invented here:** the Deflated Sharpe Ratio (Bailey
  and Lopez de Prado), the factor-zoo and multiple-testing critique (Harvey, Liu, and Zhu), and the
  replication record (Hou-Xue-Zhang, McLean-Pontiff).

Penrose is the inference-governance layer on top of that stack. The contribution is **integration and
self-calibration**, not a "first" or a discovery of alpha: it takes a claim or a generated candidate,
deflates by the real search size, holds a single-use confirmation set, validates its own detector, and
returns a calibrated verdict. Use it *alongside* those tools, not instead of them.

## Quickstart

```bash
git clone https://github.com/PattersonResearch/Penrose && cd Penrose
pip install -e .             # editable: Penrose runs the scripts that ship in the clone

# the guided demo runs the clean-room path (no key, no external data for the core):
jupyter notebook notebooks/penrose_demo.ipynb

# or from the command line (no key needed):
penrose eval                 # ground-truth: planted strategies with known verdicts (106/106)
python scripts/worked_example_process_conditional.py  # start here: identical returns, opposite verdicts by search scope
make calib-nulls            # the 5-null specificity battery (0/300)
make calib-sensitivity      # the detection-threshold sweep
make connections            # the brain's advisory connection-discovery

# these need the CZ data download and/or a model key (see the notebook):
make cz-referee             # referee the published factor literature (needs the CZ data)
penrose dream -n 10 --generate-only  # preregister a generated search (needs a model key)
penrose dream -n 10          # generate candidates and send eligible ones through the referee (needs a key)
make dash                   # the researcher dashboard (localhost)
```

The core (`eval`, `calib-*`, `connections`) runs on `pip install -e .` alone, with no API key and no
external data. The full pipeline (ingesting a paper) and the literature/generator experiments need a
model key and/or a data download; the notebook walks through both. Commands that need a key fail with a
clear message, never a crash, if one is not set.

### Model configuration

Every model-backed stage (extraction, spec generation, implementation, fidelity) talks to one
**OpenAI-compatible** endpoint through a single seam, so any compatible provider works with no code
change. It defaults to **GLM (`glm-5.2`)** — an inexpensive, capable model that runs the whole pipeline
well, so an agent or a batch job can referee at low cost. Three environment variables configure it:

```bash
export PENROSE_LLM_API_KEY=...                     # your provider key
export PENROSE_LLM_BASE_URL=https://api.z.ai/...   # any OpenAI-compatible endpoint (z.ai, OpenAI, Ollama, LiteLLM, ...)
export PENROSE_LLM_DEFAULT_MODEL=glm-5.2           # optional; defaults to glm-5.2
```

For a genuinely independent fidelity check (a different model judging the reconstruction), also set
`PENROSE_LLM_VERIFIER_MODEL` / `PENROSE_LLM_VERIFIER_BASE_URL` / `PENROSE_LLM_VERIFIER_API_KEY`; unset,
the check falls back to the default provider and each result records whether it was independent.

## MCP server (optional)

An agent can query Penrose over the [Model Context Protocol](https://modelcontextprotocol.io):

```
pip install -e ".[mcp]"
penrose-mcp                 # runs the read-only MCP server
penrose-mcp --management    # opt-in proposal/bookkeeping management tools
```

It exposes five **read-only** tools: `penrose_verdicts`, `penrose_proposals`,
`penrose_principles` (distilled cross-run proposals), `penrose_data_requests`, and `penrose_status`.

With `--management` (or `PENROSE_MCP_MANAGEMENT=1`), it also exposes guarded management tools:
`penrose_fetch_verdict`, `penrose_register_cohort`, and `penrose_run_claim`. These can fetch one
verdict, write deflation-ledger bookkeeping, or run a paper/claim through the normal sandboxed pipeline;
they only return proposals/bookkeeping.

By design it **exposes operations, not escape hatches**: every tool only reads results Penrose already
produced unless management mode is explicitly enabled. Nothing over MCP can approve or promote a
verdict (the P9 sign-off stays human), write the approved corpus, run a model-written module outside
the Docker sandbox, or touch the single-use holdout outside the guarded pipeline path — so an agent can
operate Penrose without being able to make Penrose fool itself. `mcp` is an optional extra; the core
install never requires it.

## Results

- **Referee a generator.** Of 16 factors a generative system ([Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)) produced on real
  data: **14/16 killed per-factor, 0/16 survive deflation across the full search,** including
  positive-Sharpe factors a naive backtester certifies, which Penrose kills as regime-fragile.
- **Referee the published literature.** Across all 212 Chen-Zimmermann anomalies, survival is a
  *range*: **~48% per-anomaly down to ~3%** when deflated by the whole 212-anomaly search. The
  dependence of "survival" on deflation scope is itself a finding.
- **Post-publication decay.** Decay is universal (**~52%**, reproducing McLean-Pontiff). Penrose does
  *not* beat it, but its survivors retain ~**4x the post-publication return** of its kills, so the
  verdict sorts anomalies by post-decay value.
- **Calibration.** Placebo: 0/100 noise signals certified. 5-null battery: 0/300. The detection
  floor is a sample-power artifact (it falls with more history and breadth toward the realistic
  0.02 to 0.05 IC range), not a fixed limit.

See [`docs/PENROSE_SYSTEMS_PAPER.md`](docs/PENROSE_SYSTEMS_PAPER.md) for the full system write-up, and
[`docs/FPES_STANDARD_PAPER.md`](docs/FPES_STANDARD_PAPER.md) for the underlying evidence standard. All points listed here are reproducible with the information found in this repository.
For a narrative tour see [`docs/THE_PENROSE_EDGE.md`](docs/THE_PENROSE_EDGE.md), and
[`docs/WORKED_EXAMPLE.md`](docs/WORKED_EXAMPLE.md) walks through the deterministic process-conditional example step by step.

## Pennie, your research assistant

Pennie is the chat assistant built into the Penrose dashboard. She is a skeptical research
collaborator, not an oracle and not a trader. Her one job is to help you turn a rough idea into a
single hypothesis Penrose can actually test, then hand it to the pipeline. She is the chat box in the
diagram above, the one reading from the corpus of invalidations.

**How she works.** Pennie is a chat loop with two things wired in. First, a deliberately skeptical
role that steers every conversation toward a testable claim: she pushes you to pin down the five things
Penrose needs, which are a single falsifiable directional claim, a signal-to-forward-return mechanism
with no look-ahead, data that plausibly exists, a measurable horizon with non-overlapping trades, and an
explicit falsifier. Second, **corpus grounding**: before each reply she runs a read-only,
discovery-safe retrieval over the corpus of invalidations and prior findings, and uses what surfaces to
ground the conversation (for example, "Penrose already killed a similar funding-carry claim as
regime-fragile"). When the idea is testable you click **Prepare Hypothesis**, and Pennie rewrites the
whole conversation into one clean labeled thesis (Claim, Mechanism, Scope, Horizon, Data, Falsifier).
You review it, and only if you confirm does it enter the inbox for the next run. Nothing is submitted
behind your back.

**What Pennie can do**

- Discuss signals, mechanisms, data sources, costs, and regimes, and sharpen a vague idea into a single
  falsifiable hypothesis.
- Ground answers in Penrose's own prior findings, the corpus of invalidations, discovery-safe.
- Read an attached paper's text in-conversation for discussion (it is not auto-queued for testing).
- Draft a clean, ingestable thesis and, on your confirmation, queue it for the falsification pipeline.

**What Pennie cannot (and will not) do**

- She never says an idea "works," "is profitable," or "makes money." She cannot know that; only the
  backtest decides. She will instead surface look-ahead, overlapping-window inflation, missing data, and
  cost problems.
- She does not run the backtest or assign a verdict. Pennie shapes the *input*; the falsification
  pipeline and its gates do the judging.
- She never sees or touches the locked holdout. Her corpus access is discovery-safe by construction, so
  talking to Pennie cannot contaminate a future confirmation.
- She does not auto-submit. A human confirms before any thesis enters the inbox.
- She is an optional, local, model-backed convenience. The core test suite, calibration, and worked
  example need no API key; Pennie does, because she is a generation path.

### Retrieval

Pennie's chat path grounds replies in prior corpus findings for context only, never as proof. The
corpus (`dashboard/corpus.json`) is built up as you run the pipeline, so on a fresh clone it is empty
and retrieval simply returns nothing until you have accumulated results. Retrieval runs entirely in
this repo: `pip install -e ".[embed]"` enables in-process FastEmbed vector search with
`BAAI/bge-small-en-v1.5`; without that optional extra, retrieval falls back to deterministic lexical
scoring. No external embedding service is required.

## Limitations

- **Reconstruction fidelity** is the central risk for prose inputs ("you tested a broken approximation
  of my strategy"), and a depreciating moat as foundation models improve. The strongest use is
  refereeing *code-complete* candidates, where this risk disappears.
- **Claim-type routing is English keyword-based.** Claims are routed (descriptive-statistical,
  trading-strategy, provided-series-statistic, or structural) by English-text cues; non-English or
  unusually phrased claims fall through to the `trading_strategy` default. A provided-series claim (one
  that supplies its own pre-computed statistic series) is executed deterministically and, because its
  construction is supplied rather than reproduced, is capped at `watch` until reconstructed from
  primitives. That fail-open is conservative (tested as a strategy, not
  mis-specialized), never a crash.
- **Deflation is not magic on the first look.** On the first single-claim run in a family, DSR is
  effectively PSR; deflation engages as Penrose sees multiple trials, registered generator
  candidates, or populated partitions in the same search family.
- **Low-breadth detection floor:** marginal single-asset edges are genuinely unresolvable on short
  samples, hence the `underpowered` label rather than a false kill.
- **Look-ahead defense is layered.** Generated modules run in Docker and are checked for static leak
  idioms; the dynamic truncated-bundle check is strongest when it runs on the same execution path.
- **Fidelity verification can use an independent verifier, but defaults to the same provider.**
  Set `PENROSE_LLM_VERIFIER_MODEL` (and, for a genuinely independent endpoint,
  `PENROSE_LLM_VERIFIER_BASE_URL` / `PENROSE_LLM_VERIFIER_API_KEY`) to route the fidelity check to a
  different model/provider and reduce correlated implementation-and-judging errors; unset, it falls
  back to the same provider, and each result records whether the check was independent.
- **Holdout confirmation is gated and scarce.** It is single-use per claim, must pass the configured
  holdout evidence threshold, and production runs with modeled costs are capped at `watch` even when
  the statistical path is strong.
- **Generated hypotheses have no external evidentiary anchor.** Dream results are labeled
  `generated_hypothesis`, their fidelity is only LLM self-consistency, they never inspect the locked
  holdout during triage, and they are capped at `watch` until independently confirmed.
- This is a **research prototype.** The contribution is methodology and measurement, not a product,
  and not a source of trading advice.

See [ROADMAP.md](ROADMAP.md) for where this is going, and [AGENTS.md](AGENTS.md) if you are pointing a
coding agent at the repo.

## FAQ

**Does Penrose need an LLM to run?** Only to referee a real claim. The keyless core — `penrose eval`,
the calibration batteries, `connections`, and the worked example — runs with no API key and no network,
because it uses planted strategies with known code and has nothing to reconstruct. Refereeing an actual
claim uses a model for the reconstruction stages: extraction from prose, spec generation, module
implementation, and the fidelity check. The deterministic **provided-series** path is the exception — it
tests a supplied statistic series with no generated code.

**Is the pipeline's model the same as an agent that drives Penrose?** No — they are separate roles. The
pipeline's *internal* model (`PENROSE_LLM_DEFAULT_MODEL`, default `glm-5.2`) is called programmatically
inside the stages to reconstruct a claim. An *external* agent — via the CLI, a runner script, or the MCP
`penrose_run_claim` tool — operates Penrose from the outside: it submits claims and reads verdicts, and
never touches the reconstruction. They can be, and usually are, different models (an orchestrating agent
can be one model while the pipeline uses cheap `glm-5.2` internally), and you can drive Penrose entirely
by hand with the CLI and use no external agent at all.

**Does it need more than one model provider?** No. One OpenAI-compatible provider runs the whole
pipeline. A *second, independent* provider is optional and affects only one gate: point the fidelity
check at a different model (`PENROSE_LLM_VERIFIER_MODEL` / `_BASE_URL` / `_API_KEY`) so a reconstruction
is not graded by the model that wrote it. Unset, the check falls back to the default provider and each
verdict records whether it was independent.

**Which model should I use?** The default is `glm-5.2` — cheap enough to run the full pipeline
unattended and capable enough for the reconstruction and fidelity stages. Any OpenAI-compatible endpoint
works (see [Model configuration](#model-configuration)); use a stronger model when reconstruction
fidelity on difficult prose claims matters more than cost.

**Do I need a GPU or any special hardware?** No. Penrose orchestrates and evaluates; the model runs
wherever your endpoint is (a hosted API or a local server). Untrusted generated code runs in a Docker
sandbox, so Docker is required for the reconstruction path but no GPU is.


## Why is it called Penrose?

Penrose is named for the [Penrose process](https://en.wikipedia.org/wiki/Penrose_process), Roger
Penrose's mechanism for extracting energy from a rotating black hole. Almost everything that enters a black hole is
swallowed, but a rare fraction escapes the ergosphere carrying away *more* energy than it arrived with. The gauntlet of real conditions, deflation, costs, regimes, and a single-use locked holdout, is the black hole, and nearly every edge that enters is destroyed. What falls in is not wasted: it becomes the corpus of invalidations that Penrose mines for new candidates. The rare
claim that escapes falsification is the fraction that gets out enriched, the small part that survived the thing that kills almost everything. It is never guaranteed profit, only the candidate worth confirming.

## References

1. Bailey, D. H., and López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107. <https://ssrn.com/abstract=2460551>
2. Harvey, C. R., Liu, Y., and Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68. <https://doi.org/10.1093/rfs/hhv059>
3. Hou, K., Xue, C., and Zhang, L. (2020). Replicating Anomalies. *Review of Financial Studies*, 33(5), 2019-2133. <https://doi.org/10.1093/rfs/hhy131>
4. Chen, A. Y., and Zimmermann, T. (2022). Open Source Cross-Sectional Asset Pricing. *Critical Finance Review*, 11(2), 207-264. <https://doi.org/10.1561/104.00000112> Code/data: <https://github.com/OpenSourceAP/CrossSection>
5. McLean, R. D., and Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1), 5-32. <https://doi.org/10.1111/jofi.12365>
6. Politis, D. N., and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313. <https://doi.org/10.1080/01621459.1994.10476870>
7. Li, Y., Yang, X., Yang, X., Xu, M., Wang, X., Liu, W., and Bian, J. (2025). R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. <https://arxiv.org/abs/2505.15155> Code: <https://github.com/microsoft/RD-Agent>
8. Planton, J. (2026). AlphaSeeker: A Framework for Systematic Alpha-Seed Discovery from Tick Data. TFM Quantitative Trading Ltd, working paper. SSRN. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6823559>
9. Stephan, R. (2026). Sequential Tradeability Testing for Alpha Signals. SSRN working paper. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6922558> Replication: <https://github.com/ranystephan/sequential_tradeability>

## License

Apache-2.0 (see `LICENSE`). The framework is open; if you build on it, a citation to the paper is
appreciated.

## Contact

Bugs, questions, and ideas: please use **GitHub Issues / Discussions**, the primary channel.

To follow the project, **Star/Watch** the repository. For anything else, including interest in a hosted
referee or a team deployment down the line, email **hello@penrose.systems**. No product is for sale
today; this is a research release.

## A note from the author

Thank you for taking the time to look at Penrose. I built it because I wanted an honest, shared
reference for the quant community, a place where a claim has to survive real scrutiny before anyone
believes it, and where what fails is kept and learned from instead of quietly discarded. It is a
research prototype, not a finished product, and I would genuinely rather you try to break it than take
it on faith. If it helps you, or if you find where it is wrong, I would love to hear from you.

Charles Patterson
