# Penrose: A Reference System for Reproducible, Power-Aware Evaluation of Quantitative Performance Claims

**Preprint draft — not peer reviewed**

**Author:** Charles Patterson · Patterson Research  
**Contact:** charles.s.patterson@gmail.com  
**Date:** June 2026

---

## Abstract

Empirical claims about investment performance are difficult to evaluate consistently. The relevant evidence is often distributed across prose, code, data, cost assumptions, search histories, and selectively reported backtests. Even when accepted statistical tools are available, applying them to a third-party claim requires a substantial systems layer: source ingestion, grounded claim extraction, implementation reconstruction, data provenance, isolation of untrusted code, multiple-testing accounting, holdout control, robustness analysis, audit logging, and human authorization.

We present **Penrose**, an open research prototype and reference implementation for a broader evidence protocol for quantitative performance claims. Penrose accepts papers, manually submitted theses, code-complete strategies, and registered machine-generated hypotheses. It converts each source into claim-level evaluation records, routes claims to reviewed modules or constructs claim-scoped implementations, executes untrusted generated code in a constrained Docker sandbox, and subjects resulting return streams to a power-aware robustness stack. The system reports `kill`, `underpowered`, `watch`, and `research-supported` outcomes together with routing states such as `needs_data`, `pending_module`, and `cannot_replicate`. It separates statistical evidence from implementation fidelity, distinguishes missing evidence from negative evidence, preserves a single-use holdout, tracks the multiple-testing denominator of registered generator searches, and prevents unattended model output from becoming committed knowledge without human approval.

This paper focuses on the complete system rather than proposing the underlying evidence standard. We describe the architecture, trust boundaries, data contract, reconstruction and self-repair loop, robustness engine, local corpus, advisory connection discovery, researcher interfaces, and generator bridge. We also report calibration and application results produced by the accompanying experimental scripts: placebo and injected-signal controls, a five-null battery, native-breadth sensitivity analysis, evaluation of machine-generated factors, and analysis of published equity anomalies. The current implementation is a local single-user prototype. Costs and capacity are modeled rather than established from fills; prose-to-code fidelity remains a central threat; several data domains lack production-grade point-in-time adapters; generated modules do not yet receive every causality check available to trusted modules; and independent replication and fresh-data confirmation remain future work. Penrose should therefore be understood as inspectable evaluation infrastructure and a reference implementation, not as an oracle, publication authority, trading system, or substitute for peer review.

**Keywords:** backtest overfitting; deflated Sharpe ratio; multiple testing; reproducibility; research automation; quantitative trading; evaluation infrastructure

**JEL classification:** C12, C18, C52, C63, G11, G12

**Code and artifact availability.** The reference implementation is open source at https://github.com/PattersonResearch/Penrose. The reproducible core of this paper — the evaluation-invariant suite, the calibration controls, and the deterministic process-conditional worked example — is current as of the v0.2.0 release and was re-measured against it. The keyless core (`pip install -e .`) reproduces the evaluation-invariant suite (`python scripts/eval_suite.py`, 93/93), the unit tests (`python -m pytest`, 137 passed / 2 skipped in the public distribution), and the deterministic process-conditional worked example (`scripts/worked_example_process_conditional.py`) — none of which require an API key, network access, or external data. The application studies (the published-anomaly and machine-generated-factor referee runs) were prepared against the v0.1.0 release and additionally require external datasets and credentials documented in the release; they were not re-run under v0.2.0. The v0.2.0 changes only tighten verdicts (stricter, order-independent deflation plus added robustness gates), so those reported kills stand. The companion paper, *A Power-Aware Evidence Standard for Empirical Investment-Performance Claims* (FPES), specifies the implementation-neutral evidence protocol that this system operationalizes. The current public release is v0.4.1, which adds, on top of the v0.2.0 baseline above: an opt-in tail-risk (widow-maker) gate, an input-side data-granularity check, contrastive principles, an agent-readable principle surface, and a futures data adapter (v0.3.0); and a power-aware verdict taxonomy tested against a frozen realistic-edge floor with an enforcing Monte-Carlo calibration control, first-in-family anti-mining deflation and post-sample caps, honest `engine_error` routing, an edge-free offline fallback, non-destructive append-only decision logging, intraday and fundamentals data adapters, and an opt-in human-gated management surface (v0.4.x). These are not described in this paper, and the reproducible numbers above remain those measured against v0.2.0.

## 1. Introduction

The bottleneck in evaluating a quantitative performance claim is rarely one statistical formula. A reviewer must first determine what the paper actually claims, recover the decision rule, identify what information was available at each decision time, reconstruct the strategy, obtain appropriate data, specify costs, account for the search that produced the reported specification, and separate a failed strategy from a failed reconstruction. These tasks become more difficult when claims are generated at machine scale.

The financial-statistics literature supplies many of the required components. Probabilistic and deflated Sharpe measures account for sampling variation, non-normality, and selection over multiple trials [1]. Factor-zoo research demonstrates that the effective number of attempted specifications materially changes evidentiary thresholds [2]. Large replication studies show that many published anomalies weaken under standardized construction and stronger significance requirements [3,4]. Post-publication studies document substantial decay in reported predictability [5]. None of these results, however, automatically turns a PDF, repository, or generator output into an auditable evaluation.

Penrose addresses that systems problem. Its design target is a claim-level pipeline:

```text
source
  -> sanitized ingestion
  -> grounded claim extraction
  -> falsifiability and economic preflight
  -> trusted-module routing or claim-scoped reconstruction
  -> sandboxed execution where required
  -> power-aware statistical and economic evaluation
  -> provenance-rich report and review proposal
  -> human-authorized corpus entry
```

The system is designed as a reference implementation of an implementation-neutral evidence protocol. The associated standard should define what evidence a performance claim must disclose and how results should be reported. Penrose demonstrates one way to operationalize that standard. A paper need not use Penrose to satisfy the protocol, and Penrose's output does not determine publication, truth, or investment suitability.

This systems paper makes four contributions.

1. **An end-to-end claim-evaluation architecture.** Penrose connects source ingestion, grounded extraction, reconstruction, data contracts, isolated execution, robustness analysis, reporting, and human review in one inspectable workflow.
2. **Explicit trust and authorization boundaries.** Source text is treated as untrusted data; generated code is isolated; holdout access is controlled; generated hypotheses are treated as unanchored; and machine-produced conclusions remain proposals until a named human approves them.
3. **Power- and provenance-aware operational semantics.** Missing data, failed reconstruction, insufficient power, structural invalidity, and statistical weakness are represented as different states rather than collapsed into rejection.
4. **A registered interface to research generators.** Penrose freezes the complete emitted candidate cohort, records the declared search budget before testing, and charges discarded or blocked candidates to the multiple-testing denominator.

The contribution is integration and enforcement. Penrose does not introduce a new estimator, discover a new anomaly, or claim to be the first system to automate financial research review. It combines established statistical ideas with software controls needed to apply them reproducibly to third-party and machine-generated claims.

## 2. Scope and design goals

### 2.1 Evaluation target

The unit of evaluation is a falsifiable claim, not an entire paper. A source may produce multiple claims, each with a separate implementation, data requirement, verdict, and rationale. The present implementation is most appropriate for claims about:

- abnormal or risk-adjusted returns;
- predictive factors and signals;
- trading-strategy performance;
- portfolio improvements measured through return streams;
- machine-generated factors or strategies;
- event or market claims that can be translated into a terminal quantitative test.

The architecture can represent qualitative or non-return claims, but the current deterministic backtest engine is not a universal evaluator of causal corporate-finance research, structural estimation, surveys, theory, or management quality. Claims without measurable outcomes and horizons are routed away from full-rigor performance evaluation.

### 2.2 Non-goals

Penrose is not a broker, portfolio manager, execution service, signal vendor, or capital allocator. It places no orders and has no path to live capital. A positive result means that a claim survived the configured evaluation under disclosed assumptions. It does not mean that the effect will persist, that the strategy is suitable for an investor, or that the claim is true in every market and regime.

Penrose is also not intended to monopolize an evidence standard. The standard and the implementation should remain separable:

- the protocol specifies evidence requirements and report semantics;
- Penrose is one conforming implementation;
- alternative implementations should be able to produce comparable claim-level evidence records;
- governance of a community standard should not depend on Penrose's maintainers.

### 2.3 Design principles

The implementation follows seven principles.

**Fail visibly.** Missing data, unavailable Docker, malformed module output, extraction failure, and fidelity uncertainty produce explicit states. The system avoids silently substituting success, failure, or a fabricated series.

**Ground claims in sources.** An externally extracted claim must contain a source span that occurs in the ingested text. A metric presented as verbatim must also occur in the source.

**Separate reconstruction from evaluation.** A strategy may fail because the claim is weak, because the module is unfaithful, or because required data are unavailable. These are different conclusions.

**Protect the holdout.** Kill gates operate on the in-sample and ordinary out-of-sample windows. The final holdout is consulted only for an otherwise strong external claim and is single-use in the production workflow.

**Charge search degrees of freedom.** Candidate count, family scope, regime partitions, and preregistered generator cohorts affect the deflation denominator.

**Attach power to negative results.** A non-result on a sample unable to resolve a realistic effect is `underpowered`, not evidence that the effect is absent.

**Require human authorization for durable knowledge.** Models and backtests may create review proposals. Only the review path can construct the write-capable corpus client.

## 3. System architecture

### 3.1 Local-first deployment

Penrose 0.2 is a Python 3.9+ local application packaged through `pyproject.toml`. Its core dependencies are NumPy, pandas, SciPy, PyArrow, pypdf, and PyYAML. Matplotlib, Databento, Docker, and LLM access are feature-specific dependencies. The knowledge store is a native SQLite component of the standard library and requires no external service, runtime, or network. The installed `penrose` command exposes paper evaluation, verdict inspection, data-blocker inspection, status, calibration, planted-strategy evaluation, and registered hypothesis generation.

The current system is optimized for a single researcher operating a local workspace. State is stored in files:

- source archives and dream-run artifacts;
- run and decision JSONL logs;
- a review queue;
- per-source Markdown reports;
- a machine-readable analysis index;
- a multiple-testing ledger;
- data-request records;
- dashboard state and advisory connection records;
- a native, isolated SQLite knowledge store (atoms and typed edges).

This choice makes the prototype inspectable and easy to reproduce, but it is not a multi-user transactional service. Concurrent writes are guarded only where the implementation specifically uses file locks, notably the trial ledger and dream-run summary.

### 3.2 Pipeline stages

`run_source()` is the primary orchestrator. It executes:

| Stage | Responsibility | Principal output |
|---|---|---|
| P1 | Parse, hash, archive, and flag untrusted source content | `IngestedSource` |
| relevance | Skip clearly off-domain sources before expensive processing; fail open | relevance record |
| P2 | Extract falsifiable claims with source spans | `Claim[]` |
| P3 | Classify falsifiability | route or `unfalsifiable` |
| P4 | Apply a deterministic cost-floor precheck | pass or cost failure |
| P5 | Detect exact or semantic duplicates against prior committed material | novelty record |
| P6 | Route to a reviewed module or generate a claim-scoped module specification and implementation | executable module or routing state |
| P7 | Run the return stream through the robustness and power engine | metrics bundle |
| P8 | Produce a claim-level verdict and rationale | `Decision` |
| P9 | Require named human approval before corpus promotion | committed atom and typed edges |

The data bundle is fetched once per source and shared across its claims. Each claim nevertheless receives independent access tracking, routing, metrics, verdict logic, charting, and review records.

### 3.3 Trusted and untrusted execution paths

Penrose maintains two module classes.

**Trusted operator modules** live under `modules/<module_id>/impl.py`, declare stable routing metadata, and may execute in the Penrose process. The repository currently contains reviewed implementations for a macro-signal/BTC-volatility translation and crypto perpetual funding carry. Trusted modules can declare a primary strategy class and aliases. Alias collisions are surfaced and resolved by first registration rather than silently depending on directory iteration order.

**Auto-generated modules** live on a quarantined `_auto` shelf, are bound to a specific claim identifier, are marked `__auto_generated__`, and are excluded from reusable routing vocabulary. They are never silently promoted into the trusted registry and never execute in the host process. The host reads their metadata through Python's abstract syntax tree rather than importing them.

This split is important. A module generated to approximate one paper is not automatically a reusable implementation of every claim with a similar label.

## 4. Ingestion and grounded claim extraction

### 4.1 Source formats and archival record

P1 accepts PDF, plain text, and Markdown. PDF text is extracted with pypdf. Parse failures return a sanitized empty source with an explicit error flag rather than terminating the entire run. The source record includes:

- source identifier and inferred title;
- page and character counts;
- a truncated SHA-256 digest of extracted text;
- prompt-injection pattern flags;
- an archive metadata record.

The current archive writes metadata and derived source artifacts. It does not consistently copy the original external PDF into the archive in the examined code path; therefore the source file supplied to the run remains part of the reproducibility package.

### 4.2 Source text as untrusted data

The ingester detects phrases resembling prompt injection, including instructions to ignore prior prompts, adopt an assistant role, or approve a result. Detected text is retained because provenance requires exact quotations. “Sanitization” therefore means contextual isolation and flagging, not destructive redaction. Downstream roles receive source material as quoted user data rather than as system instructions.

This is a useful control but not a formal proof of prompt-injection resistance. The system still depends on model behavior, truncates paper text to 24,000 characters for the primary extraction role, and does not currently use a full-document deterministic parser.

### 4.3 Relevance screening

When model access is available, a low-cost relevance screen checks whether at least one claim appears testable against the implemented or plausibly collectable data domains. The current declared envelope emphasizes crypto spot, perpetual funding, crypto volatility, prediction-market series, and weather. The screen fails open: errors or malformed model output allow the source to proceed.

### 4.4 Claim extraction

The P2 role extracts directional, measurable claims rather than motivations or literature summaries. Every accepted external claim must provide:

- statement;
- mechanism;
- scope;
- horizon;
- verbatim source span;
- optional verbatim claimed metric;
- proposed strategy class.

Whitespace-normalized containment checks verify that the source span occurs in the ingested text. The same check applies to a claimed metric quotation. Claims without grounded spans are dropped.

The extractor receives a bounded vocabulary of trusted module classes. It may reuse a class only when the claim clearly fits; otherwise it proposes a new concise class. This improves routing while avoiding an unbounded prompt containing one-off generated modules.

The offline fallback is intentionally explicit. Without a configured model, Penrose attempts to load a hand-authored claim file and marks the provenance as a manual fallback. If no such file exists, extraction returns no claims rather than pretending that extraction occurred.

### 4.5 Falsifiability classification

P3 classifies a claim as deterministic-testable, generated-module-testable, qualitative-only, or unfalsifiable. Claims without a measurable outcome and horizon are rejected from the quantitative path. In offline mode the current stub assumes testability and marks that assumption; this is a development convenience, not equivalent evidence.

## 5. Reconstruction, self-repair, and fidelity

### 5.1 Module specification

When no trusted module is authorized for a claim, Penrose creates a versioned YAML `ModuleSpec` containing:

- a module identifier and strategy class;
- a tradeable translation of the claim;
- required data series;
- signal and position logic;
- a kill criterion;
- expected data coverage and frequency;
- unresolved assumptions;
- implementation notes.

Specifications are retained even when implementation fails. This creates an auditable boundary between what the paper said, how the evaluator translated it, and what code was eventually run.

### 5.2 Claim-scoped code generation

The module implementer receives only a machine-readable catalog of available bundle keys and a restricted contract:

```python
run(bundle, claim, cost_frac) -> {
    "ok": True,
    "net": pandas.Series,
    "positions": pandas.Series,
    "bars_per_year": float,
    ...
}
```

Optional outputs activate additional tests:

- `payoff` and `position_signed` enable the permutation test;
- `wf_frame` enables the implemented volatility walk-forward routine;
- `prices` or `regime_schemes` can supply additional point-in-time regime partitions.

If required data are absent, the module must return a structured `data_unavailable` result. It is instructed not to invent data or access undeclared resources.

As of v0.2.0, module generation is faithful to claim type: claims are routed by type — descriptive, trading-strategy, or structural — so that a descriptive claim is implemented as a statistic test rather than forced into a trading backtest, and only claims that are genuinely about a tradeable rule are evaluated as one. An optional independent fidelity verifier, routed to a second model provider, can additionally check that the generated module matches the claim type and proposition (see §5.8).

### 5.3 Static checks

Before execution, generated code is parsed and checked for:

- imports outside an allowlist containing NumPy, pandas, `math`, and `__future__`;
- direct file opening, dynamic evaluation, compilation, or pickle operations;
- common negative-shift and forward-index look-ahead patterns;
- required module metadata and `run()` definition.

These checks are defense in depth, not the security boundary. They are necessarily incomplete against adversarial Python.

### 5.4 Sandboxed execution

The categorical boundary is Docker. If Docker is unavailable, automatic implementation is disabled; Penrose does not fall back to unsandboxed execution.

The sandbox uses:

- no network;
- a read-only container root;
- a temporary filesystem for `/tmp`;
- a non-root user;
- one CPU;
- 512 MB memory;
- a process limit;
- a wall-clock timeout;
- a read-only bind mount containing the generated module;
- a temporary work mount containing only serialized data, claim metadata, and outputs;
- no application secrets.

Input and output interchange uses Parquet and JSON. The parent does not deserialize pickle from the untrusted process. The child receives a minimal shim implementing only the bundle access needed by the module contract.

The container image is built locally from Python 3.11 slim with NumPy, pandas, and PyArrow. Building the image may require network access once; running a module does not.

### 5.5 Runtime contract validation

After the sandbox run, the host validates the returned artifacts. A successful result must have:

- finite `net` and `positions` Series;
- a `DatetimeIndex`;
- identical indexes and lengths;
- at least ten trades for implementation validation;
- a finite and plausible `bars_per_year`;
- no near-constant nonzero return stream that would imply artificial infinite Sharpe;
- no annualized Sharpe above the implementation sanity cap;
- well-formed optional payoff, signed-position, and walk-forward carriers.

The host also checks that `bars_per_year` is broadly consistent with the calendar span implied by the return index. A clean `data_unavailable` result is contract-valid and routes to a blocker rather than a kill.

### 5.6 Self-repair

Code generation is attempted up to three times. Validation errors and the previous code are fed back to the implementer. The loop can repair missing keys, malformed shapes, unavailable-series handling, unsafe imports, and contract violations. If all attempts fail, Penrose writes a non-executable rejected stub and leaves the claim in `pending_module`.

This is reconstruction assistance, not proof of semantic correctness. The loop optimizes for contract conformance and executable behavior; it cannot establish that the economic interpretation is faithful.

### 5.7 Causality checks

For trusted in-process modules, validation can rerun the module on a bundle truncated to the first 70% of each series. If return values on overlapping early dates change after later data are removed, the module is rejected for look-ahead.

The auto-generated sandbox path performs this second truncated-bundle container run after the full sandbox run and compares overlapping early net values. Static look-ahead heuristics remain a first line of defense for obvious negative shifts and future indexing, but the dynamic comparison is the production causality check for generated modules.

### 5.8 Fidelity refuter

Statistical validity cannot detect whether a module tests the wrong proposition. Penrose therefore includes an adversarial model role that compares the claim, module code, and specification. The refuter searches for wrong signals, directions, horizons, instruments, constant positions, proxy substitutions, and look-ahead.

The refuter defaults to unverified. Errors are inconclusive rather than faithful. An unverified module cannot authorize trusted-module reuse or the strongest positive verdict. A high-confidence finding of unfaithfulness changes the outcome to `cannot_replicate`, records the original statistical verdict, and excludes the result from principle formation.

This control remains model-mediated and nondeterministic. As of v0.2.0 the verifier role can be
routed to an independent provider with `PENROSE_LLM_VERIFIER_BASE_URL`, `PENROSE_LLM_VERIFIER_API_KEY`,
and `PENROSE_LLM_VERIFIER_MODEL`; independence is taken to hold when either the provider endpoint or
the model differs from the implementer's. When all are unset it defaults to the same model family as
the implementer, which leaves correlated blind spots. The strongest future fidelity gate is
independent reproduction of the paper's reported statistic before evaluating novel holdout
performance.

## 6. Data handling and provenance

### 6.1 Data contract

The F7a contract exposes `Series`, `Unavailable`, and `DataBundle`. Every available series carries:

- a logical name;
- a timestamp-indexed pandas Series;
- a provenance string;
- a unit;
- coverage metadata and notes.

Unavailable data carry a reason. The bundle never uses a missing value to imply that a source exists.

`DataBundle.get()` supports exact keys and conservative normalized aliases to tolerate naming drift such as `price.eth_usd_spot_daily` versus `eth_spot_daily`. Alias normalization removes filler tokens but does not fuzzy-match across distinct asset identifiers.

### 6.2 Implemented sources

The examined implementation can assemble:

- BTC daily close from Coinbase or Kraken public endpoints;
- BTC implied volatility from Deribit DVOL when sufficient history is available;
- Kalshi-derived macro signals from a sibling local data catalog when present;
- crypto spot, perpetual funding, and weather series exposed by that catalog;
- user-configured Databento daily series with local request-keyed caching;
- synthetic macro and implied-volatility fallbacks, explicitly labeled synthetic;
- point-in-time BTC volatility and trend regime labels derived from trailing price history.

The catalog is optional and external to the core package. Databento is bring-your-own-key and entitlement dependent. Network or adapter failures fail open into absent series or labeled synthetic fallbacks rather than terminating a paper run.

### 6.3 Synthetic data semantics

Synthetic fallback data are intended to exercise the pipeline, not support empirical claims. Verdicts record whether a module consumed synthetic data. Trusted in-process modules are tracked at the series-access level. Because the sandbox shim does not return an access log, auto-generated modules currently use the conservative bundle-level synthetic flag.

Any result involving synthetic inputs is provisional. The system's current report also displays a source-level warning if any bundle series is synthetic, which can be more conservative than the per-verdict access record.

### 6.4 Point-in-time regimes

Penrose derives volatility and trend labels from trailing BTC price history. Modules may condition on these preregistered labels instead of fitting a regime detector inside the strategy. The same labels can be used by the robustness engine to test whether returns are concentrated in one state.

This arrangement prevents a generated module from freely optimizing a regime boundary, but it does not make the labels universally appropriate. They are BTC-derived and should not be presented as generic regimes for unrelated assets without additional adapters.

### 6.5 Data blockers

When a module lacks data, Penrose writes a structured request with the claim, strategy class, missing series, source, and raw reason. The decision is `needs_data`, and the claim can be rerun when the catalog changes. Data absence is never treated as falsification.

## 7. Robustness and power engine

### 7.1 Time partitions

The current harness uses an ordered 50/30/20 split:

- 50% initial sample;
- 30% ordinary out-of-sample sample;
- 20% locked final holdout.

All kill gates operate on the first 80%, or specifically the 30% OOS segment. The final 20% is excluded from three-fold consistency, regime analysis, capacity estimation, bootstrap evaluation, permutation testing, and walk-forward evaluation.

### 7.2 Sharpe statistics and multiple testing

The engine computes conventional annualized Sharpe, Probabilistic Sharpe Ratio, and Deflated Sharpe Ratio. The implementation corrects for sample skewness and kurtosis through PSR and, when there are multiple trials with nonzero cross-trial Sharpe variance, compares the observed per-trade Sharpe against a best-of-\(N\) benchmark [1]. On the first isolated run of a family, deflation is inert and the score is effectively PSR plus the rest of the robustness stack; deflation strengthens as the family or generator ledger accumulates observed trials. The falsification-gate battery described in Section 8 treats this PSR/DSR path as a certification statistic rather than as a tradeability score.

Trials are scoped to a family composed of strategy class and coarse data domain for external claims. Generated searches use stable domain-level families so a model cannot reset its denominator by inventing new strategy-class names.

The ledger:

- deduplicates strategies by identifier;
- records family, generation source, cohort, denominator, Sharpe, DSR, and sample size;
- uses file locking and atomic replacement;
- preregisters all emitted generator candidates before any candidate reaches testing;
- counts duplicates, conceptual candidates, data blockers, and failed implementations through the cohort denominator;
- adds populated regime partitions as additional looks.

As of v0.2.0 the DSR multiple-testing denominator is pre-registered as a per-family cohort before evaluation rather than accumulated from a running trial count during the run. This is an order-independent correctness fix: the deflation a claim faces no longer depends on the order in which siblings happened to be evaluated. The change only tightens — it removes a path by which an early-evaluated claim could be deflated against a smaller denominator than its later-evaluated family members.

Family scoping avoids making an unrelated weather claim pay for every crypto experiment. It also introduces a policy choice: the correct family is not identifiable from code alone. Reports should therefore disclose the chosen scope and, for research use, include sensitivity to narrower and wider denominators.

### 7.3 Three-fold sign stability

The non-holdout sample is divided into three contiguous folds. All three must have positive Sharpe. This detects effects concentrated in one temporal segment and is treated as a structural failure rather than a low-power null.

As of v0.2.0 the engine adds combinatorial purged cross-validation (CPCV, López de Prado) as an independent robustness axis: the non-holdout sample is split into multiple groups whose train/test combinations are evaluated with purging and embargoing between adjacent observations, so an apparent edge must survive across many recombined train/test partitions rather than a single chronological fold ordering.

### 7.4 Stationary block bootstrap

The OOS return stream is resampled with a Politis–Romano-style stationary block bootstrap [7]. The output includes:

- confidence intervals for mean edge and Sharpe;
- empirical probabilities of positive edge and Sharpe;
- median and 95th-percentile drawdown;
- an indicator that the edge interval includes zero.

The bootstrap is deterministic for a configured seed. If a borderline survivor's edge interval includes zero, the verdict may become a kill or, when the sample lacks adequate detection power, `underpowered`. Section 8 adds calibration nulls that ask whether this bootstrap-assisted path certifies persistent but drift-destroyed or exactly dead processes.

### 7.5 Permutation test

When the module supplies signed positions and raw payoff, the engine shuffles payoff relative to position and tests whether observed signal-payoff alignment exceeds the randomized null. The cost term is intentionally excluded because an unchanged cost cancels across permutations. Profitability after costs remains the responsibility of net returns, DSR, and the bootstrap.

### 7.6 Walk-forward evaluation

The implemented walk-forward routine repeatedly refits signal standardization for volatility strategies over anchored or rolling training windows and reports per-window and aggregate OOS Sharpe. Inconsistent windows can trigger `walk_forward_drift`.

This routine is presently specialized to a carrier with `signal`, future realized volatility, and implied volatility columns. Penrose does not yet provide a fully generic estimator-refit protocol for every model class.

### 7.7 Regime kill lens

The regime lens partitions the non-holdout stream by exogenous calendar buckets and optional point-in-time market labels:

- weekday versus weekend;
- day of week;
- trading session for intraday indexes;
- preregistered volatility and trend states when available.

It then drops the single best bucket. If less than 25% of the overall per-trade edge remains, the strategy is marked regime-fragile. The lens does not select a preferred regime for trading; it is a falsification test. Each populated bucket increases the trial count.

As of v0.2.0 a claim may also pre-register a declared regime scope: rather than being penalized for not working in every state, a claim can declare the regime within which it is asserted to hold and be tested inside that declared scope. This is adherence-gated — the declaration is honored only when the strategy actually confines its activity to the declared regime, so a claim cannot quietly trade outside its stated scope while still receiving the narrower test.

One known issue remains in the repository roadmap: timezone mismatch between a naive trade index and UTC regime labels can silently prevent some optional regime partitions from being applied. Calendar partitions remain available, but this degradation should be fixed and tested before relying on all advertised market-state cuts.

### 7.8 Costs and capacity

Modules charge a configured fractional trading cost. The P4 preflight also computes the configured prediction-market fee curve. The harness estimates capacity using a linear impact coefficient and annualized turnover, and bootstraps a capacity interval conditional on positive edge.

These are explicit models, not measured execution. The configuration's cost provenance is currently `modeled`. Consequently, a result that would otherwise be `research-supported` is capped at `watch` until costs and capacity are supported by measured fills. The RD-Agent and Chen–Zimmermann experiment scripts override this cap to study the statistical path; those experimental overrides must not be confused with the production system's evidentiary state.

### 7.9 Power and minimum detectable effect

Every backtested decision reports an approximate minimum detectable per-bar effect:

\[
\mathrm{MDE} \approx \frac{z_{\mathrm{certify}}}{\sqrt{n_{\mathrm{OOS}}}}.
\]

The implementation also annualizes this value using the reported bar frequency. If a claim fails only through a low deflated score or a bootstrap interval spanning zero, and the computed MDE exceeds the configured realistic effect floor, Penrose reclassifies the result as `underpowered`.

Structural failures remain kills regardless of this calculation:

- in-sample-only sign instability;
- regime fragility;
- walk-forward drift;
- absence of signal-payoff alignment;
- explicitly negative DSR behavior.

The MDE is an operational approximation, not a complete prospective power calculation. It does not model every form of serial dependence, cross-sectional breadth, estimator uncertainty, or nonlinear payoff. Its value is semantic discipline: the report states what the sample could plausibly resolve.

## 8. The falsification gate battery

Penrose's verdict is produced by a battery of falsification gates rather than by a single score. The system uses established statistical controls as refutation instruments: DSR for selection-aware evidence [1], family and generator denominators for multiple testing [2], stationary bootstrap intervals for small-sample uncertainty [7], persistence-matched and dead-state nulls for sequential calibration [9], and corpus-neighbor context as an interpretability advisory [8]. The gates are deliberately asymmetric. A gate may block or qualify a claim, but no gate asserts that a claim is economically exploitable.

**P1-P9 stack.** The existing pipeline gates enforce the non-statistical evidence boundary. P1 treats source text as untrusted data; P2 requires grounded claim spans; P3 refuses non-falsifiable claims; P4 applies a deterministic fee preflight; P5 rejects exact or semantic duplicates; P6 separates reviewed modules from claim-scoped generated implementations; P7 runs the statistical and economic robustness engine; P8 emits the claim-level verdict; and P9 is the only write-capable human authorization path. The invariant is provenance separation: a claim cannot become durable knowledge merely because a model, source, or backtest produced favorable text.

**PSR/DSR and search-denominator gate.** The DSR path discriminates a true OOS return stream from a selected winner under a disclosed number of trials [1] once Penrose has actually seen more than one trial with nonzero cross-trial variance. For a single isolated family run, the score is PSR and the other robustness gates carry the evidence burden. As the family ledger grows, the null becomes the best Sharpe expected after searching a family of specifications; its alternative is an OOS stream that remains exceptional after deflation. The invariant is that generator cohorts, repeated trials, and populated regime partitions increase the denominator instead of being hidden after selection.

**Multiple-testing family gate.** Family scoping applies the factor-zoo lesson that the relevant hypothesis count is part of the evidence [2]. Its null is not one isolated backtest, but the search process that generated the tested specification. Generated searches therefore preregister all emitted candidates, including duplicates and candidates blocked before P7. The invariant is denominator persistence: renaming a generated strategy class cannot reset the family.

**Bootstrap uncertainty gate.** The stationary block bootstrap [7] discriminates a point estimate from a return stream whose OOS mean edge is distinguishable from zero under dependence-preserving resampling. The null is a borderline stream whose confidence interval includes zero; the alternative is a stream whose edge remains positive across the configured resamples. The invariant is small-sample honesty: when the interval includes zero, a survivor cannot graduate merely because the analytic Sharpe statistic is flattering.

**Power gate.** The power-aware reclassification discriminates negative evidence from inadequate resolution. Its null is a non-structural failure on a sample whose minimum detectable effect exceeds the configured realistic IC floor; its alternative is a sample large enough to resolve that effect. The invariant is semantic precision: low DSR or a zero-crossing bootstrap interval on an underpowered sample becomes `underpowered`, while structural failures such as sign instability, regime fragility, walk-forward drift, and signal-payoff misalignment remain `kill`.

**G1: persistence-matched null.** The persistence calibration uses real-return or explicitly tagged calibration-return series, destroys drift by per-period or block sign flips, and trades the sign of the previous return. The per-period null destroys both drift and persistence; the block null destroys long-run drift while preserving within-block direction. Following the sequential-testing concern in Stephan [9], the output is not a new verdict label but an activation-gap table: how often block-preserved noise reaches `watch+` relative to per-period noise. The invariant is that neither null may reach `research-supported`, and the per-period full-noise control must not reach `watch+` in the eval invariant.

**G2: power-to-resolution annotation.** Underpowered verdicts now carry a structured resolution estimate: needed OOS bars, needed cross-sectional breadth at the configured realistic IC floor, current MDE IC, and the basis of the approximation. The null it discriminates is a dead-end `underpowered` label with no operational next step; the alternative is a non-result whose resolution requirement is visible. The invariant is additivity: the annotation changes no verdict label and is `None` for adequately powered results.

**G3: explicit dead-state null.** The dead-state calibration adds a zero-mean persistent AR(1) return process. This is an operational stand-in for Stephan's three-state dead/alive concern [9]: a detector that cannot represent exactly dead signals may let noise wander across an activation boundary. Penrose does not implement a Bayesian dead-state prior, but its `kill`/`underpowered` split and this dead-but-persistent null verify that a genuinely zero-drift process is not sent to `watch` or `research-supported`. The invariant is `CALIB-DEAD-1`: dead but persistent processes must remain `kill`, `underpowered`, or `insufficient_data`.

**G4: cost-sensitivity surface.** Cost sensitivity reports the round-trip cost at which a survivor would flip to failure by DSR or bootstrap edge interval. This addresses the sequential-tradeability observation that admission can swing as modeled costs vary [9], while keeping Penrose's role as referee rather than payoff optimizer. The invariant is default additivity: the breakeven field is reported, and an optional stricter cost gate is disabled by default so configured-cost verdicts remain unchanged.

**G5: corpus-isolation advisory.** The corpus advisory queries committed prior atoms for neighboring claims and recurring mechanism families. It follows the interpretability intuition in Planton [8] that isolated results without related candidates or plausible mechanisms deserve lower prior confidence. Penrose keeps this advisory out of calibration: an empty corpus returns `isolation_score=None`, a populated corpus reports neighbors and mechanism-family presence, and neither path changes the verdict. The invariant is corpus independence: accumulated local knowledge may contextualize a claim, but it cannot make the same return stream pass or fail differently.

Together these gates make the verdict battery falsification-oriented. A positive label means that the claim survived a disclosed sequence of refutations under current data, costs, power, and provenance; it is not a claim of profitability, deployment readiness, or truth.

## 9. Verdicts and evidence states

The implemented system uses the following principal outcomes.

| Outcome | Meaning |
|---|---|
| `kill` | A tested claim failed a configured gate with adequate evidence or a structural failure |
| `underpowered` | The design did not establish an edge, but the sample could not resolve the configured realistic effect |
| `watch` | The claim remains potentially interesting but lacks the strongest confirmation or is capped by trust conditions |
| `research-supported` | The statistical path and final holdout passed under the configured assumptions |
| `needs_data` | Required inputs are unavailable; the claim was not falsified |
| `pending_module` | No valid implementation is available; a specification awaits implementation or review |
| `cannot_replicate` | The implementation is judged unfaithful, so its statistical result is not attributed to the claim |
| `insufficient_data` | The return stream is too short for the configured minimum |
| `off_domain` | The source was screened as outside the current test envelope |

Positive verdicts are further constrained:

- modeled costs cap the production result at `watch`;
- generated hypotheses and chat submissions are unanchored and capped at `watch`;
- unverified fidelity withholds the strongest verdict;
- synthetic inputs make results provisional;
- dream triage cannot access the final holdout.

Current limitations that matter for interpreting positive evidence:

- deflation is inert on the first single-claim run of a family and engages only as Penrose observes multiple trials or registered search breadth;
- the strongest look-ahead defense is a dynamic truncated-bundle comparison, while static heuristics remain a fallback for idioms the dynamic path cannot execute;
- the fidelity verifier can be routed to an independent provider through `PENROSE_LLM_VERIFIER_BASE_URL`/`_API_KEY`/`_MODEL`, but the default remains the same model family as implementation;
- holdout confirmation is gated and, under modeled costs, cannot move the production label above `watch`.

The label `research-supported` should be read as a system state, not a certification of truth. A community-facing standard would benefit from implementation-neutral evidence badges such as reproduced, statistically supported, economically supported, prospectively validated, and independently replicated. Penrose's internal states can map to such badges, but the present code does not yet implement that complete standards-layer vocabulary.

## 10. Holdout and generator controls

### 10.1 Single-use production holdout

The first qualifying external strategy atomically claims a lock file keyed to the distinct claim identity. A second attempt for the same claim is refused, while a different claim receives its own one-time consultation. If the holdout is too small or evaluation fails before completion, the lock is released; otherwise the file records the strategy, number of bars, holdout Sharpe, and holdout PSR.

The lock is local to the Penrose instance and keyed by claim identity, not a sophisticated per-dataset or rotating holdout service. Experiment scripts use isolated temporary locks and may force evaluation for repeated calibration. Those scripts are research harnesses, not the production holdout policy.

### 10.2 Registered native hypothesis generation

`penrose dream` is a source adapter built into Penrose. Before generation it writes a manifest containing:

- safe run identifier;
- model;
- corpus snapshot hash;
- declared generation budget;
- registration timestamp;
- artifact directory.

It then stores the evidence packet and capability manifest, calls the generator for exactly \(N\) candidates, and writes the complete raw output immutably. Normalization preserves duplicate and inadmissible candidates. Before testing, every emitted candidate is registered in the trial ledger with a cohort denominator equal to the larger of the declared budget and actual emitted count.

Only unique candidates labeled `testable_now` or `testable_with_new_module` proceed. Others remain archived and counted. Eligible candidates become canonical `Claim` objects and enter P3 through P8.

During dream triage:

- `PENROSE_HOLDOUT_MODE=readonly` prevents any holdout access, even through a forced call;
- the source type is `generated_hypothesis`;
- positive results are capped at `watch`;
- the corpus packet excludes synthetic, fidelity-suspect, pending-module, and cannot-replicate records;
- generated artifacts and run summaries are idempotent by run identifier.

This closes an important local selection-bias channel: the generator cannot report only its preferred candidates without the discarded population affecting deflation. It does not solve opaque upstream search. A third-party generator can still submit \(K\) finalists after privately trying an unknown \(N\). Penrose needs a disclosed or conservatively bounded upstream denominator to correct that process.

### 10.3 Generator bridge

The repository includes an RD-Agent-style bridge for cross-sectional factor matrices. It forms dollar-neutral, unit-gross, cross-sectional z-weighted portfolios and evaluates causal forward returns. A prepass loads all factor identifiers into one family, after which every factor is judged against the full search.

The bridge is a script-level integration rather than a stable package API. The real-run adapter assumes external RD-Agent workspace paths and data formats. General connectors for other generators, signed manifests, and schema-versioned handoffs remain future work.

## 11. Corpus, provenance, and human authorization

### 11.1 Flat-file corpus

The system stores reports, source archives, decisions, analysis records, review proposals, data requests, and connection summaries locally as flat files. These artifacts are the canonical record and support dashboard rendering, dream evidence packets, and audit; the knowledge store (Section 11.2) indexes them and is rebuildable from them.

Each decision records the claim, verdict, kill reason, rationale, metrics, and timestamps. Backtested claims also receive an equity-curve chart when Matplotlib is available and a compact row in the analysis index.

### 11.2 Native knowledge store

A native, in-repo SQLite store holds atoms and typed edges under Penrose-specific slugs, with no external service, runtime, or network dependency. It seeds its index from the flat-file records of Section 11.1 and is rebuildable from them, so a fresh installation reproduces the knowledge layer without migration. `BrainReader` exposes only get, search, graph, and list operations. `PromotionClient` inherits those reads but requires a nonempty approver identity and is the sole path that exposes writes. Retrieval is lexical by default and embedding-based when an optional in-process embedder is installed.

The code-level authorization boundary is structural: the orchestrator never constructs `PromotionClient`. Only the P9 approval command does.

### 11.3 Review queue

Decisions, principles, and module specifications enter an Action Required queue. A human can inspect, approve, or reject them through the CLI. Approval writes the atom and provenance edges; rejection records a reason and leaves the proposal uncommitted.

The dashboard displays the queue but does not currently perform approval. This avoids turning a browser action into an implicit knowledge commit.

### 11.4 Principle and connection discovery

The pipeline contains a conservative same-reason principle proposal requiring at least three eligible kills. Separately, `brain_connect.py` performs deterministic advisory analysis over the flat-file verdict corpus:

- clusters structural kills by domain and reason;
- identifies shared failure modes across domains;
- proposes age-decayed principles;
- creates text-similarity links between claims.

Underpowered and fidelity-suspect outcomes are excluded from structural principles. The output explicitly informs but never gates future verdicts. The connection module does not import the verdict pipeline, reducing the risk of accidental feedback.

This is corpus analysis, not independent replication. Similarity uses lightweight text matching rather than a validated semantic model, and domain classification is keyword based.

## 12. Researcher interfaces and workflows

### 12.1 Command line

The CLI supports:

```text
penrose run --paper <path> [--no-llm]
penrose verdicts -n <count>
penrose data-requests
penrose status
penrose eval
penrose calibrate placebo|injection
penrose dream -n <budget> [--generate-only] [--run-id <id>]
```

Make targets expose the full calibration battery, literature and generator scripts, connection discovery, dashboard, review flow, and reset/archive utilities.

### 12.2 Reports

The Markdown report pairs favorable metrics with deflating context: DSR with trial count, OOS Sharpe with capacity, and edge with cost. It lists data provenance and warns about synthetic inputs. Reports state that outputs are proposals awaiting review.

### 12.3 Dashboard

The local dashboard is a static HTML application served by a standard-library HTTP server. It reads project state and overlays it onto placeholder-safe views for:

- Home;
- Action Required;
- Reports;
- Connections;
- Data Sources.

The server refreshes cached HTML and retains the last good render if one input source fails. It exposes assembled live state and stage progress.

The dashboard includes narrowly scoped write endpoints:

- preflight a draft thesis;
- submit a text thesis to the inbox;
- upload a validated PDF to the inbox.

These endpoints cannot approve proposals or write corpus state. They validate same-origin localhost requests and require a random per-launch token stored with mode 0600. The preflight's current falsifiability check uses an offline stub for speed, so it should be understood as interface guidance rather than the same classifier used in a model-enabled full run.

### 12.4 Operational workflow

A typical paper workflow is:

1. submit a PDF, Markdown file, or text thesis;
2. run P1–P8;
3. inspect claim spans, module specifications, data provenance, and verdicts;
4. resolve `needs_data` and `pending_module` states;
5. review fidelity findings and implementation code;
6. approve or reject proposals at P9;
7. rerun advisory connection discovery as the corpus grows.

A generated-search workflow begins with a registered budget, immutable candidate archive, and holdout-free triage. Graduation beyond `watch` requires a separate future-data or independent confirmation process that is not yet implemented.

## 13. Calibration and empirical applications

The experimental scripts are included with the repository, but not every reported result is produced by the default unit-test suite or by a keyless clean install. The numbers below should therefore be treated as application results tied to the archived data and external setups described by the scripts. The exception is the process-conditional worked example (Section 13.8), which is deterministic and runs on a clean keyless install; we recommend it as the first artifact a reader reproduces.

### 13.1 Planted-strategy evaluation

`scripts/eval_suite.py` supplies strategies with known qualitative behavior to test system invariants and discrimination. Repository records report 93/93 evaluation invariants passing (June 2026), together with 137 passing pytest tests (2 skipped, requiring optional dependencies) in the public distribution. These checks cover pipeline and security invariants as well as statistical behavior; they are not 93 independent scientific experiments.

### 13.2 Placebo specificity

The placebo script applies independent AR(1) signals to real BTC returns. The project reports 0/100 placebo signals reaching the strongest positive state. A multiple-testing variant registers a mined family so that trial count and observed cross-trial variance affect DSR.

### 13.3 Injected-signal sensitivity

The injection scripts add known signal-return dependence to real return streams and sweep information coefficient. The expected transition is kill or underpowered at low signal strength, watch in an intermediate region, and strongest statistical support above the detection threshold. Sensitivity scripts vary history, cost, and cross-sectional breadth.

The native-breadth experiment reports that the detectable information-coefficient floor falls from approximately 0.18 in a one-asset setup toward approximately 0.02 with 100 assets. This is consistent with the intuition that breadth increases information ratio, but it is an empirical property of the implemented synthetic panel and decision thresholds rather than a universal guarantee.

### 13.4 Five-null battery

The five-null script covers:

- white noise;
- regime-switching volatility without mean signal;
- bid-ask bounce;
- a zero-forward-information cross-sectional factor;
- GARCH volatility clustering with zero mean.

The project reports 0/300 strongest certifications under deterministic seeds and a 5-basis-point turnover cost. At zero cost, the statistical gates alone leak approximately 8% on the bid-ask-bounce null; a 2-basis-point cost removes the apparent edge. This result demonstrates that the cost gate is load-bearing for microstructure artifacts rather than decorative.

### 13.5 Refereeing machine-generated factors

The RD-Agent application uses a cross-sectional factor bridge on a CSI300 subset. Project records report 16 generated factors, with 14 killed under per-factor evaluation and no factor surviving full-search deflation. Positive OOS Sharpe alone was insufficient for factors whose returns were concentrated in a regime.

This experiment has important qualifications:

- the run covered approximately 100 names and 487 days;
- external agent execution was limited by environment timeouts;
- cost assumptions were set to zero in the reported generator run;
- the script-level handoff is not yet a one-command reproducible package;
- the result concerns one generator run, not all agentic research systems.

### 13.6 Refereeing published anomalies

The Chen–Zimmermann script evaluates 212 monthly long-short anomaly return series. The project reports:

| Deflation scope | Surviving `watch` or stronger |
|---|---:|
| anomaly evaluated alone | 102/212, approximately 48% |
| all anomalies in one family | 6/212, approximately 3% |
| category proxy | 8/212, approximately 4% |

The roughly sixteen-fold range is a systems result about trial-family policy: an evaluator cannot report “survival” without disclosing the search denominator. Treating all 212 anomalies as one historical search is intentionally conservative and may over-penalize authors who did not observe the full zoo; evaluating each alone ignores field-level selection. Penrose currently records one selected scope per run. A standards-conforming report should show scope sensitivity.

### 13.7 Post-publication decay

The companion script uses study sample-end and publication metadata to compare in-sample and post-publication anomaly returns. Project records report approximately 52% aggregate post-publication decay. Survivor and killed groups exhibit broadly similar percentage decay, while the surviving group retains roughly four times the absolute post-publication monthly return of the killed group.

This analysis is descriptive and reuses Penrose verdicts derived from the same return panel. It does not establish that the system prospectively selects future alpha.

### 13.8 Process-conditional verdict (worked example)

`scripts/worked_example_process_conditional.py` isolates the central property that distinguishes the system from a backtester: a verdict is a function not only of a return series but of the search lineage that produced it. The script constructs a single, byte-identical synthetic return series `R` and scores it twice through the real backtest and verdict path (`run_backtest` followed by `p8_verdict`), changing nothing except the registered trial ledger:

| Process | Registered trials | Probabilistic Sharpe | Deflated Sharpe | Verdict |
|---|---:|---:|---:|---|
| A: preregistered single hypothesis | 10 (1 declared + 9 regime partitions) | 0.9997 | 0.9997 | `watch` |
| B: selected best of 200 | 209 | 0.9997 | 0.8976 | `kill` (`no_oos_edge`) |

Every other gate — minimum out-of-sample bars, three-fold sign stability, regime fragility, bootstrap edge interval, permutation alignment, walk-forward consistency, and holdout — passes identically for A and B, and the probabilistic Sharpe is identical; only the deflation-dependent quantities (trial count and deflated Sharpe) differ, and the deflated Sharpe alone crosses the kill threshold for B. Process A is capped at `watch` rather than `research-supported` by the modeled-cost provenance rule (Section 9), which is the system's honest behavior, not a tuning of this example.

The example is deterministic, requires no network, language model, or external data, and runs on a clean install, which is why we recommend it as the first reproduction. It is a constructed illustration of the deflation mechanism, engineered so that `R` sits just inside the kill boundary; it is not a claim that `R` is profitable or that the margin is wide. It demonstrates only that identical evidence yields different, defensible conclusions under different declared search processes — the property a conventional backtester cannot express.

## 14. Reproducibility and auditability

### 14.1 Clean installation

The core is intended to install with:

```bash
pip install -e .
penrose eval
pytest -q
```

The package vendors the exact Sharpe, PSR, DSR, fee, and linear-capacity functions used by the harness. This avoids dependence on a sibling research repository.

### 14.2 Determinism

Bootstrap, permutation, calibration, and synthetic-data generators use explicit seeds. Dream and extraction model calls are cached and record model, token counts, cost, and prompt hashes where implemented. Nevertheless:

- extraction and fidelity remain model dependent;
- the primary extraction temperature is 0.1 rather than exactly zero;
- network data can change;
- external catalog contents are not pinned by the core repository;
- experiment scripts may alter configuration, temporary ledgers, and holdout behavior.

A publishable artifact should include data snapshots, exact environment metadata, model identifiers, prompt hashes, configuration, and immutable result manifests.

### 14.3 Audit history

The repository documents adversarial review findings that changed verdict validity, including:

- a `max(PSR, DSR)` combination that made deflation ineffective;
- holdout leakage into kill gates;
- premature holdout burning;
- an unbounded global trial ledger;
- unsafe import-execution of persisted generated modules;
- non-reproducible null seeds;
- future-value filling in realized-volatility construction;
- malformed carrier and annualization paths.

The current tests encode guards for many of these failures. Audit logs are useful evidence of hardening, but they are not a substitute for an independent security review or statistical replication.

### 14.4 Reproducing the applications

The keyless core can run planted evaluations and synthetic controls. Other applications require additional assets:

- Chen–Zimmermann data in the expected local Parquet layout;
- an RD-Agent workspace and corresponding price panel;
- optional pre-collected series from a local data catalog (`PENROSE_DATA_DIR`, bring-your-own);
- an LLM endpoint for extraction, specification, reconstruction, fidelity, and dreaming;
- Docker for generated-module execution;
- vendor credentials for Databento or authenticated Kalshi access.

These requirements should be captured in a release manifest before archival publication. The current scripts contain user-specific default paths in some external experiment adapters.

## 15. Limitations and threat model

### 15.1 Reconstruction remains the central scientific risk

A statistically rigorous test of the wrong implementation is not a valid refutation. Source spans, specifications, controlled routing, fidelity review, and `cannot_replicate` reduce this risk but do not eliminate it. Penrose is strongest for code-complete candidates and weakest for underspecified prose.

### 15.2 Security isolation is substantial but not formally verified

Generated code runs with no network, constrained resources, a read-only root, and limited mounts. The implementation has not undergone a container-escape assessment. Docker itself is a privileged dependency, and image construction is part of the trusted computing base.

### 15.3 Causality validation is stronger, but not complete

The runtime truncated-data look-ahead check now runs on both trusted in-process modules and generated modules executed through the sandbox path. It catches modules whose overlapping early returns change when later data are removed. This does not prove full causal correctness: static heuristics and the dynamic rerun are layered defenses, and unusual leaks may still require code review or independent replication.

### 15.4 Statistical gates are heterogeneous

The walk-forward implementation is volatility specific. The MDE is approximate. The three-fold rule is intentionally strict. Regime partitions can be sparse. DSR depends on the chosen family and observed cross-trial Sharpe variance. These controls should be reported as a panel, not compressed into a universal scalar score.

### 15.5 Holdout policy does not scale

The production lock is single-use per distinct claim identity, which avoids one survivor blocking every later claim in a local instance. It still does not provide a dataset-scoped, rotating, multi-institution holdout service. Calibration scripts necessarily bypass the production policy with isolated forced holdouts. A production standard requires dataset-scoped, claim-scoped, access-logged holdout governance.

### 15.6 Economic validity is modeled

Transaction costs, slippage, borrow, financing, and market impact are not generally measured from execution. The production cap to `watch` is therefore appropriate. Capacity estimates are conditional and based on a linear impact model.

### 15.7 Data coverage is narrow and partially external

The implemented core does not yet offer production-grade point-in-time adapters for broad equities, options, corporate actions, borrow, prediction-market order books, or high-frequency execution. Some declared capabilities in planning documents remain unimplemented module slots. A user-supplied local catalog (`PENROSE_DATA_DIR`) improves coverage for pre-collected series but is not packaged into the core artifact; live keyless venues and keyed vendor adapters cover the rest.

### 15.8 Corpus intelligence is advisory and immature

The local corpus is useful for provenance and retrieval, but its size is small. Principle discovery is deterministic and conservative, and similarity is lexical by default or embedding-based when an in-process embedder is installed. The system does not yet demonstrate that accumulated principles improve future evaluation quality.

### 15.9 Independent confirmation is not implemented

Generated hypotheses are capped at `watch`, but there is no automated fresh-data confirmation round that can promote them. There is also no general independent-team replication workflow or final standards badge for replication.

### 15.10 Experimental claims require artifact packaging

The reported calibration, generator, and literature results are present in project documents and scripts, but this manuscript was prepared from the live repository rather than a frozen public release. An arXiv artifact should pin commit, data hashes, command lines, outputs, and environment.

## 16. Roadmap toward a standards-conforming reference implementation

The highest-priority engineering work is not adding more verdict labels. It is strengthening evidence boundaries:

1. extend the now-shipped sandbox causality rerun (§5.7) to catch additional look-ahead idioms beyond a single truncation point;
2. add deterministic reproduction of the source's reported in-sample statistic and a first-class `cannot_replicate` stage;
3. replace global holdout burning with dataset-scoped, access-logged confirmation pools;
4. provide trial-family sensitivity reports rather than one hidden denominator;
5. ship versioned vendor adapters with point-in-time and survivorship grades;
6. separate measured from modeled costs at the field level and ingest execution evidence;
7. freeze extraction prompts, model identifiers, data hashes, and environment manifests per run;
8. define an independent confirmation round for generated hypotheses;
9. expose a versioned machine-readable evidence report that alternative implementations can produce;
10. subject the sandbox, verdict logic, and calibration claims to external review.

Longer-term work includes generic estimator-aware walk-forward interfaces, a standard connector for research generators, multi-user authorization, shared but reproduce-not-trust corpora, and governance mechanisms outside the Penrose project.

## 17. Discussion

Penrose's main lesson is that performance-claim evaluation is a systems problem. Statistical rigor can be defeated upstream by an invented claim span, downstream by an unfaithful implementation, operationally by a reused holdout, economically by omitted costs, or institutionally by allowing machine output to write accepted knowledge.

The system therefore treats evaluation as a chain of evidence. A strong Sharpe ratio is insufficient if:

- the return stream was selected from an undisclosed search;
- the code read future data;
- the module implemented a proxy rather than the claim;
- the final holdout was repeatedly inspected;
- costs were modeled optimistically;
- the data were synthetic or unavailable;
- the sample could not resolve the claimed effect;
- a generated candidate lacked an external evidentiary anchor.

Conversely, a negative statistical result is insufficient to reject a claim when the evaluator could not reconstruct it, lacked the necessary data, or had inadequate power. The benefit of a system such as Penrose is not that it makes these judgments infallible. It makes them explicit, inspectable, and harder to collapse into a marketing-friendly pass/fail number.

For the broader community, the appropriate role is analogous to an artifact evaluator or reproducibility instrument. Authors can use the protocol before submission; reviewers can inspect a standardized evidence appendix; editors can require claim-level disclosures; and independent teams can reproduce reports with another implementation. Penrose can serve as a concrete test bed for that process, but adoption requires that the protocol remain open and that no single vendor or codebase become the sole authority.

### 17.1 Relation to existing quant-research systems

The closest open system is Microsoft Qlib, together with its companion factor-mining agent RD-Agent. Qlib provides a feature-representation engine (its expression language plus the Alpha158/Alpha360 handlers), a model zoo, and a configurable backtester; RD-Agent adds LLM-driven automated factor generation. A source-level review clarifies that these systems and Penrose are complementary rather than competing: Qlib is oriented toward *discovering and deploying* return-ranking models, whereas Penrose is oriented toward *independently falsifying* a claim after accounting for the search that produced it. Three observations are relevant. First, Qlib's benchmark model suite predicts a single fixed label (a cross-sectionally standardized one-day-forward return) over one shared feature set and one fixed chronological split; the resulting information-coefficient tables are an internally consistent model comparison, not evidence of independent, out-of-sample, post-cost alpha, and a tuned linear model is competitive with the deep architectures. Second, Qlib's backtester models transaction costs, market impact, and liquidity as user-set parameters rather than quantities calibrated to observed fills, and several defaults (fills at the closing price, zero modeled impact) are optimistic. Third, and most important for the present work, neither Qlib nor RD-Agent implements deflation for the size of the search, multiple-testing correction, a single-use locked holdout, or power-aware non-results; RD-Agent's selection loop re-scores the same test segment on every iteration, which makes the reported winners susceptible to sequential overfitting. Penrose is therefore best understood as the validation layer that consumes the output of generators such as Qlib/RD-Agent — it can referee their factors directly, as demonstrated in Section 13 — rather than as another research environment competing with them.

## 18. Conclusion

We presented Penrose, a local-first reference system for evaluating quantitative performance claims. The implemented system covers source ingestion, grounded extraction, relevance and falsifiability screening, economic preflight, semantic deduplication, trusted-module routing, specification generation, sandboxed reconstruction with self-repair, provenance-aware data assembly, multiple-testing accounting, robustness and power analysis, locked holdouts, claim-level verdicts, immutable generated-search registration, reports, a researcher dashboard, advisory corpus analysis, and a human authorization gate.

The implementation is broad enough to demonstrate what a performance-evidence protocol requires beyond statistical formulas. It is not yet broad or validated enough to act as an academic certification authority. Its strongest current properties are inspectability, explicit failure states, search-denominator accounting, isolation of generated code, power-aware negative results, and separation of machine proposals from human-committed knowledge. Its largest weaknesses are reconstruction fidelity, modeled execution assumptions, incomplete point-in-time data coverage, asymmetric causality testing for generated modules, simplistic holdout governance, and the absence of independent confirmation.

Penrose should therefore be evaluated as infrastructure: a reference implementation that makes a proposed evidence standard concrete, falsifiable, and open to competing implementations.

## References

1. Bailey, D. H., and López de Prado, M. (2014). “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.” *Journal of Portfolio Management*, 40(5), 94–107.
2. Harvey, C. R., Liu, Y., and Zhu, H. (2016). “…and the Cross-Section of Expected Returns.” *Review of Financial Studies*, 29(1), 5–68.
3. Hou, K., Xue, C., and Zhang, L. (2020). “Replicating Anomalies.” *Review of Financial Studies*, 33(5), 2019–2133.
4. Chen, A. Y., and Zimmermann, T. (2022). “Open Source Cross-Sectional Asset Pricing.” *Critical Finance Review*, 11(2), 207–264.
5. McLean, R. D., and Pontiff, J. (2016). “Does Academic Research Destroy Stock Return Predictability?” *Journal of Finance*, 71(1), 5–32.
6. Li, Y., Yang, X., Yang, X., Xu, M., Wang, X., Liu, W., and Bian, J. (2025). “R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization.” arXiv:2505.15155.
7. Politis, D. N., and Romano, J. P. (1994). “The Stationary Bootstrap.” *Journal of the American Statistical Association*, 89(428), 1303–1313.
8. Planton, J. (2026). “AlphaSeeker: A Framework for Systematic Alpha-Seed Discovery from Tick Data.” TFM Quantitative Trading Ltd, working paper. SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6823559
9. Stephan, R. (2026). “Sequential Tradeability Testing for Alpha Signals.” SSRN working paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6922558

## Appendix A. Implemented capability matrix

| Capability | Implemented status | Evidence boundary |
|---|---|---|
| PDF, text, and Markdown ingestion | Implemented | pypdf; parse failures become flagged empty sources |
| Prompt-injection flagging | Implemented | Pattern detection and role separation; not formal prompt security |
| Grounded external claim extraction | Implemented with LLM | Source and metric spans verified against extracted text |
| Offline extraction | Limited | Requires hand-authored claims file |
| Relevance screening | Implemented with LLM | Fails open; current domain list is narrow |
| Falsifiability classification | Implemented with LLM | Offline stub assumes testable |
| Cost preflight | Implemented | Current P4 defaults are specialized and not a universal cost model |
| Exact and semantic deduplication | Implemented | Semantic mode depends on optional embeddings and committed corpus |
| Trusted module registry | Implemented | Two reviewed strategy classes observed |
| Module specification generation | Implemented | Model-mediated; specifications require review |
| Auto-implementation and three-attempt repair | Implemented | Requires LLM and Docker |
| Static generated-code checks | Implemented | Defense in depth, not a sandbox |
| Container isolation | Implemented | Docker required; no unsandboxed fallback |
| Pickle-free sandbox result IPC | Implemented | Parquet and JSON |
| Runtime contract validation | Implemented | Shape, finiteness, frequency, and sanity constraints |
| Truncated-data look-ahead test | Implemented | Trusted path and sandbox path; static heuristics remain defense in depth |
| Fidelity refuter | Implemented with LLM | Nondeterministic; strongest replication gate remains roadmap |
| Provenance-carrying data contract | Implemented | Some inputs come from an optional local catalog (`PENROSE_DATA_DIR`) |
| Public crypto and DVOL adapters | Implemented best effort | Network dependent |
| Databento adapter | Implemented bring-your-own-key | User entitlement required |
| Broad vendor adapter framework | Roadmap | FRED/equity/options adapters not implemented |
| Synthetic fallbacks | Implemented and labeled | Provisional use only |
| Point-in-time vol/trend labels | Implemented for BTC-derived regimes | Not universal across assets |
| DSR, PSR, Sharpe | Implemented | Family and variance assumptions disclosed |
| Registered search ledger | Implemented | Unknown upstream generator search remains unsolved |
| Three-fold stability | Implemented | Strict sign requirement |
| Stationary block bootstrap | Implemented | Seeded |
| Signal-payoff permutation | Implemented when module supplies carriers | Tests alignment, not net profitability |
| Walk-forward | Partial | Specialized to volatility carrier |
| Regime kill lens | Implemented | Known timezone-degradation issue for optional labels |
| Capacity point estimate and interval | Implemented | Linear modeled impact; interval conditional on positive edge |
| Power-aware reclassification | Implemented | Approximate MDE |
| Single-use holdout | Implemented locally | Global lock; rotating governance absent |
| Generated-search preregistration | Implemented | Native dream runs only |
| Holdout-free generated triage | Implemented | Positive results capped at `watch` |
| Reports and charts | Implemented | Charts require optional Matplotlib |
| Local dashboard | Implemented | Approval remains CLI-only |
| Draft preflight and inbox submission | Implemented | Preflight uses a testability stub |
| Human approval firewall | Implemented | The sole knowledge-write path; requires an approver identity |
| Advisory connection discovery | Implemented | Deterministic, lexical/keyword based, never gates |
| Independent replication badge | Roadmap | No general workflow |
| Fresh-data confirmation for generated claims | Roadmap | Required to graduate beyond `watch` |
