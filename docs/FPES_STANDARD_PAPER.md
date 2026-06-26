# A Power-Aware Evidence Standard for Empirical Investment-Performance Claims

**Working preprint — not peer reviewed**

**Author:** Charles Patterson · Patterson Research  
**Contact:** charles.s.patterson@gmail.com  
**Version:** FPES proposal 0.1  
**Date:** June 2026

---

## Abstract

Empirical investment research is commonly summarized by a reported return, Sharpe ratio, information coefficient, or \(t\)-statistic. These summaries are difficult to compare across papers because they often omit the size of the research search, the power of the evaluation, the provenance of the data and costs, and the distinction between computational failure, structural invalidity, statistical weakness, and economic irrelevance. We propose the **Financial Performance Evidence Standard (FPES)**, an implementation-neutral, claim-level protocol for evaluating empirical claims of abnormal return, return predictability, or superior portfolio performance. FPES does not certify that a paper is true and does not replace peer review. It produces a versioned evidence profile over four independently reported dimensions: computational reproducibility, statistical support, robustness, and economic validity. The standard requires an explicit claim contract, point-in-time information rules, disclosure of search breadth, an untouched evaluation design, uncertainty and minimum-detectable-effect reporting, multiple-testing scope sensitivity, and implementation-cost analysis. Negative findings are power-aware: failure to reject a null is classified as inconclusive when the design could not reliably detect the effect claimed by the authors. Structural defects such as look-ahead, target leakage, or benchmark contamination remain invalidating regardless of power.

We define evidence levels for reproduced, statistically supported, economically supported, prospectively validated, and independently replicated claims, while preserving non-cumulative diagnostic states such as not assessable, cannot reproduce, structurally invalid, and inconclusive due to low power. We also specify governance, report schemas, conformance requirements, and a calibration program based on known nulls, injected effects, cross-implementation agreement, and blinded replication. Preliminary results from one prototype implementation illustrate why the standard is needed: among 212 published anomaly return series, the fraction labeled as surviving ranged from approximately 48% under claim-local evaluation to approximately 3% under a whole-corpus deflation assumption. These figures are not presented as validation of FPES; they demonstrate that conclusions can depend more on an unstated definition of the research family than on the reported point estimate. FPES therefore requires scope sensitivity rather than a single hidden multiple-testing denominator. The intended contribution is a common, auditable language for evidence behind investment-performance claims that can be implemented by journals, authors, replication teams, or competing software systems without dependence on a particular vendor.

**Keywords:** empirical asset pricing; investment performance; reproducibility; multiple testing; statistical power; backtest overfitting; transaction costs; evidence standards

**JEL classification:** C12, C18, C52, G11, G12, G14

**Code and artifact availability.** A conforming reference implementation, Penrose, is open source at https://github.com/PattersonResearch/Penrose (v0.2.0). The keyless core reproduces its calibration controls and a deterministic process-conditional worked example without an API key or external data; the application figures cited in §9 additionally require external datasets documented in the release and should be rerun from a frozen public artifact.

---

## 1. Introduction

Investment-performance claims are unusually exposed to selection. Researchers can vary signals, transformations, assets, universes, horizons, portfolio weights, risk models, transaction-cost assumptions, sample boundaries, and reporting metrics. A sufficiently broad search can produce an apparently compelling backtest even when no exploitable relation exists. Publication then reveals only a small portion of the search that generated the reported result.

The finance literature has developed important responses to this problem. Harvey, Liu, and Zhu (2016) argue that the large and expanding factor zoo requires substantially higher evidentiary thresholds than conventional single-test inference. Bailey and López de Prado (2014) propose the Deflated Sharpe Ratio to account for selection, non-normality, and finite samples. Hou, Xue, and Zhang (2020) show that many published anomalies do not survive standardized replication. McLean and Pontiff (2016) document substantial out-of-sample and post-publication decay. Chen and Zimmermann (2022) provide a large open replication resource for cross-sectional return predictors.

These contributions establish that conventional reporting is incomplete, but they do not by themselves define a shared operational standard for evaluating an individual performance claim. Current practice leaves several questions unresolved:

1. What exactly is the unit being evaluated: a paper, a model, a table, or a specific claim?
2. What minimum information must be disclosed before evaluation is possible?
3. How should a failed test be distinguished from an underpowered test?
4. Which multiple-testing family should be used when the true search history is partly unknown?
5. How should statistical support be separated from tradability?
6. What does it mean for an automated or human evaluator to conform to the same standard?
7. How can evidence remain comparable when data are proprietary or costs are modeled rather than observed?

We propose the **Financial Performance Evidence Standard (FPES)** to answer these questions. FPES evaluates a precisely stated performance claim and emits a structured evidence profile. It is designed for empirical claims that a signal, strategy, factor, model, or portfolio construction rule produces abnormal return, predictive information, or superior risk-adjusted performance.

FPES has five design principles.

First, the unit of evaluation is the **claim**, not the paper. A paper may contain several claims with different data, power, and implementation assumptions. Whole-paper approval conceals this heterogeneity.

Second, evidence dimensions remain separate. Computational reproduction, statistical support, robustness, and economic validity answer different questions and should not be collapsed into an opaque score.

Third, negative evidence is power-aware. A valid but weak design that cannot detect the authors' claimed effect is inconclusive, not proof that the effect is absent.

Fourth, multiple-testing assumptions are reported as a sensitivity profile. When the research search is not fully known, a single correction chosen by the evaluator creates false precision.

Fifth, the standard is implementation-neutral. No software product, institution, or proprietary model is required to issue an FPES report. A conforming implementation must disclose its methods, pass public conformance tests, and emit an auditable report.

FPES is not a publication rule, an investment recommendation, or a truth oracle. Editors may accept theoretically important work with inconclusive empirical evidence; investors may reject statistically supported effects because of mandate, capacity, or operational constraints. The standard supplies a common description of the evidence so those decisions are not made from an unqualified headline statistic.

### 1.1 Contributions

This paper contributes:

- a claim-level contract for empirical investment-performance assertions;
- four independently reported evidence dimensions;
- a power-aware diagnostic taxonomy;
- a required multiple-testing scope-sensitivity analysis;
- cumulative evidence levels that distinguish reproduction, statistical support, economic support, prospective validation, and independent replication;
- a machine-readable, implementation-neutral report structure;
- a governance and conformance model; and
- a calibration and validation agenda for establishing the operating characteristics of the standard.

### 1.2 Status of the proposal and evidence

FPES is presently a proposed standard, not an adopted academic rule. The definitions and requirements below are normative proposals. The empirical results in Section 9 are preliminary illustrations produced by a prototype evaluation system. They have not yet been independently replicated, and repository materials identify required reruns and unresolved validation work. No claim in this manuscript should be read as evidence that FPES already has calibrated field-wide error rates.

---

## 2. Scope and unit of evaluation

### 2.1 In-scope claims

FPES applies to empirical claims including:

- positive abnormal returns or alpha;
- return predictability measured by an information coefficient, forecasting loss, or portfolio return;
- superior Sharpe ratio, information ratio, drawdown, utility, or other risk-adjusted performance;
- profitable trading strategies;
- improved portfolio construction or allocation performance;
- machine-generated factors or strategies;
- performance attributed to alternative data, machine learning, or complex model selection; and
- robustness or persistence of a previously reported performance effect.

The standard can be specialized through domain modules for cross-sectional equity factors, time-series strategies, high-frequency trading, derivatives, digital assets, or portfolio allocation. The core evidence requirements remain common.

### 2.2 Out-of-scope claims

FPES does not by itself evaluate:

- purely theoretical propositions;
- causal claims whose primary estimand is not investment performance;
- accounting, corporate-finance, survey, or market-design results without a performance claim;
- legal or regulatory conclusions;
- qualitative investment theses without a falsifiable quantitative prediction; or
- the suitability of an investment for a particular person or institution.

Such work may use other reproducibility or causal-inference standards. If an otherwise out-of-scope paper contains an investment-performance claim, FPES may be applied to that claim alone.

### 2.3 The claim contract

Every evaluated claim must be restated as a **claim contract** before testing. At minimum:

> Using information set \(I_t\), rule \(S\) applied to universe \(U\) at decision time \(t\) produces outcome \(Y_{t+h}\) over horizon \(h\), with performance relative to benchmark \(B\), under implementation assumptions \(C\), and is claimed to exceed threshold \(\delta\).

The contract must identify:

| Field | Required content |
|---|---|
| Claim identifier | Stable identifier for the individual claim |
| Population and universe | Instruments, venues, regions, eligibility and delisting rules |
| Decision time | Exact timestamp or market event at which a decision is made |
| Information set | Data available at the decision time, including publication and revision lags |
| Signal or rule | Deterministic specification, executable code, or complete mathematical definition |
| Outcome horizon | The interval over which performance is realized |
| Primary estimand | Alpha, mean return, Sharpe, IC, utility gain, loss differential, or other prespecified quantity |
| Benchmark | Zero, market model, factor model, investable alternative, or incumbent method |
| Direction and threshold | Sign and minimum effect the claim asserts |
| Evaluation population | Time period, cross-section, regimes, or future data to which the claim generalizes |
| Cost and capacity assumptions | Fees, spread, slippage, financing, borrow, impact, and scale |
| Search provenance | Number and structure of candidate signals, specifications, and model variants considered |

If the source is ambiguous, the evaluator must not silently choose a favorable interpretation. It must either obtain author confirmation, evaluate explicitly labeled alternative interpretations, or return **not assessable**.

### 2.4 One paper, multiple claims

Claims must be separated when they differ materially in outcome, horizon, universe, benchmark, or implementation. A paper cannot inherit the strongest evidence level earned by one result for all other results. The paper-level summary is a list or distribution of claim-level profiles, not a single scientific truth rating.

---

## 3. Mandatory disclosure package

An FPES evaluation requires a disclosure package sufficient to determine what was tested, what information was available, and how the result was selected.

### 3.1 Data provenance and timing

The package must disclose:

- data vendor, dataset version, extraction date, and licensing constraints;
- raw and adjusted fields used;
- corporate-action, delisting, and survivorship treatment;
- timestamps and time zones;
- publication and revision lags for macroeconomic, fundamental, analyst, or alternative data;
- universe construction at each decision date;
- missing-value and outlier treatment;
- whether the dataset is point-in-time;
- all transformations performed before model fitting; and
- any manual data corrections.

The information rule is stricter than a chronological train/test split. A value dated at \(t\) is not usable at \(t\) if it was published, revised, or operationally available later.

### 3.2 Research and model-selection provenance

The package must report, to the extent known:

- number of candidate signals considered;
- number of assets, horizons, transformations, lags, filters, and portfolio variants;
- hyperparameter search space and optimization procedure;
- discarded specifications;
- prior related searches by the same project or system;
- the rule by which the reported specification was selected;
- whether the primary hypothesis and metric were preregistered; and
- which data partitions were viewed during development.

The standard distinguishes **declared search breadth** from **effective search breadth**. Highly correlated variants need not be counted as fully independent, but any effective-trial adjustment must be justified and reported. Missing search history is not interpreted as one trial.

### 3.3 Code and execution environment

The preferred package contains:

- source code or an executable research artifact;
- dependency lock file or environment specification;
- deterministic seeds where applicable;
- commands that regenerate the primary tables and figures;
- expected output checksums or tolerances;
- hardware requirements; and
- a license sufficient for evaluation.

When code or data cannot be shared, the report must distinguish legal or contractual unavailability from computational failure. Restricted-data claims may still be evaluated through a controlled enclave, journal data editor, or independent licensed replicator, but they cannot receive an unrestricted public-reproduction designation.

### 3.4 Performance and implementation assumptions

The package must disclose:

- position sizing and gross/net exposure;
- rebalancing frequency and execution timing;
- turnover definition;
- commissions and exchange fees;
- bid-ask spread and slippage;
- market impact model;
- shorting and borrow availability;
- financing and margin costs;
- treatment of halted, stale, or non-tradable instruments;
- assets under management or notional scale;
- constraints and risk controls; and
- whether costs are observed, estimated from contemporaneous quotes, or modeled.

### 3.5 Missing disclosure

Missing information is itself reportable evidence. It must not automatically imply fraud or invalidity. Depending on materiality, it produces:

- **not assessable** when the claim cannot be reconstructed or tested;
- a lower reproducibility level;
- a range of statistical conclusions under alternative search scopes; or
- economic support explicitly conditional on stated assumptions.

---

## 4. Evidence dimensions

FPES reports four dimensions independently. Implementations may add domain-specific diagnostics, but they must not replace these dimensions with a single proprietary score.

### 4.1 Dimension A: computational reproducibility

This dimension asks whether an evaluator can regenerate the reported empirical object.

### 4.1.1 Required checks

The evaluator should:

1. reconstruct the environment;
2. regenerate the primary sample and portfolio;
3. reproduce the primary statistic and uncertainty estimate;
4. compare tables and figures within declared tolerances;
5. inspect discrepancies in signs, units, dates, universes, and annualization; and
6. record all manual interventions.

Tolerance must be tied to numerical precision, stochastic estimation, or sampling uncertainty. It must not be widened after observing a discrepancy.

### 4.1.2 Reproducibility states

| State | Meaning |
|---|---|
| A0 — Not assessable | Data, code, definition, or access is insufficient |
| A1 — Cannot reproduce | Adequate materials were supplied, but the principal result cannot be regenerated |
| A2 — Partially reproduced | Direction or broad result is recovered, but material discrepancies remain |
| A3 — Reproduced within tolerance | Primary result and relevant sample are regenerated |
| A4 — Reproduced from independent reconstruction | Result is recovered without relying on the authors' executable implementation |

A1 is not equivalent to a statistical rejection of the economic claim. It states that the reported evidence cannot be established from the supplied materials.

### 4.2 Dimension B: statistical support

This dimension asks whether the estimated effect is distinguishable from the relevant null after accounting for dependence, non-normality, selection, and statistical power.

### 4.2.1 Minimum statistical report

Every claim must report:

- point estimate in economically interpretable units;
- uncertainty interval;
- sample size and effective sample size;
- sampling frequency and degree of overlap;
- distributional and dependence diagnostics relevant to the estimator;
- prespecified primary test;
- claimed effect threshold;
- minimum detectable effect or power curve;
- declared and sensitivity-adjusted multiple-testing result; and
- results relative to the prespecified benchmark.

No particular statistic is universally mandated. A Sharpe-based strategy may use probabilistic or deflated Sharpe methods; a forecasting claim may use a loss-differential test; a cross-sectional factor may use portfolio and regression evidence. The chosen method must match the estimand and data-generating structure.

### 4.2.2 Dependence and effective sample size

Nominal observations must not be treated as independent when returns overlap, positions persist, assets share common shocks, or repeated cross-sections are clustered. The report must state the dependence model and use an appropriate method such as block resampling, heteroskedasticity/autocorrelation-consistent inference, clustering, or another justified procedure.

### 4.2.3 Prespecified benchmark

The primary comparison must be stated before evaluation. Zero return is often too weak. Depending on the claim, the benchmark may be:

- a passive investable portfolio;
- a standard factor model;
- the incumbent forecasting model;
- a simpler nested model;
- a cost-equivalent heuristic; or
- the best available non-proprietary alternative.

Changing the benchmark after observing results creates an additional specification search and must be recorded as such.

### 4.3 Dimension C: robustness

Robustness asks whether support persists under perturbations that should not eliminate a genuine claim. It is not permission to run an unlimited garden of tests and report survivors.

### 4.3.1 Core robustness checks

Subject to the domain, the core includes:

- an untouched out-of-sample or prospective evaluation;
- walk-forward or rolling evaluation when parameters are fitted;
- stability across prespecified time segments;
- sensitivity to plausible portfolio-construction choices;
- sensitivity to data cleaning and universe definitions;
- placebo, permutation, or negative-control tests;
- leakage and look-ahead audit;
- block-bootstrap or equivalent dependence-aware uncertainty;
- comparison across economically relevant regimes; and
- benchmark sensitivity.

### 4.3.2 Robustness family disclosure

Robustness tests are themselves a search. The evaluator must define the robustness battery before examining its outputs or disclose which tests were added after results were observed. A claim should not fail merely because one arbitrary perturbation was chosen from many, nor pass because favorable perturbations were selectively reported.

### 4.3.3 Structural invalidity

Some failures invalidate the empirical design independently of statistical power:

- use of future information;
- target leakage;
- survivorship or universe construction that uses future membership;
- contamination of an asserted holdout;
- non-causal alignment of signal and return;
- benchmark contamination;
- material mismatch between the stated claim and tested implementation; or
- arithmetic or unit errors that generate the reported effect.

The evaluator must identify the defect and its causal relevance. “Structural invalidity” is not a label for every unstable estimate.

### 4.4 Dimension D: economic validity

Economic validity asks whether the supported statistical effect remains meaningful under implementable conditions.

### 4.4.1 Minimum economic report

The evaluator must report:

- gross and net performance;
- turnover;
- explicit fees;
- spread and slippage assumptions;
- financing and borrow assumptions;
- delayed-execution sensitivity;
- capacity or scale sensitivity;
- exposure to standard systematic risks;
- tail and drawdown behavior;
- investable benchmark comparison; and
- operational assumptions required to implement the strategy.

### 4.4.2 Cost-provenance hierarchy

Costs must be labeled:

| Provenance | Meaning |
|---|---|
| Modeled | Generic or assumed schedule not fitted to the evaluated strategy's executions |
| Quote-calibrated | Estimated from contemporaneous spreads, depth, or venue schedules |
| Paper-execution observed | Measured from prospective simulated orders against available market conditions |
| Live-execution observed | Measured from actual prospective fills at disclosed scale |

A claim evaluated only with modeled costs may receive conditional economic support, but the report must not describe it as execution validated.

### 4.4.3 Capacity

Capacity estimates must include their model, market data, participation assumptions, uncertainty, and scale. A single linear-impact number without empirical calibration is a scenario, not an established capacity estimate.

---

## 5. Power-aware inference

The central distinction in FPES is between **unsupported with adequate power** and **inconclusive because the evaluation could not resolve the claimed effect**.

### 5.1 Claimed effect and smallest effect of interest

Each claim must state:

- the effect reported by the authors, \(\hat{\delta}\);
- the minimum effect asserted or economically relevant, \(\delta_{\mathrm{claim}}\); and
- where appropriate, a smallest effect of interest, \(\delta_{\mathrm{SEI}}\), below which the effect is not economically meaningful.

If authors do not state a threshold, the evaluator may report power over a grid rather than invent one.

### 5.2 Minimum detectable effect

The evaluator must report the minimum detectable effect (MDE) at declared type-I error and target power, using a design-appropriate variance estimate. For a simple independent standardized mean, an approximation is

\[
\mathrm{MDE} \approx \left(z_{1-\alpha} + z_{1-\beta}\right)
\frac{\sigma}{\sqrt{n}},
\]

but financial data generally require adjustments for dependence, overlapping horizons, cross-sectional correlation, estimation error, and multiple testing. FPES does not prescribe the simple approximation when its assumptions fail.

The report must give the MDE in the same units as the claim whenever possible: monthly return, annualized Sharpe, alpha, information coefficient, utility gain, or forecast-loss improvement.

### 5.3 Decision logic

A negative statistical result is classified as:

- **statistically unsupported** if the design had adequate power to detect \(\delta_{\mathrm{claim}}\) or \(\delta_{\mathrm{SEI}}\), yet the required support was absent; or
- **inconclusive — underpowered** if the design could not reliably detect that effect.

A confidence interval that excludes economically meaningful effects can support a negative conclusion even if it includes zero. Conversely, a wide confidence interval containing both substantial positive and negative effects is inconclusive.

### 5.4 Power does not excuse structural defects

Power classification applies to uncertainty about an otherwise valid estimand. It does not rehabilitate look-ahead, leakage, contaminated holdouts, or a mistranslated strategy. A structurally invalid design does not estimate the intended effect.

### 5.5 Breadth and dependence

Cross-sectional breadth can increase power, but nominal asset count is not independent breadth. Common exposures, correlated signals, shared liquidity, and repeated use of the same time periods reduce effective breadth. Any information-ratio or IC-based power argument must estimate or bound this dependence. The fundamental-law relation between information coefficient and breadth is useful intuition, not a substitute for a design-specific power calculation.

---

## 6. Multiple testing and search-scope sensitivity

### 6.1 Why a single denominator is inadequate

The relevant search may include the specification shown in a paper, all variants tried by the authors, a generator's complete candidate population, a laboratory's related research program, or a broader published literature. These scopes answer different questions. Treating the submitted specification as the only trial is often too permissive; treating every historical anomaly as if it were jointly searched by every author may be too conservative.

FPES therefore requires a **scope-sensitivity profile**.

The consequence is concrete and can be exhibited on a single series. In the reference implementation, a deterministic worked example (`scripts/worked_example_process_conditional.py`) scores one byte-identical return series under two declared scopes — a single preregistered test, and selection of the best of 200 trials — holding the series, costs, and all non-deflation robustness gates fixed. The preregistered scope yields a surviving verdict; the best-of-200 scope yields a kill, on identical returns, with the deflated Sharpe alone crossing the threshold. The verdict is thus a function of the declared search, not of the return series in isolation. A conforming report that omits the search scope is therefore not merely incomplete; it is undefined, because the same evidence supports opposite conclusions under different scopes.

### 6.2 Required scopes

When feasible, the report must show at least:

1. **Claim-local scope:** the submitted specification is treated as the sole primary test. This is a descriptive lower bound on the selection penalty, not evidence that no search occurred.
2. **Declared-search scope:** correction uses all candidates and specifications disclosed for the project.
3. **Research-program sensitivity:** results are shown over plausible larger effective search sizes or related families.

For generated research, the declared scope must include every candidate generated or evaluated in the registered cohort, not only candidates retained by the generator.

### 6.3 Unknown search breadth

If search history is incomplete, the evaluator should report a survival function over effective trial counts:

\[
n \mapsto E(n),
\]

where \(E(n)\) is the evidence measure or decision at assumed effective search size \(n\).

The report should identify:

- the largest effective search size under which support remains;
- assumptions used to translate raw candidates into effective trials;
- whether the result is stable across reasonable family definitions; and
- which conclusion depends on unverifiable author disclosure.

### 6.4 Choice of correction

FPES does not require one universal correction. Depending on the setting, conforming methods may include family-wise error control, false-discovery-rate control, stepwise resampling, data-snooping tests, Sharpe deflation, hierarchical models, or selective-inference procedures. The implementation must explain:

- the error quantity controlled;
- the family of hypotheses;
- dependence assumptions;
- whether the method controls discovery or evaluates a selected winner; and
- how the correction interacts with model selection and robustness tests.

### 6.5 Registration and trial ledgers

Prospective research should register:

- candidate-generation budget;
- primary metric;
- family definition;
- data partitions;
- stopping rule; and
- confirmation protocol.

A trial ledger must be append-only or otherwise auditable, concurrency-safe, and resistant to deletion of failed candidates. The standard does not require public disclosure of proprietary signal definitions before evaluation, but it does require verifiable accounting of search breadth.

---

## 7. Diagnostic states and evidence levels

FPES separates diagnostic states from positive evidence levels.

### 7.1 Diagnostic states

| State | Definition |
|---|---|
| Not assessable | Required definition, data, code, access, or search information is insufficient |
| Cannot reproduce | Adequate materials were supplied, but the reported result cannot be regenerated |
| Structurally invalid | A design defect prevents estimation of the stated claim |
| Inconclusive — underpowered | The valid evaluation cannot resolve the claimed or economically meaningful effect |
| Statistically unsupported | The evaluation had adequate power, but required statistical support was absent |
| Economically unsupported | Statistical support exists, but the effect does not survive implementation or benchmark requirements |

These states must include reasons and supporting metrics. They are not ordinal scores: “cannot reproduce” and “underpowered” identify different problems.

### 7.2 Positive evidence levels

| Level | Short label | Minimum interpretation |
|---|---|---|
| FPES-R | Reproduced | Primary computational result reproduced within declared tolerance |
| FPES-S | Statistically supported | FPES-R plus adequate, selection-aware statistical support and required robustness checks |
| FPES-E | Economically supported | FPES-S plus net economic value under disclosed, plausibly calibrated implementation assumptions |
| FPES-X | Prospectively validated | FPES-E plus confirmation on data or executions unavailable during development |
| FPES-I | Independently replicated | The claim is recovered by an independent team or implementation using independently sourced data or reconstruction |

FPES-R through FPES-X are cumulative. FPES-I is an orthogonal distinction and must specify what was independent: operator, code, data source, sample, or all four.

Examples:

> **FPES-R/S; economic validity inconclusive.** The result was reproduced and survived declared-search correction, but borrow-cost data were unavailable.

> **FPES-R; statistically inconclusive due to power.** The implementation was reproduced, but the untouched sample could detect only effects larger than the authors' claimed alpha.

> **Structurally invalid.** The portfolio used index constituents selected with future membership information; no statistical level assigned.

### 7.3 No universal pass/fail

FPES deliberately avoids a universal “paper passed” badge. Journals and institutions may define policies such as “performance claims must reach FPES-R and disclose the full profile,” but the standard itself reports evidence. A binary gate would discard information, incentivize threshold gaming, and conflate scientific and editorial judgments.

---

## 8. Conforming implementations and reports

### 8.1 Implementation neutrality

A conforming FPES implementation may be:

- a reproducible analysis script;
- a journal replication service;
- an academic laboratory workflow;
- a commercial system;
- a controlled data-enclave process; or
- a manual audit supported by standard statistical software.

No implementation receives exclusive authority to define or issue FPES evidence levels.

### 8.2 Required implementation disclosures

Each implementation must publish:

- supported FPES version;
- statistical methods and thresholds;
- domain modules;
- default assumptions;
- random seeds or stochastic reproducibility protocol;
- known limitations;
- conformance-test results;
- software and dependency version;
- conflicts of interest; and
- whether human judgment altered the machine-generated report.

An implementation may be stricter than the core standard, but it must identify additional requirements and must still emit the common fields.

### 8.3 Machine-readable report

Every report must contain at least:

```yaml
fpes_version: "0.1"
claim_id: "persistent-identifier"
claim_contract:
  universe: "..."
  decision_time: "..."
  information_set: "..."
  outcome_horizon: "..."
  primary_estimand: "..."
  benchmark: "..."
  claimed_effect: "..."
search_scope:
  claim_local: 1
  declared_candidates: null
  sensitivity_grid: [1, 10, 50, 100, 500]
reproducibility:
  state: "A3"
statistical:
  point_estimate: null
  interval: null
  effective_sample_size: null
  mde: null
  power_for_claimed_effect: null
robustness:
  prespecified_tests: []
  structural_findings: []
economic:
  gross_result: null
  net_result: null
  cost_provenance: "modeled"
diagnostic_state: null
evidence_levels: ["FPES-R"]
implementation:
  name: "..."
  version: "..."
  report_hash: "..."
```

The final schema should use a formally versioned JSON Schema or equivalent. The example is illustrative, not yet a ratified schema.

### 8.4 Auditability

Reports should be content-addressed or digitally signed, retain input hashes, and identify all post hoc changes. Restricted inputs may be represented by secure hashes and reviewed by an authorized replicator. A report without enough provenance to determine what was tested is not conforming.

### 8.5 Conformance suite

The standard should publish fixed cases including:

- reproducible positive controls;
- pure-noise nulls;
- autocorrelated and fat-tailed nulls;
- bid-ask-bounce and stale-price placebos;
- explicit look-ahead and leakage traps;
- known underpowered effects;
- known multiple-search winners;
- cost-fragile strategies; and
- intentionally incomplete disclosure packages.

The suite should test report semantics as well as numerical output. For example, a valid low-power null must not be mislabeled as structurally invalid, and a leakage case must not be rescued by a large sample.

---

## 9. Preliminary empirical motivation

This section reports existing prototype results only to motivate design choices. These results are not a validation of FPES, have not been independently replicated, and should be rerun from a frozen public artifact before publication.

### 9.1 Multiple-testing scope can dominate the conclusion

A prototype implementation evaluated 212 monthly long-short anomaly return series from the Chen–Zimmermann open cross-section. The reported survival count depended strongly on the assumed family:

| Scope used in prototype | Surviving claims | Share |
|---|---:|---:|
| Claim-local evaluation | 102 / 212 | approximately 48% |
| Category proxy | 8 / 212 | approximately 4% |
| Whole-corpus family | 6 / 212 | approximately 3% |

The exact “whole-corpus” implementation used approximately 212 candidate trials plus a small number of regime partitions, rather than literally 212 trials. Repository audit notes state that this distinction has negligible numerical effect on the six-survivor count but should be reported accurately. The category grouping was a sign-group proxy rather than a validated taxonomy.

The result does not establish that 3% or 48% is the correct replication rate. The whole-corpus family may over-penalize papers whose authors did not jointly search all anomalies; the claim-local family under-penalizes any undisclosed research search. The result supports the FPES requirement to publish scope sensitivity and avoid an unexplained single denominator.

### 9.2 Preliminary null and injection controls

The prototype repository reports:

- zero “research-supported” outcomes in a deterministic five-null battery of 300 draws when a 5-basis-point turnover cost is included;
- an approximately 8% leak for a bid-ask-bounce placebo when statistical gates are used at zero cost, eliminated at transaction costs of at least 2 basis points;
- zero certification at injected IC \(=0\) across tested cross-sectional breadths; and
- a decline in the injected-IC certification threshold from roughly 0.18 at one asset to roughly 0.02 at 100 assets in one simulation design.

These are useful engineering controls, but they are not yet sufficient evidence of calibrated false-positive or false-negative rates. The null battery is finite and partly synthetic; cost assumptions determine one result; the breadth experiment uses a specific injected data-generating process; and the repository roadmap calls for deterministic reruns before publication. FPES validation requires the broader program in Section 10.

### 9.3 Preliminary generator case

The prototype also reports an evaluation of 16 factors generated by an automated research system on a limited CSI300 panel. Fourteen were rejected under per-factor robustness evaluation, two remained in an intermediate state, and none survived full-search deflation. This is consistent with the selection problem FPES addresses, but it is not a field-level estimate: the sample is small, the time span is short, costs were not measured, and operational failures limited the generator run.

A source-level review of that generator family (an open automated factor-mining agent built on a widely used quant-research library) is what motivates the standard's mandatory requirements rather than the prototype's particular numbers. The generator proposes candidates, has a model write and debug their code, and backtests each cumulative factor set — but it records no trial count in any significance calculation, applies no multiple-testing correction or deflated performance statistic, and re-scores the *same* fixed test segment on every iteration, so the model-selection signal is the test set itself. The underlying library, similarly, reports information-coefficient and return point estimates with no confidence intervals, no purged or embargoed cross-validation, and no single-use holdout. These are not isolated defects; they are the default posture of contemporary generate-and-backtest tooling. FPES is the response: requiring a preregistered search budget (Section 6), a single-use locked evaluation set, and power-aware non-results so that a candidate's apparent edge cannot be an unaccounted artifact of the search that produced it.

---

## 10. Calibration and validation plan

A proposed standard earns authority through measured operating characteristics and independent use, not through completeness of its checklist.

### 10.1 Phase I: specification freeze and reference cases

Before estimating field-wide performance:

1. freeze FPES 0.1 definitions and report schema;
2. publish a public conformance suite;
3. implement at least two independent evaluators;
4. preregister calibration metrics and thresholds; and
5. archive all benchmark datasets and seeds where licensing permits.

### 10.2 Phase II: false-positive calibration

Test null environments that preserve realistic nuisance structure:

- independent white-noise signals on real returns;
- volatility clustering with zero conditional mean;
- regime-switching volatility without return predictability;
- bid-ask bounce and stale-price artifacts;
- zero-forward-IC cross-sectional factors;
- randomized event dates;
- time-shifted signals;
- shuffled labels within dependence-preserving blocks; and
- multiple-search experiments in which the selected winner is known to be null.

For each, estimate the frequency of every diagnostic state and evidence level, not merely the strongest false certification. Calibration should be stratified by sample length, frequency, breadth, turnover, tail behavior, and search size.

### 10.3 Phase III: sensitivity and power calibration

Inject known effects into realistic residual structures and estimate:

\[
P(\text{FPES state or level}\mid \delta, n, b, c, m),
\]

where \(\delta\) is effect size, \(n\) is time-series length, \(b\) is effective breadth, \(c\) is cost or turnover environment, and \(m\) is search size.

Required outputs include:

- detection curves rather than one threshold;
- empirical coverage of uncertainty intervals;
- agreement between reported and realized power;
- sensitivity to non-normality and dependence;
- effect of correlation among trials; and
- rates at which true but fragile effects are classified as robust.

### 10.4 Phase IV: historical benchmark corpus

Apply FPES prospectively, under a frozen version, to:

- published anomalies with open return series;
- papers with author-supplied code and data;
- known failed replications;
- strategies with documented post-publication decay;
- simple investable benchmarks; and
- selected machine-generated candidate populations with complete trial ledgers.

The benchmark corpus must avoid tuning FPES thresholds to reproduce a preferred replication narrative. A development subset may be used for specification design; a separate locked corpus must be reserved for validation.

### 10.5 Phase V: blinded and independent replication

Independent teams should receive claim packages without the prototype verdict and report:

- claim-contract agreement;
- reconstruction agreement;
- structural-finding agreement;
- quantitative estimate agreement;
- diagnostic-state agreement; and
- evidence-level agreement.

Disagreements should be adjudicated publicly and used to refine ambiguous standard language. High inter-implementation disagreement would indicate that FPES is not sufficiently specified.

### 10.6 Phase VI: prospective prediction

The strongest validation is prospective. For claims evaluated at time \(T\), record:

- FPES profile at \(T\);
- subsequent performance on data generated after \(T\);
- realized costs where observable; and
- whether evidence levels rank future persistence or economic value.

The standard should not be optimized solely for predicting positive returns. It should be evaluated on calibration: claims with similar evidence profiles should have similar subsequent outcome distributions.

### 10.7 Success criteria

Quantitative criteria should be preregistered after pilot studies. At minimum, a mature FPES release should demonstrate:

- controlled false certification under registered null families;
- uncertainty intervals with acceptable empirical coverage;
- power reports that track realized detection rates;
- stable semantics across independent implementations;
- low rates of structural-error misclassification;
- explicit sensitivity where search scope is uncertain; and
- prospective association between stronger evidence profiles and more persistent net performance.

No single criterion proves validity. The standard must publish failures and recalibrate versioned releases rather than silently changing thresholds.

---

## 11. Governance and institutional adoption

### 11.1 Separation of standard and implementation

The organization governing FPES should not require use of a particular implementation. Commercial and open-source systems may compete on automation, supported domains, security, and user experience while conforming to the same report contract.

### 11.2 Governance body

A credible governance body should include:

- empirical asset-pricing researchers;
- econometricians and statisticians;
- journal editors or data editors;
- market-microstructure and execution specialists;
- quantitative practitioners;
- research-software and reproducibility experts; and
- public-interest or conflict-of-interest representation.

No implementation vendor should control the voting majority.

### 11.3 Change process

Each change should include:

- public proposal;
- stated failure mode addressed;
- empirical evidence;
- backward-compatibility analysis;
- open comment period;
- conformance-suite changes; and
- versioned release notes.

Reports must remain interpretable under the version that generated them. A later standard must not retroactively relabel old reports without issuing a new evaluation.

### 11.4 Conflicts and appeals

Evaluators must disclose financial, employment, authorship, and commercial conflicts. Authors should be able to appeal:

- an incorrect claim contract;
- a reconstruction error;
- an inappropriate data or cost assumption;
- a misdefined search family; or
- a conformance violation.

Appeals should amend the report history rather than erase the original report.

### 11.5 Adoption path

A realistic sequence is:

1. voluntary claim appendices;
2. replication-workshop or course pilots;
3. journal-recognized evidence profiles;
4. editor-requested FPES reports for performance claims; and
5. prospective badges or registry entries after independent governance is established.

The initial standard should remain narrow. Attempting to cover all financial research would dilute the statistical and institutional problem FPES is designed to solve.

---

## 12. Limitations and open questions

### 12.1 No standard eliminates researcher discretion

Family definition, benchmark choice, smallest effect of interest, robustness battery, and cost scenario require judgment. FPES makes that judgment visible; it cannot remove it.

### 12.2 Unknown search histories

Authors and automated systems may omit failed trials. Scope sensitivity bounds the consequence but cannot recover an unobserved search. Cryptographic or institutional trial registration may help prospectively.

### 12.3 Proprietary data and strategies

Full public reproduction may be impossible. Controlled replication can provide evidence, but outsiders must trust the enclave or designated evaluator. FPES should distinguish restricted verification from public reproducibility.

### 12.4 Statistical support is not permanence

Even well-supported effects can decay after discovery because of arbitrage, crowding, market change, or publication. Prospective validation is therefore a stronger level, not a promise of permanent alpha.

### 12.5 Economic models are conditional

Spread, impact, borrow, and capacity estimates are strategy- and scale-specific. Modeled costs can be informative without being definitive. Reports must preserve their conditional nature.

### 12.6 Reconstruction of prose claims

Automated reconstruction can test the wrong strategy. A conforming evaluator must establish implementation fidelity by reproducing the authors' reported statistic, obtaining author confirmation, or independently reconstructing the method. An unverified translation cannot support a substantive negative verdict.

### 12.7 Holdout exhaustion

Repeated claims can exhaust a shared holdout even if each claim is evaluated once. Rotating holdouts, prospective data accrual, hierarchical error budgets, or external confirmation are needed for high-throughput research systems.

### 12.8 Domain heterogeneity

Monthly factor portfolios and high-frequency execution strategies cannot share identical robustness or cost modules. FPES requires a common core plus validated domain extensions.

### 12.9 Threshold gaming

Any recognized evidence level can become a target. Continuous metrics, full profiles, versioned methods, and prospective audits reduce but do not eliminate gaming.

### 12.10 Validation evidence remains incomplete

The current prototype evidence is limited. Required work includes independent implementation, frozen reruns, broader nulls, blinded replication, measured execution costs, and prospective follow-up. Until then, FPES should be described as a standards proposal supported by motivating examples.

---

## 13. Discussion

FPES changes the question from “Did this paper pass?” to “What kind of evidence supports this specific performance claim, under which assumptions?”

This shift matters for authors because an underpowered result no longer has to be defended as either success or failure. It matters for reviewers because common fields expose missing search provenance, contaminated holdouts, and cost assumptions without forcing every referee to invent a private checklist. It matters for editors because computational reproduction can be required independently of statistical or economic conclusions. It matters for readers because two claims with the same reported Sharpe can receive different profiles when one was selected from a thousand candidates and the other was prospectively registered.

The most consequential requirement may be multiple-testing scope sensitivity. Preliminary anomaly results show a roughly sixteen-fold difference in the number of survivors between claim-local and whole-corpus assumptions. Neither endpoint is automatically correct. The scientific defect is presenting one endpoint without saying which research process it represents.

The second consequential requirement is power-aware language. A field that labels every non-significant result a failed anomaly will overstate negative evidence, especially for short samples, low-frequency strategies, and realistic small effects. A field that labels every underpowered result promising will preserve unfalsifiable claims. Reporting the claimed effect, MDE, and interval makes the distinction operational.

The third is separation between statistical and economic evidence. The preliminary microstructure null illustrates that statistical gates may accept a pattern generated by bid-ask mechanics while small transaction costs eliminate it. This is not a reason to let arbitrary cost assumptions rescue statistical testing. It is evidence that statistical support and tradability answer different questions and both must be reported.

If adopted, FPES would be most useful as a common appendix and report format rather than a centralized authority. Its success would be measured by whether independent researchers can produce comparable assessments, identify disagreements precisely, and improve the calibration of empirical performance claims.

---

## 14. Conclusion

Empirical investment-performance research needs a reporting standard that reflects how such claims are produced: through noisy data, broad searches, weak effects, and implementation frictions. FPES proposes a claim-level, power-aware, selection-aware evidence profile with independent dimensions for reproducibility, statistical support, robustness, and economic validity.

The standard does not declare papers true, replace peer review, or require a particular software system. It requires evaluators to state what was tested, what information was available, what search was conducted, what effect could have been detected, how conclusions change under plausible search scopes, and whether performance survives implementable costs.

The proposal is ready for criticism and calibration, not institutional mandate. Its next necessary steps are a frozen specification, machine-readable schema, public conformance suite, multiple independent implementations, blinded replication, and prospective validation. If those steps succeed, FPES could provide finance with a shared language for distinguishing reproduced computation, statistical evidence, economic plausibility, and genuine independent confirmation.

---

## References

Bibliographic details below were verified against publisher or arXiv records.

1. Bailey, D. H., & López de Prado, M. (2014). “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.” *Journal of Portfolio Management*, 40(5), 94–107.

2. Chen, A. Y., & Zimmermann, T. (2022). “Open Source Cross-Sectional Asset Pricing.” *Critical Finance Review*, 11(2), 207–264.

3. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). New York: McGraw-Hill. ISBN 978-0-07-024882-3.

4. Harvey, C. R., Liu, Y., & Zhu, H. (2016). “…and the Cross-Section of Expected Returns.” *Review of Financial Studies*, 29(1), 5–68.

5. Hou, K., Xue, C., & Zhang, L. (2020). “Replicating Anomalies.” *Review of Financial Studies*, 33(5), 2019–2133.

6. McLean, R. D., & Pontiff, J. (2016). “Does Academic Research Destroy Stock Return Predictability?” *Journal of Finance*, 71(1), 5–32.

7. Nikolopoulos, S. D. (2026). “Spurious Predictability in Financial Machine Learning.” arXiv:2604.15531. *[Preprint; the associated QuantAudit package is not yet publicly released.]*

8. Politis, D. N., & Romano, J. P. (1994). “The Stationary Bootstrap.” *Journal of the American Statistical Association*, 89(428), 1303–1313.

---

## Appendix A. Minimum FPES claim appendix

An author-facing appendix may be kept short if it links to complete artifacts.

1. **Claim:** one-sentence claim contract.
2. **Primary estimand and threshold:** metric, direction, and smallest meaningful effect.
3. **Information rule:** what was known at each decision time.
4. **Data:** source, version, point-in-time status, universe, exclusions.
5. **Search:** number and structure of candidate specifications.
6. **Development design:** train, validation, test, and holdout use.
7. **Primary result:** point estimate, interval, and dependence adjustment.
8. **Power:** MDE and power for the claimed effect.
9. **Scope sensitivity:** claim-local, declared-search, and wider-search results.
10. **Robustness:** prespecified battery and structural audit.
11. **Economics:** gross/net performance, turnover, costs, borrow, capacity, provenance.
12. **Reproduction:** code/data access and execution instructions.
13. **Evidence profile:** diagnostic state and earned FPES levels.

## Appendix B. Example human-readable report

> **Claim:** A monthly long-short equity factor produces positive risk-adjusted return after costs in U.S. common stocks.
>
> **Reproducibility:** FPES-R. The reported return series and primary Sharpe were reproduced within prespecified tolerance.
>
> **Statistical evidence:** Supported under claim-local and 24-specification declared-search scopes; inconclusive when sensitivity exceeds 180 effective trials. The 95% interval for net monthly return is [0.08%, 0.61%]. MDE at 80% power is 0.21% per month.
>
> **Robustness:** No detected look-ahead. The effect survives block resampling and the prespecified post-2005 split but weakens under micro-cap exclusion.
>
> **Economic evidence:** Conditional. Net return remains positive under quote-calibrated spread and fee estimates at \$25 million, but borrow cost is modeled and capacity above \$80 million is not established.
>
> **Evidence profile:** FPES-R/S; conditional FPES-E not awarded pending borrow-cost verification. Search-scope sensitivity is material.

This example is illustrative and does not represent an evaluated paper.
