# The Penrose Edge

> **Maturity status (read this first).** Penrose has two tiers, and they are at very different stages.
>
> - **The Referee — shipping and validated.** The falsification engine (multiple-testing deflation,
>   power-aware verdicts, the discovery/confirmation firewall, the honesty-calibration battery) is built,
>   tested, and demonstrated end-to-end (see `WORKED_EXAMPLE.md`: the same return series earns different
>   verdicts under different search lineages). **This is the product today.**
> - **The Scientist & Synthesizer — R&D, engineering-in-place, generative quality UNPROVEN.** The pieces
>   that extract principles from the corpus of past kills and form *new candidate hypotheses* exist and
>   are calibrated against noise (a noise corpus yields **0** confirmed survivors — it does not
>   hallucinate). But the actual generative claim — that accumulated kills produce candidate hypotheses
>   that survive independent confirmation and beat a naive guess — has **not** been tested, because it
>   requires a deep corpus of real kills that does not yet exist (today: ~8 concepts, 0 promoted
>   principles). Treat everything below about the Synthesizer as a research direction, not a delivered
>   capability.
>
> Penrose forms **candidate hypotheses from the corpus of kills**. It does **not** find alpha, and nothing
> it generates is "alpha" — a candidate that survives confirmation is "research-supported," still subject
> to human P9 review.

## Why not just backtest the strategy?

The shortest answer:

> A backtester evaluates an implementation. Penrose extracts reusable knowledge from the experiment.

A backtest estimates what would have happened if a particular set of rules had been traded on
historical data. Penrose asks a larger set of questions:

- What claim was the strategy supposed to test?
- Did the implementation faithfully test that claim?
- How was the strategy discovered and selected?
- How many alternatives were tried before this result was chosen?
- What actually explains the observed performance?
- Which explanations are inconsistent with the evidence?
- Under what conditions does the result survive or fail?
- Was the experiment capable of detecting the claimed effect?
- What can be learned from this experiment and reused in future research?

The backtest is evidence. Penrose audits the chain of reasoning from idea, to implementation, to
evidence, to conclusion—and uses what survives that audit to make the next research cycle smarter.

## Penrose is larger than its referee

Penrose began with falsification because trustworthy learning requires a trustworthy filter. But
falsification is the quality-control mechanism, not the full product.

The larger system is:

> A self-correcting quantitative research engine that turns experiments into reusable market
> knowledge and uses that knowledge to generate new candidate hypotheses (never "alpha" — candidates
> to be tested, not winners).

Its complete loop is:

```text
hypothesize
    → implement
    → simulate
    → falsify and explain
    → extract concepts
    → accumulate evidence
    → synthesize new hypotheses
    → independently confirm
```

This supports three logically distinct functions:

### Researcher

Conversationally formalizes human ideas, designs experiments, and creates implementations.

### Scientist

Tests claims, competing explanations, statistical credibility, regime dependence, costs, power, and
implementation fidelity. It extracts concepts from both successful and failed experiments.

### Synthesizer

Finds relationships across the concept corpus and proposes new hypotheses, strategy families, and
risk overlays that were not present in any single prior experiment.

The Scientist includes the independent Referee. The Synthesizer can propose but cannot certify its own
work. Every synthesized candidate must re-enter the same preregistered confirmation process.

This distinction matters: Penrose is not merely a validation engine with a generator attached. It is
a learning system whose accumulated knowledge changes what it chooses to investigate next.

---

## Three different objects

Quantitative research contains at least three distinct objects:

### Thesis

> Extreme perpetual-futures funding predicts subsequent price reversal.

### Strategy

> Short assets in the highest funding decile and hold for eight hours.

### Simulation

> These historical orders, under these fill and cost assumptions, produced this return series.

A conventional backtester principally evaluates the simulation. It may help express the strategy,
but it generally does not maintain or test the relationship between the strategy and its thesis.

Penrose treats all three as first-class objects.

It asks whether the strategy genuinely tests the thesis, whether the simulation is a credible test of
the strategy, and what conclusions the resulting evidence supports.

---

## The key thought experiment

Two researchers submit exactly the same strategy and receive exactly the same backtest:

- OOS Sharpe: 1.6
- 500 trades
- Positive after modeled costs
- Stable across three periods

Researcher A formed one economically motivated hypothesis, specified it before examining the test
data, and tested it once.

Researcher B generated 20,000 strategies, repeatedly inspected their results, and submitted the
winner.

A conventional backtester returns the same report for both. The submitted code and return series are
identical.

Penrose should not return the same conclusion:

- Researcher A may have produced meaningful evidence.
- Researcher B may have won a 20,000-ticket lottery.

The difference is not present in the return series. It exists in the research process that produced
the strategy.

> Backtesting evaluates the result conditional on a strategy. Penrose evaluates the credibility of
> the result conditional on the process that produced it.

---

## A profitable strategy does not necessarily validate its thesis

Suppose a funding-reversal strategy is profitable.

A backtest can show the trades, P&L, Sharpe ratio, drawdowns, and sensitivity to costs. Penrose goes
further by testing competing explanations:

- Did extreme funding actually predict price reversal?
- Did the return instead come from collecting funding payments?
- Was the strategy unintentionally long momentum?
- Was nearly all profit earned in one volatility regime?
- Did a small number of crisis observations dominate the result?
- Would a simpler delta-neutral carry strategy capture the same effect with less risk?

Penrose might conclude:

```text
Observed result:
The submitted strategy was historically profitable.

Supported concept:
Funding carry persisted while basis conditions remained stable.

Rejected explanation:
Extreme funding did not independently predict price reversal.

Fragility:
Performance collapsed during rapid basis expansion and deleveraging.

Next hypothesis:
Capture funding carry while reducing exposure when basis instability rises.
```

The strategy worked, but its stated thesis did not. The useful result is not merely the equity curve.
It is the more accurate concept recovered from the experiment.

Penrose should not claim to prove causality from historical data. It identifies plausible
explanations, tests competing explanations, eliminates those inconsistent with the evidence, and
records the surviving concepts with appropriate uncertainty.

---

## A failed backtest does not always disprove a thesis

Consider a plausible cross-sectional factor tested using 18 months of data and 40 assets. The
estimated Sharpe is 0.2.

A backtester may label the strategy unsuccessful.

Penrose asks whether the experiment had enough information to resolve the proposed effect. If the
sample could reliably detect only an information coefficient above 0.11, while realistic effects in
the domain are around 0.02–0.05, the proper conclusion is:

> The experiment is underpowered. The available evidence cannot distinguish a realistic edge from
> noise.

That is different from saying the thesis is false.

Penrose separates:

- Evidence that contradicts a thesis.
- Evidence too weak to support it.
- An experiment too weak to tell.
- Structural failures such as leakage, regime concentration, or implementation mismatch.

This prevents “we could not detect it” from silently becoming “it does not exist.”

---

## Better backtesting versus research governance

More realistic fills, commissions, latency, order books, margin rules, and corporate actions produce
a better simulation. Penrose can consume those simulations through adapters, but it does not need to
own every execution model.

Its distinctive role begins where simulation ends: determining what may legitimately be inferred
from the result.

The closest analogy is clinical research:

| Quantitative research | Clinical research |
|---|---|
| Strategy implementation | Treatment protocol |
| Backtest | Observed trial outcome |
| Search registration | Trial registration |
| Locked holdout | Prespecified endpoint |
| Implementation fidelity | Protocol adherence |
| Multiple-testing correction | Multiple-endpoint correction |
| Power and minimum detectable effect | Sample-size calculation |
| Verdict | Evidentiary conclusion |

A physiological simulator and a clinical-trial protocol are not substitutes. One models an outcome;
the other governs whether the evidence supports a claim.

Likewise:

> A backtester generates observations. Penrose governs inference.

---

## Conversational strategy creation is part of the distinction

Conversational strategy creation should not merely mean “an AI writes Python.”

Its deeper purpose is to turn an informal belief into an explicit research contract.

A user begins with:

> I think extreme funding means the market is overextended.

Penrose helps resolve the hidden degrees of freedom:

- Does the thesis predict continuation or reversal?
- Over what horizon?
- Is it cross-sectional or within each asset?
- Is funding the proposed mechanism or only a proxy?
- What outcome would falsify the thesis?
- Which competing explanations must be controlled?
- What information was available at each decision time?
- Which choices will be fixed before testing?
- What effect size would be economically meaningful?
- How many variants may be attempted?

The result is not only strategy code. It is a thesis, an implementation, an experiment design, and a
record of the choices made before the outcome was known.

---

## The internal firewall: Researcher and Referee

Penrose should contain two logically separate roles.

### The Researcher

The conversational Researcher can:

- Clarify an economic thesis.
- Define assets, horizon, universe, and implementation.
- Identify necessary point-in-time data.
- Generate strategy code or adapter specifications.
- Suggest alternative formulations.
- Diagnose failed experiments.
- Combine prior findings into new hypotheses.

### The Referee

The Referee should:

- Freeze the hypothesis and declared search budget.
- Record every attempted candidate and variation.
- Verify that the implementation tests the stated claim.
- Prevent development from inspecting protected confirmation data.
- Apply robustness, power, provenance, and multiple-testing controls.
- Produce a calibrated verdict.
- Refuse to relax standards because the Researcher wants a winner.

The Researcher may propose and argue. It cannot change the Referee's rules.

Without this separation, conversational iteration can repeatedly learn from the supposed test set
until “out of sample” becomes training data with extra steps.

---

## From experiment storage to compounding research

Most backtesting systems operate at one of the first three levels:

### Level 0 — No memory

Run a backtest and display the results.

### Level 1 — Result memory

Save strategies, parameters, metrics, and charts.

### Level 2 — Optimization memory

Use previous scores to search nearby parameter combinations.

### Level 3 — Experimental memory

Remember attempted variants, data lineage, search history, failures, and evidentiary strength.

### Level 4 — Conceptual memory

Extract mechanisms, regime conditions, alternative explanations, contradictions, and reusable
principles.

### Level 5 — Generative research memory

Use those principles and failure patterns to propose materially new hypotheses.

Most conventional backtesters stop at Levels 0–2. Trying more parameters is not the same as learning
from experiments.

Penrose's opportunity is to reach Levels 3–5.

---

## What the corpus should contain

The primary unit of memory should not be:

```text
Strategy X
Parameters Y
Sharpe Z
```

It should be closer to:

```text
Claim:
Extreme funding predicts price reversal over eight hours.

Evidence:
Three adequately powered experiments contradict the claim.
One small experiment weakly supports it.
The pooled directional effect is near zero.

Alternative explanation:
Positive returns came from funding accrual rather than price reversal.

Boundary:
Carry performance failed during rapid basis expansion.

Reusable principle:
Funding is more reliable as compensation than as a directional signal.

Implementation consequences:
Remain approximately delta-neutral.
Monitor basis instability.
Avoid unnecessary rebalancing.

Provenance:
Linked experiments, datasets, code versions, costs, and verdicts.
```

This lets Penrose reason across strategies that look different in code but test the same underlying
proposition.

Failures become useful evidence rather than discarded optimizer runs.

---

## How the corpus can generate better strategies

Suppose the corpus contains these supported or repeatedly observed findings:

```text
A. Funding carry is persistent but suffers during deleveraging.
B. Basis instability often precedes deleveraging.
C. Volatility-based risk controls are more stable than directional filters.
D. Frequent rebalancing consumes marginal carry through costs.
```

Penrose can synthesize a new candidate:

> Collect funding carry, reduce exposure as basis instability rises, size inversely to trailing
> volatility, and rebalance only after a material exposure threshold is crossed.

No prior strategy needed to contain that complete design. The hypothesis is assembled from tested
mechanisms, boundaries, and implementation constraints.

This is not guaranteed alpha. The new strategy remains a new hypothesis and must be preregistered and
tested against genuinely fresh evidence.

The compounding loop is:

```text
experiments
    → validated concepts and failure patterns
    → synthesis of new hypotheses
    → preregistered implementations
    → adversarial validation
    → a richer corpus
```

The system does not merely search the neighborhood of yesterday's best parameters. It uses yesterday's
experiments to decide what class of idea deserves to be tested tomorrow.

---

## Cross-strategy synthesis

The corpus need not be confined to variations of one strategy. Penrose can abstract findings across
asset classes and strategy families.

Suppose separate experiments establish:

```text
Crypto:
Funding carry survives except during basis instability.

Equities:
Volatility scaling reduces crash concentration.

Futures:
Trend filters add most value during liquidity transitions.

Prediction markets:
Liquidity provision becomes dangerous when spreads signal informed flow.
```

These implementations are different, but Penrose may identify a shared concept:

> Compensation-based strategies are often implicitly short a market-instability state. Their yield
> is most dangerous when financing, liquidity, or information conditions begin to destabilize.

That cross-family principle can produce several new candidates:

- Reduce crypto carry as basis dispersion expands.
- De-risk equity volatility-selling strategies as financing conditions tighten.
- Scale futures carry according to liquidity deterioration.
- Withdraw prediction-market liquidity when spread and flow conditions imply information arrival.
- Construct a shared instability overlay across several otherwise unrelated yield strategies.

No individual backtest contains that complete idea. It emerges by reasoning across the corpus.

Penrose can therefore synthesize candidate alpha from:

- Supported mechanisms.
- Repeated failure modes.
- Regime and market-state boundaries.
- Hidden common exposures.
- Cross-asset analogues.
- Cost, turnover, liquidity, and capacity constraints.
- Complementary weak effects.
- Contradictory findings that imply a conditional relationship.

The important wording is **candidate alpha**. A generated hypothesis is not alpha merely because it
was assembled from supported concepts. It becomes credible only after independent evidence confirms
the new combination.

Penrose does not just search strategy space. It learns a model of the structure of that space from
every experiment.

---

## Discovery and confirmation must remain separate

A corpus creates a new source of overfitting: Penrose can overfit its own accumulated research.

Therefore corpus-driven generation needs two explicit modes.

### Discovery

- Use the corpus freely.
- Combine concepts and failure patterns.
- Explore alternative mechanisms.
- Compare many candidate implementations.
- Record the complete candidate population and search budget.

### Confirmation

- Freeze the selected hypothesis and implementation.
- Use data, markets, periods, or evidence that did not contribute to its creation.
- Prohibit the generator from inspecting the final holdout.
- Distinguish reused evidence from independent replication.
- Apply multiple-testing correction for the complete discovery process.

Validated components do not make their recombination automatically valid. They provide a better prior,
not a free conclusion.

---

## What Penrose should and should not own

Penrose should own:

- Conversational thesis formation.
- Evidence-informed strategy and hypothesis synthesis.
- Experiment specification and preregistration.
- Search and revision lineage.
- Claim-to-code fidelity.
- Backtest-adapter orchestration.
- Statistical and structural falsification.
- Power-aware verdicts.
- Explanation and competing-hypothesis analysis.
- Concept extraction.
- A provenance-linked research corpus.
- Cross-strategy and cross-family concept abstraction.
- Corpus-informed candidate-alpha generation.

Penrose does not need to own:

- Brokerage.
- Live execution.
- Exchange connectivity.
- Every asset-specific fill model.
- Every market-data feed.
- A universal event-driven simulator.

Those capabilities can be supplied by adapters such as Qlib, LEAN, Mendl, NautilusTrader, vectorbt,
or custom institutional infrastructure.

The adapter supplies observations. Penrose decides what those observations justify believing and what
they teach the research process.

---

## Honest boundary: when Penrose really is just a better backtest

The criticism is valid if Penrose only:

- Receives a return series.
- Runs additional robustness statistics.
- Applies stricter thresholds.
- Produces a red, yellow, or green badge.

That would be a sophisticated validation harness, but still fundamentally a better backtest.

Penrose earns the larger category only if it evaluates information a normal backtester cannot see:

- The original thesis.
- The mechanism and competing explanations.
- The strategy's fidelity to the thesis.
- The complete discovery and revision process.
- The number of attempted alternatives.
- The independence of confirmation evidence.
- The experiment's ability to resolve the claimed effect.
- The relationship between this experiment and prior evidence.
- The reusable concepts learned from success or failure.

The product test is:

> Could Penrose give two different verdicts for the same return series because the results came from
> different research processes?

If not, it is a better backtester.

If yes—because of search lineage, preregistration, fidelity, provenance, power, and evidence
independence—it is a research-validation system.

A second test is:

> Can Penrose conclude that a profitable strategy does not validate its stated thesis?

If not, it is a better backtester.

If yes, it is evaluating knowledge claims rather than merely scoring implementations.

A third test is:

> Can a killed strategy produce a trustworthy concept that improves the next hypothesis?

If not, Penrose stores results.

If yes, Penrose compounds research.

---

## Common challenges and concise answers

### “Why can't I just backtest it?”

Because a backtest tells you how one implementation behaved under one simulation. It does not tell
you whether the result was selected from thousands of failures, whether the implementation tests the
stated idea, what explains the return, whether the experiment had enough power, or what knowledge
should transfer to the next strategy.

### “Isn't Penrose just a stricter backtester?”

It would be if it only added statistical gates. The intended product tracks the complete reasoning
and research process, tests competing explanations, governs inference, and turns experiments into
reusable concepts.

### “Does Penrose discover alpha?”

No. Penrose is a referee, not an alpha generator, and it finds no new alpha. As an unproven research
direction (not a delivered capability), the Synthesizer can *propose* candidate hypotheses from
accumulated evidence across many experiments and strategy families, combining supported mechanisms,
regime boundaries, shared risk exposures, cost constraints, and failure patterns into ideas that may
not have appeared in any prior strategy. These are candidates *to be tested*, capped below
`research-supported` until they survive genuinely independent confirmation, and never a claim of
profit. The Synthesizer proposes; the Referee judges.

### “Why does search history matter if the final backtest is good?”

Because the best result from 20,000 attempts is expected to look unusually good even when every
candidate is noise. Identical performance carries different evidentiary weight when it was predicted
once versus selected from a large search.

### “Does Penrose tell us why a strategy worked?”

It cannot casually prove causality from observational history. It decomposes exposures, tests
competing explanations, rejects explanations inconsistent with the evidence, and records which
mechanisms remain plausible.

### “Why retain failed strategies?”

Failures reveal which mechanisms, regimes, implementations, and assumptions do not hold. Structured
negative evidence can prevent repeated dead ends and can expose a more accurate principle than the
original successful-looking strategy.

### “Why not let Qlib, RD-Agent, or Mendl do this?”

They can be excellent generators, optimizers, simulators, or execution systems. Penrose can use them
through adapters. Its distinct responsibility is maintaining the research contract, complete search
lineage, independent confirmation boundary, calibrated inference, and reusable concept corpus.

### “Can the corpus really create more profitable strategies?”

It can create better-informed candidates by combining supported mechanisms, known boundaries, and
failure constraints. Profitability is never guaranteed, and corpus-derived candidates still require
fresh confirmation. The advantage is a progressively better-informed hypothesis search across
strategy families, not a promise of returns.

### “Is Penrose a falsification engine or an alpha-generation engine?”

A falsification engine. It is a self-correcting research referee, not an alpha generator. Falsification
prevents weak or contaminated conclusions from entering its knowledge base. Concept extraction turns
trustworthy experiments into reusable knowledge. Synthesis (an unproven research direction, not a
delivered capability) uses that knowledge to propose new candidate hypotheses to test. These are
stages of one loop, not a promise of returns.

---

## Positioning language

### One sentence

> A backtester tells you how a strategy performed; Penrose learns what the experiment teaches and
> uses that knowledge to generate what should be tested next.

### Short version

> Describe a thesis. Build the test. Try to kill it. Learn from it. Generate what comes next.

### Product description

> Penrose is an open-source conversational quantitative research engine. It turns informal market
> beliefs into explicit, reproducible experiments; evaluates those experiments with an independent,
> power-aware referee; and converts both successes and failures into provenance-linked concepts that
> improve future research and generate new candidate alpha across strategy families.

### Category

> Penrose is an open-source, self-correcting quantitative research engine.

Its methodological core is independent falsification. Its compounding asset is a corpus of tested
market concepts. Its generative output is evidence-informed candidate alpha.

### The durable distinction

> Backtesters optimize strategies. Penrose compounds market knowledge—and uses it to invent new
> strategies.

### The research loop

```text
converse
    → hypothesize
    → formalize
    → implement
    → simulate through an adapter
    → explain and falsify
    → extract and update concepts
    → synthesize candidate alpha
    → independently confirm
```

The backtest is the measurement instrument inside that loop. The accumulated, validated knowledge is
the Penrose edge. Falsification makes that knowledge trustworthy; synthesis makes it generative.
