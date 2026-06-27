# How Penrose decides: every test it runs, in plain language

Penrose is a referee for quantitative research. You hand it a claim, something like "this strategy
earns a Sharpe of 1.8," and Penrose spends its effort trying to prove that the claim is luck,
overfitting, or an artifact of how it was found. Whatever survives that gauntlet earns a verdict.
Nothing is taken on trust, including (especially) a pretty backtest.

This page explains every test Penrose runs, which we call **gates**, without the math. If you can read
a backtest and you have been burned by one before, you can read this. The formal version of each gate,
with the statistics and the citations, lives in the academic write-up (the Penrose systems paper); this
page is the plain-language companion to it.

The one idea to hold onto: **most ideas should die, and a kill is a service.** A tool that blesses your
strategy is flattering you. A tool that tells you exactly how your strategy could be fooling you is
doing the job.

---

## The four verdicts

Every claim that makes it through reconstruction comes out as one of four words. Penrose deliberately
refuses to collapse them into a single score, because "this is wrong" and "I cannot tell yet" are
completely different statements and hiding the difference is how people get hurt.

| Verdict | Plain meaning |
|---|---|
| **kill** | The evidence does not hold up. The edge is luck, overfitting, look-ahead, or it lives in one lucky pocket of history. |
| **underpowered** | Penrose cannot tell. The data is too thin to resolve a realistic edge from noise. This is *not* a kill. The idea might be real; you just have not given it enough evidence to prove it. |
| **watch** | The edge survived the statistics but has not cleared every bar (for example, the costs are modeled rather than measured, or the locked holdout has not confirmed it). Promising, not certified. |
| **research-supported** | The strongest verdict Penrose gives. The edge survived every gate, including a one-time locked holdout. Note the word: *supported*, not *proven*, and never *profitable*. Penrose is a referee, not a fund. |

Penrose makes **no claim that anything is tradeable or profitable.** It only tells you whether the
evidence for a claim survives honest testing.

---

## How a claim travels through Penrose

Before any gate fires, Penrose has to turn a claim into something testable:

1. **Ingest and extract.** A paper, a write-up, or a code-complete strategy comes in. Penrose pulls out
   the specific, falsifiable claim and checks that the words it extracted actually appear in the source
   (no inventing claims the author never made).
2. **Reconstruct in a sandbox.** Penrose rebuilds the strategy as runnable code in an isolated
   container, with no network and no access to anything it should not see.
3. **Split the data and lock a holdout.** The history is divided. Most gates only ever see the early
   part. A final slice is locked away and can be opened exactly once.
4. **Run the gates.** Everything below.
5. **Return a verdict** plus a report that pairs every flattering number with its deflating context.

The most important structural rule is the **firewall**: the part of Penrose that searches for and
builds ideas can never peek at the locked confirmation data. This is the single discipline that
separates honest research from accidentally tuning your strategy until it fits the answer.

---

## The gates

For each gate: what it catches, an example of a strategy that dies on it, and why you should want it
even when it kills your favorite idea.

### 1. Reconstruction fidelity
**Catches:** a strategy that was described loosely enough that the tested version is not the real one.
**Dies here:** a paper says "buy on momentum" without defining the lookback, the rebalance, or the
universe, so the rebuilt version is a guess. **Why you want it:** if Penrose tested a broken copy of
your idea, every downstream verdict is meaningless. It flags this rather than pretending.

### 2. The firewall and the locked holdout
**Catches:** the most common self-deception in quant research, reusing your test data until the
strategy passes. **Dies here:** a strategy that was tweaked over and over against the same out-of-sample
period, so that period quietly became training data. **Why you want it:** the holdout can be opened
once. After that it is burned. This is what makes a `research-supported` verdict mean something.

### 3. Deflated Sharpe Ratio, the spine
**Catches:** an impressive Sharpe that is impressive only because you tried hundreds of variations and
kept the best one. **Dies here:** you test 200 parameter combinations, the winner shows Sharpe 2.0, but
once Penrose accounts for the 200 tries, the deflated Sharpe says that result is roughly what pure luck
would have produced. **Why you want it:** this is the single most important number in the system. The
Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) corrects a Sharpe for how many things you
tried and for fat-tailed returns. Without it, "best of many backtests" looks like skill. **One honest
caveat:** deflation scales with the size of the search Penrose has actually seen. On a first, single
isolated claim there is nothing yet to deflate, so the score is the Probabilistic Sharpe Ratio plus
the rest of the robustness stack; the deflation strengthens as a family or generator accumulates more
trials.

### 4. Three-fold sign stability
**Catches:** an edge that only exists in one stretch of time. **Dies here:** a strategy that made all
its money in 2020 and is flat or negative in the other thirds of the sample. **Why you want it:** a
real edge should at least keep the same sign across different chunks of history. One lucky era is not an
edge.

### 5. Regime fragility
**Catches:** an edge that lives entirely in one market condition. **Dies here:** a strategy whose entire
profit comes from high-volatility days, or only from one trend direction, and collapses everywhere
else. **Why you want it:** if dropping a single bucket of conditions erases the edge, the edge was that
bucket, not your signal.

### 6. Bootstrap edge interval
**Catches:** an edge that is not distinguishable from zero once you account for small-sample noise.
**Dies here:** a strategy with a positive average return whose resampled confidence interval comfortably
includes zero. **Why you want it:** the headline average can be positive while the honest uncertainty
band still straddles "no edge at all." Penrose resamples the returns thousands of times (a stationary
block bootstrap, Politis and Romano, 1994) to see the real spread.

### 7. Permutation, the signal-alignment check
**Catches:** a strategy that profits by accident rather than because the signal actually lines up with
the outcome. **Dies here:** you shuffle the link between the signal and the returns it supposedly
predicts, and the strategy makes just as much money shuffled as unshuffled. **Why you want it:** if your
signal carries no real information, breaking the connection between signal and payoff should hurt. If it
does not, there was nothing there.

### 8. Walk-forward consistency
**Catches:** parameters that drift, so the settings that worked early stop working later. **Dies here:**
a strategy that needs different parameters in each window to keep performing, which means it was fit to
the past, not to the market. **Why you want it:** a durable edge does not need to be re-tuned every year
to survive.

### 9. Cost and capacity
**Catches:** an edge that exists on paper but not after you pay to trade it, or one that evaporates at
any meaningful size. **Dies here:** a high-turnover strategy whose gross edge is real but smaller than
the round-trip cost of capturing it. **Why you want it:** gross returns are a fantasy. The only edge
that matters is the one left after costs and market impact.

### 10. Power-aware "underpowered" labeling
**Catches:** the mistake of calling a thin-data result "dead" when you simply could not have detected a
real edge with so little data. **Dies here:** nothing dies here; this gate is the opposite, it *rescues*
results from a false kill. **Why you want it:** Penrose computes the smallest edge your data could even
detect. If that floor sits above the size of a realistic edge, the honest answer is "I cannot tell,"
not "this is worthless." A skeptic that rejects everything is just a broken machine that always says no.

---

## The newer gates

These five extend the battery. They come from two ideas in the published literature (see refs 8 and 9
in the README): a sequential view of validation, and a candidate-discovery process with an
interpretability check.

### 11. Persistence-matched null
**Catches:** the risk of certifying a signal that only looks real because it drifted in one direction
for a long time. **Dies here:** this is a calibration test, not a per-strategy gate. Penrose builds fake
signals with no genuine edge but with realistic persistence, and checks that it never certifies them.
**Why you want it:** it lets Penrose state honestly what it is actually responding to. The honest
admission it produces: Penrose certifies *persistent direction*, and cannot by itself tell a real edge
apart from slow-moving style or regime exposure. Knowing the limit is part of trusting the tool.

### 12. Resolution estimate on "underpowered"
**Catches:** the dead-end feeling of an `underpowered` verdict. **Dies here:** nothing; this gate turns a
"cannot tell" into a recipe. **Why you want it:** instead of just saying "not enough evidence," Penrose
tells you roughly how much more would settle it, for example how many more out-of-sample trades you
need, or how many names you would have to test across to resolve a realistic edge at cross-sectional
breadth. It treats validation as something you can continue, not a single pass-or-fail.

### 13. Dead-state null
**Catches:** a subtle calibration failure where a method cannot represent a truly dead signal and so
drifts into certifying noise. **Dies here:** a genuinely dead signal (zero edge, realistic structure)
that a weaker tool might wave through. **Why you want it:** Penrose explicitly checks that a signal with
no edge lands on `kill` or `underpowered`, never on a survivor verdict. It is a guard against the tool
fooling itself.

### 14. Cost-sensitivity (breakeven cost)
**Catches:** a verdict that looks robust at one cost assumption but flips at a slightly higher one.
**Dies here:** a thin-margin strategy that is "supported" at the modeled cost but a kill if costs are a
hair higher than assumed. **Why you want it:** rather than judging at a single cost point, Penrose
reports the cost level at which the verdict flips. "Survives up to here, dies above" tells you how much
of your edge is really just an optimistic fee assumption.

### 15. Corpus isolation and mechanism
**Catches:** an idea with no relatives and no story. **Dies here:** nothing automatically; this one is
advisory. **Why you want it:** Penrose remembers everything it has ever invalidated. A claim that sits
in a well-populated, previously-tested family with a plausible mechanism deserves more confidence than a
lone result with no neighbors and no reason to work. This turns Penrose's growing record of what does
not work into an active prior on what comes next. (On a fresh install the memory is empty, so this gate
simply stays quiet until you have run enough claims to fill it.)

### 16. Combinatorial purged cross-validation (CPCV)
**Catches:** an edge that survives one chronological train/test split but not the many recombined
splits a careful evaluator would try. **Dies here:** a strategy whose apparent edge depends on the exact
ordering of its in-sample and out-of-sample windows. **Why you want it:** CPCV (Lopez de Prado) builds
many train/test partitions with purging and embargoing between adjacent observations, so a real edge has
to hold across recombinations, not just one lucky cut. An independent overfitting axis next to the
bootstrap and walk-forward.

### 17. Declared regime scope
**Catches:** the unfairness of killing a claim that never *claimed* to work everywhere. **Dies here:**
nothing extra; this is the honest counterpart to the regime kill-lens. **Why you want it:** a claim may
pre-register the regime it asserts it holds in (e.g. high-volatility) and be tested *within* that scope
instead of penalized for being flat elsewhere. It is adherence-gated: the narrower test is granted only
when the strategy actually confines its activity to the declared regime, so nothing trades outside its
stated scope while still claiming the easier exam.

### 18. Tail-risk / widow-maker (opt-in)
**Catches:** a stable, well-deflated strategy that nonetheless has a bounded upside and an unbounded
downside, the short-volatility / positive-carry profile that works 95% of the time and then a rare
event annihilates it. **Dies here:** a fat-left-tail payoff (strongly negative skew, a left tail far
heavier than the right) even when the average edge is positive and every other gate passes. **Why you
want it:** the other gates catch overfitting and instability; they do *not* catch a genuinely stable
edge whose payoff shape is a time bomb. This gate is off by default (so it never silently moves a
verdict) and reports the tail diagnostics on every run; enable it to kill or cap such payoffs.

### 19. Data-granularity check (at the input boundary)
**Catches:** the silent, confident-but-wrong verdict that results from feeding data at the wrong
sampling frequency, e.g. intraday bars to a rule written for daily data. **Dies here:** nothing is
killed; the mismatch is flagged before it can corrupt anything. **Why you want it:** a wrong-frequency
input is invisible to every other gate and poisons every statistic downstream with full confidence.
Penrose infers a series' actual frequency at the data boundary and warns on a mismatch, the input-side
counterpart to the existing check that a strategy's output bars-per-year matches its calendar span.

---

## The philosophy, in one paragraph

Penrose is built on a deliberately uncomfortable idea: the most valuable output of a research process is
usually a **clean, well-documented invalidation**, not a winner. Every gate above is a different way for
a claim to fail honestly, and every failure Penrose records makes the next claim easier to judge. If you
came here hoping for a machine that confirms your edge, this is the wrong tool. If you came here hoping
for one that tells you the truth about your edge before the market does, you are in the right place.
