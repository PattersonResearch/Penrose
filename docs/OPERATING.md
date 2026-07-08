# Operating Penrose: from a claim to a verdict

This guide is for anyone, a person or an agent, who wants to **run** Penrose on a trading claim rather
than work on its code. If you are here to develop Penrose, read [AGENTS.md](../AGENTS.md) instead.

Penrose is a referee. You hand it a performance claim; it reconstructs and stress-tests that claim, and
returns a calibrated verdict about whether the edge is believable given how it was found. Your job is
to hand it a clean, falsifiable claim and to read the result honestly.

## Prerequisites

```bash
pip install -e .          # editable install; Penrose runs from the clone
# The core (eval, calibration, backtests) runs with no API key and no network.
# Ingesting a paper (P2 extraction) or the generative paths need PENROSE_LLM_API_KEY.
```

Confirm the install is sound before you trust any verdict it produces:

```bash
python scripts/eval_suite.py           # must print 106/106 passed
python scripts/calibration_placebo.py  # placebo control: no no-edge signal may be certified
```

## The loop

```
a source (paper, tweet, thesis, idea)
   -> 1. TRIAGE     is there a falsifiable, quantitative performance claim here?
   -> 2. FORMALIZE  write the structured Claim
   -> 3. SUBMIT     penrose run --claims  /  run_source  /  the MCP tool
   -> 4. VERDICT    read the verdict, or the honest routing state
   -> 5. P9         a human, and only a human, promotes a survivor
```

Most of the value is in steps 1 and 2. A raw source is usually under-specified or out of scope; turning
it into a precise, testable claim is the real work, and it is exactly what Penrose is designed to
receive.

## 1. Triage: is there a claim to test?

Penrose referees claims about **investment performance**: an abnormal or risk-adjusted return, a
tradeable signal, a factor, a return predictability, a strategy edge. Before formalizing, decide which
of three the source is:

- **Formalizable** — a specific rule or strategy with (ideally) a stated metric. Go to step 2.
- **In scope but under-specified** — a real performance idea missing the pieces needed to test it
  (rules, universe, metric). Send it back for specification; do not submit a vague claim.
- **Out of scope** — commentary, a modeling or estimation improvement with no tradeable claim, market
  news, or pure theory. Do not submit; it will correctly come back `off_domain`.

The scope gate is a **scope** check, not a data check: a claim is in scope even when Penrose lacks the
data to test it (that becomes `needs_data`), and out of scope only when there is no tradeable
performance claim at all.

## 2. Formalize into a Claim

Fill these fields (a `penrose.brain.Claim`, or an inline JSON object for the CLI or MCP):

- `statement` — the precise, falsifiable performance claim, in one sentence.
- `mechanism` — why it should work (the hypothesized edge).
- `scope` — asset class, universe, and period the claim applies to.
- `horizon` — decision frequency (`daily`, `intraday/5-minute`, `monthly`, ...). Be honest here: the
  input-side granularity gate uses it, and a wrong-frequency test is silently invalid.
- `claimed_metric_quote` — the number the source claims (Sharpe, return, IC), so the referee has a
  target. If the source gives none, say so; the verdict is still meaningful.
- `applicable_strategy_class` — a controlled-vocabulary class that routes the claim to a module.
- `sample_period` (optional) — the source's own evaluation window `{start, end}`. When present, Penrose
  can require post-sample evidence before it trusts a survivor.
- `expected_edge` (optional) — the claimed net edge per trade, so the fee gate can actually evaluate it.

## 3. Submit

Three equivalent doors, all guarded (nothing is auto-approved, and the P9 human gate always holds):

```bash
# A. Submit already-structured claims (best: skips lossy prose re-extraction)
penrose run --claims claims.json         # claims.json = a JSON list of Claim objects

# B. Submit the source itself and let P2 extract the claims (needs PENROSE_LLM_API_KEY)
penrose run --paper path.pdf             # use when the source has precise numbers a summary lacks
```

```python
# C. In-process, for scripts
from penrose.pipeline.run import run_source
run_source(claims_override=[claim], use_llm=False)
```

For an **external agent**, the MCP server exposes the same submit path as a tool. Start it in
management mode and call `penrose_run_claim`:

```bash
PENROSE_MCP_MANAGEMENT=1 penrose-mcp      # or: penrose-mcp --management
```

## 4. Read the verdict, honestly

Penrose returns either a **verdict** or an honest **routing state**. Both are features, not failures.

Verdicts (a real gated judgment on a real backtest):

- `kill` — the claim did not survive falsification.
- `underpowered` — the data could not resolve an edge of the claimed size. This is not a rejection, it
  is an inconclusive, and it ships a resolution estimate (how much more data or breadth would settle it).
- `watch` — survived the kill gates but not certified: borderline, or capped because costs are modeled
  rather than measured, or a generated hypothesis not yet independently confirmed. A provisional survivor.
- `research-supported` — cleared the full stack, including the single-use locked holdout. It means
  "survived falsification," not "profitable," and still requires human review.

Routing states (an honest stop before a verdict):

- `off_domain` — no tradeable performance claim (triage should catch this first).
- `needs_data` — in scope, but the data is not held; it names exactly what is missing.
- `pending_module` — a spec was generated but no trusted module exists yet.
- `cannot_replicate` — a module was built, but the fidelity check judged it unfaithful to the claim.
- `engine_error` — an internal failure; the claim was not tested. Route it to a human. Do not treat it
  as a data gap or a kill.

Never fabricate a verdict, a number, or data. A routing state is the right answer when it is the true
one.

## 5. The human gate (P9)

A `research-supported` verdict is flagged for human review; it is never auto-promoted. Promotion into
the approved corpus is a P9 decision made by a person:

```bash
python -m penrose.pipeline.p9_review list                    # what is pending
python -m penrose.pipeline.p9_review approve <i> --approver <you>
```

No agent surface (the CLI, the MCP, or a script) can cross P9. This is a hard boundary, not a
convenience, and you must not try to route around it.

## The agent's contract

- Formalize and submit faithfully; never fabricate data, a claim, or a verdict.
- Treat `needs_data` / `pending_module` / `engine_error` as honest stops, not obstacles to route around.
- Never cross P9. Promotion to the approved corpus is a human decision.

## Read-only inspection (safe to expose)

Without management mode, the MCP server is read-only: `penrose_verdicts`, `penrose_status`,
`penrose_data_requests`, `penrose_principles`, `penrose_proposals`. An agent can pull "what has been
decided" and "what data is missing" with no ability to run or promote anything, so the read-only server
is safe to hand to an untrusted orchestrator.
