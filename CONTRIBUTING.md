# Contributing to Penrose

Thanks for your interest. Penrose is an open **research referee** for quantitative performance claims — it
evaluates whether a strategy's evidence holds up, with explicit multiple-testing and power accounting. It is
**not** a trading system and does not generate or hold positions.

## Project intent

Penrose is open-sourced research-first: the goal is a **transparent, calibrated, reproducible** standard for
evaluating quantitative claims that others can adopt, audit, and extend. We welcome contributions that make
the referee more rigorous or more broadly usable — and we are deliberately conservative about anything that
weakens a control to make a result look better.

The two highest-value contribution surfaces:

1. **Data adapters** — point-in-time, leakage-safe sources for new domains (equities, futures, FX, macro).
   One reference adapter ships; the contract is defined in `src/penrose/data/contract.py`.
2. **Strategy modules** — reviewed, deterministic implementations of strategy classes the pipeline can route
   claims to.

Also welcome: additional calibration controls, robustness gates, documentation, and bug fixes.

## Non-negotiables (these are the point of the project)

- **Never weaken a gate or a test to make a result pass.** If a change breaks a calibration or evaluation
  invariant, the change is wrong, not the gate. The eval suite (`python scripts/eval_suite.py`) and the
  placebo (`python scripts/calibration_placebo.py`) must stay green.
- **Discovery and confirmation stay separated.** Nothing on the discovery side may read reserved/confirmation
  data. PRs that cross this firewall will be declined.
- **Determinism.** Reproducibility is a feature. Avoid nondeterministic ordering, unseeded randomness, or
  wall-clock dependence in evaluation paths.
- **No alpha claims.** Penrose evaluates claims; it does not assert profitability. Keep language to verdicts
  (`kill` / `underpowered` / `watch` / `research-supported`) — never "alpha" or "profitable."

## Getting started

```bash
git clone <repo>
cd penrose
pip install -e .
# optional in-process embeddings (vector retrieval; lexical fallback works without it):
pip install -e '.[embed]'

python scripts/eval_suite.py                          # 82/82 invariants
python scripts/calibration_placebo.py                 # placebo: no no-edge signal certified
python scripts/worked_example_process_conditional.py  # the process-conditional verdict demo
python -m pytest -q
```

No API key, network, or external service is required for the test suite, the calibration scripts, or the
worked example. A language-model key is only needed for the optional ingestion/generation paths.

## Pull requests

- Keep changes focused; explain *why*, not just *what*.
- Add a deterministic regression test for any bug fix or new gate.
- Keep the full gate green (eval + placebo + pytest).
- Be explicit if a change touches verdict logic — those PRs get the most scrutiny.

## License

By contributing, you agree your contributions are licensed under the repository's LICENSE.
