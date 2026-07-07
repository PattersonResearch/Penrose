# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, and similar) working in this repository. Humans:
see [README.md](README.md) for the project overview and [CONTRIBUTING.md](CONTRIBUTING.md) for the
contribution rules. This file is the short, operational version for agents.

## What this project is

Penrose is a **falsification referee for quantitative trading claims**. It ingests a claim (a paper, a
thesis, a code-complete strategy, or a machine-generated hypothesis), reconstructs it in a sandbox,
runs it through a robustness and power stack, validates its own detector, and returns a calibrated
verdict. It is **not** a backtester and **not** an alpha generator, and it makes **no claim that any
strategy is profitable**. DSR deflation scales with the search Penrose has actually seen, floored by
a conservative effective-trials prior for external claims (config.DEFLATION_PRIOR). See
[docs/GATES.md](docs/GATES.md) for every gate in plain language.

## Operating Penrose (running a claim)

To *run* Penrose on a claim (rather than work on its code), the loop is: shape a falsifiable claim,
submit it, read the verdict, stop at the human gate. Full walkthrough: [docs/OPERATING.md](docs/OPERATING.md).

The minimum:

1. **Install** with `pip install -e .` (the core runs keyless; ingesting a paper needs a model). The
   model seam is one OpenAI-compatible endpoint configured by env: `PENROSE_LLM_API_KEY`,
   `PENROSE_LLM_BASE_URL` (any compatible provider), and `PENROSE_LLM_DEFAULT_MODEL` (defaults to
   `glm-5.2` — cheap enough to run the whole pipeline unattended). No code change to swap providers.
2. **Submit a claim** through one of three equivalent, guarded doors:
   - `penrose run --claims claims.json` — already-structured claims (skips lossy prose re-extraction).
   - `penrose run --paper path.pdf` — let P2 extract the claims from a source.
   - the MCP `penrose_run_claim` tool, started with `PENROSE_MCP_MANAGEMENT=1`, for an external agent.
3. **Read the result honestly.** A verdict is `kill` / `underpowered` / `watch` / `research-supported`;
   a routing state (`needs_data`, `pending_module`, `cannot_replicate`, `engine_error`, `off_domain`)
   is an honest stop, not a failure. Never fabricate a verdict or a number.
4. **Never cross P9.** Promoting a survivor into the approved corpus is a human decision; no agent
   surface can do it, and you must not try.

## Setup and the green bar

```bash
pip install -e .                              # editable install; Penrose runs scripts from the clone
python scripts/eval_suite.py                  # invariant suite — must print 101/101 passed
python -m pytest -q                           # unit tests — must stay green
python scripts/calibration_placebo.py         # placebo: no no-edge signal may be certified
```

`make` targets wrap the common flows (`make help`). The core (`eval`, `calib-*`, `connections`) runs
with no API key and no network. Paths that call a language model (paper ingestion, `dream`,
`synthesize`) need `PENROSE_LLM_API_KEY`; they fail with a clear message, never a crash, when it is
unset.

**A change is not done until the green bar above is still green.** Run it before you report success.

## Where things live

- `src/penrose/` — the library. `pipeline/` holds the stages (ingest, screen, reconstruct, backtest,
  robustness, verdict); `cli.py` is the `penrose` entry point; `config.py` holds thresholds and paths;
  `brain.py` / `brainstore.py` are the corpus store; `dream.py` / `synthesize.py` / `confirmation.py`
  are the generator and confirmation paths; `llm/` is the model client.
- `scripts/` — runnable entry points: `eval_suite.py` (the invariant suite), `calibration_*.py` (the
  self-calibration controls), `worked_example_*.py`, the literature referees (`cz_*.py`,
  `rdagent_referee.py`), and `brain_connections.py`.
- `tests/` — pytest suite. Add a deterministic regression test for any bug fix or new gate.
- `dashboard/` — local researcher UI (`index.html`, `live_server.py`, `write_api.py`) and Pennie, the
  chat assistant.
- `modules/` — reviewed, deterministic strategy modules the pipeline can route claims to.
- `docs/` — `GATES.md` (plain-language gate reference) and assets.

## Non-negotiable invariants (do not violate these)

1. **Never weaken a gate or a test to make a result pass.** If a change breaks a calibration or
   evaluation invariant, the change is wrong, not the gate.
2. **Keep discovery and confirmation separated.** Nothing on the discovery side may read the
   reserved/confirmation data. The single-use locked holdout confirms a distinct claim exactly once
   and then burns for that claim. Do not add a path that reads it during discovery or triage.
3. **Determinism.** Seed all randomness explicitly. No wall-clock dependence, no unseeded RNG, no
   nondeterministic ordering in evaluation paths.
4. **No alpha claims.** The verdict vocabulary is `kill` / `underpowered` / `watch` /
   `research-supported` (plus routing states). Never emit "profitable", "tradeable", or "alpha" as a
   verdict, and never describe a result that way.
5. **Fail gracefully.** Every path that can hit missing data, a missing key, an empty corpus, or a
   degenerate input must produce a clear, user-facing message, never a raw traceback.

## Conventions

- Match the surrounding code's style, naming, and comment density.
- New gates and calibration controls must ship with a deterministic test and keep `eval_suite.py`
  green. Verdict-logic changes get the most scrutiny; call them out explicitly.
- The corpus of invalidations may **inform** a human; it never **gates** a verdict automatically.
- This is a `0.x` research prototype: interfaces may change. Prefer additive, reversible changes.
