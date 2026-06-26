# CLAUDE.md

This project's agent guidance lives in [AGENTS.md](AGENTS.md). Please read it first.

It covers what Penrose is, how to set up and run the green bar (`pip install -e .`,
`python scripts/eval_suite.py` must print 93/93, `python -m pytest -q`), where things live, and the
non-negotiable invariants (never weaken a gate, keep discovery and confirmation separated, determinism,
no alpha claims, fail gracefully).
