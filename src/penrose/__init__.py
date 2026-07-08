"""penrose — a falsification-first research pipeline for quantitative trading claims.

This is the v1 cold-start build (design source of truth: README.md).
It ingests a source (paper/chat/dream), extracts per-claim theses, runs the
cheap-kill filters, generates and runs a backtest module, and proposes
kill/underpowered/watch/research-supported verdicts plus principles — which a human commits
through the review gate. Backtests propose; humans commit.
"""
__version__ = "0.6.0"
