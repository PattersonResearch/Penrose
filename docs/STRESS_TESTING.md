# How Penrose is stress-tested

A research referee is only as trustworthy as its own discipline. Penrose is built so its checks can be
*verified*, not just asserted. This page summarizes how the system is calibrated and adversarially reviewed.
The relevant scripts are in `scripts/` and run on a clean, keyless install.

## It proves it has power before it claims anything

- **Placebo (specificity).** Pure-noise signals applied to real returns yield **zero** strongest-state
  certifications (`scripts/calibration_placebo.py`, and the five-null battery `scripts/calibration_nulls.py`).
  A referee that certifies noise is worthless; this one doesn't.
- **Injection (sensitivity).** A *planted* signal is recovered: as injected signal strength rises, verdicts
  move from `kill`/`underpowered` through `watch` to `research-supported` at the expected threshold.
- **Process-conditional verdict.** A single, byte-identical return series is certified or killed depending
  only on the declared search lineage that produced it — the multiple-testing problem made concrete and
  runnable (`scripts/worked_example_process_conditional.py`). This is the cleanest demonstration of why
  Penrose is not a backtester, and it runs with no network, key, or data.

## It is reviewed adversarially, not just written

Changes are pushed through a build → independent-review → correct loop, where reviewers (including a separate
model and direct re-execution of the real pipeline) try to *refute* each result before it is accepted.
Findings are reproduced from the code, not taken on faith; controls that turn out to be no-ops are fixed.
Several real defects — a search that silently hid committed results, a "demonstration" whose claimed verdict
did not survive the real pipeline — were caught and closed this way before release.

## What this does and does not establish

These controls establish that the referee **discriminates signal from noise, accounts for search, and does
not certify the void**. They do **not** establish that any surviving claim is profitable. A `research-supported`
verdict means "survived falsification," not "will make money" — and still requires human review. Penrose is
inspectable evaluation infrastructure, not an oracle, a publication authority, or a trading system.

Reproduce any of the above yourself: every script named here ships in the repository.
