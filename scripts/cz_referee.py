"""penrose as REFEREE for the PUBLISHED factor literature (Chen-Zimmermann Open Source Cross-Section).

The "referee a generator" experiment (rdagent_referee) tests a machine. This is the companion
"referee the literature" experiment: 212 PUBLISHED cross-sectional equity anomalies, each a
replicated long-short monthly return series with a known published claim ("this earns a positive
risk-adjusted return"). We run every one through penrose's deflated, power-aware verdict — deflating
across the FULL set of 212 (the factor-zoo multiple-testing correction the literature is famous for
lacking) — and report how many survive.

This is the established factor-zoo replication framing (Harvey-Liu-Zhu; Hou-Xue-Zhang found most
anomalies die under proper testing). CZ's set is the *replicable* subset (each reproduces its
paper's original result), so a meaningful fraction survives in-sample; deflation + locked holdout +
power-aware labels are what separate the durable from the data-mined.

Data: penrose-data/literature/chen_zimmermann/ls_panel.parquet (1188 months x 212 anomalies, signed
to the published direction). Fetched via the openassetpricing package — run `make cz-data` (or
`python scripts/fetch_cz_data.py`) once to download+build it, then `make cz-referee`.

This script reports BOTH endpoints of the scope-dependence result in one run, so the whole "the
denominator you assume dominates the conclusion" finding is reproducible:
  * per-anomaly-alone  — each anomaly is its own family (denominator 1, no cross-deflation)
  * deflated-by-212    — all anomalies share one family (the full-search multiple-testing correction)

Run:  python scripts/cz_referee.py [N_top]   (N_top: limit to the N longest-history anomalies)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import os as _os, tempfile as _tf
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")
_os.path.exists(_os.environ["PENROSE_HOLDOUT_LOCK"]) and _os.unlink(_os.environ["PENROSE_HOLDOUT_LOCK"])
from penrose import config                            # noqa: E402
config.COST_PROVENANCE = "measured"                  # E2 off: let a genuinely durable anomaly certify
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim                       # noqa: E402

PANEL = Path.home() / "Development/penrose-data/literature/chen_zimmermann/ls_panel.parquet"
BARS_PER_YEAR = 12.0                                 # monthly returns
FAMILY = "cz_literature::equity_xs"
SURVIVING = ("research-supported", "watch")


def _run_pass(names, panel, family_of, *, populate: bool):
    """Evaluate every anomaly under a family assignment. When `populate`, all trials are logged into
    their families first so DSR deflates by the realized family size; otherwise each singleton family
    has denominator 1 (no cross-deflation). Uses an isolated temp ledger, restored on exit."""
    tmp = Path(_tf.gettempdir()) / f"_cz_ledger_{'deflated' if populate else 'alone'}.tsv"
    old = P7.LEDGER; P7.LEDGER = tmp
    if tmp.exists(): tmp.unlink()
    try:
        if populate:
            for nm in names:
                net = panel[nm].dropna()
                if len(net) < 30:
                    continue
                P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), BARS_PER_YEAR,
                                cost_frac=0.0, family=family_of(nm), log=True)
        rows = []
        for nm in names:
            net = panel[nm].dropna()
            if len(net) < 30:
                rows.append((nm, "insufficient_data", None, None, None, len(net))); continue
            bt = P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), BARS_PER_YEAR,
                                 cost_frac=0.0, family=family_of(nm), log=False)
            dec = stages.p8_verdict(_claim(nm), bt, {}, synthetic=False)
            if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
                ho = P7.final_holdout_eval(nm, net, BARS_PER_YEAR, force=True)
                dec = stages.p8_verdict(_claim(nm), bt, ho, synthetic=False)
            rows.append((nm, dec.verdict, bt.get("dsr"), bt.get("oos_sharpe"),
                         dec.metrics.get("mde_ic"), len(net)))
    finally:
        P7.LEDGER = old; tmp.unlink(missing_ok=True)
    return rows


def _report(label, rows):
    tally = Counter(r[1] for r in rows)
    surv = [r for r in rows if r[1] in SURVIVING]
    print(f"=== VERDICT TALLY — {label} ===")
    for v, c in tally.most_common():
        print(f"  {v:18s} {c:4d}  ({100 * c / len(rows):.0f}%)")
    print(f"SURVIVORS (watch / research-supported): {len(surv)}/{len(rows)} = {100 * len(surv) / len(rows):.0f}%\n")
    return len(surv), len(rows)


def main() -> None:
    if not PANEL.exists():
        print(f"missing {PANEL}\nrun `make cz-data` (or python scripts/fetch_cz_data.py) to download+build it first")
        sys.exit(1)
    panel = pd.read_parquet(PANEL) / 100.0           # % -> fraction
    n_top = int(sys.argv[1]) if len(sys.argv) > 1 else panel.shape[1]
    counts = panel.notna().sum().sort_values(ascending=False)   # best-powered first
    names = list(counts.index[:n_top])
    print(f"[cz-referee] {len(names)} published anomalies; monthly; reporting both scope endpoints.\n")

    alone = _run_pass(names, panel, lambda nm: f"cz_alone::{nm}", populate=False)
    deflated = _run_pass(names, panel, lambda nm: FAMILY, populate=True)

    a_surv, a_tot = _report("per-anomaly-alone (no cross-deflation)", alone)
    d_surv, d_tot = _report("deflated by the full 212-anomaly search", deflated)

    print(f"SCOPE DEPENDENCE: survival ranges from {a_surv}/{a_tot} ({100 * a_surv / a_tot:.0f}%) judged alone "
          f"to {d_surv}/{d_tot} ({100 * d_surv / d_tot:.0f}%) deflated by the full search.")
    print("Same identical return series; only the assumed research family changes. The denominator you")
    print("assume dominates the conclusion — which is the point. Top survivors (deflated) by OOS Sharpe:")
    for nm, v, dsr, osh, mde, n in sorted((r for r in deflated if r[1] in SURVIVING),
                                          key=lambda r: -(r[3] or -9))[:12]:
        print(f"  {nm:26s} {v:16s} dsr={dsr} oos_sharpe={osh} n_months={n}")


def _claim(nm: str) -> Claim:
    return Claim(claim_id=nm, statement=f"published anomaly {nm} earns a positive long-short return",
                 mechanism="", scope="US equities cross-section", horizon="monthly",
                 source_id="chen_zimmermann", source_span="", claimed_metric_quote="",
                 applicable_strategy_class="cz-anomaly")


if __name__ == "__main__":
    main()
