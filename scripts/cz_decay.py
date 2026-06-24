"""Post-publication decay analysis on the Chen-Zimmermann anomalies — and the NOVEL question:
do penrose's SURVIVORS decay LESS than the ones it killed?

McLean & Pontiff (2016) showed published anomalies lose ~26% of their return out-of-sample (after
the original study's sample ends) and ~58% post-publication — strong evidence that much "alpha" is
in-sample mining + arbitraged-away once public. We reproduce that AND connect it to penrose's
verdicts: if penrose's `research-supported`/`watch` survivors decay LESS than its kills, then
penrose's deflated power-aware verdict is identifying DURABLE structure, not just long-sample luck.
If they decay the same, that's the honest, humbling finding (even survivors aren't immune).

Per anomaly we split the monthly LS returns into:
  in-sample      : up to SampleEndYear (the original study's window)
  post-sample    : SampleEndYear .. publication Year (out-of-sample, pre-publication)
  post-pub       : after publication Year
and compare mean monthly return across periods, grouped by penrose's verdict.

Run:  python scripts/cz_decay.py
"""
from __future__ import annotations

import os as _os
import sys
import tempfile as _tf
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")

from penrose import config                            # noqa: E402
config.COST_PROVENANCE = "measured"
from penrose.pipeline import p7_backtest as P7, stages  # noqa: E402
from penrose.brain import Claim                       # noqa: E402

LIT = Path.home() / "Development/penrose-data/literature/chen_zimmermann"
FAMILY = "cz_literature::equity_xs"


def _claim(nm):
    return Claim(claim_id=nm, statement="", mechanism="", scope="", horizon="",
                 source_id="cz", source_span="", claimed_metric_quote="", applicable_strategy_class="cz")


def main() -> None:
    if not (LIT / "ls_panel.parquet").exists():
        print(f"[cz-decay] Chen-Zimmermann data not found at {LIT}")
        print("           This experiment needs the published-anomaly panel, which is a separate")
        print("           data download. See notebooks/penrose_demo.ipynb for how to obtain it.")
        sys.exit(1)
    panel = pd.read_parquet(LIT / "ls_panel.parquet") / 100.0
    doc = pd.read_parquet(LIT / "signal_doc.parquet").set_index("Acronym")
    panel.index = pd.to_datetime(panel.index)
    names = [c for c in panel.columns if c in doc.index and panel[c].notna().sum() >= 60
             and pd.notna(doc.loc[c, "Year"]) and pd.notna(doc.loc[c, "SampleEndYear"])]
    print(f"[cz-decay] {len(names)} anomalies with publication metadata + >=60 months.\n")

    # penrose verdict per anomaly (deflated across the full search, as in cz_referee)
    tmp = Path("/tmp/_czd.tsv"); old = P7.LEDGER; P7.LEDGER = tmp; tmp.unlink(missing_ok=True)
    verdict = {}
    try:
        for nm in names:
            net = panel[nm].dropna()
            P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), 12.0, cost_frac=0.0, family=FAMILY, log=True)
        for nm in names:
            net = panel[nm].dropna()
            bt = P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), 12.0, cost_frac=0.0, family=FAMILY, log=False)
            d = stages.p8_verdict(_claim(nm), bt, {}, False)
            if d.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
                ho = P7.final_holdout_eval(nm, net, 12.0, force=True); d = stages.p8_verdict(_claim(nm), bt, ho, False)
            verdict[nm] = d.verdict
    finally:
        P7.LEDGER = old; tmp.unlink(missing_ok=True)

    # decay per anomaly: mean monthly return in each period
    rows = []
    for nm in names:
        s = panel[nm].dropna()
        end_y, pub_y = int(doc.loc[nm, "SampleEndYear"]), int(doc.loc[nm, "Year"])
        ins = s[s.index.year <= end_y]
        post_samp = s[(s.index.year > end_y) & (s.index.year <= pub_y)]
        post_pub = s[s.index.year > pub_y]
        if len(ins) < 24 or len(post_pub) < 24:
            continue
        rows.append({"name": nm, "verdict": verdict.get(nm, "?"),
                     "ins_mean": ins.mean(), "postsamp_mean": post_samp.mean() if len(post_samp) else np.nan,
                     "postpub_mean": post_pub.mean(),
                     "decay_pct": 100 * (1 - post_pub.mean() / ins.mean()) if ins.mean() != 0 else np.nan})
    df = pd.DataFrame(rows)

    print("=== POST-PUBLICATION DECAY (mean monthly LS return, % decay = 1 - postpub/insample) ===")
    print(f"  N anomalies with clean pre/post split: {len(df)}")
    print(f"  in-sample mean monthly:   {df['ins_mean'].mean()*100:.3f}%")
    print(f"  post-publication mean:    {df['postpub_mean'].mean()*100:.3f}%")
    overall = 100 * (1 - df['postpub_mean'].mean() / df['ins_mean'].mean())
    print(f"  OVERALL decay:            {overall:.0f}%   (McLean-Pontiff found ~58% post-publication)")

    print("\n=== THE NOVEL QUESTION: do penrose's SURVIVORS decay less than its kills? ===")
    grp = {"survivors (watch/research-supported)": ["watch", "research-supported"],
           "underpowered": ["underpowered"], "killed": ["kill"]}
    for label, vs in grp.items():
        sub = df[df["verdict"].isin(vs)]
        if len(sub) == 0:
            print(f"  {label:42s} n=0"); continue
        dec = 100 * (1 - sub['postpub_mean'].mean() / sub['ins_mean'].mean()) if sub['ins_mean'].mean() else np.nan
        print(f"  {label:42s} n={len(sub):3d}  in-samp={sub['ins_mean'].mean()*100:.3f}%  "
              f"post-pub={sub['postpub_mean'].mean()*100:.3f}%  decay={dec:.0f}%")

    surv = df[df["verdict"].isin(["watch", "research-supported"])].sort_values("postpub_mean", ascending=False)
    print("\n  penrose's survivors, individually (do they hold up post-publication?):")
    for _, r in surv.iterrows():
        print(f"    {r['name']:16s} in-samp={r['ins_mean']*100:.2f}%  post-pub={r['postpub_mean']*100:.2f}%  "
              f"decay={r['decay_pct']:.0f}%  ({r['verdict']})")
    print("\nReading: if survivors decay LESS than kills, penrose's deflated verdict identifies durable")
    print("structure (not long-sample luck). If they decay the same, even survivors obey McLean-Pontiff")
    print("— an honest result: a falsifier rejects the fragile, but cannot make a real edge immune to decay.")


if __name__ == "__main__":
    main()
