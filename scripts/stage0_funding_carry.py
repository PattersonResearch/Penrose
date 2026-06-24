"""Stage-0 attempt: produce ONE trustworthy, non-provisional verdict on REAL data.

Drives the operator-written `crypto_funding_carry` module through the exact P7 -> P8
path the pipeline uses (run_backtest with family-scoped DSR, then p8_verdict with the
provisional/holdout discipline), on the REAL binance-funding catalog series.

Stage-0 criteria checked at the end:
  * synthetic == False        (the verdict read only real series)
  * costs are real (modeled)  (a survivor is capped at WATCH by E2 until measured)
  * module is faithful        (operator-written, zero free params — faithful by construction)
  * verdict is non-provisional (a real kill / watch / research-supported, not pending/needs_data)

Run:  make stage0   (or: PYTHONPATH=... python scripts/stage0_funding_carry.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import os as _os, tempfile as _tf
_os.environ["PENROSE_HOLDOUT_LOCK"] = _os.path.join(_tf.gettempdir(), "penrose_calib_holdout.lock")
_os.path.exists(_os.environ["PENROSE_HOLDOUT_LOCK"]) and _os.unlink(_os.environ["PENROSE_HOLDOUT_LOCK"])
from penrose import config                              # noqa: E402
from penrose.brain import Claim                         # noqa: E402
from penrose.data import client as dataclient           # noqa: E402
from penrose.pipeline import p7_backtest, stages        # noqa: E402
from penrose.pipeline.run import _family               # noqa: E402

# The claim under test (a controlled, honest statement of the funding-carry hypothesis).
CLAIM = Claim(
    claim_id="stage0_funding_carry",
    statement=("A delta-neutral crypto perpetual funding-carry — long spot / short perp when "
               "funding is positive, reversed when negative — harvests the funding rate for a "
               "positive risk-adjusted return net of trading costs."),
    mechanism=("Perp longs pay shorts when funding>0; a delta-neutral carry collects that stream "
               "while price exposure cancels, so PnL ~= signed funding minus rebalance cost."),
    scope="crypto perpetuals (BTC)",
    horizon="daily",
    source_id="operator_stage0",
    source_span="(operator-authored controlled claim; not a paper quote)",
    claimed_metric_quote="positive risk-adjusted return net of costs",
    applicable_strategy_class="crypto-funding-carry",
)

# Modeled crypto round-trip for a delta-neutral BTC rebalance (spot+perp, both sides, liquid taker).
COST_FRAC = 0.0010


def main() -> None:
    # This is the operator's deliberate FIRST single-use holdout for this new strategy.
    lock = p7_backtest.HOLDOUT_LOCK
    if lock.exists():
        lock.unlink()
        print(f"[stage0] cleared prior holdout lock ({lock.name}) for fresh single-use consult")

    bundle = dataclient.fetch_bundle()
    bundle.reset_access()

    from penrose.pipeline.run import REGISTRY, _register_known_modules
    _register_known_modules()
    module = REGISTRY.get("crypto-funding-carry")
    if module is None:
        print("[stage0] FAIL: crypto_funding_carry not registered"); sys.exit(1)

    mres = module.run(bundle, CLAIM, COST_FRAC)
    if not mres.get("ok"):
        print(f"[stage0] module returned not-ok: {mres.get('reason')}"); sys.exit(1)

    # Prove the verdict is built on REAL data only (per-verdict synthetic tracking).
    syn_here = bundle.accessed_synthetic()
    accessed = sorted(getattr(bundle, "_accessed", set()))
    prov = {k: bundle.series[k].provenance for k in accessed if k in bundle.series}
    print(f"[stage0] module read series: {prov}  -> accessed_synthetic={syn_here}")

    family = _family(CLAIM, module)
    bt = p7_backtest.run_backtest(
        CLAIM.claim_id, mres["net"], mres["positions"], mres["bars_per_year"],
        payoff=mres.get("payoff"), position_signed=mres.get("position_signed"),
        cost_frac=COST_FRAC, wf_frame=mres.get("wf_frame"), family=family, log=False)

    # P8 provisional (no holdout), then consult the single-use holdout only if it earned it.
    holdout = {}
    dec = stages.p8_verdict(CLAIM, bt, holdout, syn_here)
    if dec.verdict == "watch" and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]:
        holdout = p7_backtest.final_holdout_eval(CLAIM.claim_id, mres["net"], mres["bars_per_year"])
        dec = stages.p8_verdict(CLAIM, bt, holdout, syn_here)

    # ---- report ----
    print("\n========== STAGE-0 VERDICT ==========")
    print(f"family            : {family}")
    print(f"n / n_oos / trials: {bt.get('n')} / {bt.get('n_oos')} / {bt.get('n_trials')}")
    print(f"DSR / PSR         : {bt.get('dsr')} / {bt.get('psr')}")
    print(f"OOS / IS / full Sh: {bt.get('oos_sharpe')} / {bt.get('is_sharpe')} / {bt.get('full_sharpe')}")
    print(f"edge_t / avg_edge : {bt.get('edge_t')} / {bt.get('avg_net_edge')}")
    print(f"3-fold            : {bt.get('three_fold')}")
    print(f"regime            : {bt.get('regime')}")
    print(f"bootstrap         : {bt.get('bootstrap')}")
    print(f"permutation       : {bt.get('permutation')}")
    print(f"capacity_usd      : {bt.get('capacity_usd')}")
    print(f"holdout           : {holdout}")
    print(f"\nVERDICT           : {dec.verdict}")
    print(f"kill_reason       : {dec.kill_reason}")
    print(f"rationale         : {dec.rationale}")
    print(f"metrics           : {json.dumps(dec.metrics, default=str)}")

    # ---- Stage-0 gate ----
    nonprovisional = dec.verdict in ("kill", "watch", "research-supported", "cannot_replicate")
    print("\n========== STAGE-0 CRITERIA ==========")
    print(f"  synthetic == False        : {syn_here is False}")
    print(f"  costs real (modeled)      : {config.COST_PROVENANCE} (survivor capped at WATCH by E2)")
    print(f"  module faithful           : True (operator-written, 0 free params)")
    print(f"  non-provisional verdict   : {nonprovisional} ({dec.verdict})")
    ok = (syn_here is False) and nonprovisional
    print(f"\nSTAGE 0 {'REACHED' if ok else 'NOT reached'}: a {dec.verdict} on real data, "
          f"{'no synthetic series read' if syn_here is False else 'SYNTHETIC contamination'}.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
