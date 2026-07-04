"""Deterministic honesty calibration for the Penrose concept/synthesis/confirmation loop."""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from penrose import config  # noqa: E402
from penrose.brain import Claim  # noqa: E402
from penrose.concepts import extract  # noqa: E402
from penrose.confirmation import validate_firewall  # noqa: E402
from penrose.corpus import build  # noqa: E402
from penrose.dream import create_manifest, record_candidates, register_search  # noqa: E402
from penrose.pipeline import p7_backtest as p7, stages  # noqa: E402
from penrose.synthesize import normalize  # noqa: E402


def _noise_concepts() -> list[dict]:
    rows = []
    for domain, family in (("noise-a", "noise-carry"), ("noise-b", "noise-yield")):
        for i in range(3):
            rows.append({
                "concept_id": f"{domain}-{i}", "source_claim_id": f"{domain}-{i}",
                "statement": "random observations happened to be positive",
                "surviving_explanation": "", "evidence_direction": "positive",
                "created_at": "2026-01-01T00:00:00+00:00",
                "data_provenance": {
                    "data_domain": domain, "strategy_family": family,
                    "datasets": [f"{domain}-discovery"],
                    "periods": [{"start": "2020-01-01", "end": "2022-12-31"}],
                },
            })
    return rows


def _placebo_loop() -> tuple[int, bool]:
    """Run noise through corpus -> synthesis normalization/registration -> firewall -> P7/P8."""
    graph = build(_noise_concepts(), min_support=3, current_year=2026)
    mechanisms = [n for n in graph["nodes"] if n.get("level") == "cross_family_mechanism"]
    if not mechanisms:
        return -1, False
    raw = [{
        "statement": "Random noise predicts future returns",
        "mechanism": "placebo only", "scope": "synthetic", "horizon": "1d",
        "strategy_class": "noise-placebo", "candidate_class": "testable_now",
        "inspired_by": [mechanisms[0]["node_id"]],
        "falsifier": "no positive OOS evidence",
    }]
    claims, normalized = normalize("calib-noise", raw, {"nodes": graph["nodes"]})
    if not normalized[0]["admitted"]:
        return -1, False

    with tempfile.TemporaryDirectory(prefix="penrose-calib-") as td:
        root = Path(td)
        old_ledger = p7.LEDGER
        p7.LEDGER = root / "ledger.tsv"
        try:
            manifest = create_manifest(
                run_id="calib-noise", generation_budget=1, model="deterministic",
                corpus_snapshot_hash="noise", root=root / "run")
            manifest = record_candidates(manifest, raw)
            manifest = register_search(manifest, claims, normalized)

            epoch = {"epoch_id": "noise-confirm-1", "start": "2024-01-01",
                     "end": "2025-12-31", "data_domains": ["noise-confirm"],
                     "datasets": ["noise-confirm-reserve"]}
            allowed, _ = validate_firewall(
                {"data_provenance": normalized[0]["data_provenance"]}, epoch)
            if not allowed:
                return -1, False

            rng = np.random.default_rng(20260622)
            idx = pd.date_range("2024-01-01", periods=240, freq="D")
            net = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
            bt = p7.run_backtest(
                "calib-noise-confirm", net, pd.Series(1.0, index=idx), 365.0,
                family="generated::noise", generation_source="synthesized",
                search_cohort_id="calib-noise",
                search_denominator=manifest["effective_search_denominator"])
            confirmation_claim = Claim(
                claim_id="calib-noise-confirm", statement=raw[0]["statement"],
                mechanism=raw[0]["mechanism"], scope="synthetic", horizon="1d",
                source_id="calib", source_span=raw[0]["statement"], claimed_metric_quote="",
                applicable_strategy_class="noise-placebo", source_type="confirmation",
                search_cohort_id="calib-noise",
                search_denominator=manifest["effective_search_denominator"])
            decision = stages.p8_verdict(confirmation_claim, bt, {}, synthetic=False)
            survived = int(decision.verdict == "research-supported")
            return survived, True
        finally:
            p7.LEDGER = old_ledger


def main() -> None:
    survived, loop_ran = _placebo_loop()

    killed = extract({
        "claim_id": "known-kill", "statement": "noise has a durable pattern",
        "verdict": "kill", "kill_reason": "low_edge_t", "run_at": "2026-01-01",
        "metrics": {"dsr": 0.1, "n_oos": 200},
        "competing_explanations": [{
            "explanation": "noise is a supported mechanism", "verdict": "survives"}],
    }, use_llm=False)
    fidelity_ok = killed is not None and killed.surviving_explanation == ""

    epoch = {"epoch_id": "r", "start": "2024-01-01", "end": "2025-01-01",
             "data_domains": ["reserved"], "datasets": []}
    allowed, reason = validate_firewall(
        {"data_provenance": {"data_domains": ["reserved"],
                             "periods": [{"start": "2020-01-01", "end": "2021-01-01"}]}},
        epoch)
    firewall_ok = not allowed and "intersects" in reason

    rows = [
        ("synthesizer placebo: full deterministic loop ran", loop_ran),
        ("synthesizer placebo: noise confirmation survivors = 0", survived == 0),
        ("concept fidelity: kill cannot claim a surviving supported mechanism", fidelity_ok),
        ("confirmation firewall: reserve-touching provenance is refused", firewall_ok),
    ]
    for name, ok in rows:
        print(f"{name}: {'OK' if ok else 'FAIL'}")
    print(f"placebo survivors: {survived}")
    raise SystemExit(0 if all(ok for _, ok in rows) else 1)


if __name__ == "__main__":
    main()
