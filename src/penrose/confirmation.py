"""Independent confirmation firewall for frozen synthesized candidates.

Each candidate receives a distinct, explicitly configured reserve epoch. The reserve bundle is
loaded only here, is never exposed to synthesis, and is passed into the unchanged P3-P9 pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config
from .brain import Claim
from .corpus import configured_reserve_epochs, provenance_checkable, reserve_intersects
from .dream import _atomic_json, _read_jsonl


def validate_firewall(candidate: dict, epoch: dict | None = None) -> tuple[bool, str]:
    epochs = [epoch] if epoch else configured_reserve_epochs()
    if not epochs:
        return False, "confirmation reserve is unconfigured; distinct data epochs are required"
    provenance = candidate.get("data_provenance") or {}
    if not provenance_checkable(provenance, {"epochs": epochs}):
        return False, "refused: candidate provenance is incomplete for reserve comparison"
    for item in epochs:
        if reserve_intersects(provenance, {"epochs": [item]}):
            return False, (
                f"refused: candidate/corpus provenance intersects confirmation epoch "
                f"{item.get('epoch_id')}")
    return True, "ok"


def _load(run_id: str) -> tuple[Path, dict, list[dict]]:
    root = config.SYNTHESIS_ARCHIVES / run_id
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(
            f"synthesis run not found: '{run_id}'. No registered run with that id "
            f"(looked in {root}). List runs under {config.SYNTHESIS_ARCHIVES} or "
            "create one first with `penrose synthesize`.")
    manifest = json.loads((root / "manifest.json").read_text())
    return root, manifest, _read_jsonl(root / "candidates.normalized.jsonl")


def _claim(run_id: str, row: dict, epoch: dict, whole_search: int) -> Claim:
    raw = row.get("raw") or {}
    return Claim(
        claim_id=f"{row['claim_id']}-confirm-{epoch['epoch_id']}",
        statement=str(raw.get("statement", "")),
        mechanism=str(raw.get("mechanism", "")),
        scope=str(raw.get("scope", "")),
        horizon=str(raw.get("horizon", "")),
        source_id=f"{run_id}-confirmation-{epoch['epoch_id']}",
        source_span=str(raw.get("statement", "")),
        claimed_metric_quote="",
        applicable_strategy_class=str(raw.get("strategy_class", "synthesized-family")),
        source_type="confirmation",
        search_cohort_id=run_id,
        search_denominator=whole_search,
        raw_hypothesis_id=row.get("raw_hypothesis_id"),
        data_provenance={
            "confirmation_reserve_id": config.CONFIRMATION_RESERVE.get("reserve_id"),
            "confirmation_epoch_id": epoch["epoch_id"],
            "periods": [{"start": epoch["start"], "end": epoch["end"]}],
            "data_domains": epoch.get("data_domains", []),
            "datasets": epoch.get("datasets", []),
            "discovery_search_denominator": whole_search,
        },
    )


def _restrict_bundle(bundle, epoch: dict):
    """Enforce an optional dataset-level reserve allowlist in addition to the time window."""
    allowed = set(map(str, epoch.get("datasets") or []))
    if not allowed:
        return bundle
    bundle.series = {k: v for k, v in bundle.series.items() if k in allowed}
    return bundle


def confirm_run(run_id: str) -> dict:
    root, manifest, rows = _load(run_id)
    result_path = root / "confirmation.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if manifest.get("status") not in {"complete", "generated_only", "no_admitted_candidates"}:
        return {"status": "refused", "reason": "discovery run is not frozen/complete"}
    admitted = [r for r in rows if r.get("admitted")]
    if not admitted:
        return {"status": "refused", "reason": "no frozen admitted candidates"}
    epochs = configured_reserve_epochs()
    if len(epochs) < len(admitted):
        return {
            "status": "refused",
            "reason": "not enough independent confirmation epochs",
            "candidates": len(admitted),
            "epochs": len(epochs),
        }

    failures = []
    assignments = []
    for row, epoch in zip(admitted, epochs):
        provenance = row.get("data_provenance") or {}
        ok, reason = validate_firewall({"data_provenance": provenance}, epoch)
        if not ok:
            failures.append({"claim_id": row.get("claim_id"), "epoch_id": epoch["epoch_id"],
                             "reason": reason})
        assignments.append((row, epoch))
    if failures:
        return {"status": "refused",
                "reason": "confirmation firewall rejected candidate provenance",
                "failures": failures}

    whole_search = max(int(manifest.get("generation_budget", 0)),
                       int(manifest.get("candidates_generated", 0)))
    source = root / "source.md"
    if not source.exists():
        return {"status": "refused", "reason": f"frozen synthesis source missing: {source}"}

    from .data import client as dataclient
    from .pipeline.run import run_source

    results = []
    for row, epoch in assignments:
        checkpoint = root / "confirmation_results" / f"{epoch['epoch_id']}.json"
        if checkpoint.exists():
            results.append(json.loads(checkpoint.read_text()))
            continue
        # This is the data firewall: confirmation explicitly loads a held-aside time window and
        # injects that bundle into the Referee. Discovery's default bundle is never fetched here.
        try:
            bundle = dataclient.fetch_bundle(start=epoch["start"], end=epoch["end"])
        except Exception as e:  # fail-soft, but never substitute discovery data
            return {"status": "refused", "reason":
                    f"confirmation epoch {epoch['epoch_id']} could not load: "
                    f"{type(e).__name__}: {e}"}
        if tuple(map(str, bundle.requested_window)) != (epoch["start"], epoch["end"]):
            return {"status": "refused",
                    "reason": f"confirmation loader returned wrong window for {epoch['epoch_id']}"}
        bundle = _restrict_bundle(bundle, epoch)
        if epoch.get("datasets") and not bundle.series:
            return {"status": "refused",
                    "reason": f"confirmation epoch {epoch['epoch_id']} loaded none of its "
                              "reserved datasets"}

        claim = _claim(run_id, row, epoch, whole_search)
        lock = root / "confirmation_locks" / f"{epoch['epoch_id']}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists():
            return {"status": "refused",
                    "reason": f"confirmation epoch {epoch['epoch_id']} was already consumed "
                              "without a completed checkpoint; rotate the reserve"}
        old_lock = os.environ.get("PENROSE_HOLDOUT_LOCK")
        old_mode = os.environ.get("PENROSE_HOLDOUT_MODE")
        os.environ["PENROSE_HOLDOUT_LOCK"] = str(lock)
        os.environ.pop("PENROSE_HOLDOUT_MODE", None)
        try:
            try:
                result = run_source(
                    source, use_llm=True, claims_override=[claim],
                    source_type="confirmation", bundle_override=bundle)
            except Exception as e:
                consumed = lock.exists()
                failure = {
                    "status": "refused",
                    "reason": (
                        f"confirmation epoch {epoch['epoch_id']} failed after its holdout "
                        "was consumed; rotate the reserve"
                        if consumed else
                        f"confirmation epoch {epoch['epoch_id']} failed before holdout "
                        "consumption and may be retried"
                    ),
                    "epoch_id": epoch["epoch_id"],
                    "holdout_consumed": consumed,
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
                _atomic_json(
                    root / "confirmation_failures" / f"{epoch['epoch_id']}.json", failure)
                return failure
        finally:
            if old_lock is None:
                os.environ.pop("PENROSE_HOLDOUT_LOCK", None)
            else:
                os.environ["PENROSE_HOLDOUT_LOCK"] = old_lock
            if old_mode is None:
                os.environ.pop("PENROSE_HOLDOUT_MODE", None)
            else:
                os.environ["PENROSE_HOLDOUT_MODE"] = old_mode
        item = {"claim_id": claim.claim_id, "epoch_id": epoch["epoch_id"],
                "window": [epoch["start"], epoch["end"]],
                "datasets": epoch.get("datasets", []), "pipeline_run": result}
        _atomic_json(checkpoint, item)
        results.append(item)

    payload = {
        "status": "complete",
        "reserve_id": config.CONFIRMATION_RESERVE.get("reserve_id"),
        "whole_search_denominator": whole_search,
        "candidate_results": results,
    }
    _atomic_json(result_path, payload)
    return payload
