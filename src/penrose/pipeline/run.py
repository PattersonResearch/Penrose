"""Run a paper end-to-end through P1–P9.

Backtests PROPOSE; humans COMMIT. This orchestrator runs P1–P8 and writes
proposals to the review queue + decisions log + brain-via-archives; it does NOT
promote anything to the brain as committed knowledge. Promotion happens only in
p9_review.approve(), which constructs the read-write PromotionClient.

Usage:
    python -m penrose.pipeline.run                 # scans inbox/, falls back to staged paper
    python -m penrose.pipeline.run --paper path.pdf
    python -m penrose.pipeline.run --paper path.pdf --no-llm   # force fallback (claims.py)
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import functools
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import __version__
from .. import config
from .. import regime as regime_lib
from .. import worker_control
from ..audit import AuditLog, config_fingerprint, platform_tuple
from ..brain import BrainReader, Claim, source_is_unanchored, validate_source_type
from ..data import client as dataclient
from ..report import write_report
from ..strategy_family import declared_strategy_family, normalize_strategy_family
from ..trace import project_trace_record
from . import stages, p7_backtest
# NB: claims.py is per-paper P2 output and gitignored (cold-start = none), so it is NOT imported
# here — a fresh clone has no claims.py. extract.fallback_claims loads it lazily when present.
from . import (
    p1_ingest, extract, spec_gen, impl_gen, relevance, charts, fidelity, sandbox,
    fidelity_memory, provided_series, event_market, predictive_regression, factor_spanning,
    cross_sectional_sort, event_study, forecast_skill, formulaic_signal, robustness,
)
from .human_review import human_review_explanation

MAX_CLAIM_WORKERS = worker_control.MAX_REQUESTED_WORKERS
_JSONL_DEFER = threading.local()
_JSONL_WRITE_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()
_HOLDOUT_LOCK = threading.Lock()
_REGISTRY_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _defer_jsonl_events():
    prior = getattr(_JSONL_DEFER, "events", None)
    events: list[tuple[Path, dict]] = []
    _JSONL_DEFER.events = events
    try:
        yield events
    finally:
        _JSONL_DEFER.events = prior


def _append_jsonl(path, obj) -> None:
    deferred = getattr(_JSONL_DEFER, "events", None)
    if deferred is not None:
        deferred.append((Path(path), obj))
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_WRITE_LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(obj, default=str) + "\n")


def _emit_jsonl_events(events: list[tuple[Path, dict]]) -> None:
    for path, obj in events:
        _append_jsonl(path, obj)


def _effective_traces_path() -> Path:
    default = config.ROOT / "reports" / "traces.jsonl"
    configured = Path(getattr(config, "TRACES", default))
    if configured != default:
        return configured
    if Path(config.DECISIONS_LOG) != config.ROOT / "decisions.jsonl":
        return Path(config.DECISIONS_LOG).parent / "reports" / "traces.jsonl"
    if Path(config.REPORTS) != config.ROOT / "reports":
        return Path(config.REPORTS) / "traces.jsonl"
    return configured


def _emit_trace(claim, dec, rec, run_log) -> None:
    try:
        _append_jsonl(_effective_traces_path(), project_trace_record(claim, dec, rec, run_log))
    except Exception:  # noqa: BLE001 — observability must never change control flow
        pass


def _audit_warn(exc: BaseException) -> None:
    print(f"[penrose] audit emission failed: {exc}", file=sys.stderr)


def _audit_call(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - audit must never change verdict control flow
        _audit_warn(exc)


def _audit_seeds() -> dict:
    return {
        "bootstrap": (getattr(config, "BOOTSTRAP", {}) or {}).get("seed"),
        "permutation": (getattr(config, "PERMUTATION", {}) or {}).get("seed"),
        "regime_fragility": (getattr(config, "REGIME_FRAGILITY", {}) or {}).get("seed"),
        "cpcv": (getattr(config, "CPCV", {}) or {}).get("seed"),
    }


def _audit_reproducibility_class(source_type: str) -> str:
    if source_type in {"generated_hypothesis", "synthesized_hypothesis", "confirmation"}:
        return "OPEN"
    return "GATED" if getattr(config, "DATA_DIR", None) else "OPEN"


def _audit_data_sources(run_log: dict) -> dict:
    return {
        "source_id": run_log.get("source_id"),
        "source_title": run_log.get("source_title"),
        "paper_path": run_log.get("paper_path"),
        "content_sha256": (run_log.get("idempotency") or {}).get("content_sha256"),
        "provenance": run_log.get("provenance"),
    }


def _audit_gate_events(audit_log: AuditLog, dec, rec: dict) -> None:
    metrics = getattr(dec, "metrics", {}) or {}
    p7 = ((rec or {}).get("stages") or {}).get("P7") or {}
    gates = []
    if metrics.get("dsr") is not None or metrics.get("psr") is not None:
        gates.append(("deflation", {
            "psr": metrics.get("psr"),
            "dsr": metrics.get("dsr"),
            "n_trials": metrics.get("n_trials"),
            "verdict": getattr(dec, "verdict", None),
            "kill_reason": getattr(dec, "kill_reason", None),
        }))
    if metrics.get("holdout") is not None:
        gates.append(("holdout", {
            "holdout": metrics.get("holdout"),
            "verdict": getattr(dec, "verdict", None),
        }))
    if metrics.get("power_sufficient") is not None or metrics.get("resolution") is not None:
        gates.append(("power", {
            "power_sufficient": metrics.get("power_sufficient"),
            "mde_ic": metrics.get("mde_ic"),
            "mde_sharpe_ann": metrics.get("mde_sharpe_ann"),
            "resolution": metrics.get("resolution"),
            "verdict": getattr(dec, "verdict", None),
        }))
    if metrics.get("tail") is not None or metrics.get("tail_asymmetric") is not None:
        gates.append(("tail", {
            "tail": metrics.get("tail"),
            "tail_asymmetric": metrics.get("tail_asymmetric"),
            "verdict": getattr(dec, "verdict", None),
        }))
    if metrics.get("implausible") is not None or p7.get("implausible") is not None:
        gates.append(("implausibility", {
            "implausible": metrics.get("implausible", p7.get("implausible")),
            "verdict": getattr(dec, "verdict", None),
        }))
    if metrics.get("fidelity") is not None or metrics.get("fidelity_suspect") is not None:
        gates.append(("fidelity", {
            "fidelity": metrics.get("fidelity"),
            "fidelity_suspect": metrics.get("fidelity_suspect"),
            "fidelity_unverified": metrics.get("fidelity_unverified"),
            "verdict": getattr(dec, "verdict", None),
            "kill_reason": getattr(dec, "kill_reason", None),
        }))
    for gate, detail in gates:
        _audit_call(
            audit_log.stage,
            "gate",
            "gate_outcome",
            inputs={"claim_id": getattr(dec, "claim_id", None), "gate": gate},
            outputs=detail,
            detail={"claim_id": getattr(dec, "claim_id", None), "gate": gate, **detail},
        )


def _emit_audit_run_events(audit_log: AuditLog | None, run_log: dict, decisions: list | None = None) -> None:
    if audit_log is None:
        return
    _audit_call(audit_log.stage, "P1", "enter",
                inputs={"paper_path": run_log.get("paper_path")},
                detail={"paper_path": run_log.get("paper_path")})
    if run_log.get("p1") is not None:
        _audit_call(audit_log.stage, "P1", "exit",
                    outputs=run_log.get("p1"), detail=run_log.get("p1"))
    if run_log.get("relevance") is not None:
        _audit_call(audit_log.stage, "relevance", "enter",
                    inputs={"source_id": run_log.get("source_id")},
                    detail={"source_id": run_log.get("source_id")})
        _audit_call(audit_log.stage, "relevance", "exit",
                    outputs=run_log.get("relevance"), detail=run_log.get("relevance"))
    if run_log.get("p2") is not None:
        _audit_call(audit_log.stage, "P2", "enter",
                    inputs={"source_id": run_log.get("source_id")},
                    detail={"source_id": run_log.get("source_id")})
        _audit_call(audit_log.stage, "P2", "exit",
                    outputs=run_log.get("p2"), detail=run_log.get("p2"))
    decision_by_claim = {
        getattr(dec, "claim_id", ""): dec for dec in (decisions or [])
        if getattr(dec, "claim_id", "")
    }
    for rec in run_log.get("claims", []) or []:
        claim_id = rec.get("claim_id")
        stages_seen = rec.get("stages") or {}
        if not isinstance(stages_seen, dict):
            continue
        for stage, payload in stages_seen.items():
            _audit_call(audit_log.stage, str(stage), "enter",
                        inputs={"claim_id": claim_id},
                        detail={"claim_id": claim_id})
            _audit_call(audit_log.stage, str(stage), "exit",
                        outputs=payload,
                        detail={"claim_id": claim_id, "payload": payload})
        dec = decision_by_claim.get(claim_id)
        if dec is not None:
            _audit_gate_events(audit_log, dec, rec)


def _finalize_audit_run(audit_log: AuditLog | None, audit_path: Path | None,
                        run_log: dict, decisions: list | None = None) -> None:
    if audit_log is None or audit_path is None:
        return
    try:
        _emit_audit_run_events(audit_log, run_log, decisions)
        run_log["audit_head_hash"] = audit_log.head_hash()
        run_log["audit_path"] = str(audit_path)
    except Exception as exc:  # noqa: BLE001 - audit must never change verdict control flow
        _audit_warn(exc)


def _llm_available() -> bool:
    """True iff an LLM API key is configured."""
    return bool(os.environ.get("PENROSE_LLM_API_KEY"))


def resolve_worker_count(requested: int | str | None) -> int:
    return worker_control.resolve_worker_count(requested)


def _claim_worker_resolution(value: int | str | None = None) -> worker_control.WorkerCountResolution:
    raw = os.environ.get("PENROSE_MAX_CLAIM_WORKERS") if value is None else value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        # Operator specified nothing -> conservative parallel default = min(4, auto): 4 on capable
        # hardware, auto-reduced on constrained machines so a small box is never swamped. `auto`
        # (the full hardware max, up to the cap) and an explicit `--workers N`/1 remain available.
        raw = worker_control.default_worker_count()
    return worker_control.resolve_worker_count_details(raw)


def _log_claim_worker_resolution(resolution: worker_control.WorkerCountResolution) -> None:
    print(
        "[penrose] claim workers: "
        f"resolved={resolution.count} requested={resolution.requested!r} "
        f"bound={resolution.bound} "
        f"ceilings(cpu={resolution.cpu_ceiling}, ram={resolution.ram_ceiling}, cap={resolution.hard_cap})",
        file=sys.stderr,
    )


def _isolated_bundle_access(bundle):
    """Shallow clone a bundle for one claim run, with independent access tracking.

    Series values are shared read-only; the series mapping and `_accessed` set are per-claim.
    Auto-fetch additions therefore cannot leak into another claim's provenance cap.
    """
    try:
        clone = copy.copy(bundle)
    except Exception:  # noqa: BLE001
        clone = bundle
    if hasattr(bundle, "series"):
        try:
            object.__setattr__(clone, "series", dict(getattr(bundle, "series", {}) or {}))
        except Exception:  # noqa: BLE001
            pass
    if hasattr(bundle, "fallback_substitutions"):
        try:
            object.__setattr__(
                clone, "fallback_substitutions",
                list(getattr(bundle, "fallback_substitutions", []) or []),
            )
        except Exception:  # noqa: BLE001
            pass
    for attr, value in (("_accessed", set()), ("_norm_index", None), ("_norm_index_keys", None)):
        try:
            object.__setattr__(clone, attr, value)
        except Exception:  # noqa: BLE001
            pass
    return clone


def _accessed_keys(bundle, *, conservative_for_auto: bool = False) -> list[str]:
    if conservative_for_auto:
        keys = getattr(bundle, "series", {}).keys() if hasattr(bundle, "series") else []
    else:
        keys = getattr(bundle, "_accessed", None) or []
    return sorted(str(k) for k in keys)


def _run_claim_tasks(items, worker_count: int, fn, on_error=None, governor=None):
    # P-1: a worker must NEVER propagate. In the parallel path a raised worker aborts `as_completed`
    # and discards EVERY sibling worker's deferred writes (their successful decisions vanish from the
    # ledger — worse than serial, where prior claims are already committed). `on_error` converts an
    # unguarded worker crash into a per-claim engine_error result so isolation matches the serial intent.
    def _safe(item):
        try:
            if worker_count <= 1 or governor is None:
                return fn(item)
            acquired = False
            try:
                governor.acquire()
                acquired = True
            except Exception as exc:  # noqa: BLE001
                print(f"[penrose] worker governor failed open: {exc}", file=sys.stderr)
                return fn(item)
            try:
                return fn(item)
            finally:
                if acquired:
                    try:
                        governor.release()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[penrose] worker governor release failed open: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            if on_error is None:
                raise
            return on_error(item, exc)

    if worker_count <= 1 or len(items) <= 1:
        return [_safe(item) for item in items]
    out = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_safe, item): item for item in items}
        for fut in as_completed(futures):
            out.append(fut.result())
    return sorted(out, key=lambda r: r.get("claim_i", 0))


def _processed_set() -> set[str]:
    """Filenames already processed (so the loop advances through inbox/ instead of
    re-running pdfs[0] forever). Reset by `make reset`."""
    p = config.PROCESSED_PAPERS
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).get("processed", []))
    except Exception:  # noqa: BLE001
        return set()


def _record_processed_source(source_id: str, paper_path: Path, text_sha256: str) -> None:
    existing = {}
    if config.PROCESSED_PAPERS.exists():
        try:
            raw = json.loads(config.PROCESSED_PAPERS.read_text())
            existing = raw if isinstance(raw, dict) else {}
        except Exception:  # noqa: BLE001
            existing = {}
    processed = set(existing.get("processed", []) or [])
    processed.add(paper_path.name)
    sources = dict(existing.get("sources", {}) or {})
    sources[source_id] = {
        "paper": paper_path.name,
        "content_sha256": text_sha256,
        "completed_at": _now(),
    }
    config.PROCESSED_PAPERS.parent.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_PAPERS.write_text(json.dumps({
        **existing,
        "processed": sorted(processed),
        "sources": sources,
        "updated_at": _now(),
    }, indent=2))


def _processed_source_entry(source_id: str) -> dict:
    if not config.PROCESSED_PAPERS.exists():
        return {}
    try:
        raw = json.loads(config.PROCESSED_PAPERS.read_text())
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    sources = raw.get("sources", {}) or {}
    entry = sources.get(source_id, {})
    return entry if isinstance(entry, dict) else {}


def _decision_row_source_id(row: dict) -> str:
    source_id = str(row.get("source_id") or "")
    if source_id:
        return source_id
    claim_id = str(row.get("claim_id") or "")
    m = re.match(r"^(.+)-c\d+(?:$|-)", claim_id)
    return m.group(1) if m else ""


def _supersede_decision_rows(source_id: str, run_id: str) -> int:
    """Non-destructive supersede (P0 fix, 2026-07-04 decisions-loss incident).

    decisions.jsonl is APPEND-ONLY: a line once written to it is NEVER deleted or
    rewritten. Before this fix, this function truncated the file, removing every prior
    row for `source_id` written by a different run -- called BEFORE any replacement row
    necessarily existed (the zero-claims / off-domain early-return paths called it too),
    so a --force re-run that (for any reason) produced no new decisions permanently
    erased the prior ones. That is exactly what happened to the two funding_drift_claim
    rows on 2026-07-04.

    Now: for every prior ACTIVE decision belonging to `source_id` from a different run,
    APPEND one supersession-marker row (never touch the old bytes). The marker reuses the
    original `decision_id` so decision-id-keyed readers (e.g. brainstore's rebuild, which
    upserts by id) naturally treat it as that decision's latest state, but the original
    line's bytes remain in the file forever -- recoverable by anyone who scans history
    instead of only the latest state per id. A row that is itself the CURRENT latest state
    for its identity because this run wrote a fresh row with the same decision_id (the
    common case: the same claim re-verdicted) needs no marker at all -- the append order
    already makes the new row win; nothing is marked twice, and nothing is ever removed.
    """
    path = config.DECISIONS_LOG
    if not path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    marked = 0
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows: list[dict] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # unparseable lines are left alone; never touched, never lost

        # Latest known state per identity (decision_id, falling back to claim_id for
        # legacy rows lacking one) -- so a decision that already has a fresher row in the
        # file (this run's own re-verdict, or an earlier marker) is never re-marked.
        latest: dict[str, dict] = {}
        order: list[str] = []
        for row in rows:
            ident = str(row.get("decision_id") or row.get("claim_id") or "")
            if not ident:
                continue
            if ident not in latest:
                order.append(ident)
            latest[ident] = row

        to_mark: list[dict] = []
        for ident in order:
            row = latest[ident]
            if _decision_row_source_id(row) != source_id:
                continue
            if row.get("run_id") == run_id:
                continue  # this run's own row -- nothing to supersede
            if row.get("verdict") == "superseded":
                continue  # already marked; do not re-mark
            to_mark.append(row)

        if to_mark:
            now = _now()
            lines_to_append = []
            for row in to_mark:
                marker = dict(row)
                marker["verdict"] = "superseded"
                marker["kill_reason"] = None
                marker["prior_verdict"] = row.get("verdict")
                marker["prior_run_id"] = row.get("run_id")
                marker["rationale"] = (
                    f"superseded by run {run_id} for source {source_id}; the prior decision "
                    f"row (verdict={row.get('verdict')!r}) is preserved above, never deleted"
                )
                marker["run_id"] = run_id
                marker["superseded_by_run_id"] = run_id
                marker["type"] = "supersession_marker"
                marker["logged_at"] = now
                lines_to_append.append(json.dumps(marker, default=str))
                marked += 1
            with path.open("a") as f:  # APPEND ONLY -- never write_text/replace the file
                f.write("\n".join(lines_to_append) + "\n")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return marked


def _inbox_pdfs() -> list[Path]:
    inbox = config.INBOX
    pdfs = sorted([p for p in inbox.iterdir() if p.suffix.lower() == ".pdf"]) if inbox.exists() else []
    if not pdfs:  # repo-root fallback (test convenience)
        pdfs = sorted(config.ROOT.glob("*.pdf"))
    return pdfs


def _find_paper(cli_path: str | None) -> Path | None:
    """Explicit --paper always wins. Otherwise return the first UNPROCESSED inbox paper
    (None when every paper has been run — the loop's natural stop condition)."""
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            print(f"penrose: paper not found: {p}")
            print("  Check the path, or drop a PDF into inbox/ and run `penrose run` (no --paper).")
            raise SystemExit(1)
        return p
    done = _processed_set()
    for p in _inbox_pdfs():
        if p.name not in done:
            return p
    return None


# --- module registry (cold-start: empty by default) ------------------------- #
# In v1 the operator registers modules they've authored (e.g. macro_vol_btc)
# before running the paper those modules target. A new paper with no module
# triggers ModuleSpec generation at P6.
REGISTRY: dict[str, object] = {}   # strategy_class -> module object with .run()
_REGISTRY_ALIAS_OWNERS: dict[str, str] = {}
_REGISTRY_CANONICAL_OWNERS: dict[str, str] = {}
_REGISTRY_CANONICAL_MODULES: dict[str, object] = {}

SPEC_SELF_CORRECTION_MAX_ATTEMPTS = 3


def _canonical_strategy_class(alias: str) -> str:
    return str(alias or "").replace("-", "_")


def _register_known_modules() -> None:
    """Best-effort registration of operator-supplied modules present on disk.

    Each module lives at modules/<id>/impl.py with a `run(bundle, claim, cost)`
    method. Module's `__strategy_class__` (and optional `__strategy_class_aliases__`)
    declare which claims it handles.
    """
    modules_dir = config.MODULES
    if not modules_dir.exists():
        return
    for mod_dir in modules_dir.iterdir():
        if not mod_dir.is_dir() or mod_dir.name.startswith("_"):
            continue
        impl = mod_dir / "impl.py"
        if not impl.exists():
            continue
        try:
            import importlib.util
            # D-002: decide whether this is an auto-generated (UNTRUSTED) module STATICALLY,
            # before any exec. `exec_module` runs the file's top-level code in penrose's process,
            # outside the Docker sandbox — a persisted malicious auto-impl would fire on every
            # run. Auto-generated modules are skipped here entirely (they're paper-specific and
            # never registered for cross-claim routing anyway, A-019; they execute ONLY in the
            # sandbox at backtest time). Only trusted operator modules are import-exec'd.
            meta = impl_gen.ast_module_meta(impl.read_text())
            if meta["auto_generated"]:
                continue
            spec = importlib.util.spec_from_file_location(
                f"modules.{mod_dir.name}.impl", impl)
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(config.ROOT / "src"))
            spec.loader.exec_module(module)
            if getattr(module, "__auto_generated__", False):
                continue                          # belt-and-suspenders (should be caught above)
            primary = getattr(module, "__strategy_class__", None)
            aliases = set(getattr(module, "__strategy_class_aliases__", []) or [])
            if primary:
                aliases.add(primary)
            if hasattr(module, "run") and aliases:
                owner = str(getattr(module, "__module_id__", "") or mod_dir.name)
                for alias in aliases:
                    canonical = _canonical_strategy_class(alias)
                    existing_owner = (
                        _REGISTRY_CANONICAL_OWNERS.get(canonical)
                        or _REGISTRY_ALIAS_OWNERS.get(alias)
                    )
                    if (existing_owner is not None and existing_owner != owner
                            and _canonical_strategy_class(existing_owner) == _canonical_strategy_class(owner)):
                        _REGISTRY_ALIAS_OWNERS[alias] = existing_owner
                        REGISTRY[alias] = _REGISTRY_CANONICAL_MODULES.get(canonical, module)
                        continue
                    # Trusted-first-wins: never let a later trusted module silently clobber an
                    # alias another already claimed (a real operator foot-gun — dir-iteration
                    # order would otherwise decide routing nondeterministically). First registrant
                    # keeps the alias; the collision is surfaced, not hidden.
                    if existing_owner is not None and existing_owner != owner:
                        print(f"[penrose] strategy_class alias collision: {alias!r} already owned by "
                              f"{existing_owner}; "
                              f"{mod_dir.name} not registered for it", file=sys.stderr)
                        continue
                    _REGISTRY_CANONICAL_OWNERS[canonical] = owner
                    _REGISTRY_CANONICAL_MODULES[canonical] = module
                    _REGISTRY_ALIAS_OWNERS[alias] = owner
                    REGISTRY[alias] = module
        except Exception as e:  # noqa: BLE001
            print(f"[penrose] module {mod_dir.name} failed to load: {e}", file=sys.stderr)


def _data_domain(claim) -> str:
    """Coarse data-domain bucket for C1 family scoping, inferred from the claim text/class."""
    t = ((getattr(claim, "statement", "") or "") + " " +
         (getattr(claim, "applicable_strategy_class", "") or "")).lower()
    if any(k in t for k in ("polymarket", "kalshi", "prediction market", "election", "event contract")):
        return "prediction_market"
    if any(k in t for k in ("weather", "temperature", "noaa", "hdd", "cdd")):
        return "weather"
    if any(k in t for k in ("btc", "eth", "sol", "crypto", "bitcoin", "funding", "perp", "altcoin")):
        return "crypto"
    if any(k in t for k in ("equity", "equities", "stock", "spx", "s&p", "sp500", "nasdaq", "share")):
        return "equities"
    if any(k in t for k in ("cpi", "fed", "recession", "inflation", "treasury", "bond", "macro", "rate")):
        return "macro"
    return "general"


def _is_preregistered_single_cohort(claim, source=None) -> bool:
    try:
        if source is None and hasattr(claim, "preregistered_single_cohort"):
            return bool(getattr(claim, "preregistered_single_cohort"))
        return fidelity_memory.is_preregistered_single_cohort(claim, source)
    except Exception:  # noqa: BLE001
        return False


def _stamp_resolved_claim_type(claim, claim_type: str) -> str:
    claim_type = str(claim_type or "").strip() or fidelity_memory.DEFAULT_CLAIM_TYPE
    try:
        setattr(claim, "resolved_claim_type", claim_type)
    except Exception:  # noqa: BLE001
        pass
    return claim_type


def _authoritative_claim_type(claim, spec: dict | None = None, source=None) -> str:
    """Resolve the run's single claim_type decision.

    The source-aware classifier can see declarations omitted from extracted claim fields.
    Once resolved, keep that value on the Claim so later no-source paths cannot drift.
    """
    stamped = str(getattr(claim, "resolved_claim_type", "") or "").strip()
    if stamped:
        return stamped
    spec_type = str((spec or {}).get("claim_type") or "").strip()
    if spec_type:
        return _stamp_resolved_claim_type(claim, spec_type)
    if source is not None:
        try:
            return _stamp_resolved_claim_type(
                claim, fidelity_memory.classify_claim_type(claim, source)
            )
        except Exception:  # noqa: BLE001
            pass
    try:
        return fidelity_memory.classify_claim_type(claim)
    except Exception:  # noqa: BLE001
        return fidelity_memory.DEFAULT_CLAIM_TYPE


def _family(claim, module, source=None, spec: dict | None = None) -> str:
    """C1: the multiple-testing family = strategy class + data domain. DSR deflation counts only
    sibling strategies in the same family, so unrelated domains don't raise each other's bar."""
    if (
        _authoritative_claim_type(claim, spec, source) == "provided_series_statistic"
        and _is_preregistered_single_cohort(claim, source)
    ):
        return f"provided_series_statistic::{getattr(claim, 'claim_id', 'unknown')}"
    if _authoritative_claim_type(claim, spec, source) == "predictive_regression":
        return f"predictive_regression::{_data_domain(claim)}"
    if _authoritative_claim_type(claim, spec, source) == "factor_spanning":
        return f"factor_spanning::{_data_domain(claim)}"
    if _authoritative_claim_type(claim, spec, source) == "cross_sectional_sort":
        return f"cross_sectional_sort::{_data_domain(claim)}"
    if _authoritative_claim_type(claim, spec, source) == "event_study":
        return f"event_study::{_data_domain(claim)}"
    if _authoritative_claim_type(claim, spec, source) == "forecast_skill":
        return f"forecast_skill::{_data_domain(claim)}"
    if _authoritative_claim_type(claim, spec, source) == "formulaic_signal":
        return f"formulaic_signal::{_data_domain(claim)}"
    if source_is_unanchored(getattr(claim, "source_type", "external_source")):
        return f"generated::{_data_domain(claim)}"
    cls = (getattr(claim, "applicable_strategy_class", "") or
           getattr(module, "__strategy_class__", "") or "unknown")
    # PEN-07: only the operator-registered controlled vocabulary may define a family. A
    # self-declared novel class would mint a fresh n=1 family per paper (denominator reset).
    if _canonical_strategy_class(cls) not in _REGISTRY_CANONICAL_OWNERS:
        return f"external::{_data_domain(claim)}"
    return f"{cls}::{_data_domain(claim)}"


def _structured_strategy_family(claim, source=None, spec: dict | None = None) -> dict:
    raw = (spec or {}).get("strategy_family")
    if raw is None:
        raw = getattr(claim, "strategy_family", None)
    family = declared_strategy_family(claim, source, raw=raw)
    try:
        setattr(claim, "strategy_family", family)
    except Exception:  # noqa: BLE001
        pass
    return family


def _generation_source_for(claim: Claim, spec: dict | None = None, source=None) -> str:
    if _authoritative_claim_type(claim, spec, source) == "provided_series_statistic":
        return "provided_series_statistic"
    if _authoritative_claim_type(claim, spec, source) == "predictive_regression":
        return "predictive_regression"
    if _authoritative_claim_type(claim, spec, source) == "factor_spanning":
        return "factor_spanning"
    if _authoritative_claim_type(claim, spec, source) == "cross_sectional_sort":
        return "cross_sectional_sort"
    if _authoritative_claim_type(claim, spec, source) == "event_study":
        return "event_study"
    if _authoritative_claim_type(claim, spec, source) == "forecast_skill":
        return "forecast_skill"
    if _authoritative_claim_type(claim, spec, source) == "formulaic_signal":
        return "formulaic_signal"
    if claim.source_type == "confirmation":
        return "confirmation"
    if source_is_unanchored(claim.source_type):
        return "generated"
    return "paper"


def _cohort_id(source_id: str, family: str) -> str:
    digest = hashlib.sha256(family.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:p7:{digest}"


def _register_run_cohorts(ready_for_backtest: list[dict], source_id: str, source=None) -> None:
    """Register the run's P7-ready claims as per-family cohorts before backtesting.

    This fixes the paper/external path's order-dependent running denominator. The cohort is
    built only from claims that survived P3-P6 and are about to enter P7; if ledger registration
    fails, leave newly computed cohort fields unset so P7 falls back to the prior running count.
    """
    if not ready_for_backtest:
        return
    grouped: dict[str, list[dict]] = {}
    prepared: list[tuple[Claim, str, str, int]] = []
    rows: list[dict] = []
    for item in ready_for_backtest:
        claim = item["claim"]
        spec = item.get("spec")
        claim_type = _authoritative_claim_type(claim, spec, source)
        preregistered_single_cohort = (
            claim_type == "provided_series_statistic"
            and fidelity_memory.is_preregistered_single_cohort(claim, source)
        )
        try:
            setattr(claim, "preregistered_single_cohort", preregistered_single_cohort)
        except Exception:  # noqa: BLE001
            pass
        family = _family(claim, item["module"], source, spec)
        _structured_strategy_family(claim, source, spec)
        item["family"] = family
        if claim_type == "provided_series_statistic" and preregistered_single_cohort:
            cohort_id = f"{source_id}:provided_series:{claim.claim_id}"
            denominator = 1
            prepared.append((claim, family, cohort_id, denominator))
            rows.append({
                "strategy": claim.claim_id,
                "family": family,
                "generation_source": _generation_source_for(claim, spec, source),
                "search_cohort_id": cohort_id,
                "search_denominator": denominator,
            })
            continue
        grouped.setdefault(family, []).append(item)

    for family, items in sorted(grouped.items()):
        # The per-run cohort size is this run's P7-ready claim count only. Ignore any
        # stale cohort fields if a Claim object is accidentally reused across runs.
        denominator = len(items)
        cohort_id = _cohort_id(source_id, family)
        for item in items:
            claim = item["claim"]
            prepared.append((claim, family, cohort_id, denominator))
            rows.append({
                "strategy": claim.claim_id,
                "family": family,
                "generation_source": _generation_source_for(claim, item.get("spec"), source),
                "search_cohort_id": cohort_id,
                "search_denominator": denominator,
            })
    try:
        p7_backtest.register_trials(rows)
    except Exception as e:  # noqa: BLE001
        print(f"[penrose] cohort pre-registration failed; falling back to running count: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return
    for claim, _family_name, cohort_id, denominator in prepared:
        claim.search_cohort_id = cohort_id
        claim.search_denominator = denominator


def _cleanup_run_cohorts(ready_for_backtest: list[dict]) -> None:
    cohort_ids = {
        str(getattr(item["claim"], "search_cohort_id", "") or "")
        for item in ready_for_backtest
        if _generation_source_for(item["claim"], item.get("spec")) in {
            "paper", "provided_series_statistic", "predictive_regression",
            "factor_spanning", "cross_sectional_sort", "event_study", "forecast_skill",
            "formulaic_signal",
        }
    }
    try:
        p7_backtest.cleanup_unscored_paper_cohorts(cohort_ids)
    except Exception as e:  # noqa: BLE001
        print(f"[penrose] cohort cleanup failed; continuing: {type(e).__name__}: {e}",
              file=sys.stderr)


def _holdout_unreachable_reason(claim: Claim, bt: dict | None = None) -> str | None:
    bt = bt or {}
    measured_costs = (
        getattr(config, "COST_PROVENANCE", "modeled") == "measured"
        or str(bt.get("cost_provenance") or "").strip().lower() == "measured"
    )
    if not measured_costs:
        return ("holdout not consulted: research-supported unreachable under modeled costs - "
                "preserved for a measured-cost run")
    if source_is_unanchored(claim.source_type):
        return ("holdout not consulted: research-supported unreachable for unanchored source - "
                "preserved for external confirmation")
    return None


def _maybe_consult_holdout(claim: Claim, bt: dict, mres: dict, dec, synthetic: bool):
    holdout = {}
    if not (dec.verdict == "watch"
            and (bt.get("dsr") or 0) >= config.DSR_DECISION["watch_band"][1]):
        return dec, holdout
    reason = _holdout_unreachable_reason(claim, bt)
    if reason:
        holdout = {"not_consulted": True, "reason": reason}
        dec.metrics["holdout"] = holdout
        dec.rationale = f"{dec.rationale}; {reason}"
        return dec, holdout
    holdout = p7_backtest.final_holdout_eval(
        claim.claim_id, mres["net"], mres["bars_per_year"])
    dec = stages.p8_verdict(claim, bt, holdout, synthetic)   # may upgrade -> research-supported
    return dec, holdout


def run_source(paper_path: Path, *, use_llm: bool | None = None,
               claims_override: list[Claim] | None = None,
               source_type: str = "external_source",
               bundle_override=None,
               force: bool = False,
               max_claims: int | None = None,
               max_claim_workers: int | str | None = None,
               principal: str = "cli") -> dict:
    """Run one paper end-to-end. Returns the run log (also written to runs.jsonl)."""
    config.ensure_output_dirs()
    source_type = validate_source_type(source_type)
    if claims_override is not None:
        for claim in claims_override:
            claim.source_type = source_type
    reader = BrainReader()
    run_log: dict = {"run_at": _now(), "paper_path": str(paper_path), "claims": [],
                     "source_type": source_type}
    claim_worker_resolution = _claim_worker_resolution(max_claim_workers)
    claim_workers = claim_worker_resolution.count
    _log_claim_worker_resolution(claim_worker_resolution)
    claim_governor = worker_control.configure_claim_governor(claim_workers)
    run_log["claim_workers"] = claim_workers
    _set_pipeline_status("running")        # dashboard status dot -> green while we run
    _progress("ingest", paper=paper_path.name)

    # ---- P1: ingest -------------------------------------------------------- #
    source = p1_ingest.sanitize(paper_path)
    source_id = source.source_id
    title = source.title or source_id
    run_id = f"{source_id}-{uuid.uuid4().hex}"
    audit_path = Path(config.AUDIT) / f"{run_id}.jsonl"
    audit_log = AuditLog(run_id, principal, audit_path)
    run_log["source_id"] = source_id
    run_log["source_title"] = source.title
    run_log["p1"] = {"n_pages": source.n_pages, "n_chars": source.n_chars,
                     "sha": source.text_sha256,
                     "injection_flags": source.injection_flags}
    run_log["idempotency"] = {
        "source_id": source_id,
        "run_id": run_id,
        "content_sha256": source.text_sha256,
        "force": bool(force),
        "superseded_decisions": 0,
    }
    _audit_call(
        audit_log.envelope,
        __version__,
        config_fingerprint(),
        _audit_data_sources(run_log),
        _audit_seeds(),
        platform_tuple(),
        _audit_reproducibility_class(source_type),
    )
    prior = _processed_source_entry(source_id)
    if (
        not force
        and prior.get("completed_at")
        and prior.get("content_sha256") == source.text_sha256
    ):
        msg = "already processed (unchanged); use --force to re-run"
        print(f"[penrose] {source_id}: {msg}", file=sys.stderr)
        run_log["idempotency"]["skipped"] = True
        run_log["note"] = msg
        _finalize_audit_run(audit_log, audit_path, run_log, [])
        _append_jsonl(config.ROOT / "runs.jsonl", run_log)
        _record_processed_source(source_id, paper_path, source.text_sha256)
        _progress(None)
        return run_log

    # archive the source record
    archive_kind = ("dreams" if source_type == "generated_hypothesis"
                    else "syntheses" if source_type in {"synthesized_hypothesis", "confirmation"}
                    else "papers")
    arch = config.ARCHIVES / archive_kind / source_id
    arch.mkdir(parents=True, exist_ok=True)
    metadata_path = arch / "metadata.json"
    try:
        with metadata_path.open("x") as metadata_file:
            metadata_file.write(json.dumps({
            "source_id": source_id, "title": source.title,
            "ingested_at": _now(), "sha": source.text_sha256,
            "n_pages": source.n_pages, "injection_flags": source.injection_flags,
            }, indent=2))
    except FileExistsError:
        pass

    if use_llm is None:
        use_llm = _llm_available()

    # ---- relevance gate (pre-P2): skip off-domain papers before the expensive stages --- #
    # A paper with no claim testable against our data domains (e.g. nuclear physics) would
    # otherwise burn LLM budget and clutter the data-request backlog. Fails open.
    if claims_override is None and use_llm and getattr(config, "RELEVANCE_GATE", True):
        _progress("relevance", paper=title)
        rel = relevance.screen(source.title, source.text)
        run_log["relevance"] = rel
        if not rel.get("relevant", True):
            print(f"[penrose] off-domain, skipping P2+: {rel.get('reason')}", file=sys.stderr)
            _progress(None)
            out = _finish_offdomain(source, source_id, rel, run_log)
            # FIX 1 (2026-07-04 data-loss incident): an off-domain classification does not
            # re-adjudicate any specific claim, so no prior decision for this source is
            # touched here — superseding on a non-result is exactly the destructive pattern
            # that erased funding_drift_claim's rows. Nothing to mark; nothing removed.
            _finalize_audit_run(audit_log, audit_path, run_log, [])
            _append_jsonl(config.ROOT / "runs.jsonl", run_log)
            _record_processed_source(source_id, paper_path, source.text_sha256)
            return out

    # register operator-supplied modules FIRST, so P2 can reuse their strategy classes
    # (controlled-vocabulary routing — the LLM names an existing class when a claim fits,
    # instead of inventing a unique one that never routes). Cold-start aware.
    _register_known_modules()
    run_log["registry"] = list(REGISTRY.keys())
    known = _known_classes()

    # ---- P2: claim extraction --------------------------------------------- #
    _progress("extract", paper=title)
    if claims_override is not None:
        claims = list(claims_override)
        p2_prov = {
            "mode": "source-adapter",
            "source_type": source_type,
            "n_extracted": len(claims),
            "note": "canonical claims supplied by source adapter; paper-specific P2 skipped",
        }
    elif use_llm:
        try:
            claims, p2_prov = extract.extract_claims(source, known_classes=known)
            p2_prov["mode"] = "llm"
        except Exception as e:  # noqa: BLE001
            print(f"[penrose] P2 LLM call failed ({e}); falling back to manual", file=sys.stderr)
            claims, p2_prov = extract.fallback_claims(source)
            p2_prov["mode"] = "fallback-after-error"
            p2_prov["extraction_error"] = str(e)
    else:
        claims, p2_prov = extract.fallback_claims(source)
        p2_prov["mode"] = "manual"

    run_log["p2"] = p2_prov
    if not claims:
        run_log["note"] = "no claims extracted; pipeline ends here"
        # FIX 3 (fail-soft violation, 2026-07-04 incident): a 0-claim result on a
        # non-trivial source is surfaced LOUDLY (an engine_error decision + review-queue
        # entry), never treated as a silent success — see _zero_extraction_is_suspicious.
        if _zero_extraction_is_suspicious(source, p2_prov):
            _zero_extraction_engine_error(source, source_id, p2_prov, run_log)
            run_log["engine_error"] = True
        # FIX 1: no claim was re-adjudicated by a zero-extraction result, so no prior
        # decision for this source is superseded here — this exact call, on this exact
        # branch, is what erased funding_drift_claim's rows on 2026-07-04. Never call it
        # on a non-result.
        _finalize_audit_run(audit_log, audit_path, run_log, [])
        _append_jsonl(config.ROOT / "runs.jsonl", run_log)
        _record_processed_source(source_id, paper_path, source.text_sha256)
        _progress(None)
        return run_log
    if max_claims is not None:
        max_claims = max(0, int(max_claims))
        run_log["max_claims"] = max_claims
        claims = claims[:max_claims]

    # ---- one data pull shared across claims ------------------------------- #
    bundle = bundle_override if bundle_override is not None else dataclient.fetch_bundle()
    provenance = bundle.provenance_summary()
    synthetic = bundle.any_synthetic()
    cost_frac = config.VOL_TRADE_COST["deribit_roundtrip_bps_of_vega"] / 1e4

    decisions = []
    specs_generated = []
    ready_for_backtest = []
    auto_fetch_attempted_series: set[str] = set()

    claims_for_phase1 = claims
    if claim_workers > 1 and len(claims) > 1:
        def _phase1_one(item: tuple[int, Claim]) -> dict:
            _ci, claim = item
            claim_started = time.monotonic()
            rec: dict = {"claim_id": claim.claim_id, "statement": claim.statement, "stages": {}}
            rec["stages"]["P1"] = {"sanitized": True}
            local_bundle = _isolated_bundle_access(bundle)
            local_run_log = {**run_log, "claims": []}
            local_decisions = []
            local_ready = []
            local_specs = []
            with _defer_jsonl_events() as events:
                for _once in (None,):
                    _progress("evaluate", detail=f"P3–P6 routing: {_short_name(claim)}",
                              paper=title, claim_i=_ci, claim_n=len(claims))
                    if use_llm:
                        try:
                            p3 = extract.classify_claim(claim)
                        except Exception as e:  # noqa: BLE001
                            print(f"[penrose] P3 LLM failed ({e}); using stub", file=sys.stderr)
                            p3 = extract.classify_claim_stub(claim)
                    else:
                        p3 = extract.classify_claim_stub(claim)
                    rec["stages"]["P3"] = p3
                    if p3["killed"]:
                        local_decisions.append(
                            _kill(claim, p3["reason"], p3["note"], synthetic, rec, local_run_log))
                        continue
                    _authoritative_claim_type(claim, source=source)

                    p4 = stages.p4_fee_curve(claim, expected_edge=claim.expected_edge)
                    rec["stages"]["P4"] = p4
                    if p4["killed"]:
                        local_decisions.append(
                            _kill(claim, p4["reason"], p4["note"], synthetic, rec, local_run_log))
                        continue

                    p5 = stages.p5_dedup(claim, reader)
                    rec["stages"]["P5"] = p5
                    if p5["killed"]:
                        local_decisions.append(
                            _kill(claim, p5["reason"], p5["note"], synthetic, rec, local_run_log))
                        continue

                    cls = claim.applicable_strategy_class or ""
                    module = REGISTRY.get(cls)
                    spec = None
                    if module is not None and use_llm and getattr(config, "FIDELITY_CHECK", False):
                        try:
                            _mc = Path(getattr(module, "__file__", "") or "").read_text()
                        except Exception:  # noqa: BLE001
                            _mc = ""
                        _rf = _assess_fidelity_safe(claim, _mc)
                        if not _rf.get("verified", False):
                            rec["stages"]["P6_route_fidelity"] = {
                                "reused": False, "reason": _rf.get("note", "")[:140],
                                "confidence": _rf.get("confidence", 0),
                            }
                            module = None
                    if _claim_budget_exceeded(claim_started):
                        local_decisions.append(_skip(
                            claim, "timeout",
                            "skipped: per-claim time budget exceeded before module routing",
                            rec, local_run_log))
                        continue
                    if module is None:
                        pre_fid = {}
                        prior_divergences: list[str] | None = None
                        missing_inputs: list[str] = []
                        binding_review: dict | None = None
                        max_spec_attempts = (
                            SPEC_SELF_CORRECTION_MAX_ATTEMPTS
                            if use_llm and getattr(config, "FIDELITY_CHECK", False)
                            else 1
                        )
                        for spec_attempt in range(1, max_spec_attempts + 1):
                            spec = spec_gen.generate_spec(
                                claim, source, use_llm=use_llm,
                                prior_divergences=prior_divergences)
                            _record_spec_inputs(rec, spec)
                            _authoritative_claim_type(claim, spec, source)
                            local_specs.append(spec)
                            binding_review = (
                                _predictive_regression_binding_review(claim, spec)
                                or _factor_spanning_binding_review(claim, spec)
                                or _cross_sectional_sort_binding_review(claim, spec)
                                or _event_study_binding_review(claim, spec)
                                or _forecast_skill_binding_review(claim, spec)
                                or _formulaic_signal_binding_review(claim, spec)
                            )
                            if binding_review:
                                explanation = binding_review.get("explanation") or {}
                                rec["stages"]["P6"] = {
                                    "module_id": None,
                                    "spec_generated": True,
                                    "auto_implemented": False,
                                    "spec_path": str(spec.get("_path", "")),
                                    "binding_uncertainty": binding_review.get("reason"),
                                    "human_review": explanation,
                                    "note": f"{binding_review.get('kind')} -> needs_review",
                                }
                                local_decisions.append(_needs_review(
                                    claim,
                                    binding_review.get("reason"),
                                    rec,
                                    local_run_log,
                                    metrics={
                                        "claim_type": spec.get("claim_type"),
                                        "binding_uncertainty": binding_review.get("reason"),
                                        "binding_detail": binding_review.get("detail", {}),
                                        "spec_path": str(spec.get("_path", "")),
                                    },
                                    review=explanation,
                                ))
                                break
                            try:
                                missing_inputs = _missing_spec_inputs_from_bundle(spec, local_bundle)
                            except Exception:  # noqa: BLE001
                                missing_inputs = None
                            if missing_inputs:
                                rec["stages"]["P6_data_availability"] = {
                                    "blocked": True,
                                    "inputs_requested": list(rec.get("inputs_requested", [])),
                                    "missing_series": missing_inputs,
                                    "spec_path": str(spec.get("_path", "")),
                                }
                                local_decisions.append(_needs_data(
                                    claim, "data_unavailable: " + ", ".join(missing_inputs),
                                    rec, local_run_log))
                                break
                            pre_fid = (
                                _assess_spec_fidelity_safe(claim, spec)
                                if use_llm and getattr(config, "FIDELITY_CHECK", False)
                                else {}
                            )
                            if use_llm and getattr(config, "FIDELITY_CHECK", False):
                                rec["stages"].setdefault("P6_pre_fidelity_attempts", []).append({
                                    "attempt": spec_attempt,
                                    "blocked": _fidelity_confidently_unfaithful(pre_fid),
                                    "faithful": pre_fid.get("faithful"),
                                    "confidence": pre_fid.get("confidence", 0),
                                    "divergences": pre_fid.get("divergences", []),
                                    "spec_path": str(spec.get("_path", "")),
                                })
                            if not _fidelity_confidently_unfaithful(pre_fid):
                                break
                            _persist_fidelity_rejection(claim, spec, pre_fid)
                            if spec_attempt >= max_spec_attempts:
                                local_decisions.append(
                                    _cannot_replicate_unfaithful_spec(
                                        claim, spec, pre_fid, rec, local_run_log))
                                break
                            prior_divergences = list(pre_fid.get("divergences") or [])
                            if not prior_divergences and pre_fid.get("note"):
                                prior_divergences = [str(pre_fid.get("note"))]
                        if binding_review:
                            continue
                        if missing_inputs:
                            continue
                        if _fidelity_confidently_unfaithful(pre_fid):
                            continue
                        claim_type = _authoritative_claim_type(claim, spec, source)
                        if claim_type == "provided_series_statistic":
                            module = provided_series.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "provided_series_statistic"},
                                    "deterministic_provided_series": True}
                        elif claim_type == "event_market_strategy":
                            module = event_market.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "event_market_strategy"},
                                    "deterministic_event_market": True}
                        elif claim_type == "predictive_regression":
                            module = predictive_regression.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "predictive_regression"},
                                    "deterministic_regression": True}
                        elif claim_type == "factor_spanning":
                            module = factor_spanning.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "factor_spanning"},
                                    "deterministic_factor_spanning": True}
                        elif claim_type == "cross_sectional_sort":
                            module = cross_sectional_sort.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "cross_sectional_sort"},
                                    "deterministic_cross_sectional_sort": True}
                        elif claim_type == "event_study":
                            module = event_study.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "event_study"},
                                    "deterministic_event_study": True}
                        elif claim_type == "forecast_skill":
                            module = forecast_skill.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "forecast_skill"},
                                    "deterministic_forecast_skill": True}
                        elif claim_type == "formulaic_signal":
                            module = formulaic_signal.build_module(spec, claim)
                            impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                                    "validation": {"deterministic": "formulaic_signal"},
                                    "deterministic_formulaic_signal": True}
                        elif not config.AUTO_IMPLEMENT_MODULES:
                            impl = {"ok": False, "reason": "auto-impl disabled"}
                        elif not (sandbox.docker_available() and sandbox.ensure_image()):
                            impl = {"ok": False, "reason": "auto-impl requires Docker sandbox (not available); "
                                    "operator must implement, or start Docker"}
                        else:
                            impl = impl_gen.try_implement(
                                spec, claim, local_bundle, cost_frac, use_llm=use_llm)
                        if impl.get("ok"):
                            deterministic_module = (
                                impl.get("deterministic_provided_series")
                                or impl.get("deterministic_event_market")
                                or impl.get("deterministic_regression")
                                or impl.get("deterministic_factor_spanning")
                                or impl.get("deterministic_cross_sectional_sort")
                                or impl.get("deterministic_event_study")
                                or impl.get("deterministic_forecast_skill")
                                or impl.get("deterministic_formulaic_signal")
                            )
                            if not deterministic_module:
                                with _REGISTRY_LOCK:
                                    _register_known_modules()
                            module = impl["module"]
                            rec["stages"]["P6"] = {
                                "module_id": impl["module_id"], "spec_generated": True,
                                "auto_implemented": not deterministic_module,
                                "deterministic_provided_series": bool(
                                    impl.get("deterministic_provided_series")),
                                "deterministic_event_market": bool(
                                    impl.get("deterministic_event_market")),
                                "deterministic_regression": bool(
                                    impl.get("deterministic_regression")),
                                "deterministic_factor_spanning": bool(
                                    impl.get("deterministic_factor_spanning")),
                                "deterministic_cross_sectional_sort": bool(
                                    impl.get("deterministic_cross_sectional_sort")),
                                "deterministic_event_study": bool(
                                    impl.get("deterministic_event_study")),
                                "deterministic_forecast_skill": bool(
                                    impl.get("deterministic_forecast_skill")),
                                "deterministic_formulaic_signal": bool(
                                    impl.get("deterministic_formulaic_signal")),
                                "spec_path": str(spec.get("_path", "")),
                                "validation": impl.get("validation", {}),
                                "note": (
                                    "deterministic predictive-regression executor; backtesting"
                                    if impl.get("deterministic_regression") else
                                    "deterministic factor-spanning executor; backtesting"
                                    if impl.get("deterministic_factor_spanning") else
                                    "deterministic cross-sectional-sort executor; backtesting"
                                    if impl.get("deterministic_cross_sectional_sort") else
                                    "deterministic event-study executor; backtesting"
                                    if impl.get("deterministic_event_study") else
                                    "deterministic forecast-skill executor; backtesting"
                                    if impl.get("deterministic_forecast_skill") else
                                    "deterministic formulaic-signal executor; backtesting"
                                    if impl.get("deterministic_formulaic_signal") else
                                    "deterministic event-market executor; backtesting"
                                    if impl.get("deterministic_event_market") else
                                    "deterministic provided-series executor; backtesting"
                                    if impl.get("deterministic_provided_series")
                                    else "spec auto-implemented + validated on the live bundle; backtesting"
                                )}
                        elif impl.get("needs_review"):
                            audit = impl.get("no_progress") or {}
                            explanation = human_review_explanation(
                                "auto_impl_no_progress",
                                {
                                    "reason": impl.get("reason"),
                                    "attempts": audit.get("attempts_tried")
                                    or audit.get("consecutive_attempts"),
                                },
                            )
                            rec["stages"]["P6"] = {
                                "module_id": None, "spec_generated": True,
                                "auto_implemented": False,
                                "auto_impl_reason": str(impl.get("reason", ""))[:240],
                                "auto_impl_no_progress": audit,
                                "human_review": explanation,
                                "spec_path": str(spec.get("_path", "")),
                                "note": "ModuleSpec generated; auto-impl no-progress guard -> needs_review"}
                            rec["stages"]["P6_auto_impl_no_progress"] = audit
                            local_decisions.append(_needs_review(
                                claim, impl.get("reason"), rec, local_run_log,
                                metrics={"auto_impl_no_progress": audit},
                                review=explanation))
                            continue
                        else:
                            rec["stages"]["P6"] = {
                                "module_id": None, "spec_generated": True,
                                "auto_implemented": False,
                                "auto_impl_reason": str(impl.get("reason", ""))[:140],
                                "spec_path": str(spec.get("_path", "")),
                                "note": "ModuleSpec generated; auto-impl declined -> pending operator"}
                            _append_jsonl(config.REVIEW_QUEUE,
                                          {"type": "module_spec", "queued_at": _now(),
                                           "status": "pending",
                                           "proposal_id": f"{claim.claim_id}-spec",
                                           "name": _short_name(claim),
                                           "claim_id": claim.claim_id, "strategy_class": cls,
                                           "spec_path": str(spec.get("_path", "")),
                                           "statement": claim.statement, "meaning": claim.mechanism})
                            local_decisions.append(_skip(
                                claim, "module_unavailable",
                                "spec generated; auto-impl declined ("
                                + str(impl.get("reason", ""))[:60] + "); awaiting operator",
                                rec, local_run_log))
                            continue
                    else:
                        rec["stages"]["P6"] = {"module_id": getattr(module, "__module_id__", "unknown"),
                                               "spec_generated": False}
                    if _claim_budget_exceeded(claim_started):
                        local_decisions.append(_skip(
                            claim, "timeout",
                            "skipped: per-claim time budget exceeded before backtest",
                            rec, local_run_log))
                        continue
                    local_ready.append({
                        "claim": claim,
                        "module": module,
                        "rec": rec,
                        "claim_i": _ci,
                        "spec": spec,
                    })
            return {"claim_i": _ci, "decisions": local_decisions, "ready": local_ready,
                    "specs": local_specs, "claims": list(local_run_log.get("claims", [])),
                    "events": list(events)}

        phase1_items = list(enumerate(claims, 1))

        def _phase1_on_error(item, exc):  # P-1: isolate a worker crash to its own claim
            _eci, _eclaim = item
            _erec = {"claim_id": _eclaim.claim_id, "statement": _eclaim.statement, "stages": {}}
            return {"claim_i": _eci, "ready": [], "specs": [], "claims": [], "events": [],
                    "decisions": [_engine_error(_eclaim, "parallel worker (phase1)", exc, _erec, run_log)]}

        for result in _run_claim_tasks(
                phase1_items, claim_workers, _phase1_one, _phase1_on_error, claim_governor):
            _emit_jsonl_events(result.get("events", []))
            decisions.extend(result.get("decisions", []))
            specs_generated.extend(result.get("specs", []))
            ready_for_backtest.extend(result.get("ready", []))
            run_log["claims"].extend(result.get("claims", []))
        ready_for_backtest.sort(key=lambda item: item.get("claim_i", 0))
        claims_for_phase1 = []

    for _ci, claim in enumerate(claims_for_phase1, 1):
        claim_started = time.monotonic()
        _progress("evaluate", detail=f"P3–P6 routing: {_short_name(claim)}",
                  paper=title, claim_i=_ci, claim_n=len(claims))
        rec: dict = {"claim_id": claim.claim_id, "statement": claim.statement, "stages": {}}
        rec["stages"]["P1"] = {"sanitized": True}

        # ---- P3: falsifiability ------------------------------------------- #
        if use_llm:
            try:
                p3 = extract.classify_claim(claim)
            except Exception as e:  # noqa: BLE001
                print(f"[penrose] P3 LLM failed ({e}); using stub", file=sys.stderr)
                p3 = extract.classify_claim_stub(claim)
        else:
            p3 = extract.classify_claim_stub(claim)
        rec["stages"]["P3"] = p3
        if p3["killed"]:
            decisions.append(_kill(claim, p3["reason"], p3["note"], synthetic, rec, run_log))
            continue
        _authoritative_claim_type(claim, source=source)

        # ---- P4: fee curve ------------------------------------------------ #
        p4 = stages.p4_fee_curve(claim, expected_edge=claim.expected_edge)
        rec["stages"]["P4"] = p4
        if p4["killed"]:
            decisions.append(_kill(claim, p4["reason"], p4["note"], synthetic, rec, run_log))
            continue

        # ---- P5: dedup ---------------------------------------------------- #
        p5 = stages.p5_dedup(claim, reader)
        rec["stages"]["P5"] = p5
        if p5["killed"]:
            decisions.append(_kill(claim, p5["reason"], p5["note"], synthetic, rec, run_log))
            continue

        # ---- P6: module routing (+ spec + auto-implement) ----------------- #
        cls = claim.applicable_strategy_class or ""
        module = REGISTRY.get(cls)
        spec = None
        # B1: fidelity GATES routing. If a claim matched an EXISTING module, verify that module
        # actually implements THIS claim before trusting a backtest on it. If the refuter calls
        # it unfaithful, don't reuse — fall through to a fresh spec + auto-impl for this claim.
        # (This catches the over-collapse — many distinct claims -> one module — BEFORE the
        # backtest, not just flagging it after.)
        if module is not None and use_llm and getattr(config, "FIDELITY_CHECK", False):
            try:
                _mc = Path(getattr(module, "__file__", "") or "").read_text()
            except Exception:  # noqa: BLE001
                _mc = ""
            _rf = _assess_fidelity_safe(claim, _mc)
            if not _rf.get("verified", False):
                rec["stages"]["P6_route_fidelity"] = {
                    "reused": False, "reason": _rf.get("note", "")[:140],
                    "confidence": _rf.get("confidence", 0),
                }
                module = None      # unverified reuse is not an authorization to test this claim
        if _claim_budget_exceeded(claim_started):
            decisions.append(_skip(claim, "timeout",
                                   "skipped: per-claim time budget exceeded before module routing",
                                   rec, run_log))
            continue
        if module is None:
            # cold-start / novel class -> generate spec, then TRY to auto-implement it
            pre_fid = {}
            prior_divergences: list[str] | None = None
            missing_inputs: list[str] = []
            binding_review: dict | None = None
            max_spec_attempts = (
                SPEC_SELF_CORRECTION_MAX_ATTEMPTS
                if use_llm and getattr(config, "FIDELITY_CHECK", False)
                else 1
            )
            for spec_attempt in range(1, max_spec_attempts + 1):
                spec = spec_gen.generate_spec(
                    claim, source, use_llm=use_llm,
                    prior_divergences=prior_divergences)
                _record_spec_inputs(rec, spec)
                _authoritative_claim_type(claim, spec, source)
                specs_generated.append(spec)
                binding_review = (
                    _predictive_regression_binding_review(claim, spec)
                    or _factor_spanning_binding_review(claim, spec)
                    or _cross_sectional_sort_binding_review(claim, spec)
                    or _event_study_binding_review(claim, spec)
                    or _forecast_skill_binding_review(claim, spec)
                    or _formulaic_signal_binding_review(claim, spec)
                )
                if binding_review:
                    explanation = binding_review.get("explanation") or {}
                    rec["stages"]["P6"] = {
                        "module_id": None,
                        "spec_generated": True,
                        "auto_implemented": False,
                        "spec_path": str(spec.get("_path", "")),
                        "binding_uncertainty": binding_review.get("reason"),
                        "human_review": explanation,
                        "note": f"{binding_review.get('kind')} -> needs_review",
                    }
                    decisions.append(_needs_review(
                        claim,
                        binding_review.get("reason"),
                        rec,
                        run_log,
                        metrics={
                            "claim_type": spec.get("claim_type"),
                            "binding_uncertainty": binding_review.get("reason"),
                            "binding_detail": binding_review.get("detail", {}),
                            "spec_path": str(spec.get("_path", "")),
                        },
                        review=explanation,
                    ))
                    break
                try:
                    missing_inputs = _missing_spec_inputs_from_bundle(spec, bundle)
                except Exception:  # noqa: BLE001 — pre-fidelity availability check must fail open
                    missing_inputs = None
                if missing_inputs:
                    rec["stages"]["P6_data_availability"] = {
                        "blocked": True,
                        "inputs_requested": list(rec.get("inputs_requested", [])),
                        "missing_series": missing_inputs,
                        "spec_path": str(spec.get("_path", "")),
                    }
                    decisions.append(_needs_data(
                        claim, "data_unavailable: " + ", ".join(missing_inputs), rec, run_log))
                    break
                pre_fid = (
                    _assess_spec_fidelity_safe(claim, spec)
                    if use_llm and getattr(config, "FIDELITY_CHECK", False)
                    else {}
                )
                if use_llm and getattr(config, "FIDELITY_CHECK", False):
                    rec["stages"].setdefault("P6_pre_fidelity_attempts", []).append({
                        "attempt": spec_attempt,
                        "blocked": _fidelity_confidently_unfaithful(pre_fid),
                        "faithful": pre_fid.get("faithful"),
                        "confidence": pre_fid.get("confidence", 0),
                        "divergences": pre_fid.get("divergences", []),
                        "spec_path": str(spec.get("_path", "")),
                    })
                if not _fidelity_confidently_unfaithful(pre_fid):
                    break
                _persist_fidelity_rejection(claim, spec, pre_fid)
                if spec_attempt >= max_spec_attempts:
                    decisions.append(_cannot_replicate_unfaithful_spec(claim, spec, pre_fid, rec, run_log))
                    break
                prior_divergences = list(pre_fid.get("divergences") or [])
                if not prior_divergences and pre_fid.get("note"):
                    prior_divergences = [str(pre_fid.get("note"))]
            if binding_review:
                continue
            if missing_inputs:
                continue
            if _fidelity_confidently_unfaithful(pre_fid):
                continue
            claim_type = _authoritative_claim_type(claim, spec, source)
            if claim_type == "provided_series_statistic":
                module = provided_series.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "provided_series_statistic"},
                        "deterministic_provided_series": True}
            elif claim_type == "event_market_strategy":
                module = event_market.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "event_market_strategy"},
                        "deterministic_event_market": True}
            elif claim_type == "predictive_regression":
                module = predictive_regression.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "predictive_regression"},
                        "deterministic_regression": True}
            elif claim_type == "factor_spanning":
                module = factor_spanning.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "factor_spanning"},
                        "deterministic_factor_spanning": True}
            elif claim_type == "cross_sectional_sort":
                module = cross_sectional_sort.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "cross_sectional_sort"},
                        "deterministic_cross_sectional_sort": True}
            elif claim_type == "event_study":
                module = event_study.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "event_study"},
                        "deterministic_event_study": True}
            elif claim_type == "forecast_skill":
                module = forecast_skill.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "forecast_skill"},
                        "deterministic_forecast_skill": True}
            elif claim_type == "formulaic_signal":
                module = formulaic_signal.build_module(spec, claim)
                impl = {"ok": True, "module": module, "module_id": module.__module_id__,
                        "validation": {"deterministic": "formulaic_signal"},
                        "deterministic_formulaic_signal": True}
            # Auto-implementation REQUIRES the Docker sandbox — model-written code never execs unsandboxed.
            # No Docker -> no auto-implement (the claim stays pending_module for the operator).
            elif not config.AUTO_IMPLEMENT_MODULES:
                impl = {"ok": False, "reason": "auto-impl disabled"}
            elif not (sandbox.docker_available() and sandbox.ensure_image()):
                impl = {"ok": False, "reason": "auto-impl requires Docker sandbox (not available); "
                        "operator must implement, or start Docker"}
            else:
                impl = impl_gen.try_implement(spec, claim, bundle, cost_frac, use_llm=use_llm)
            if impl.get("ok"):
                deterministic_module = (
                    impl.get("deterministic_provided_series")
                    or impl.get("deterministic_event_market")
                    or impl.get("deterministic_regression")
                    or impl.get("deterministic_factor_spanning")
                    or impl.get("deterministic_cross_sectional_sort")
                    or impl.get("deterministic_event_study")
                    or impl.get("deterministic_forecast_skill")
                    or impl.get("deterministic_formulaic_signal")
                )
                if not deterministic_module:
                    _register_known_modules()      # discover the just-written, validated module
                module = impl["module"]
                rec["stages"]["P6"] = {"module_id": impl["module_id"], "spec_generated": True,
                                       "auto_implemented": not deterministic_module,
                                       "deterministic_provided_series": bool(
                                           impl.get("deterministic_provided_series")),
                                       "deterministic_event_market": bool(
                                           impl.get("deterministic_event_market")),
                                       "deterministic_regression": bool(
                                           impl.get("deterministic_regression")),
                                       "deterministic_factor_spanning": bool(
                                           impl.get("deterministic_factor_spanning")),
                                       "deterministic_cross_sectional_sort": bool(
                                           impl.get("deterministic_cross_sectional_sort")),
                                       "deterministic_event_study": bool(
                                           impl.get("deterministic_event_study")),
                                       "deterministic_forecast_skill": bool(
                                           impl.get("deterministic_forecast_skill")),
                                       "deterministic_formulaic_signal": bool(
                                           impl.get("deterministic_formulaic_signal")),
                                       "spec_path": str(spec.get("_path", "")),
                                       "validation": impl.get("validation", {}),
                                       "note": (
                                           "deterministic predictive-regression executor; backtesting"
                                           if impl.get("deterministic_regression") else
                                           "deterministic factor-spanning executor; backtesting"
                                           if impl.get("deterministic_factor_spanning") else
                                           "deterministic cross-sectional-sort executor; backtesting"
                                           if impl.get("deterministic_cross_sectional_sort") else
                                           "deterministic event-study executor; backtesting"
                                           if impl.get("deterministic_event_study") else
                                           "deterministic forecast-skill executor; backtesting"
                                           if impl.get("deterministic_forecast_skill") else
                                           "deterministic formulaic-signal executor; backtesting"
                                           if impl.get("deterministic_formulaic_signal") else
                                           "deterministic event-market executor; backtesting"
                                           if impl.get("deterministic_event_market") else
                                           "deterministic provided-series executor; backtesting"
                                           if impl.get("deterministic_provided_series")
                                           else "spec auto-implemented + validated on the live bundle; backtesting"
                                       )}
                # fall through to P7 with the freshly-built module
            elif impl.get("needs_review"):
                audit = impl.get("no_progress") or {}
                explanation = human_review_explanation(
                    "auto_impl_no_progress",
                    {
                        "reason": impl.get("reason"),
                        "attempts": audit.get("attempts_tried")
                        or audit.get("consecutive_attempts"),
                    },
                )
                rec["stages"]["P6"] = {"module_id": None, "spec_generated": True,
                                       "auto_implemented": False,
                                       "auto_impl_reason": str(impl.get("reason", ""))[:240],
                                       "auto_impl_no_progress": audit,
                                       "human_review": explanation,
                                       "spec_path": str(spec.get("_path", "")),
                                       "note": "ModuleSpec generated; auto-impl no-progress guard -> needs_review"}
                rec["stages"]["P6_auto_impl_no_progress"] = audit
                decisions.append(_needs_review(
                    claim, impl.get("reason"), rec, run_log,
                    metrics={"auto_impl_no_progress": audit},
                    review=explanation))
                continue
            else:
                rec["stages"]["P6"] = {"module_id": None, "spec_generated": True,
                                       "auto_implemented": False,
                                       "auto_impl_reason": str(impl.get("reason", ""))[:140],
                                       "spec_path": str(spec.get("_path", "")),
                                       "note": "ModuleSpec generated; auto-impl declined -> pending operator"}
                _append_jsonl(config.REVIEW_QUEUE,
                              {"type": "module_spec", "queued_at": _now(), "status": "pending",
                               "proposal_id": f"{claim.claim_id}-spec", "name": _short_name(claim),
                               "claim_id": claim.claim_id, "strategy_class": cls,
                               "spec_path": str(spec.get("_path", "")),
                               "statement": claim.statement, "meaning": claim.mechanism})
                decisions.append(_skip(claim, "module_unavailable",
                                       "spec generated; auto-impl declined ("
                                       + str(impl.get("reason", ""))[:60] + "); awaiting operator",
                                       rec, run_log))
                continue
        else:
            rec["stages"]["P6"] = {"module_id": getattr(module, "__module_id__", "unknown"),
                                   "spec_generated": False}
        if _claim_budget_exceeded(claim_started):
            decisions.append(_skip(claim, "timeout",
                                   "skipped: per-claim time budget exceeded before backtest",
                                   rec, run_log))
            continue
        ready_for_backtest.append({
            "claim": claim,
            "module": module,
            "rec": rec,
            "claim_i": _ci,
            "spec": spec,
        })

    _register_run_cohorts(ready_for_backtest, source_id, source)

    if claim_workers > 1 and ready_for_backtest:
        def _phase2_one(_ready: dict) -> dict:
            claim = _ready["claim"]
            module = _ready["module"]
            rec = _ready["rec"]
            _ci = _ready["claim_i"]
            spec = _ready.get("spec")
            local_bundle = _isolated_bundle_access(bundle)
            local_run_log = {**run_log, "claims": []}
            local_auto_fetch_attempted: set[str] = set()
            local_decisions = []
            claim_started = time.monotonic()
            local_provenance = provenance
            local_synthetic = synthetic
            with _defer_jsonl_events() as events:
                for _once in (None,):
                    # ---- P7: backtest (pre-existing OR just auto-implemented module) -- #
                    _progress("evaluate", detail=f"P7 backtest: {_short_name(claim)}",
                              paper=title, claim_i=_ci, claim_n=len(claims))
                    local_bundle.reset_access()
                    is_auto = bool(getattr(module, "__auto_generated__", False))
                    mres = _run_module_once(module, local_bundle, claim, cost_frac, is_auto)
                    if not isinstance(mres, dict) or not mres.get("ok"):
                        mres = mres if isinstance(mres, dict) else {
                            "ok": False, "reason": "module returned non-dict"}
                        if mres.get("needs_review"):
                            local_decisions.append(_needs_review(
                                claim, mres.get("reason") or "module requested review",
                                rec, local_run_log))
                            continue
                        if not _is_data_unavailable_reason(mres.get("reason")):
                            local_decisions.append(_engine_error(
                                claim, "module run",
                                RuntimeError(str(mres.get("reason") or "module returned not-ok")),
                                rec, local_run_log))
                            continue
                        retry = _auto_fetch_and_retry_missing_series(
                            module, local_bundle, claim, cost_frac, is_auto, mres,
                            local_auto_fetch_attempted)
                        if retry.get("attempted"):
                            rec["stages"]["P7_auto_fetch"] = {
                                "attempted": retry.get("attempted", []),
                                "added": retry.get("added", []),
                                "status": retry.get("status", "miss"),
                            }
                        if retry.get("retried"):
                            local_provenance = local_bundle.provenance_summary()
                            local_synthetic = local_bundle.any_synthetic()
                            mres = retry.get("mres")
                            if isinstance(mres, dict) and mres.get("ok"):
                                pass
                            elif not isinstance(mres, dict) or not _is_data_unavailable_reason(mres.get("reason")):
                                local_decisions.append(_engine_error(
                                    claim, "module run",
                                    RuntimeError(str(retry.get("reason")
                                                     or "module retry returned not-ok")),
                                    rec, local_run_log))
                                continue
                            else:
                                local_decisions.append(_needs_data(
                                    claim, retry.get("reason", mres.get("reason")), rec, local_run_log,
                                    auto_fetch_attempted=retry.get("attempted")))
                                continue
                        else:
                            local_decisions.append(_needs_data(
                                claim, mres.get("reason"), rec, local_run_log,
                                auto_fetch_attempted=retry.get("attempted")))
                            continue
                    if not isinstance(mres, dict) or not mres.get("ok"):
                        local_decisions.append(_needs_data(claim, mres.get("reason"), rec, local_run_log))
                        continue
                    try:
                        mcode_for_coverage = Path(getattr(module, "__file__", "") or "").read_text()
                    except Exception:  # noqa: BLE001
                        mcode_for_coverage = ""
                    coverage = fidelity.variable_coverage_check(mcode_for_coverage, spec)
                    if coverage.get("needs_review"):
                        rec["stages"]["P6_variable_coverage"] = coverage
                        local_decisions.append(_needs_review(
                            claim, coverage.get("reason"), rec, local_run_log,
                            metrics={"variable_coverage": coverage}))
                        continue
                    _missing = [k for k in ("net", "positions", "bars_per_year") if k not in mres]
                    if _missing:
                        local_decisions.append(_needs_data(
                            claim, f"data_unavailable: module result missing keys {_missing}",
                            rec, local_run_log))
                        continue
                    family = _family(claim, module, source, spec)
                    structured_family = _structured_strategy_family(claim, source, spec)
                    reg_schemes = {}
                    for _rk in ("btc_vol_regime", "btc_trend_regime"):
                        _rs = local_bundle.series.get(_rk)
                        if (_rs is not None and getattr(_rs, "available", False)
                                and getattr(_rs, "data", None) is not None):
                            reg_schemes[_rk] = _rs.data
                    if mres.get("regime_schemes"):
                        if not isinstance(mres["regime_schemes"], dict):
                            local_decisions.append(_needs_data(
                                claim, "data_unavailable: module regime_schemes must be a mapping",
                                rec, local_run_log))
                            continue
                        reg_schemes.update(mres["regime_schemes"])
                    elif mres.get("prices") is not None:
                        try:
                            reg_schemes.update(regime_lib.regime_schemes(mres["prices"]))
                        except Exception:  # noqa: BLE001
                            pass
                    reg_schemes = reg_schemes or None
                    try:
                        bt = p7_backtest.run_backtest(
                            claim.claim_id, mres["net"], mres["positions"], mres["bars_per_year"],
                            payoff=mres.get("payoff"), position_signed=mres.get("position_signed"),
                            cost_frac=cost_frac, wf_frame=mres.get("wf_frame"), family=family,
                            generation_source=_generation_source_for(claim, spec, source),
                            search_cohort_id=claim.search_cohort_id,
                            search_denominator=claim.search_denominator,
                            preregistered_single_cohort=_is_preregistered_single_cohort(claim, source),
                            declared_grid_size=p7_backtest.declared_grid_size(spec),
                            regime_schemes=reg_schemes,
                            declared_regime=claim.declared_regime)
                    except Exception as e:  # noqa: BLE001
                        local_decisions.append(_engine_error(claim, "P7 backtest", e, rec, local_run_log))
                        continue
                    bt.setdefault("claim_type", _authoritative_claim_type(claim, spec, source))
                    for _mk in ("event_market", "cost_provenance", "capacity_provenance"):
                        if _mk in mres:
                            bt[_mk] = mres[_mk]
                    if claim.source_type == "external_source":
                        sample_end = (claim.sample_period or {}).get("end") if claim.sample_period else None
                        data_end = None
                        post_years = None
                        try:
                            data_end_ts = mres["net"].dropna().index.max()
                            data_end = str(data_end_ts.date())
                            if sample_end:
                                post_years = (
                                    data_end_ts.tz_localize(None) - pd.Timestamp(sample_end)
                                ).days / 365.25
                        except Exception:  # noqa: BLE001
                            post_years = None
                        bt["post_sample"] = {
                            "sample_end": sample_end,
                            "data_end": data_end,
                            "post_years": post_years,
                            "declared": claim.sample_period is not None,
                        }
                    rec["stages"]["P7"] = bt
                    if _claim_budget_exceeded(claim_started):
                        local_decisions.append(_skip(
                            claim, "timeout", "skipped: per-claim time budget exceeded before verdict",
                            rec, local_run_log))
                        continue
                    syn_here = local_synthetic if is_auto else local_bundle.accessed_synthetic()
                    holdout = {}
                    try:
                        dec = stages.p8_verdict(claim, bt, holdout, syn_here)
                        if _maybe_attach_parameter_fragility(
                                bt, module, local_bundle, claim, cost_frac, spec, dec):
                            dec = stages.p8_verdict(claim, bt, holdout, syn_here)
                        with _HOLDOUT_LOCK:
                            dec, holdout = _maybe_consult_holdout(claim, bt, mres, dec, syn_here)
                        dec.metrics["corpus_isolation"] = stages.corpus_isolation(claim, reader)
                    except Exception as e:  # noqa: BLE001
                        local_decisions.append(_engine_error(claim, "P8 verdict", e, rec, local_run_log))
                        continue
                    rec["stages"]["holdout"] = holdout
                    if use_llm and getattr(config, "FIDELITY_CHECK", False):
                        _progress("evaluate", detail=f"fidelity check: {_short_name(claim)}",
                                  paper=title, claim_i=_ci, claim_n=len(claims))
                        try:
                            mcode = Path(getattr(module, "__file__", "") or "").read_text()
                        except Exception:  # noqa: BLE001
                            mcode = ""
                        fid = _assess_fidelity_safe(claim, mcode, spec)
                        bt["fidelity"] = fid
                        dec.metrics["fidelity"] = fid
                        dec.metrics["independent_verifier"] = bool(fid.get("independent_verifier", False))
                        if fid.get("checked") is False:
                            dec.metrics["fidelity_provenance"] = "unknown"
                        else:
                            base_provenance = dec.metrics.get("fidelity_provenance", "unknown")
                            dec.metrics["fidelity_provenance"] = (
                                f"{base_provenance}; independent_verifier="
                                f"{str(bool(fid.get('independent_verifier', False))).lower()}"
                            )
                        if not fid.get("verified", False):
                            dec.metrics["fidelity_unverified"] = True
                            dec.rationale += f". FIDELITY UNVERIFIED: {fid.get('note', 'inconclusive')}."
                            if dec.verdict == "research-supported":
                                dec.verdict = "watch"
                                dec.rationale += " Strongest verdict withheld."
                        if (not fid.get("faithful", False)
                                and fid.get("confidence", 0) >= config.FIDELITY_KILL_CONFIDENCE):
                            _persist_fidelity_rejection(claim, spec, fid)
                            dec.metrics["fidelity_suspect"] = True
                            dec.metrics["original_verdict"] = dec.verdict
                            dec.verdict = "cannot_replicate"
                            dec.kill_reason = "unfaithful_module"
                            dec.rationale += (
                                f". CANNOT_REPLICATE — module unfaithful to the claim "
                                f"(conf {fid['confidence']:.2f}): {fid.get('note', '')}")
                    rec["stages"]["P8"] = {"verdict": dec.verdict, "kill_reason": dec.kill_reason,
                                           "rationale": dec.rationale}
                    try:
                        from .. import explanations
                        _inputs = explanations.visible_inputs(
                            mres["net"], market=mres.get("market_returns"),
                            momentum=mres.get("momentum_proxy"), simpler_net=mres.get("simpler_net"),
                            visible_frac=p7_backtest.IS_FRAC + p7_backtest.OOS_FRAC)
                        competing = explanations.analyze(
                            _inputs["net"], mres["bars_per_year"],
                            market=_inputs["market"], momentum=_inputs["momentum"],
                            crisis_windows=mres.get("crisis_windows"),
                            simpler_net=_inputs["simpler_net"])
                    except Exception:  # noqa: BLE001
                        competing = []
                    rec["competing_explanations"] = competing
                    local_decisions.append(dec)
                    _queue_and_log(claim, dec, local_run_log, rec)
                    chart = charts.render_backtest_chart(claim.claim_id, _short_name(claim),
                                                         mres["net"], bt, dec.verdict)
                    analysis_record = {
                        "claim_id": claim.claim_id, "source_id": source_id, "source_title": title,
                        "statement": claim.statement[:240], "mechanism": claim.mechanism,
                        "verdict": dec.verdict,
                        "kill_reason": dec.kill_reason,
                        "metrics": {k: bt.get(k) for k in ("dsr", "psr", "oos_sharpe", "n_trades",
                                                           "n_oos", "three_fold", "regime",
                                                           "capacity_usd", "edge_t")},
                        "fidelity": dec.metrics.get("fidelity"),
                        "fidelity_provenance": dec.metrics.get("fidelity_provenance"),
                        "competing_explanations": competing,
                        "data_provenance": {
                            **(getattr(claim, "data_provenance", {}) or {}),
                            "data_domain": _data_domain(claim), "strategy_family": family,
                            "strategy_family_structured": structured_family,
                            "datasets": _accessed_keys(local_bundle, conservative_for_auto=is_auto),
                            "periods": ([{"start": str(local_bundle.requested_window[0]),
                                          "end": str(local_bundle.requested_window[1])}]
                                        if getattr(local_bundle, "requested_window", None) else []),
                            "fallback_substitutions": list(
                                getattr(local_bundle, "fallback_substitutions", [])),
                            "bundle": local_provenance,
                        },
                        "source_type": claim.source_type,
                        "declared_regime": claim.declared_regime,
                        "synthetic": syn_here, "chart": chart, "run_at": _now()}
                    _append_jsonl(config.ANALYSIS_INDEX, analysis_record)
                    try:
                        from ..concepts import extract_and_append
                        concept = extract_and_append(analysis_record, use_llm=bool(use_llm))
                        if concept:
                            rec["concept_id"] = concept["concept_id"]
                    except Exception as e:  # noqa: BLE001
                        rec["concept_error"] = f"{type(e).__name__}: {e}"[:160]
            return {"claim_i": _ci, "decisions": local_decisions,
                    "claims": list(local_run_log.get("claims", [])), "events": list(events),
                    "provenance": local_provenance}

        try:
            def _phase2_on_error(item, exc):  # P-1: isolate a worker crash to its own claim
                return {"claim_i": item["claim_i"], "claims": [], "events": [],
                        "decisions": [_engine_error(item["claim"], "parallel worker (phase2)", exc,
                                                    item["rec"], run_log)]}

            phase2_results = _run_claim_tasks(
                ready_for_backtest, claim_workers, _phase2_one, _phase2_on_error, claim_governor)
            for result in phase2_results:
                _emit_jsonl_events(result.get("events", []))
                decisions.extend(result.get("decisions", []))
                run_log["claims"].extend(result.get("claims", []))
                # Merge each worker's provenance (auto-fetched series land in the worker's bundle CLONE,
                # so the run-level summary must aggregate them or a parallel run under-reports what data
                # it used vs serial). Union of per-series keys; base keys are shared so update is safe.
                if isinstance(result.get("provenance"), dict):
                    provenance.update(result["provenance"])
        finally:
            _cleanup_run_cohorts(ready_for_backtest)
        ready_for_backtest = []

    try:
        for _ready in ready_for_backtest:
            claim = _ready["claim"]
            module = _ready["module"]
            rec = _ready["rec"]
            _ci = _ready["claim_i"]
            spec = _ready.get("spec")
            claim_started = time.monotonic()
            # ---- P7: backtest (pre-existing OR just auto-implemented module) -- #
            _progress("evaluate", detail=f"P7 backtest: {_short_name(claim)}",
                      paper=title, claim_i=_ci, claim_n=len(claims))
            bundle.reset_access()                 # track which series THIS module reads (per-verdict synthetic)
            is_auto = bool(getattr(module, "__auto_generated__", False))
            mres = _run_module_once(module, bundle, claim, cost_frac, is_auto)
            if not isinstance(mres, dict) or not mres.get("ok"):
                mres = mres if isinstance(mres, dict) else {"ok": False, "reason": "module returned non-dict"}
                if mres.get("needs_review"):
                    decisions.append(_needs_review(
                        claim, mres.get("reason") or "module requested review", rec, run_log))
                    continue
                if not _is_data_unavailable_reason(mres.get("reason")):
                    decisions.append(_engine_error(
                        claim, "module run", RuntimeError(str(mres.get("reason") or "module returned not-ok")),
                        rec, run_log))
                    continue
                # A module that can't get its data is NOT a falsified claim — it's a data
                # BLOCKER. Record a structured request to the data backlog and mark the claim
                # needs_data (re-runnable once the catalog gains the series), never kill it.
                retry = _auto_fetch_and_retry_missing_series(
                    module, bundle, claim, cost_frac, is_auto, mres, auto_fetch_attempted_series)
                if retry.get("attempted"):
                    rec["stages"]["P7_auto_fetch"] = {
                        "attempted": retry.get("attempted", []),
                        "added": retry.get("added", []),
                        "status": retry.get("status", "miss"),
                    }
                if retry.get("retried"):
                    provenance = bundle.provenance_summary()
                    synthetic = bundle.any_synthetic()
                    mres = retry.get("mres")
                    if isinstance(mres, dict) and mres.get("ok"):
                        pass
                    elif not isinstance(mres, dict) or not _is_data_unavailable_reason(mres.get("reason")):
                        decisions.append(_engine_error(
                            claim, "module run",
                            RuntimeError(str(retry.get("reason") or "module retry returned not-ok")),
                            rec, run_log))
                        continue
                    else:
                        decisions.append(_needs_data(
                            claim, retry.get("reason", mres.get("reason")), rec, run_log,
                            auto_fetch_attempted=retry.get("attempted")))
                        continue
                else:
                    decisions.append(_needs_data(
                        claim, mres.get("reason"), rec, run_log,
                        auto_fetch_attempted=retry.get("attempted")))
                    continue
            if not isinstance(mres, dict) or not mres.get("ok"):
                decisions.append(_needs_data(claim, mres.get("reason"), rec, run_log))
                continue
            try:
                mcode_for_coverage = Path(getattr(module, "__file__", "") or "").read_text()
            except Exception:  # noqa: BLE001
                mcode_for_coverage = ""
            coverage = fidelity.variable_coverage_check(mcode_for_coverage, spec)
            if coverage.get("needs_review"):
                rec["stages"]["P6_variable_coverage"] = coverage
                decisions.append(_needs_review(claim, coverage.get("reason"), rec, run_log,
                                               metrics={"variable_coverage": coverage}))
                continue
            # B-005: an ok-module may still omit required keys -> guard before subscript (no KeyError
            # aborting the paper). Treat a malformed result as a data blocker, not a crash.
            _missing = [k for k in ("net", "positions", "bars_per_year") if k not in mres]
            if _missing:
                decisions.append(_needs_data(
                    claim, f"data_unavailable: module result missing keys {_missing}", rec, run_log))
                continue
            family = _family(claim, module, source, spec)        # C1: strategy_class + data_domain
            structured_family = _structured_strategy_family(claim, source, spec)
            # Pre-registered point-in-time MARKET-regime labels for the kill-lens. PRIMARY source is
            # the bundle's regime catalog series (btc_vol_regime / btc_trend_regime) — the SAME labels
            # a module conditions on via bundle.get(), so the lens partitions by exactly what the
            # strategy could have used. A module may also hand back its own labels (regime_schemes) or
            # the price series it traded (prices) to derive more. All are exogenous + trailing -> not
            # data-snooping; each populated bucket still inflates the DSR trial count (no free pass).
            reg_schemes = {}
            for _rk in ("btc_vol_regime", "btc_trend_regime"):
                _rs = bundle.series.get(_rk)
                if _rs is not None and getattr(_rs, "available", False) and getattr(_rs, "data", None) is not None:
                    reg_schemes[_rk] = _rs.data
            if mres.get("regime_schemes"):
                if not isinstance(mres["regime_schemes"], dict):
                    decisions.append(_needs_data(
                        claim, "data_unavailable: module regime_schemes must be a mapping", rec, run_log))
                    continue
                reg_schemes.update(mres["regime_schemes"])
            elif mres.get("prices") is not None:
                try:
                    reg_schemes.update(regime_lib.regime_schemes(mres["prices"]))
                except Exception:  # noqa: BLE001 — a label-build failure must not abort the paper
                    pass
            reg_schemes = reg_schemes or None
            try:
                bt = p7_backtest.run_backtest(
                    claim.claim_id, mres["net"], mres["positions"], mres["bars_per_year"],
                    payoff=mres.get("payoff"), position_signed=mres.get("position_signed"),
                    cost_frac=cost_frac, wf_frame=mres.get("wf_frame"), family=family,
                    generation_source=_generation_source_for(claim, spec, source),
                    search_cohort_id=claim.search_cohort_id,
                    search_denominator=claim.search_denominator,
                    preregistered_single_cohort=_is_preregistered_single_cohort(claim, source),
                    declared_grid_size=p7_backtest.declared_grid_size(spec),
                    regime_schemes=reg_schemes,
                    declared_regime=claim.declared_regime)
            except Exception as e:  # noqa: BLE001 — C-005: one bad backtest must not abort the paper
                decisions.append(_engine_error(claim, "P7 backtest", e, rec, run_log))
                continue
            bt.setdefault("claim_type", _authoritative_claim_type(claim, spec, source))
            for _mk in ("event_market", "cost_provenance", "capacity_provenance"):
                if _mk in mres:
                    bt[_mk] = mres[_mk]
            if claim.source_type == "external_source":
                sample_end = (claim.sample_period or {}).get("end") if claim.sample_period else None
                data_end = None
                post_years = None
                try:
                    data_end_ts = mres["net"].dropna().index.max()
                    data_end = str(data_end_ts.date())
                    if sample_end:
                        post_years = (data_end_ts.tz_localize(None) - pd.Timestamp(sample_end)).days / 365.25
                except Exception:  # noqa: BLE001 - non-datetime or empty indices leave cap conservative
                    post_years = None
                bt["post_sample"] = {
                    "sample_end": sample_end,
                    "data_end": data_end,
                    "post_years": post_years,
                    "declared": claim.sample_period is not None,
                }
            rec["stages"]["P7"] = bt
            if _claim_budget_exceeded(claim_started):
                decisions.append(_skip(claim, "timeout",
                                       "skipped: per-claim time budget exceeded before verdict",
                                       rec, run_log))
                continue

            # Per-verdict synthetic flag: only synthetic if THIS module actually READ a synthetic
            # series (in-process tracking). Sandboxed modules can't be tracked from here -> fall
            # back to the conservative bundle-level flag.
            syn_here = synthetic if is_auto else bundle.accessed_synthetic()

            # ---- P8: verdict -------------------------------------------------- #
            # Run the verdict provisionally WITHOUT the holdout first, so all robustness KILL
            # gates (3-fold, regime, bootstrap, permutation) decide before the single-use holdout
            # is ever touched. Only a genuine survivor — passed every gate AND deflated-score >=
            # the research-supported band — earns the holdout consultation. This stops a
            # robustness-killed claim from wasting the lock (A-003); the gates never saw it (A-002).
            holdout = {}
            try:
                dec = stages.p8_verdict(claim, bt, holdout, syn_here)
                if _maybe_attach_parameter_fragility(
                        bt, module, bundle, claim, cost_frac, spec, dec):
                    dec = stages.p8_verdict(claim, bt, holdout, syn_here)
                with _HOLDOUT_LOCK:
                    dec, holdout = _maybe_consult_holdout(claim, bt, mres, dec, syn_here)
                dec.metrics["corpus_isolation"] = stages.corpus_isolation(claim, reader)
            except Exception as e:  # noqa: BLE001 — C-005: a verdict-stage error isolates to this claim
                decisions.append(_engine_error(claim, "P8 verdict", e, rec, run_log))
                continue
            rec["stages"]["holdout"] = holdout

            # ---- VERIFY: module-fidelity refuter (does the code faithfully test the claim?) -- #
            # The statistical gates can't see translation drift; only a reader comparing claim
            # to code can. An unfaithful module's verdict is untrustworthy -> flag it, and never
            # let an unfaithful module graduate to research-supported.
            if use_llm and getattr(config, "FIDELITY_CHECK", False):
                _progress("evaluate", detail=f"fidelity check: {_short_name(claim)}",
                          paper=title, claim_i=_ci, claim_n=len(claims))
                try:
                    mcode = Path(getattr(module, "__file__", "") or "").read_text()  # real path (B-011)
                except Exception:  # noqa: BLE001
                    mcode = ""
                fid = _assess_fidelity_safe(claim, mcode, spec)
                bt["fidelity"] = fid
                dec.metrics["fidelity"] = fid
                dec.metrics["independent_verifier"] = bool(fid.get("independent_verifier", False))
                if fid.get("checked") is False:
                    dec.metrics["fidelity_provenance"] = "unknown"
                else:
                    base_provenance = dec.metrics.get("fidelity_provenance", "unknown")
                    dec.metrics["fidelity_provenance"] = (
                        f"{base_provenance}; independent_verifier="
                        f"{str(bool(fid.get('independent_verifier', False))).lower()}"
                    )
                if not fid.get("verified", False):
                    dec.metrics["fidelity_unverified"] = True
                    dec.rationale += f". FIDELITY UNVERIFIED: {fid.get('note', 'inconclusive')}."
                    if dec.verdict == "research-supported":
                        dec.verdict = "watch"
                        dec.rationale += " Strongest verdict withheld."
                if not fid.get("faithful", False) and fid.get("confidence", 0) >= config.FIDELITY_KILL_CONFIDENCE:
                    # B3: an unfaithful module's result is NOT a trusted kill OR survivor — it's a
                    # distinct `cannot_replicate` verdict (a translation failure), excluded from the
                    # corpus, never a principle.
                    _persist_fidelity_rejection(claim, spec, fid)
                    dec.metrics["fidelity_suspect"] = True
                    dec.metrics["original_verdict"] = dec.verdict
                    dec.verdict = "cannot_replicate"
                    dec.kill_reason = "unfaithful_module"
                    dec.rationale += (f". CANNOT_REPLICATE — module unfaithful to the claim "
                                      f"(conf {fid['confidence']:.2f}): {fid.get('note', '')}")

            rec["stages"]["P8"] = {"verdict": dec.verdict, "kill_reason": dec.kill_reason,
                                   "rationale": dec.rationale}
            # Scientist depth is advisory and fail-soft: it refines the concept but cannot alter P8.
            try:
                from .. import explanations
                # WP3 is forbidden from reading the final locked holdout. Match P7's visible
                # IS+OOS window exactly, and align every optional explanatory series to it.
                _inputs = explanations.visible_inputs(
                    mres["net"], market=mres.get("market_returns"),
                    momentum=mres.get("momentum_proxy"), simpler_net=mres.get("simpler_net"),
                    visible_frac=p7_backtest.IS_FRAC + p7_backtest.OOS_FRAC)
                competing = explanations.analyze(
                    _inputs["net"], mres["bars_per_year"],
                    market=_inputs["market"], momentum=_inputs["momentum"],
                    crisis_windows=mres.get("crisis_windows"),
                    simpler_net=_inputs["simpler_net"])
            except Exception:  # noqa: BLE001
                competing = []
            rec["competing_explanations"] = competing
            decisions.append(dec)
            _queue_and_log(claim, dec, run_log, rec)

            # ---- Analysis report: chart the backtested equity curve + index it --- #
            chart = charts.render_backtest_chart(claim.claim_id, _short_name(claim),
                                                 mres["net"], bt, dec.verdict)
            analysis_record = {
                "claim_id": claim.claim_id, "source_id": source_id, "source_title": title,
                "statement": claim.statement[:240], "mechanism": claim.mechanism,
                "verdict": dec.verdict,
                "kill_reason": dec.kill_reason,
                "metrics": {k: bt.get(k) for k in ("dsr", "psr", "oos_sharpe", "n_trades",
                                                   "n_oos", "three_fold", "regime",
                                                   "capacity_usd", "edge_t")},
                "fidelity": dec.metrics.get("fidelity"),
                "fidelity_provenance": dec.metrics.get("fidelity_provenance"),
                "competing_explanations": competing,
                "data_provenance": {
                    **(getattr(claim, "data_provenance", {}) or {}),
                    "data_domain": _data_domain(claim), "strategy_family": family,
                    "strategy_family_structured": structured_family,
                    "datasets": sorted(
                        (getattr(bundle, "_accessed", None) or bundle.series.keys())
                        if is_auto else (getattr(bundle, "_accessed", None) or [])),
                    "periods": ([{"start": str(bundle.requested_window[0]),
                                  "end": str(bundle.requested_window[1])}]
                                if getattr(bundle, "requested_window", None) else []),
                    "fallback_substitutions": list(getattr(bundle, "fallback_substitutions", [])),
                    "bundle": provenance,
                },
                "source_type": claim.source_type,
                "declared_regime": claim.declared_regime,
                "synthetic": syn_here, "chart": chart, "run_at": _now()}
            _append_jsonl(config.ANALYSIS_INDEX, analysis_record)
            # WP1: concept extraction cannot abort a paper and has no path back into the verdict.
            try:
                from ..concepts import extract_and_append
                concept = extract_and_append(analysis_record, use_llm=bool(use_llm))
                if concept:
                    rec["concept_id"] = concept["concept_id"]
            except Exception as e:  # noqa: BLE001
                rec["concept_error"] = f"{type(e).__name__}: {e}"[:160]

    finally:
        _cleanup_run_cohorts(ready_for_backtest)

    claim_order = {getattr(c, "claim_id", ""): i for i, c in enumerate(claims)}
    decisions.sort(key=lambda d: claim_order.get(getattr(d, "claim_id", ""), len(claim_order)))

    # ---- P8 aggregate: principle proposal (sub-threshold if <3 same-class kills) #
    _progress("finalize", paper=title)
    # A kill from an unfaithful/unverified module is untrustworthy — it must NOT count toward
    # a principle (3 mis-impl kills could otherwise mint a false "principle"). (A-017)
    def _principle_eligible(d):
        m = getattr(d, "metrics", {}) or {}
        if m.get("fidelity_suspect") or m.get("fidelity_unverified"):
            return False                                   # A-017: unfaithful kills never form a principle
        if d.verdict == "kill" and m.get("replicated_in_sample") is False:
            return False                                   # B2: a kill we never reproduced in-sample isn't corpus-worthy
        return True
    trustworthy = [d for d in decisions if _principle_eligible(d)]
    principle = stages.propose_principle(trustworthy)
    run_log["principle_proposed"] = asdict(principle) if principle else None
    if principle:
        _append_jsonl(config.REVIEW_QUEUE,
                      {"type": "principle", "queued_at": _now(), **asdict(principle)})

    run_log["specs_generated"] = len(specs_generated)
    run_log["claims_extracted"] = len(claims)
    run_log["decisions"] = [{"claim_id": d.claim_id, "verdict": d.verdict,
                             "kill_reason": getattr(d, "kill_reason", None)}
                            for d in decisions if hasattr(d, "claim_id")]
    run_log["decision_metrics"] = {
        d.claim_id: {"resolution": (getattr(d, "metrics", {}) or {}).get("resolution")}
        for d in decisions if hasattr(d, "claim_id")
    }

    # ---- write report + dashboard live.json ------------------------------- #
    report_path = write_report(source_id, source.title, claims, decisions,
                               provenance, principle)
    run_log["report"] = str(report_path)
    run_log["provenance"] = provenance
    _write_live(source.title, source_id, decisions, provenance, principle, synthetic,
                specs_generated, p2_prov.get("mode", "unknown"), use_llm)
    run_log["idempotency"]["superseded_decisions"] = _supersede_decision_rows(source_id, run_id)
    _finalize_audit_run(audit_log, audit_path, run_log, decisions)
    _append_jsonl(config.ROOT / "runs.jsonl", run_log)
    _record_processed_source(source_id, paper_path, source.text_sha256)
    _progress(None)
    return run_log


# --- helpers --------------------------------------------------------------- #

def _kill(claim, reason, note, synthetic, rec, run_log):
    from ..brain import Decision
    dec = Decision(decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                   verdict="kill", kill_reason=reason,
                   rationale=(note or "") + (" (synthetic signal)" if synthetic else ""),
                   metrics={"synthetic_signal": synthetic}, revisit_at="2026-07-01")
    rec["stages"]["P8"] = {"verdict": "kill", "kill_reason": reason}
    run_log["claims"].append(rec)
    _queue_and_log(claim, dec, run_log, rec)
    return dec


def _short_name(claim) -> str:
    """A short human-readable label so a proposal can be referenced by name, not just id."""
    s = (getattr(claim, "applicable_strategy_class", "") or "").strip()
    if s and s.lower() not in ("", "unknown", "none"):
        return s[:60]
    words = (getattr(claim, "statement", "") or "").split()
    return (" ".join(words[:8])[:60]) or claim.claim_id


def _record_spec_inputs(rec: dict, spec: dict | None) -> None:
    try:
        inputs = (spec or {}).get("inputs", [])
        if not isinstance(inputs, list):
            return
        out: list[str] = []
        seen: set[str] = set()
        for raw in inputs:
            name = str(raw or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        if out:
            rec["inputs_requested"] = out
    except Exception:  # noqa: BLE001 — trace metadata only
        pass


def _series_ref_name(value) -> str:
    try:
        return fidelity._series_ref_name(value)
    except Exception:  # noqa: BLE001
        if isinstance(value, dict):
            return str(value.get("series") or value.get("base_series") or "").strip()
        return str(value or "").strip()


def _predictive_regression_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when a deterministic regression binding is uncertain."""
    if not isinstance(spec, dict) or spec.get("claim_type") != "predictive_regression":
        return None
    try:
        inputs = spec.get("inputs") or []
        predictor = spec.get("predictor") or (inputs[0] if inputs else "")
        target = spec.get("target") or (inputs[1] if len(inputs) > 1 else "")
        provenance = spec.get("binding_provenance") or {}
        predictor_prov = provenance.get("predictor") if isinstance(provenance, dict) else {}
        target_prov = provenance.get("target") if isinstance(provenance, dict) else {}
        predictor_prov = predictor_prov if isinstance(predictor_prov, dict) else {}
        target_prov = target_prov if isinstance(target_prov, dict) else {}
        unknowns = " ".join(str(u or "").lower() for u in (spec.get("unknowns") or []))
        unresolved = (
            not _series_ref_name(predictor)
            or not _series_ref_name(target)
            or str(predictor_prov.get("kind") or "").lower() == "unresolved"
            or str(target_prov.get("kind") or "").lower() == "unresolved"
            or "unresolved binding" in unknowns
            or "not both resolved" in unknowns
        )
        detail = {
            "reason": "unresolved binding" if unresolved else "unconfirmed binding",
            "predictor": predictor,
            "target": target,
            "binding_provenance": provenance,
            "confirmed": {"predictor": False, "target": False},
        }
        if unresolved:
            try:
                detail["confirmed"] = {
                    "predictor": bool(
                        predictor_prov
                        and str(predictor_prov.get("kind") or "").lower() != "unresolved"
                        and fidelity._binding_matches_spec_value(predictor_prov, predictor)
                        and fidelity._binding_score_ok(predictor_prov)
                    ),
                    "target": bool(
                        target_prov
                        and str(target_prov.get("kind") or "").lower() != "unresolved"
                        and fidelity._binding_matches_spec_value(target_prov, target)
                        and fidelity._binding_score_ok(target_prov)
                    ),
                }
            except Exception:  # noqa: BLE001
                pass
            return {
                "kind": "predictive_regression_binding_uncertain",
                "reason": "predictive_regression_binding_unresolved",
                "detail": detail,
                "explanation": human_review_explanation(
                    "predictive_regression_binding_uncertain", detail),
            }
        if provenance:
            text = fidelity._regression_claim_text(claim, spec)
            predictor_confirmed = bool(
                predictor_prov
                and fidelity._binding_matches_spec_value(predictor_prov, predictor)
                and fidelity._binding_score_ok(predictor_prov)
            )
            target_confirmed = bool(
                target_prov
                and fidelity._binding_matches_spec_value(target_prov, target)
                and fidelity._binding_score_ok(target_prov)
            )
            detail["confirmed"] = {
                "predictor": predictor_confirmed,
                "target": target_confirmed,
            }
            if not (
                predictor_confirmed
                and target_confirmed
                and fidelity._ordered_binding_provenance_verified(text, predictor_prov, target_prov)
            ):
                return {
                    "kind": "predictive_regression_binding_uncertain",
                    "reason": "predictive_regression_binding_unconfirmed",
                    "detail": detail,
                    "explanation": human_review_explanation(
                        "predictive_regression_binding_uncertain", detail),
                }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "predictive_regression_binding_uncertain",
            "reason": "predictive_regression_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation(
                "predictive_regression_binding_uncertain", detail),
        }
    return None


def _factor_spanning_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when a deterministic spanning binding is uncertain."""
    if not isinstance(spec, dict) or spec.get("claim_type") != "factor_spanning":
        return None
    try:
        inputs = spec.get("inputs") or []
        candidate = spec.get("candidate_factor") or (inputs[0] if inputs else "")
        benchmarks = spec.get("benchmark_factors") or inputs[1:]
        if isinstance(benchmarks, str):
            benchmarks = [benchmarks]
        provenance = spec.get("binding_provenance") or {}
        candidate_prov = provenance.get("candidate_factor") if isinstance(provenance, dict) else {}
        candidate_prov = candidate_prov if isinstance(candidate_prov, dict) else {}
        benchmark_set_prov = provenance.get("benchmark_set") if isinstance(provenance, dict) else {}
        benchmark_set_prov = benchmark_set_prov if isinstance(benchmark_set_prov, dict) else {}
        unknowns = " ".join(str(u or "").lower() for u in (spec.get("unknowns") or []))
        candidate_confirmed = bool(
            candidate_prov
            and str(candidate_prov.get("kind") or "").lower() != "unresolved"
            and fidelity._binding_matches_spec_value(candidate_prov, candidate)
            and fidelity._binding_score_ok(candidate_prov)
        )
        benchmark_confirmed = bool(
            str(spec.get("benchmark_set") or "").lower() in {"capm", "ff3", "ff5", "carhart"}
            and benchmarks
            and all(_series_ref_name(b) for b in benchmarks)
            and (
                not benchmark_set_prov
                or bool(benchmark_set_prov.get("confirmed", True))
            )
        )
        unresolved = (
            not _series_ref_name(candidate)
            or not benchmarks
            or str(candidate_prov.get("kind") or "").lower() == "unresolved"
            or "candidate factor series was not resolved" in unknowns
            or "unresolved binding" in unknowns
        )
        detail = {
            "reason": "unresolved binding" if unresolved else "unconfirmed binding",
            "candidate_factor": candidate,
            "benchmark_set": spec.get("benchmark_set"),
            "benchmark_factors": benchmarks,
            "binding_provenance": provenance,
            "confirmed": {
                "candidate_factor": candidate_confirmed,
                "benchmark_set": benchmark_confirmed,
            },
        }
        if unresolved or not (candidate_confirmed and benchmark_confirmed):
            return {
                "kind": "factor_spanning_binding_uncertain",
                "reason": (
                    "factor_spanning_binding_unresolved"
                    if unresolved else
                    "factor_spanning_binding_unconfirmed"
                ),
                "detail": detail,
                "explanation": human_review_explanation(
                    "factor_spanning_binding_uncertain", detail),
            }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "factor_spanning_binding_uncertain",
            "reason": "factor_spanning_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation(
                "factor_spanning_binding_uncertain", detail),
        }
    return None


def _panel_declared(value) -> bool:
    if isinstance(value, dict):
        raw = value.get("path") or value.get("table") or value.get("table_path") or value.get("panel_path")
    else:
        raw = value
    text = str(raw or "").strip()
    return bool(text and text not in {"returns_panel", "characteristic_panel"})


def _cross_sectional_sort_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when a deterministic sort binding is uncertain."""
    if not isinstance(spec, dict) or spec.get("claim_type") != "cross_sectional_sort":
        return None
    try:
        panel_inputs = spec.get("panel_inputs") or {}
        provenance = spec.get("binding_provenance") or {}
        characteristic = str(spec.get("characteristic") or "").strip()
        unknowns = " ".join(str(x).lower() for x in (spec.get("unknowns") or []))
        char_prov = provenance.get("characteristic") if isinstance(provenance, dict) else {}
        char_prov = char_prov if isinstance(char_prov, dict) else {}
        characteristic_confirmed = (
            bool(characteristic)
            and str(char_prov.get("kind") or "").lower() != "unresolved"
            and bool(char_prov.get("confirmed", True))
            and "characteristic was not resolved" not in unknowns
        )
        returns_declared = _panel_declared(panel_inputs.get("returns"))
        characteristic_declared = _panel_declared(panel_inputs.get("characteristic"))
        if characteristic_confirmed and returns_declared and characteristic_declared:
            return None
        detail = {
            "reason": "unresolved binding" if not characteristic_confirmed else "unconfirmed panel binding",
            "characteristic": characteristic,
            "panel_inputs": panel_inputs,
            "binding_provenance": provenance,
            "confirmed": {
                "characteristic": characteristic_confirmed,
                "returns_panel": returns_declared,
                "characteristic_panel": characteristic_declared,
            },
        }
        reason = (
            "cross_sectional_sort_binding_unresolved"
            if not characteristic_confirmed else
            "cross_sectional_sort_binding_unconfirmed"
        )
        return {
            "kind": "cross_sectional_sort_binding_uncertain",
            "reason": reason,
            "detail": detail,
            "explanation": human_review_explanation(
                "cross_sectional_sort_binding_uncertain", detail),
        }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "cross_sectional_sort_binding_uncertain",
            "reason": "cross_sectional_sort_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation(
                "cross_sectional_sort_binding_uncertain", detail),
        }


def _event_study_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when deterministic event-study bindings are uncertain."""
    if not isinstance(spec, dict) or spec.get("claim_type") != "event_study":
        return None
    try:
        inputs = spec.get("inputs") or []
        return_series = spec.get("return_series") or (inputs[0] if inputs else "")
        baseline = str(spec.get("baseline") or "mean_adjusted").strip().lower()
        market_series = spec.get("market_series") or ""
        provenance = spec.get("binding_provenance") or {}
        ret_prov = provenance.get("return_series") if isinstance(provenance, dict) else {}
        ret_prov = ret_prov if isinstance(ret_prov, dict) else {}
        unknowns = " ".join(str(x).lower() for x in (spec.get("unknowns") or []))
        window = spec.get("window")
        try:
            window_ok = (
                isinstance(window, (list, tuple))
                and len(window) >= 2
                and int(window[0]) <= int(window[1])
            )
        except (TypeError, ValueError):
            window_ok = False
        return_confirmed = bool(
            _series_ref_name(return_series)
            and str(ret_prov.get("kind") or "").lower() != "unresolved"
            and "return series was not resolved" not in unknowns
        )
        baseline_ok = baseline in {"mean_adjusted", "market_model"}
        market_ok = baseline != "market_model" or bool(_series_ref_name(market_series))
        if return_confirmed and window_ok and baseline_ok and market_ok:
            return None
        detail = {
            "reason": "unresolved binding",
            "return_series": return_series,
            "event_calendar": spec.get("event_calendar"),
            "window": window,
            "baseline": baseline,
            "market_series": market_series,
            "binding_provenance": provenance,
            "confirmed": {
                "return_series": return_confirmed,
                "window": window_ok,
                "baseline": baseline_ok,
                "market_series": market_ok,
            },
        }
        return {
            "kind": "event_study_binding_uncertain",
            "reason": "event_study_binding_unresolved",
            "detail": detail,
            "explanation": human_review_explanation("event_study_binding_uncertain", detail),
        }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "event_study_binding_uncertain",
            "reason": "event_study_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation("event_study_binding_uncertain", detail),
        }


def _forecast_benchmark_method(value) -> str:
    if isinstance(value, dict):
        value = value.get("method") or value.get("family") or value.get("benchmark") or value.get("kind")
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"random_walk", "rw", "naive", "persistence", "last_value"}:
        return "random_walk"
    if text in {"historical_mean", "expanding_mean", "mean", "expanding_historical_mean"}:
        return "historical_mean"
    return ""


def _forecast_skill_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when deterministic forecast-skill bindings are uncertain."""
    if not isinstance(spec, dict) or spec.get("claim_type") != "forecast_skill":
        return None
    try:
        inputs = spec.get("inputs") or []
        model = spec.get("model_forecast") or (inputs[0] if inputs else "")
        target = spec.get("target") or (inputs[1] if len(inputs) > 1 else "")
        benchmark = spec.get("benchmark") or spec.get("benchmark_forecast") or spec.get("benchmark_series") or ""
        benchmark_series = _series_ref_name(benchmark)
        benchmark_method = _forecast_benchmark_method(benchmark)
        provenance = spec.get("binding_provenance") or {}
        model_prov = provenance.get("model_forecast") if isinstance(provenance, dict) else {}
        target_prov = provenance.get("target") if isinstance(provenance, dict) else {}
        benchmark_prov = provenance.get("benchmark") if isinstance(provenance, dict) else {}
        model_prov = model_prov if isinstance(model_prov, dict) else {}
        target_prov = target_prov if isinstance(target_prov, dict) else {}
        benchmark_prov = benchmark_prov if isinstance(benchmark_prov, dict) else {}
        unknowns = " ".join(str(x).lower() for x in (spec.get("unknowns") or []))
        model_confirmed = bool(
            _series_ref_name(model)
            and str(model_prov.get("kind") or "").lower() != "unresolved"
            and (
                not model_prov
                or (
                    fidelity._binding_matches_spec_value(model_prov, model)
                    and fidelity._binding_score_ok(model_prov)
                )
            )
        )
        target_confirmed = bool(
            _series_ref_name(target)
            and str(target_prov.get("kind") or "").lower() != "unresolved"
            and (
                not target_prov
                or (
                    fidelity._binding_matches_spec_value(target_prov, target)
                    and fidelity._binding_score_ok(target_prov)
                )
            )
        )
        benchmark_confirmed = False
        if benchmark_series:
            benchmark_confirmed = bool(
                benchmark_series not in {_series_ref_name(model), _series_ref_name(target)}
                and (
                    not benchmark_prov
                    or (
                        str(benchmark_prov.get("kind") or "").lower() != "unresolved"
                        and fidelity._binding_matches_spec_value(benchmark_prov, benchmark_series)
                        and fidelity._binding_score_ok(benchmark_prov)
                    )
                )
            )
        elif benchmark_method:
            benchmark_confirmed = benchmark_method in {"random_walk", "historical_mean"}
        unresolved = (
            not model_confirmed
            or not target_confirmed
            or not benchmark_confirmed
            or "benchmark forecast was not resolved" in unknowns
            or "model forecast and target series were not both resolved" in unknowns
        )
        if not unresolved:
            return None
        detail = {
            "reason": "unresolved binding",
            "model_forecast": model,
            "target": target,
            "benchmark": benchmark,
            "binding_provenance": provenance,
            "confirmed": {
                "model_forecast": model_confirmed,
                "target": target_confirmed,
                "benchmark": benchmark_confirmed,
            },
        }
        return {
            "kind": "forecast_skill_binding_uncertain",
            "reason": "forecast_skill_binding_unresolved",
            "detail": detail,
            "explanation": human_review_explanation("forecast_skill_binding_uncertain", detail),
        }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "forecast_skill_binding_uncertain",
            "reason": "forecast_skill_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation("forecast_skill_binding_uncertain", detail),
        }


def _formulaic_signal_binding_review(claim, spec: dict | None) -> dict | None:
    """Return a needs_review payload when formulaic-signal bindings are structurally invalid."""
    del claim
    if not isinstance(spec, dict) or spec.get("claim_type") != "formulaic_signal":
        return None
    try:
        required = fidelity._formulaic_signal_names(spec)
        inputs = spec.get("inputs") or []
        declared = {str(x or "").strip() for x in inputs if str(x or "").strip()}
        signal = str(spec.get("signal") or "").strip()
        trade_series = str(spec.get("trade_series") or "").strip()
        position_map = str(spec.get("position_map") or "sign").strip().lower()
        confirmed = {
            "signal_parses": required is not None,
            "trade_series": bool(trade_series),
            "position_map": position_map in {"sign", "zscore_clip"},
            "inputs_exact": bool(required is not None and declared == required),
        }
        if all(confirmed.values()):
            return None
        detail = {
            "reason": "invalid formula or unresolved binding",
            "signal": signal,
            "trade_series": trade_series,
            "funding_pnl_series": spec.get("funding_pnl_series"),
            "position_map": position_map,
            "declared_inputs": sorted(declared),
            "required_inputs": sorted(required or []),
            "confirmed": confirmed,
        }
        return {
            "kind": "formulaic_signal_binding_uncertain",
            "reason": "formulaic_signal_binding_unresolved",
            "detail": detail,
            "explanation": human_review_explanation("formulaic_signal_binding_uncertain", detail),
        }
    except Exception as exc:  # noqa: BLE001
        detail = {"reason": f"binding confirmation failed: {type(exc).__name__}"}
        return {
            "kind": "formulaic_signal_binding_uncertain",
            "reason": "formulaic_signal_binding_check_error",
            "detail": detail,
            "explanation": human_review_explanation("formulaic_signal_binding_uncertain", detail),
        }


def _skip(claim, reason, note, rec, run_log):
    """Spec generated, no module yet — awaiting implementation. This is NOT a backtest
    verdict: it is a distinct 'pending_module' state. Logged to the decisions ledger for
    provenance, but deliberately NOT re-queued — the ModuleSpec proposal is the single
    Action Required item for this claim (prevents the duplicate + empty 'watch' entries)."""
    from ..brain import Decision
    dec = Decision(decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                   verdict="pending_module", kill_reason=None, rationale=note,
                   metrics={"skip_reason": reason}, revisit_at="2026-07-01")
    rec["stages"]["P8"] = {"verdict": "pending_module", "skip_reason": reason}
    run_log["claims"].append(rec)
    _append_jsonl(config.DECISIONS_LOG,
                  asdict(dec) | _decision_log_metadata(claim, run_log))   # log, do NOT queue
    _emit_trace(claim, dec, rec, run_log)
    return dec


def _cannot_replicate_unfaithful_spec(claim, spec: dict, fid: dict, rec, run_log):
    """Early fidelity block for a generated spec.

    This is not a statistical verdict and never promotes credibility. It only saves the
    wasted auto-impl/backtest when the refuter has already found a high-confidence
    translation failure in the spec itself.
    """
    from ..brain import Decision

    note = fid.get("note") or "; ".join(fid.get("divergences") or []) or "unfaithful spec"
    dec = Decision(
        decision_id=f"{claim.claim_id}-d1",
        claim_id=claim.claim_id,
        verdict="cannot_replicate",
        kill_reason="unfaithful_spec",
        rationale=(f"CANNOT_REPLICATE — generated ModuleSpec is unfaithful to the claim "
                   f"(conf {float(fid.get('confidence', 0) or 0):.2f}): {str(note)[:240]}"),
        metrics={
            "fidelity": fid,
            "fidelity_suspect": True,
            "independent_verifier": bool(fid.get("independent_verifier", False)),
            "fidelity_provenance": (
                "spec-fidelity; independent_verifier="
                f"{str(bool(fid.get('independent_verifier', False))).lower()}"
            ),
            "spec_path": str(spec.get("_path", "")),
            "claim_type": spec.get("claim_type"),
        },
        revisit_at="2026-07-01",
    )
    rec["stages"]["P6_pre_fidelity"] = {
        "blocked": True,
        "confidence": fid.get("confidence", 0),
        "divergences": fid.get("divergences", []),
        "spec_path": str(spec.get("_path", "")),
    }
    rec["stages"]["P8"] = {"verdict": "cannot_replicate", "kill_reason": "unfaithful_spec"}
    run_log["claims"].append(rec)
    _append_jsonl(config.DECISIONS_LOG, asdict(dec) | _decision_log_metadata(claim, run_log))
    _emit_trace(claim, dec, rec, run_log)
    return dec


def _open_data_requests() -> list[dict]:
    """Read the data-request backlog -> open requests, newest first, deduped by
    request_id (latest record wins; a later status:'sourced'/'closed' retires it)."""
    path = config.DATA_REQUESTS
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[r.get("request_id", line)] = r
    out = [r for r in latest.values() if r.get("status", "open") == "open"]
    out.sort(key=lambda r: r.get("requested_at", ""), reverse=True)
    return out


# Non-data tokens the missing-series extractor must never log as a dataset: English
# prose fragments and status markers an auto-impl emits in its reason string. Without
# this guard the backlog fills with phantom requests like 'the' / 'cannot_operationalize'
# (observed live), polluting the data shopping list the whole flywheel depends on.
_MISSING_SERIES_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
    "as", "at", "is", "are", "be", "this", "that", "these", "those", "it", "its",
    "long", "short", "leg", "legs", "side", "only", "both", "net", "gross",
    "cannot_operationalize", "unspecified", "none", "nan", "null", "tbd", "unknown", "the_",
})


@functools.lru_cache(maxsize=1)
def _catalog_series_names() -> frozenset:
    """Known catalog series names (fail-open to empty), to whitelist real requests."""
    try:
        dd = str(config.DATA_DIR)
        if dd not in sys.path:
            sys.path.insert(0, dd)
        import loader as catalog  # type: ignore
        return frozenset(catalog.available()) if hasattr(catalog, "available") else frozenset()
    except Exception:  # noqa: BLE001 — never break parsing on a catalog hiccup
        return frozenset()


def _looks_like_series(tok: str) -> bool:
    """Real series keys are snake_case / dotted / carry a digit; bare prose words don't."""
    return ("_" in tok) or ("." in tok) or any(c.isdigit() for c in tok)


def _parse_missing_series(reason: str) -> list[str]:
    """Pull the series names out of a module's 'data_unavailable: a, b, c' reason.

    Drops non-data tokens: stopwords/status-markers, and bare prose words that are
    neither a known catalog series nor structured like a series key. A legitimate
    request for data we do NOT hold (e.g. 'crsp_daily_stock_returns') still passes,
    because it is structured even though it is absent from the catalog.
    """
    r = str(reason or "")
    tail = r.split("data_unavailable:", 1)[1] if "data_unavailable:" in r else r
    catalog = _catalog_series_names()
    out, seen = [], set()
    for p in re.split(r"[,;]", tail):
        m = re.search(r"[a-zA-Z][a-zA-Z0-9_.]{2,}", p)
        if not m:
            continue
        tok = m.group(0).strip(" ._")
        low = tok.lower()
        if not tok or low in _MISSING_SERIES_STOPWORDS:
            continue
        if (tok in catalog or low in catalog or _looks_like_series(tok)) and low not in seen:
            seen.add(low)
            out.append(tok)
    return out or ["unspecified"]


def _missing_spec_inputs_from_bundle(spec: dict, bundle) -> list[str] | None:
    """Return declared spec inputs that the actual run bundle cannot provide.

    This deliberately asks DataBundle.get(), not the catalog, because module execution
    sees the bundle: catalog series plus bundle-native live/synthetic/vendor series and
    the contract's alias rules. Any uncertainty fails open to None so the existing module
    data_unavailable path remains the backstop.
    """
    had_accessed = False
    prior_accessed = set()
    try:
        if isinstance(spec, dict) and spec.get("claim_type") == "event_market_strategy":
            return []
        if isinstance(spec, dict) and spec.get("claim_type") == "cross_sectional_sort":
            return []
        if isinstance(spec, dict) and spec.get("claim_type") == "event_study":
            inputs = []
            return_name = spec.get("return_series") or ""
            if return_name:
                inputs.append(return_name)
            if str(spec.get("baseline") or "").strip().lower() == "market_model":
                market_name = spec.get("market_series") or ""
                if market_name:
                    inputs.append(market_name)
            spec = {**spec, "inputs": inputs}
        if isinstance(spec, dict) and spec.get("claim_type") == "forecast_skill":
            inputs = []
            model_name = spec.get("model_forecast") or ""
            target_name = spec.get("target") or ""
            if model_name:
                inputs.append(model_name)
            if target_name:
                inputs.append(target_name)
            benchmark = (
                spec.get("benchmark")
                or spec.get("benchmark_forecast")
                or spec.get("benchmark_series")
                or ""
            )
            benchmark_name = _series_ref_name(benchmark)
            if benchmark_name:
                inputs.append(benchmark_name)
            spec = {**spec, "inputs": inputs}
        if isinstance(spec, dict) and spec.get("claim_type") == "formulaic_signal":
            required = fidelity._formulaic_signal_names(spec)
            if required is not None:
                spec = {**spec, "inputs": sorted(required)}
        inputs = spec.get("inputs", []) if isinstance(spec, dict) else []
        if inputs is None:
            return []
        if not isinstance(inputs, list):
            return None
        resolver = getattr(bundle, "get", None)
        if not callable(resolver):
            return None
        had_accessed = hasattr(bundle, "_accessed")
        if had_accessed:
            prior_accessed = set(getattr(bundle, "_accessed") or set())
        missing: list[str] = []
        seen: set[str] = set()
        for raw in inputs:
            if isinstance(raw, dict) and raw.get("kind") == "derived_series":
                name = str(raw.get("base_series") or "").strip()
            elif isinstance(raw, dict):
                name = str(
                    raw.get("series")
                    or raw.get("id")
                    or raw.get("name")
                    or raw.get("key")
                    or ""
                ).strip()
            else:
                name = str(raw or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            series = resolver(name)
            if series is None or getattr(series, "available", True) is False:
                missing.append(name)
        return missing
    except Exception:  # noqa: BLE001 — verdict logic fails open on availability uncertainty
        return None
    finally:
        if had_accessed:
            object.__setattr__(bundle, "_accessed", prior_accessed)
        elif hasattr(bundle, "_accessed"):
            object.__setattr__(bundle, "_accessed", set())


def _run_module_once(module, bundle, claim, cost_frac, is_auto: bool):
    if is_auto:
        # untrusted model-written code -> run in the Docker sandbox, NEVER in-process
        return sandbox.run_in_container(getattr(module, "__file__", ""), bundle, claim, cost_frac)
    try:
        return module.run(bundle, claim, cost_frac)   # operator/trusted module, in-process
    except TypeError:
        # legacy signature: run(bundle, channel, cost_frac)
        try:
            return _run_legacy_module(module, bundle, claim, cost_frac)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"engine_error: legacy module raised {type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001 — a buggy module must not abort the whole paper (A-027)
        return {"ok": False, "reason": f"engine_error: module raised {type(e).__name__}: {e}"}


def _maybe_attach_parameter_fragility(bt: dict, module, bundle, claim, cost_frac, spec, dec) -> bool:
    """Attach parameter-fragility metrics only for provisional survivor candidates."""
    try:
        if getattr(dec, "verdict", None) not in ("watch", "research-supported"):
            return False
        gate = getattr(config, "FRAGILITY_GATE", {})
        if not gate.get("enabled", True):
            return False
        min_configs = int(gate.get("min_configs", 4))
        if p7_backtest.declared_grid_size(spec) < min_configs:
            return False
        if not getattr(module, "__supports_param_override__", False):
            return False
        bt["parameter_fragility"] = robustness.parameter_fragility(
            module, bundle, claim, cost_frac, spec, p7_backtest.OOS_FRAC)
        return True
    except Exception as exc:  # noqa: BLE001
        bt["parameter_fragility"] = {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}
        return True


def _is_data_unavailable_reason(reason) -> bool:
    return str(reason or "").startswith("data_unavailable:")


def _auto_fetch_and_retry_missing_series(
        module, bundle, claim, cost_frac, is_auto: bool, mres: dict,
        attempted_series: set[str] | None = None) -> dict:
    """Try one conservative vendor fetch pass for missing series, then rerun once.

    Any miss, disabled vendor, bad spec, failed fetch, or malformed result returns
    a non-retried status so the caller keeps today's needs_data behavior.
    """
    attempted: list[str] = []
    added: list[str] = []
    attempted_series = attempted_series if attempted_series is not None else set()
    try:
        from ..data.contract import Series
        from ..data import vendors
    except Exception:  # noqa: BLE001
        return {"retried": False, "attempted": attempted, "added": added, "status": "unavailable"}

    for missing in _parse_missing_series(mres.get("reason")):
        if not missing or missing in getattr(bundle, "series", {}):
            continue
        try:
            resolved = vendors.resolve_vendor_spec(missing)
        except Exception:  # noqa: BLE001
            resolved = None
        if resolved is None:
            # No DEFAULT_SERIES spec. If auto-source is enabled (opt-in), try to resolve +
            # ARCHIVE the requested series from a real source (fred/tiingo/coingecko) so it
            # is never re-queried. Fail-open + gated OFF by default -> no change otherwise.
            if missing not in attempted_series:
                try:
                    from ..data import client as _dataclient
                    if _dataclient.resolve_missing_series(bundle, missing) is not None:
                        attempted_series.add(missing)
                        attempted.append(missing)
                        added.append(missing)
                        continue
                except Exception:  # noqa: BLE001 — auto-source is fail-open
                    pass
            continue
        key, spec = resolved
        if key in attempted_series:
            continue
        attempted_series.add(key)
        attempted.append(key)
        if key in getattr(bundle, "series", {}):
            continue
        try:
            mod = vendors.enabled_adapters().get(spec.get("vendor"))
            if mod is None:
                continue
            fetched = mod.fetch(spec)
            if fetched is None:
                continue
            s, prov = fetched
            if s is None or len(s) == 0:
                continue
            if getattr(s.index, "tz", None) is None:
                s.index = s.index.tz_localize("UTC")
            else:
                s.index = s.index.tz_convert("UTC")
            grade = getattr(mod, "PROVENANCE_GRADE", "as_displayed")
            s.name = key
            bundle.series[key] = Series(
                key, s, f"{prov} [{grade}]", spec.get("unit", ""),
                note=f"vendor:auto_fetch:{mod.NAME}:{key}")
            added.append(key)
        except Exception:  # noqa: BLE001 — fail-open; one bad fetch is a miss
            continue

    if not added:
        return {"retried": False, "attempted": attempted, "added": added, "status": "miss"}

    try:
        bundle.reset_access()
        retry_res = _run_module_once(module, bundle, claim, cost_frac, is_auto)
    except Exception as e:  # noqa: BLE001
        retry_res = {"ok": False, "reason": f"data_unavailable: auto-fetch retry raised {type(e).__name__}: {e}"}
    if not isinstance(retry_res, dict):
        retry_res = {"ok": False, "reason": "module returned non-dict after auto-fetch retry"}
    return {"retried": True, "attempted": attempted, "added": added,
            "status": "retried", "mres": retry_res,
            "reason": retry_res.get("reason", mres.get("reason"))}


# FIX 3 / 2026-07-04 incident: minimum non-trivial paper size (chars) below which a
# zero-claim extraction is treated as a legitimate "nothing here" result rather than a
# suspicious engine failure. Guards trivial/blank test fixtures while still catching the
# regression class the incident surfaced: a real paper (funding_drift_claim-sized or
# larger) that silently produced zero claims.
MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION = 80


def _zero_extraction_is_suspicious(source, p2_prov: dict) -> bool:
    """True when a 0-claim extraction result must be a LOUD engine_error rather than a
    quiet "no claims extracted" note (FIX 3, 2026-07-04 decisions-loss incident).

    Two suspicious signatures, either sufficient on its own:
      1. mode == "fallback-after-error": extract_claims itself RAISED (network error,
         un-parseable/truncated JSON after retries, etc.) and run_source silently
         downgraded to the manual claims.py fallback, which returns [] whenever (as in
         every agentic/paper run) no hand-authored claims.py exists for this source. An
         ENGINE failure wearing a "no claims" costume, not a genuinely empty paper.
      2. mode == "llm" with n_extracted == 0 on a NON-TRIVIAL source: the model itself
         returned an empty claims list for real body text. This can be a correct
         judgment, but it is exactly the vocabulary-confusion failure documented for
         admin/pre-registration-styled sources (verdict/supersedes/kill_reason prose
         reads as an audit log, not a research paper) -- silently trusting either
         explanation is the fail-soft violation. Always surface it for a human.
    Fails open (False, i.e. quiet) for a trivial/empty source — nothing to extract in the
    first place — and never raises.
    """
    try:
        mode = str((p2_prov or {}).get("mode") or "")
        if mode == "fallback-after-error":
            return True
        n_chars = int(getattr(source, "n_chars", 0) or 0)
        return mode == "llm" and n_chars >= MIN_CHARS_FOR_LOUD_ZERO_EXTRACTION
    except Exception:  # noqa: BLE001 — the loud-failure guard must itself never crash the run
        return False


def _zero_extraction_engine_error(source, source_id: str, p2_prov: dict, run_log: dict):
    """Loud, human-readable engine_error decision for a suspicious zero-claim extraction.

    Written as a SOURCE-level decision (there is no Claim object — extraction itself is
    what failed) so the run can never look like a silent success. Deliberately does NOT
    call _supersede_decision_rows: no claim was re-adjudicated here, so no prior decision
    for this source is superseded by this non-result (that ordering is FIX 1)."""
    from ..brain import Decision
    claim_id = f"{source_id}-c0"
    mode = str((p2_prov or {}).get("mode") or "unknown")
    detail = str((p2_prov or {}).get("extraction_error") or (p2_prov or {}).get("error") or "").strip()
    retry_note = ""
    if (p2_prov or {}).get("retry_on_zero"):
        retry_note = " after retry-on-zero extraction hardening ran once"
    rationale = (
        f"engine error: P2 claim extraction returned ZERO claims for a non-trivial source "
        f"({getattr(source, 'n_chars', 0)} chars, p2_mode={mode}){retry_note}"
        + (f": {detail}" if detail else "")
        + ". Very likely an extraction failure (LLM/network hiccup, or a vocabulary-"
          "confusion misjudgment reading the source as an admin/report document rather "
          "than a research paper) — not an empty paper. Needs operator review; re-run "
          "once the cause is addressed."
    )
    dec = Decision(decision_id=f"{claim_id}-err", claim_id=claim_id, verdict="engine_error",
                   kill_reason=None, rationale=rationale[:600],
                   metrics={"stage": "P2 extraction", "p2_mode": mode,
                           "n_chars": getattr(source, "n_chars", 0)})
    rec = {"claim_id": claim_id,
           "source_id": source_id,
           "stages": {"P2": p2_prov,
                      "P8": {"verdict": "engine_error", "stage": "P2 extraction"}}}
    run_log["claims"].append(rec)
    meta = {"source_id": source_id,
            "run_id": (run_log.get("idempotency") or {}).get("run_id", ""),
            "logged_at": _now()}
    _append_jsonl(config.DECISIONS_LOG, asdict(dec) | meta)
    _emit_trace({"claim_id": claim_id, "source_id": source_id}, dec, rec, run_log)
    _append_jsonl(config.REVIEW_QUEUE,
                  {"type": "engine_error", "queued_at": _now(), "status": "pending",
                   "proposal_id": f"{claim_id}-err", "name": source_id,
                   "claim_id": claim_id, "claim_statement": "(P2 extraction produced no claims)",
                   "rationale": dec.rationale, **asdict(dec)})
    print(f"[penrose] engine error (P2 extraction) for {source_id}: zero claims extracted "
          f"(p2_mode={mode}); see decisions.jsonl / review_queue.jsonl ({claim_id}-err)",
          file=sys.stderr)
    return dec


def _needs_data(claim, reason, rec, run_log, auto_fetch_attempted=None):
    """Module returned data_unavailable -> record a structured data REQUEST
    and return a `needs_data` blocker decision. Not a kill (the claim isn't falsified, just
    untestable until the catalog gains the series). Logged, not queued to review — the data
    request itself is the actionable artifact, surfaced via the dashboard data-requests panel."""
    from ..brain import Decision
    missing = _parse_missing_series(reason)
    attempted = [str(x) for x in (auto_fetch_attempted or []) if str(x)]
    req = {"request_id": f"{claim.claim_id}-data",
           "claim_id": claim.claim_id,
           "source_id": getattr(claim, "source_id", ""),
           "strategy_class": getattr(claim, "applicable_strategy_class", "") or "",
           "statement": getattr(claim, "statement", "")[:240],
           "missing_series": missing,
           "raw_reason": str(reason or "")[:240],
           "status": "open",
           "requested_at": _now()}
    if attempted:
        req["auto_fetch_attempted"] = attempted
    _append_jsonl(config.DATA_REQUESTS, req)
    needs = ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "")
    auto_fetch_note = ""
    if attempted:
        auto_fetch_note = " Auto-fetch attempted: " + ", ".join(attempted[:4])
        if len(attempted) > 4:
            auto_fetch_note += "…"
        auto_fetch_note += "."
    metrics = {"missing_series": missing}
    if attempted:
        metrics["auto_fetch_attempted"] = attempted
    dec = Decision(decision_id=f"{claim.claim_id}-d1", claim_id=claim.claim_id,
                   verdict="needs_data", kill_reason=None,
                   rationale=f"untestable until the data catalog provides: {needs}. "
                             f"Logged to the F7b data-request backlog; re-runnable once sourced."
                             f"{auto_fetch_note}",
                   metrics=metrics, revisit_at="2026-07-01")
    rec["stages"]["P8"] = {"verdict": "needs_data", "missing_series": missing}
    if attempted:
        rec["stages"]["P8"]["auto_fetch_attempted"] = attempted
    run_log["claims"].append(rec)
    _append_jsonl(config.DECISIONS_LOG,
                  asdict(dec) | _decision_log_metadata(claim, run_log))   # log, do NOT queue
    _emit_trace(claim, dec, rec, run_log)
    return dec


def _needs_review(claim, reason, rec, run_log, metrics=None, review=None):
    """Soft-stop for implementation-fidelity defects that need a human/agent fix.

    This is not a falsification verdict. It means the implementation cannot be trusted to
    answer the claim, so P7/P8 are not allowed to manufacture a kill from it.
    """
    from ..brain import Decision
    note = str(reason or "implementation needs review")[:300]
    if review is None:
        review = human_review_explanation("generic", {"reason": note})
    review = {
        "what": str((review or {}).get("what") or "Routed to human review.")[:500],
        "why": str((review or {}).get("why") or note)[:1000],
        "action": str((review or {}).get("action") or "Inspect the review item, correct the blocker, then re-run.")[:500],
    }
    dec = Decision(decision_id=f"{claim.claim_id}-review", claim_id=claim.claim_id,
                   verdict="needs_review", kill_reason=None,
                   rationale=(
                       f"needs_review: {review['what']} Why: {review['why']} "
                       f"Action: {review['action']} Existing reason: {note}"
                   ),
                   metrics={**(metrics or {}), "human_review": review},
                   revisit_at="2026-07-01")
    rec["stages"]["P8"] = {"verdict": "needs_review", "reason": note,
                           "human_review": review}
    run_log["claims"].append(rec)
    _append_jsonl(config.DECISIONS_LOG,
                  asdict(dec) | _decision_log_metadata(claim, run_log))
    _emit_trace(claim, dec, rec, run_log)
    _append_jsonl(config.REVIEW_QUEUE,
                  {"type": "needs_review", "queued_at": _now(), "status": "pending",
                   "proposal_id": dec.decision_id, "name": _short_name(claim),
                   "claim_id": claim.claim_id, "claim_statement": claim.statement,
                   "rationale": dec.rationale, "human_review": review, **asdict(dec)})
    return dec


def _engine_error(claim, stage: str, err: BaseException, rec, run_log):
    """Internal failures are review items, not data blockers and not verdict kills."""
    from ..brain import Decision
    err_text = f"{type(err).__name__}: {err}"
    rationale = (
        f"engine error during {stage}: {err_text}"[:300]
        + " — needs operator attention; claim untested."
    )
    dec = Decision(decision_id=f"{claim.claim_id}-err", claim_id=claim.claim_id,
                   verdict="engine_error", kill_reason=None, rationale=rationale,
                   metrics={"stage": stage, "error": err_text[:500]})
    rec["stages"]["P8"] = {"verdict": "engine_error", "stage": stage, "error": err_text[:240]}
    run_log["claims"].append(rec)
    _append_jsonl(config.DECISIONS_LOG,
                  asdict(dec) | _decision_log_metadata(claim, run_log))
    _emit_trace(claim, dec, rec, run_log)
    _append_jsonl(config.REVIEW_QUEUE,
                  {"type": "engine_error", "queued_at": _now(), "status": "pending",
                   "proposal_id": f"{claim.claim_id}-err", "name": _short_name(claim),
                   "claim_id": claim.claim_id, "claim_statement": claim.statement,
                   "rationale": dec.rationale, **asdict(dec)})
    print(f"[penrose] engine error ({stage}) for {claim.claim_id}: {err_text}", file=sys.stderr)
    return dec


def _decision_log_metadata(claim, run_log) -> dict:
    source_id = getattr(claim, "source_id", "") or run_log.get("source_id", "")
    idempotency = run_log.get("idempotency", {}) if isinstance(run_log, dict) else {}
    out = {
        "source_id": source_id,
        "run_id": idempotency.get("run_id", ""),
        "logged_at": _now(),
        "strategy_class": getattr(claim, "applicable_strategy_class", "") or "",
    }
    try:
        family = normalize_strategy_family(getattr(claim, "strategy_family", None))
        if family is None:
            family = _structured_strategy_family(claim)
        out["strategy_family"] = family
    except Exception:  # noqa: BLE001
        pass
    return out


def _queue_and_log(claim, dec, run_log, rec=None) -> None:
    _append_jsonl(config.DECISIONS_LOG,
                  asdict(dec) | _decision_log_metadata(claim, run_log))
    _emit_trace(claim, dec, rec, run_log)
    _append_jsonl(config.REVIEW_QUEUE,
                  {"type": "decision", "queued_at": _now(), "status": "pending",
                   "proposal_id": dec.decision_id, "name": _short_name(claim),
                   "claim_statement": claim.statement, "meaning": claim.mechanism,
                   **asdict(dec)})
    if not any(c.get("claim_id") == claim.claim_id for c in run_log["claims"]):
        run_log["claims"].append({"claim_id": claim.claim_id, "verdict": dec.verdict})


def _known_classes(limit: int = 20) -> dict:
    """{primary strategy class: short description} for registered modules — passed to P2
    so the LLM reuses an existing class when a claim fits (controlled-vocabulary routing).

    EXCLUDES auto-generated modules (they're paper-specific and proliferate; left in, the
    prompt grows unboundedly and exhausts glm's thinking budget -> empty extractions) and
    caps the count so the P2 prompt stays small. Auto modules are still registered for
    exact-match routing in P6 — they just aren't advertised as reusable vocabulary."""
    out, seen = {}, set()
    for mod in REGISTRY.values():
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if getattr(mod, "__auto_generated__", False):
            continue
        cls = getattr(mod, "__strategy_class__", None)
        if cls and cls not in out:
            out[cls] = getattr(mod, "__description__", "") or cls
        if len(out) >= limit:
            break
    return out


def _run_legacy_module(module, bundle, claim, cost_frac):
    """Backward-compat: original macro_vol_btc signature was run(bundle, channel, cost).
    Infer the channel from the claim TEXT (not just the id) so routed LLM claims pick
    the right channel."""
    text = " ".join([getattr(claim, "claim_id", ""), getattr(claim, "statement", ""),
                     getattr(claim, "mechanism", "")]).lower()
    # C-008: a bare "rate" substring misroutes "unemployment rate" / "exchange rate" /
    # "volatility rate" to the fed channel. Require a monetary-policy qualifier instead, so only
    # an actual policy-rate claim routes to "fed".
    _fed_terms = ("fed", "monetary", "kxfed", "fed funds", "interest rate",
                  "policy rate", "rate hike", "rate cut", "fomc")
    channel = "fed" if any(k in text for k in _fed_terms) else "recession"
    return module.run(bundle, channel, cost_frac)


def _fidelity_unknown(error: Exception) -> dict:
    return {
        "faithful": False,
        "verified": False,
        "checked": False,
        "confidence": 0.0,
        "divergences": [],
        "independent_verifier": False,
        "error": f"{type(error).__name__}: {error}"[:240],
        "note": f"fidelity check unavailable: {type(error).__name__}: {error}"[:240],
    }


def _assess_fidelity_safe(claim, module_code: str, spec: dict | None = None) -> dict:
    try:
        fid = fidelity.assess(claim, module_code, spec=spec)
    except Exception as e:  # noqa: BLE001 — fidelity must not abort a source run
        return _fidelity_unknown(e)
    if not isinstance(fid, dict):
        return _fidelity_unknown(TypeError("fidelity assessor returned non-dict"))
    if "checked" not in fid:
        fid = dict(fid)
        note = str(fid.get("note", "") or "").lower()
        fid["checked"] = not (
            "not checked" in note
            or "inconclusive" in note
            or "errored" in note
            or "unavailable" in note
        )
    if "independent_verifier" not in fid:
        fid = dict(fid)
        fid["independent_verifier"] = False
    return fid


def _spec_fidelity_payload(spec: dict) -> str:
    return json.dumps({
        "module_spec_only": True,
        "strategy_class": spec.get("strategy_class"),
        "claim_type": spec.get("claim_type"),
        "claim_translation": spec.get("claim_translation"),
        "inputs": spec.get("inputs"),
        "signal_logic": spec.get("signal_logic"),
        "statistic_logic": spec.get("statistic_logic"),
        "kill_criterion": spec.get("kill_criterion"),
        "unknowns": spec.get("unknowns"),
    }, sort_keys=True, default=str)


def _assess_spec_fidelity_safe(claim, spec: dict) -> dict:
    try:
        fid = fidelity.assess(claim, _spec_fidelity_payload(spec), spec=spec)
    except Exception as e:  # noqa: BLE001 — pre-check must fail open into the normal path
        return _fidelity_unknown(e)
    if not isinstance(fid, dict):
        return _fidelity_unknown(TypeError("fidelity assessor returned non-dict"))
    if "checked" not in fid:
        fid = dict(fid)
        note = str(fid.get("note", "") or "").lower()
        fid["checked"] = not (
            "not checked" in note
            or "inconclusive" in note
            or "errored" in note
            or "unavailable" in note
        )
    if "independent_verifier" not in fid:
        fid = dict(fid)
        fid["independent_verifier"] = False
    return fid


def _fidelity_confidently_unfaithful(fid: dict) -> bool:
    return (
        isinstance(fid, dict)
        and fid.get("checked", True) is not False
        and fid.get("faithful") is False
        and float(fid.get("confidence", 0) or 0) >= config.FIDELITY_KILL_CONFIDENCE
    )


def _persist_fidelity_rejection(claim, spec: dict | None, fid: dict) -> None:
    try:
        claim_type = _authoritative_claim_type(claim, spec)
        strategy_class = (
            (spec or {}).get("strategy_class")
            or getattr(claim, "applicable_strategy_class", "")
            or "unspecified"
        )
        divergences = list(fid.get("divergences") or [])
        if not divergences and fid.get("note"):
            divergences = [fid.get("note")]
        fidelity_memory.append_rejection(
            strategy_class=strategy_class,
            claim_type=claim_type,
            divergences=divergences,
            note=str(fid.get("note", "")),
        )
    except Exception:  # noqa: BLE001
        pass


def _claim_budget_exceeded(started_at: float) -> bool:
    budget = float(getattr(config, "CLAIM_TIME_BUDGET_SECONDS", 0) or 0)
    return budget > 0 and (time.monotonic() - started_at) > budget


def _set_pipeline_status(status: str) -> None:
    """Update only the pipeline_status (+ updated_at) in live.json, preserving the
    rest. Lets the dashboard dot show running(green)/action_required(amber)/idle."""
    try:
        cur = json.loads(config.LIVE_JSON.read_text()) if config.LIVE_JSON.exists() else {}
    except Exception:  # noqa: BLE001
        cur = {}
    cur["pipeline_status"] = status
    cur["updated_at"] = _now()
    config.LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_JSON.write_text(json.dumps(cur, indent=2, default=str))


# Ordered macro-stages for the dashboard activity bar (the per-claim loop = "evaluate").
_STAGE_ORDER = ["ingest", "relevance", "extract", "evaluate", "finalize"]
_STAGE_LABEL = {
    "ingest": "P1 · ingesting & sanitizing",
    "relevance": "relevance gate",
    "extract": "P2 · extracting claims",
    "evaluate": "P3–P8 · evaluating claims",
    "finalize": "P8/P9 · principle + report",
}


def _progress(stage: str | None, detail: str = "", *, paper: str = "",
              claim_i: int | None = None, claim_n: int | None = None) -> None:
    """Write live per-stage progress for the dashboard activity panel. stage=None -> idle/done.
    Best-effort: never let progress I/O break a run."""
    try:
        with _PROGRESS_LOCK:
            if stage is None:
                payload = {"running": False, "updated_at": _now()}
            else:
                payload = {
                    "running": True,
                    "paper": paper,
                    "stage": stage,
                    "stage_no": (_STAGE_ORDER.index(stage) + 1) if stage in _STAGE_ORDER else 0,
                    "stage_total": len(_STAGE_ORDER),
                    "stage_label": _STAGE_LABEL.get(stage, stage),
                    "detail": detail,
                    "claim_i": claim_i,
                    "claim_n": claim_n,
                    "updated_at": _now(),
                }
            config.PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
            config.PROGRESS_JSON.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:  # noqa: BLE001
        pass


def _pending_count() -> int:
    try:
        if config.REVIEW_QUEUE.exists():
            return sum(json.loads(l).get("status") == "pending"
                       for l in config.REVIEW_QUEUE.read_text().splitlines() if l.strip())
    except Exception:  # noqa: BLE001
        pass
    return 0


def _finish_offdomain(source, source_id, rel, run_log) -> dict:
    """Short-circuit for a paper the relevance gate flagged off-domain: no P2+, no LLM spend.
    Write a minimal report + live.json so the dashboard explains the skip, mark the run, return."""
    reason = rel.get("reason", "no testable claim against penrose data domains")
    report = (f"**{source.title}**\n\nsource_id: {source_id}\n\n"
              f"Skipped before claim extraction — the relevance gate found no claim testable "
              f"against penrose's data domains.\n\nReason: {reason}\n")
    (config.REPORTS).mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS / f"{source_id}.md"
    report_path.write_text(report)
    run_log["report"] = str(report_path)
    run_log["off_domain"] = True
    live = {
        "pipeline_status": "action_required" if (_pending_count() or _open_data_requests()) else "idle",
        "updated_at": _now(),
        "source_title": f"{source_id} — {source.title}",
        "notice": f"Off-domain paper — skipped before extraction. {reason}",
        "data_provenance": {},
        "synthetic_warning": False,
        "p2_mode": "skipped-off-domain",
        "stats": {"sources": 1, "claims": 0, "kills": 0, "watch": 0, "supported": 0,
                  "pending_module": 0, "needs_data": 0, "principles": 0,
                  "modules": len(REGISTRY), "specs_pending": 0},
        "data_requests": _open_data_requests(),
        "decisions": [],
    }
    config.LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_JSON.write_text(json.dumps(live, indent=2, default=str))
    return run_log


def _write_live(source_title, source_id, decisions, provenance, principle,
                synthetic, specs_generated, p2_mode, use_llm) -> None:
    # synthetic warning fires only if a backtest actually CONSUMED synthetic data —
    # not just because the generic data bundle contains a synthetic series. A paper
    # whose claims were all killed/deferred before P7 never touched it.
    backtested_any = any(((getattr(d, "metrics", {}) or {}).get("dsr") is not None
                          or (getattr(d, "metrics", {}) or {}).get("oos_sharpe") is not None)
                         for d in decisions)
    synthetic_used = bool(synthetic and backtested_any)
    live = {
        "pipeline_status": "action_required" if (_pending_count() or _open_data_requests()) else "idle",
        "updated_at": _now(),
        "status_badge": "RESEARCH ENGINE — NO LIVE TRADING",
        "source_title": f"{source_id} — {source_title}",
        "data_provenance": provenance,
        "synthetic_warning": synthetic_used,
        "p2_mode": p2_mode,
        "llm_active": bool(use_llm),
        "stats": {
            "sources": 1,
            "claims": len(decisions),
            "kills": sum(getattr(d, "verdict", "") == "kill" for d in decisions),
            "watch": sum(getattr(d, "verdict", "") == "watch" for d in decisions),
            "supported": sum(getattr(d, "verdict", "") == "research-supported" for d in decisions),
            "pending_module": sum(getattr(d, "verdict", "") == "pending_module" for d in decisions),
            "needs_data": sum(getattr(d, "verdict", "") == "needs_data" for d in decisions),
            "engine_errors": sum(getattr(d, "verdict", "") == "engine_error" for d in decisions),
            "principles": 1 if principle else 0,
            "modules": len(REGISTRY),
            "specs_pending": specs_generated if isinstance(specs_generated, int) else len(specs_generated),
        },
        "data_requests": _open_data_requests(),
        "decisions": [
            {"claim_id": getattr(d, "claim_id", ""), "verdict": getattr(d, "verdict", ""),
             "kill_reason": getattr(d, "kill_reason", None),
             "rationale": getattr(d, "rationale", ""),
             "metrics": getattr(d, "metrics", {})}
            for d in decisions
        ],
    }
    config.LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.LIVE_JSON.write_text(json.dumps(live, indent=2, default=str))


def _run_and_report(paper: Path, use_llm: bool, *, force: bool = False,
                    max_claims: int | None = None,
                    max_claim_workers: int | str | None = None) -> None:
    try:
        out = run_source(
            paper, use_llm=use_llm, force=force, max_claims=max_claims,
            max_claim_workers=max_claim_workers)
    except BaseException:  # noqa: BLE001 — incl. KeyboardInterrupt/SystemExit (B-015): never leave 'running'
        _set_pipeline_status("error")
        _progress(None)
        raise
    print(json.dumps({
        "run_at": out["run_at"], "source_id": out.get("source_id"),
        "claims_extracted": len(out.get("claims", [])),
        "specs_generated": out.get("specs_generated", 0),
        "report": out.get("report"),
        "p2_mode": out.get("p2", {}).get("mode"),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(prog="penrose-run")
    ap.add_argument("--paper", help="path to PDF; default = next UNPROCESSED inbox/ paper")
    ap.add_argument("--all", action="store_true",
                    help="process EVERY unprocessed inbox paper this invocation")
    ap.add_argument("--no-llm", action="store_true",
                    help="force fallback (claims.py / stub P3) even if API key is set")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if this source was already processed; prior rows are superseded")
    ap.add_argument("--max-claims", type=int,
                    help=("process at most N extracted claims from this source; setting "
                          "PENROSE_CLAIM_TIME_BUDGET_SECONDS makes budget skips wall-clock-dependent"))
    ap.add_argument("--workers",
                    help=("per-claim worker threads; default PENROSE_MAX_CLAIM_WORKERS or 1, "
                          f"accepts an int or auto; ints clamped to {MAX_CLAIM_WORKERS}"))
    args = ap.parse_args()
    use_llm = (not args.no_llm)

    if args.paper:
        _run_and_report(Path(args.paper) if Path(args.paper).exists()
                        else _find_paper(args.paper), use_llm, force=args.force,
                        max_claims=args.max_claims, max_claim_workers=args.workers)
        return

    if args.all:
        done = _processed_set()
        pending = [p for p in _inbox_pdfs() if p.name not in done]
        if not pending:
            print(json.dumps({"status": "all_processed", "inbox": len(_inbox_pdfs())}))
            return
        for i, paper in enumerate(pending, 1):
            print(f"[penrose] ({i}/{len(pending)}) {paper.name}", file=sys.stderr)
            try:
                _run_and_report(
                    paper, use_llm, force=args.force, max_claims=args.max_claims,
                    max_claim_workers=args.workers)
            except Exception as e:  # noqa: BLE001 — one bad paper must not stop the batch
                # Record the failure VISIBLY (not silent data loss, A-025), then keep
                # this invocation moving to the next paper.
                _append_jsonl(config.ROOT / "failed_papers.jsonl",
                              {"paper": paper.name, "error": str(e)[:300], "at": _now()})
                print(json.dumps({"paper": paper.name, "error": str(e)[:200]}), file=sys.stderr)
        return

    paper = _find_paper(None)
    if paper is None:
        print(json.dumps({"status": "all_processed",
                          "inbox": len(_inbox_pdfs()),
                          "note": "every inbox paper already run; `make reset` to reprocess"}))
        return
    _run_and_report(
        paper, use_llm, force=args.force, max_claims=args.max_claims,
        max_claim_workers=args.workers)


if __name__ == "__main__":
    main()
