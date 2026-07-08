"""MCP server for Penrose. Optional: ``pip install penrose[mcp]``.

EXPOSE OPERATIONS, NOT ESCAPE HATCHES. The default server is strictly READ-ONLY:
it reads data Penrose already wrote (verdicts, proposals, data-requests, status)
or computes read-only distillation. An explicit management mode adds guarded
proposal/bookkeeping tools. NOTHING here crosses P9, writes the approved brain /
PRINCIPLES_LOG / trusted brainstore / decisions-as-approved-knowledge, runs a
model-written module outside the Docker sandbox, or touches the holdout outside
the guarded pipeline path.

The ``mcp`` dependency is OPTIONAL and imported lazily here ONLY, so importing
``penrose`` / the CLI / the eval suite never requires it.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_MCP_MISSING_MSG = (
    "penrose MCP requires the optional 'mcp' dependency.\n"
    "install with:  pip install penrose[mcp]"
)

_READ_ONLY_TOOLS = {
    "penrose_verdicts",
    "penrose_proposals",
    "penrose_principles",
    "penrose_data_requests",
    "penrose_status",
}
_MANAGEMENT_TOOLS = {
    "penrose_fetch_verdict",
    "penrose_register_cohort",
    "penrose_run_claim",
}
_MAX_COHORT_STRATEGIES = 1000
_MAX_RUN_CLAIMS = 20
# A single cohort can only OVER-deflate (n_trials is a max), so this bounds a denial-of-service
# where a caller registers a huge denominator to force everything in a family to `underpowered`.
_MAX_COHORT_DENOMINATOR = 10_000

# Lazily populated so importing this module stays light and never requires mcp.
views = None
run_source = None
register_trials = None


def _err(message: str, **extra: Any) -> dict:
    return {"ok": False, "error": str(message), **extra}


def _management_imports() -> tuple[Any, Any, Any]:
    """Load the only Penrose internals reachable from management tools."""
    global views, run_source, register_trials
    if views is None:
        from . import views as _views
        views = _views
    if run_source is None:
        from .pipeline.run import run_source as _run_source
        run_source = _run_source
    if register_trials is None:
        from .pipeline.p7_backtest import register_trials as _register_trials
        register_trials = _register_trials
    return views, run_source, register_trials


def _read_imports() -> Any:
    global views
    if views is None:
        from . import views as _views
        views = _views
    return views


def _single_verdict(claim_id: str) -> dict:
    try:
        cid = str(claim_id or "").strip()
        if not cid:
            return _err("claim_id is required")
        _views = _read_imports()
        for row in reversed(list(_views.verdicts(1000) or [])):
            if str(row.get("claim_id") or "") == cid:
                return {"ok": True, "claim_id": cid, "verdict": row}
        return _err(f"no verdict found for claim_id={cid}", claim_id=cid)
    except Exception as e:  # noqa: BLE001
        return _err(f"verdict lookup failed: {type(e).__name__}: {e}")


def _register_cohort(cohort_id: str, family: str, denominator: int,
                     strategies: list[str] | None = None) -> dict:
    try:
        cohort = str(cohort_id or "").strip()
        fam = str(family or "").strip()
        if not cohort:
            return _err("cohort_id is required")
        if not fam:
            return _err("family is required")
        try:
            denom = int(denominator)
        except (TypeError, ValueError):
            return _err("denominator must be an integer >= 1")
        if denom < 1:
            return _err("denominator must be >= 1")
        if denom > _MAX_COHORT_DENOMINATOR:
            return _err(f"denominator is capped at {_MAX_COHORT_DENOMINATOR} "
                        "(a larger cohort would over-deflate the whole family)")
        if strategies is None:
            strategies = []
        if not isinstance(strategies, list):
            return _err("strategies must be a list of strings")
        raw_strategies = [str(s or "").strip() for s in strategies]
        clean = [s for s in raw_strategies if s]
        if len(clean) > _MAX_COHORT_STRATEGIES:
            return _err(f"strategies is capped at {_MAX_COHORT_STRATEGIES} entries")
        rows = [
            {
                "strategy": strategy,
                "family": fam,
                "generation_source": "mcp_management",
                "search_cohort_id": cohort,
                "search_denominator": denom,
            }
            for strategy in (clean or [f"{cohort}:registered"])
        ]
        _, _, _register_trials = _management_imports()
        _register_trials(rows)
        return {"ok": True, "registered": denom, "cohort_id": cohort}
    except Exception as e:  # noqa: BLE001
        return _err(f"cohort registration failed: {type(e).__name__}: {e}")


def _claim_obj(row: dict, idx: int) -> SimpleNamespace:
    statement = str(row.get("statement") or row.get("claim") or "").strip()
    if not statement:
        raise ValueError("claim.statement is required")
    claim_id = str(row.get("claim_id") or row.get("id") or f"mcp-claim-{idx}").strip()
    return SimpleNamespace(
        claim_id=claim_id,
        statement=statement,
        mechanism=str(row.get("mechanism") or statement).strip(),
        scope=str(row.get("scope") or "unspecified").strip(),
        horizon=str(row.get("horizon") or "unspecified").strip(),
        source_id=str(row.get("source_id") or "mcp_inline_claim").strip(),
        source_span=str(row.get("source_span") or statement).strip(),
        claimed_metric_quote=str(row.get("claimed_metric_quote") or "").strip(),
        applicable_strategy_class=str(row.get("applicable_strategy_class") or "").strip(),
        source_type="external_source",
        search_cohort_id=row.get("search_cohort_id"),
        search_denominator=row.get("search_denominator"),
        raw_hypothesis_id=row.get("raw_hypothesis_id"),
        data_provenance=dict(row.get("data_provenance") or {}),
        declared_regime=row.get("declared_regime"),
    )


def _normalize_claims(claim: Any, max_claims: int) -> list[SimpleNamespace]:
    rows = claim if isinstance(claim, list) else [claim]
    if not rows or rows == [None]:
        raise ValueError("claim is required when paper_path is not provided")
    if len(rows) > max_claims:
        raise ValueError(f"claim list exceeds max_claims cap ({max_claims})")
    out = []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("claim must be a mapping or list of mappings")
        out.append(_claim_obj(row, idx))
    return out


def _proposal_rows(run_log: dict) -> list[dict]:
    rows = run_log.get("decisions")
    if not isinstance(rows, list):
        rows = run_log.get("claims") if isinstance(run_log.get("claims"), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["status"] = "proposed"
        out.append(item)
    return out


def _inbox_paper(paper_path: str) -> Path | None:
    """Resolve `paper_path` iff it is an existing file INSIDE config.INBOX; else None.
    Prevents the run tool from being an arbitrary-file-read primitive on an agent surface."""
    try:
        from . import config
        inbox = Path(config.INBOX).resolve()
        p = Path(str(paper_path).strip()).resolve()
        if not p.is_file():
            return None
        p.relative_to(inbox)          # raises ValueError if p is not under inbox
        return p
    except Exception:  # noqa: BLE001 — any resolution error -> refuse
        return None


def _run_claim(paper_path: str | None = None, claim: Any = None,
               max_claims: int = 5) -> dict:
    try:
        try:
            cap = int(max_claims)
        except (TypeError, ValueError):
            return _err("max_claims must be an integer")
        if cap < 1:
            return _err("max_claims must be >= 1")
        if cap > _MAX_RUN_CLAIMS:
            return _err(f"max_claims is capped at {_MAX_RUN_CLAIMS}")
        has_paper = bool(str(paper_path or "").strip())
        has_claim = claim is not None
        if has_paper == has_claim:
            return _err("provide exactly one of paper_path or claim")
        _, _run_source, _ = _management_imports()
        if has_paper:
            # Defense-in-depth for an agent-facing surface: restrict to the inbox so this run
            # tool can't be used as an arbitrary-file-read primitive (audit LOW). The operator
            # (or agent) stages papers in the inbox; paths outside it are refused.
            resolved = _inbox_paper(paper_path)
            if resolved is None:
                return _err("paper_path must be a file inside the penrose inbox/ directory")
            run_log = _run_source(
                resolved,
                claims_override=None,
                source_type="external_source",
                max_claims=cap,
            )
        else:
            claims_override = _normalize_claims(claim, cap)
            text = "\n\n".join(c.statement for c in claims_override)
            with tempfile.TemporaryDirectory(prefix="penrose-mcp-") as td:
                path = Path(td) / "inline_claim.txt"
                path.write_text(text)
                run_log = _run_source(
                    path,
                    claims_override=claims_override,
                    source_type="external_source",
                    max_claims=cap,
                )
        if not isinstance(run_log, dict):
            return _err("run_source returned a non-dict result")
        return {
            "ok": True,
            "proposals": _proposal_rows(run_log),
            "run": run_log,
            "note": "MCP run output is proposal/bookkeeping only; P9 human review remains required.",
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"claim run failed: {type(e).__name__}: {e}")


def build_server(management: bool = False):
    """Build the FastMCP server. Raises ImportError (with a clear install
    message) if the optional ``mcp`` package is absent."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # optional dependency
        raise ImportError(_MCP_MISSING_MSG) from e

    _views = _read_imports()

    server = FastMCP("penrose")

    @server.tool()
    def penrose_verdicts(limit: int = 20) -> list[dict]:
        """Recent backtested verdicts (kill / underpowered / watch / research-supported
        + kill_reason + key metrics). READ-ONLY; approval stays human (P9)."""
        return _views.verdicts(limit)

    @server.tool()
    def penrose_proposals() -> list[dict]:
        """The propose-only principle proposals (status: proposed). READ-ONLY; promotion
        to the approved brain requires the human P9 review path, not this server."""
        return _views.proposals()

    @server.tool()
    def penrose_principles() -> list[dict]:
        """Distilled cross-run advisory principle proposals from the full corpus.
        READ-ONLY compute; never writes the approved brain."""
        return _views.principles()

    @server.tool()
    def penrose_data_requests() -> list[dict]:
        """Open `needs_data` blockers — the 'one dataset away from testable' list.
        READ-ONLY."""
        return _views.data_requests()

    @server.tool()
    def penrose_status() -> dict:
        """Current pipeline status. READ-ONLY."""
        return _views.status()

    @server.tool()
    def penrose_triage(top: int = 15, source: str | None = None) -> dict:
        """Failure-cluster analysis across the trace corpus: verdict distribution, per-stage drop-off,
        and the top recurring failure signatures (where claims die, and why). READ-ONLY."""
        return _views.triage(top=top, source=source)

    if management:
        _management_imports()

        @server.tool()
        def penrose_fetch_verdict(claim_id: str) -> dict:
            """Read one verdict proposal by claim id. Produces read-only proposal context;
            P9 human review remains required."""
            return _single_verdict(claim_id)

        @server.tool()
        def penrose_register_cohort(cohort_id: str, family: str, denominator: int,
                                    strategies: list[str] | None = None) -> dict:
            """Register deflation bookkeeping for a disclosed cohort. Produces bookkeeping
            only; P9 human review remains required."""
            return _register_cohort(cohort_id, family, denominator, strategies)

        @server.tool()
        def penrose_run_claim(paper_path: str | None = None, claim: Any = None,
                              max_claims: int = 5) -> dict:
            """Run a paper or inline claim through the guarded pipeline. Produces verdict
            proposals only; P9 human review remains required."""
            return _run_claim(paper_path=paper_path, claim=claim, max_claims=max_claims)

        @server.tool()
        def penrose_mine_principles() -> dict:
            """Distill cross-run principle PROPOSALS from the full corpus and persist them to the
            propose-only store (status: proposed). Produces proposals ONLY — promotion into the
            approved brain is the human P9 path, NEVER this server."""
            from .learning import persist_distilled_proposals
            out = persist_distilled_proposals()
            return {**out, "distilled": len(out.get("distilled") or [])}

    return server


def main() -> int:
    parser = argparse.ArgumentParser(prog="penrose-mcp")
    parser.add_argument("--management", action="store_true",
                        help="enable guarded proposal/bookkeeping management tools")
    args = parser.parse_args()
    management = args.management or os.environ.get("PENROSE_MCP_MANAGEMENT") == "1"
    try:
        server = build_server(management=management)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    if management:
        print("[penrose-mcp] management mode enabled: proposals/bookkeeping only; P9 stays human",
              file=sys.stderr)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
