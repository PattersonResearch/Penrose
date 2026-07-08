"""penrose command-line interface — the single operations layer.

`pyproject.toml` wires this to the `penrose` command via [project.scripts]. Read-only commands
(verdicts / data-requests / status) only read JSON state and import nothing heavy, so they stay
fast and work even when the backtest stack can't import. `run` pulls in the full pipeline.

  penrose run [--paper P] [--all] [--no-llm]   ingest + evaluate a paper (default: next inbox paper)
  penrose verdicts [-n N]                       recent backtested verdicts (reports/analysis_index)
  penrose triage [--json] [--top N]             trace drop-offs and recurring failure clusters
  penrose data-requests                         open data blockers (what a claim needs)
  penrose status                                pipeline status badge
  penrose eval                                  run the planted-strategy eval suite (ground truth)
  penrose calibrate {placebo,injection,synth}   run a calibration control
  penrose dream [-n N] [--generate-only]         registered hypothesis generation + P3-P9 triage
  penrose synthesize [-n N] [--generate-only]    corpus-grounded registered discovery
  penrose confirm <run_id>                       test frozen candidates on blind reserve
  penrose brain-rebuild                          rebuild native BrainStore from flat files
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
import json
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _cmd_verdicts(args) -> int:
    from . import views
    rows = views.verdicts(args.n)
    if not rows:
        print("No backtested verdicts yet (reports/analysis_index.jsonl is empty)."); return 0
    print(f"{'verdict':<18} {'syn':<4} {'claim_id':<26} statement")
    print("-" * 100)
    for r in rows:
        syn = "yes" if r.get("synthetic") else "no"
        print(f"{str(r.get('verdict')):<18} {syn:<4} {str(r.get('claim_id'))[:26]:<26} "
              f"{str(r.get('statement') or '')[:50]}")
    return 0


def _cmd_data_requests(args) -> int:
    from . import views
    rows = views.data_requests()
    if not rows:
        print("No open data requests."); return 0
    print(f"Open F7b data blockers ({len(rows)}):")
    for r in rows:
        miss = ", ".join(r.get("missing_series", []) or [])[:70]
        print(f"  {str(r.get('claim_id'))[:30]:<30} needs: {miss}")
    return 0


def _cmd_triage(args) -> int:
    from . import config
    from .trace import load_trace_rows, triage_report

    rows, loaded_from = load_trace_rows(Path(config.TRACES), Path(config.DECISIONS_LOG))
    if not rows:
        msg = "No traces or decisions found yet (reports/traces.jsonl and decisions.jsonl are empty)."
        if args.json:
            print(json.dumps({"status": "empty", "message": msg}, sort_keys=True))
        else:
            print(msg)
        return 0
    report = triage_report(rows, top=args.top, source=args.source)
    report["input"] = loaded_from
    if args.json:
        print(json.dumps(report, sort_keys=True))
        return 0

    source_note = f" source={args.source}" if args.source else ""
    print(f"Trace triage ({report['total']} claims{source_note}; input={loaded_from})")
    print("")
    print("Verdict distribution:")
    for verdict, count in report["verdict_distribution"].items():
        print(f"  {verdict:<20} {count}")
    print("")
    print("Per-stage drop-off:")
    for stage, count in report["stage_dropoff"].items():
        print(f"  {stage:<24} {count}")
    print("")
    print("Top recurring failure clusters:")
    if not report["failure_clusters"]:
        print("  none")
        return 0
    print(f"  {'count':>5} {'signature':<16} {'verdict':<18} {'exit_stage':<22} example")
    for cluster in report["failure_clusters"]:
        reason = cluster.get("kill_reason") or cluster.get("gate_outcome") or ""
        print(f"  {cluster['count']:>5} {str(cluster['failure_signature'])[:16]:<16} "
              f"{str(cluster.get('verdict') or '')[:18]:<18} "
              f"{str(cluster.get('exit_stage') or '')[:22]:<22} "
              f"{str(cluster.get('example_claim_id') or '')}  {str(reason)[:48]}")
    return 0


def _cmd_status(args) -> int:
    from . import views
    d = views.status()
    if d.get("pipeline_status") == "idle" and d.get("note"):
        print("idle (no dashboard/live.json)"); return 0
    if d.get("pipeline_status") == "unknown" and "status_badge" not in d:
        print("unknown"); return 1
    print(f"pipeline: {d.get('pipeline_status', 'unknown')}  ({d.get('status_badge', '')})")
    print(f"updated:  {d.get('updated_at') or '?'}")
    st = d.get("stats") or {}
    if st:
        print(f"stats:    {json.dumps(st)}")
    return 0


def _cmd_run(args) -> int:
    # Heavy: pulls the full pipeline + backtest stack. Lazy-imported so the read-only commands
    # above don't pay for it (and still work if the stats stack can't import).
    from .pipeline import run as runmod
    if args.all and (args.json or args.claims):
        raise ValueError("--all cannot be combined with --json or --claims")
    if args.json or args.claims:
        paper = _resolve_run_paper(args.paper, args.claims, runmod)
        if paper is None:
            print(json.dumps({
                "source_id": None,
                "verdicts": [],
                "principle": None,
                "status": "all_processed",
                "inbox": len(runmod._inbox_pdfs()),
                "note": "every inbox paper already run; `make reset` to reprocess",
            }, sort_keys=True))
            return 0
        claims_override = _load_claims_file(args.claims, _source_id_for(paper)) if args.claims else None
        out = runmod.run_source(
            paper,
            use_llm=not args.no_llm,
            claims_override=claims_override,
            force=args.force,
            max_claims=args.max_claims,
            max_claim_workers=args.workers,
        )
        if args.json:
            print(json.dumps(_run_json_object(out), sort_keys=True, default=str))
        else:
            print(json.dumps({
                "run_at": out["run_at"], "source_id": out.get("source_id"),
                "claims_extracted": len(out.get("claims", [])),
                "specs_generated": out.get("specs_generated", 0),
                "report": out.get("report"),
                "p2_mode": out.get("p2", {}).get("mode"),
            }, indent=2))
        return 0
    argv = []
    if args.paper:
        argv += ["--paper", args.paper]
    if args.all:
        argv += ["--all"]
    if args.no_llm:
        argv += ["--no-llm"]
    if args.force:
        argv += ["--force"]
    if args.max_claims is not None:
        argv += ["--max-claims", str(args.max_claims)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    sys.argv = ["penrose-run"] + argv
    runmod.main()
    return 0


def _source_id_for(path: Path) -> str:
    return path.stem or "claims-file"


def _resolve_run_paper(paper_arg: str | None, claims_arg: str | None, runmod) -> Path | None:
    if paper_arg:
        paper = Path(paper_arg)
        if not paper.exists():
            raise FileNotFoundError(f"paper not found: {paper}")
        return paper
    if claims_arg:
        claims_path = Path(claims_arg)
        if not claims_path.exists():
            raise FileNotFoundError(f"claims file invalid: not found: {claims_path}")
        return claims_path
    paper = runmod._find_paper(None)
    return paper


def _load_claims_file(path_arg: str, fallback_source_id: str):
    from .brain import Claim
    path = Path(path_arg)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"claims file invalid: not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"claims file invalid: malformed JSON at line {e.lineno} column {e.colno}") from None
    except OSError as e:
        raise ValueError(f"claims file invalid: cannot read {path}: {e}") from None
    if not isinstance(raw, list):
        raise ValueError("claims file invalid: top-level JSON value must be a list")
    claim_fields = {f.name: f for f in fields(Claim)}
    required = [
        f.name for f in fields(Claim)
        if f.default is MISSING and f.default_factory is MISSING and f.name != "source_id"
    ]
    claims = []
    for i, item in enumerate(raw, 1):
        prefix = f"claims file invalid: claim {i}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        unknown = sorted(set(item) - set(claim_fields))
        if unknown:
            raise ValueError(f"{prefix} has unknown field(s): {', '.join(unknown)}")
        missing = [name for name in required if name not in item]
        if missing:
            raise ValueError(f"{prefix} missing required field(s): {', '.join(missing)}")
        payload = dict(item)
        payload.setdefault("source_id", fallback_source_id)
        try:
            claims.append(Claim(**payload))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{prefix} invalid: {e}") from None
    return claims


def _run_json_object(out: dict) -> dict:
    skipped = bool((out.get("idempotency") or {}).get("skipped"))
    metrics_by_claim = out.get("decision_metrics") or {}
    verdicts = []
    for row in list(out.get("decisions") or []):
        item = dict(row)
        if item.get("claim_id") in metrics_by_claim:
            item["resolution"] = (metrics_by_claim.get(item.get("claim_id")) or {}).get("resolution")
        verdicts.append(item)
    return {
        "source_id": out.get("source_id"),
        "verdicts": verdicts,
        "principle": out.get("principle_proposed"),
        "status": "skipped" if skipped else "complete",
    }


def _json_error_object(message: str) -> dict:
    return {"source_id": None, "verdicts": [], "principle": None,
            "status": "error", "error": message}


def _run_repo_script(script_rel: str) -> int:
    """Run a script that ships with the cloned repo. Fails gracefully (no raw Errno 2)
    when penrose was installed non-editably and the repo scripts aren't on disk."""
    import subprocess
    from . import config
    path = config.ROOT / "scripts" / script_rel
    if not path.exists():
        print(f"penrose: cannot find {script_rel} on disk (looked in {path.parent}).")
        print("  This command runs scripts that ship with the cloned repository, so it needs")
        print("  an editable install from a clone:")
        print("    git clone <repo> && cd Penrose && pip install -e .")
        return 1
    return subprocess.run([sys.executable, str(path)]).returncode


def _cmd_eval(args) -> int:
    return _run_repo_script("eval_suite.py")


def _cmd_calibrate(args) -> int:
    script = {
        "placebo": "calibration_placebo.py",
        "injection": "calibration_injection.py",
        "synth": "calibration_synthesizer.py",
    }[args.control]
    return _run_repo_script(script)


def _cmd_dream(args) -> int:
    from .dream import run_dream
    out = run_dream(n=args.n, generate_only=args.generate_only, run_id=args.run_id)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("status") != "failed" else 1


def _cmd_synthesize(args) -> int:
    from .synthesize import run_synthesis
    out = run_synthesis(n=args.n, generate_only=args.generate_only, run_id=args.run_id)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("status") != "failed" else 1


def _cmd_confirm(args) -> int:
    from .confirmation import confirm_run
    out = confirm_run(args.run_id)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("status") == "complete" else 2


def _cmd_brain_rebuild(args) -> int:
    from . import brainstore
    counts = brainstore.rebuild_from_flat_files()
    print(json.dumps(counts, sort_keys=True))
    return 0


def _cmd_proposals(args) -> int:
    from .brain import read_proposals
    rows = read_proposals()
    if args.json:
        print(json.dumps(rows, sort_keys=True))
        return 0
    if not rows:
        print("No proposed principles.")
        return 0
    print(f"{'status':<10} {'n':<4} {'domain':<24} {'kill_reason':<22} statement")
    print("-" * 100)
    for r in rows:
        print(f"{str(r.get('status', 'proposed')):<10} {str(r.get('n_observations', '')):<4} "
              f"{str(r.get('domain', ''))[:24]:<24} {str(r.get('kill_reason', ''))[:22]:<22} "
              f"{str(r.get('statement', ''))[:48]}")
    return 0


def _cmd_principles(args) -> int:
    from .learning import persist_distilled_proposals
    out = persist_distilled_proposals()  # guarded distill+persist (CR-2/CR2-1); shared with the MCP tool
    if args.json:
        print(json.dumps(out, sort_keys=True))
    else:
        print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="penrose",
                                 description="Falsification-first research pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="ingest + evaluate a paper")
    p.add_argument("--paper", help="path to a PDF/MD; default = next unprocessed inbox paper")
    p.add_argument("--all", action="store_true", help="process every unprocessed inbox paper")
    p.add_argument("--no-llm", action="store_true", help="force the no-LLM fallback path")
    p.add_argument("--force", action="store_true",
                   help="re-run sources and supersede prior decision rows for that source_id")
    p.add_argument("--max-claims", type=int,
                   help=("process at most N extracted claims; setting "
                         "PENROSE_CLAIM_TIME_BUDGET_SECONDS makes budget skips wall-clock-dependent"))
    p.add_argument("--workers",
                   help="per-claim worker threads; default min(4, auto) (auto-reduced on small hardware); accepts int or 'auto'")
    p.add_argument("--json", action="store_true",
                   help="print one parseable JSON result object to stdout")
    p.add_argument("--claims", help="path to a JSON list of structured Claim objects")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("verdicts", help="recent backtested verdicts")
    p.add_argument("-n", type=int, default=20, help="how many to show (default 20)")
    p.set_defaults(fn=_cmd_verdicts)

    p = sub.add_parser("triage", help="analyze per-claim trace drop-offs and failure clusters")
    p.add_argument("--json", action="store_true", help="print a stable machine-readable report")
    p.add_argument("--top", type=int, default=15, help="maximum recurring failure clusters to show")
    p.add_argument("--source", help="filter to one source_id")
    p.set_defaults(fn=_cmd_triage)

    sub.add_parser("data-requests", help="open F7b data blockers").set_defaults(fn=_cmd_data_requests)
    sub.add_parser("status", help="pipeline status").set_defaults(fn=_cmd_status)
    sub.add_parser("eval", help="run the planted-strategy eval suite").set_defaults(fn=_cmd_eval)

    p = sub.add_parser("calibrate", help="run a calibration control")
    p.add_argument("control", choices=["placebo", "injection", "synth"])
    p.set_defaults(fn=_cmd_calibrate)

    p = sub.add_parser("dream", help="generate a registered hypothesis search and triage it")
    p.add_argument("-n", type=int, default=10, help="pre-registered generation budget (default 10)")
    p.add_argument("--generate-only", action="store_true",
                   help="persist + register candidates, but do not run P3-P9")
    p.add_argument("--run-id", help="stable id for replay/testing; existing ids are idempotent")
    p.set_defaults(fn=_cmd_dream)

    p = sub.add_parser("synthesize", help="generate a registered corpus-grounded discovery search")
    p.add_argument("-n", type=int, default=10, help="pre-registered generation budget")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--run-id", help="stable id for replay/testing")
    p.set_defaults(fn=_cmd_synthesize)

    p = sub.add_parser("confirm", help="confirm a frozen synthesis on the blind reserve")
    p.add_argument("run_id")
    p.set_defaults(fn=_cmd_confirm)

    sub.add_parser("brain-rebuild", help="rebuild native BrainStore from flat files").set_defaults(
        fn=_cmd_brain_rebuild)
    p = sub.add_parser("proposals", help="read propose-only principle proposals")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_proposals)
    p = sub.add_parser("principles", help="distill cross-run proposed principles")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_principles)
    p = sub.add_parser("distill", help="alias for principles")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_principles)

    args = ap.parse_args(argv)
    # CLI error boundary: expected runtime problems (missing API key, missing run/file)
    # become a clean one-line message, never a raw traceback. Programming errors still surface.
    try:
        return args.fn(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        if getattr(args, "json", False):
            print(json.dumps(_json_error_object(str(e)), sort_keys=True))
        else:
            print(f"penrose: {e}")
        return 1
    except KeyboardInterrupt:
        print("penrose: interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
