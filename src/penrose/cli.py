"""penrose command-line interface — the single operations layer.

`pyproject.toml` wires this to the `penrose` command via [project.scripts]. Read-only commands
(verdicts / data-requests / status) only read JSON state and import nothing heavy, so they stay
fast and work even when the backtest stack can't import. `run` pulls in the full pipeline.

  penrose run [--paper P] [--all] [--no-llm]   ingest + evaluate a paper (default: next inbox paper)
  penrose verdicts [-n N]                       recent backtested verdicts (reports/analysis_index)
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
    from . import config
    rows = _read_jsonl(config.ANALYSIS_INDEX)[-args.n:]
    if not rows:
        print("No backtested verdicts yet (reports/analysis_index.jsonl is empty)."); return 0
    print(f"{'verdict':<18} {'syn':<4} {'claim_id':<26} statement")
    print("-" * 100)
    for r in rows:
        syn = "yes" if r.get("synthetic") else "no"
        print(f"{str(r.get('verdict')):<18} {syn:<4} {str(r.get('claim_id'))[:26]:<26} "
              f"{str(r.get('statement',''))[:50]}")
    return 0


def _cmd_data_requests(args) -> int:
    from . import config
    rows = [r for r in _read_jsonl(config.DATA_REQUESTS) if r.get("status", "open") == "open"]
    # dedupe by claim_id, keep latest
    latest = {r.get("claim_id"): r for r in rows}
    if not latest:
        print("No open data requests."); return 0
    print(f"Open F7b data blockers ({len(latest)}):")
    for r in latest.values():
        miss = ", ".join(r.get("missing_series", []) or [])[:70]
        print(f"  {str(r.get('claim_id'))[:30]:<30} needs: {miss}")
    return 0


def _cmd_status(args) -> int:
    from . import config
    live = config.ROOT / "dashboard" / "live.json"
    if not live.exists():
        print("idle (no dashboard/live.json)"); return 0
    try:
        d = json.loads(live.read_text())
    except (json.JSONDecodeError, OSError):
        print("unknown"); return 1
    print(f"pipeline: {d.get('pipeline_status', 'unknown')}  ({d.get('status_badge', '')})")
    print(f"updated:  {d.get('updated_at', '?')}")
    st = d.get("stats") or {}
    if st:
        print(f"stats:    {json.dumps(st)}")
    return 0


def _cmd_run(args) -> int:
    # Heavy: pulls the full pipeline + backtest stack. Lazy-imported so the read-only commands
    # above don't pay for it (and still work if the stats stack can't import).
    from .pipeline import run as runmod
    argv = []
    if args.paper:
        argv += ["--paper", args.paper]
    if args.all:
        argv += ["--all"]
    if args.no_llm:
        argv += ["--no-llm"]
    sys.argv = ["penrose-run"] + argv
    runmod.main()
    return 0


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="penrose",
                                 description="Falsification-first research pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="ingest + evaluate a paper")
    p.add_argument("--paper", help="path to a PDF/MD; default = next unprocessed inbox paper")
    p.add_argument("--all", action="store_true", help="process every unprocessed inbox paper")
    p.add_argument("--no-llm", action="store_true", help="force the no-LLM fallback path")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("verdicts", help="recent backtested verdicts")
    p.add_argument("-n", type=int, default=20, help="how many to show (default 20)")
    p.set_defaults(fn=_cmd_verdicts)

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

    args = ap.parse_args(argv)
    # CLI error boundary: expected runtime problems (missing API key, missing run/file)
    # become a clean one-line message, never a raw traceback. Programming errors still surface.
    try:
        return args.fn(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"penrose: {e}")
        return 1
    except KeyboardInterrupt:
        print("penrose: interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
