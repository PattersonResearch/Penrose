"""Reset penrose to a fresh state — archive, never hard-delete.

Wipes the accumulated run + brain state (decisions, kills, typed edges, the review
queue, reports, ingested papers, ledgers) so you can iterate from clean, but moves
everything into reset_archive/<timestamp>/ first. Nothing is lost; you can untar a
prior brain to roll back.

Leaves penrose in the same state a fresh clone would be in (then re-initialises an
empty brain so it's immediately usable). Run: `make reset`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLAT = ["runs.jsonl", "decisions.jsonl", "review_queue.jsonl", "data_requests.jsonl",
        "processed_papers.json", "backtest_ledger.tsv", ".holdout_burned",
        "dashboard/live.json", "dashboard/progress.json"]
DIRS = ["reports", "archives/papers", "run_archive"]   # archive contents of these (incl. reports/charts + analysis_index)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ROOT / "reset_archive" / ts
    dest.mkdir(parents=True, exist_ok=True)
    moved = []

    # brain (atoms / kills / typed edges) -> tarball, then MOVE aside (don't delete).
    # We keep a recoverable backup until re-init succeeds; if re-init fails we can
    # restore it, and main() exits non-zero so the failure isn't silently ignored.
    brain = ROOT / ".brain"
    brain_backup = None
    if brain.exists():
        with tarfile.open(dest / "brain.tgz", "w:gz") as t:
            t.add(brain, arcname=".brain")
        brain_backup = ROOT / ".brain.reset-bak"
        if brain_backup.exists():
            shutil.rmtree(brain_backup)
        shutil.move(str(brain), str(brain_backup))
        moved.append(".brain (atoms, kills, edges) -> brain.tgz")

    for rel in FLAT:
        p = ROOT / rel
        if p.exists():
            shutil.move(str(p), str(dest / Path(rel).name))
            moved.append(rel)

    for rel in DIRS:
        d = ROOT / rel
        if d.exists() and any(d.iterdir()):
            sub = dest / rel.replace("/", "__")
            sub.mkdir(parents=True, exist_ok=True)
            for item in list(d.iterdir()):
                shutil.move(str(item), str(sub / item.name))
            moved.append(f"{rel}/*")

    # re-initialise an empty brain so penrose stays immediately usable post-reset
    reinit = "skipped"
    reinit_ok = True
    # The native SQLite knowledge store needs no external init; it rebuilds lazily from the
    # flat-file records on first use, so a reset leaves it immediately usable.
    reinit_ok = True
    reinit = "native store (no external init needed)"

    # Transactional outcome: only discard the brain backup once re-init succeeded.
    # On failure, restore EVERYTHING from the snapshot so the box is fully
    # recoverable (otherwise the next --all re-processes every paper against the
    # restored brain -> duplicates), then exit 1.
    if not reinit_ok:
        # restore brain: remove any partial new brain, move the backup back
        if brain_backup is not None and brain_backup.exists():
            if brain.exists():
                shutil.rmtree(brain)
            shutil.move(str(brain_backup), str(brain))

        # restore FLAT files: move each archived file back to its original path
        for rel in FLAT:
            archived = dest / Path(rel).name
            if archived.exists():
                target = ROOT / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(archived), str(target))

        # restore DIR contents: move each archived item back into its dir
        for rel in DIRS:
            sub = dest / rel.replace("/", "__")
            if sub.exists():
                target_dir = ROOT / rel
                target_dir.mkdir(parents=True, exist_ok=True)
                for item in list(sub.iterdir()):
                    back = target_dir / item.name
                    if back.exists():
                        if back.is_dir():
                            shutil.rmtree(back)
                        else:
                            back.unlink()
                    shutil.move(str(item), str(back))
    elif brain_backup is not None and brain_backup.exists():
        # success: discard the now-redundant brain backup; keep the archive.
        shutil.rmtree(brain_backup)

    print(f"reset: archived to {dest.relative_to(ROOT)}")
    for m in (moved or ["(nothing to archive — already clean)"]):
        print(f"  archived {m}")
    print(f"empty brain re-init: {reinit}")
    if reinit_ok:
        print("penrose is now fresh. Roll back any piece from reset_archive/ if needed.")
    else:
        print("RESET FAILED: re-init did not succeed; prior brain restored. "
              "Archived copies remain in reset_archive/.")
        sys.exit(1)


if __name__ == "__main__":
    main()
