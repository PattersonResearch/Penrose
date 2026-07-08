#!/usr/bin/env python
"""Move legacy holdout burn-state into .holdout/.

One-shot migration for the Phase 0 corpus re-score campaign. It intentionally
does not run as part of normal startup; a human reviews and runs it once.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose import config  # noqa: E402
from penrose.pipeline import p7_backtest as p7  # noqa: E402

_KNOWN_LOCK_KEYS = (
    "strategy",
    "evaluation",
    "bars",
    "digest",
    "burned_at",
    "holdout_sharpe",
    "holdout_psr",
    "nbars",
)


class MigrationError(RuntimeError):
    pass


def _legacy_files() -> tuple[Path | None, list[Path]]:
    root = Path(config.ROOT)
    base = root / ".holdout_burned"
    locks = sorted(root.glob(".holdout_burned.*.lock"))
    return (base if base.exists() else None), locks


def _field(text: str, key: str) -> str | None:
    match = re.search(rf"(?<!\S){re.escape(key)}=", text)
    if not match:
        return None
    start = match.end()
    next_key = re.search(
        rf"\s(?:{'|'.join(re.escape(k) for k in _KNOWN_LOCK_KEYS)})=",
        text[start:],
    )
    end = start + next_key.start() if next_key else len(text)
    value = text[start:end].strip()
    return value or None


def _redacted_lock_text(src: Path, *, require_strategy: bool) -> tuple[str | None, str]:
    original = src.read_text(errors="replace")
    strategy = _field(original, "strategy")
    if require_strategy and not strategy:
        raise MigrationError(f"cannot migrate {src.name}: missing strategy= identity")
    if not strategy:
        strategy = "legacy-global-holdout"
    burned_at = _field(original, "burned_at")
    if not burned_at:
        burned_at = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat()
    digest = _field(original, "digest")
    if not digest:
        digest = hashlib.sha256(f"{strategy}|{burned_at}|{src.name}".encode("utf-8")).hexdigest()[:16]
    text = f"strategy={strategy} digest={digest} burned_at={burned_at}\n"
    if "holdout_sharpe" in text:
        raise MigrationError(f"redaction failed for {src.name}: holdout_sharpe remains")
    return strategy, text


def _archive_legacy(files: list[Path]) -> Path:
    archive_dir = Path(config.ROOT) / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    archive = archive_dir / f"holdout_legacy_{stamp}.tar.gz"
    if archive.exists():
        raise MigrationError(f"archive already exists: {archive}")
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
    return archive


def _verify_migration(files_in: int, output_paths: list[Path], identities: list[str], had_base: bool) -> None:
    lock_outputs = [p for p in output_paths if p.parent == Path(config.HOLDOUT_DIR) / "locks"]
    base_outputs = [p for p in output_paths if p == Path(config.HOLDOUT_DIR) / "burned"]
    expected_out = len(lock_outputs) + (1 if had_base else 0)
    if files_in != expected_out:
        raise MigrationError(f"mismatch: files_in={files_in} files_out={expected_out}")
    if had_base and len(base_outputs) != 1:
        raise MigrationError("mismatch: legacy base log did not migrate to .holdout/burned")
    for path in output_paths:
        if not path.exists():
            raise MigrationError(f"mismatch: migrated output missing: {path}")
        if "holdout_sharpe" in path.read_text(errors="replace"):
            raise MigrationError(f"redaction failed: holdout_sharpe remains in {path}")
    # HR-2: the runtime burn check keys on the EXACT lock path `.holdout/locks/<slug>.<digest>.lock`, and
    # we preserved that filename verbatim from the legacy record — so a lock file existing at its migrated
    # path (verified above) IS the engine reporting it burned for the original raw identity. Re-checking via
    # a re-parsed (whitespace-stripped) identity would key on a DIFFERENT digest and falsely fail; the
    # authoritative check is that every legacy lock now exists at its identity-preserving destination.
    if had_base and p7._holdout_burned_lock("__legacy_global_probe__") is None:
        raise MigrationError("burn-check failed after migration for legacy base burn log")
    base, locks = _legacy_files()
    if base is not None or locks:
        remaining = [str(p) for p in ([base] if base else []) + locks]
        raise MigrationError(f"legacy files remain after migration: {remaining[:5]}")


def migrate() -> dict[str, object]:
    if os.environ.get("PENROSE_HOLDOUT_LOCK"):
        raise MigrationError("PENROSE_HOLDOUT_LOCK is set; refusing to migrate isolated test state")
    base, locks = _legacy_files()
    legacy = ([base] if base else []) + locks
    if not legacy:
        print("holdout migration: moved=0 archived=none verified=ok")
        return {"moved": 0, "archive": None, "verified": True}

    holdout_dir = Path(config.HOLDOUT_DIR)
    lock_dir = holdout_dir / "locks"
    planned: list[tuple[Path, Path, str | None, str]] = []
    identities: list[str] = []

    if base is not None:
        identity, text = _redacted_lock_text(base, require_strategy=False)
        planned.append((base, holdout_dir / "burned", identity, text))
    legacy_prefix = ".holdout_burned."
    for src in locks:
        identity, text = _redacted_lock_text(src, require_strategy=True)
        if identity is None:
            raise MigrationError(f"cannot migrate {src.name}: missing strategy= identity")
        # HR-2: PRESERVE the legacy filename's slug.digest EXACTLY. The burn record already encodes the
        # RAW identity's digest (`_claim_lock_filename` hashes the raw, unstripped name). Re-deriving the
        # destination from the parsed `strategy=` field — which `_field` whitespace-STRIPS — produces a
        # DIFFERENT digest for any identity with surrounding whitespace, and the runtime burn check (which
        # keys on the raw name) would then miss the migrated lock and silently UN-BURN it. Stripping only
        # the `.holdout_burned.` prefix yields precisely `_claim_holdout_lock(raw_name).name`.
        if not src.name.startswith(legacy_prefix):
            raise MigrationError(f"unexpected legacy lock name (no prefix): {src.name}")
        dst = lock_dir / src.name[len(legacy_prefix):]
        planned.append((src, dst, identity, text))
        identities.append(identity)

    destinations = [dst for _, dst, _, _ in planned]
    if len(destinations) != len(set(destinations)):
        raise MigrationError("mismatch: multiple legacy locks map to the same migrated path")
    conflicts = [dst for dst in destinations if dst.exists()]
    if conflicts:
        raise MigrationError(f"destination already exists; refusing to overwrite: {conflicts[:5]}")

    archive = _archive_legacy(legacy)
    lock_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for src, dst, _, text in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        src.unlink()
        output_paths.append(dst)

    _verify_migration(len(legacy), output_paths, identities, base is not None)
    print(f"holdout migration: moved={len(legacy)} archived={archive.relative_to(config.ROOT)} verified=ok")
    return {"moved": len(legacy), "archive": archive, "verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        migrate()
    except MigrationError as exc:
        print(f"holdout migration FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
