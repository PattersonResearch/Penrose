import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import migrate_holdout_state  # noqa: E402
from penrose import config  # noqa: E402
from penrose.pipeline import p7_backtest as p7  # noqa: E402


def _net():
    idx = pd.date_range("2024-01-01", periods=240, freq="D", tz="UTC")
    rng = np.random.default_rng(20260707)
    return pd.Series(0.004 + rng.normal(0.0, 0.01, len(idx)), index=idx)


def _isolate_holdout(tmp_path, monkeypatch):
    monkeypatch.delenv("PENROSE_HOLDOUT_LOCK", raising=False)
    monkeypatch.delenv("PENROSE_HOLDOUT_MODE", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")


def test_legacy_root_lock_is_honored_by_burn_check(tmp_path, monkeypatch):
    _isolate_holdout(tmp_path, monkeypatch)
    legacy = p7._legacy_claim_holdout_lock("legacy-claim")
    legacy.write_text("strategy=legacy-claim holdout_sharpe=2.1 burned_at=2026-01-01T00:00:00+00:00")

    result = p7.final_holdout_eval("legacy-claim", _net(), 252.0)

    assert result.get("refused") is True
    assert "already burned" in result["reason"]
    assert not p7._claim_holdout_lock("legacy-claim").exists()


def test_new_burn_writes_to_holdout_locks_not_repo_root(tmp_path, monkeypatch):
    _isolate_holdout(tmp_path, monkeypatch)

    result = p7.final_holdout_eval("new-claim", _net(), 252.0)

    assert result.get("refused") is not True
    assert p7._claim_holdout_lock("new-claim").exists()
    assert len(list((tmp_path / ".holdout" / "locks").glob("*.lock"))) == 1
    assert list(tmp_path.glob(".holdout_burned*")) == []


def test_migration_redacts_holdout_sharpe_and_is_idempotent(tmp_path, monkeypatch):
    _isolate_holdout(tmp_path, monkeypatch)
    legacy = p7._legacy_claim_holdout_lock("migrate-claim")
    legacy.write_text(
        "strategy=migrate-claim holdout_sharpe=1.234 "
        "digest=abcdef1234567890 burned_at=2026-01-01T00:00:00+00:00"
    )

    first = migrate_holdout_state.migrate()
    migrated = p7._claim_holdout_lock("migrate-claim")
    second = migrate_holdout_state.migrate()

    assert first["moved"] == 1
    assert second["moved"] == 0
    assert not legacy.exists()
    assert migrated.exists()
    assert "holdout_sharpe" not in migrated.read_text()
    assert p7._holdout_burned_lock("migrate-claim") == migrated


def test_migration_preserves_burn_for_whitespace_identity_hr2(tmp_path, monkeypatch):
    """HR-2: a legacy lock whose identity has surrounding whitespace must STAY burned across migration.
    The migration must preserve the legacy filename's slug.digest, not re-derive from the stripped
    strategy= field (which would produce a different digest and silently un-burn the identity)."""
    import importlib
    from penrose import config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    (tmp_path / ".holdout" / "locks").mkdir(parents=True, exist_ok=True)
    from penrose.pipeline import p7_backtest as p7
    import scripts.migrate_holdout_state as M
    name = "  spaced identity  "
    legacy = p7._legacy_claim_holdout_lock(name)
    legacy.write_text(f"strategy={name} digest=x burned_at=2026\n")
    assert p7._holdout_burned_lock(name) is not None      # burned before
    M.migrate()
    assert not legacy.exists()                            # relocated
    assert p7._holdout_burned_lock(name) is not None      # STILL burned — no un-burn
