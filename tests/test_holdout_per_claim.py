import json

import numpy as np
import pandas as pd


def _net():
    idx = pd.date_range("2024-01-01", periods=240, freq="D", tz="UTC")
    rng = np.random.default_rng(20240624)
    return pd.Series(0.004 + rng.normal(0.0, 0.01, len(idx)), index=idx)


def test_holdout_lock_is_per_distinct_claim(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import p7_backtest as p7

    monkeypatch.delenv("PENROSE_HOLDOUT_LOCK", raising=False)
    monkeypatch.delenv("PENROSE_HOLDOUT_MODE", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    net = _net()

    first = p7.final_holdout_eval("claim-a", net, 252.0)
    second = p7.final_holdout_eval("claim-b", net, 252.0)
    third = p7.final_holdout_eval("claim-c", net, 252.0)
    repeat = p7.final_holdout_eval("claim-a", net, 252.0)

    assert first.get("refused") is not True
    assert second.get("refused") is not True
    assert third.get("refused") is not True
    assert repeat.get("refused") is True
    assert "claim-a" in repeat["reason"]
    assert len(list((tmp_path / ".holdout" / "locks").glob("*.lock"))) == 3
    assert list(tmp_path.glob(".holdout_burned.*.lock")) == []


def test_holdout_burn_refusal_does_not_echo_stats(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import p7_backtest as p7

    monkeypatch.delenv("PENROSE_HOLDOUT_LOCK", raising=False)
    monkeypatch.delenv("PENROSE_HOLDOUT_MODE", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "HOLDOUT_DIR", tmp_path / ".holdout")
    net = _net()

    first = p7.final_holdout_eval("claim-leak-check", net, 252.0)
    second = p7.final_holdout_eval("claim-leak-check", net, 252.0)
    lock_text = next((tmp_path / ".holdout" / "locks").glob("*.lock")).read_text()

    assert "holdout_sharpe" in first
    assert second.get("refused") is True
    # PEN-14: burned locks and refusal payloads must never echo holdout statistics.
    assert "holdout_sharpe" not in lock_text
    assert "holdout_sharpe" not in json.dumps(second)


def test_holdout_readonly_blocks_even_force(tmp_path, monkeypatch):
    from penrose.pipeline import p7_backtest as p7

    lock = tmp_path / "isolated.lock"
    monkeypatch.setenv("PENROSE_HOLDOUT_LOCK", str(lock))
    monkeypatch.setenv("PENROSE_HOLDOUT_MODE", "readonly")

    result = p7.final_holdout_eval("claim-readonly", _net(), 252.0, force=True)

    assert result.get("refused") is True
    assert "read-only" in result["reason"]
    assert not lock.exists()
