"""Tests for the auto-source-and-archive data layer (src/penrose/data/auto_source.py).

Hermetic by default: the adapter is MOCKED, so these run WITHOUT network or API keys.
A single live FRED smoke test runs ONLY if FRED_API_KEY is set (otherwise it skips),
to prove the real end-to-end loop.

Run with:
  python -m pytest tests/test_auto_source.py -v
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from penrose import config
from penrose.data import auto_source, client
from penrose.data.contract import DataBundle, Series


def _fake_series(n: int = 60) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(range(1, n + 1), index=idx, dtype="float64", name="x")


@pytest.fixture
def enable_auto_source(monkeypatch):
    """Turn the opt-in gate ON for a test (default is OFF)."""
    monkeypatch.setattr(config, "AUTO_SOURCE",
                        {"enabled": True, "max_stale_days": 7,
                         "allow_sources": ["fred", "tiingo", "coingecko"]},
                        raising=False)


# --------------------------------------------------------------------------- #
# (a) Idempotency — a second resolve within max_stale_days does NOT re-fetch.
# --------------------------------------------------------------------------- #
def test_idempotent_within_stale_window(tmp_path, monkeypatch, enable_auto_source):
    calls = {"n": 0}

    def _fake_fetch(desc):
        calls["n"] += 1
        return _fake_series(), "fred-api:DGS10"

    monkeypatch.setattr(auto_source, "_fetch_from_source", _fake_fetch)

    spec = {"source": "fred", "id": "DGS10"}
    first = auto_source.resolve_and_archive("us_10y_treasury", spec=spec, data_dir=tmp_path)
    assert first is not None and first["cached"] is False
    assert calls["n"] == 1
    parquet = tmp_path / "vendor" / "us_10y_treasury.parquet"
    assert parquet.exists()

    # Second call within max_stale_days: served from the fresh archive, adapter NOT called.
    second = auto_source.resolve_and_archive("us_10y_treasury", spec=spec, data_dir=tmp_path)
    assert second is not None and second["cached"] is True
    assert calls["n"] == 1, "adapter must not be re-invoked for a fresh archive"
    assert len(second["series"]) == len(first["series"])


def test_stale_archive_is_refetched(tmp_path, monkeypatch):
    """An archive OLDER than max_stale_days is re-fetched (staleness bound honored)."""
    monkeypatch.setattr(config, "AUTO_SOURCE",
                        {"enabled": True, "max_stale_days": 0,
                         "allow_sources": ["fred"]}, raising=False)
    calls = {"n": 0}

    def _fake_fetch(desc):
        calls["n"] += 1
        return _fake_series(), "fred-api:DGS10"

    monkeypatch.setattr(auto_source, "_fetch_from_source", _fake_fetch)
    spec = {"source": "fred", "id": "DGS10"}
    auto_source.resolve_and_archive("us_10y_treasury", spec=spec, data_dir=tmp_path)
    # max_stale_days=0 -> the just-written archive is already "stale" -> refetch.
    auto_source.resolve_and_archive("us_10y_treasury", spec=spec, data_dir=tmp_path)
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# (b) Disabled by default — a miss stays needs_data (bundle unchanged).
# --------------------------------------------------------------------------- #
def test_disabled_by_default_is_no_op(tmp_path, monkeypatch):
    # Default config: AUTO_SOURCE.enabled is False.
    assert config.AUTO_SOURCE.get("enabled") is False
    # Even if the adapter WOULD succeed, a disabled gate must not source anything.
    monkeypatch.setattr(auto_source, "_fetch_from_source",
                        lambda desc: (_fake_series(), "fred-api:DGS10"))
    bundle = DataBundle()
    out = client.resolve_missing_series(bundle, "DGS10")
    assert out is None
    assert "DGS10" not in bundle.series
    assert bundle.get("DGS10") is None            # honest needs_data preserved


# --------------------------------------------------------------------------- #
# (c) Graceful — an unknown/unsourceable name returns None, never raises.
# --------------------------------------------------------------------------- #
def test_unsourceable_name_returns_none(tmp_path, enable_auto_source):
    # No explicit hint + no heuristic match -> None, no exception.
    out = auto_source.resolve_and_archive("totally_unknown_thing_xyz", data_dir=tmp_path)
    assert out is None
    assert not (tmp_path / "vendor" / "totally_unknown_thing_xyz.parquet").exists()


def test_detect_source_heuristics_and_hints():
    allow = ["fred", "tiingo", "coingecko"]
    # Explicit hint (preferred).
    d = auto_source.detect_source("anything", spec={"source": "fred", "id": "DGS10"}, allow_sources=allow)
    assert d == {"source": "fred", "id": "DGS10"}
    # Nested hint under a "source" key + extra field.
    d = auto_source.detect_source(
        "x", spec={"source": {"source": "tiingo", "id": "AAPL", "field": "adjClose"}}, allow_sources=allow)
    assert d["source"] == "tiingo" and d["id"] == "AAPL" and d["field"] == "adjClose"
    # Heuristic: FRED-style all-caps id.
    assert auto_source.detect_source("CPIAUCSL", allow_sources=allow) == {"source": "fred", "id": "CPIAUCSL"}
    # Heuristic: known equity ticker -> tiingo (NOT fred, even though all-caps).
    assert auto_source.detect_source("AAPL", allow_sources=allow) == {"source": "tiingo", "id": "AAPL"}
    # Unknown lowercase -> None.
    assert auto_source.detect_source("weird_local_thing", allow_sources=allow) is None
    # A disallowed source in an explicit hint is NOT silently proxied to a heuristic.
    assert auto_source.detect_source("DGS10", spec={"source": "bloomberg", "id": "DGS10"},
                                     allow_sources=allow) is None


def test_graceful_when_adapter_returns_none(tmp_path, monkeypatch, enable_auto_source):
    monkeypatch.setattr(auto_source, "_fetch_from_source", lambda desc: None)
    out = auto_source.resolve_and_archive("us_10y_treasury",
                                          spec={"source": "fred", "id": "DGS10"}, data_dir=tmp_path)
    assert out is None


# --------------------------------------------------------------------------- #
# (d) A catalog entry is appended with provenance + sourced_at + status vendor.
# --------------------------------------------------------------------------- #
def test_catalog_entry_registered(tmp_path, monkeypatch, enable_auto_source):
    import yaml

    monkeypatch.setattr(auto_source, "_fetch_from_source",
                        lambda desc: (_fake_series(), "fred-api:DGS10"))
    auto_source.resolve_and_archive("us_10y_treasury",
                                    spec={"source": "fred", "id": "DGS10"}, data_dir=tmp_path)
    cat = yaml.safe_load((tmp_path / "catalog.yaml").read_text())
    entry = cat["series"]["us_10y_treasury"]
    assert entry["provenance"] == "fred-api:DGS10"
    assert entry["status"] == "vendor"
    assert entry["adapter"] == "col"
    assert entry["date_col"] == "date" and entry["value_col"] == "value"
    assert entry["pit"] is False              # as-collected, not point-in-time
    assert entry["path"] == "vendor/us_10y_treasury.parquet"
    assert isinstance(entry["sourced_at"], str) and entry["sourced_at"].endswith("Z")


def test_registration_does_not_clobber_existing(tmp_path, monkeypatch, enable_auto_source):
    """A second registration must NOT overwrite a hand-authored catalog entry, and must
    preserve unrelated content/comments in the file."""
    cat = tmp_path / "catalog.yaml"
    cat.write_text(
        "# my hand-authored catalog\n"
        "series:\n"
        "  us_10y_treasury: {domain: rates, path: custom.parquet, provenance: MINE, status: vendor}\n",
        encoding="utf-8")
    monkeypatch.setattr(auto_source, "_fetch_from_source",
                        lambda desc: (_fake_series(), "fred-api:DGS10"))
    auto_source.resolve_and_archive("us_10y_treasury",
                                    spec={"source": "fred", "id": "DGS10"}, data_dir=tmp_path)
    import yaml
    loaded = yaml.safe_load(cat.read_text())
    assert loaded["series"]["us_10y_treasury"]["provenance"] == "MINE"   # untouched
    assert "# my hand-authored catalog" in cat.read_text()               # comment preserved


# --------------------------------------------------------------------------- #
# Client wiring: enabled -> a miss is sourced + folded into the bundle.
# --------------------------------------------------------------------------- #
def test_client_wiring_folds_series_into_bundle(tmp_path, monkeypatch, enable_auto_source):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(auto_source, "_fetch_from_source",
                        lambda desc: (_fake_series(), "fred-api:DGS10"))
    bundle = DataBundle()
    out = client.resolve_missing_series(bundle, "CPIAUCSL")   # heuristic -> fred
    assert isinstance(out, Series)
    assert isinstance(bundle.series.get("CPIAUCSL"), Series)
    assert bundle.series["CPIAUCSL"].provenance == "fred-api:DGS10"


# --------------------------------------------------------------------------- #
# Live smoke (only with a real key) — proves the true end-to-end loop.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"),
                    reason="no FRED_API_KEY; live smoke skipped (unit tests cover the loop)")
def test_live_fred_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_SOURCE",
                        {"enabled": True, "max_stale_days": 7, "allow_sources": ["fred"]},
                        raising=False)
    res = auto_source.resolve_and_archive(
        "us_10y_treasury", spec={"source": "fred", "id": "DGS10", "start": "2024-01-01"},
        data_dir=tmp_path)
    assert res is not None and res["cached"] is False
    assert len(res["series"]) > 50
    assert (tmp_path / "vendor" / "us_10y_treasury.parquet").exists()
    # Idempotent second call: cached, no re-fetch.
    res2 = auto_source.resolve_and_archive(
        "us_10y_treasury", spec={"source": "fred", "id": "DGS10"}, data_dir=tmp_path)
    assert res2 is not None and res2["cached"] is True
