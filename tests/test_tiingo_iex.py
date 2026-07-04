from __future__ import annotations

import os
import urllib.parse

import pandas as pd
import pytest

from penrose.data import granularity
from penrose.data import vendors
from penrose.data.vendors import tiingo_iex


def _sample_rows():
    return [
        {
            "date": "2024-01-02T14:30:00.000Z",
            "open": 470.00,
            "high": 470.25,
            "low": 469.90,
            "close": 470.10,
            "volume": 1200,
        },
        {
            "date": "2024-01-02T14:35:00.000Z",
            "open": 470.10,
            "high": 470.40,
            "low": 470.05,
            "close": 470.35,
            "volume": 1400,
        },
        {
            "date": "2024-01-02T14:35:00.000Z",
            "open": 470.11,
            "high": 470.41,
            "low": 470.06,
            "close": 470.36,
            "volume": 1450,
        },
        {
            "date": "2024-01-02T14:40:00.000Z",
            "open": None,
            "high": 470.50,
            "low": 470.20,
            "close": 470.30,
            "volume": 1300,
        },
        {
            "date": "2024-01-02T14:40:00.000Z",
            "open": 470.36,
            "high": 470.55,
            "low": 470.20,
            "close": 470.45,
            "volume": 1500,
        },
    ]


def _query(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


def test_available_reflects_key_never_raises_and_grade(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert tiingo_iex.available() is False

    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    assert tiingo_iex.available() is True
    assert tiingo_iex.PROVENANCE_GRADE == "as_displayed"
    assert tiingo_iex.NAME == "tiingo_iex"
    assert tiingo_iex._VENUE == "single_venue"

    class BadEnv:
        def get(self, _key):
            raise RuntimeError("broken env")

    class BadOS:
        environ = BadEnv()

    monkeypatch.setattr(tiingo_iex, "os", BadOS)
    assert tiingo_iex.available() is False


def test_registered_but_not_default_series():
    assert vendors.ADAPTERS[tiingo_iex.NAME] is tiingo_iex
    assert all(spec.get("vendor") != tiingo_iex.NAME for spec in vendors.DEFAULT_SERIES.values())


def test_default_fetch_is_price_only_and_url_omits_volume(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    seen = {}

    def fake_download(url):
        seen["url"] = url
        return _sample_rows()

    monkeypatch.setattr(tiingo_iex, "_download_json", fake_download)
    out = tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03")
    assert out is not None
    df, provenance = out

    assert list(df.columns) == ["open", "high", "low", "close"]
    assert "volume" not in df.columns
    assert provenance == "tiingo-iex:SPY:5min:single_venue"
    assert df.attrs["venue"] == "single_venue"

    params = _query(seen["url"])
    assert params["columns"] == ["date,open,high,low,close"]
    assert "volume" not in params["columns"][0]
    assert params["resampleFreq"] == ["5min"]
    assert params["token"] == ["dummy"]


def test_include_volume_includes_and_marks_single_venue(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    seen = {}

    def fake_download(url):
        seen["url"] = url
        return _sample_rows()

    monkeypatch.setattr(tiingo_iex, "_download_json", fake_download)
    out = tiingo_iex.fetch_intraday(
        "SPY", "2024-01-02", "2024-01-03", freq="5min", include_volume=True
    )
    assert out is not None
    df, provenance = out

    # volume column is self-documenting -> misuse of df["volume"] is a KeyError, not a 5%-sample
    assert list(df.columns) == ["open", "high", "low", "close", "volume_single_venue"]
    assert "volume" not in df.columns
    assert df.attrs["volume_venue"] == "single_venue"
    assert "single_venue" in provenance
    assert _query(seen["url"])["columns"] == ["date,open,high,low,close,volume"]


def test_price_only_drops_unasked_volume_and_stub_and_freq_guard(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    # defense-in-depth (V4): a chatty server returns volume unasked -> price-only frame drops it
    monkeypatch.setattr(tiingo_iex, "_download_json", lambda _url: _sample_rows())
    out = tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03", freq="5min")
    assert out is not None
    df, _ = out
    assert "volume" not in df.columns and "volume_single_venue" not in df.columns
    # V2: a daily+ freq on the intraday-only endpoint is refused (no network touch)
    assert tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03", freq="1day") is None
    # V3: the daily-protocol stub is present and fail-open, so a uniform registry iteration is safe
    assert tiingo_iex.fetch({}) is None


def test_tz_aware_utc_intraday_spacing_and_dup_na_drop(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    monkeypatch.setattr(tiingo_iex, "_download_json", lambda _url: _sample_rows())

    out = tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03")
    assert out is not None
    df, _ = out

    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert list(df.index.strftime("%H:%M:%S")) == ["14:30:00", "14:35:00", "14:40:00"]
    assert df.loc[pd.Timestamp("2024-01-02T14:35:00Z"), "close"] == pytest.approx(470.36)
    assert df.index.to_series().diff().dropna().dt.total_seconds().tolist() == [300.0, 300.0]
    assert granularity.frequency_label(granularity.infer_bars_per_year(df)) == "intraday"


def test_empty_returns_none(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    monkeypatch.setattr(tiingo_iex, "_download_json", lambda _url: [])
    assert tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03") is None


def test_malformed_returns_none_fail_open(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "dummy")
    for malformed in ({"bad": "shape"}, "not-json-list", [{"date": "bad", "open": 1.0}]):
        monkeypatch.setattr(tiingo_iex, "_download_json", lambda _url, rows=malformed: rows)
        assert tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03") is None

    def boom(_url):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(tiingo_iex, "_download_json", boom)
    assert tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03") is None


@pytest.mark.skipif(
    not os.environ.get("TIINGO_API_KEY") or not os.environ.get("PENROSE_IEX_LIVE"),
    reason="TIINGO_API_KEY and PENROSE_IEX_LIVE required for live Tiingo IEX test",
)
def test_tiingo_iex_live_fetch_spy_5min():
    out = tiingo_iex.fetch_intraday("SPY", "2024-01-02", "2024-01-03", freq="5min")
    assert out is not None
    df, provenance = out
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert df.index.tz is not None
    assert "tiingo-iex:SPY:5min:single_venue" == provenance
