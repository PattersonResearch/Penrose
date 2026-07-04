from __future__ import annotations

import pandas as pd

from penrose.data import granularity
from penrose.data import vendors
from penrose.data.contract import DataBundle
from penrose.data.vendors import pysystemtrade


def _write_intraday_fixture(root, instrument="TEST", days=5, bars_per_day=8):
    price_dir = root / "data" / "futures" / "adjusted_prices_csv"
    price_dir.mkdir(parents=True)
    idx = []
    for day in pd.date_range("2024-01-01", periods=days, freq="D"):
        idx.extend(pd.date_range(day + pd.Timedelta(hours=9), periods=bars_per_day, freq="15min"))
    df = pd.DataFrame({"DATETIME": idx, "price": range(len(idx))})
    path = price_dir / f"{instrument}.csv"
    df.to_csv(path, index=False)
    return df


def test_pysystemtrade_available_only_with_configured_nonempty_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("PENROSE_FUTURES_DIR", raising=False)
    monkeypatch.delenv("PYSYS_DIR", raising=False)
    assert pysystemtrade.available() is False

    monkeypatch.setenv("PENROSE_FUTURES_DIR", str(tmp_path))
    assert pysystemtrade.available() is False

    _write_intraday_fixture(tmp_path)
    assert pysystemtrade.available() is True


def test_pysystemtrade_fetch_resamples_intraday_to_daily(tmp_path, monkeypatch):
    raw = _write_intraday_fixture(tmp_path, days=6, bars_per_day=16)
    monkeypatch.setenv("PENROSE_FUTURES_DIR", str(tmp_path))
    monkeypatch.delenv("PYSYS_DIR", raising=False)

    result = pysystemtrade.fetch({"instrument": "TEST"})

    assert result is not None
    series, provenance = result
    assert series.index.tz is not None
    assert str(series.index.tz) == "UTC"
    assert granularity.frequency_label(granularity.infer_bars_per_year(series)) == "daily"
    assert len(series) < len(raw) / 4
    assert "pysystemtrade" in provenance
    assert "intraday->daily" in provenance
    assert "back-adjusted continuous" in provenance


def test_pysystemtrade_fetch_missing_instrument_returns_none(tmp_path, monkeypatch):
    _write_intraday_fixture(tmp_path)
    monkeypatch.setenv("PENROSE_FUTURES_DIR", str(tmp_path))

    assert pysystemtrade.fetch({"instrument": "NOPE"}) is None


def test_pysystemtrade_fetch_rejects_path_traversal(tmp_path, monkeypatch):
    # A secret CSV with the right columns sits OUTSIDE the data dir; a traversal instrument
    # name must NOT read it (swarm audit found this leaked before the fix).
    _write_intraday_fixture(tmp_path)
    secret = tmp_path.parent / "secret.csv"
    pd.DataFrame({"DATETIME": pd.date_range("2024-01-01", periods=3, freq="D"),
                  "price": [999, 1000, 1001]}).to_csv(secret, index=False)
    monkeypatch.setenv("PENROSE_FUTURES_DIR", str(tmp_path))

    for bad in ("../../secret", "../../../secret", "SP500/../X", "/etc/passwd", "..\\..\\secret"):
        assert pysystemtrade.fetch({"instrument": bad}) is None


def test_pysystemtrade_defaults_fold_into_bundle_when_available(tmp_path, monkeypatch):
    _write_intraday_fixture(tmp_path, instrument="SP500")
    monkeypatch.setenv("PENROSE_FUTURES_DIR", str(tmp_path))
    for name in ("fred", "polygon", "tiingo", "alpaca", "alphavantage", "stooq", "kenfrench"):
        monkeypatch.setattr(vendors.ADAPTERS[name], "available", lambda: False)

    bundle = DataBundle(requested_window=("2024-01-01", "2024-01-10"))
    vendors.add_vendor_series(bundle)

    assert "futures_sp500" in bundle.series
    assert "futures_us10" not in bundle.series
    s = bundle.series["futures_sp500"]
    assert granularity.frequency_label(granularity.infer_bars_per_year(s.data)) == "daily"
    assert "pysystemtrade-adjusted" in s.provenance
    assert "as_displayed" in s.provenance
