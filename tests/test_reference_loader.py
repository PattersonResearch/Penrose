from __future__ import annotations

import sys

import pandas as pd

from penrose import config
from penrose.data.contract import DataBundle, Series, load_catalog_loader
from penrose.data.loader_protocol import CatalogLoaderProtocol


def _reference_dir():
    return config.ROOT / "examples" / "reference_loader"


def test_reference_loader_satisfies_protocol_and_shapes():
    loader = load_catalog_loader(_reference_dir())

    assert isinstance(loader, CatalogLoaderProtocol)
    assert loader.available() == ["btc_spot_close", "equity_spy_close"]
    assert loader.domains() == ["crypto", "equity"]
    assert loader.domain_of("equity_spy_close") == "equity"
    assert "equity_spy_close" in loader.describe_brief("equity_spy_close")
    assert loader.describe("equity_spy_close")["agg"] == "last"

    result = loader.load_series("equity_spy_close")
    assert result is not None
    series, provenance = result
    assert isinstance(series, pd.Series)
    assert provenance == "reference-csv:spy-close:v1"
    assert str(series.dtype) == "float64"
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.tz is not None
    assert list(series.index.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    ]


def test_client_catalog_path_loads_reference_series(monkeypatch):
    from penrose.data import client

    sys.modules.pop("loader", None)
    monkeypatch.setattr(config, "DATA_DIR", _reference_dir())
    bundle = DataBundle(requested_window=("2024-01-01", "2024-01-05"))

    client._add_catalog_series(bundle)

    loaded = bundle.series["equity_spy_close"]
    assert isinstance(loaded, Series)
    assert loaded.provenance == "reference-csv:spy-close:v1"
    assert loaded.note == "catalog:equity_spy_close"
    assert loaded.data.index.tz is not None
    assert list(loaded.data.astype(float)) == [470.25, 472.10, 471.35, 474.00, 473.50]
    sys.modules.pop("loader", None)


def test_resample_ohlc_collapses_intraday_to_daily_utc():
    from penrose.data.granularity import resample_ohlc

    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [12.0, 12.5, 13.0, 15.0],
            "low": [9.5, 10.5, 11.0, 12.5],
            "close": [11.5, 12.0, 12.5, 14.0],
        },
        index=pd.to_datetime([
            "2024-01-01 09:30",
            "2024-01-01 16:00",
            "2024-01-02 09:30",
            "2024-01-02 16:00",
        ]),
    )

    daily = resample_ohlc(frame)

    assert list(daily.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert daily.index.tz is not None
    assert daily.loc[pd.Timestamp("2024-01-01", tz="UTC")].to_dict() == {
        "open": 10.0,
        "high": 12.5,
        "low": 9.5,
        "close": 12.0,
    }
    assert daily.loc[pd.Timestamp("2024-01-02", tz="UTC")].to_dict() == {
        "open": 12.0,
        "high": 15.0,
        "low": 11.0,
        "close": 14.0,
    }
