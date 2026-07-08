# BYO Data Contract

Penrose can read local, pre-collected data through `PENROSE_DATA_DIR`. Point the variable at a
directory containing `loader.py`; Penrose imports that module and calls a small catalog surface:

```python
load_series(name: str) -> tuple[pandas.Series, str] | None
available() -> list[str]
domain_of(name: str) -> str | None
domains() -> list[str]
describe(name: str) -> dict
describe_brief(name: str) -> str
```

The typed reference is `penrose.data.loader_protocol.CatalogLoaderProtocol`.

## Series Shape

`load_series()` returns `(series, provenance)` or `None` if the series cannot be delivered. The series
must be a scalar daily time series:

- `pandas.Series`
- `DatetimeIndex`, UTC daily timestamps
- float values
- sorted, deterministic, point-in-time

If a returned index is timezone-naive, Penrose localizes it to UTC before placing it in a
`penrose.data.contract.Series`. Return `None` for missing files, missing columns, empty data, or
unsupported names. Do not fabricate fallback values in a loader.

There is no OHLC or bar primitive in this catalog seam. OHLC is four scalar daily series by convention:
`asset_open`, `asset_high`, `asset_low`, and `asset_close`.

## Intraday Data

Collapse intraday data to daily before returning it from `load_series()`, and declare the aggregation in
catalog metadata. Supported aggregation names for scalar series should be explicit:

- `last`
- `sum`
- `mean`
- `count`
- `first`
- `max`
- `min`

For OHLC input, use the usual daily collapse: open = first, high = max, low = min, close = last. Penrose
also exposes `penrose.data.granularity.resample_ohlc(frame)` for that generic conversion.

## Minimal Example

The reference implementation is in `examples/reference_loader/`:

```bash
PENROSE_DATA_DIR=examples/reference_loader penrose run --claims claims.json
```

It contains:

- `loader.py` implementing `CatalogLoaderProtocol`
- `catalog.yaml` with names, domains, units, provenance, aggregation, and descriptions
- tiny CSV files under `data/`

The sample catalog publishes `equity_spy_close` and `btc_spot_close` as daily scalar close series.
Use it as a template for private catalogs; keep private data outside the repository and point
`PENROSE_DATA_DIR` at that directory.
