# Reference Loader

This directory is a minimal bring-your-own data catalog for Penrose:

```bash
PENROSE_DATA_DIR=examples/reference_loader penrose run --claims claims.json
```

`loader.py` reads `catalog.yaml` and local CSV files from `data/`. Each CSV has:

```csv
date,value
2024-01-01,470.25
```

The loader returns `(pandas.Series, provenance)`, where the series is a scalar daily UTC float series.
There is no OHLC/bar object in this contract; publish bars as four named scalar series such as
`asset_open`, `asset_high`, `asset_low`, and `asset_close`.

To add a series, add a CSV under `data/`, then add a `catalog.yaml` entry with `file`, `domain`,
`unit`, `provenance`, `agg`, and `description`.
