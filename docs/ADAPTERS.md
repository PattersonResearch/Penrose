# Data adapters

Penrose refers claims against real data, but it never reads a raw vendor format directly. A **data
adapter** is a small translator: it fetches data from some source and hands it back in one of Penrose's
standard **contracts**. Adapters are the seam where new data sources plug in — contributions welcome.

## The three contracts

Penrose reads exactly three shapes (`src/penrose/data/`):

| Contract | Shape | For |
|---|---|---|
| **`Series`** | a single daily time series (UTC dates → float) | time-series strategies (a rate, a spread, one asset's returns) |
| **`Panel`** | dates × entity columns → float | cross-sectional claims (rank many assets, form long/short factors) |
| **`EventMarketPanel`** | per-event sets of strikes, each with a price and a binary outcome | prediction-market / bracket claims (Kalshi, Polymarket) |

Every contract carries `provenance` and is validated on construction. An adapter returns one of these, or
`Unavailable` when it honestly cannot (never a fabricated value).

For local pre-collected scalar series, the bring-your-own catalog seam is documented in
[DATA_CONTRACT.md](DATA_CONTRACT.md). The runnable reference is
[`examples/reference_loader/`](../examples/reference_loader/).

## Writing an adapter

Adapters live in `src/penrose/data/vendors/` (e.g. `stooq`, `polygon`, `tiingo`, `fred`) and
`src/penrose/data/` (e.g. `sec_edgar`). A new one must:

- **Satisfy a contract, don't invent a shape.** Return `Series` / `Panel` / `EventMarketPanel`. Extend a
  contract only with a design note + tests.
- **No look-ahead, checked at the boundary.** Only data known at/before the decision time may enter a
  row; frequency is checked so intraday data can't be silently treated as daily, and a settlement/outcome
  field must not leak information available only after the market closes.
- **Deterministic.** No wall-clock dependence, no unseeded randomness, stable ordering.
- **Fail gracefully.** Missing keys, empty pulls, and degenerate inputs return a clear message or
  `Unavailable` — never a raw traceback.
- **Keyless where possible; document any key.** Several adapters (Stooq, SEC EDGAR, Ken French) need no
  key; those that do fail with a clear message when it's unset.
- **Test it.** Ship deterministic tests, including a look-ahead-rejection test, and keep
  `python scripts/eval_suite.py` green.

## Examples to copy

- `Series` from a keyless vendor: `src/penrose/data/vendors/stooq.py`.
- `Panel` (cross-sectional) source: `src/penrose/data/sec_edgar.py` (point-in-time fundamentals).
- `EventMarketPanel` (bracket markets): `src/penrose/data/event_market.py` + the bracket backtest in
  `src/penrose/pipeline/event_market_backtest.py`.

Adapters are referee inputs for faithfully reconstructing what a claim describes — they are **not**
signal generators and assert no edge.
