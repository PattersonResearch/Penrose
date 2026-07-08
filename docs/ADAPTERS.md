# Data adapters

Penrose refers claims against real data, but it never reads a raw vendor format directly. A **data
adapter** is a small translator: it fetches data from some source and hands it back in one of Penrose's
standard **contracts**. Adapters are the seam where new data sources plug in — contributions welcome.

## The four contracts

Penrose reads exactly four shapes (`src/penrose/data/`):

| Contract | Shape | For |
|---|---|---|
| **`Series`** | a single daily time series (UTC dates → float) | time-series strategies (a rate, a spread, one asset's returns) |
| **`Panel`** | dates × entity columns → float | cross-sectional claims (rank many assets, form long/short factors) |
| **`EventCalendar`** | sorted, de-duplicated event timestamps | event-study claims (earnings, announcements, listings, index additions) |
| **`EventMarketPanel`** | per-event sets of strikes, each with a price and a binary outcome | prediction-market / bracket claims (Kalshi, Polymarket) |

Every contract carries `provenance` and is validated on construction. An adapter returns one of these, or
`Unavailable` when it honestly cannot (never a fabricated value).

For local pre-collected scalar series, the bring-your-own catalog seam is documented in
[DATA_CONTRACT.md](DATA_CONTRACT.md). The runnable reference is
[`examples/reference_loader/`](../examples/reference_loader/).

## Claim-type adapters

Some claim types reuse an existing data contract but need a deterministic front end before P7:

| Claim type | Contract | Executor | Notes |
|---|---|---|---|
| `predictive_regression` | two `Series` inputs: predictor, target | `src/penrose/pipeline/predictive_regression.py` | aligns `X_t` with `Y_t+h`, freezes sign and z-score moments on the in-sample prefix, emits the standard P7 net/positions/bars_per_year triple; no trading overlay |
| `factor_spanning` | `Series` inputs: candidate factor plus benchmark factors | `src/penrose/pipeline/factor_spanning.py` | fits candidate factor returns on declared benchmark factors using only the in-sample prefix, freezes betas, emits benchmark-hedged residual alpha to P7; no trading overlay |
| `cross_sectional_sort` | declared `Panel` inputs: returns plus characteristic | `src/penrose/pipeline/cross_sectional_sort.py` + `src/penrose/data/panel_load.py` | loads declared panel tables, requires survivorship-corrected returns, calls `data.xsection.form_factor`, synthesizes bucket-membership positions, and emits the P7 triple; no trading overlay |
| `event_study` | one return `Series` plus declared `EventCalendar` | `src/penrose/pipeline/event_study.py` + `src/penrose/data/event_calendar_load.py` | loads a declared event-date table, estimates the baseline strictly before each event, emits per-event CAR as the P7 net series, and annualizes by events/year; no trading overlay |
| `forecast_skill` | `Series` inputs: model forecast, realized target, optional benchmark forecast | `src/penrose/pipeline/forecast_skill.py` | emits the squared-loss differential `(B_t-Y_t)^2 - (F_t-Y_t)^2`; implied random-walk or historical-mean benchmarks are declared in the spec and constructed strictly from `Y` through `t-1`; no trading overlay |
| `event_market_strategy` | `EventMarketPanel` | `src/penrose/pipeline/event_market.py` + `event_market_backtest.py` | bracket-market pricing and entry rules emit the same P7 triple |

## Writing an adapter

Adapters live in `src/penrose/data/vendors/` (e.g. `stooq`, `polygon`, `tiingo`, `fred`) and
`src/penrose/data/` (e.g. `sec_edgar`). A new one must:

- **Satisfy a contract, don't invent a shape.** Return `Series` / `Panel` / `EventCalendar` /
  `EventMarketPanel`. Extend a contract only with a design note + tests.
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
- Declared panel-table loader for sort claims: `src/penrose/data/panel_load.py`.
- Declared event-calendar loader for event-study claims: `src/penrose/data/event_calendar_load.py`.
- `EventMarketPanel` (bracket markets): `src/penrose/data/event_market.py` + the bracket backtest in
  `src/penrose/pipeline/event_market_backtest.py`.
- Predictive-regression claim type over two `Series`: `src/penrose/pipeline/predictive_regression.py`.
- Factor-spanning claim type over `Series`: `src/penrose/pipeline/factor_spanning.py`.
- Cross-sectional-sort claim type over two declared `Panel` tables:
  `src/penrose/pipeline/cross_sectional_sort.py`.
- Event-study claim type over a return `Series` and declared event calendar:
  `src/penrose/pipeline/event_study.py`.
- Forecast-skill claim type over model-forecast and target `Series`:
  `src/penrose/pipeline/forecast_skill.py`.

Adapters are referee inputs for faithfully reconstructing what a claim describes — they are **not**
signal generators and assert no edge.
