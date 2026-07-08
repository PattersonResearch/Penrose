# Data Acquisition Standard

**A verdict is only as trustworthy as its weakest input series.** This standard sets the bar for what data
is acceptable to feed Penrose. It sits *above* the wire contracts: [DATA_CONTRACT.md](DATA_CONTRACT.md) and
[ADAPTERS.md](ADAPTERS.md) define *how Penrose reads data*; this defines *what data is trustworthy enough to
referee on*.

It exists because of a concrete failure: in an early corpus re-score, kills had been adjudicated on
synthetic/proxy series (a `synthetic` provenance, or cross-venue-contaminated legs — perp from one venue,
funding from another, spot from a third). Those were not trustworthy verdicts. Every field and gate below
exists to make the weakest link *visible*, never to hide it.

**Scope.** Applies to (a) every series in a Penrose data catalog, (b) every vendor adapter, and (c) every
bring-your-own catalog a third party points `PENROSE_DATA_DIR` at. Conformance to the `CatalogLoaderProtocol`
is *necessary but not sufficient* for production-tier verdicts — a BYO catalog must also clear this bar.

---

## 1. Required-fields contract

Every registered series declares these. A series missing a required field, or whose declared value does not
match what the loader actually measures (coverage, granularity), **fails catalog validation** — it is not
registered, and any claim needing it routes to `needs_data`. This is a lint check CI runs against the
catalog, kept green like the eval suite.

| Field | Meaning | Req | Validation |
|---|---|---|---|
| `domain` | asset/data class (`crypto-spot`, `equity`, `macro`, `prediction-market-macro`, …) | ✔ | must be in the controlled vocabulary (extended by PR, not free text) |
| `provenance` | vendor/method tag (`okx-swap`, `fred`, `kalshi-live`, `synthetic`) | ✔ | non-empty; `synthetic`/`derived`/`proxy` MUST be flagged (§2.1) |
| `coverage` | `[start, end]` observed range | ✔ | must equal the loaded index min/max, checked at load — not merely declared |
| `granularity` | native cadence (tick/1m/1h/1d) + the `agg` used to collapse to daily | ✔ | verified against actual index spacing at load, not trusted from YAML |
| `day_basis` | `utc` or `local` (which calendar day a date-key means) | ✔ | loader refuses silent cross-basis joins |
| `status` | `static` / `vendor` / `frozen_snapshot` / `alias` / `derived` | ✔ | `derived`/`alias` must name their source series |
| `adapter` | loader shape (`wide_col`, `long_filter`, `col`, or a vendor/ BYO adapter) | ✔ | must resolve to a registered adapter |
| `max_stale_days` | staleness bound | ✔ for non-`static` | absent → treated as always-stale (fail **closed**) |
| `unit` | `usd` / `pct` / `degF` / … | ✔ | enforced by the `Series`/`Panel` dataclasses |
| `pit` | point-in-time correct? (`true` = as-known-at-`t`; `false` = as-collected) | ✔ | see §3.4; drives verdict disclosure |
| `survivorship` | for panels: `corrected` / `uncorrected` / `unknown` | ✔ for `Panel` | see §3.3 |
| `redistribution` | licensing class (§5) | ✔ for anything shipped publicly | one of: public / attribution / derived-ok / none |
| `fingerprint` | content hash at fetch/snapshot time | ✔ for receipts | computed by the loader, never hand-entered |
| `note` | methodology caveats (e.g. an OHLC-from-hourly-close proxy) | recommended when `derived`/`proxy` | — |

> **Contract gap (implementation follow-up):** today's `Series`/`Panel`/`CatalogLoaderProtocol` have no
> `fingerprint`, `pit`, or `survivorship` fields. Making §3.3–3.5 machine-enforced needs three minimal,
> backward-compatible dataclass/protocol extensions — tracked as tasks, not part of this doc.

---

## 2. Invariants (testable, not aspirational)

### 2.1 No-silent-proxy
A missing or insufficient series **never** triggers synthesis, cross-domain interpolation, or substitution
with a look-alike. It returns `Unavailable` and the claim routes to `needs_data`. Deriving `high`/`low` from
same-series hourly closes with a declared `okx-swap-derived` provenance + a caveat note is *allowed and
labeled*; standing in a *different* asset/venue/domain for a missing one is *not* — that is the exact
contamination this standard retires. **Test:** every adapter asserts that an out-of-coverage or nonexistent
series returns `Unavailable`, never a fabricated `Series`.

### 2.2 Provenance-required
No `Series`/`Panel` may be constructed without a non-empty, catalog-registered provenance. Anonymous inline
arrays are not a valid data source in the referee path. Provenance must resolve to a documented source, not
an arbitrary string.

### 2.3 No-look-ahead
Only data knowable at/before decision time `t` may enter row `t`. Three failure modes, each with its own
test:
1. **Restatement/vintage leakage** — macro series (GDP, payrolls, CPI) get revised; using the *final*
   revised value at `t` when only the *first print* was knowable is look-ahead. PIT-sensitive claims must
   use vintaged sources (§3.4). *This is the subtlest and least-tested case.*
2. **Settlement/outcome leakage** — an event market's outcome/settlement must not enter pre-resolution
   features (see ADAPTERS.md).
3. **Frequency masquerading** — intraday data silently treated as daily (checked via the granularity gate).

---

## 3. Quality gates

Each: what it checks · why · how to satisfy · where enforced.

1. **Granularity declared + verified** — native cadence matches the declared `granularity`/`agg`; a silently
   wrong frequency corrupts every downstream statistic. Run the granularity check at *catalog-load* time,
   fail loudly on declared ≠ measured. *Enforced by catalog lint.*
2. **Staleness bound** — for any non-`static` series, `now − last_obs ≤ max_stale_days`; a "live" feed that
   silently went stale is indistinguishable from a healthy one unless bounded. A periodic audit asserts
   freshness. (`frozen_snapshot` exists precisely to keep a *known*-frozen series from being misread as a
   failed refresh.) *Enforced by a catalog audit job.*
3. **Survivorship (panels)** — any `Panel` used for cross-sectional claims must retain delisted/dead entities
   for the window they were alive, using *point-in-time universe membership*, not today's membership applied
   backward. Survivorship bias is the single most common way a backtest overstates an edge, and it is
   invisible unless audited. The panel declares `survivorship: corrected | uncorrected | unknown`. *Enforced
   by adapter review + the `survivorship` field.*
4. **Point-in-time** — distinguish "as known today" (`pit: false`, a static snapshot) from "as known on date
   `t`" (`pit: true`, vintaged). Most static catalogs are `pit: false` — a real limitation that must be
   *surfaced, not buried*. PIT-sensitive claim types (restated macro, index membership, fundamentals) demand
   `pit: true` sources or the verdict carries a mandatory caveat. *Enforced by verdict-level disclosure
   (§6) + catalog lint.*
5. **Fingerprint** — a content hash (e.g. sha256 of serialized index+values) computed at fetch/snapshot
   time. "The same series name" can silently drift between runs (revendored, refetched, backfilled); a claim
   re-adjudicated months later must prove it saw the same data — or explicitly show it didn't. *Carried on
   the `Series`/`Panel`; surfaced into the receipt (§6).*

---

## 4. Meeting the standard by domain — capabilities, with examples

**Normative text binds to *capabilities*, never to brands.** The tables below state the capability a source
must have per domain, then list *examples known to meet it* — free and paid alike — **non-exhaustive,
unranked, alphabetized**, each tied to the criterion it satisfies.

> **No affiliation.** Penrose has no commercial relationship with any source named here. Inclusion means a
> source is known to satisfy the stated criterion, not that alternatives don't and not a recommendation to
> buy.

**Equities**
- *Daily EOD prices, keyless:* Ken French library (factors, 1926+, minimally revised); Stooq (EOD, current-
  list bias if a universe is built from today's tickers).
- *Survivorship-corrected panel + PIT fundamentals* (§3.3–3.4): CRSP-class datasets (delisting returns
  included by construction); Compustat-Snapshot-class point-in-time fundamentals. Institutional-licensed.

**Crypto — spot & derivatives**
- *Keyless spot/derivatives history:* public exchange REST (Binance/OKX/Bybit — note some funding-history
  endpoints cap server-side, so deep funding needs a deeper source). Exchange APIs rarely track delisted
  pairs → build a dead-token registry for cross-sectional crypto panels.
- *Survivorship-aware universe / cross-venue aggregation* (reduces the single-venue contamination this
  standard exists to prevent): CoinGecko-class (tracks dead/delisted tokens); CryptoCompare/CCData-class.
- *Audit-grade tick + deep derivatives history:* Amberdata-class; Kaiko-class. Licensed.

**On-chain**
- *Keyless chain metrics:* blockchain.com (BTC); Dune / The Graph (protocol-dependent; community-query
  correctness varies — audit before trusting); Etherscan (keyed free tier).
- *Deep aggregated on-chain metrics:* Glassnode-class. Licensed for redistribution.

**Macro**
- *Vintaged / point-in-time* (§3.4 — **required for any "trader reacts to the print" thesis**): ALFRED
  (vintage-stamped, public domain). **FRED serves the latest revision by default — using it for a PIT claim
  silently injects look-ahead.** This is the single easiest look-ahead bug to introduce, because FRED is the
  reflex.
- *First-release calendar:* BLS (pair with a vintaged CPI/payrolls source rather than trusting the API's
  latest value as PIT).

**Prediction markets**
- *Full settled-market history, keyless/keyed:* Kalshi API; Polymarket Gamma / subgraph; Dune dashboards.
  Settled outcomes are genuinely realized (not revised) — but must not leak settlement into pre-resolution
  features.

**Factors**
- *Long, minimally-restated factor returns:* AQR data library (house methodology — do not silently mix with
  Fama-French construction); Ken French library.

---

## 5. Licensing / redistribution (read before committing data)

- **Public domain (safe to redistribute; prefer as OSS default):** US government data — FRED/ALFRED, BLS.
- **Exchange APIs:** fetching for research is generally fine; bulk raw redistribution usually isn't —
  publish *derived* daily series (Penrose's convention), not raw tick dumps.
- **Prediction markets:** check current ToS before bulk redistribution; safe to ship *derived signals*
  (e.g. a volume-weighted |Δprob|), not raw books.
- **Freemium/paid vendors** (CoinGecko/Tiingo/Glassnode/Kaiko/Amberdata-class): free tiers often bar
  redistribution even of derived metrics. **Default posture: never commit paid-vendor raw or lightly-derived
  data into the public repo** — keep it behind the BYO seam in a private catalog.
- **Institutional (CRSP/Compustat-class):** strict no-redistribution, licensed only. Reference by adapter for
  licensed users; never check into a public catalog.

---

## 6. Tiers, disclosure, and reproducibility receipts

**Two tiers, and a verdict declares which it ran under — computed automatically from the *worst* tier of any
series it consumed, never asserted by hand.**

- **Demo / keyless (out-of-the-box):** Penrose runs a *real* referee with no signup — Ken French, Stooq,
  FRED (**explicitly not PIT**), public exchange REST, Kalshi/Polymarket, keyless on-chain. Demo-tier
  verdicts that touch a non-PIT or non-survivorship-corrected source carry a visible banner: *"as-collected /
  non-PIT data — not production-grade."*
- **Production:** every consumed series satisfies all quality gates (§3) at `pit: true` / `survivorship:
  corrected` where the claim type demands it. This is the enforcement hook for capital-consequential
  verdicts, not a suggestion.

**Reproducibility receipt.** For every series a verdict's `DataBundle` actually accessed, the receipt records
`provenance`, `coverage`, `fingerprint`, `pit`, tier contribution, and whether it was ever `Unavailable`
(routed to `needs_data`) rather than silently skipped. The fingerprint is computed once at the loader
boundary and carried unchanged into the receipt — recomputing at receipt time would defeat the "prove you
saw the same bytes" purpose. A verdict whose receipt cannot show a fingerprint for every series it touched is
flagged, not certified.

---

*This standard is normative for production-tier verdicts. Where it names a source, that is evidence a
capability can be met — not an endorsement. Where reality falls short of the bar (most static catalogs are
`pit: false`), the rule is to **disclose it in the verdict**, never to paper over it.*
