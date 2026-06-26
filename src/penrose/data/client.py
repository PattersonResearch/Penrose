"""Data client (v1, no separate service yet).

Prefers live venues; falls back to a clearly-tagged synthetic generator when
live history is too shallow or unreachable. Every series carries provenance so a
verdict built on synthetic data is never mistaken for one built on real fills.

Live sources attempted:
  * BTC daily close  -> Binance public klines (no key, reliable)
  * Kalshi macro      -> elections.kalshi.com candlesticks (best-effort; shallow)
  * BTC implied vol   -> Deribit DVOL (best-effort)

The synthetic generator embeds a MODEST true relationship (macro signal weakly
forecasts next-5d realized vol) plus a volatility risk premium, so the pipeline's
verdict is earned by DSR/holdout/fees rather than rigged either way.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .contract import Series, Unavailable, DataBundle

WINDOW = ("2023-01-01", "2026-03-31")          # the paper's sample
_SEED = 20260618


def _http_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------------------- #
# Live: BTC daily close from Binance public klines
# --------------------------------------------------------------------------- #
def _coinbase_btc_daily(start: str, end: str) -> pd.Series | None:
    """Coinbase Exchange daily candles (US-accessible). 300 candles/request."""
    rows = {}
    cur = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    while cur < end_ts:
        hi = min(cur + pd.Timedelta(days=299), end_ts)
        url = (f"https://api.exchange.coinbase.com/products/BTC-USD/candles?"
               f"granularity=86400&start={cur.isoformat()}&end={hi.isoformat()}")
        batch = _http_json(url)
        if not batch:
            break
        for c in batch:                       # [time, low, high, open, close, volume]
            rows[int(c[0])] = float(c[4])
        cur = hi + pd.Timedelta(days=1)
    if not rows:
        return None
    idx = pd.to_datetime(sorted(rows), unit="s", utc=True)
    return pd.Series([rows[int(t.timestamp())] for t in idx], index=idx, name="btc_price")


def _kraken_btc_daily(start: str, end: str) -> pd.Series | None:
    since = int(pd.Timestamp(start, tz="UTC").timestamp())
    js = _http_json(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1440&since={since}")
    res = js.get("result", {})
    key = next((k for k in res if k != "last"), None)
    if not key:
        return None
    rows = res[key]
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
    s = pd.Series([float(r[4]) for r in rows], index=idx, name="btc_price")
    return s[(s.index >= pd.Timestamp(start, tz="UTC")) & (s.index <= pd.Timestamp(end, tz="UTC"))]


def _binance_btc_daily(start: str, end: str) -> Series | Unavailable:
    """BTC daily close, US-accessible sources first (Binance is 451 from US)."""
    for name, fn in (("coinbase-live", _coinbase_btc_daily), ("kraken-live", _kraken_btc_daily)):
        try:
            s = fn(start, end)
            if s is not None and len(s) > 200:
                s = s[~s.index.duplicated()].sort_index()
                return Series("btc_price", s, name, "USD", note=f"{len(s)} daily closes")
        except Exception:  # noqa: BLE001
            continue
    return Unavailable("btc_price", "no US-accessible daily BTC source returned >200 closes")


# --------------------------------------------------------------------------- #
# Live (best-effort): Deribit DVOL daily
# --------------------------------------------------------------------------- #
def _deribit_dvol_daily(start: str, end: str) -> Series | Unavailable:
    try:
        t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data?"
               f"currency=BTC&start_timestamp={t0}&end_timestamp={t1}&resolution=43200")
        js = _http_json(url)
        data = js.get("result", {}).get("data", [])
        if not data:
            return Unavailable("btc_implied_vol", "deribit dvol empty")
        idx = pd.to_datetime([d[0] for d in data], unit="ms", utc=True)
        # data rows: [ts, open, high, low, close]; use close, to daily, /100 -> frac
        iv = pd.Series([d[4] / 100.0 for d in data], index=idx, name="btc_implied_vol")
        iv = iv.resample("1D").last().dropna()
        return Series("btc_implied_vol", iv, "deribit-live", "annualized_vol_frac",
                      note=f"{len(iv)} DVOL closes")
    except Exception as e:  # noqa: BLE001
        return Unavailable("btc_implied_vol", f"deribit fetch failed: {e}")


# --------------------------------------------------------------------------- #
# Synthetic generator (tagged) — embeds a modest true signal + vol risk premium
# --------------------------------------------------------------------------- #
def _synthetic_bundle(start: str, end: str, btc_close: pd.Series | None) -> dict:
    rng = np.random.default_rng(_SEED)
    days = pd.date_range(start, end, freq="1D", tz="UTC")
    n = len(days)

    # Realized-vol regime: slow-moving latent state (calm/normal/stress)
    regime = np.zeros(n)
    state = 0.6
    for i in range(n):
        state = 0.985 * state + 0.015 * rng.normal(0.6, 0.25)
        regime[i] = max(0.15, min(1.6, state))
    base_rv = 0.45 + 0.55 * regime          # annualized realized vol, ~0.4–1.2

    # Macro signals: |Δ^vw| daily. Each loads on the regime with noise, plus
    # event spikes (FOMC / data prints). Fed channel leads realized vol; recession
    # channel is slower / steadier (matches the paper's OOS-stability finding).
    def signal(persist, load, spike_p, lead):
        s = np.zeros(n)
        prev = 0.0
        for i in range(n):
            drift = load * (regime[(i + lead) % n] - 0.6)
            prev = persist * prev + (1 - persist) * abs(rng.normal(drift, 0.04))
            if rng.random() < spike_p:
                prev += abs(rng.normal(0.10, 0.05))
            s[i] = max(0.0, prev)
        return pd.Series(s, index=days)

    kxfed = signal(persist=0.5, load=0.18, spike_p=0.04, lead=3)        # leads RV by ~3d
    kxrec = signal(persist=0.8, load=0.10, spike_p=0.01, lead=5)        # slower, steadier

    # Realized vol actually depends (modestly) on the lagged signals + regime.
    eps = rng.normal(0, 0.06, n)
    rv = (base_rv
          + 0.35 * pd.Series(kxfed).shift(3).fillna(0).values        # true predictive load
          + 0.25 * pd.Series(kxrec).shift(5).fillna(0).values
          + eps)
    rv = np.clip(rv, 0.10, 2.0)
    realized_vol = pd.Series(rv, index=days, name="btc_realized_vol_5d")

    # Implied vol = realized vol smoothed + a positive risk premium + noise.
    # Construct from a realized series via a fixed VRP transform so implied and
    # realized stay COUPLED to the SAME underlying. We define the transform once
    # here and reuse it below if we substitute a real realized series, otherwise
    # the synthetic-fallback implied would track `rv` while realized tracks real
    # prices -> the two become decoupled and payoff=pos*(real_rv - syn_implied)
    # is a systematic loss (false-negative kill of macro_vol_btc).
    vrp = 0.06 + 0.02 * rng.normal(0, 1, n)   # one VRP draw, length n, reused per-date below

    def _implied_from_realized(rv_series: pd.Series) -> pd.Series:
        """Mirror the synthetic VRP shape: smoothed realized + fixed premium + VRP.
        Aligns the per-date VRP draw to whatever index `rv_series` carries so the
        vol-risk-premium relationship is preserved on the real calendar too."""
        vrp_aligned = pd.Series(vrp, index=days).reindex(rv_series.index).fillna(0.06)
        iv_vals = (rv_series.ewm(span=7).mean() + 0.04 + vrp_aligned)
        return pd.Series(np.clip(iv_vals.values, 0.12, 2.2),
                         index=rv_series.index, name="btc_implied_vol")

    implied_vol = _implied_from_realized(pd.Series(rv, index=days))

    # If we have a real BTC price, prefer real realized vol computed from it.
    if btc_close is not None and len(btc_close) > 30:
        ret = np.log(btc_close).diff()
        rv_real = ret.rolling(5).std() * np.sqrt(365)
        # Point-in-time: never fill a value at time t with data from after t.
        # Drop the leading rolling-window NaNs (do NOT bfill them with a future value),
        # align to the requested calendar, and forward-fill only. No interpolation
        # (which would average future neighbors) — this series is used as both feature
        # and target, so any look-ahead would leak the answer.
        rv_real = rv_real.dropna()
        rv_real = rv_real.reindex(days).ffill()
        rv_real = rv_real.dropna()        # drop dates before the first real observation
        rv_real.name = "btc_realized_vol_5d"
        realized_vol = rv_real
        # Re-derive the synthetic-fallback implied from the REAL realized series so
        # implied and realized stay coupled (same underlying, same VRP shape). Only
        # the Deribit-live path (real DVOL) keeps a truly-real implied; here there
        # is no real DVOL, so the implied proxy MUST track the realized we expose.
        implied_vol = _implied_from_realized(realized_vol)

    return {
        "kxfed_signal": Series("kxfed_signal", kxfed, "synthetic", "abs_prob_change",
                               note="monetary-policy channel |Δvw|; embeds modest true RV load"),
        "kxrecssnber_signal": Series("kxrecssnber_signal", kxrec, "synthetic", "abs_prob_change",
                                     note="recession channel |Δvw|; slower/steadier"),
        "btc_realized_vol_5d": Series("btc_realized_vol_5d", realized_vol,
                                      "binance-derived" if btc_close is not None else "synthetic",
                                      "annualized_vol_frac", note="5d rolling realized vol"),
        "btc_implied_vol_syn": Series("btc_implied_vol", implied_vol, "synthetic",
                                      "annualized_vol_frac", note="DVOL proxy + vol risk premium"),
    }


def _kalshi_signal(name: str, start: str, end: str) -> "Series | None":
    """Load the REAL Kalshi macro |Δvw| signal from the local data catalog (config.DATA_DIR,
    set via PENROSE_DATA_DIR), clipped to the requested window. Returns None if the catalog/file
    is unavailable or too short, so the caller falls back to the synthetic generator (and tags
    provenance honestly). This is the one pre-collected series with no free live API; everything
    else uses keyless live venues or a keyed vendor adapter."""
    import sys
    from .. import config
    dd = str(config.DATA_DIR)
    try:
        if dd not in sys.path:
            sys.path.insert(0, dd)
        import loader as catalog
        r = catalog.load_series(name)
    except Exception:  # noqa: BLE001 — a bad reference must never crash the bundle
        return None
    if r is None:
        return None
    s, prov = r
    # The catalog loader returns tz-NAIVE UTC dates; the rest of the bundle (btc/synthetic)
    # is tz-AWARE UTC. Localize so a module that cross-joins this signal with btc vol doesn't
    # hit "Cannot join tz-naive with tz-aware".
    if getattr(s.index, "tz", None) is None:
        s.index = s.index.tz_localize("UTC")
    s = s[(s.index >= pd.Timestamp(start, tz="UTC")) & (s.index <= pd.Timestamp(end, tz="UTC"))]
    if len(s) < 30:                       # too little real history to backtest -> use synthetic
        return None
    s.name = name
    return Series(name, s, prov, "abs_prob_change",
                  note=f"REAL Kalshi |Δvw| macro signal (catalog:{name})")


def fetch_bundle(start: str = WINDOW[0], end: str = WINDOW[1]) -> DataBundle:
    """Assemble the bundle the macro_vol module needs, real-where-cheap, synthetic
    elsewhere, every piece tagged with provenance."""
    bundle = DataBundle(requested_window=(start, end))

    btc = _binance_btc_daily(start, end)
    bundle.series["btc_price"] = btc
    btc_close = btc.data if isinstance(btc, Series) else None

    syn = _synthetic_bundle(start, end, btc_close)
    # Prefer the REAL Kalshi macro |Δvw| signal (local data catalog); fall back to the
    # synthetic generator only when no real history is available. This removes the synthetic
    # tag from any verdict whose module reads the macro signal where real data now exists.
    for _nm in ("kxfed_signal", "kxrecssnber_signal"):
        _real = _kalshi_signal(_nm, start, end)
        bundle.series[_nm] = _real if _real is not None else syn[_nm]
    rv = syn["btc_realized_vol_5d"]
    if btc_close is not None:
        rv.provenance = f"{btc.provenance.split('-')[0]}-derived"   # match real price source
    bundle.series["btc_realized_vol_5d"] = rv

    dvol = _deribit_dvol_daily(start, end)
    if isinstance(dvol, Series) and dvol.coverage[2] >= 200:
        bundle.series["btc_implied_vol"] = dvol
    else:
        bundle.series["btc_implied_vol"] = syn["btc_implied_vol_syn"]
        if isinstance(dvol, Unavailable):
            bundle.series["btc_implied_vol"].note += f" (deribit fallback: {dvol.reason})"

    _add_catalog_series(bundle)          # real series via the optional local data catalog
    _add_databento_series(bundle)        # BYO point-in-time market data (if the user configured it)
    _add_regime_series(bundle)           # pre-registered point-in-time vol/trend regime labels
    _add_vendor_series(bundle)           # BYO data-vendor adapters (FRED + skeletons), fail-open
    return bundle


def _add_vendor_series(bundle: DataBundle) -> None:
    """Fold every ENABLED data-vendor adapter's configured series into the bundle.
    BYO + fail-open: no key / no package / broken vendor -> the series is simply
    absent (honest needs_data downstream), never a crash. Each series carries
    provenance + grade. The whole call is wrapped so a vendor import error can
    never break a run."""
    try:
        from .vendors import add_vendor_series
        add_vendor_series(bundle)
    except Exception:  # noqa: BLE001 — vendor framework must never crash fetch_bundle
        return


def _add_regime_series(bundle: DataBundle) -> None:
    """Add PRE-REGISTERED, point-in-time market-regime LABEL series to the bundle so any module —
    including a sandboxed auto-implemented one — can CONDITION on a regime via `bundle.get(...)`
    like any other series, WITHOUT importing penrose (the sandbox forbids it) and without fitting
    the boundary (it's fixed upstream in penrose.regime). These same labels feed the engine's
    regime kill-lens. Derived from REAL btc_price with strictly trailing windows -> no look-ahead,
    not synthetic. Hard rule: a pre-registered feature you may condition on, never a fitted detector
    inside the strategy."""
    from .. import regime as _rg
    btc = bundle.series.get("btc_price")
    if not isinstance(btc, Series) or btc.data is None or len(btc.data.dropna()) < 160:
        return
    px = btc.data.dropna()
    src = btc.provenance or "btc"
    for key, fn, desc in (
        ("btc_vol_regime", _rg.vol_regime,
         "point-in-time trailing-vol tercile {low_vol,mid_vol,high_vol}; fixed boundary, no look-ahead"),
        ("btc_trend_regime", _rg.trend_regime,
         "point-in-time trailing-MA trend {uptrend,downtrend}; fixed boundary, no look-ahead"),
    ):
        if key in bundle.series:
            continue
        try:
            lab = fn(px)
        except Exception:  # noqa: BLE001
            continue
        if lab is not None and len(lab) >= 40:
            bundle.series[key] = Series(key, lab, f"{src}-regime-pit", "regime_label", note=desc)


def _add_databento_series(bundle: DataBundle) -> None:
    """Fold the user's configured Databento series (config.DATABENTO_SERIES) into the bundle.
    BYO + fail-open: no key / no package / no entitlement -> the series is simply absent
    (honest needs_data downstream), never a crash. Cached so re-runs don't re-bill."""
    from .. import config
    specs = getattr(config, "DATABENTO_SERIES", {}) or {}
    if not specs:
        return
    try:
        from . import databento as dbento
    except Exception:  # noqa: BLE001
        return
    if not dbento.available():
        return
    for name, spec in specs.items():
        if name in bundle.series or not isinstance(spec, dict):
            continue
        try:
            r = dbento.fetch_daily(
                spec["dataset"], spec["symbol"], spec["start"], spec["end"],
                schema=spec.get("schema", "ohlcv-1d"),
                field=spec.get("field", "close"),
                stype_in=spec.get("stype_in", "raw_symbol"))
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            continue
        s, prov = r
        bundle.series[name] = Series(name, s, prov, "", note=f"databento:{name}")


def _add_catalog_series(bundle: DataBundle) -> None:
    """Add every series from the optional local data catalog (config.DATA_DIR, set via
    PENROSE_DATA_DIR) to the bundle, so auto-implemented modules can request real
    SOL/LINK/funding/weather/etc. Never overrides a series penrose already populated; silently
    skips if the catalog or a referenced file is missing (the module just sees data_unavailable)."""
    import sys
    from .. import config
    dd = str(config.DATA_DIR)
    try:
        if dd not in sys.path:
            sys.path.insert(0, dd)
        import loader as catalog  # <PENROSE_DATA_DIR>/loader.py
    except Exception:  # noqa: BLE001
        return
    if not hasattr(catalog, "available") or not hasattr(catalog, "load_series"):
        return
    for name in catalog.available():
        if name in bundle.series:
            continue
        try:
            r = catalog.load_series(name)
        except Exception:  # noqa: BLE001
            continue
        if r is None:
            continue
        s, prov = r
        # D-003: the catalog loader returns tz-NAIVE dates, but the rest of the bundle
        # (btc_price / synthetic) is tz-AWARE UTC. A module that cross-joins a catalog series
        # (eth/sol spot, weather) with a bundle-native one would hit "Cannot join tz-naive with
        # tz-aware" and degrade to needs_data even though the data exists. Localize to UTC so
        # every series in the bundle shares one tz convention (matches _kalshi_signal).
        if getattr(s.index, "tz", None) is None:
            s.index = s.index.tz_localize("UTC")
        bundle.series[name] = Series(name, s, prov, "", note=f"catalog:{name}")
