"""Tiingo IEX intraday adapter — BYO key, price-only by default.

This is a data adapter, not strategy construction. It fetches and shapes
intraday OHLC bars from Tiingo's IEX endpoint and makes no claim that any
strategy is profitable. IEX volume is single-venue, so volume is withheld unless
the caller explicitly opts in and the returned data is tagged accordingly.

BYO model: set `TIINGO_API_KEY` (https://www.tiingo.com). With no key, this
adapter is unavailable. All network, HTTP, and parse failures fail open to None.

Grade = "as_displayed": intraday bars as Tiingo displays them, not
point-in-time vintages.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

NAME = "tiingo_iex"
PROVENANCE_GRADE = "as_displayed"
_VENUE = "single_venue"  # IEX is one lit venue (~2-5% of consolidated US volume)

_BASE = "https://api.tiingo.com/iex"
_PRICE_COLUMNS = ("date", "open", "high", "low", "close")


def available() -> bool:
    """True iff TIINGO_API_KEY is set. Never raises."""
    try:
        return bool(os.environ.get("TIINGO_API_KEY"))
    except Exception:  # noqa: BLE001
        return False


def _download_json(url: str):
    """Return decoded JSON from `url`, or None on any network/parse error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "penrose/0.1", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _parse_rows(rows, ticker: str, include_volume: bool) -> pd.DataFrame | None:
    """Parse Tiingo IEX rows into a UTC-indexed OHLC DataFrame. None on failure."""
    try:
        if not isinstance(rows, list) or not rows:
            return None
        columns = list(_PRICE_COLUMNS[1:])
        if include_volume:
            columns.append("volume")

        records = []
        dates = []
        for row in rows:
            if not isinstance(row, dict) or row.get("date") is None:
                continue
            dates.append(row.get("date"))
            records.append({col: row.get(col) for col in columns})
        if not records:
            return None

        idx = pd.to_datetime(dates, utc=True, errors="coerce")
        df = pd.DataFrame(records, index=idx)
        df = df[~df.index.isna()]
        if df.empty:
            return None
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.empty:
            return None
        df = df[columns]
        if include_volume:
            # Self-documenting column name so a downstream consumer cannot silently treat
            # IEX's single-venue (~2-5% of tape) volume as consolidated volume — a misuse
            # of `df["volume"]` becomes a KeyError instead of a wrong number.
            df = df.rename(columns={"volume": "volume_single_venue"})
        df.index.name = "date"
        df.attrs["source"] = NAME
        df.attrs["venue"] = _VENUE
        if include_volume:
            df.attrs["volume_venue"] = _VENUE
        df.attrs["ticker"] = str(ticker)
        return df
    except Exception:  # noqa: BLE001
        return None


def fetch_intraday(
    ticker,
    start,
    end,
    *,
    freq: str = "5min",
    include_volume: bool = False,
) -> tuple[pd.DataFrame, str] | None:
    """Intraday OHLC(+opt. single-venue volume) bars for a ticker.

    Returns a tz-aware UTC-indexed DataFrame plus provenance string, or None.
    `freq` is a Tiingo intraday resampleFreq ("1min", "5min", "30min", ...).
    By default the request is price-only: IEX volume is single-venue and must not
    be used as a relative-volume signal. Provenance:
    "tiingo-iex:<ticker>:<freq>:single_venue". Fail-open -> None.
    """
    if not available() or not ticker or not start or not end:
        return None
    if any(u in str(freq).lower() for u in ("day", "week", "month", "year")):
        return None  # IEX endpoint is intraday-only; a daily+ freq is a caller error
    try:
        columns = list(_PRICE_COLUMNS)
        if include_volume:
            columns.append("volume")
        params = {
            "startDate": str(start),
            "endDate": str(end),
            "resampleFreq": str(freq),
            "columns": ",".join(columns),
            "token": os.environ["TIINGO_API_KEY"],
        }
        safe_ticker = urllib.parse.quote(str(ticker))
        url = f"{_BASE}/{safe_ticker}/prices?{urllib.parse.urlencode(params)}"
        rows = _download_json(url)
        df = _parse_rows(rows, str(ticker), bool(include_volume))
        if df is None or df.empty:
            return None
        return df, f"tiingo-iex:{ticker}:{freq}:{_VENUE}"
    except Exception:  # noqa: BLE001
        return None


def fetch(spec: dict):  # noqa: D401 — daily-protocol stub; this adapter is intraday-only
    """Daily vendor-protocol stub: always None. Use fetch_intraday() for IEX bars.

    Present so a uniform enabled_adapters() iteration that calls fetch(spec) cannot
    AttributeError on this intraday-only adapter. Fail-open by construction.
    """
    return None
