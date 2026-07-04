"""SEC EDGAR fundamentals panel adapter.

This module is a data-adapter primitive for Penrose reconstruction workflows:
it fetches public company filings, extracts requested accounting concepts, and
assembles point-in-time ``Panel`` objects by filing date. It does not construct
factors, select concepts for a caller, generate signals, or make any alpha
claim.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading
import time
import urllib.request

import pandas as pd

from .panel import Panel
from .xsection import asof_panel

NAME = "sec_edgar"
PROVENANCE_GRADE = "point_in_time"

_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC's data.sec.gov REQUIRES a name+email-format User-Agent (a URL-only UA is rejected, which
# would silently fail-open to empty panels). This default is a conforming contact so the adapter
# works out of the box; users SHOULD still set SEC_EDGAR_UA to their own name + email.
_DEFAULT_UA = "Penrose OSS research penrose-oss@pattersonresearch.org"
_MIN_INTERVAL_SEC = 0.12  # ~8.3 req/s, below SEC's 10 req/s ceiling.
_rate_lock = threading.Lock()  # serialize the pacing so concurrent callers can't burst past the ceiling
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "penrose_sec_edgar_cache")
_CACHE_TTL_SEC = 24 * 60 * 60
_MAX_JSON_BYTES = 128 * 1024 * 1024
_last_call = 0.0

# concept -> ordered list of (namespace, tag, unit) candidates. Filers tag the
# same economic quantity differently, so the first candidate with data wins.
CONCEPTS = {
    "book_equity": [
        ("us-gaap", "StockholdersEquity", "USD"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ],
    "assets": [("us-gaap", "Assets", "USD")],
    "op_income": [("us-gaap", "OperatingIncomeLoss", "USD")],
    "gross_profit": [("us-gaap", "GrossProfit", "USD")],
    "revenue": [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    ],
    # CAVEAT: the fallback chain interleaves point-in-time share COUNTS (cover-page /
    # balance-sheet outstanding) with period WEIGHTED-AVERAGE shares. A filer resolved via a
    # later fallback contributes a slightly different quantity than one resolved early, so
    # cross-filer `shares` are not perfectly comparable. The adapter supplies data; a caller
    # who needs strict comparability should prefer the point-in-time tags only.
    "shares": [
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
        ("us-gaap", "CommonStockSharesOutstanding", "shares"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    ],
}


def available() -> bool:
    """Return True for this keyless adapter; network errors fail open elsewhere."""
    return True


def _user_agent() -> str:
    ua = os.environ.get("SEC_EDGAR_UA", "").strip()
    return ua or _DEFAULT_UA


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, name)


def _read_json_cache(path: str):
    try:
        st = os.stat(path)
        if st.st_size <= 0 or (time.time() - st.st_mtime) >= _CACHE_TTL_SEC:
            return None
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_JSON_BYTES + 1)
        if not raw or len(raw) > _MAX_JSON_BYTES:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 - corrupt or partial cache is ignored.
        return None


def _write_json_cache(path: str, obj) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(json.dumps(obj).encode("utf-8"))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception:  # noqa: BLE001 - cache failures never affect data assembly.
        return


def _get(url: str):
    """Rate-limited SEC JSON GET. Returns parsed JSON or None on any error."""
    global _last_call
    # Serialize pacing so concurrent callers can't both skip the sleep and burst past SEC's ceiling.
    with _rate_lock:
        wait = _MIN_INTERVAL_SEC - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(_MAX_JSON_BYTES + 1)
            if not raw or len(raw) > _MAX_JSON_BYTES:
                return None
            if str(resp.headers.get("Content-Encoding", "")).lower() == "gzip":
                raw = gzip.decompress(raw)
                if len(raw) > _MAX_JSON_BYTES:      # bound the DECOMPRESSED size too (gzip-bomb guard)
                    return None
            return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 - network/HTTP/parse errors fail open.
        return None


def ticker_cik_map() -> dict[str, str]:
    """Return ``{TICKER: CIK10}`` from SEC's public map, cached and fail-open.

    This is a data-adapter lookup for point-in-time panel assembly, not factor
    construction. Any network, cache, or parse error returns ``{}``.
    """
    cache = _cache_path("ticker_cik.json")
    cached = _read_json_cache(cache)
    if isinstance(cached, dict):
        return {str(k).upper(): str(v).zfill(10) for k, v in cached.items()}

    raw = _get(_CIK_MAP_URL)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    try:
        for row in raw.values():
            ticker = str(row["ticker"]).upper()
            cik = str(row["cik_str"]).zfill(10)
            if ticker and cik:
                out[ticker] = cik
    except Exception:  # noqa: BLE001
        return {}
    if out:
        _write_json_cache(cache, out)
    return out


def _companyfacts(cik10: str):
    """Return raw SEC companyfacts JSON for one CIK, disk-cached and fail-open."""
    cik = str(cik10).zfill(10)
    if not cik.isdigit() or len(cik) != 10:
        return None
    cache = _cache_path(f"CIK{cik}.json")
    cached = _read_json_cache(cache)
    if isinstance(cached, dict):
        return cached

    data = _get(f"{_BASE}/CIK{cik}.json")
    if isinstance(data, dict):
        _write_json_cache(cache, data)
        return data
    return None


def _extract_concept_records(facts: dict, concept: str) -> pd.DataFrame | None:
    if not isinstance(facts, dict) or concept not in CONCEPTS:
        return None

    rows = None
    for namespace, tag, unit in CONCEPTS[concept]:
        try:
            candidate = facts["facts"][namespace][tag]["units"][unit]
        except (KeyError, TypeError):
            candidate = None
        if candidate:
            rows = candidate
            break
    if not rows:
        return None

    try:
        df = pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return None
    if not {"end", "filed", "val"}.issubset(df.columns):
        return None

    out = df.loc[:, ["end", "filed", "val"]].copy()
    out["end"] = pd.to_datetime(out["end"], errors="coerce", utc=True)
    out["filed"] = pd.to_datetime(out["filed"], errors="coerce", utc=True)
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out = out.dropna(subset=["end", "filed", "val"]).sort_values("filed", kind="mergesort")
    if len(out) == 0:
        return None
    return out.reset_index(drop=True)


def concept_records(ticker, concept, *, cik_map=None) -> pd.DataFrame | None:
    """Return one ticker/concept's point-in-time records ``[end, filed, val]``.

    ``filed`` is the availability date used for no-look-ahead assembly; both
    dates are parsed as tz-aware UTC. This data-adapter primitive extracts
    public filing data only. It does not build factors or assert an edge.
    Missing mappings, failed fetches, malformed filings, and unknown concepts
    return ``None``.
    """
    try:
        cmap = cik_map if cik_map is not None else ticker_cik_map()
        cik = cmap.get(str(ticker).upper()) if isinstance(cmap, dict) else None
        if not cik:
            return None
        facts = _companyfacts(str(cik).zfill(10))
        return _extract_concept_records(facts, str(concept))
    except Exception:  # noqa: BLE001 - public adapter surface is fail-open.
        return None


def fundamentals_panel(tickers, concept, dates, *, cik_map=None, lag_days=0) -> Panel:
    """Assemble a point-in-time SEC fundamentals ``Panel`` by filing date.

    For each mapped ticker, the adapter extracts the requested SEC concept and
    delegates point-in-time alignment to ``xsection.asof_panel``. A value is
    visible only once ``filed + lag_days <= date``. Unmapped, failed, malformed,
    or empty entities are absent. This is a data-adapter primitive, not factor
    construction, signal generation, or an alpha claim.
    """
    try:
        target_dates = pd.DatetimeIndex(dates)
        if target_dates.tz is None:
            target_dates = target_dates.tz_localize("UTC")
        else:
            target_dates = target_dates.tz_convert("UTC")
    except Exception:  # noqa: BLE001
        target_dates = pd.DatetimeIndex([], tz="UTC")

    provenance = f"{NAME}:{concept}:{PROVENANCE_GRADE}"
    try:
        if tickers is None:
            symbols: list[str] = []
        elif isinstance(tickers, str):
            symbols = [tickers.upper()]
        else:
            symbols = [str(t).upper() for t in tickers]
    except Exception:  # noqa: BLE001
        symbols = []

    if not symbols:
        return Panel(str(concept), pd.DataFrame(index=target_dates), provenance, kind="characteristic")

    records: dict[str, pd.DataFrame] = {}
    try:
        cmap = cik_map if cik_map is not None else ticker_cik_map()
        if not isinstance(cmap, dict):
            cmap = {}
        lag = pd.Timedelta(days=max(0, int(lag_days or 0)))
    except Exception:  # noqa: BLE001
        cmap = {}
        lag = pd.Timedelta(0)

    for symbol in symbols:
        rec = concept_records(symbol, concept, cik_map=cmap)
        if rec is None or len(rec) == 0:
            continue
        if lag != pd.Timedelta(0):
            rec = rec.copy()
            rec["filed"] = pd.to_datetime(rec["filed"], errors="coerce", utc=True) + lag
            rec = rec.dropna(subset=["filed"])
        if len(rec) > 0:
            records[symbol] = rec

    try:
        panel = asof_panel(records, str(concept), target_dates)
        return Panel(
            name=str(concept),
            data=panel.data,
            provenance=f"{provenance}; {panel.provenance}",
            kind="characteristic",
            note="SEC EDGAR data-adapter panel; point-in-time by filing date; not factor construction",
        )
    except Exception:  # noqa: BLE001 - assembly failures degrade to an empty panel.
        return Panel(str(concept), pd.DataFrame(index=target_dates), provenance, kind="characteristic")
