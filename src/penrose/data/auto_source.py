"""Auto-source-and-archive data layer (OPT-IN, default OFF via config.AUTO_SOURCE).

Today Penrose reads series from a local catalog (PENROSE_DATA_DIR/catalog.yaml ->
vendor/*.parquet), populated MANUALLY by fetch_*.py scripts. When a run needs a series
the catalog lacks, it returns `needs_data`. This module closes the automatic loop for a
commercial / self-hosted deployment that opts in:

    on a MISS  ->  resolve the source  ->  fetch via an existing vendor adapter
               ->  ARCHIVE locally (parquet)  ->  REGISTER in catalog.yaml  ->  continue

The archive is the point: a series is pulled ONCE and reused forever (idempotent +
staleness-bounded), so the same source is never re-queried.

Discipline (docs/DATA_ACQUISITION_STANDARD.md):
  * Provenance + `sourced_at` recorded on every archived series.
  * `pit: false` (as-collected) unless the source is genuinely point-in-time — none of
    FRED/Tiingo/CoinGecko as wired here are, so all are marked `pit: false`.
  * NO silent proxies: only the series actually requested, from the declared/detected
    source, is archived. A universe/panel request is NOT survivorship-corrected here.
  * NO secret ever touches the catalog or code — adapters read keys from env only.

Fail-open everywhere: any failure returns None and the caller keeps today's honest
`needs_data` behavior. Never raises.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# A small, conservative allowlist of well-known US equity tickers for the HEURISTIC
# fallback (no explicit source hint). Kept deliberately tight: an unknown symbol simply
# fails to resolve (-> needs_data) rather than guessing a wrong source. An explicit
# `source` hint on the request bypasses this entirely and is always preferred.
_KNOWN_EQUITY_TICKERS = frozenset({
    "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "NVDA", "TSLA", "NFLX", "AMD",
    "INTC", "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "XLF", "XLE",
    "XLK", "VTI", "VOO", "BRK.B", "JPM", "BAC", "V", "MA", "DIS", "KO", "PEP",
})

# A FRED-style series id: all-caps alphanumerics, length >= 3, at least one letter
# (e.g. DGS10, CPIAUCSL, UNRATE, T10YIE). Deliberately excludes short pure-alpha
# tickers, which route to the equity allowlist above.
_FRED_ID_RE = re.compile(r"^[A-Z][A-Z0-9]{2,}$")

# CoinGecko coin ids for the heuristic fallback (lowercase slugs only reachable via an
# explicit hint in practice; kept minimal).
_KNOWN_COINGECKO_IDS = frozenset({"bitcoin", "ethereum", "solana"})

_DOMAIN_BY_SOURCE = {"fred": "macro", "tiingo": "equity", "coingecko": "crypto"}
_UNIT_BY_SOURCE = {"fred": "", "tiingo": "usd", "coingecko": "usd"}


# --------------------------------------------------------------------------- #
# Config access (fail-open defaults so the module is import-safe in isolation)
# --------------------------------------------------------------------------- #
def _cfg() -> dict:
    try:
        from .. import config
        cfg = getattr(config, "AUTO_SOURCE", None)
        if isinstance(cfg, dict):
            return cfg
    except Exception:  # noqa: BLE001
        pass
    return {"enabled": False, "max_stale_days": 7, "allow_sources": ["fred", "tiingo", "coingecko"]}


def _data_dir(data_dir) -> Path | None:
    if data_dir:
        return Path(data_dir)
    try:
        from .. import config
        return Path(config.DATA_DIR)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Source detection
# --------------------------------------------------------------------------- #
def detect_source(name: str, spec=None, allow_sources=None) -> dict | None:
    """Return a normalized source descriptor for `name`, or None.

    A descriptor is {"source": <adapter>, "id": <vendor id>, ...optional fetch kwargs}.

    Detection order (PREFERRED first):
      1. An EXPLICIT hint. Either the request `spec` IS a source dict
         ({"source": "fred", "id": "DGS10"}) or it carries one under a "source" key
         ({"source": {"source": "tiingo", "id": "AAPL", "field": "adjClose"}}).
      2. A light HEURISTIC on the bare name: a FRED-style all-caps id -> fred; a
         known equity ticker -> tiingo; a known CoinGecko slug -> coingecko.

    Only sources in `allow_sources` are returned. Returns None (-> caller stays
    needs_data) when nothing can be determined. Never raises.
    """
    if allow_sources is None:
        allow_sources = _cfg().get("allow_sources", ["fred", "tiingo", "coingecko"])
    allow = set(allow_sources or ())
    try:
        hint = _extract_hint(spec)
        if hint is not None:
            src = str(hint.get("source", "")).strip().lower()
            vid = hint.get("id") or hint.get("series_id") or hint.get("symbol") or name
            if src in allow and vid:
                out = {"source": src, "id": str(vid)}
                for k in ("field", "start", "end"):
                    if hint.get(k) is not None:
                        out[k] = hint[k]
                return out
            # An explicit hint naming a disallowed/unknown source is NOT overridden by a
            # heuristic guess — that would be a silent proxy. Fall through to None.
            if src:
                return None
        # Heuristic fallback on the bare name.
        return _heuristic_source(name, allow)
    except Exception:  # noqa: BLE001 — detection is fail-open
        return None


def _extract_hint(spec) -> dict | None:
    if not isinstance(spec, dict):
        return None
    inner = spec.get("source")
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        return spec               # spec itself is the source hint dict
    return None


def _heuristic_source(name: str, allow: set) -> dict | None:
    if not name:
        return None
    token = name.strip()
    if token in _KNOWN_EQUITY_TICKERS and "tiingo" in allow:
        return {"source": "tiingo", "id": token}
    if token.lower() in _KNOWN_COINGECKO_IDS and "coingecko" in allow:
        return {"source": "coingecko", "id": token.lower()}
    if "fred" in allow and _FRED_ID_RE.match(token) and token not in _KNOWN_EQUITY_TICKERS:
        return {"source": "fred", "id": token}
    return None


# --------------------------------------------------------------------------- #
# Adapter dispatch — reuse the existing certified/experimental adapters
# --------------------------------------------------------------------------- #
def _fetch_from_source(desc: dict):
    """Call the matching adapter. Return (pd.Series, provenance) or None. Never raises."""
    src = desc.get("source")
    vid = desc.get("id")
    try:
        if src == "fred":
            from .vendors import fred
            spec = {"series_id": vid}
            if desc.get("start"):
                spec["start"] = desc["start"]
            if desc.get("end"):
                spec["end"] = desc["end"]
            return fred.fetch(spec)
        if src == "tiingo":
            from .vendors import tiingo
            spec = {"symbol": vid, "field": desc.get("field", "close")}
            if desc.get("start"):
                spec["start"] = desc["start"]
            if desc.get("end"):
                spec["end"] = desc["end"]
            return tiingo.fetch(spec)
        if src == "coingecko":
            return _fetch_coingecko(vid)
    except Exception:  # noqa: BLE001 — adapter failure is a miss, never a crash
        return None
    return None


def _fetch_coingecko(coin_id: str):
    """CoinGecko daily market cap (demo tier). No shared adapter exists, so this mirrors
    the data-repo fetch_vendors.py path. Requires COINGECKO_API_KEY; returns None otherwise."""
    key = os.environ.get("COINGECKO_API_KEY")
    if not key or not coin_id:
        return None
    import json
    import urllib.request
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days=365")
    req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1",
                                               "x-cg-demo-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            arr = (json.loads(r.read().decode()).get("market_caps")) or []
    except Exception:  # noqa: BLE001
        return None
    if not arr:
        return None
    idx = pd.to_datetime([int(p[0]) for p in arr], unit="ms", utc=True)
    s = pd.Series([float(p[1]) for p in arr], index=idx, name=str(coin_id))
    s = s.resample("1D").last().dropna()
    return (s, f"coingecko-api:{coin_id}") if not s.empty else None


# --------------------------------------------------------------------------- #
# Archive + catalog registration
# --------------------------------------------------------------------------- #
def _write_parquet(path: Path, s: pd.Series) -> None:
    """Write a single series as (date, value) parquet — matches fetch_vendors.py `_write`
    (tz-naive UTC dates, numeric value, NaNs dropped)."""
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame({"date": idx, "value": pd.to_numeric(s.values, errors="coerce")}).dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _register_catalog_entry(catalog_path: Path, name: str, entry: dict) -> None:
    """Append a `series[name]` entry to catalog.yaml, idempotently and WITHOUT clobbering
    an existing entry. Uses a targeted text insert right after the `series:` line so any
    hand-authored comments/formatting in an existing catalog are PRESERVED; falls back to
    a yaml round-trip only when the file is absent or has no `series:` block."""
    import yaml
    # Never re-register: honor an existing entry exactly as-is.
    if catalog_path.exists():
        try:
            existing = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict) and name in (existing.get("series") or {}):
                return
        except Exception:  # noqa: BLE001 — unreadable catalog: fall through to text path
            existing = None

    inline = yaml.safe_dump({name: entry}, default_flow_style=True,
                            sort_keys=False, width=10_000).strip()
    # yaml.safe_dump wraps a top-level mapping as `{name: {...}}`; strip the outer braces
    # so we emit `name: {...}` at 2-space indent under `series:`.
    if inline.startswith("{") and inline.endswith("}"):
        inline = inline[1:-1].strip()
    new_line = f"  {inline}\n"

    if catalog_path.exists():
        text = catalog_path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        for i, ln in enumerate(lines):
            if re.match(r"^series\s*:\s*$", ln.strip()) or ln.rstrip("\n") == "series:":
                lines.insert(i + 1, new_line)
                catalog_path.write_text("".join(lines), encoding="utf-8")
                return
        # No `series:` block found — append one.
        sep = "" if text.endswith("\n") else "\n"
        catalog_path.write_text(text + f"{sep}series:\n{new_line}", encoding="utf-8")
        return

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(f"series:\n{new_line}", encoding="utf-8")


def _build_entry(name: str, desc: dict, provenance: str, max_stale_days: int) -> dict:
    """Catalog entry for the archived series: `col` adapter (date/value), status vendor,
    provenance + sourced_at, pit:false (as-collected). Matches the local data catalog
    shape for FRED/Tiingo vendor series."""
    src = desc.get("source", "")
    return {
        "domain": _DOMAIN_BY_SOURCE.get(src, "unknown"),
        "path": f"vendor/{name}.parquet",
        "adapter": "col",
        "date_col": "date",
        "value_col": "value",
        "provenance": provenance,
        "status": "vendor",
        "sourced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_stale_days": int(max_stale_days),
        "pit": False,           # none of FRED/Tiingo/CoinGecko as wired here is point-in-time
        "day_basis": "utc",
        "unit": _UNIT_BY_SOURCE.get(src, ""),
    }


# --------------------------------------------------------------------------- #
# Load an archived parquet back into a tz-aware daily series
# --------------------------------------------------------------------------- #
def _load_archived(path: Path, name: str) -> pd.Series | None:
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if "date" not in df or "value" not in df or df.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=idx.normalize(), name=name).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if not s.empty else None


def _is_fresh(path: Path, max_stale_days: int) -> bool:
    """True if the archived parquet was written within `max_stale_days` (by mtime). This
    is the idempotency guard: pull once, reuse — a fresh archive is never re-fetched."""
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        return age_days <= float(max_stale_days)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def resolve_and_archive(name: str, *, spec=None, data_dir=None) -> dict | None:
    """Resolve `name` to a source, fetch via the matching adapter, archive it locally, and
    register it in catalog.yaml. Return a descriptor dict, or None on any failure.

    Return dict keys: name, source, provenance, unit, path, series (tz-aware UTC daily
    pd.Series), sourced_at, cached (True if served from a fresh existing archive).

    IDEMPOTENT: if <data_dir>/vendor/<name>.parquet already exists AND is within
    `max_stale_days`, the archive is reused WITHOUT re-fetching (this is the whole point).

    Never raises. On an unsourceable name, disabled/unknown source, or failed fetch it
    returns None so the caller keeps today's honest `needs_data` behavior.
    """
    try:
        if not name:
            return None
        # Name-safety: a series name is a catalog identifier, never a path fragment. Reject anything with a
        # slash/backslash/`..` or non-identifier char so `vendor/<name>.parquet` can never escape vendor/
        # (defends the explicit-hint path where detect_source's whitelist wouldn't otherwise gate the name).
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", name):
            return None
        cfg = _cfg()
        max_stale = int(cfg.get("max_stale_days", 7))
        dd = _data_dir(data_dir)
        if dd is None:
            return None
        parquet = dd / "vendor" / f"{name}.parquet"

        # 1) Idempotency: a fresh existing archive is reused, never re-fetched.
        if parquet.exists() and _is_fresh(parquet, max_stale):
            s = _load_archived(parquet, name)
            if s is not None:
                return {"name": name, "source": None, "provenance": _archived_provenance(dd, name),
                        "unit": "", "path": str(parquet), "series": s,
                        "sourced_at": None, "cached": True}

        # 2) Resolve the source (explicit hint preferred, else heuristic).
        desc = detect_source(name, spec=spec, allow_sources=cfg.get("allow_sources"))
        if desc is None:
            return None

        # 3) Fetch via the matching adapter.
        fetched = _fetch_from_source(desc)
        if fetched is None:
            return None
        s, provenance = fetched
        if s is None or len(s) == 0:
            return None
        if getattr(s.index, "tz", None) is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")

        # 4) Archive + register (idempotent, non-clobbering).
        _write_parquet(parquet, s)
        entry = _build_entry(name, desc, provenance, max_stale)
        _register_catalog_entry(dd / "catalog.yaml", name, entry)

        return {"name": name, "source": desc.get("source"), "provenance": provenance,
                "unit": entry.get("unit", ""), "path": str(parquet), "series": s,
                "sourced_at": entry["sourced_at"], "cached": False}
    except Exception:  # noqa: BLE001 — the whole loop is fail-open
        return None


def _archived_provenance(data_dir: Path, name: str) -> str:
    """Best-effort read of the registered provenance for a cached series (fail-open)."""
    try:
        import yaml
        cat = yaml.safe_load((data_dir / "catalog.yaml").read_text(encoding="utf-8")) or {}
        return str((cat.get("series") or {}).get(name, {}).get("provenance", "") or "archived")
    except Exception:  # noqa: BLE001
        return "archived"
