"""Data-vendor adapter framework (generalizes the BYO Databento pattern).

Each vendor is a small module exposing a common protocol:

    NAME: str
    PROVENANCE_GRADE: "point_in_time" | "as_displayed"
    available() -> bool
        True ONLY if the vendor's API-key env var is set AND any needed lib is
        importable. Never raises.
    fetch(spec: dict) -> tuple[pandas.Series, str] | None
        Series indexed by tz-aware UTC dates; second item is a provenance string.
        Returns None on ANY failure — must never raise.

BYO model (mirrors BYO LLM tokens / Databento): the USER brings their own vendor
account. With no key / no package, that vendor is simply UNAVAILABLE — its series
never enter the bundle, the claim falls to an honest `needs_data`, and a run never
crashes. No license burden on penrose.

Vendor-series CONFIG lives HERE (DEFAULT_SERIES), not in config.py, so the whole
framework — registry, specs, fold-in — is self-contained in this package.

HARD RULE — fail-open everywhere: a missing or broken vendor must NEVER raise or
crash `fetch_bundle`, and must NEVER override a series the bundle already carries.

Provenance grades:
  * point_in_time — survivorship-aware, as-of-date snapshots (e.g. Databento). The
    strongest defense against look-ahead; a kill on such data is worth far more.
  * as_displayed   — current/revised values as the vendor displays them today
    (e.g. FRED, most equity-bar vendors). Honest, but may embed revisions.

Only FRED is CERTIFIED (wired + live-key tested). The others are framework-ready
EXPERIMENTAL skeletons until verified against a live key.
"""
from __future__ import annotations

from . import alpaca, alphavantage, fred, polygon, tiingo

# Adapter registry: NAME -> module. Each module implements the protocol above.
ADAPTERS = {
    fred.NAME: fred,
    polygon.NAME: polygon,
    tiingo.NAME: tiingo,
    alpaca.NAME: alpaca,
    alphavantage.NAME: alphavantage,
}

# Logical bundle key -> {"vendor": <NAME>, ...vendor-specific spec}.
# Kept HERE (not config.py). Empty by default: the framework is opt-in — a user
# adds entries (or sets keys) to enable real fetches. Example shape:
#   "us_10y_treasury": {"vendor": "fred", "series_id": "DGS10",
#                       "start": "2023-01-01", "unit": "pct", "field": "us_10y"},
DEFAULT_SERIES: dict[str, dict] = {
    # FRED 10y treasury constant-maturity yield — the canonical certified example.
    # Active whenever FRED_API_KEY is set; otherwise fail-open (simply absent).
    "us_10y_treasury": {
        "vendor": "fred",
        "series_id": "DGS10",
        "start": "2023-01-01",
        "unit": "pct",
    },
    # Alpha Vantage SPY daily close — the canonical certified equities example.
    # Active whenever ALPHAVANTAGE_API_KEY is set; otherwise fail-open (simply absent).
    # outputsize="compact" (~100 trading days): "full" is an Alpha Vantage PREMIUM feature,
    # so the free tier must use compact, and the adapter fails open on the premium/throttle note.
    "us_equity_spy": {
        "vendor": "alphavantage",
        "symbol": "SPY",
        "field": "4. close",
        "outputsize": "compact",
        "unit": "usd",
    },
}


def enabled_adapters() -> dict:
    """Return {NAME: module} for every adapter whose `available()` is True.

    Fail-open: an adapter whose `available()` raises is treated as unavailable.
    """
    out = {}
    for name, mod in ADAPTERS.items():
        try:
            if mod.available():
                out[name] = mod
        except Exception:  # noqa: BLE001 — a broken adapter is just unavailable
            continue
    return out


def add_vendor_series(bundle) -> None:
    """Fold every ENABLED adapter's configured series into `bundle`, vendor-tagged.

    Fail-open at every level: a missing/broken vendor, a bad spec, or a failed
    fetch is silently skipped — it NEVER raises and NEVER crashes `fetch_bundle`.
    An existing series in the bundle is NEVER overridden.

    Each added series carries provenance (vendor string) and grade (the adapter's
    PROVENANCE_GRADE), appended to the provenance so a verdict can weight by it.
    """
    try:
        from ..contract import Series  # local import keeps the package import-light
    except Exception:  # noqa: BLE001
        return

    enabled = enabled_adapters()
    if not enabled:
        return

    for key, spec in (DEFAULT_SERIES or {}).items():
        try:
            if key in getattr(bundle, "series", {}):
                continue                      # never override an existing series
            if not isinstance(spec, dict):
                continue
            mod = enabled.get(spec.get("vendor"))
            if mod is None:
                continue                      # vendor disabled / unknown -> skip
            r = mod.fetch(spec)
            if r is None:
                continue
            s, prov = r
            if s is None or len(s) == 0:
                continue
            grade = getattr(mod, "PROVENANCE_GRADE", "as_displayed")
            s.name = key
            bundle.series[key] = Series(
                key, s, f"{prov} [{grade}]", spec.get("unit", ""),
                note=f"vendor:{mod.NAME}:{key}")
        except Exception:  # noqa: BLE001 — one bad series never breaks the rest
            continue
