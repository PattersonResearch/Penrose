"""Data contract — what penrose asks any data service for, and the shape it
gets back. Contract design: explicit `unavailable` rather
than silent nulls; provenance per series; point-in-time semantics.

v1 has no separate service yet, so the client in client.py implements the
contract directly against live venues, and falls back to a clearly-tagged
synthetic generator when live history is too shallow (authorized: build the
pipeline now, fix data fidelity later).
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Optional

import pandas as pd

from .loader_protocol import CatalogLoaderProtocol

LOG = logging.getLogger(__name__)

# Filler tokens stripped when normalizing a series name, so naming drift from auto-generated
# modules ('price.eth_usd_spot_daily') still resolves to the catalog key ('eth_spot_daily').
_KEY_FILLER = {"usd", "close", "px", "last", "value", "series", "the", "rate"}


def _key_tokens(name: str, *, drop_filler: bool) -> list[str]:
    s = (name or "").lower().strip()
    if "." in s:                      # drop a 'price.' / 'crypto.' style namespace prefix
        s = s.split(".")[-1]
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if drop_filler:
        toks = [t for t in toks if t not in _KEY_FILLER]
    return toks


def _norm_key(name: str) -> str:
    toks = _key_tokens(name, drop_filler=True)
    return "_".join(sorted(toks))     # sorted -> order-insensitive; distinct entity tokens still separate


def load_catalog_loader(data_dir):
    """Import <data_dir>/loader.py without leaving data_dir on sys.path."""
    dd = str(data_dir)
    sentinel = object()
    prior_loader = sys.modules.get("loader", sentinel)
    prior_path = list(sys.path)
    try:
        if dd not in sys.path:
            sys.path.insert(0, dd)
        sys.modules.pop("loader", None)
        return import_module("loader")
    finally:
        if prior_loader is sentinel:
            sys.modules.pop("loader", None)
        else:
            sys.modules["loader"] = prior_loader
        sys.path[:] = prior_path


@dataclass
class Series:
    """One time series with provenance attached to every delivery."""
    name: str
    data: pd.Series                      # DatetimeIndex (UTC, daily) -> float
    provenance: str                      # "kalshi-live" | "binance-live" | "synthetic" | ...
    unit: str
    available: bool = True
    note: str = ""

    @property
    def coverage(self) -> tuple[Optional[str], Optional[str], int]:
        if self.data is None or len(self.data) == 0:
            return None, None, 0
        return (str(self.data.index[0].date()), str(self.data.index[-1].date()), len(self.data))


@dataclass
class Unavailable:
    """Returned instead of a Series when the contract cannot be satisfied.
    Routed to Action Required (P9), never interpolated silently."""
    name: str
    reason: str
    available: bool = False


@dataclass
class DataBundle:
    """Everything one module run needs, each piece carrying its provenance."""
    series: dict = field(default_factory=dict)        # name -> Series | Unavailable
    requested_window: tuple = ("", "")
    fallback_substitutions: list[str] = field(default_factory=list)

    def get(self, name: str):
        """Resolve a series by name, tolerating naming drift from auto-generated modules.
        Tries the exact key, then a normalized alias (strips a 'price.'/'crypto.' prefix,
        drops '_usd'/'close'/'daily' filler, collapses separators) so a module asking for
        'price.eth_usd_spot_daily' still finds 'eth_spot_daily' instead of a false blocker.
        Never fuzzy-matches across distinct entities — only exact normalized-key hits."""
        s = self.series.get(name)
        if s is not None:
            self._note_access(name)
            return s
        if not name:
            return None
        target = _norm_key(name)
        if not target:                    # all-filler query (e.g. "rate"/"USD Close"): never alias-match
            return None
        # Rebuild the alias index if it's missing or stale. `series` is mutable, so a
        # series added after the first get() must still be visible to alias lookup.
        # Invalidate on the KEY SET (not the length): a same-length mutation such as a
        # rename, or a remove+add, leaves the length unchanged but the keys different,
        # which would otherwise serve a stale alias index.
        keys = frozenset(self.series.keys())
        if (not hasattr(self, "_norm_index") or self._norm_index is None
                or getattr(self, "_norm_index_keys", None) != keys):
            # Skip empty normalized keys so an all-filler catalog name can never become
            # a catch-all that swallows unrelated all-filler queries.
            idx = {}
            for k in self.series:
                nk = _norm_key(k)
                if nk:
                    idx.setdefault(nk, []).append(k)
            object.__setattr__(self, "_norm_index", idx)
            object.__setattr__(self, "_norm_index_keys", keys)
        hit = _unique_alias_hit(target, self._norm_index, keys)
        if hit:
            LOG.info("resolved data series alias %r -> %r", name, hit)
            self._note_access(hit)
            return self.series.get(hit)
        return None

    # --- per-verdict synthetic tracking: which series did THIS module actually read? -------
    # `any_synthetic()` is bundle-level (trips if a synthetic series merely EXISTS), which
    # falsely flags a verdict that never touched it. These let the caller scope the synthetic
    # flag to what the module actually consumed.
    def _note_access(self, key: str) -> None:
        acc = getattr(self, "_accessed", None)
        if acc is None:
            acc = set(); object.__setattr__(self, "_accessed", acc)
        acc.add(key)

    def reset_access(self) -> None:
        object.__setattr__(self, "_accessed", set())

    def accessed_synthetic(self) -> bool:
        for k in (getattr(self, "_accessed", None) or set()):
            v = self.series.get(k)
            if isinstance(v, Series) and getattr(v, "provenance", "") == "synthetic":
                return True
        return False

    def provenance_summary(self) -> dict:
        out = {}
        for k, v in self.series.items():
            if isinstance(v, Series):
                lo, hi, n = v.coverage
                out[k] = {"provenance": v.provenance, "from": lo, "to": hi, "n": n, "note": v.note}
            else:
                out[k] = {"provenance": "unavailable", "reason": v.reason}
        return out

    def any_synthetic(self) -> bool:
        return any(isinstance(v, Series) and v.provenance == "synthetic"
                   for v in self.series.values())

    def granularity_warnings(self, expected: str = "daily") -> list[dict]:
        """Advisory: report any available series whose inferred sampling frequency does not match
        `expected`. A wrong-frequency input (e.g. intraday bars where a rule assumes daily) silently
        corrupts every downstream statistic; this surfaces it at the data boundary. Non-gating and
        fail-open (a series whose frequency cannot be inferred is never flagged)."""
        from .granularity import check_granularity
        out = []
        for k, v in self.series.items():
            if isinstance(v, Series) and getattr(v, "data", None) is not None:
                chk = check_granularity(v.data, expected=expected)
                if not chk["ok"]:
                    out.append({"name": k, **chk})
        return out


def _unique_alias_hit(target: str, index: dict[str, list[str]], all_keys=None) -> str | None:
    """Resolve only a unique high-confidence alias hit.

    Exact normalized-key matches are allowed, as are unique normalized prefix/suffix
    matches for multi-token names. Ambiguous buckets deliberately miss.
    """
    direct = index.get(target) or []
    if len(direct) == 1:
        hit = direct[0]
        return None if _has_qualifier_sibling(hit, all_keys or ()) else hit
    if direct:
        return None

    target_parts = target.split("_")
    if len(target_parts) < 2:
        return None
    hits = []
    for nk, keys in index.items():
        if len(keys) != 1:
            continue
        nk_parts = nk.split("_")
        if len(nk_parts) < 2:
            continue
        if (nk.startswith(f"{target}_") or nk.endswith(f"_{target}")
                or target.startswith(f"{nk}_") or target.endswith(f"_{nk}")):
            hits.append(keys[0])
    unique_hits = set(hits)
    if len(unique_hits) != 1:
        return None
    hit = hits[0]
    return None if _has_qualifier_sibling(hit, all_keys or ()) else hit


def _has_qualifier_sibling(hit: str, all_keys) -> bool:
    """Return True when another key is the same base plus one qualifier."""
    base = set(_key_tokens(hit, drop_filler=True))
    if not base:
        return False
    for other in all_keys:
        if other == hit:
            continue
        other_tokens = set(_key_tokens(other, drop_filler=False))
        if base.issubset(other_tokens) and len(other_tokens - base) == 1:
            return True
    return False
