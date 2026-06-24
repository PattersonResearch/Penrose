"""Data contract — what penrose asks any data service for, and the shape it
gets back. Contract design: explicit `unavailable` rather
than silent nulls; provenance per series; point-in-time semantics.

v1 has no separate service yet, so the client in client.py implements the
contract directly against live venues, and falls back to a clearly-tagged
synthetic generator when live history is too shallow (authorized: build the
pipeline now, fix data fidelity later).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Filler tokens stripped when normalizing a series name, so naming drift from auto-generated
# modules ('price.eth_usd_spot_daily') still resolves to the catalog key ('eth_spot_daily').
_KEY_FILLER = {"usd", "close", "px", "last", "value", "series", "the", "rate"}


def _norm_key(name: str) -> str:
    s = (name or "").lower().strip()
    if "." in s:                      # drop a 'price.' / 'crypto.' style namespace prefix
        s = s.split(".")[-1]
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in _KEY_FILLER]
    return "_".join(sorted(toks))     # sorted -> order-insensitive; distinct entity tokens still separate


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
                    idx[nk] = k
            object.__setattr__(self, "_norm_index", idx)
            object.__setattr__(self, "_norm_index_keys", keys)
        hit = self._norm_index.get(target)
        if hit:
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
