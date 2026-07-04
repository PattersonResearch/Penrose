"""Panel data contract for cross-sectional reconstruction primitives.

The Panel type is a provenance-carrying data boundary for dates x entities
data. It supports reconstruction of claims that describe cross-sectional
portfolios; it is not a signal generator and makes no alpha claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Panel:
    """A cross-sectional reconstruction panel: UTC dates x entity columns -> float.

    These panels are referee inputs for faithfully reconstructing a claim's
    described portfolio, not recommendations, signal generation, or evidence of
    an edge. ``kind`` describes the values so transforms can guard obvious
    misuse: ``"return"``, ``"price"``, or ``"characteristic"``.
    """

    name: str
    data: pd.DataFrame
    provenance: str
    kind: str = "characteristic"
    unit: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Panel.data must be a pandas DataFrame")
        if self.kind not in {"return", "price", "characteristic"}:
            raise ValueError("Panel.kind must be 'return', 'price', or 'characteristic'")
        if not self.data.columns.is_unique:
            raise ValueError("Panel.data columns must be unique entity ids")

        df = self.data.copy()
        if len(df) == 0 and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex([], tz="UTC")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Panel.data index must be a pandas DatetimeIndex")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        if df.index.has_duplicates:
            raise ValueError("Panel.data index must not contain duplicate dates")

        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        # Drop all-NaN ENTITIES (a column with no data anywhere is useless), but KEEP all-NaN
        # DATES: a date with no cross-section is still a valid point on the time axis, and
        # transforms like cross_sectional_zscore intentionally emit NaN rows for sparse dates.
        df = df.dropna(axis=1, how="all")
        object.__setattr__(self, "data", df.astype(float))

    @property
    def coverage(self) -> tuple[Optional[str], Optional[str], int, int]:
        """Return ``(first_date, last_date, n_dates, n_entities)``."""
        if self.data is None or len(self.data) == 0:
            return None, None, 0, 0
        return (
            str(self.data.index[0].date()),
            str(self.data.index[-1].date()),
            len(self.data),
            len(self.data.columns),
        )
