"""Protocol for bring-your-own local catalog loaders.

Penrose imports ``<PENROSE_DATA_DIR>/loader.py`` and calls this small module
surface. The loader stays outside the package; this Protocol documents the
contract without coupling Penrose to any private catalog implementation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


CatalogSeriesResult = tuple[pd.Series, str] | None


@runtime_checkable
class CatalogLoaderProtocol(Protocol):
    """Runtime shape of ``<PENROSE_DATA_DIR>/loader.py``.

    A catalog series is a scalar daily time series: ``pandas.Series`` with a
    ``DatetimeIndex`` at UTC daily granularity mapped to ``float`` values.
    If the loader returns a timezone-naive index, Penrose treats it as UTC
    before placing it in a ``penrose.data.contract.Series``. The second tuple
    element is a provenance string naming the source/version of the delivered
    data. Return ``None`` when a requested series cannot be delivered; never
    fabricate values.

    Penrose has no OHLC/bar primitive in this catalog seam. Publish bars as
    four scalar series by convention, e.g. ``asset_open``, ``asset_high``,
    ``asset_low``, and ``asset_close``.
    """

    def load_series(self, name: str) -> CatalogSeriesResult:
        """Return ``(series, provenance)`` for ``name``, or ``None`` if unavailable.

        ``series`` must be sorted, point-in-time, daily UTC, and float-like.
        ``provenance`` should identify the source and version or extraction
        date closely enough for a verdict report to audit where the values came
        from.
        """
        ...

    def available(self) -> list[str]:
        """Return stable catalog series names accepted by ``load_series``."""
        ...

    def domain_of(self, name: str) -> str | None:
        """Return a coarse domain tag for ``name`` such as ``equity`` or ``rates``."""
        ...

    def domains(self) -> list[str]:
        """Return stable, sorted domain tags represented by this catalog."""
        ...

    def describe(self, name: str) -> dict:
        """Return human-readable metadata for ``name``.

        Penrose treats this as advisory metadata for routing/spec generation;
        it is not a data payload and should not be required to reconstruct the
        series.
        """
        ...

    def describe_brief(self, name: str) -> str:
        """Return a one-line description for prompts and diagnostics."""
        ...
