"""Pure cross-sectional transforms for reconstructing described portfolios.

These functions are reconstruction primitives for Penrose's referee workflow.
They assemble and transform the portfolio a claim describes; they do not
generate signals, select parameters, or make any alpha claim.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .panel import Panel


Eligibility = Callable[[pd.Timestamp], set[str]]


def _prov(name: str, *parts: str) -> str:
    return f"{name}(" + ", ".join(p for p in parts if p) + ")"


def _utc_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a pandas DatetimeIndex")
    if dates.tz is None:
        return dates.tz_localize("UTC")
    return dates.tz_convert("UTC")


def _utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _empty_series(name: str, provenance: str) -> pd.Series:
    s = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=name)
    s.attrs["provenance"] = provenance
    s.attrs["note"] = "empty reconstruction result"
    return s


def _offset_alias(rule: str):
    s = str(rule).strip().upper()
    m = re.fullmatch(r"(\d*)M", s)
    if m:
        return pd.offsets.MonthEnd(int(m.group(1) or 1))
    y = re.fullmatch(r"(\d*)Y", s)
    if y:
        # annual == 12 month-ends, NOT YearEnd (which anchors to Dec-31 and would shorten a
        # hold from a non-December rebalance, e.g. Jan-31 + '1Y' -> Dec-31 = 11 months).
        return pd.offsets.MonthEnd(12 * int(y.group(1) or 1))
    return pd.tseries.frequencies.to_offset(rule)


def _resample_alias(rule: str) -> str:
    s = str(rule).strip().upper()
    m = re.fullmatch(r"(\d*)M", s)
    if m:
        return f"{m.group(1) or 1}ME"
    y = re.fullmatch(r"(\d*)Y", s)
    if y:
        return f"{12 * int(y.group(1) or 1)}ME"     # annual rebalance == every 12 month-ends
    return rule


def winsorize(panel: Panel, lower: float, upper: float) -> Panel:
    """Clip a reconstruction panel to caller-declared bounds, not an alpha signal.

    Each value is clipped to ``[lower, upper]`` and provenance records the source
    panel and bounds. Empty panels return an empty panel.
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("winsorize requires finite bounds with lower <= upper")
    return Panel(
        name=f"{panel.name}_winsorized",
        data=panel.data.clip(lower=lower, upper=upper),
        provenance=_prov("winsorize", panel.provenance, f"lower={lower}", f"upper={upper}"),
        kind=panel.kind,
        unit=panel.unit,
        note="reconstruction primitive; clipped to caller-declared bounds",
    )


def cross_sectional_rank(panel: Panel, *, pct: bool = True) -> Panel:
    """Rank entities within each date for reconstruction, not alpha generation.

    Ties use pandas ``method="first"`` so equal values resolve deterministically
    by column order. With ``pct=True``, ranks are normalized by the number of
    non-missing entities on that date.
    """
    ranked = panel.data.rank(axis=1, method="first", pct=pct, na_option="keep")
    return Panel(
        name=f"{panel.name}_rank",
        data=ranked,
        provenance=_prov("cross_sectional_rank", panel.provenance, f"pct={pct}"),
        kind="characteristic",
        unit="rank_pct" if pct else "rank",
        note="reconstruction primitive; deterministic ties by entity column order",
    )


def cross_sectional_zscore(panel: Panel) -> Panel:
    """Per-date cross-sectional z-score for reconstruction, not signal discovery.

    Dates with fewer than two observations, or zero cross-sectional variance,
    produce NaN for that date.
    """
    counts = panel.data.count(axis=1)
    means = panel.data.mean(axis=1)
    stds = panel.data.std(axis=1)
    z = panel.data.sub(means, axis=0).div(stds.replace(0.0, np.nan), axis=0)
    z[counts < 2] = np.nan
    return Panel(
        name=f"{panel.name}_zscore",
        data=z,
        provenance=_prov("cross_sectional_zscore", panel.provenance),
        kind="characteristic",
        unit="zscore",
        note="reconstruction primitive; per-date demeaned and scaled",
    )


def asof_panel(
    records: dict[str, pd.DataFrame],
    field: str,
    dates: pd.DatetimeIndex,
    *,
    date_col: str = "filed",
    val_col: str = "val",
) -> Panel:
    """Assemble a point-in-time characteristic panel for reconstruction only.

    For each entity and target date, the value is the most recent record with
    ``date_col <= target``. Later records are ignored by construction.
    """
    target_dates = _utc_index(dates).sort_values()
    out = pd.DataFrame(index=target_dates)
    if not records:
        return Panel(field, out, _prov("asof_panel", "empty records"), kind="characteristic")

    for entity in sorted(records):
        rec = records[entity]
        if not isinstance(rec, pd.DataFrame):
            raise TypeError("asof_panel records values must be pandas DataFrames")
        if date_col not in rec.columns or val_col not in rec.columns:
            raise ValueError(f"asof_panel record for {entity!r} must include {date_col!r} and {val_col!r}")
        if len(rec) == 0:
            out[entity] = np.nan
            continue

        tmp = rec[[date_col, val_col]].copy()
        filed = pd.to_datetime(tmp[date_col], errors="coerce", utc=True)
        tmp[date_col] = filed
        tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col]).sort_values(date_col, kind="mergesort")
        tmp = tmp.drop_duplicates(subset=[date_col], keep="last")
        if len(tmp) == 0:
            out[entity] = np.nan
            continue

        series = tmp.set_index(date_col)[val_col].sort_index()
        aligned = series.reindex(series.index.union(target_dates)).sort_index().ffill().reindex(target_dates)
        out[entity] = aligned.to_numpy(dtype=float)

    return Panel(
        name=field,
        data=out,
        provenance=_prov("asof_panel", f"field={field}", "point-in-time filed<=target"),
        kind="characteristic",
        note="reconstruction primitive; later filings ignored",
    )


def _rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    if len(index) == 0:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.Series(1, index=index).resample(_resample_alias(rule)).last().dropna().index


def _window_end(start: pd.Timestamp, hold: str, max_date: pd.Timestamp) -> pd.Timestamp:
    end = start + _offset_alias(hold)
    return min(end, max_date)


def _leg_return(row: pd.Series, names: list[str], weight_row: pd.Series | None) -> float:
    vals = row.reindex(names).dropna()
    if len(vals) == 0:
        return np.nan
    if weight_row is None:
        return float(vals.mean())
    w = pd.to_numeric(weight_row.reindex(vals.index), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]
    common = vals.index.intersection(w.index)
    if len(common) == 0 or float(w.loc[common].sum()) <= 0:
        return np.nan
    ww = w.loc[common] / float(w.loc[common].sum())
    return float((vals.loc[common] * ww).sum())


def _leg_returns(win: pd.DataFrame, names: list[str], weight_row: pd.Series | None) -> pd.Series:
    block = win.reindex(columns=names)
    if weight_row is None:
        return block.mean(axis=1)

    w = pd.to_numeric(weight_row.reindex(names), errors="coerce").replace([np.inf, -np.inf], np.nan)
    w = w.where(w > 0)
    denom = block.notna().mul(w, axis=1).sum(axis=1)
    ww = block.notna().mul(w, axis=1).div(denom, axis=0)
    out = block.mul(ww).sum(axis=1)
    out[denom <= 0] = np.nan
    return out


def _bucket_membership(values: pd.Series, n_buckets: int) -> tuple[list[str], list[str]]:
    ranked = values.sort_values(kind="mergesort").rank(method="first")
    bucket = np.ceil(ranked * n_buckets / len(ranked)).astype(int).clip(1, n_buckets)
    low = list(bucket.index[bucket == 1])
    high = list(bucket.index[bucket == n_buckets])
    return low, high


def form_factor(
    returns: Panel,
    signal: Panel,
    *,
    n_buckets: int = 10,
    rebalance: str = "ME",
    hold: str = "1M",
    weights: Panel | None = None,
    high_minus_low: bool = True,
    min_names: int = 20,
    eligible: Eligibility | None = None,
) -> pd.Series:
    """Reconstruct a described cross-sectional long-short factor, never an alpha.

    At each rebalance date, entities are sorted by signal values known as of
    that date. The selected top and bottom buckets are then held over the next
    holding window. Returns inside that holding window never influence bucket
    formation. Weights, if supplied, are read as of the rebalance date.
    """
    if returns.kind != "return":
        raise ValueError("form_factor returns Panel must have kind='return'")
    if signal.kind == "return":
        raise ValueError("form_factor signal Panel must be a characteristic/price, not returns")
    if weights is not None and weights.kind == "return":
        raise ValueError("form_factor weights Panel must not have kind='return'")
    if n_buckets < 2:
        raise ValueError("form_factor requires n_buckets >= 2")
    if min_names < 1:
        raise ValueError("form_factor requires min_names >= 1")
    if returns.data.empty or signal.data.empty:
        return _empty_series("factor", _prov("form_factor", returns.provenance, signal.provenance))

    ret = returns.data.sort_index()
    sig = signal.data.sort_index()
    cols = list(ret.columns.intersection(sig.columns))
    if weights is not None:
        cols = list(pd.Index(cols).intersection(weights.data.columns))
    if not cols:
        return _empty_series("factor", _prov("form_factor", returns.provenance, signal.provenance))

    ret = ret.reindex(columns=cols)
    sig = sig.reindex(columns=cols)
    wdf = weights.data.reindex(columns=cols).sort_index() if weights is not None else None
    pieces: list[pd.Series] = []
    membership: dict[str, dict[str, list[str]]] = {}

    for rb in _rebalance_dates(ret.index, rebalance):
        sig_known = sig.loc[:rb]
        if sig_known.empty:
            continue
        sig_row = sig_known.iloc[-1].dropna()
        names = list(sig_row.index)
        if eligible is not None:
            names = [n for n in names if n in eligible(rb)]
            sig_row = sig_row.reindex(names).dropna()
        if len(sig_row) < min_names:
            continue

        low, high = _bucket_membership(sig_row, n_buckets)
        if not low or not high:
            continue

        win = ret[(ret.index > rb) & (ret.index <= _window_end(rb, hold, ret.index[-1]))]
        if win.empty:
            continue

        weight_row = None
        if wdf is not None:
            wk = wdf.loc[:rb]
            if wk.empty:
                continue
            weight_row = wk.iloc[-1]

        high_ret = _leg_returns(win, high, weight_row)
        low_ret = _leg_returns(win, low, weight_row)
        diff = high_ret - low_ret
        piece = (diff if high_minus_low else -diff).astype(float)
        pieces.append(piece)
        membership[str(rb.date())] = {"high": high, "low": low}

    provenance = _prov(
        "form_factor",
        returns.provenance,
        signal.provenance,
        getattr(weights, "provenance", "") if weights is not None else "equal-weight",
        f"n_buckets={n_buckets}",
        f"rebalance={rebalance}",
        f"hold={hold}",
    )
    if not pieces:
        return _empty_series("factor", provenance)

    # Blend overlapping legs: when hold > rebalance, several active cohorts cover the same
    # date, so AVERAGE them (the standard overlapping-portfolio reconstruction). When
    # hold <= rebalance each date has exactly one leg, so the mean is a no-op (identical output).
    out = pd.concat(pieces).groupby(level=0).mean().sort_index()
    out.name = "factor"
    out.attrs["provenance"] = provenance
    out.attrs["membership"] = membership
    out.attrs["note"] = "reconstruction primitive; buckets formed from signals known at rebalance"
    return out


def liquidity_screen(dvol: Panel, *, top_n: int, asof: pd.Timestamp, lookback: str = "3M") -> set[str]:
    """Select liquid entities using trailing dollar-volume for reconstruction.

    The median dollar-volume window ends at ``asof`` and never reads future
    rows. The returned set is an eligibility aid, not a recommended universe.
    """
    if top_n < 1:
        raise ValueError("liquidity_screen requires top_n >= 1")
    if dvol.data.empty:
        return set()
    asof_utc = _utc_ts(asof)
    start = asof_utc - _offset_alias(lookback)
    win = dvol.data[(dvol.data.index > start) & (dvol.data.index <= asof_utc)]
    if win.empty:
        return set()
    med = win.median(axis=0, skipna=True).dropna()
    if med.empty:
        return set()
    ranked = med.sort_values(ascending=False, kind="mergesort")
    return set(ranked.head(top_n).index.astype(str))


def momentum_signal(returns: Panel, lookback_m: int = 12, skip_m: int = 1) -> Panel:
    """Build a prior-window momentum characteristic for reconstruction only.

    This thin helper reflects a caller-described prior-return characteristic; it
    is not a recommended signal and carries no alpha claim.
    """
    if returns.kind != "return":
        raise ValueError("momentum_signal requires a returns Panel with kind='return'")
    if lookback_m < 1 or skip_m < 0:
        raise ValueError("momentum_signal requires lookback_m >= 1 and skip_m >= 0")
    monthly = (1.0 + returns.data).resample("ME").prod() - 1.0
    vals = (1.0 + monthly.shift(skip_m + 1)).rolling(lookback_m, min_periods=lookback_m).apply(np.prod, raw=True) - 1.0
    aligned = vals.reindex(returns.data.index.union(vals.index)).sort_index().ffill().reindex(returns.data.index)
    return Panel(
        name=f"{returns.name}_momentum",
        data=aligned,
        provenance=_prov("momentum_signal", returns.provenance, f"lookback_m={lookback_m}", f"skip_m={skip_m}"),
        kind="characteristic",
        unit="return",
        note="reconstruction-only helper; not a recommended signal",
    )
