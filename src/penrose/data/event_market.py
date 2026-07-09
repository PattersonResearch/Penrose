"""Event-market data contract for bracket reconstruction primitives.

The EventMarketPanel type is a provenance-carrying data boundary for settled
binary bracket markets. It supports faithful reconstruction of prediction
market claims; it is not a signal generator and makes no edge claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


EVENT_MARKET_COLUMNS = [
    "event_id",
    "decision_time",
    "close_time",
    "strike_low",
    "strike_high",
    "entry_price",
    "outcome",
    "underlying",
]

WEATHER_TAIL_COLUMNS = [
    "ticker",
    "city",
    "close_date",
    "p_close",
    "outcome",
    "volume",
    "open_interest",
    "is_tail",
]


@dataclass
class EventMarketPanel:
    """A settled event/bracket-market reconstruction panel.

    One row represents one binary bracket for an event. ``decision_time`` is
    the point at which the strategy may use ``underlying`` and ``entry_price``;
    adapters are responsible for ensuring those fields are causal.
    """

    name: str
    data: pd.DataFrame
    provenance: str
    kind: str = "event_market"
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("EventMarketPanel.data must be a pandas DataFrame")
        if self.kind != "event_market":
            raise ValueError("EventMarketPanel.kind must be 'event_market'")

        if len(self.data) == 0 and len(self.data.columns) == 0:
            df = pd.DataFrame(columns=EVENT_MARKET_COLUMNS)
        else:
            df = coerce_event_market_frame(self.data)

        missing = [c for c in EVENT_MARKET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"EventMarketPanel.data missing required columns: {', '.join(missing)}")
        extra = [c for c in df.columns if c not in EVENT_MARKET_COLUMNS]
        if extra:
            raise ValueError(f"EventMarketPanel.data has unsupported columns: {', '.join(extra)}")

        df = df.loc[:, EVENT_MARKET_COLUMNS].copy()
        try:
            df["decision_time"] = _utc_datetime_series(df["decision_time"], "decision_time")
            df["close_time"] = _utc_datetime_series(df["close_time"], "close_time")
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"EventMarketPanel.data {exc}") from None

        try:
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="raise").astype(float)
            df["strike_low"] = pd.to_numeric(df["strike_low"], errors="raise").astype(float)
            df["strike_high"] = pd.to_numeric(df["strike_high"], errors="raise").astype(float)
            df["outcome"] = pd.to_numeric(df["outcome"], errors="raise")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"EventMarketPanel.data numeric fields are invalid: {exc}") from None

        if df["event_id"].isna().any():
            raise ValueError("EventMarketPanel.data event_id must not contain nulls")
        if df["underlying"].isna().any():
            raise ValueError("EventMarketPanel.data underlying must not contain nulls")
        if df["decision_time"].isna().any() or df["close_time"].isna().any():
            raise ValueError("EventMarketPanel.data decision_time and close_time must not contain nulls")
        if df["entry_price"].isna().any():
            raise ValueError("EventMarketPanel.data entry_price must not contain nulls")
        if not df["entry_price"].between(0.0, 1.0, inclusive="both").all():
            raise ValueError("EventMarketPanel.data entry_price must be in [0, 1]")
        if not df["outcome"].isin([0, 1]).all():
            raise ValueError("EventMarketPanel.data outcome must be in {0, 1}")
        if (df["close_time"] < df["decision_time"]).any():
            raise ValueError("EventMarketPanel.data close_time must be >= decision_time")

        df["event_id"] = df["event_id"].astype(str)
        df["outcome"] = df["outcome"].astype(int)
        df = df.sort_values(
            ["decision_time", "event_id", "strike_low", "strike_high"],
            kind="mergesort",
        ).reset_index(drop=True)
        object.__setattr__(self, "data", df)

    @property
    def coverage(self) -> tuple[Optional[str], Optional[str], int, int]:
        """Return ``(first_decision, last_decision, n_events, n_brackets)``."""
        if self.data is None or len(self.data) == 0:
            return None, None, 0, 0
        return (
            self.data["decision_time"].iloc[0].isoformat(),
            self.data["decision_time"].iloc[-1].isoformat(),
            int(self.data["event_id"].nunique()),
            len(self.data),
        )


def _utc_datetime_series(values: pd.Series, name: str) -> pd.Series:
    out = []
    for value in values:
        try:
            ts = pd.Timestamp(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{name} must be parseable datetimes: {exc}") from None
        if pd.isna(ts):
            out.append(pd.NaT)
        elif ts.tzinfo is None:
            out.append(ts.tz_localize("UTC"))
        else:
            out.append(ts.tz_convert("UTC"))
    return pd.Series(out, index=values.index, name=name, dtype="datetime64[ns, UTC]")


def coerce_event_market_frame(data: pd.DataFrame, *, preserve_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return a generic event-market frame, accepting the Kalshi weather raw shape.

    The weather tail primitive consumes ticker/city/liquidity/tail flags from the
    row ``underlying`` dict. Keeping those fields there avoids creating a second
    event-market contract while still letting declared raw weather tables load.
    """
    df = data.copy()
    if all(c in df.columns for c in EVENT_MARKET_COLUMNS):
        if preserve_extra:
            keep = EVENT_MARKET_COLUMNS + [c for c in preserve_extra if c in df.columns]
            return df.loc[:, keep].copy()
        return df
    if not all(c in df.columns for c in WEATHER_TAIL_COLUMNS):
        return df

    out = pd.DataFrame(index=df.index)
    out["event_id"] = df["ticker"].astype(str)
    out["decision_time"] = df["close_date"]
    out["close_time"] = df["close_date"]
    out["strike_low"] = df["strike_low"] if "strike_low" in df.columns else 0.0
    out["strike_high"] = df["strike_high"] if "strike_high" in df.columns else 1.0
    out["entry_price"] = df["p_close"]
    out["outcome"] = df["outcome"]

    def _underlying(row) -> dict:
        return {
            "ticker": row["ticker"],
            "city": row["city"],
            "close_date": row["close_date"],
            "p_close": row["p_close"],
            "volume": row["volume"],
            "open_interest": row["open_interest"],
            "is_tail": row["is_tail"],
        }

    out["underlying"] = df.apply(_underlying, axis=1)
    for col in preserve_extra:
        if col in df.columns:
            out[col] = df[col]
    return out
