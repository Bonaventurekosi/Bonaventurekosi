"""Timestamped events (earnings, filings, news) and when the market can act on them.

An event has a symbol, the time it became public, a kind, an optional numeric
value (an earnings surprise, a sentiment score) and some text. Times are naive
US/Eastern, the exchange's clock; anything timezone-aware is converted on load.

The one rule that keeps event trading honest: an event belongs to the first bar
that *closes* at or after its timestamp. Earnings released at 4:30 pm therefore
belong to the next session, and nothing can be traded on before it was public.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

COLUMNS = ["symbol", "time", "kind", "value", "text", "source"]
MARKET_TZ = "America/New_York"


def to_market_time(values: Any) -> pd.DatetimeIndex:
    """Naive US/Eastern timestamps from anything datetime-like."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_convert(MARKET_TZ).tz_localize(None)
    return idx.as_unit("ns")


def session_cutoffs(timeline: Any, close: str = "16:00") -> pd.DatetimeIndex:
    """The moment each bar's information is complete.

    Daily bars are stamped at midnight and close at ``close``; intraday bars
    are treated as complete at their own timestamp, which is conservative.
    """
    tl = pd.DatetimeIndex(timeline).as_unit("ns")
    if len(tl) and (tl == tl.normalize()).all():
        hours, minutes = (int(x) for x in close.split(":"))
        return tl + pd.Timedelta(hours=hours, minutes=minutes)
    return tl


class EventStore:
    def __init__(self, rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None):
        df = pd.DataFrame(rows) if rows is not None else pd.DataFrame(columns=COLUMNS)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan if col == "value" else ""
        df = df[COLUMNS].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df["time"] = to_market_time(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        for col in ("symbol", "kind", "text", "source"):
            df[col] = df[col].fillna("").astype(str)
        df = (df.drop_duplicates(["symbol", "time", "kind", "text"])
                .sort_values(["time", "symbol", "kind"], kind="stable").reset_index(drop=True))
        self.frame = df
        self._index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sym, part in df.groupby("symbol", sort=False):
            self._index[sym] = (part["time"].to_numpy().astype("datetime64[ns]").astype("int64"),
                                part.index.to_numpy())

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame)

    @property
    def symbols(self) -> list[str]:
        return sorted(self._index)

    def counts(self) -> dict[str, int]:
        return self.frame["kind"].value_counts().to_dict()

    def filter(self, kinds: Sequence[str] | None = None, symbols: Sequence[str] | None = None,
               start: Any = None, end: Any = None) -> "EventStore":
        df = self.frame
        if kinds is not None:
            df = df[df["kind"].isin(list(kinds))]
        if symbols is not None:
            df = df[df["symbol"].isin(list(symbols))]
        if start is not None:
            df = df[df["time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["time"] <= pd.Timestamp(end)]
        return EventStore(df)

    def merge(self, other: "EventStore") -> "EventStore":
        """Both sets of events, without duplicates."""
        return EventStore(pd.concat([self.frame, other.frame], ignore_index=True))

    # -- timing --------------------------------------------------------------
    def sessions(self, timeline: Any, close: str = "16:00") -> np.ndarray:
        """Index into ``timeline`` of the first bar that closes at or after each
        event; -1 for events after the last bar."""
        cut = session_cutoffs(timeline, close).asi8
        times = self.frame["time"].to_numpy().astype("datetime64[ns]").astype("int64")
        idx = np.searchsorted(cut, times, side="left")
        idx[idx >= len(cut)] = -1
        return idx

    def between(self, symbol: str, after: Any, upto: Any,
                kinds: Sequence[str] | None = None) -> pd.DataFrame:
        """Events for ``symbol`` with ``after < time <= upto``."""
        entry = self._index.get(symbol)
        if entry is None:
            return self.frame.iloc[0:0]
        times, rows = entry
        lo = np.searchsorted(times, pd.Timestamp(after).value, side="right")
        hi = np.searchsorted(times, pd.Timestamp(upto).value, side="right")
        out = self.frame.loc[rows[lo:hi]]
        return out[out["kind"].isin(list(kinds))] if kinds is not None else out

    # -- files ---------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.frame.copy()
        out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(path, index=False)
        return path

    @classmethod
    def from_csv(cls, path: str | Path) -> "EventStore":
        """Read events. Needs ``symbol`` and ``time`` columns; ``kind``, ``value``,
        ``text`` and ``source`` are optional."""
        df = pd.read_csv(path, dtype={"symbol": str, "kind": str, "text": str, "source": str})
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = {"symbol", "time"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: events need columns {sorted(missing)}")
        return cls(df)
