"""Validated OHLCV bars for a set of symbols on a shared timeline."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise one symbol's bars: lower-case columns, sorted unique datetime
    index, no missing or non-positive prices, and highs/lows that actually
    bracket the open and close."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "close" not in df.columns:
        raise ValueError("bars need at least a 'close' column")
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = df.index.as_unit("ns")  # one resolution everywhere
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    df["volume"] = df["volume"].fillna(0.0)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df.astype(float)


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year, measured from the data (about 252 for daily equities,
    365 for daily crypto, and so on for any bar size)."""
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return (len(index) - 1) / years if years > 0 else 252.0


class Dataset:
    def __init__(self, frames: Mapping[str, pd.DataFrame], clean: bool = True):
        if not frames:
            raise ValueError("no data")
        self.frames = {s: (clean_frame(df) if clean else df) for s, df in frames.items()}
        empty = [s for s, df in self.frames.items() if df.empty]
        if empty:
            raise ValueError(f"no usable bars for {empty}")
        self.symbols = list(self.frames)
        timeline = self.frames[self.symbols[0]].index
        for sym in self.symbols[1:]:
            timeline = timeline.union(self.frames[sym].index)
        self.timeline: pd.DatetimeIndex = timeline
        self.arrays = {
            s: (df.index.asi8, *(df[c].to_numpy() for c in COLUMNS))
            for s, df in self.frames.items()
        }

    def __len__(self) -> int:
        return len(self.timeline)

    @property
    def periods_per_year(self) -> float:
        return infer_periods_per_year(self.timeline)

    def closes(self) -> pd.DataFrame:
        return pd.DataFrame({s: df["close"] for s, df in self.frames.items()},
                            index=self.timeline)

    def slice(self, start: int, stop: int | None = None) -> "Dataset":
        """Bars whose timestamp falls in ``timeline[start:stop]``."""
        window = self.timeline[start:stop]
        if not len(window):
            raise ValueError("empty slice")
        lo, hi = window[0], window[-1]
        frames = {s: df.loc[lo:hi] for s, df in self.frames.items()}
        return Dataset({s: df for s, df in frames.items() if len(df)}, clean=False)

    def benchmark(self) -> pd.Series:
        """Equal-weight, rebalanced-every-bar return index of the universe."""
        rets = self.closes().ffill().pct_change(fill_method=None)
        return (1.0 + rets.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
