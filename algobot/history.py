"""Append-only bar storage.

``BarHistory`` is the only view of the market a strategy ever gets. It holds
bars up to "now" and nothing later, which is what makes the same strategy code
safe to run in a backtest and against a live feed.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME = range(5)


class BarHistory:
    def __init__(self, symbols: Iterable[str], capacity: int = 512):
        self.symbols: list[str] = list(symbols)
        self._n: dict[str, int] = {s: 0 for s in self.symbols}
        self._data: dict[str, np.ndarray] = {
            s: np.empty((5, capacity)) for s in self.symbols
        }
        self._last_ts: dict[str, Any] = {s: None for s in self.symbols}
        self.now: Any = None

    # -- writing -----------------------------------------------------------
    def append(self, symbol: str, ts: Any, o: float, h: float, l: float,
               c: float, v: float = 0.0) -> None:
        n = self._n[symbol]
        arr = self._data[symbol]
        if n == arr.shape[1]:
            grown = np.empty((5, arr.shape[1] * 2))
            grown[:, :n] = arr
            self._data[symbol] = arr = grown
        arr[_OPEN, n] = o
        arr[_HIGH, n] = h
        arr[_LOW, n] = l
        arr[_CLOSE, n] = c
        arr[_VOLUME, n] = v
        self._n[symbol] = n + 1
        self._last_ts[symbol] = ts
        self.now = ts

    # -- reading -----------------------------------------------------------
    def __contains__(self, symbol: str) -> bool:
        return symbol in self._n

    def size(self, symbol: str) -> int:
        return self._n.get(symbol, 0)

    def last_ts(self, symbol: str) -> Any:
        return self._last_ts.get(symbol)

    def _get(self, symbol: str, row: int, n: int | None) -> np.ndarray:
        end = self._n[symbol]
        start = 0 if n is None else max(0, end - n)
        view = self._data[symbol][row, start:end]
        view.flags.writeable = False
        return view

    def open(self, symbol: str, n: int | None = None) -> np.ndarray:
        return self._get(symbol, _OPEN, n)

    def high(self, symbol: str, n: int | None = None) -> np.ndarray:
        return self._get(symbol, _HIGH, n)

    def low(self, symbol: str, n: int | None = None) -> np.ndarray:
        return self._get(symbol, _LOW, n)

    def close(self, symbol: str, n: int | None = None) -> np.ndarray:
        return self._get(symbol, _CLOSE, n)

    def volume(self, symbol: str, n: int | None = None) -> np.ndarray:
        return self._get(symbol, _VOLUME, n)

    def last(self, symbol: str) -> float:
        """Most recent close, or NaN if the symbol has no bars yet."""
        n = self._n.get(symbol, 0)
        return float(self._data[symbol][_CLOSE, n - 1]) if n else float("nan")

    def prices(self) -> dict[str, float]:
        return {s: self.last(s) for s in self.symbols if self._n[s]}

    @classmethod
    def from_frames(cls, frames: Mapping[str, Any]) -> "BarHistory":
        """Bulk-load from ``{symbol: DataFrame[open, high, low, close, volume]}``."""
        hist = cls(frames.keys())
        for sym, df in frames.items():
            for ts, o, h, l, c, v in zip(
                df.index, df["open"].to_numpy(), df["high"].to_numpy(),
                df["low"].to_numpy(), df["close"].to_numpy(),
                df["volume"].to_numpy(),
            ):
                hist.append(sym, ts, o, h, l, c, v)
        return hist
