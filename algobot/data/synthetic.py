"""Seeded synthetic markets.

These exist so strategies can be developed and tested without a data vendor,
and so tests can check behaviour on markets whose structure is known (a
trending series, a mean-reverting one, a cointegrated pair). They say nothing
about how a strategy will do on real prices.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def gbm_path(rng, n: int, mu: float = 0.0003, sigma: float = 0.012, s0: float = 100.0):
    """Geometric random walk with drift: no exploitable structure."""
    return s0 * np.exp(np.cumsum(rng.normal(mu - 0.5 * sigma ** 2, sigma, n)))


def trend_path(rng, n: int, mu: float = 0.0009, sigma: float = 0.011,
               persistence: float = 0.992, s0: float = 100.0):
    """Drift that flips sign occasionally: long, persistent up and down legs."""
    flips = rng.random(n) > persistence
    direction = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    return s0 * np.exp(np.cumsum(direction * mu + rng.normal(0.0, sigma, n)))


def range_path(rng, n: int, kappa: float = 0.06, sigma: float = 0.012, s0: float = 100.0):
    """Ornstein-Uhlenbeck in log price: always pulled back toward ``s0``."""
    noise = rng.normal(0.0, sigma, n).tolist()
    x, out = 0.0, []
    for eps in noise:
        x += -kappa * x + eps
        out.append(x)
    return s0 * np.exp(np.array(out))


def regime_path(rng, n: int, s0: float = 100.0, persistence: float = 0.99,
                mu: float = 0.0010, sigma: float = 0.011, kappa: float = 0.08):
    """Alternates between up-trends, down-trends and sideways ranges."""
    noise = rng.normal(0.0, 1.0, n).tolist()
    switch = (rng.random(n) > persistence).tolist()
    choice = rng.integers(0, 3, n).tolist()
    regime, x, centre, out = 0, 0.0, 0.0, []
    for i in range(n):
        if switch[i]:
            regime, centre = choice[i], x
        if regime == 0:
            x += mu + sigma * noise[i]
        elif regime == 1:
            x += -mu + 1.3 * sigma * noise[i]
        else:
            x += -kappa * (x - centre) + sigma * noise[i]
        out.append(x)
    return s0 * np.exp(np.array(out))


def cointegrated_path(rng, base: np.ndarray, beta: float = 0.8, kappa: float = 0.08,
                      sigma: float = 0.01, s0: float = 50.0):
    """A series tied to ``base``: log price = beta * log base + a mean-reverting spread."""
    spread = np.log(range_path(rng, len(base), kappa=kappa, sigma=sigma, s0=1.0))
    return s0 * np.exp(beta * (np.log(base) - np.log(base[0])) + spread)


def ohlcv_from_close(rng, close: np.ndarray, index: pd.DatetimeIndex,
                     base_volume: float = 1e6) -> pd.DataFrame:
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1.0 + rng.normal(0.0, 0.002, n))
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = base_volume * np.exp(rng.normal(0.0, 0.3, n)) * (1.0 + 10.0 * np.abs(close / prev - 1.0))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume}, index=index)


_PROCESSES = {"gbm": gbm_path, "trend": trend_path, "range": range_path,
              "regime": regime_path}
_MIX = ("regime", "trend", "range", "gbm")


def synthetic_universe(symbols: Sequence[str], bars: int = 2520, seed: int = 7,
                       kind: str = "mixed", start: str = "2014-01-02", freq: str = "B",
                       pairs: Sequence[Sequence[str]] = ()) -> dict[str, pd.DataFrame]:
    """Bars for ``symbols``. ``kind`` is one process for all, or "mixed" to cycle
    through regime / trend / range / random-walk. Each ``[a, b]`` in ``pairs``
    makes ``b`` cointegrated with ``a``."""
    if kind != "mixed" and kind not in _PROCESSES:
        raise ValueError(f"unknown synthetic kind {kind!r}")
    index = pd.date_range(start, periods=bars, freq=freq)
    closes: dict[str, np.ndarray] = {}
    rngs = {s: np.random.default_rng([seed, i]) for i, s in enumerate(symbols)}
    followers = {b: a for a, b in pairs}
    for i, sym in enumerate(symbols):
        if sym in followers:
            continue
        process = _PROCESSES[_MIX[i % len(_MIX)] if kind == "mixed" else kind]
        closes[sym] = process(rngs[sym], bars)
    for b, a in followers.items():
        if a not in closes or b not in rngs:
            raise ValueError(f"pair [{a}, {b}] must name two symbols in the universe")
        closes[b] = cointegrated_path(rngs[b], closes[a])
    return {s: ohlcv_from_close(rngs[s], closes[s], index) for s in symbols}
