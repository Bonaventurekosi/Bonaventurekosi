"""Meta-strategy: run several strategies side by side and blend their books."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .. import indicators as ind
from .base import Strategy, create, register


@register
class Ensemble(Strategy):
    """Blend child strategies into one book.

    Each child is a normal strategy spec plus two optional keys:

    ``weight``  share of capital before any adaptation (default 1, normalised).
    ``regime``  "trend", "range" or "any". A gated child only trades a symbol
                while that symbol's ADX says the market is in that regime. Use
                it on per-symbol signal strategies; gating one leg of a spread
                or ranking strategy would unbalance it.

    ``method`` decides how capital moves between children over time:

    ``static``       fixed weights.
    ``inverse_vol``  more capital to children whose returns have been calmer.
    ``adaptive``     more capital to children that have been working lately
                     (softmax of the t-statistic of their recent returns).
    """

    name = "ensemble"
    summary = "Blends several strategies, optionally re-weighting by performance or market regime."
    PARAMS = {"children": (), "method": "static", "lookback": 63, "eta": 1.0,
              "adx_period": 14, "adx_range": 20.0, "adx_trend": 25.0}

    def setup(self):
        if self.method not in ("static", "inverse_vol", "adaptive"):
            raise ValueError("ensemble: method must be static, inverse_vol or adaptive")
        if not self.children:
            raise ValueError("ensemble: needs at least one child strategy")
        self._kids = [create(spec, self.symbols) for spec in self.children]
        base = np.array([float(spec.get("weight", 1.0)) for spec in self.children])
        if base.min() < 0 or base.sum() <= 0:
            raise ValueError("ensemble: child weights must be non-negative")
        self._base = base / base.sum()
        self._gates = [spec.get("regime", "any") for spec in self.children]
        if any(g not in ("any", "trend", "range") for g in self._gates):
            raise ValueError("ensemble: regime must be any, trend or range")
        self._prev: list[dict[str, float] | None] = [None] * len(self._kids)
        self._rets = [deque(maxlen=self.lookback) for _ in self._kids]
        self._stamps: dict[str, object] = {}
        self._regime: dict[str, str] = {}
        self.allocation = self._base.copy()  # exposed for inspection

    # -- helpers -----------------------------------------------------------
    def _bar_returns(self, h) -> dict[str, float]:
        """Return of each symbol over the bar that just closed."""
        rets = {}
        for sym in h.symbols:
            stamp = h.last_ts(sym)
            if h.size(sym) >= 2 and stamp != self._stamps.get(sym):
                c = h.close(sym, 2)
                rets[sym] = c[1] / c[0] - 1.0
            self._stamps[sym] = stamp
        return rets

    def _symbol_regime(self, sym, h) -> str | None:
        n = self.adx_period * 8
        if h.size(sym) < self.adx_period * 3:
            return None
        value = ind.adx(h.high(sym, n), h.low(sym, n), h.close(sym, n), self.adx_period)[0][-1]
        if math.isnan(value):
            return None
        prev = self._regime.get(sym)
        if value >= self.adx_trend:
            prev = "trend"
        elif value <= self.adx_range:
            prev = "range"
        elif prev is None:
            prev = "trend" if value >= (self.adx_trend + self.adx_range) / 2 else "range"
        self._regime[sym] = prev
        return prev

    def _allocate(self) -> np.ndarray:
        if self.method == "static":
            return self._base
        min_obs = min(20, self.lookback)
        if any(len(r) < min_obs for r in self._rets):
            return self._base
        stats = [(np.mean(r), np.std(r), len(r)) for r in self._rets]
        if self.method == "inverse_vol":
            vols = np.array([sd for _, sd, _ in stats])
            if np.any(vols <= 0):
                return self._base
            raw = self._base / vols
        else:
            tstat = np.array([m / sd * math.sqrt(n) if sd > 0 else 0.0
                              for m, sd, n in stats])
            raw = self._base * np.exp(self.eta * (tstat - tstat.max()))
        return raw / raw.sum()

    # -- main ----------------------------------------------------------------
    def target_weights(self, h):
        rets = self._bar_returns(h)
        for i, prev in enumerate(self._prev):
            if prev is not None:
                self._rets[i].append(sum(w * rets.get(s, 0.0) for s, w in prev.items()))

        books = [kid.target_weights(h) for kid in self._kids]
        self._prev = books
        self.allocation = alloc = self._allocate()

        gated = any(g != "any" for g in self._gates)
        regimes = {s: self._symbol_regime(s, h) for s in self.universe(h)} if gated else {}

        out: dict[str, float] = {}
        for book, share, gate in zip(books, alloc, self._gates):
            for sym, w in book.items():
                if gate != "any" and regimes.get(sym) != gate:
                    w = 0.0
                out[sym] = out.get(sym, 0.0) + share * w
        return out
