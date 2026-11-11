"""Inventory-style strategies: grid trading and cost averaging."""
from __future__ import annotations

import math

import pandas as pd

from .base import SignalStrategy, Strategy, register


@register
class GridTrading(SignalStrategy):
    name = "grid"
    summary = "Buys each step down and sells each step up around an anchor price."
    # The anchor is, in order of precedence: ``anchor_price`` if given, the
    # ``anchor_window``-bar average if non-zero, else the first price seen.
    PARAMS = {"step_pct": 0.02, "levels": 5, "mode": "long", "anchor_window": 0,
              "anchor_price": None}

    def setup(self):
        if self.mode not in ("long", "neutral"):
            raise ValueError("grid: mode must be 'long' or 'neutral'")
        if self.step_pct <= 0 or self.levels < 1:
            raise ValueError("grid: step_pct and levels must be positive")
        self._anchor: dict[str, float] = {}
        self._line: dict[str, int] = {}

    def min_bars(self):
        return max(1, self.anchor_window)

    def signal(self, sym, h):
        price = h.last(sym)
        if self.anchor_price is not None:
            anchor = float(self.anchor_price)
        elif self.anchor_window:
            anchor = h.close(sym, self.anchor_window).mean()
        else:
            anchor = self._anchor.setdefault(sym, price)
        x = (price / anchor - 1.0) / self.step_pct  # position in grid steps
        line = self._line.get(sym)
        if line is None:
            line = math.trunc(x)
        elif x <= line - 1:   # fell through one or more grid lines: buy
            line = math.ceil(x)
        elif x >= line + 1:   # rose through one or more grid lines: sell
            line = math.floor(x)
        self._line[sym] = line
        k = max(-self.levels, min(self.levels, line))
        if self.mode == "long":
            # Half invested at the anchor, full at the bottom, flat at the top.
            return (self.levels - k) / (2.0 * self.levels)
        return -k / self.levels


@register
class CostAveraging(Strategy):
    name = "dca"
    summary = "Builds a position in fixed increments on a schedule, buying more on dips."
    # ``start`` (a date) delays the first purchase; bars before it are ignored.
    PARAMS = {"every": 21, "increment": 0.05, "max_weight": 1.0,
              "dip_window": 50, "dip_threshold": 0.05, "dip_boost": 2.0, "start": None}

    def setup(self):
        self._w: dict[str, float] = {}
        self._calls = 0
        self._start = pd.Timestamp(str(self.start)) if self.start is not None else None

    def target_weights(self, h):
        syms = [s for s in self.universe(h) if h.size(s)]
        if self._start is not None and h.now < self._start:
            syms = []
        if syms and self._calls % self.every == 0:
            for sym in syms:
                boost = 1.0
                if self.dip_window and h.size(sym) >= self.dip_window:
                    avg = h.close(sym, self.dip_window).mean()
                    if h.last(sym) < avg * (1.0 - self.dip_threshold):
                        boost = self.dip_boost
                room = self.max_weight - sum(self._w.values())
                add = min(self.increment * boost / len(syms), max(0.0, room))
                self._w[sym] = self._w.get(sym, 0.0) + add
        if syms:
            self._calls += 1
        out = {s: 0.0 for s in self.universe(h)}
        out.update(self._w)
        return out
