"""Trend-following strategies."""
from __future__ import annotations

import numpy as np

from .. import indicators as ind
from .base import SignalStrategy, register


@register
class MovingAverageCross(SignalStrategy):
    name = "ma_cross"
    summary = "Long when the fast moving average is above the slow one, short below."
    PARAMS = {"fast": 20, "slow": 100, "kind": "ema"}

    def setup(self):
        if not 0 < self.fast < self.slow:
            raise ValueError("ma_cross: need 0 < fast < slow")
        if self.kind not in ("ema", "sma"):
            raise ValueError("ma_cross: kind must be 'ema' or 'sma'")

    def min_bars(self):
        return self.slow + 1

    def signal(self, sym, h):
        if self.kind == "sma":
            c = h.close(sym, self.slow)
            fast, slow = c[-self.fast:].mean(), c.mean()
        else:
            c = h.close(sym, self.slow * 4)
            fast, slow = ind.ema_last(c, self.fast), ind.ema_last(c, self.slow)
        return 1.0 if fast > slow else -1.0


@register
class MACDTrend(SignalStrategy):
    name = "macd"
    summary = "Follows the sign of the MACD histogram."
    PARAMS = {"fast": 12, "slow": 26, "smooth": 9}

    def setup(self):
        if not 0 < self.fast < self.slow:
            raise ValueError("macd: need 0 < fast < slow")

    def min_bars(self):
        return self.slow + self.smooth

    def signal(self, sym, h):
        c = h.close(sym, (self.slow + self.smooth) * 4)
        hist = ind.macd(c, self.fast, self.slow, self.smooth)[2][-1]
        return float(np.sign(hist))


@register
class DonchianBreakout(SignalStrategy):
    name = "donchian"
    summary = "Turtle-style channel breakout: enter on an N-bar extreme, exit on a shorter one."
    PARAMS = {"entry": 55, "exit": 20}

    def setup(self):
        if not 0 < self.exit <= self.entry:
            raise ValueError("donchian: need 0 < exit <= entry")
        self._state: dict[str, int] = {}

    def min_bars(self):
        return self.entry + 1

    def signal(self, sym, h):
        # Channels exclude the current bar, so a breakout is a close beyond them.
        highs, lows = h.high(sym, self.entry + 1)[:-1], h.low(sym, self.entry + 1)[:-1]
        c = h.last(sym)
        state = self._state.get(sym, 0)
        if c > highs.max():
            state = 1
        elif c < lows.min():
            state = -1
        elif state == 1 and c < lows[-self.exit:].min():
            state = 0
        elif state == -1 and c > highs[-self.exit:].max():
            state = 0
        self._state[sym] = state
        return float(state)


@register
class SupertrendFollow(SignalStrategy):
    name = "supertrend"
    summary = "Rides the ATR-based Supertrend line."
    PARAMS = {"period": 10, "mult": 3.0}

    def min_bars(self):
        return self.period * 3

    def signal(self, sym, h):
        n = max(self.period * 10, 100)
        direction = ind.supertrend(h.high(sym, n), h.low(sym, n), h.close(sym, n),
                                   self.period, self.mult)[1]
        return float(direction[-1])


@register
class TimeSeriesMomentum(SignalStrategy):
    name = "tsmom"
    summary = "Blends the sign of past returns over several horizons."
    PARAMS = {"lookbacks": (21, 63, 126, 252)}

    def setup(self):
        self.lookbacks = tuple(int(x) for x in self.lookbacks)
        if not self.lookbacks or min(self.lookbacks) < 1:
            raise ValueError("tsmom: lookbacks must be positive")

    def min_bars(self):
        return max(self.lookbacks) + 1

    def signal(self, sym, h):
        c = h.close(sym, max(self.lookbacks) + 1)
        return float(np.mean([np.sign(c[-1] / c[-1 - lb] - 1.0) for lb in self.lookbacks]))


@register
class KeltnerBreakout(SignalStrategy):
    name = "keltner_breakout"
    summary = "Volatility breakout: enter beyond an ATR band around the EMA, exit at the EMA."
    PARAMS = {"window": 20, "atr_window": 14, "mult": 1.5}

    def setup(self):
        self._state: dict[str, int] = {}

    def min_bars(self):
        return max(self.window, self.atr_window) * 3

    def signal(self, sym, h):
        n = max(self.window, self.atr_window) * 5
        c = h.close(sym, n)
        mid = ind.ema_last(c, self.window)
        band = self.mult * ind.atr(h.high(sym, n), h.low(sym, n), c, self.atr_window)[-1]
        last = c[-1]
        state = self._state.get(sym, 0)
        if last > mid + band:
            state = 1
        elif last < mid - band:
            state = -1
        elif (state == 1 and last < mid) or (state == -1 and last > mid):
            state = 0
        self._state[sym] = state
        return float(state)
