"""Mean-reversion strategies."""
from __future__ import annotations

from .. import indicators as ind
from .base import SignalStrategy, register


def reversion_step(state: int, locked: bool, z: float, entry: float, exit_: float,
                   stop: float | None) -> tuple[int, bool]:
    """One step of the fade-the-extreme state machine shared with pairs trading.

    Enter against a move of ``entry`` sigmas, leave once it has come back to
    within ``exit_`` sigmas, and bail out (then wait for normality) at ``stop``.
    """
    if state == 0:
        if locked:
            locked = abs(z) >= exit_
        elif z <= -entry:
            state = 1
        elif z >= entry:
            state = -1
    elif state == 1:
        if z >= -exit_:
            state = 0
        elif stop is not None and z <= -stop:
            state, locked = 0, True
    else:
        if z <= exit_:
            state = 0
        elif stop is not None and z >= stop:
            state, locked = 0, True
    return state, locked


@register
class BollingerReversion(SignalStrategy):
    name = "bollinger"
    summary = "Fades moves beyond the Bollinger bands and exits near the mean."
    PARAMS = {"window": 20, "entry_z": 2.0, "exit_z": 0.5, "stop_z": None}

    def setup(self):
        if self.exit_z >= self.entry_z:
            raise ValueError("bollinger: exit_z must be below entry_z")
        self._state: dict[str, tuple[int, bool]] = {}

    def min_bars(self):
        return self.window

    def signal(self, sym, h):
        c = h.close(sym, self.window)
        sd = c.std()
        z = (c[-1] - c.mean()) / sd if sd > 0 else 0.0
        state, locked = self._state.get(sym, (0, False))
        state, locked = reversion_step(state, locked, z, self.entry_z, self.exit_z,
                                       self.stop_z)
        self._state[sym] = (state, locked)
        return float(state)


@register
class RSIReversion(SignalStrategy):
    name = "rsi_reversion"
    summary = "Buys short-term oversold dips in an uptrend (and the mirror image)."
    PARAMS = {"period": 2, "entry": 10.0, "exit": 60.0, "trend_window": 200}

    def setup(self):
        self._state: dict[str, int] = {}

    def min_bars(self):
        return max(self.trend_window, self.period * 10) + 1

    def signal(self, sym, h):
        r = ind.rsi(h.close(sym, self.period * 10 + 50), self.period)[-1]
        last = h.last(sym)
        if self.trend_window:
            trend = h.close(sym, self.trend_window).mean()
            up, down = last > trend, last < trend
        else:
            up = down = True
        state = self._state.get(sym, 0)
        if state == 0:
            if r < self.entry and up:
                state = 1
            elif r > 100.0 - self.entry and down:
                state = -1
        elif state == 1 and r > self.exit:
            state = 0
        elif state == -1 and r < 100.0 - self.exit:
            state = 0
        self._state[sym] = state
        return float(state)
