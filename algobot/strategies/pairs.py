"""Statistical arbitrage on pairs of related symbols."""
from __future__ import annotations

import numpy as np

from .base import Strategy, register
from .reversion import reversion_step


@register
class PairsTrading(Strategy):
    name = "pairs"
    summary = "Trades the spread between two co-moving symbols back toward its mean."
    PARAMS = {"pairs": None, "window": 60, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0}

    def setup(self):
        if self.exit_z >= self.entry_z:
            raise ValueError("pairs: exit_z must be below entry_z")
        # (a, b) -> (state, locked, hedge ratio frozen at entry)
        self._state: dict[tuple[str, str], tuple[int, bool, float]] = {}

    def _pairs(self, h):
        if self.pairs:
            return [tuple(p) for p in self.pairs]
        syms = self.universe(h)
        return [(syms[i], syms[i + 1]) for i in range(0, len(syms) - 1, 2)]

    def target_weights(self, h):
        pairs = self._pairs(h)
        out = {s: 0.0 for s in self.universe(h)}
        if not pairs:
            return out
        share = 1.0 / len(pairs)
        n = self.window
        for a, b in pairs:
            if a not in h or b not in h or min(h.size(a), h.size(b)) < n:
                continue
            y, x = np.log(h.close(a, n)), np.log(h.close(b, n))
            xm, ym = x.mean(), y.mean()
            var = ((x - xm) ** 2).mean()
            if var <= 0:
                continue
            beta = ((x - xm) * (y - ym)).mean() / var
            resid = y - (ym + beta * (x - xm))
            sd = resid.std()
            z = resid[-1] / sd if sd > 0 else 0.0

            state, locked, held_beta = self._state.get((a, b), (0, False, 1.0))
            was = state
            state, locked = reversion_step(state, locked, z, self.entry_z,
                                           self.exit_z, self.stop_z)
            if state != 0 and was == 0:
                if beta <= 0:  # not a hedgeable relationship right now
                    state = 0
                else:
                    held_beta = beta
            self._state[(a, b)] = (state, locked, held_beta)
            if state:
                # Long the spread = long a, short beta units of b; gross = share.
                gross = 1.0 + abs(held_beta)
                out[a] = out.get(a, 0.0) + state * share / gross
                out[b] = out.get(b, 0.0) - state * held_beta * share / gross
        return out
