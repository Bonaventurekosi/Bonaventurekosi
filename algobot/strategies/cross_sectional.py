"""Strategies that allocate across a universe rather than symbol by symbol."""
from __future__ import annotations

import numpy as np

from .base import Strategy, register


class Rebalanced(Strategy):
    """Recomputes weights every ``rebalance_every`` bars and holds in between."""

    PARAMS = {"rebalance_every": 21}

    def setup(self):
        self._held: dict[str, float] = {}
        self._since: int | None = None

    def target_weights(self, h):
        if self._since is None or self._since >= self.rebalance_every:
            fresh = self.allocate(h)
            if fresh is not None:
                self._held, self._since = fresh, 0
        if self._since is not None:
            self._since += 1
        out = {s: 0.0 for s in self.universe(h)}
        out.update(self._held)
        return out

    def allocate(self, h) -> dict[str, float] | None:
        """Fresh weights, or None when there is not enough data yet."""
        raise NotImplementedError


def _trailing_return(h, sym, lookback, skip=0):
    c = h.close(sym, lookback + skip + 1)
    return c[-1 - skip] / c[0] - 1.0


@register
class CrossSectionalMomentum(Rebalanced):
    name = "xs_momentum"
    summary = "Ranks the universe by past return; long the winners, optionally short the losers."
    PARAMS = {"lookback": 126, "skip": 5, "top_k": None, "long_short": False}

    def allocate(self, h):
        need = self.lookback + self.skip + 1
        ranked = sorted((s for s in self.universe(h) if h.size(s) >= need),
                        key=lambda s: _trailing_return(h, s, self.lookback, self.skip),
                        reverse=True)
        k = self.top_k or max(1, len(ranked) // 3)
        if len(ranked) < (2 * k if self.long_short else k) or not ranked:
            return None
        if not self.long_short:
            return {s: 1.0 / k for s in ranked[:k]}
        out = {s: 0.5 / k for s in ranked[:k]}
        out.update({s: -0.5 / k for s in ranked[-k:]})
        return out


@register
class DualMomentum(Rebalanced):
    name = "dual_momentum"
    summary = "Holds the strongest assets only while their own return is positive, else the safe asset or cash."
    PARAMS = {"lookback": 126, "top_k": 1, "safe_asset": None}

    def allocate(self, h):
        need = self.lookback + 1
        risky = [s for s in self.universe(h)
                 if s != self.safe_asset and h.size(s) >= need]
        if len(risky) < self.top_k:
            return None
        scored = sorted(((_trailing_return(h, s, self.lookback), s) for s in risky),
                        reverse=True)
        out: dict[str, float] = {}
        slot = 1.0 / self.top_k
        for ret, sym in scored[:self.top_k]:
            if ret > 0:
                out[sym] = slot
            elif self.safe_asset and h.size(self.safe_asset):
                out[self.safe_asset] = out.get(self.safe_asset, 0.0) + slot
        return out


@register
class InverseVolatility(Rebalanced):
    name = "inverse_vol"
    summary = "Naive risk parity: weights proportional to 1 / volatility."
    PARAMS = {"window": 60}

    def allocate(self, h):
        inv = {}
        for sym in self.universe(h):
            if h.size(sym) < self.window + 1:
                continue
            vol = np.diff(np.log(h.close(sym, self.window + 1))).std()
            if vol > 0:
                inv[sym] = 1.0 / vol
        if not inv:
            return None
        total = sum(inv.values())
        return {s: v / total for s, v in inv.items()}


@register
class BuyAndHold(Strategy):
    name = "buy_and_hold"
    summary = "Constant weights (equal by default). The baseline everything else must beat."
    PARAMS = {"weights": None}

    def target_weights(self, h):
        live = [s for s in self.universe(h) if h.size(s)]
        if self.weights:
            return {s: float(self.weights.get(s, 0.0)) for s in live}
        return {s: 1.0 / len(live) for s in live}
