"""A learning strategy: online logistic regression on price features.

The model only ever trains on labels that are already known. At each bar it
first learns from the sample taken ``horizon`` bars ago (whose outcome has now
been observed) and only then predicts from the current features, so a backtest
of it is a genuine walk-forward test.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .. import indicators as ind
from .base import SignalStrategy, register

_N_FEATURES = 9
_WINDOW = 64


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def features(close: np.ndarray) -> np.ndarray:
    """Volatility-normalised momentum, stretch and regime features, plus a bias."""
    lr = np.diff(np.log(close))
    vol = lr[-20:].std() + 1e-12
    sd20 = close[-20:].std()
    move20 = close[-1] - close[-21]
    path20 = np.abs(np.diff(close[-21:])).sum()
    f = np.array([
        lr[-1] / vol,
        lr[-5:].sum() / (vol * math.sqrt(5)),
        lr[-10:].sum() / (vol * math.sqrt(10)),
        lr[-20:].sum() / (vol * math.sqrt(20)),
        lr[-60:].sum() / (vol * math.sqrt(60)),
        (close[-1] - close[-20:].mean()) / sd20 if sd20 > 0 else 0.0,
        ind.rsi(close[-30:], 14)[-1] / 50.0 - 1.0,
        lr[-5:].std() / vol - 1.0,
        move20 / path20 if path20 > 0 else 0.0,
        1.0,
    ])
    return np.clip(f, -5.0, 5.0)


@register
class OnlineLogistic(SignalStrategy):
    name = "ml_logistic"
    summary = "Online logistic regression predicting the direction of the next few bars."
    PARAMS = {"horizon": 5, "learning_rate": 0.02, "l2": 1e-4, "min_train": 250,
              "threshold": 0.03, "gain": 4.0, "pooled": True}

    def setup(self):
        self._w: dict[str, np.ndarray] = {}
        self._seen: dict[str, int] = {}
        self._pending: dict[str, deque] = {}

    def min_bars(self):
        return _WINDOW

    def _key(self, sym):
        return "*" if self.pooled else sym

    def signal(self, sym, h):
        key = self._key(sym)
        w = self._w.setdefault(key, np.zeros(_N_FEATURES + 1))
        x = features(h.close(sym, _WINDOW))
        price = h.last(sym)

        queue = self._pending.setdefault(sym, deque())
        queue.append((x, price))
        if len(queue) > self.horizon:
            x_old, price_old = queue.popleft()
            label = 1.0 if price > price_old else 0.0
            p_old = _sigmoid(float(w @ x_old))
            w -= self.learning_rate * ((p_old - label) * x_old + self.l2 * w)
            self._seen[key] = self._seen.get(key, 0) + 1

        if self._seen.get(key, 0) < self.min_train:
            return 0.0
        edge = _sigmoid(float(w @ x)) - 0.5
        if abs(edge) < self.threshold:
            return 0.0
        return max(-1.0, min(1.0, 2.0 * edge * self.gain))
