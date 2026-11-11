"""Position sizing overlay: volatility targeting and a leverage cap."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .history import BarHistory


@dataclass
class SizingConfig:
    # Annualised portfolio volatility to aim for; None leaves weights as-is.
    target_vol: float | None = None
    vol_window: int = 40
    # Hard cap on the sum of absolute weights after scaling.
    max_leverage: float = 1.0
    # Bars per year; filled in from the data when left as None.
    periods_per_year: float | None = None


class Sizer:
    def __init__(self, config: SizingConfig | None = None):
        self.config = config or SizingConfig()

    def apply(self, weights: Mapping[str, float], history: BarHistory) -> dict[str, float]:
        cfg = self.config
        out = {s: float(w) for s, w in weights.items() if math.isfinite(w)}
        if cfg.target_vol is not None:
            scale = self._vol_scale(out, history)
            if scale is not None:
                out = {s: w * scale for s, w in out.items()}
        gross = sum(abs(w) for w in out.values())
        if gross > cfg.max_leverage > 0:
            k = cfg.max_leverage / gross
            out = {s: w * k for s, w in out.items()}
        return out

    def _vol_scale(self, weights: Mapping[str, float], history: BarHistory) -> float | None:
        """Forecast the book's volatility from recent covariance and scale to target."""
        cfg = self.config
        active = [s for s, w in weights.items() if w != 0.0]
        if not active:
            return None
        n = cfg.vol_window
        if any(history.size(s) < n + 1 for s in active):
            return None
        rets = np.vstack([np.diff(np.log(history.close(s, n + 1))) for s in active])
        w = np.array([weights[s] for s in active])
        cov = np.atleast_2d(np.cov(rets))
        var = float(w @ cov @ w)
        if not math.isfinite(var) or var <= 0:
            return None
        vol = math.sqrt(var * (cfg.periods_per_year or 252.0))
        return cfg.target_vol / vol
