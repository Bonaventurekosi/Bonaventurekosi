"""Risk manager: the last word on what the bot is allowed to hold.

It runs once per bar, after the strategy and sizing, and can only reduce risk:
clamp weights, close positions that hit a stop, or flatten everything when a
circuit breaker trips. Stops are evaluated on bar closes and executed on the
next bar, the same way in a backtest as against a live feed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .types import Position


@dataclass
class RiskConfig:
    # Per-symbol absolute weight cap and total (gross) cap.
    max_position_weight: float = 1.0
    max_gross_leverage: float = 1.0
    long_only: bool = False
    # Per-position exits, as fractions of the average entry price.
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    # Exit when price gives back this fraction from its best level since entry.
    trailing_stop_pct: float | None = None
    # Bars a symbol stays untradable after one of the exits above fires.
    cooldown_bars: int = 5
    # Flatten everything once equity is this far below its peak...
    max_drawdown_pct: float | None = None
    # ...and stay out for this many bars (None: until restarted by hand).
    drawdown_halt_bars: int | None = None
    # Flatten for the rest of the day after losing this fraction intraday.
    daily_loss_limit_pct: float | None = None


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.bar = 0
        self.peak_equity = 0.0
        self.halted = False
        self.halt_until: int | None = None
        self.day: Any = None
        self.day_start_equity = 0.0
        self.day_halted = False
        self._last_equity = 0.0
        self._extreme: dict[str, float] = {}
        self._side: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        self.events: list[tuple[Any, str, str]] = []

    # -- main entry point --------------------------------------------------
    def apply(
        self,
        targets: Mapping[str, float],
        *,
        equity: float,
        positions: Mapping[str, Position],
        prices: Mapping[str, float],
        now: Any = None,
    ) -> dict[str, float]:
        cfg = self.config
        self.bar += 1
        flat = {s: 0.0 for s in set(targets) | set(positions)}

        # A day's loss is measured from the last equity seen on the previous day,
        # so the limit also means something on daily bars.
        day = now.date() if hasattr(now, "date") else None
        if day != self.day or self.day_start_equity <= 0:
            base = self._last_equity if self._last_equity > 0 else equity
            self.day, self.day_start_equity, self.day_halted = day, base, False
        self._last_equity = equity

        if self._drawdown_halt(equity, now):
            return flat
        if cfg.daily_loss_limit_pct is not None:
            if not self.day_halted and equity <= self.day_start_equity * (1 - cfg.daily_loss_limit_pct):
                self.day_halted = True
                self._log(now, "daily_loss_limit", f"equity {equity:.2f}")
            if self.day_halted:
                return flat

        self._check_stops(positions, prices, now)

        out: dict[str, float] = {}
        cap = cfg.max_position_weight
        for sym, w in targets.items():
            if not math.isfinite(w) or self._cooldown.get(sym, 0) > self.bar:
                w = 0.0
            if cfg.long_only and w < 0:
                w = 0.0
            out[sym] = max(-cap, min(cap, w))
        gross = sum(abs(w) for w in out.values())
        if gross > cfg.max_gross_leverage > 0:
            k = cfg.max_gross_leverage / gross
            out = {s: w * k for s, w in out.items()}
        return out

    # -- circuit breaker ---------------------------------------------------
    def _drawdown_halt(self, equity: float, now: Any) -> bool:
        cfg = self.config
        self.peak_equity = max(self.peak_equity, equity)
        if cfg.max_drawdown_pct is None:
            return False
        if self.halted and self.halt_until is not None and self.bar >= self.halt_until:
            self.halted, self.halt_until = False, None
            self.peak_equity = equity  # start measuring afresh after the pause
            self._log(now, "resume", f"equity {equity:.2f}")
        if not self.halted and equity <= self.peak_equity * (1 - cfg.max_drawdown_pct):
            self.halted = True
            if cfg.drawdown_halt_bars is not None:
                self.halt_until = self.bar + cfg.drawdown_halt_bars
            self._log(now, "max_drawdown", f"equity {equity:.2f} peak {self.peak_equity:.2f}")
        return self.halted

    # -- per-position exits --------------------------------------------------
    def _check_stops(self, positions: Mapping[str, Position],
                     prices: Mapping[str, float], now: Any) -> None:
        cfg = self.config
        for sym in list(self._side):
            pos = positions.get(sym)
            if pos is None or pos.qty == 0:
                self._side.pop(sym, None)
                self._extreme.pop(sym, None)
        if (cfg.stop_loss_pct is None and cfg.take_profit_pct is None
                and cfg.trailing_stop_pct is None):
            return
        for sym, pos in positions.items():
            price = prices.get(sym)
            if pos.qty == 0 or price is None or pos.avg_price <= 0:
                continue
            side = 1 if pos.qty > 0 else -1
            if self._side.get(sym) != side:
                self._side[sym] = side
                self._extreme[sym] = pos.avg_price
            best = self._extreme[sym]
            best = max(best, price) if side > 0 else min(best, price)
            self._extreme[sym] = best
            gain = side * (price / pos.avg_price - 1.0)
            giveback = side * (1.0 - price / best)
            reason = None
            if cfg.stop_loss_pct is not None and gain <= -cfg.stop_loss_pct:
                reason = "stop_loss"
            elif cfg.take_profit_pct is not None and gain >= cfg.take_profit_pct:
                reason = "take_profit"
            elif cfg.trailing_stop_pct is not None and giveback >= cfg.trailing_stop_pct:
                reason = "trailing_stop"
            if reason and self._cooldown.get(sym, 0) <= self.bar:
                self._cooldown[sym] = self.bar + max(1, cfg.cooldown_bars)
                self._log(now, reason, f"{sym} at {price:.4f}")

    def _log(self, now: Any, kind: str, detail: str) -> None:
        self.events.append((now, kind, detail))

    # -- persistence (so a restart does not forget a tripped breaker) --------
    def state_dict(self) -> dict:
        return {"peak_equity": self.peak_equity, "halted": self.halted,
                "halt_until": self.halt_until, "bar": self.bar,
                "last_equity": self._last_equity}

    def load_state(self, state: Mapping) -> None:
        self.peak_equity = float(state.get("peak_equity", 0.0))
        self.halted = bool(state.get("halted", False))
        self.halt_until = state.get("halt_until")
        self.bar = int(state.get("bar", 0))
        self._last_equity = float(state.get("last_equity", 0.0))

    def clear_halt(self) -> None:
        """Resume after a tripped drawdown breaker, measuring from current equity."""
        self.halted, self.halt_until, self.peak_equity = False, None, 0.0
