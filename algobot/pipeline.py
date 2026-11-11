"""The decision pipeline: strategy -> sizing -> risk -> orders.

This is the single code path shared by the backtester, the paper/live runner
and the one-shot planner. Whatever a backtest did on a bar is exactly what the
live bot would do given the same history and the same account state.
"""
from __future__ import annotations

from typing import Any, Mapping

from .execution.orders import ExecutionConfig, build_orders
from .history import BarHistory
from .risk import RiskManager
from .sizing import Sizer
from .strategies.base import Strategy
from .types import Decision, Position


class Pipeline:
    def __init__(self, strategy: Strategy, sizer: Sizer | None = None,
                 risk: RiskManager | None = None,
                 execution: ExecutionConfig | None = None):
        self.strategy = strategy
        self.sizer = sizer or Sizer()
        self.risk = risk or RiskManager()
        self.execution = execution or ExecutionConfig()

    def observe(self, history: BarHistory) -> dict[str, float]:
        """Show the strategy a bar without trading on it (warm-up, replay)."""
        return self.strategy.target_weights(history)

    def step(self, history: BarHistory, *, equity: float,
             positions: Mapping[str, Position],
             prices: Mapping[str, float] | None = None,
             now: Any = None) -> Decision:
        prices = prices if prices is not None else history.prices()
        raw = self.strategy.target_weights(history)
        sized = self.sizer.apply(raw, history)
        targets = self.risk.apply(sized, equity=equity, positions=positions,
                                  prices=prices,
                                  now=now if now is not None else history.now)
        volumes = None
        if self.execution.max_participation is not None:
            volumes = {s: float(history.volume(s, 1)[-1])
                       for s in history.symbols if history.size(s)}
        quantities = {s: p.qty for s, p in positions.items() if p.qty != 0.0}
        orders = build_orders(targets, quantities, equity, prices,
                              self.execution, volumes)
        return Decision(raw=raw, sized=sized, targets=targets, orders=orders)
