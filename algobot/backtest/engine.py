"""Event-driven backtester.

Timing, which is where backtests usually cheat:

1. Orders decided on bar ``t`` fill at the *open* of bar ``t+1``, moved against
   you by the slippage setting, and pay commission.
2. Bar ``t+1`` is then added to the history the strategy can see.
3. The book is marked at that bar's close and the strategy decides again.

So a decision never uses a price from its own future, and no fill happens at a
price the strategy had already seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.dataset import Dataset
from ..execution.orders import CostModel
from ..history import BarHistory
from ..pipeline import Pipeline
from ..portfolio import Portfolio
from ..types import Fill, Order, Trade
from .metrics import compute_metrics, format_metrics


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    # Annual rate charged on the market value of short positions.
    borrow_rate: float = 0.0
    # Annual rate charged on borrowed money when gross long exceeds equity.
    margin_rate: float = 0.0
    # Bars per year for annualising; measured from the data when None.
    periods_per_year: float | None = None


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    exposure: pd.Series
    weights: pd.DataFrame
    fills: list[Fill]
    trades: list[Trade]
    risk_events: list[tuple[Any, str, str]]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    periods_per_year: float
    bankrupt: bool = False
    final_positions: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = f"  {'':<26}{'strategy':>14}{'benchmark':>14}"
        body = format_metrics(self.metrics, self.benchmark_metrics)
        span = f"  {self.equity.index[0]} -> {self.equity.index[-1]}  ({len(self.equity)} bars)"
        tail = "\n  ** account wiped out **" if self.bankrupt else ""
        return f"{span}\n{head}\n{body}{tail}"


class Backtester:
    """Runs one pipeline over one dataset. Single use: strategies carry state."""

    def __init__(self, data: Dataset | Mapping[str, pd.DataFrame], pipeline: Pipeline,
                 costs: CostModel | None = None, config: BacktestConfig | None = None):
        self.dataset = data if isinstance(data, Dataset) else Dataset(data)
        self.pipeline = pipeline
        self.costs = costs or CostModel()
        self.config = config or BacktestConfig()
        self._used = False

    def run(self, trade_start: Any = None) -> BacktestResult:
        """Run the backtest. With ``trade_start`` the strategy still sees every
        earlier bar (to warm up) but no order is placed before that time, and
        the result only covers the period from then on."""
        if self._used:
            raise RuntimeError("Backtester already ran; build a new one")
        self._used = True
        ds, cfg, pipe, costs = self.dataset, self.config, self.pipeline, self.costs
        ppy = cfg.periods_per_year or ds.periods_per_year
        if pipe.sizer.config.periods_per_year is None:
            pipe.sizer.config = replace(pipe.sizer.config, periods_per_year=ppy)

        symbols, arrays = ds.symbols, ds.arrays
        timeline, t_ns = ds.timeline, ds.timeline.asi8
        n = len(timeline)
        start_ns = pd.Timestamp(trade_start).value if trade_start is not None else None
        sizes = {s: len(arrays[s][0]) for s in symbols}
        ptr = {s: 0 for s in symbols}

        pf = Portfolio(cfg.initial_cash)
        hist = BarHistory(symbols, capacity=n + 1)
        last: dict[str, float] = {}
        pending: list[Order] = []
        equity = np.empty(n)
        exposure = np.empty(n)
        weights: list[dict[str, float]] = []
        traded = financing = 0.0
        bankrupt = False
        end = n

        for i in range(n):
            t, now = t_ns[i], timeline[i]
            live = {}
            for s in symbols:
                p = ptr[s]
                if p < sizes[s] and arrays[s][0][p] == t:
                    live[s] = p

            # 1. Yesterday's decisions fill at today's open.
            for order in pending:
                p = live.get(order.symbol)
                if p is None:
                    continue  # no bar for this symbol now; it is re-decided below
                px = costs.fill_price(float(arrays[order.symbol][1][p]), order.qty)
                pf.apply_fill(Fill(now, order.symbol, order.qty, px,
                                   costs.commission(order.qty, px), order.reason))
                traded += abs(order.qty * px)
            pending = []

            # 2. The bar becomes visible.
            for s, p in live.items():
                _, o, h, l, c, v = arrays[s]
                hist.append(s, now, o[p], h[p], l[p], c[p], v[p])
                last[s] = float(c[p])
                ptr[s] = p + 1

            # 3. Financing, then mark to market at the close.
            if cfg.borrow_rate or cfg.margin_rate:
                longs = shorts = 0.0
                for s, pos in pf.positions.items():
                    if pos.qty > 0:
                        longs += pos.qty * last[s]
                    elif pos.qty < 0:
                        shorts -= pos.qty * last[s]
                loan = max(0.0, longs - pf.equity(last))
                charge = (shorts * cfg.borrow_rate + loan * cfg.margin_rate) / ppy
                pf.cash -= charge
                financing += charge
            eq = pf.equity(last)
            equity[i] = eq
            exposure[i] = pf.gross_value(last) / eq if eq > 0 else 0.0
            weights.append(pf.weights(last))
            if eq <= 0:
                equity[i], bankrupt, end = 0.0, True, i + 1
                break

            # 4. Decide for the next bar.
            decision = pipe.step(hist, equity=eq, positions=pf.positions,
                                 prices=last, now=now)
            if start_ns is None or t >= start_ns:
                pending = decision.orders

        index = timeline[:end]
        eq_s = pd.Series(equity[:end], index=index, name="equity")
        ex_s = pd.Series(exposure[:end], index=index, name="exposure")
        w_df = pd.DataFrame(weights, index=index).reindex(columns=symbols).fillna(0.0)
        bench = ds.benchmark().reindex(index).ffill()
        if trade_start is not None:
            keep = index >= pd.Timestamp(trade_start)
            eq_s, ex_s, w_df, bench = eq_s[keep], ex_s[keep], w_df[keep], bench[keep]
        if len(bench):
            bench = bench / bench.iloc[0] * cfg.initial_cash
        bench.name = "benchmark"

        return BacktestResult(
            equity=eq_s, benchmark=bench, exposure=ex_s, weights=w_df,
            fills=pf.fills, trades=pf.trades, risk_events=list(pipe.risk.events),
            metrics=compute_metrics(eq_s, periods_per_year=ppy, trades=pf.trades,
                                    traded_notional=traded, exposure=ex_s,
                                    fees=pf.fees_paid + financing),
            benchmark_metrics=compute_metrics(bench, periods_per_year=ppy),
            periods_per_year=ppy, bankrupt=bankrupt,
            final_positions=pf.quantities(),
        )
