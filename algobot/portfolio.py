"""Cash and position accounting with average-cost P&L and round-trip trades."""
from __future__ import annotations

from typing import Mapping

from .types import Fill, Position, Trade

_EPS = 1e-12


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Portfolio:
    def __init__(self, cash: float = 100_000.0):
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self._open: dict[str, Trade] = {}

    # -- queries -----------------------------------------------------------
    def qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.qty if pos else 0.0

    def quantities(self) -> dict[str, float]:
        return {s: p.qty for s, p in self.positions.items() if abs(p.qty) > _EPS}

    def equity(self, prices: Mapping[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            if abs(pos.qty) > _EPS:
                total += pos.qty * prices[sym]
        return total

    def gross_value(self, prices: Mapping[str, float]) -> float:
        return sum(abs(p.qty) * prices[s] for s, p in self.positions.items()
                   if abs(p.qty) > _EPS)

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {s: p.qty * prices[s] / eq for s, p in self.positions.items()
                if abs(p.qty) > _EPS}

    # -- updates -----------------------------------------------------------
    def apply_fill(self, fill: Fill) -> None:
        q, px = fill.qty, fill.price
        if abs(q) <= _EPS:
            return
        pos = self.positions.setdefault(fill.symbol, Position())
        self.cash -= q * px + fill.commission
        self.fees_paid += fill.commission
        self.fills.append(fill)
        old = pos.qty

        if abs(old) <= _EPS or _sign(old) == _sign(q):
            # Opening or adding: blend the average price.
            total = abs(old) + abs(q)
            pos.avg_price = (abs(old) * pos.avg_price + abs(q) * px) / total
            pos.qty = old + q
            trade = self._open.get(fill.symbol)
            if trade is None:
                trade = self._start_trade(fill, pos.qty)
            trade.max_qty = max(trade.max_qty, abs(pos.qty))
            trade.fees += fill.commission
            return

        # Reducing, closing or flipping.
        closing = min(abs(q), abs(old))
        pnl = closing * (px - pos.avg_price) * _sign(old)
        self.realized_pnl += pnl
        trade = self._open.get(fill.symbol)
        if trade is not None:
            trade.gross_pnl += pnl
            trade.fees += fill.commission
        new = old + q
        if abs(new) <= _EPS:
            pos.qty, pos.avg_price = 0.0, 0.0
            self._close_trade(fill)
        elif _sign(new) != _sign(old):
            pos.qty, pos.avg_price = new, px
            self._close_trade(fill)
            self._start_trade(fill, new).max_qty = abs(new)
        else:
            pos.qty = new

    def _start_trade(self, fill: Fill, qty: float) -> Trade:
        trade = Trade(symbol=fill.symbol, side=_sign(qty), entry_ts=fill.ts,
                      entry_price=fill.price)
        self._open[fill.symbol] = trade
        return trade

    def _close_trade(self, fill: Fill) -> None:
        trade = self._open.pop(fill.symbol, None)
        if trade is not None:
            trade.exit_ts, trade.exit_price = fill.ts, fill.price
            self.trades.append(trade)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "positions": {s: {"qty": p.qty, "avg_price": p.avg_price}
                          for s, p in self.positions.items() if abs(p.qty) > _EPS},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Portfolio":
        pf = cls(d.get("initial_cash", d["cash"]))
        pf.cash = float(d["cash"])
        pf.realized_pnl = float(d.get("realized_pnl", 0.0))
        pf.fees_paid = float(d.get("fees_paid", 0.0))
        for sym, p in d.get("positions", {}).items():
            pf.positions[sym] = Position(float(p["qty"]), float(p["avg_price"]))
        return pf
