"""Plain data types shared across the framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Order:
    """A market order. ``qty`` is signed: positive buys, negative sells."""

    symbol: str
    qty: float
    reason: str = "rebalance"


@dataclass(frozen=True)
class Fill:
    ts: Any
    symbol: str
    qty: float
    price: float
    commission: float = 0.0
    reason: str = ""

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class Trade:
    """A round trip: from a position being opened until it is flat or flipped."""

    symbol: str
    side: int  # +1 long, -1 short
    entry_ts: Any
    entry_price: float
    exit_ts: Any = None
    exit_price: float = 0.0
    max_qty: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def closed(self) -> bool:
        return self.exit_ts is not None


@dataclass
class Decision:
    """Everything one pipeline step produced, kept for logging and inspection."""

    raw: dict[str, float] = field(default_factory=dict)
    sized: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)
