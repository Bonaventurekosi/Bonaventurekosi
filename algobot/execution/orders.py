"""Trading costs and the translation of target weights into orders."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..types import Order


@dataclass
class CostModel:
    """Proportional costs. ``slippage_bps`` moves every fill against you."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    min_commission: float = 0.0

    def fill_price(self, ref_price: float, qty: float) -> float:
        side = 1.0 if qty > 0 else -1.0
        return ref_price * (1.0 + side * self.slippage_bps / 1e4)

    def commission(self, qty: float, price: float) -> float:
        if qty == 0:
            return 0.0
        return max(self.min_commission, abs(qty * price) * self.commission_bps / 1e4)


@dataclass
class ExecutionConfig:
    # Skip trades smaller than this fraction of equity (kills churn).
    min_trade_weight: float = 0.005
    # Whole-unit trading when False; quantities are truncated to ``lot_size``.
    fractional: bool = True
    lot_size: float = 1.0
    # Cap each order at this fraction of the last bar's volume.
    max_participation: float | None = None
    # Cap each order's notional at this fraction of equity.
    max_order_weight: float | None = None


def build_orders(
    targets: Mapping[str, float],
    quantities: Mapping[str, float],
    equity: float,
    prices: Mapping[str, float],
    cfg: ExecutionConfig,
    volumes: Mapping[str, float] | None = None,
) -> list[Order]:
    """Orders that move ``quantities`` to ``targets`` (fractions of equity).

    ``targets`` is the complete desired book: anything held but not listed is
    sold. Sells come first so they fund the buys.
    """
    orders: list[Order] = []
    if equity <= 0:
        return orders
    for sym in sorted(set(targets) | set(quantities)):
        price = prices.get(sym)
        if price is None or not math.isfinite(price) or price <= 0:
            continue
        current = quantities.get(sym, 0.0)
        weight = targets.get(sym, 0.0)
        if not math.isfinite(weight):
            weight = 0.0
        target_qty = weight * equity / price
        if not cfg.fractional:
            target_qty = math.trunc(target_qty / cfg.lot_size) * cfg.lot_size
        delta = target_qty - current
        closing = target_qty == 0.0 and current != 0.0

        limit = math.inf
        if cfg.max_order_weight is not None:
            limit = cfg.max_order_weight * equity / price
        if cfg.max_participation is not None and volumes is not None:
            limit = min(limit, cfg.max_participation * volumes.get(sym, 0.0))
        if abs(delta) > limit:
            delta = math.copysign(limit, delta)
            if not cfg.fractional:
                delta = math.trunc(delta / cfg.lot_size) * cfg.lot_size
            closing = False

        if delta == 0.0:
            continue
        if not closing and abs(delta) * price < cfg.min_trade_weight * equity:
            continue
        orders.append(Order(sym, delta, "exit" if closing else "rebalance"))
    orders.sort(key=lambda o: o.qty * prices[o.symbol])
    return orders
