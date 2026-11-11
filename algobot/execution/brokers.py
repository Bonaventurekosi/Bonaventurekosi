"""Brokers: where orders go when the bot runs outside a backtest.

``PaperBroker`` simulates fills locally and is the default. ``AlpacaBroker``
and ``CCXTBroker`` talk to real venues; both default to the venue's own paper
or sandbox environment and need an explicit opt-in to touch real money.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..portfolio import Portfolio
from ..types import Fill, Order, Position
from .orders import CostModel


@dataclass
class Account:
    equity: float
    cash: float


class Broker(ABC):
    name = "broker"
    real_money = False

    def update_prices(self, prices: Mapping[str, float]) -> None:
        """Latest prices, for brokers that mark positions locally."""

    @abstractmethod
    def account(self) -> Account:
        ...

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        ...

    @abstractmethod
    def submit(self, order: Order, ref_price: float, now: Any = None) -> dict[str, Any]:
        """Send a market order. Returns a small status record."""

    def cancel_open_orders(self) -> None:
        """Cancel anything still working from an earlier bar (no-op by default)."""

    def state(self) -> dict | None:
        return None

    def load_state(self, state: Mapping) -> None:
        pass


class PaperBroker(Broker):
    """Fills every order immediately at the reference price plus costs."""

    name = "paper"

    def __init__(self, cash: float = 100_000.0, costs: CostModel | None = None):
        self.portfolio = Portfolio(cash)
        self.costs = costs or CostModel()
        self._prices: dict[str, float] = {}

    def update_prices(self, prices):
        self._prices.update(prices)

    def account(self) -> Account:
        return Account(self.portfolio.equity(self._prices), self.portfolio.cash)

    def positions(self) -> dict[str, Position]:
        return {s: Position(p.qty, p.avg_price)
                for s, p in self.portfolio.positions.items() if p.qty != 0.0}

    def submit(self, order, ref_price, now=None):
        px = self.costs.fill_price(ref_price, order.qty)
        fee = self.costs.commission(order.qty, px)
        self.portfolio.apply_fill(Fill(now, order.symbol, order.qty, px, fee, order.reason))
        self._prices.setdefault(order.symbol, ref_price)
        return {"status": "filled", "price": round(px, 6), "commission": round(fee, 4)}

    def state(self):
        return self.portfolio.to_dict()

    def load_state(self, state):
        self.portfolio = Portfolio.from_dict(state)


class AlpacaBroker(Broker):
    """Alpaca Trading API (stocks, ETFs and crypto). Paper endpoint by default.

    Alpaca only accepts fractional quantities on long, day, market orders for
    stocks; if a strategy shorts stocks, set ``execution.fractional: false``.
    """

    name = "alpaca"
    PAPER = "https://paper-api.alpaca.markets"
    LIVE = "https://api.alpaca.markets"

    def __init__(self, key: str, secret: str, paper: bool = True,
                 symbols: Sequence[str] = (), time_in_force: str = "day",
                 session: Any = None, timeout: float = 20.0):
        self.base = self.PAPER if paper else self.LIVE
        self.real_money = not paper
        self.time_in_force, self.timeout = time_in_force, timeout
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._session = session
        # Alpaca reports crypto positions without the slash (BTCUSD).
        self._alias = {s.replace("/", ""): s for s in symbols}

    def _call(self, method: str, path: str, **kwargs):
        if self._session is None:
            import requests
            self._session = requests.Session()
        resp = self._session.request(method, self.base + path, headers=self._headers,
                                     timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def account(self) -> Account:
        acct = self._call("GET", "/v2/account")
        return Account(float(acct["equity"]), float(acct["cash"]))

    def positions(self) -> dict[str, Position]:
        out = {}
        for row in self._call("GET", "/v2/positions") or []:
            qty = abs(float(row["qty"])) * (-1.0 if row.get("side") == "short" else 1.0)
            sym = self._alias.get(row["symbol"], row["symbol"])
            out[sym] = Position(qty, float(row["avg_entry_price"]))
        return out

    def submit(self, order, ref_price, now=None):
        body = {
            "symbol": order.symbol,
            "qty": f"{abs(order.qty):.9f}".rstrip("0").rstrip("."),
            "side": "buy" if order.qty > 0 else "sell",
            "type": "market",
            "time_in_force": "gtc" if "/" in order.symbol else self.time_in_force,
        }
        resp = self._call("POST", "/v2/orders", json=body)
        return {"status": resp.get("status", "submitted"), "id": resp.get("id")}

    def cancel_open_orders(self) -> None:
        self._call("DELETE", "/v2/orders")

    def market_open(self) -> bool:
        return bool(self._call("GET", "/v2/clock")["is_open"])


class CCXTBroker(Broker):
    """Spot crypto through the optional ``ccxt`` package. Long-only.

    Exchanges do not report an average entry price for spot balances, so the
    risk manager's per-position stops (which need one) are inactive here; the
    drawdown and daily-loss breakers still work.
    """

    name = "ccxt"

    def __init__(self, exchange: Any, symbols: Sequence[str], sandbox: bool = True):
        self.exchange = exchange
        self.symbols = list(symbols)
        self.real_money = not sandbox
        quotes = {s.split("/")[1] for s in self.symbols}
        if len(quotes) != 1:
            raise ValueError("CCXTBroker: all symbols must share one quote currency")
        self.quote = quotes.pop()
        self._prices: dict[str, float] = {}

    @classmethod
    def connect(cls, exchange: str, symbols: Sequence[str], key: str, secret: str,
                sandbox: bool = True) -> "CCXTBroker":
        import ccxt  # optional dependency
        client = getattr(ccxt, exchange)({"apiKey": key, "secret": secret,
                                          "enableRateLimit": True})
        if sandbox:
            client.set_sandbox_mode(True)
        return cls(client, symbols, sandbox)

    def update_prices(self, prices):
        self._prices.update(prices)

    def _balances(self) -> dict[str, float]:
        return {k: float(v or 0.0) for k, v in self.exchange.fetch_balance()["total"].items()}

    def account(self) -> Account:
        bal = self._balances()
        cash = bal.get(self.quote, 0.0)
        held = sum(bal.get(s.split("/")[0], 0.0) * self._prices.get(s, 0.0)
                   for s in self.symbols)
        return Account(cash + held, cash)

    def positions(self) -> dict[str, Position]:
        bal = self._balances()
        return {s: Position(bal[s.split("/")[0]], 0.0) for s in self.symbols
                if bal.get(s.split("/")[0], 0.0) > 0}

    def submit(self, order, ref_price, now=None):
        resp = self.exchange.create_order(order.symbol, "market",
                                          "buy" if order.qty > 0 else "sell",
                                          abs(order.qty))
        return {"status": resp.get("status", "submitted"), "id": resp.get("id")}
