"""Paper-trading sessions that run in the background while the app is open."""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..config import build_costs, build_pipeline
from ..data import make_provider
from ..execution.brokers import PaperBroker
from ..live import LiveRunner
from ..tasks import symbols_of

_BAR_SECONDS = {"1d": 86400, "1day": 86400, "day": 86400, "4h": 14400, "4hour": 14400,
                "1h": 3600, "1hour": 3600, "60m": 3600, "30m": 1800, "15m": 900, "5m": 300}


def bar_seconds(cfg: Mapping[str, Any]) -> float | None:
    """How long a bar lasts, so the bot waits for it to finish before acting."""
    live, data = cfg.get("live") or {}, cfg.get("data") or {}
    if live.get("bar_seconds"):
        return float(live["bar_seconds"])
    size = str(data.get("interval") or data.get("timeframe") or "1d").lower()
    return _BAR_SECONDS.get(size)


class PaperSession:
    """One setup trading a simulated account against its live price feed."""

    def __init__(self, setup: str, cfg: Mapping[str, Any], state_dir: str | Path,
                 provider: Any = None, poll_seconds: float | None = None):
        live = dict(cfg.get("live") or {})
        self.setup = setup
        self.cash = float(live.get("cash", (cfg.get("backtest") or {}).get("initial_cash", 100_000.0)))
        self.broker = PaperBroker(self.cash, build_costs(cfg))
        self.poll_seconds = float(poll_seconds if poll_seconds is not None
                                  else live.get("poll_seconds", 300))
        self.runner = LiveRunner(
            build_pipeline(cfg), provider or make_provider(cfg["data"]), self.broker,
            symbols_of(cfg), history_bars=int(live.get("history_bars", 500)),
            refresh_bars=int(live.get("refresh_bars", 30)), bar_seconds=bar_seconds(cfg),
            poll_seconds=self.poll_seconds, max_order_value=live.get("max_order_value"),
            state_path=Path(state_dir) / f"app-{setup}.json")
        self.started = pd.Timestamp.now().floor("s")
        self.last_check: pd.Timestamp | None = None
        self.checks = 0
        self.error: str | None = None
        self.curve: deque = deque(maxlen=2000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=f"paper-{self.setup}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def check_once(self) -> None:
        """Look for a new completed bar and trade it if there is one."""
        with self._lock:
            try:
                decision = self.runner.tick()
                self.error = None
                if decision is not None:
                    self.curve.append([str(self.runner.history.now),
                                       round(self.broker.account().equity, 2)])
            except Exception as exc:   # a feed hiccup must not kill the session
                self.error = f"{type(exc).__name__}: {exc}"
            self.last_check = pd.Timestamp.now().floor("s")
            self.checks += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self.poll_seconds)

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                account = self.broker.account()
                equity, cash = round(account.equity, 2), round(account.cash, 2)
            except KeyError:           # positions restored but no prices seen yet
                equity, cash = None, round(self.broker.portfolio.cash, 2)
            now = self.runner.history.now
            return {
                "setup": self.setup, "running": self.running, "started": str(self.started),
                "last_check": str(self.last_check) if self.last_check is not None else None,
                "checks": self.checks, "poll_seconds": self.poll_seconds, "error": self.error,
                "last_bar": str(now) if now is not None else None,
                "last_traded": str(self.runner.last_traded) if self.runner.last_traded is not None else None,
                "start_cash": self.cash, "equity": equity, "cash": cash,
                "positions": {s: round(p.qty, 6) for s, p in self.broker.positions().items()},
                "orders": [{"time": str(f.ts), "symbol": f.symbol,
                            "side": "buy" if f.qty > 0 else "sell", "qty": round(abs(f.qty), 6),
                            "price": round(f.price, 4), "reason": f.reason}
                           for f in self.broker.portfolio.fills[-50:]][::-1],
                "curve": list(self.curve),
            }
