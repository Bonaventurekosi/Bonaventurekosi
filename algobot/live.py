"""Running the pipeline against a live (or replayed) bar feed.

The runner polls a data provider. Each new *completed* bar is appended to the
history and the strategy is called exactly once for it, as in a backtest. Only
the newest bar leads to orders; older bars that arrived while the bot was down
are replayed without trading so strategy state stays correct.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .data.dataset import infer_periods_per_year
from .execution.brokers import Broker
from .history import BarHistory
from .pipeline import Pipeline
from .types import Decision

log = logging.getLogger("algobot")


def feed(pipeline: Pipeline, history: BarHistory, frames: Mapping[str, pd.DataFrame],
         hold_last: bool = True, already_seen: Any = None) -> bool:
    """Append bars newer than what ``history`` holds, one timestamp at a time.

    The strategy observes every new timestamp except, when ``hold_last`` is
    set, the newest, which is left for the caller to trade on with
    ``pipeline.step``. Returns True when such a bar is waiting.

    Bars up to ``already_seen`` are stored but not shown to the strategy: use
    it when the strategy's state was restored from a snapshot taken then.
    """
    fresh = {}
    for sym, df in frames.items():
        if sym not in history:
            continue
        seen = history.last_ts(sym)
        fresh[sym] = df if seen is None else df[df.index > seen]
    stamps = sorted(set().union(*(df.index for df in fresh.values()))) if fresh else []
    for i, ts in enumerate(stamps):
        for sym, df in fresh.items():
            if ts in df.index:
                row = df.loc[ts]
                history.append(sym, ts, row["open"], row["high"], row["low"],
                               row["close"], row["volume"])
        if already_seen is not None and ts <= already_seen:
            continue
        if not (hold_last and i == len(stamps) - 1):
            pipeline.observe(history)
    if already_seen is not None and stamps and stamps[-1] <= already_seen:
        return False
    return bool(stamps) and hold_last


class ReplayProvider:
    """Serves historical bars as if they were arriving now. For demos and tests."""

    def __init__(self, frames: Mapping[str, pd.DataFrame], start: int):
        self.frames = dict(frames)
        timeline = pd.DatetimeIndex([])
        for df in self.frames.values():
            timeline = timeline.union(df.index)
        self.timeline = timeline
        self.cursor = max(1, min(start, len(timeline)))

    def fetch(self, symbols: Sequence[str], bars: int | None = None):
        cutoff = self.timeline[self.cursor - 1]
        out = {s: self.frames[s].loc[:cutoff] for s in symbols}
        return {s: (df.iloc[-bars:] if bars else df) for s, df in out.items()}

    def advance(self) -> bool:
        if self.cursor >= len(self.timeline):
            return False
        self.cursor += 1
        return True


class LiveRunner:
    def __init__(self, pipeline: Pipeline, provider: Any, broker: Broker,
                 symbols: Sequence[str], *, history_bars: int = 500,
                 refresh_bars: int = 30, bar_seconds: float | None = None,
                 poll_seconds: float = 60.0, state_path: str | Path | None = None,
                 dry_run: bool = False, max_order_value: float | None = None,
                 reset_halt: bool = False,
                 clock: Callable[[], pd.Timestamp] | None = None):
        self.pipeline, self.provider, self.broker = pipeline, provider, broker
        self.symbols = list(symbols)
        self.history = BarHistory(self.symbols)
        self.history_bars, self.refresh_bars = history_bars, refresh_bars
        self.bar_seconds, self.poll_seconds = bar_seconds, poll_seconds
        self.state_path = Path(state_path) if state_path else None
        self.dry_run, self.max_order_value = dry_run, max_order_value
        self.reset_halt = reset_halt
        self._clock = clock or (lambda: pd.Timestamp(
            dt.datetime.now(dt.timezone.utc)).tz_localize(None))
        self.last_traded: pd.Timestamp | None = None
        self.decisions: list[tuple[Any, Decision]] = []
        self._started = False
        self._warm = False
        self._restored = False

    # -- state -------------------------------------------------------------
    def _fingerprint(self) -> str:
        """Identifies the strategy configuration a snapshot belongs to."""
        strat = self.pipeline.strategy
        text = repr((strat.name, sorted(strat.params.items(), key=lambda kv: kv[0]),
                     strat.symbols, self.symbols))
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def _snapshot_path(self) -> Path:
        return self.state_path.with_suffix(".strategy.pkl")

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text())
        if state.get("last_traded"):
            self.last_traded = pd.Timestamp(state["last_traded"])
        if state.get("risk"):
            self.pipeline.risk.load_state(state["risk"])
        if state.get("broker") and state.get("broker_name") == self.broker.name:
            self.broker.load_state(state["broker"])
        # Restore the strategy exactly as it was after the last traded bar. The
        # snapshot is only trusted if it was written by this same configuration;
        # otherwise the strategy is rebuilt by replaying history.
        if (self.last_traded is not None and self._snapshot_path.exists()
                and state.get("strategy") == self._fingerprint()):
            try:
                with open(self._snapshot_path, "rb") as fh:
                    self.pipeline.strategy = pickle.load(fh)
                self._restored = True
            except Exception:
                log.exception("could not restore the strategy snapshot; replaying history")

    def _save_state(self) -> None:
        if not self.state_path or self.dry_run:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = self._fingerprint()
        tmp = self._snapshot_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self.pipeline.strategy, fh)
        tmp.replace(self._snapshot_path)
        state = {"last_traded": str(self.last_traded) if self.last_traded is not None else None,
                 "risk": self.pipeline.risk.state_dict(), "strategy": fingerprint,
                 "broker_name": self.broker.name, "broker": self.broker.state()}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)

    # -- bars --------------------------------------------------------------
    def _completed(self, frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Drop the bar that is still forming, if the bar length is known."""
        if not self.bar_seconds:
            return dict(frames)
        cutoff = self._clock() - pd.Timedelta(seconds=self.bar_seconds)
        return {s: df[df.index <= cutoff] for s, df in frames.items()}

    def tick(self) -> Decision | None:
        """Pull new bars; trade if a new one completed. Returns the decision."""
        if not self._started:
            self._load_state()
            self._started = True
            if self.reset_halt:
                self.pipeline.risk.clear_halt()
                log.warning("drawdown halt cleared; measuring drawdown from current equity")
        # Until one fetch has succeeded, keep asking for the full warm-up history.
        first = not self._warm
        frames = self._completed(self.provider.fetch(
            self.symbols, self.history_bars if first else self.refresh_bars))
        sizing = self.pipeline.sizer.config
        if first and sizing.periods_per_year is None and frames:
            longest = max(frames.values(), key=len)
            sizing.periods_per_year = infer_periods_per_year(longest.index)
        seen = self.last_traded if (first and self._restored) else None
        fresh = feed(self.pipeline, self.history, frames, hold_last=True, already_seen=seen)
        self._warm = True
        if not fresh:
            return None
        now = self.history.now
        if self.last_traded is not None and now <= self.last_traded:
            # Already acted on this bar before a restart: just observe it.
            self.pipeline.observe(self.history)
            return None

        prices = self.history.prices()
        self.broker.update_prices(prices)
        account = self.broker.account()
        # Only ever manage the configured symbols; anything else in the account
        # is left alone (but does count toward the equity that sizes positions).
        positions = {s: p for s, p in self.broker.positions().items() if s in self.history}
        decision = self.pipeline.step(self.history, equity=account.equity,
                                      positions=positions, prices=prices, now=now)
        if decision.orders and not self.dry_run:
            try:
                self.broker.cancel_open_orders()   # nothing stale from the last bar
            except Exception:
                log.exception("could not cancel open orders; not trading this bar")
                return None
        log.info("bar %s  equity %.2f  targets %s", now, account.equity,
                 {s: round(w, 4) for s, w in decision.targets.items() if w})
        for order in decision.orders:
            value = abs(order.qty) * prices[order.symbol]
            if self.max_order_value is not None and value > self.max_order_value:
                log.warning("SKIPPED %s %+.6f: value %.2f exceeds max_order_value",
                            order.symbol, order.qty, value)
                continue
            if self.dry_run:
                log.info("DRY RUN %s %+.6f (~%.2f) %s", order.symbol, order.qty,
                         value, order.reason)
                continue
            try:
                status = self.broker.submit(order, prices[order.symbol], now)
                log.info("ORDER %s %+.6f (~%.2f) %s -> %s", order.symbol, order.qty,
                         value, order.reason, status)
            except Exception:  # keep running; the next bar re-derives the order
                log.exception("order failed: %s %+.6f", order.symbol, order.qty)
        self.last_traded = now
        self.decisions.append((now, decision))
        self._save_state()
        return decision

    def run(self, max_ticks: int | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            try:
                self.tick()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("tick failed; retrying after the poll interval")
            ticks += 1
            if max_ticks is None or ticks < max_ticks:
                sleep(self.poll_seconds)
