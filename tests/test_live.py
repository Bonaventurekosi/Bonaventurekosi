import json

import numpy as np
import pandas as pd
import pytest

from algobot.config import build_pipeline, run_backtest
from algobot.execution.brokers import PaperBroker
from algobot.execution.orders import CostModel
from algobot.history import BarHistory
from algobot.live import LiveRunner, ReplayProvider, feed
from algobot.types import Order

CFG = {"strategy": {"type": "ensemble", "params": {"children": [
    {"type": "donchian", "params": {"entry": 20, "exit": 10}}, {"type": "bollinger"},
    {"type": "grid"}, {"type": "xs_momentum", "params": {"lookback": 30, "rebalance_every": 7}}]}}}


def replay(mixed, cfg, start, **kw):
    provider = ReplayProvider(mixed.frames, start=start)
    broker = PaperBroker(100_000.0, CostModel(0, 0))
    runner = LiveRunner(build_pipeline(cfg), provider, broker, mixed.symbols, **kw)
    runner.tick()
    while provider.advance():
        runner.tick()
    return runner, broker


def test_live_decisions_match_the_backtest(mixed):
    """The live loop must call the strategy exactly as the backtester does."""
    runner, _ = replay(mixed, CFG, start=400)
    bt = build_pipeline(CFG)
    h = BarHistory(mixed.symbols)
    expected = {}
    for i, ts in enumerate(mixed.timeline):
        for sym in mixed.symbols:
            row = mixed.frames[sym].iloc[i]
            h.append(sym, ts, row["open"], row["high"], row["low"], row["close"], row["volume"])
        expected[ts] = bt.strategy.target_weights(h)
    assert len(runner.decisions) == len(mixed) - 400 + 1
    for ts, decision in runner.decisions:
        assert decision.raw == pytest.approx(expected[ts])


def test_paper_account_tracks_targets(mixed):
    cfg = {"strategy": {"type": "buy_and_hold"}}
    runner, broker = replay(mixed, cfg, start=600)
    prices = runner.history.prices()
    weights = broker.portfolio.weights(prices)
    assert weights == pytest.approx({s: 0.25 for s in mixed.symbols}, abs=0.02)
    assert broker.account().equity == pytest.approx(broker.portfolio.equity(prices))


def test_feed_appends_only_new_bars(mixed):
    pipe = build_pipeline({"strategy": {"type": "buy_and_hold"}})
    h = BarHistory(mixed.symbols)
    first = {s: df.iloc[:50] for s, df in mixed.frames.items()}
    assert feed(pipe, h, first) is True and h.size("AAA") == 50
    assert feed(pipe, h, first) is False and h.size("AAA") == 50          # nothing new
    more = {s: df.iloc[40:53] for s, df in mixed.frames.items()}          # overlaps what we have
    assert feed(pipe, h, more) is True and h.size("AAA") == 53
    np.testing.assert_array_equal(h.close("AAA"), mixed.frames["AAA"]["close"].to_numpy()[:53])
    assert h.now == mixed.timeline[52]


def test_incomplete_bar_is_ignored(mixed):
    provider = ReplayProvider(mixed.frames, start=300)
    latest = mixed.timeline[299]
    clock = {"now": latest + pd.Timedelta(hours=6)}                       # that bar is still forming
    runner = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}), provider,
                        PaperBroker(), mixed.symbols, bar_seconds=86400,
                        clock=lambda: clock["now"])
    runner.tick()
    assert runner.history.now == mixed.timeline[298]
    clock["now"] = latest + pd.Timedelta(days=1)
    assert runner.tick() is not None and runner.history.now == latest


def test_restart_resumes_without_double_trading(mixed, tmp_path):
    state = tmp_path / "bot.json"
    cfg = {"strategy": {"type": "buy_and_hold"}, "risk": {"max_drawdown_pct": 0.5}}

    def runner_at(cursor):
        return LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=cursor),
                          PaperBroker(50_000.0, CostModel(0, 0)), mixed.symbols, state_path=state)

    first = runner_at(300)
    assert first.tick() is not None
    saved = json.loads(state.read_text())
    assert saved["last_traded"] == str(mixed.timeline[299]) and saved["broker_name"] == "paper"
    held = first.broker.positions()

    again = runner_at(300)                      # restart, no new bar yet
    assert again.tick() is None                 # already acted on this bar
    assert {s: p.qty for s, p in again.broker.positions().items()} == pytest.approx(
        {s: p.qty for s, p in held.items()})
    assert again.pipeline.risk.peak_equity == pytest.approx(50_000.0)

    later = runner_at(303)                      # restart after three more bars
    decision = later.tick()
    assert decision is not None and later.last_traded == mixed.timeline[302]


def test_dry_run_and_order_cap(mixed, tmp_path, caplog):
    state = tmp_path / "s.json"
    cfg = {"strategy": {"type": "buy_and_hold"}}
    dry = LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=300), PaperBroker(),
                     mixed.symbols, dry_run=True, state_path=state)
    decision = dry.tick()
    assert len(decision.orders) == 4 and dry.broker.positions() == {} and not state.exists()
    capped = LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=300), PaperBroker(),
                        mixed.symbols, max_order_value=1_000.0)
    with caplog.at_level("WARNING", logger="algobot"):
        capped.tick()
    assert capped.broker.positions() == {} and "exceeds max_order_value" in caplog.text


class FlakyBroker(PaperBroker):
    def submit(self, order, ref_price, now=None):
        if order.symbol == "BBB":
            raise ConnectionError("venue down")
        return super().submit(order, ref_price, now)


def test_one_failed_order_does_not_stop_the_rest(mixed):
    runner = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}),
                        ReplayProvider(mixed.frames, start=300), FlakyBroker(), mixed.symbols)
    runner.tick()
    assert set(runner.broker.positions()) == {"AAA", "CCC", "DDD"}


def test_run_loop_survives_a_failing_tick(mixed):
    class Boom:
        calls = 0

        def fetch(self, symbols, bars=None):
            Boom.calls += 1
            raise RuntimeError("feed down")

    runner = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}), Boom(),
                        PaperBroker(), mixed.symbols, poll_seconds=0)
    naps = []
    runner.run(max_ticks=3, sleep=naps.append)
    assert Boom.calls == 3 and len(naps) == 2


STATEFUL = {"strategy": {"type": "ensemble", "params": {"method": "adaptive", "lookback": 30, "children": [
    {"type": "donchian", "params": {"entry": 20, "exit": 10}}, {"type": "bollinger"},
    {"type": "grid"}, {"type": "dca", "params": {"every": 9}},
    {"type": "xs_momentum", "params": {"lookback": 30, "rebalance_every": 7}},
    {"type": "ml_logistic", "params": {"min_train": 60}}]}},
    "sizing": {"target_vol": 0.15, "periods_per_year": 252}, "risk": {"trailing_stop_pct": 0.08}}


def stateful_runner(mixed, cursor, state):
    provider = ReplayProvider(mixed.frames, start=cursor)
    runner = LiveRunner(build_pipeline(STATEFUL), provider, PaperBroker(100_000.0, CostModel(1, 2)),
                        mixed.symbols, history_bars=150, state_path=state)
    return runner, provider


def test_restart_continues_exactly_where_it_stopped(mixed, tmp_path):
    """Stop the bot mid-run and start a fresh process: every later decision must
    be identical to a run that never stopped."""
    straight, provider = stateful_runner(mixed, 300, tmp_path / "a.json")
    straight.tick()
    while provider.cursor < 500 and provider.advance():
        straight.tick()

    part1, provider = stateful_runner(mixed, 300, tmp_path / "b.json")
    part1.tick()
    while provider.cursor < 400 and provider.advance():
        part1.tick()
    part2, provider = stateful_runner(mixed, 403, tmp_path / "b.json")   # down for three bars
    part2.tick()
    assert part2._restored
    while provider.cursor < 500 and provider.advance():
        part2.tick()

    want = {ts: d for ts, d in straight.decisions}
    resumed = dict(part2.decisions)
    assert len(resumed) == 98 and max(resumed) == mixed.timeline[499]
    for ts, decision in resumed.items():
        assert decision.raw == pytest.approx(want[ts].raw), ts
    # The two missed bars were not traded, so the books differ slightly at first,
    # but the strategy itself is in exactly the same state.
    assert part2.pipeline.strategy.allocation == pytest.approx(straight.pipeline.strategy.allocation)


def test_snapshot_is_ignored_when_the_config_changes(mixed, tmp_path):
    state = tmp_path / "s.json"
    first, _ = stateful_runner(mixed, 300, state)
    first.tick()
    other = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}),
                       ReplayProvider(mixed.frames, start=302), PaperBroker(), mixed.symbols,
                       state_path=state)
    assert other.tick() is not None and not other._restored
    assert other.pipeline.strategy.name == "buy_and_hold"


class SharedAccount(PaperBroker):
    """An account that also holds things the bot was never told about."""
    cancelled = 0

    def positions(self):
        from algobot.types import Position
        return {**super().positions(), "PRIVATE": Position(100.0, 50.0)}

    def cancel_open_orders(self):
        SharedAccount.cancelled += 1


def test_only_configured_symbols_are_touched(mixed):
    runner = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}),
                        ReplayProvider(mixed.frames, start=300), SharedAccount(), mixed.symbols)
    decision = runner.tick()
    assert {o.symbol for o in decision.orders} == set(mixed.symbols)
    assert "PRIVATE" not in decision.targets and SharedAccount.cancelled == 1


def test_reset_halt(mixed, tmp_path):
    state = tmp_path / "h.json"
    cfg = {"strategy": {"type": "buy_and_hold"}, "risk": {"max_drawdown_pct": 0.1}}
    tripped = LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=300),
                         PaperBroker(), mixed.symbols, state_path=state)
    tripped.pipeline.risk.halted = True
    tripped.pipeline.risk.peak_equity = 1e9
    assert tripped.tick().orders == []
    still = LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=301),
                       PaperBroker(), mixed.symbols, state_path=state)
    assert still.tick().orders == []                       # the halt survives a restart
    resumed = LiveRunner(build_pipeline(cfg), ReplayProvider(mixed.frames, start=302),
                         PaperBroker(), mixed.symbols, state_path=state, reset_halt=True)
    assert len(resumed.tick().orders) == 4


def test_failed_first_fetch_still_warms_up_fully(mixed):
    class FlakyOnce(ReplayProvider):
        asked = []

        def fetch(self, symbols, bars=None):
            FlakyOnce.asked.append(bars)
            if len(FlakyOnce.asked) == 1:
                raise ConnectionError("feed down")
            return super().fetch(symbols, bars)

    runner = LiveRunner(build_pipeline({"strategy": {"type": "buy_and_hold"}}),
                        FlakyOnce(mixed.frames, start=300), PaperBroker(), mixed.symbols,
                        history_bars=200, refresh_bars=10, poll_seconds=0)
    runner.run(max_ticks=3, sleep=lambda _: None)
    assert FlakyOnce.asked == [200, 200, 10]            # full history again after the failure
    assert runner.history.size("AAA") == 200 and len(runner.decisions) == 1
