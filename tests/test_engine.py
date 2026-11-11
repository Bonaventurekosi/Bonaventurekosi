import numpy as np
import pandas as pd
import pytest

from algobot.backtest.engine import BacktestConfig, Backtester
from algobot.backtest.metrics import bootstrap_sharpe, compute_metrics
from algobot.data import Dataset
from algobot.execution.orders import CostModel, ExecutionConfig
from algobot.pipeline import Pipeline
from algobot.risk import RiskConfig, RiskManager
from algobot.sizing import Sizer, SizingConfig
from algobot.strategies.base import Strategy
from conftest import Fixed, make_bars

FREE = CostModel(commission_bps=0, slippage_bps=0)


def run(frames, strategy, costs=FREE, risk=None, sizing=None, execution=None, cfg=None, **kw):
    pipe = Pipeline(strategy, Sizer(sizing), RiskManager(risk),
                    execution or ExecutionConfig(min_trade_weight=0.0))
    return Backtester(frames, pipe, costs, cfg or BacktestConfig(initial_cash=1000.0)).run(**kw)


def test_orders_fill_at_next_open_never_same_bar():
    closes = [100.0, 110.0, 121.0, 133.1]
    opens = [100.0, 104.0, 112.0, 125.0]
    res = run({"X": make_bars(closes, opens)}, Fixed(weights={"X": 1.0}))
    first = res.fills[0]
    # Decided on bar 0 (close 100), filled at bar 1's open (104).
    assert first.ts == res.equity.index[1] and first.price == 104.0
    assert first.qty == pytest.approx(1000.0 / 100.0)
    assert res.equity.iloc[0] == 1000.0
    assert res.equity.iloc[1] == pytest.approx(1000.0 + 10.0 * (110.0 - 104.0))


def test_costs_are_charged():
    frames = {"X": make_bars([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])}
    res = run(frames, Fixed(weights={"X": 1.0}), CostModel(commission_bps=10, slippage_bps=20),
              execution=ExecutionConfig(min_trade_weight=0.01))
    f = res.fills[0]
    assert f.price == pytest.approx(100.2) and f.commission == pytest.approx(10 * 100.2 * 0.001)
    assert res.equity.iloc[-1] == pytest.approx(1000.0 - 10 * 0.2 - f.commission)
    # Reported costs include the slippage paid, not just the commission.
    assert res.metrics["fees"] == pytest.approx(f.commission + 10 * 0.2)
    assert res.metrics["fills"] == 1 and res.metrics["trades"] == 0


def test_short_position_profits_when_price_falls():
    closes = [100.0, 100.0, 90.0, 80.0]
    res = run({"X": make_bars(closes)}, Fixed(weights={"X": -0.5}),
              execution=ExecutionConfig(min_trade_weight=0.4))    # enter, then never rebalance
    assert len(res.fills) == 1 and res.fills[0].qty == pytest.approx(-5.0)
    assert res.equity.iloc[-1] == pytest.approx(1000.0 + 5.0 * 20.0)


class Spy(Strategy):
    """Records what it could see at each call."""
    name = "spy"

    def setup(self):
        self.seen = []

    def target_weights(self, h):
        self.seen.append((h.now, h.size("X"), h.last("X")))
        return {}


def test_strategy_sees_only_the_past():
    frames = {"X": make_bars(np.arange(1.0, 31.0))}
    spy = Spy()
    run(frames, spy)
    assert len(spy.seen) == 30
    for i, (now, size, last) in enumerate(spy.seen):
        assert now == frames["X"].index[i] and size == i + 1 and last == i + 1.0


def test_trade_start_warms_up_without_trading():
    frames = {"X": make_bars(np.linspace(100, 150, 40))}
    start = frames["X"].index[25]
    spy_res = run(frames, Fixed(weights={"X": 1.0}), trade_start=start)
    assert spy_res.equity.index[0] == start and spy_res.equity.iloc[0] == 1000.0
    assert all(f.ts > start for f in spy_res.fills) and spy_res.fills
    assert spy_res.benchmark.iloc[0] == pytest.approx(1000.0)


def test_bankruptcy_stops_the_run():
    closes = [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
    res = run({"X": make_bars(closes)}, Fixed(weights={"X": 3.0}),
              sizing=SizingConfig(max_leverage=3.0),
              risk=RiskConfig(max_position_weight=3.0, max_gross_leverage=3.0))
    assert res.bankrupt and res.equity.iloc[-1] == 0.0 and len(res.equity) == 4
    assert res.metrics["total_return"] == -1.0 and res.metrics["max_drawdown"] == -1.0


def test_drawdown_breaker_flattens_in_backtest():
    closes = np.concatenate([np.full(5, 100.0), np.linspace(100, 60, 20), np.full(10, 60.0)])
    res = run({"X": make_bars(closes)}, Fixed(weights={"X": 1.0}),
              risk=RiskConfig(max_drawdown_pct=0.15))
    assert [e[1] for e in res.risk_events] == ["max_drawdown"]
    assert res.final_positions == {}
    assert -0.22 < res.metrics["max_drawdown"] <= -0.15
    assert res.equity.iloc[-1] == res.equity.iloc[-5]          # flat after the halt


def test_financing_costs():
    frames = {"X": make_bars(np.full(253, 100.0))}
    res = run(frames, Fixed(weights={"X": -1.0}),
              cfg=BacktestConfig(initial_cash=1000.0, borrow_rate=0.10, periods_per_year=252),
              execution=ExecutionConfig(min_trade_weight=0.5))
    # Short 10 units from bar 1 on: 252 bars of borrow at 10%/yr on 1,000 of stock.
    assert len(res.fills) == 1
    assert res.equity.iloc[-1] == pytest.approx(1000.0 - 1000.0 * 0.10 * 252 / 252, rel=1e-6)


def test_symbols_with_different_calendars():
    a = make_bars(np.linspace(100, 120, 30))
    b = make_bars(np.linspace(50, 40, 30)).iloc[::2]            # trades every other day
    c = make_bars(np.linspace(10, 12, 10), start=a.index[20])   # lists late
    res = run({"A": a, "B": b, "C": c}, Fixed(weights={"A": 0.3, "B": 0.3, "C": 0.3}))
    assert len(res.equity) == 30 and np.isfinite(res.equity).all()
    assert set(res.final_positions) == {"A", "B", "C"}
    assert min(f.ts for f in res.fills if f.symbol == "C") == a.index[21]


def test_backtester_is_single_use():
    frames = {"X": make_bars([1.0, 2.0, 3.0])}
    bt = Backtester(frames, Pipeline(Fixed(weights={"X": 1.0})))
    bt.run()
    with pytest.raises(RuntimeError):
        bt.run()


def test_metrics_on_known_curve():
    eq = pd.Series([100.0, 110.0, 99.0, 108.9, 119.79])
    m = compute_metrics(eq, periods_per_year=4)
    assert m["total_return"] == pytest.approx(0.1979)
    assert m["cagr"] == pytest.approx(0.1979)                   # exactly one year of bars
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["max_drawdown_bars"] == 2
    rets = np.array([0.1, -0.1, 0.1, 0.1])
    assert m["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * 2)
    assert m["volatility"] == pytest.approx(rets.std(ddof=1) * 2)
    assert m["calmar"] == pytest.approx(0.1979 / 0.10)
    assert compute_metrics(pd.Series([100.0]))["bars"] == 1


def test_bootstrap_is_reproducible_and_sane():
    rng = np.random.default_rng(2)
    eq = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 800)))
    a, b = bootstrap_sharpe(eq, seed=1), bootstrap_sharpe(eq, seed=1)
    assert a == b and a["sharpe_p05"] < a["sharpe_p50"] < a["sharpe_p95"]
    assert 0.0 <= a["prob_loss"] <= 1.0
    assert bootstrap_sharpe(eq.iloc[:10]) == {}


def test_dataset_slice_and_benchmark():
    a, b = make_bars([10.0, 11.0, 12.1, 12.1]), make_bars([20.0, 20.0, 18.0, 19.8])
    ds = Dataset({"A": a, "B": b})
    bench = ds.benchmark()
    assert bench.iloc[1] == pytest.approx(1.05) and bench.iloc[2] == pytest.approx(1.05 * 1.0)
    part = ds.slice(1, 3)
    assert len(part) == 2 and part.timeline[0] == ds.timeline[1]
    with pytest.raises(ValueError):
        ds.slice(4, 4)
