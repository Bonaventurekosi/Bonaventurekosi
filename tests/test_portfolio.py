import pytest

from algobot.execution.orders import CostModel, ExecutionConfig, build_orders
from algobot.portfolio import Portfolio
from algobot.types import Fill


def fill(pf, sym, qty, price, fee=0.0, ts=0):
    pf.apply_fill(Fill(ts, sym, qty, price, fee))


def test_long_round_trip():
    pf = Portfolio(1000.0)
    fill(pf, "X", 10, 10.0, fee=1.0, ts=1)
    assert pf.cash == pytest.approx(899.0) and pf.qty("X") == 10
    fill(pf, "X", 10, 20.0, ts=2)                       # add: average price 15
    assert pf.positions["X"].avg_price == pytest.approx(15.0)
    fill(pf, "X", -5, 18.0, ts=3)                       # partial exit: +15
    assert pf.realized_pnl == pytest.approx(15.0) and not pf.trades
    fill(pf, "X", -15, 12.0, fee=2.0, ts=4)             # close: -45
    assert pf.realized_pnl == pytest.approx(-30.0)
    assert pf.qty("X") == 0 and pf.positions["X"].avg_price == 0
    (trade,) = pf.trades
    assert (trade.side, trade.entry_ts, trade.exit_ts, trade.max_qty) == (1, 1, 4, 20)
    assert trade.gross_pnl == pytest.approx(-30.0) and trade.fees == pytest.approx(3.0)
    assert trade.pnl == pytest.approx(-33.0)
    # Cash must reconcile exactly with realised P&L and fees.
    assert pf.cash == pytest.approx(1000.0 + pf.realized_pnl - pf.fees_paid)


def test_short_and_flip():
    pf = Portfolio(1000.0)
    fill(pf, "X", -10, 50.0)
    assert pf.cash == pytest.approx(1500.0)
    assert pf.equity({"X": 50.0}) == pytest.approx(1000.0)
    assert pf.equity({"X": 40.0}) == pytest.approx(1100.0)   # short profits as price falls
    fill(pf, "X", 25, 40.0)                                   # cover 10 (+100) and go long 15
    assert pf.realized_pnl == pytest.approx(100.0)
    assert pf.qty("X") == 15 and pf.positions["X"].avg_price == pytest.approx(40.0)
    assert len(pf.trades) == 1 and pf.trades[0].side == -1
    assert pf.equity({"X": 40.0}) == pytest.approx(1100.0)
    assert pf.weights({"X": 40.0})["X"] == pytest.approx(600.0 / 1100.0)


def test_persistence_round_trip():
    pf = Portfolio(500.0)
    fill(pf, "A", 3, 10.0, fee=0.5)
    fill(pf, "B", -2, 20.0)
    clone = Portfolio.from_dict(pf.to_dict())
    prices = {"A": 11.0, "B": 19.0}
    assert clone.equity(prices) == pytest.approx(pf.equity(prices))
    assert clone.quantities() == pf.quantities()


def test_cost_model():
    costs = CostModel(commission_bps=10, slippage_bps=5, min_commission=1.0)
    assert costs.fill_price(100.0, 1) == pytest.approx(100.05)
    assert costs.fill_price(100.0, -1) == pytest.approx(99.95)
    assert costs.commission(100, 50.0) == pytest.approx(5.0)
    assert costs.commission(1, 50.0) == pytest.approx(1.0)      # minimum applies
    assert costs.commission(0, 50.0) == 0.0


def test_build_orders_basic_and_ordering():
    cfg = ExecutionConfig(min_trade_weight=0.0)
    orders = build_orders({"A": 0.5, "B": 0.0, "C": -0.25}, {"B": 10.0}, 1000.0,
                          {"A": 10.0, "B": 20.0, "C": 5.0}, cfg)
    got = {o.symbol: o.qty for o in orders}
    assert got == pytest.approx({"A": 50.0, "B": -10.0, "C": -50.0})
    assert [o.symbol for o in orders][-1] == "A"                # buys after sells
    assert next(o for o in orders if o.symbol == "B").reason == "exit"


def test_build_orders_thresholds_and_rounding():
    prices = {"A": 10.0}
    cfg = ExecutionConfig(min_trade_weight=0.01)
    assert build_orders({"A": 0.505}, {"A": 50.0}, 1000.0, prices, cfg) == []    # 0.5% drift ignored
    assert build_orders({"A": 0.52}, {"A": 50.0}, 1000.0, prices, cfg)[0].qty == pytest.approx(2.0)
    # A full exit always goes through, however small.
    assert build_orders({"A": 0.0}, {"A": 0.3}, 1000.0, prices, cfg)[0].qty == pytest.approx(-0.3)
    # Held symbols missing from the targets are sold.
    assert build_orders({}, {"A": 5.0}, 1000.0, prices, cfg)[0].qty == pytest.approx(-5.0)
    whole = ExecutionConfig(min_trade_weight=0.0, fractional=False)
    assert build_orders({"A": 0.257}, {}, 1000.0, prices, whole)[0].qty == 25.0
    assert build_orders({"A": -0.257}, {}, 1000.0, prices, whole)[0].qty == -25.0


def test_build_orders_caps_and_bad_inputs():
    prices = {"A": 10.0, "B": float("nan")}
    capped = ExecutionConfig(min_trade_weight=0.0, max_participation=0.1)
    assert build_orders({"A": 1.0}, {}, 1000.0, prices, capped, {"A": 200.0})[0].qty == pytest.approx(20.0)
    notional = ExecutionConfig(min_trade_weight=0.0, max_order_weight=0.2)
    assert build_orders({"A": 1.0}, {}, 1000.0, prices, notional)[0].qty == pytest.approx(20.0)
    plain = ExecutionConfig(min_trade_weight=0.0)
    assert build_orders({"B": 0.5}, {}, 1000.0, prices, plain) == []             # no price, no order
    assert build_orders({"A": float("nan")}, {}, 1000.0, prices, plain) == []
    assert build_orders({"A": 0.5}, {}, 0.0, prices, plain) == []                # no equity, no order
