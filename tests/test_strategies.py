import math

import numpy as np
import pandas as pd
import pytest

from algobot.config import run_backtest
from algobot.data import Dataset, synthetic_universe
from algobot.history import BarHistory
from algobot.strategies import available, create
from algobot.strategies.base import Strategy, register
from algobot.strategies.reversion import reversion_step
from conftest import make_bars

FREE = {"costs": {"commission_bps": 0, "slippage_bps": 0}}


def spec_for(name):
    if name == "ensemble":
        return {"type": name, "params": {"method": "adaptive", "children": [
            {"type": "ma_cross", "regime": "trend"}, {"type": "bollinger", "regime": "range"},
            {"type": "pairs", "weight": 2}]}}
    return {"type": name}


def history(closes_by_symbol):
    h = BarHistory(list(closes_by_symbol))
    n = max(len(v) for v in closes_by_symbol.values())
    for i in range(n):
        for sym, closes in closes_by_symbol.items():
            if i < len(closes):
                c = float(closes[i])
                h.append(sym, i, c, c * 1.001, c * 0.999, c, 1000.0)
    return h


def walk(strategy, closes_by_symbol):
    """Feed bars one at a time; return the weights after every bar."""
    h = BarHistory(list(closes_by_symbol))
    out = []
    n = max(len(v) for v in closes_by_symbol.values())
    for i in range(n):
        for sym, closes in closes_by_symbol.items():
            c = float(closes[i])
            h.append(sym, i, c, c * 1.001, c * 0.999, c, 1000.0)
        out.append(strategy.target_weights(h))
    return out


def test_registry_lists_the_library():
    names = set(available())
    assert {"ma_cross", "macd", "donchian", "supertrend", "tsmom", "keltner_breakout",
            "bollinger", "rsi_reversion", "xs_momentum", "dual_momentum", "inverse_vol",
            "buy_and_hold", "pairs", "grid", "dca", "ml_logistic", "ensemble"} <= names
    assert all(cls.summary for cls in available().values())


def test_bad_specs_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        create({"type": "nope"})
    with pytest.raises(ValueError, match="unknown parameter"):
        create({"type": "ma_cross", "params": {"fastt": 5}})
    with pytest.raises(ValueError, match="fast < slow"):
        create({"type": "ma_cross", "params": {"fast": 50, "slow": 20}})
    with pytest.raises(ValueError, match="unknown keys"):
        create({"type": "ma_cross", "parms": {}})
    with pytest.raises(ValueError, match="shadows a method"):
        @register
        class Bad(Strategy):
            name = "bad_shadow"
            PARAMS = {"universe": 1}

            def target_weights(self, h):
                return {}


@pytest.mark.parametrize("name", sorted(available()))
def test_every_strategy_runs_and_respects_limits(name, mixed):
    res = run_backtest({"strategy": spec_for(name)}, mixed)
    assert np.isfinite(res.equity).all() and (res.equity > 0).all()
    assert res.weights.abs().sum(axis=1).max() < 1.25        # gross stays near 1 despite gaps
    assert res.exposure.max() < 1.25


@pytest.mark.parametrize("name", sorted(available()))
def test_no_lookahead(name, mixed):
    """Appending future bars must not change anything that already happened."""
    cfg = {"strategy": spec_for(name), "backtest": {"periods_per_year": 252},
           "sizing": {"target_vol": 0.12}, "risk": {"trailing_stop_pct": 0.1}}
    short = run_backtest(cfg, mixed.slice(0, 500))
    full = run_backtest(cfg, mixed)
    np.testing.assert_allclose(short.equity.to_numpy(), full.equity.to_numpy()[:500], rtol=1e-12)
    assert len(short.fills) == sum(f.ts <= short.equity.index[-1] for f in full.fills)


@pytest.mark.parametrize("name", sorted(available()))
def test_deterministic(name, mixed):
    a = run_backtest({"strategy": spec_for(name)}, mixed)
    b = run_backtest({"strategy": spec_for(name)}, mixed)
    assert a.equity.equals(b.equity)


# -- behaviour on markets with known structure ---------------------------------
def test_trend_followers_ride_a_trend_and_flip():
    closes = np.concatenate([100 * 1.004 ** np.arange(200), 100 * 1.004 ** 200 * 0.996 ** np.arange(200)])
    ds = Dataset({"X": make_bars(closes)})
    for name, params in [("ma_cross", {"fast": 10, "slow": 40}), ("donchian", {"entry": 20, "exit": 10}),
                         ("supertrend", {}), ("tsmom", {"lookbacks": [10, 20, 40]}),
                         ("keltner_breakout", {})]:
        res = run_backtest({"strategy": {"type": name, "params": params}, **FREE}, ds)
        assert res.metrics["total_return"] > 0.5, name
        assert res.final_positions["X"] < 0, name               # ends short in the downtrend
        assert res.weights["X"].iloc[150] > 0.9, name           # was long in the uptrend


def test_macd_follows_turns():
    # The histogram measures acceleration, so test it on turns rather than a steady trend.
    closes = np.concatenate([np.linspace(120, 100, 60), np.linspace(100, 130, 30),
                             np.linspace(130, 105, 30)])
    w = [x["X"] for x in walk(create({"type": "macd"}), {"X": closes})]
    assert w[20] == 0                       # still warming up
    assert w[75] == 1 and w[105] == -1      # long after the upturn, short after the downturn


def test_allow_short_false_stays_flat_in_downtrend():
    closes = 100 * 0.996 ** np.arange(200)
    res = run_backtest({"strategy": {"type": "ma_cross", "params": {"fast": 10, "slow": 40,
                                                                    "allow_short": False}}},
                       Dataset({"X": make_bars(closes)}))
    assert not res.fills and res.metrics["total_return"] == 0.0


def test_mean_reversion_profits_on_ranging_market(ranging):
    for name in ("bollinger", "grid"):
        res = run_backtest({"strategy": {"type": name}, **FREE}, ranging)
        assert res.metrics["sharpe"] > 1.0, name
    trend = run_backtest({"strategy": {"type": "ma_cross"}, **FREE}, ranging)
    assert trend.metrics["sharpe"] < 0          # and trend following is the wrong tool here


def test_trend_following_profits_on_trending_market(trending):
    res = run_backtest({"strategy": {"type": "donchian"}, **FREE}, trending)
    assert res.metrics["sharpe"] > 0.5
    fade = run_backtest({"strategy": {"type": "bollinger"}, **FREE}, trending)
    assert fade.metrics["sharpe"] < 0


def test_reversion_state_machine():
    step = lambda s, l, z: reversion_step(s, l, z, 2.0, 0.5, 4.0)
    assert step(0, False, -1.9) == (0, False)
    assert step(0, False, -2.1) == (1, False)       # buy the dip
    assert step(1, False, -1.0) == (1, False)       # hold
    assert step(1, False, -0.4) == (0, False)       # back near the mean: exit
    assert step(0, False, 2.5) == (-1, False)
    assert step(-1, False, 0.4) == (0, False)
    assert step(1, False, -4.2) == (0, True)        # stopped out, locked
    assert step(0, True, -3.0) == (0, True)         # no re-entry while still stretched
    assert step(0, True, 0.2) == (0, False)         # unlocked once normal again


def test_donchian_states():
    closes = [10.0] * 6 + [11.0, 11.5, 11.2, 9.0, 9.5, 10.2, 12.0]
    w = [x["X"] for x in walk(create({"type": "donchian", "params": {"entry": 5, "exit": 2}}),
                              {"X": closes})]
    assert w[5] == 0 and w[6] == 1 and w[7] == 1    # breakout above the 5-bar high
    assert w[8] == 1                                # 11.2 is above the 2-bar low: hold
    assert w[9] == -1                               # new 5-bar low: reverse
    assert w[12] == 1


def test_grid_buys_dips_and_sells_rallies():
    grid = create({"type": "grid", "params": {"step_pct": 0.10, "levels": 2, "mode": "neutral"}})
    prices = [100, 95, 89, 92, 99, 101, 112, 125, 140, 109, 100]
    w = [x["X"] for x in walk(grid, {"X": prices})]
    assert w[:2] == [0, 0]                   # inside the first step: nothing
    assert w[2] == 0.5                       # one line down: buy one level
    assert w[3] == 0.5 and w[4] == 0.5       # no sale until a full step back up
    assert w[5] == 0                         # back through the anchor line: sold, one step profit
    assert w[6] == -0.5 and w[7] == -1.0
    assert w[8] == -1.0                      # exposure is capped at the outer level
    assert w[10] == 0
    long_only = create({"type": "grid", "params": {"step_pct": 0.10, "levels": 2}})
    lw = [x["X"] for x in walk(long_only, {"X": prices})]
    assert lw[0] == 0.5 and lw[2] == 0.75 and min(lw) == 0.0 and max(lw) <= 1.0


def test_cross_sectional_momentum_picks_winners():
    n = 80
    data = {"UP": 100 * 1.01 ** np.arange(n), "FLAT": np.full(n, 100.0),
            "DOWN": 100 * 0.99 ** np.arange(n)}
    lo = create({"type": "xs_momentum", "params": {"lookback": 20, "skip": 2, "top_k": 1}})
    assert lo.target_weights(history(data)) == {"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}
    ls = create({"type": "xs_momentum", "params": {"lookback": 20, "skip": 2, "top_k": 1,
                                                   "long_short": True}})
    assert ls.target_weights(history(data)) == {"UP": 0.5, "FLAT": 0.0, "DOWN": -0.5}


def test_rebalanced_strategies_hold_between_rebalances():
    n = 60
    up, down = 100 * 1.01 ** np.arange(n), 100 * 0.99 ** np.arange(n)
    # Leadership swaps half way through; weights may only change on schedule.
    data = {"A": np.concatenate([up[:30], up[29] * 0.97 ** np.arange(1, 31)]),
            "B": np.concatenate([down[:30], down[29] * 1.03 ** np.arange(1, 31)])}
    strat = create({"type": "xs_momentum", "params": {"lookback": 10, "skip": 0, "top_k": 1,
                                                      "rebalance_every": 15}})
    ws = walk(strat, data)
    changes = [i for i in range(1, n) if ws[i] != ws[i - 1]]
    assert ws[9] == {"A": 0.0, "B": 0.0} and ws[10]["A"] == 1.0   # first bar with enough data
    assert changes == [10, 40] and ws[-1]["B"] == 1.0             # rebalances fall on 10, 25, 40, 55


def test_dual_momentum_goes_to_safety():
    n = 60
    falling = {"A": 100 * 0.99 ** np.arange(n), "B": 100 * 0.995 ** np.arange(n),
               "BOND": np.full(n, 100.0)}
    cash = create({"type": "dual_momentum", "params": {"lookback": 20}})
    assert sum(cash.target_weights(history(falling)).values()) == 0.0
    safe = create({"type": "dual_momentum", "params": {"lookback": 20, "safe_asset": "BOND"}})
    assert safe.target_weights(history(falling)) == {"A": 0.0, "B": 0.0, "BOND": 1.0}
    rising = dict(falling, A=100 * 1.01 ** np.arange(n))
    assert create({"type": "dual_momentum", "params": {"lookback": 20, "safe_asset": "BOND"}}
                  ).target_weights(history(rising))["A"] == 1.0


def test_inverse_vol_weights():
    rng = np.random.default_rng(0)
    data = {"CALM": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 100))),
            "WILD": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))}
    w = create({"type": "inverse_vol"}).target_weights(history(data))
    assert sum(w.values()) == pytest.approx(1.0) and w["CALM"] > 2.5 * w["WILD"]


def test_dca_accumulates_on_schedule_and_buys_dips():
    flat = {"X": np.full(40, 100.0)}
    dca = create({"type": "dca", "params": {"every": 10, "increment": 0.1, "max_weight": 0.25}})
    w = [x["X"] for x in walk(dca, flat)]
    assert w[0] == pytest.approx(0.1) and w[9] == pytest.approx(0.1)
    assert w[10] == pytest.approx(0.2) and w[20] == pytest.approx(0.25) and w[-1] == pytest.approx(0.25)
    dip = {"X": np.concatenate([np.full(30, 100.0), np.full(10, 80.0)])}
    boosted = create({"type": "dca", "params": {"every": 30, "increment": 0.1, "dip_window": 20}})
    assert [x["X"] for x in walk(boosted, dip)][-1] == pytest.approx(0.1 + 0.2)


def test_explicit_anchors_for_live_use():
    late = create({"type": "dca", "params": {"every": 5, "increment": 0.1, "start": "1970-01-01 00:00:00.000000012"}})
    h = BarHistory(["X"])
    seen = []
    for i in range(20):
        h.append("X", pd.Timestamp(i), 100.0, 100.0, 100.0, 100.0, 1.0)
        seen.append(late.target_weights(h)["X"])
    assert seen[11] == 0.0 and seen[12] == pytest.approx(0.1) and seen[17] == pytest.approx(0.2)
    pinned = create({"type": "grid", "params": {"anchor_price": 100.0, "step_pct": 0.1, "levels": 2,
                                                "mode": "neutral"}})
    # Starting below the anchor is already one level long, wherever history begins.
    assert walk(pinned, {"X": [85.0, 86.0]})[0]["X"] == 0.5


def test_pairs_trades_the_spread_market_neutral():
    ds = Dataset(synthetic_universe(["A", "B"], bars=1200, seed=5, kind="gbm", pairs=[["A", "B"]]))
    res = run_backtest({"strategy": {"type": "pairs", "params": {"pairs": [["B", "A"]]}}, **FREE}, ds)
    assert res.metrics["sharpe"] > 1.0 and res.metrics["trades"] > 20
    held = res.weights[(res.weights.abs().sum(axis=1) > 0.5)]
    assert len(held) > 50 and (np.sign(held["A"]) == -np.sign(held["B"])).all()   # always hedged
    assert held.abs().sum(axis=1).max() < 1.2


def test_ml_learns_a_predictable_market(ranging):
    res = run_backtest({"strategy": {"type": "ml_logistic", "params": {"min_train": 150}}, **FREE},
                       ranging)
    assert res.metrics["sharpe"] > 1.0
    # No trading before it has trained on enough labelled samples.
    early = [f for f in res.fills if f.ts < ranging.timeline[64 + 150 // 3]]
    assert not early


def test_ensemble_static_is_the_weighted_sum(mixed):
    kids = [{"type": "ma_cross", "weight": 3}, {"type": "bollinger", "weight": 1}]
    ens = create({"type": "ensemble", "params": {"children": kids}})
    a, b = create(kids[0]), create(kids[1])
    h = BarHistory(mixed.symbols)
    for i, ts in enumerate(mixed.timeline[:300]):
        for sym in mixed.symbols:
            row = mixed.frames[sym].iloc[i]
            h.append(sym, ts, row["open"], row["high"], row["low"], row["close"], row["volume"])
        got, wa, wb = ens.target_weights(h), a.target_weights(h), b.target_weights(h)
        for sym in mixed.symbols:
            assert got[sym] == pytest.approx(0.75 * wa[sym] + 0.25 * wb[sym])


def test_ensemble_adaptive_moves_capital_to_what_works(trending):
    kids = [{"type": "donchian"}, {"type": "bollinger"}]
    ens = create({"type": "ensemble", "params": {"children": kids, "method": "adaptive", "eta": 1.0}})
    h = BarHistory(trending.symbols)
    allocs = []
    for i, ts in enumerate(trending.timeline):
        for sym in trending.symbols:
            row = trending.frames[sym].iloc[i]
            h.append(sym, ts, row["open"], row["high"], row["low"], row["close"], row["volume"])
        ens.target_weights(h)
        allocs.append(ens.allocation.copy())
    allocs = np.array(allocs)
    assert np.allclose(allocs.sum(axis=1), 1.0)
    assert np.allclose(allocs[:20], 0.5)                      # no evidence yet: base weights
    assert allocs[300:, 0].mean() > 0.6                       # trend child earns the capital


def test_ensemble_regime_gate(mixed):
    trend_only = {"type": "ensemble", "params": {"children": [
        {"type": "buy_and_hold", "regime": "trend"}]}}
    range_only = {"type": "ensemble", "params": {"children": [
        {"type": "buy_and_hold", "regime": "range"}]}}
    a, b = run_backtest({"strategy": trend_only}, mixed), run_backtest({"strategy": range_only}, mixed)
    wa, wb = a.weights.to_numpy()[60:], b.weights.to_numpy()[60:]
    assert (wa > 0.05).any() and (wb > 0.05).any()
    # The regimes are exclusive, so the two gates (almost) never hold the same
    # symbol at once; the small overlap is the one bar it takes to trade out.
    assert ((wa > 0.05) & (wb > 0.05)).mean() < 0.05


def test_ensemble_validation():
    with pytest.raises(ValueError, match="at least one child"):
        create({"type": "ensemble"})
    with pytest.raises(ValueError, match="regime"):
        create({"type": "ensemble", "params": {"children": [{"type": "macd", "regime": "bull"}]}})
    with pytest.raises(ValueError, match="method"):
        create({"type": "ensemble", "params": {"children": [{"type": "macd"}], "method": "magic"}})
