import numpy as np
import pandas as pd
import pytest

from algobot.history import BarHistory
from algobot.risk import RiskConfig, RiskManager
from algobot.sizing import Sizer, SizingConfig
from algobot.types import Position

T0 = pd.Timestamp("2024-01-02")


def apply(rm, targets, equity=1000.0, positions=None, prices=None, now=T0):
    return rm.apply(targets, equity=equity, positions=positions or {}, prices=prices or {}, now=now)


def test_caps_and_long_only():
    rm = RiskManager(RiskConfig(max_position_weight=0.3, max_gross_leverage=0.5, long_only=True))
    out = apply(rm, {"A": 0.6, "B": 0.25, "C": -0.4, "D": float("nan")})
    assert out == pytest.approx({"A": 0.3 * 0.5 / 0.55, "B": 0.25 * 0.5 / 0.55, "C": 0.0, "D": 0.0})
    assert sum(abs(w) for w in out.values()) == pytest.approx(0.5)


def test_stop_loss_then_cooldown():
    rm = RiskManager(RiskConfig(stop_loss_pct=0.10, cooldown_bars=3))
    pos = {"A": Position(10, 100.0)}
    assert apply(rm, {"A": 1.0}, positions=pos, prices={"A": 95.0})["A"] == 1.0
    assert apply(rm, {"A": 1.0}, positions=pos, prices={"A": 89.0})["A"] == 0.0     # stop fires
    assert rm.events[-1][1] == "stop_loss"
    flat = {"A": Position(0, 0.0)}
    assert apply(rm, {"A": 1.0}, positions=flat, prices={"A": 89.0})["A"] == 0.0    # cooling down
    assert apply(rm, {"A": 1.0}, positions=flat, prices={"A": 89.0})["A"] == 0.0
    assert apply(rm, {"A": 1.0}, positions=flat, prices={"A": 89.0})["A"] == 1.0    # allowed again


def test_take_profit_and_short_stop():
    rm = RiskManager(RiskConfig(take_profit_pct=0.2, stop_loss_pct=0.1, cooldown_bars=1))
    assert apply(rm, {"A": 1.0}, positions={"A": Position(5, 100.0)}, prices={"A": 121.0})["A"] == 0.0
    assert rm.events[-1][1] == "take_profit"
    short = {"B": Position(-5, 100.0)}
    assert apply(rm, {"B": -1.0}, positions=short, prices={"B": 105.0})["B"] == -1.0
    assert apply(rm, {"B": -1.0}, positions=short, prices={"B": 111.0})["B"] == 0.0  # short loses as price rises
    assert rm.events[-1][1] == "stop_loss"


def test_trailing_stop_tracks_the_best_price():
    rm = RiskManager(RiskConfig(trailing_stop_pct=0.10, cooldown_bars=1))
    pos = {"A": Position(10, 100.0)}
    for price in (105.0, 120.0, 112.0):
        assert apply(rm, {"A": 1.0}, positions=pos, prices={"A": price})["A"] == 1.0
    assert apply(rm, {"A": 1.0}, positions=pos, prices={"A": 107.0})["A"] == 0.0     # 10.8% off the 120 high
    assert rm.events[-1][1] == "trailing_stop"
    # A fresh position starts tracking from its own entry, not the old high.
    apply(rm, {"A": 1.0}, positions={"A": Position(0, 0.0)}, prices={"A": 107.0})
    fresh = {"A": Position(10, 107.0)}
    assert apply(rm, {"A": 1.0}, positions=fresh, prices={"A": 100.0})["A"] == 1.0


def test_drawdown_breaker_halts_and_resumes():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.2, drawdown_halt_bars=2))
    pos = {"A": Position(1, 1.0)}
    assert apply(rm, {"A": 1.0}, equity=1000.0)["A"] == 1.0
    assert apply(rm, {"A": 1.0}, equity=850.0)["A"] == 1.0
    assert apply(rm, {"A": 1.0}, equity=790.0, positions=pos) == {"A": 0.0}          # tripped
    assert apply(rm, {"A": 1.0}, equity=790.0) == {"A": 0.0}
    assert apply(rm, {"A": 1.0}, equity=790.0)["A"] == 1.0                           # resumed
    assert rm.peak_equity == 790.0                                                    # measured afresh
    assert [e[1] for e in rm.events] == ["max_drawdown", "resume"]


def test_drawdown_breaker_is_permanent_by_default_and_survives_restart():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.1))
    apply(rm, {"A": 1.0}, equity=1000.0)
    apply(rm, {"A": 1.0}, equity=880.0)
    for _ in range(50):
        assert apply(rm, {"A": 1.0}, equity=2000.0) == {"A": 0.0}
    restarted = RiskManager(RiskConfig(max_drawdown_pct=0.1))
    restarted.load_state(rm.state_dict())
    assert apply(restarted, {"A": 1.0}, equity=2000.0) == {"A": 0.0}


def test_daily_loss_limit_resets_next_day():
    rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.05))
    day1, day2 = pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-03 10:00")
    assert apply(rm, {"A": 1.0}, equity=1000.0, now=day1)["A"] == 1.0
    assert apply(rm, {"A": 1.0}, equity=940.0, now=day1 + pd.Timedelta(hours=1)) == {"A": 0.0}
    assert apply(rm, {"A": 1.0}, equity=990.0, now=day1 + pd.Timedelta(hours=2)) == {"A": 0.0}
    assert apply(rm, {"A": 1.0}, equity=985.0, now=day2)["A"] == 1.0                 # new day, small move


def test_daily_loss_limit_works_on_daily_bars():
    rm = RiskManager(RiskConfig(daily_loss_limit_pct=0.05))
    days = pd.date_range("2024-01-02", periods=4, freq="B")
    assert apply(rm, {"A": 1.0}, equity=1000.0, now=days[0])["A"] == 1.0
    assert apply(rm, {"A": 1.0}, equity=930.0, now=days[1]) == {"A": 0.0}             # -7% on the day
    assert apply(rm, {"A": 1.0}, equity=930.0, now=days[2])["A"] == 1.0               # back the next day
    assert apply(rm, {"A": 1.0}, equity=900.0, now=days[3])["A"] == 1.0               # -3.2%: within limit


def test_clear_halt():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.1))
    apply(rm, {"A": 1.0}, equity=1000.0)
    assert apply(rm, {"A": 1.0}, equity=800.0) == {"A": 0.0}
    rm.clear_halt()
    assert apply(rm, {"A": 1.0}, equity=800.0)["A"] == 1.0 and rm.peak_equity == 800.0


def history_with_vol(daily_vols, n=200, seed=0):
    rng = np.random.default_rng(seed)
    h = BarHistory(list(daily_vols))
    for sym, vol in daily_vols.items():
        px = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
        for i, p in enumerate(px):
            h.append(sym, i, p, p, p, p, 1.0)
    return h


def test_vol_target_scales_to_target():
    h = history_with_vol({"A": 0.02})
    sizer = Sizer(SizingConfig(target_vol=0.10, vol_window=60, max_leverage=5.0, periods_per_year=252))
    w = sizer.apply({"A": 1.0}, h)["A"]
    realised = np.diff(np.log(h.close("A", 61))).std(ddof=1) * np.sqrt(252)
    assert w == pytest.approx(0.10 / realised)
    # The leverage cap wins when the target would need more.
    capped = Sizer(SizingConfig(target_vol=0.9, vol_window=60, max_leverage=1.5, periods_per_year=252))
    assert capped.apply({"A": 1.0}, h)["A"] == pytest.approx(1.5)


def test_vol_target_uses_correlation():
    h = history_with_vol({"A": 0.01, "B": 0.01})
    sizer = Sizer(SizingConfig(target_vol=0.10, vol_window=100, max_leverage=50.0, periods_per_year=252))
    both = sizer.apply({"A": 0.5, "B": 0.5}, h)
    single = sizer.apply({"A": 1.0}, h)
    # Two independent assets diversify, so the pair can carry more gross exposure.
    assert sum(both.values()) > single["A"] * 1.2


def test_sizer_passthrough_and_edge_cases():
    h = history_with_vol({"A": 0.01}, n=10)
    assert Sizer().apply({"A": 0.4, "B": -0.2}, h) == {"A": 0.4, "B": -0.2}
    assert Sizer(SizingConfig(max_leverage=0.3)).apply({"A": 0.4, "B": -0.2}, h) == pytest.approx(
        {"A": 0.2, "B": -0.1})
    # Not enough history to estimate volatility: leave the weights alone.
    assert Sizer(SizingConfig(target_vol=0.1, vol_window=40)).apply({"A": 0.5}, h) == {"A": 0.5}
    assert Sizer(SizingConfig(target_vol=0.1)).apply({"A": float("nan")}, h) == {}
