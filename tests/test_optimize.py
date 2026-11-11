import numpy as np
import pandas as pd
import pytest

from algobot.backtest.optimize import (best_params, expand_grid, optimize, sample_space,
                                       walk_forward)
from algobot.config import run_backtest, set_path, with_overrides

CFG = {"strategy": {"type": "bollinger", "params": {"window": 20}},
       "costs": {"commission_bps": 0, "slippage_bps": 0}}
SPACE = {"strategy.params.window": [10, 20, 40], "strategy.params.entry_z": [1.5, 2.5]}


def test_grid_and_random_spaces():
    grid = expand_grid(SPACE)
    assert len(grid) == 6 and {"strategy.params.window": 40, "strategy.params.entry_z": 1.5} in grid
    draws = sample_space({"a": [1, 2, 3], "b": {"low": 5, "high": 9, "type": "int"},
                          "c": {"low": 0.1, "high": 0.2}, "d": {"low": 1e-4, "high": 1e-1, "type": "log"}},
                         50, seed=3)
    assert len(draws) == 50 and draws == sample_space(
        {"a": [1, 2, 3], "b": {"low": 5, "high": 9, "type": "int"}, "c": {"low": 0.1, "high": 0.2},
         "d": {"low": 1e-4, "high": 1e-1, "type": "log"}}, 50, seed=3)
    assert all(d["a"] in (1, 2, 3) and 5 <= d["b"] <= 9 and isinstance(d["b"], int)
               and 0.1 <= d["c"] <= 0.2 and 1e-4 <= d["d"] <= 1e-1 for d in draws)
    assert {d["b"] for d in draws} == {5, 6, 7, 8, 9}


def test_set_path_reaches_nested_lists():
    cfg = {"strategy": {"type": "ensemble", "params": {"children": [{"type": "macd"},
                                                                    {"type": "bollinger"}]}}}
    out = with_overrides(cfg, {"strategy.params.children.1.params.window": 30,
                               "risk.stop_loss_pct": 0.05})
    assert out["strategy"]["params"]["children"][1]["params"]["window"] == 30
    assert out["risk"]["stop_loss_pct"] == 0.05
    assert "risk" not in cfg and "params" not in cfg["strategy"]["params"]["children"][1]
    set_path(cfg, "strategy.params.children.0", {"type": "tsmom"})
    assert cfg["strategy"]["params"]["children"][0] == {"type": "tsmom"}


def test_optimize_ranks_candidates_and_matches_a_direct_run(ranging):
    table = optimize(CFG, ranging, SPACE, metric="sharpe")
    assert len(table) == 6 and table["score"].is_monotonic_decreasing
    best = best_params(table, SPACE)
    direct = run_backtest(with_overrides(CFG, best), ranging)
    assert direct.metrics["sharpe"] == pytest.approx(table.loc[0, "sharpe"])
    assert isinstance(best["strategy.params.window"], int)


def test_optimize_parallel_matches_serial(ranging):
    serial = optimize(CFG, ranging, SPACE)
    parallel = optimize(CFG, ranging, SPACE, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_optimize_survives_invalid_combinations(ranging):
    cfg = {"strategy": {"type": "ma_cross"}}
    table = optimize(cfg, ranging, {"strategy.params.fast": [10, 300], "strategy.params.slow": [50]})
    assert len(table) == 2 and table.loc[0, "strategy.params.fast"] == 10
    assert "fast < slow" in table.loc[1, "error"] and np.isnan(table.loc[1, "score"])
    with pytest.raises(ValueError):
        optimize(cfg, ranging, {"strategy.params.fast": [300], "strategy.params.slow": [50]})


def test_min_trades_filter_and_random_search(ranging):
    table = optimize(CFG, ranging, SPACE, min_trades=10_000)
    assert table["score"].isna().all()
    with pytest.raises(ValueError, match="no valid candidate"):
        best_params(table, SPACE)
    rnd = optimize(CFG, ranging, {"strategy.params.window": {"low": 10, "high": 40, "type": "int"}},
                   method="random", n_iter=5, seed=1)
    assert len(rnd) == 5


def test_walk_forward_is_out_of_sample_only(ranging):
    wf = walk_forward(CFG, ranging, SPACE, train_bars=300, test_bars=150)
    assert len(wf.folds) == 4                                # (900 - 300) / 150
    assert wf.equity.index[0] == ranging.timeline[300]
    assert wf.equity.index[-1] == ranging.timeline[-1]
    assert wf.equity.index.is_unique and wf.equity.index.is_monotonic_increasing
    assert len(wf.equity) == 600 and wf.equity.iloc[0] == 100_000.0
    assert wf.metrics["sharpe"] > 0.5                        # the edge is real on this market
    # Fold 1 must equal a hand-rolled run: best params from bars 0-299, traded on 300-449.
    table = optimize(CFG, ranging.slice(0, 300), SPACE)
    params = best_params(table, SPACE)
    manual = run_backtest(with_overrides(CFG, params), ranging.slice(0, 450),
                          trade_start=ranging.timeline[300])
    assert wf.folds.loc[0, "strategy.params.window"] == params["strategy.params.window"]
    np.testing.assert_allclose(wf.equity.to_numpy()[:150], manual.equity.to_numpy())


def test_walk_forward_needs_enough_data(ranging):
    with pytest.raises(ValueError, match="need at least"):
        walk_forward(CFG, ranging, SPACE, train_bars=800, test_bars=200)
