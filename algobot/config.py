"""Turning a config file into runnable objects."""
from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .backtest.engine import BacktestConfig, BacktestResult, Backtester
from .data import Dataset, load_data
from .execution.orders import CostModel, ExecutionConfig
from .pipeline import Pipeline
from .risk import RiskConfig, RiskManager
from .sizing import Sizer, SizingConfig
from .strategies import create

SECTIONS = {"name", "data", "strategy", "sizing", "risk", "execution", "costs",
            "backtest", "optimize", "walk_forward", "live"}


def validate(cfg: Mapping[str, Any]) -> None:
    if not isinstance(cfg, Mapping):
        raise ValueError("config must be a mapping")
    unknown = set(cfg) - SECTIONS
    if unknown:
        raise ValueError(f"unknown config section(s) {sorted(unknown)}; valid: {sorted(SECTIONS)}")
    if "strategy" not in cfg:
        raise ValueError("config needs a 'strategy' section")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    validate(cfg)
    return cfg


def _section(cls, values: Mapping[str, Any] | None, name: str):
    values = dict(values or {})
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"{name}: unknown option(s) {sorted(unknown)}; valid: {sorted(valid)}")
    return cls(**values)


def build_pipeline(cfg: Mapping[str, Any]) -> Pipeline:
    """A fresh pipeline (fresh strategy and risk state) for a config."""
    return Pipeline(
        strategy=create(cfg["strategy"]),
        sizer=Sizer(_section(SizingConfig, cfg.get("sizing"), "sizing")),
        risk=RiskManager(_section(RiskConfig, cfg.get("risk"), "risk")),
        execution=_section(ExecutionConfig, cfg.get("execution"), "execution"),
    )


def build_costs(cfg: Mapping[str, Any]) -> CostModel:
    return _section(CostModel, cfg.get("costs"), "costs")


def load_dataset(cfg: Mapping[str, Any]) -> Dataset:
    return Dataset(load_data(cfg.get("data") or {}))


def run_backtest(cfg: Mapping[str, Any], dataset: Dataset | None = None,
                 trade_start: Any = None) -> BacktestResult:
    dataset = dataset if dataset is not None else load_dataset(cfg)
    bt = Backtester(dataset, build_pipeline(cfg), build_costs(cfg),
                    _section(BacktestConfig, cfg.get("backtest"), "backtest"))
    return bt.run(trade_start=trade_start)


def set_path(cfg: Any, path: str, value: Any) -> None:
    """Set ``cfg["a"]["b"][0]["c"]`` from the dotted path ``"a.b.0.c"``."""
    keys = path.split(".")
    node = cfg
    for i, key in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(node, list):
            idx = int(key)
            if last:
                node[idx] = value
            else:
                node = node[idx]
        elif last:
            node[key] = value
        else:
            if node.get(key) is None:
                node[key] = {}
            node = node[key]


def with_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    for path, value in overrides.items():
        set_path(out, path, value)
    return out
