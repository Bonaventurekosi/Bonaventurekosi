"""Market data: datasets, synthetic generators, file and network loaders."""
from __future__ import annotations

import os
from typing import Any, Mapping

import pandas as pd

from .dataset import Dataset, clean_frame, infer_periods_per_year
from .loaders import AlpacaData, CCXTData, YahooData, load_csv, load_csv_dir
from .synthetic import synthetic_universe

__all__ = ["Dataset", "clean_frame", "infer_periods_per_year", "load_data",
           "make_provider", "synthetic_universe", "load_csv", "load_csv_dir",
           "YahooData", "AlpacaData", "CCXTData"]

_COMMON = {"source", "symbols", "start", "end"}
_OPTIONS = {
    "synthetic": {"bars", "seed", "kind", "freq", "pairs"},
    "csv": {"path", "adjust"},
    "yahoo": {"interval", "range"},
    "alpaca": {"timeframe", "feed", "lookback_days"},
    "ccxt": {"exchange", "timeframe", "limit"},
}


def _check(spec: Mapping[str, Any]) -> str:
    source = spec.get("source", "synthetic")
    if source not in _OPTIONS:
        raise ValueError(f"data.source must be one of {sorted(_OPTIONS)}, got {source!r}")
    unknown = set(spec) - _COMMON - _OPTIONS[source]
    if unknown:
        raise ValueError(f"data: unknown option(s) {sorted(unknown)} for source {source!r}")
    return source


def make_provider(spec: Mapping[str, Any]):
    """A network provider with ``fetch(symbols, bars)`` for the given data spec."""
    source = _check(spec)
    opts = {k: spec[k] for k in _OPTIONS[source] if k in spec}
    if source == "yahoo":
        return YahooData(**opts)
    if source == "alpaca":
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_API_SECRET")
        return AlpacaData(key, secret, **opts)
    if source == "ccxt":
        return CCXTData(**opts)
    raise ValueError(f"data source {source!r} is not a live provider")


def load_data(spec: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Bars for a config's ``data`` section, as ``{symbol: DataFrame}``."""
    source = _check(spec)
    symbols = spec.get("symbols")
    if source == "synthetic":
        if not symbols:
            raise ValueError("data.symbols is required for synthetic data")
        opts = {k: spec[k] for k in _OPTIONS[source] if k in spec}
        if "start" in spec:
            opts["start"] = str(spec["start"])
        frames = synthetic_universe(symbols, **opts)
    elif source == "csv":
        frames = load_csv_dir(spec["path"], symbols, spec.get("adjust", True))
    else:
        if not symbols:
            raise ValueError(f"data.symbols is required for source {source!r}")
        frames = make_provider(spec).fetch(symbols)
    if source != "synthetic" and spec.get("start") is not None:
        frames = {s: df.loc[str(spec["start"]):] for s, df in frames.items()}
    if spec.get("end") is not None:
        frames = {s: df.loc[:str(spec["end"])] for s, df in frames.items()}
    return frames
