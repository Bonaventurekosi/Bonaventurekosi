"""Strategy interface and registry.

A strategy looks at a ``BarHistory`` and answers one question: what fraction of
the capital should sit in each symbol right now? Positive is long, negative is
short, and the absolute weights of one strategy never sum to more than 1.
Sizing, risk limits and order generation happen downstream, which is what lets
any strategy be combined with any other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence

from ..history import BarHistory

_REGISTRY: dict[str, type["Strategy"]] = {}


def register(cls: type["Strategy"]) -> type["Strategy"]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name")
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate strategy name {cls.name!r}")
    # Parameters become attributes, so they must not shadow a method.
    for klass in cls.__mro__:
        for key in vars(klass).get("PARAMS", {}):
            if callable(getattr(cls, key, None)):
                raise ValueError(f"{cls.__name__}: parameter {key!r} shadows a method")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type["Strategy"]]:
    return dict(sorted(_REGISTRY.items()))


def create(spec: Mapping[str, Any] | str, symbols: Sequence[str] | None = None) -> "Strategy":
    """Build a strategy from ``{"type": name, "params": {...}, "symbols": [...]}``."""
    if isinstance(spec, str):
        spec = {"type": spec}
    kind = spec.get("type")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown strategy {kind!r}; choose from {sorted(_REGISTRY)}")
    unknown = set(spec) - {"type", "params", "symbols", "weight", "regime"}
    if unknown:
        raise ValueError(f"unknown keys in strategy spec for {kind!r}: {sorted(unknown)}")
    return _REGISTRY[kind](symbols=spec.get("symbols") or symbols,
                           **dict(spec.get("params") or {}))


class Strategy(ABC):
    name: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    PARAMS: ClassVar[dict[str, Any]] = {}

    def __init__(self, symbols: Sequence[str] | None = None, **params: Any):
        merged: dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            merged.update(vars(klass).get("PARAMS", {}))
        unknown = set(params) - set(merged)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                f"valid: {sorted(merged)}")
        merged.update(params)
        self.params = merged
        for key, value in merged.items():
            setattr(self, key, value)
        self.symbols = list(symbols) if symbols else None
        self.setup()

    def setup(self) -> None:
        """Validate parameters and create state. Called once from ``__init__``."""

    def universe(self, h: BarHistory) -> list[str]:
        wanted = self.symbols if self.symbols is not None else h.symbols
        return [s for s in wanted if s in h]

    @abstractmethod
    def target_weights(self, h: BarHistory) -> dict[str, float]:
        """Desired weight per symbol given everything known up to ``h.now``.

        Called exactly once per bar, in time order.
        """


class SignalStrategy(Strategy):
    """One signal in [-1, 1] per symbol, with capital split equally."""

    PARAMS: ClassVar[dict[str, Any]] = {"allow_short": True}

    def min_bars(self) -> int:
        return 1

    @abstractmethod
    def signal(self, sym: str, h: BarHistory) -> float:
        ...

    def target_weights(self, h: BarHistory) -> dict[str, float]:
        syms = self.universe(h)
        if not syms:
            return {}
        share = 1.0 / len(syms)
        need = self.min_bars()
        out = {}
        for sym in syms:
            sig = self.signal(sym, h) if h.size(sym) >= need else 0.0
            if sig < 0 and not self.allow_short:
                sig = 0.0
            out[sym] = max(-1.0, min(1.0, float(sig))) * share
        return out
