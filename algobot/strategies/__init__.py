"""Strategy library. Importing this package registers every built-in strategy."""
from .base import SignalStrategy, Strategy, available, create, register
from . import accumulation, cross_sectional, ensemble, ml, pairs, reversion, trend  # noqa: F401

__all__ = ["Strategy", "SignalStrategy", "available", "create", "register"]
