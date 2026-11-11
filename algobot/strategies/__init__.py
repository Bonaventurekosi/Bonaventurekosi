"""Strategy library. Importing this package registers every built-in strategy."""
from .base import SignalStrategy, Strategy, available, create, register
from . import (accumulation, cross_sectional, ensemble, event_driven, ml, pairs,  # noqa: F401
               reversion, trend)

__all__ = ["Strategy", "SignalStrategy", "available", "create", "register"]
