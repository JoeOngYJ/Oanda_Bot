"""Compatibility aliases for legacy `backtesting` package."""

from importlib import import_module
import sys

_SUBPACKAGES = [
    "analysis",
    "core",
    "data",
    "execution",
    "features",
    "strategy",
    "utils",
    "visualization",
]
for _name in _SUBPACKAGES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"backtesting.{_name}")

__all__ = _SUBPACKAGES
