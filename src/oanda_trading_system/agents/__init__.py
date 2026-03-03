"""Compatibility aliases for legacy `agents` package."""

from importlib import import_module
import sys

_SUBPACKAGES = ["execution", "market_data", "monitoring", "risk", "strategy"]
for _name in _SUBPACKAGES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"agents.{_name}")

__all__ = _SUBPACKAGES
