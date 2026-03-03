"""Compatibility aliases for legacy `shared` package."""

from importlib import import_module
import sys

_SUBMODULES = ["config", "logging_config", "message_bus", "models", "utils"]
for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"shared.{_name}")

__all__ = _SUBMODULES
