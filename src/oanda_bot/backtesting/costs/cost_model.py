from __future__ import annotations

# Canonical location for stage-2 cost modeling.
# Re-export implementation currently used in features pipeline.
from oanda_bot.features.cost_model import CostModel, SpreadTable

__all__ = ["CostModel", "SpreadTable"]
