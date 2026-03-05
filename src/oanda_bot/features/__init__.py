"""Feature engineering package."""

from .feature_builder import FeatureBuilder
from .cost_model import CostModel, SpreadTable
from .labels import make_labels
from .torch_dataset import FeatureLabelDataset, feature_label_collate_fn
from .regime_mlp import RegimeMLP, RegimeTargetConfig, make_regime_targets, train_regime_mlp, predict_regime_proba
from .opportunity_model import SharedTCNEncoder, OpportunityModel, train_opportunity_model
from .direction_model import DirectionModel, train_direction_model
from .breakout_model import BreakoutModel, make_breakout_labels, train_breakout_model
from .mean_reversion_model import MeanReversionModel, make_mean_reversion_labels, train_mean_reversion_model
from .risk_model import RiskModel, make_risk_labels, train_risk_model

__all__ = [
    "indicators",
    "structure",
    "compute",
    "FeatureBuilder",
    "CostModel",
    "SpreadTable",
    "make_labels",
    "FeatureLabelDataset",
    "feature_label_collate_fn",
    "RegimeMLP",
    "RegimeTargetConfig",
    "make_regime_targets",
    "train_regime_mlp",
    "predict_regime_proba",
    "SharedTCNEncoder",
    "OpportunityModel",
    "train_opportunity_model",
    "DirectionModel",
    "train_direction_model",
    "BreakoutModel",
    "make_breakout_labels",
    "train_breakout_model",
    "MeanReversionModel",
    "make_mean_reversion_labels",
    "train_mean_reversion_model",
    "RiskModel",
    "make_risk_labels",
    "train_risk_model",
]
