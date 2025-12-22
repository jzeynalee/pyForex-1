"""
Phase 1: Predictive Foundation

Multi-head TCN backbone for risk management predictions:
- Direction probabilities
- Volatility forecasts
- Quantile predictions for price distribution
 - Trade outcome probabilities (TP hit before SL within horizon): p_long, p_short
"""

from .tcn_backbone import (
    TCNConfig,
    TradingProfile,
    MultiHeadTCN,
    RiskPrediction,
    create_tcn_for_profile
)

from .training import (
    TrainingConfig,
    RiskDataset,
    MultiHeadTCNTrainer,
    DirectionLoss,
    VolatilityLoss,
    QuantileLoss,
    MultiTaskLoss,
    compute_metrics
)

__all__ = [
    # Model
    'TCNConfig',
    'TradingProfile', 
    'MultiHeadTCN',
    'RiskPrediction',
    'create_tcn_for_profile',
    # Training
    'TrainingConfig',
    'RiskDataset',
    'MultiHeadTCNTrainer',
    'DirectionLoss',
    'VolatilityLoss',
    'QuantileLoss',
    'MultiTaskLoss',
    'compute_metrics'
]
