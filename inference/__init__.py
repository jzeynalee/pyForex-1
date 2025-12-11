# inference/__init__.py
"""
Inference module with risk-aware predictions.
"""

from .predictor import (
    RiskAwareTCNPredictor,
    HybridPredictor,
    PredictionResult,
    PredictorConfig,
    Signal,
    create_predictor,
    # Backward compatibility aliases
    TCNPredictor,
    SimpleLSTMPredictor
)

__all__ = [
    'RiskAwareTCNPredictor',
    'HybridPredictor',
    'PredictionResult',
    'PredictorConfig',
    'Signal',
    'create_predictor',
    'TCNPredictor',
    'SimpleLSTMPredictor'
]
