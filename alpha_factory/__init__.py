# alpha_factory/__init__.py
"""
Alpha Factory System for Market Structure Analysis and Alpha Generation.

This package provides a comprehensive framework for:
- Market structure identification and swing point extraction
- Feature engineering with 220+ technical indicators
- Causal analysis using Granger causality, mutual information, and lead-lag analysis
- Decision making based on market regimes and feature signals

Main Components:
- MarketData: OHLCV data processing and swing point extraction
- Features: Technical indicator extraction (uses existing utils.features_engineering)
- CausalAnalysis: Statistical causality analysis between features
- DecisionMaking: Trading decision logic and risk management
- AlphaFactory: Main orchestrator class
"""

from .market_data import MarketData, SwingPoint
from .causal_analysis import compute_causality, get_top_causal_features, create_causal_network
from .decision_making import (
    decision_function, DecisionSignal, DecisionConfig, 
    MarketRegime, DecisionType, create_decision_summary
)

__version__ = "1.0.0"
__author__ = "Alpha Factory Team"

# Convenience imports
__all__ = [
    'MarketData',
    'SwingPoint',
    'compute_causality',
    'get_top_causal_features',
    'create_causal_network',
    'decision_function',
    'DecisionSignal',
    'DecisionConfig',
    'MarketRegime',
    'DecisionType',
    'create_decision_summary',
]
