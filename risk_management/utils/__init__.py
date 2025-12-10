"""
Risk Management Utilities

Common functions used across all phases:
- Technical indicators
- Regime detection
- Data preprocessing
- Performance metrics
"""

from .indicators import (
    # Technical Indicators
    calculate_atr,
    calculate_volatility,
    calculate_adx,
    calculate_rsi,
    calculate_bollinger_bands,
    # Regime Detection
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    # Preprocessing
    create_direction_labels,
    create_volatility_labels,
    create_price_move_labels,
    normalize_features,
    apply_normalization,
    # Performance Metrics
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy,
    PerformanceReport,
    generate_performance_report
)

__all__ = [
    # Technical Indicators
    'calculate_atr',
    'calculate_volatility',
    'calculate_adx',
    'calculate_rsi',
    'calculate_bollinger_bands',
    # Regime Detection
    'MarketRegime',
    'RegimeConfig',
    'RegimeDetector',
    # Preprocessing
    'create_direction_labels',
    'create_volatility_labels',
    'create_price_move_labels',
    'normalize_features',
    'apply_normalization',
    # Performance Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_expectancy',
    'PerformanceReport',
    'generate_performance_report'
]
