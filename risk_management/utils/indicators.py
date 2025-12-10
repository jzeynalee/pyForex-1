"""
Utilities for Risk Management System

Common functions and helpers used across all phases:
- Technical indicator calculations
- Data preprocessing
- Regime detection
- Performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Technical Indicators
# =============================================================================

def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    ATR = SMA(TR, period)
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # EMA for smoother ATR
    atr = pd.Series(true_range).ewm(span=period, adjust=False).mean().values
    
    return atr


def calculate_volatility(
    close: np.ndarray,
    period: int = 20,
    annualize: bool = False,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).
    """
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    volatility = pd.Series(returns).rolling(period).std().fillna(0).values
    
    if annualize:
        volatility *= np.sqrt(periods_per_year)
    
    return volatility


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI.
    
    Returns:
        (adx, plus_di, minus_di)
    """
    # Calculate +DM and -DM
    high_diff = np.diff(high)
    low_diff = -np.diff(low)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Pad to original length
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Smooth DM values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr + 1e-8)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-8)
    
    # Calculate DX
    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / (di_sum + 1e-8)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index (RSI)."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gains = pd.Series(gains).ewm(span=period, adjust=False).mean().values
    avg_losses = pd.Series(losses).ewm(span=period, adjust=False).mean().values
    
    rs = avg_gains / (avg_losses + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = pd.Series(close).rolling(period).mean().fillna(method='bfill').values
    std = pd.Series(close).rolling(period).std().fillna(method='bfill').values
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return upper, middle, lower


# =============================================================================
# Regime Detection
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    adx_strong_trend: float = 25.0
    adx_weak_trend: float = 20.0
    volatility_high_pct: float = 75.0
    volatility_low_pct: float = 25.0
    lookback_period: int = 50


class RegimeDetector:
    """
    Detects current market regime using multiple indicators.
    
    Regimes:
    - TRENDING_STRONG: ADX > 25, clear directional movement
    - TRENDING_WEAK: ADX 20-25, some directional bias
    - RANGING: ADX < 20, price oscillating
    - VOLATILE: High volatility regardless of trend
    - LOW_VOLATILITY: Low volatility, potentially breakout setup
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._volatility_history = []
    
    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect current market regime.
        
        Args:
            high, low, close: OHLC data
            volume: Optional volume data
        
        Returns:
            (regime, indicators_dict)
        """
        # Calculate indicators
        adx, plus_di, minus_di = calculate_adx(high, low, close)
        current_adx = adx[-1]
        
        volatility = calculate_volatility(close)
        current_vol = volatility[-1]
        
        # Update volatility history for percentile
        self._volatility_history.append(current_vol)
        if len(self._volatility_history) > self.config.lookback_period:
            self._volatility_history.pop(0)
        
        # Calculate volatility percentile
        if len(self._volatility_history) >= 20:
            vol_percentile = (
                np.sum(np.array(self._volatility_history) < current_vol) /
                len(self._volatility_history) * 100
            )
        else:
            vol_percentile = 50.0
        
        indicators = {
            'adx': current_adx,
            'plus_di': plus_di[-1],
            'minus_di': minus_di[-1],
            'volatility': current_vol,
            'volatility_percentile': vol_percentile
        }
        
        # Determine regime
        regime = self._classify_regime(current_adx, vol_percentile)
        
        return regime, indicators
    
    def _classify_regime(
        self,
        adx: float,
        vol_percentile: float
    ) -> MarketRegime:
        """Classify market regime based on indicators."""
        
        # Check volatility first (overrides trend)
        if vol_percentile >= self.config.volatility_high_pct:
            return MarketRegime.VOLATILE
        
        if vol_percentile <= self.config.volatility_low_pct:
            return MarketRegime.LOW_VOLATILITY
        
        # Check trend strength
        if adx >= self.config.adx_strong_trend:
            return MarketRegime.TRENDING_STRONG
        
        if adx >= self.config.adx_weak_trend:
            return MarketRegime.TRENDING_WEAK
        
        return MarketRegime.RANGING
    
    def detect_batch(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Detect regime for entire price series.
        
        Returns array of regime values (as integers).
        """
        n = len(close)
        regimes = np.zeros(n, dtype=np.int32)
        
        # Calculate indicators for full series
        adx, _, _ = calculate_adx(high, low, close)
        volatility = calculate_volatility(close)
        
        # Rolling volatility percentile
        vol_pct = pd.Series(volatility).rolling(
            self.config.lookback_period
        ).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) * 100
        ).fillna(50).values
        
        # Classify each point
        regime_map = {
            MarketRegime.TRENDING_STRONG: 0,
            MarketRegime.TRENDING_WEAK: 1,
            MarketRegime.RANGING: 2,
            MarketRegime.VOLATILE: 3,
            MarketRegime.LOW_VOLATILITY: 4
        }
        
        for i in range(n):
            regime = self._classify_regime(adx[i], vol_pct[i])
            regimes[i] = regime_map[regime]
        
        return regimes


# =============================================================================
# Data Preprocessing
# =============================================================================

def create_direction_labels(
    close: np.ndarray,
    horizon: int = 1,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Create direction labels based on future returns.
    
    Args:
        close: Close prices
        horizon: Look-ahead periods
        threshold: Return threshold for sideways classification
    
    Returns:
        Labels: 0 (Bear), 1 (Sideways), 2 (Bull)
    """
    n = len(close)
    labels = np.ones(n, dtype=np.int32)  # Default sideways
    
    # Calculate forward returns
    for i in range(n - horizon):
        future_return = (close[i + horizon] - close[i]) / close[i]
        
        if future_return > threshold:
            labels[i] = 2  # Bull
        elif future_return < -threshold:
            labels[i] = 0  # Bear
        # else: stays 1 (Sideways)
    
    return labels


def create_volatility_labels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    horizon: int = 1
) -> np.ndarray:
    """
    Create volatility labels (realized volatility over horizon).
    """
    n = len(close)
    labels = np.zeros(n)
    
    for i in range(n - horizon):
        # Use average true range over horizon as realized volatility
        future_high = high[i + 1:i + 1 + horizon]
        future_low = low[i + 1:i + 1 + horizon]
        future_close = close[i + 1:i + 1 + horizon]
        
        if len(future_high) > 0:
            atr = calculate_atr(future_high, future_low, future_close, period=min(horizon, 5))
            labels[i] = atr[-1] if len(atr) > 0 else 0
    
    return labels


def create_price_move_labels(
    close: np.ndarray,
    horizon: int = 1
) -> np.ndarray:
    """
    Create price movement labels (actual price change over horizon).
    
    Used for quantile regression targets.
    """
    n = len(close)
    labels = np.zeros(n)
    
    for i in range(n - horizon):
        labels[i] = close[i + horizon] - close[i]
    
    return labels


def normalize_features(
    features: np.ndarray,
    method: str = 'zscore',
    clip_value: float = 5.0
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize features for model input.
    
    Args:
        features: (n_samples, n_features) array
        method: 'zscore', 'minmax', or 'robust'
        clip_value: Clip extreme values
    
    Returns:
        (normalized_features, normalization_params)
    """
    params = {}
    
    if method == 'zscore':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        normalized = (features - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'robust':
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25 + 1e-8
        normalized = (features - median) / iqr
        params = {'median': median, 'iqr': iqr}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Clip extreme values
    normalized = np.clip(normalized, -clip_value, clip_value)
    
    params['method'] = method
    params['clip_value'] = clip_value
    
    return normalized, params


def apply_normalization(
    features: np.ndarray,
    params: Dict
) -> np.ndarray:
    """Apply saved normalization parameters to new features."""
    method = params['method']
    
    if method == 'zscore':
        normalized = (features - params['mean']) / params['std']
    elif method == 'minmax':
        normalized = (features - params['min']) / (params['max'] - params['min'] + 1e-8)
    elif method == 'robust':
        normalized = (features - params['median']) / params['iqr']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.clip(normalized, -params['clip_value'], params['clip_value'])


# =============================================================================
# Performance Metrics
# =============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    excess_returns = returns - risk_free_rate / periods_per_year
    
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    sortino = np.mean(excess_returns) / downside_std
    
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Returns:
        (max_drawdown_pct, peak_idx, trough_idx)
    """
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0
    max_dd_peak = 0
    max_dd_trough = 0
    
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = i
        
        drawdown = (peak - value) / peak
        
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_peak = peak_idx
            max_dd_trough = i
    
    return max_dd * 100, max_dd_peak, max_dd_trough


def calculate_win_rate(outcomes: np.ndarray) -> float:
    """Calculate win rate from trade outcomes."""
    wins = np.sum(outcomes > 0)
    total = len(outcomes)
    return wins / total if total > 0 else 0.0


def calculate_profit_factor(
    returns: np.ndarray
) -> float:
    """Calculate profit factor (gross profits / gross losses)."""
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(
    returns: np.ndarray
) -> float:
    """
    Calculate trade expectancy.
    
    E = (Win% × Avg Win) - (Loss% × Avg Loss)
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(returns) == 0:
        return 0.0
    
    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 0
    
    return (win_rate * avg_win) - (loss_rate * avg_loss)


@dataclass
class PerformanceReport:
    """Comprehensive performance metrics."""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    expectancy: float
    total_return: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int


def generate_performance_report(
    returns: np.ndarray,
    equity_curve: Optional[np.ndarray] = None
) -> PerformanceReport:
    """Generate comprehensive performance report."""
    
    if equity_curve is None:
        equity_curve = np.cumprod(1 + returns)
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    # Calculate consecutive wins/losses
    max_cons_wins = 0
    max_cons_losses = 0
    current_streak = 0
    
    for r in returns:
        if r > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_cons_wins = max(max_cons_wins, current_streak)
        elif r < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_cons_losses = max(max_cons_losses, abs(current_streak))
        else:
            current_streak = 0
    
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    return PerformanceReport(
        total_trades=len(returns),
        win_rate=calculate_win_rate(returns),
        profit_factor=calculate_profit_factor(returns),
        sharpe_ratio=calculate_sharpe_ratio(returns),
        sortino_ratio=calculate_sortino_ratio(returns),
        max_drawdown=max_dd,
        expectancy=calculate_expectancy(returns),
        total_return=(equity_curve[-1] / equity_curve[0] - 1) * 100 if len(equity_curve) > 0 else 0,
        avg_trade_return=np.mean(returns) * 100 if len(returns) > 0 else 0,
        best_trade=np.max(returns) * 100 if len(returns) > 0 else 0,
        worst_trade=np.min(returns) * 100 if len(returns) > 0 else 0,
        avg_win=np.mean(wins) * 100 if len(wins) > 0 else 0,
        avg_loss=np.mean(losses) * 100 if len(losses) > 0 else 0,
        max_consecutive_wins=max_cons_wins,
        max_consecutive_losses=max_cons_losses
    )
