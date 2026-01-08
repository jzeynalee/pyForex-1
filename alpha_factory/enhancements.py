# alpha_factory/enhancements.py
"""
Enhancements and improvements to the Alpha Factory system based on feedback.

This module implements:
1. Stationarity checking and data preprocessing
2. Transfer Entropy for non-linear causal analysis
3. Look-ahead bias detection and prevention
4. Memory optimization with float32
5. Enhanced market structure analysis
6. Liquidity and slippage modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats, norm
from sklearn.preprocessing import StandardScaler
import warnings
import logging

logger = logging.getLogger(__name__)

class ProbabilisticRegimeDetector:
    """
    detects market regimes using soft probabilities rather than hard thresholds.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def compute_regime_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate probability distribution across regimes using Z-scores and Sigmoids.
        
        Returns:
            Dict with 'bullish', 'bearish', 'neutral', 'volatile' probabilities summing to 1.0.
        """
        if len(df) < self.window:
            return {'bullish': 0.25, 'bearish': 0.25, 'neutral': 0.25, 'volatile': 0.25}

        # 1. Feature Extraction
        closes = df['close']
        returns = closes.pct_change().dropna()
        
        # Volatility Z-Score (vs rolling baseline)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=self.window * 5).std().iloc[-1]
        if pd.isna(rolling_vol) or rolling_vol == 0: rolling_vol = current_vol
        vol_z = (current_vol - rolling_vol) / rolling_vol if rolling_vol > 0 else 0
        
        # Trend Strength (ADX-proxy using efficiency ratio)
        change = (closes.iloc[-1] - closes.iloc[-self.window])
        path = np.sum(np.abs(closes.diff().tail(self.window)))
        efficiency = abs(change / path) if path > 0 else 0
        
        # Directional Score (-1 to 1)
        # Using simple linear regression slope normalized by volatility
        y = closes.tail(self.window).values
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        normalized_slope = slope / (closes.iloc[-1] * 0.001) # Normalize to % terms
        
        # 2. Probability Calculation (Sigmoids)
        # P(Volatile) increases as Vol Z-Score increases
        p_volatile = 1 / (1 + np.exp(-(vol_z - 1.5) * 2))  # Shift center to 1.5 std devs
        
        # P(Trend) increases with efficiency
        p_trend = 1 / (1 + np.exp(-(efficiency - 0.4) * 10))
        
        # P(Direction)
        p_bull_raw = 1 / (1 + np.exp(-(normalized_slope - 0.5) * 2))
        p_bear_raw = 1 - p_bull_raw
        
        # 3. Regime Synthesis
        # Volatility consumes probability mass first
        probs = {}
        probs['volatile'] = p_volatile
        
        remaining = 1.0 - p_volatile
        probs['neutral'] = remaining * (1 - p_trend)
        
        trend_mass = remaining * p_trend
        probs['bullish'] = trend_mass * p_bull_raw
        probs['bearish'] = trend_mass * p_bear_raw
        
        # Normalize ensuring sum is 1.0
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}


def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Dict:
    """
    Check stationarity of a time series using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        significance_level: Threshold for stationarity test
        
    Returns:
        Dictionary with stationarity results and recommendations
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        # Perform ADF test
        result = adfuller(series.dropna(), autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < significance_level
        
        recommendations = []
        if not is_stationary:
            recommendations.append("Consider differencing the series")
            recommendations.append("Apply logarithmic transformation")
            recommendations.append("Use percentage returns instead of prices")
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'significance_level': significance_level,
            'recommendations': recommendations
        }
        
    except ImportError:
        logger.warning("statsmodels not available for stationarity testing")
        return {
            'is_stationary': True,  # Assume stationary for safety
            'method': 'assumed',
            'reason': 'statsmodels not available'
        }


def preprocess_for_stationarity(features: pd.DataFrame, target_col: str = 'close') -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess features to ensure stationarity for causal analysis.
    
    Args:
        features: DataFrame with features
        target_col: Target column name
        
    Returns:
        Tuple of (preprocessed features, preprocessing info)
    """
    preprocessed = features.copy()
    preprocessing_info = {}
    
    # Check target column stationarity
    if target_col in preprocessed.columns:
        target_stationarity = check_stationarity(preprocessed[target_col])
        preprocessing_info[f'{target_col}_stationarity'] = target_stationarity
        
        # Apply differencing if non-stationary
        if not target_stationarity['is_stationary']:
            preprocessed[f'{target_col}_diff'] = preprocessed[target_col].diff()
            preprocessed[f'{target_col}_pct'] = preprocessed[target_col].pct_change()
            preprocessing_info[f'{target_col}_transformation'] = 'differenced_and_pct_change'
    
    # Check feature stationarity for key features
    key_features = ['rsi', 'macd', 'atr', 'volume', 'sma_20', 'ema_12']
    
    for feature in key_features:
        if feature in preprocessed.columns:
            stationarity = check_stationarity(preprocessed[feature])
            preprocessing_info[f'{feature}_stationarity'] = stationarity
            
            # Apply transformations if needed
            if not stationarity['is_stationary']:
                if feature in ['volume', 'atr']:
                    # Log transform for positive values
                    preprocessed[f'{feature}_log'] = np.log1p(preprocessed[feature])
                    preprocessing_info[f'{feature}_transformation'] = 'log'
                else:
                    # Use differencing for other features
                    preprocessed[f'{feature}_diff'] = preprocessed[feature].diff()
                    preprocessing_info[f'{feature}_transformation'] = 'differenced'
    
    # Remove infinite and NaN values
    preprocessed = preprocessed.replace([np.inf, -np.inf], np.nan)
    preprocessed = preprocessed.fillna(method='ffill').fillna(method='bfill')
    
    return preprocessed, preprocessing_info


def compute_transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    """
    Compute Transfer Entropy between two time series.
    
    Transfer Entropy measures the reduction in uncertainty about the future
    of the target series given knowledge of the source series.
    
    Args:
        source: Source time series
        target: Target time series
        bins: Number of bins for discretization
        
    Returns:
        Transfer entropy value
    """
    try:
        # Remove NaN values
        mask = ~(np.isnan(source) | np.isnan(target))
        source_clean = source[mask]
        target_clean = target[mask]
        
        if len(source_clean) < 10:
            return 0.0
        
        # Discretize the data
        source_discrete = np.digitize(source_clean, bins=np.linspace(source_clean.min(), source_clean.max(), bins))
        target_discrete = np.digitize(target_clean, bins=np.linspace(target_clean.min(), target_clean.max(), bins))
        
        # Compute joint and marginal probabilities
        def compute_entropy(series):
            unique, counts = np.unique(series, return_counts=True)
            probabilities = counts / len(series)
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            return -np.sum(probabilities * np.log2(probabilities))
        
        # Compute entropies
        h_target = compute_entropy(target_discrete[1:])  # H(Y_t)
        h_target_given_target = compute_entropy(np.column_stack([target_discrete[:-1], target_discrete[1:]])) - compute_entropy(target_discrete[:-1])
        h_target_given_source_target = compute_entropy(np.column_stack([source_discrete[:-1], target_discrete[:-1], target_discrete[1:]])) - compute_entropy(np.column_stack([source_discrete[:-1], target_discrete[:-1]]))
        
        # Transfer Entropy
        te = h_target_given_target - h_target_given_source_target
        
        return max(0, te)  # Ensure non-negative
        
    except Exception as e:
        logger.debug(f"Transfer entropy computation failed: {e}")
        return 0.0


def enhanced_causal_analysis(features: pd.DataFrame, target_col: str = 'close') -> Dict:
    """
    Enhanced causal analysis including Transfer Entropy and stationarity checking.
    
    Args:
        features: DataFrame with features
        target_col: Target column name
        
    Returns:
        Enhanced causality results
    """
    from .causal_analysis import compute_causality
    
    # Preprocess for stationarity
    preprocessed_features, preprocessing_info = preprocess_for_stationarity(features, target_col)
    
    # Get original causality results
    original_results = compute_causality(preprocessed_features, target_col)
    
    # Add Transfer Entropy analysis
    transfer_entropy_results = {}
    feature_cols = [col for col in preprocessed_features.columns if col != target_col and col != 'time']
    
    target_values = preprocessed_features[target_col].values
    
    for feature in feature_cols:
        try:
            feature_values = preprocessed_features[feature].values
            
            # Compute Transfer Entropy
            te_source_to_target = compute_transfer_entropy(feature_values, target_values)
            te_target_to_source = compute_transfer_entropy(target_values, feature_values)
            
            transfer_entropy_results[feature] = {
                'te_source_to_target': te_source_to_target,
                'te_target_to_source': te_target_to_source,
                'net_te': te_source_to_target - te_target_to_source,
                'direction': 'source_drives' if te_source_to_target > te_target_to_source else 'target_drives'
            }
            
        except Exception as e:
            logger.debug(f"Transfer entropy failed for {feature}: {e}")
            continue
    
    # Combine results
    enhanced_results = original_results.copy()
    enhanced_results['transfer_entropy'] = transfer_entropy_results
    enhanced_results['preprocessing_info'] = preprocessing_info
    enhanced_results['stationarity_analysis'] = {
        'features_checked': len(preprocessing_info),
        'stationary_features': sum(1 for k, v in preprocessing_info.items() 
                                 if 'stationarity' in k and v.get('is_stationary', False))
    }
    
    # Update causal ranking with Transfer Entropy
    if 'causal_ranking' in enhanced_results:
        ranking = enhanced_results['causal_ranking']
        
        for feature, data in ranking.items():
            if feature in transfer_entropy_results:
                te_score = transfer_entropy_results[feature]['net_te']
                # Add Transfer Entropy to combined score
                data['combined_score'] = data['combined_score'] + abs(te_score) * 0.1
                data['transfer_entropy'] = te_score
        
        # Re-sort by updated scores
        sorted_features = sorted(ranking.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        for rank, (feature, data) in enumerate(sorted_features, 1):
            ranking[feature]['rank'] = rank
    
    return enhanced_results


def check_lookahead_bias(features: pd.DataFrame, target_col: str = 'close') -> Dict:
    """
    Check for potential look-ahead bias in feature engineering.
    
    Args:
        features: DataFrame with features
        target_col: Target column name
        
    Returns:
        Dictionary with bias analysis results
    """
    bias_analysis = {
        'potential_lookahead_issues': [],
        'safe_features': [],
        'recommendations': []
    }
    
    if target_col not in features.columns:
        return bias_analysis
    
    target_values = features[target_col].values
    
    for feature in features.columns:
        if feature in [target_col, 'time']:
            continue
        
        feature_values = features[feature].values
        
        # Check for perfect correlation (potential look-ahead)
        try:
            correlation = np.corrcoef(feature_values[~np.isnan(feature_values)], 
                                     target_values[~np.isnan(target_values)])[0, 1]
            
            if abs(correlation) > 0.95:
                bias_analysis['potential_lookahead_issues'].append({
                    'feature': feature,
                    'correlation': correlation,
                    'issue': 'Perfect correlation detected'
                })
            elif abs(correlation) < 0.1:
                bias_analysis['safe_features'].append(feature)
            else:
                bias_analysis['safe_features'].append(feature)
                
        except Exception:
            bias_analysis['safe_features'].append(feature)
    
    # Check for future information leakage
    suspicious_patterns = ['future', 'lead', 'ahead', 'tomorrow', 'next']
    for feature in features.columns:
        feature_lower = feature.lower()
        if any(pattern in feature_lower for pattern in suspicious_patterns):
            bias_analysis['potential_lookahead_issues'].append({
                'feature': feature,
                'correlation': None,
                'issue': 'Suspicious naming pattern detected'
            })
    
    # Generate recommendations
    if bias_analysis['potential_lookahead_issues']:
        bias_analysis['recommendations'].append("Review features with high correlation to target")
        bias_analysis['recommendations'].append("Ensure no future information is used")
        bias_analysis['recommendations'].append("Consider using only lagged features")
    
    return bias_analysis


def optimize_memory_usage(features: pd.DataFrame, target_dtype: str = 'float32') -> pd.DataFrame:
    """
    Optimize memory usage by converting data types.
    
    Args:
        features: DataFrame with features
        target_dtype: Target data type for numeric columns
        
    Returns:
        Memory-optimized DataFrame
    """
    optimized = features.copy()
    
    # Convert numeric columns to target dtype
    for col in optimized.select_dtypes(include=[np.number]).columns:
        optimized[col] = optimized[col].astype(target_dtype)
    
    # Convert object columns to category if they have low cardinality
    for col in optimized.select_dtypes(include=['object']).columns:
        if optimized[col].nunique() < len(optimized) * 0.5:  # Less than 50% unique
            optimized[col] = optimized[col].astype('category')
    
    # Log memory savings
    original_memory = features.memory_usage(deep=True).sum() / 1024**2  # MB
    optimized_memory = optimized.memory_usage(deep=True).sum() / 1024**2  # MB
    savings = original_memory - optimized_memory
    
    logger.info(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB (saved {savings:.1f}MB)")
    
    return optimized


def enhanced_market_structure_analysis(swing_points: List, lookback_window: int = 5) -> Dict:
    """
    Enhanced market structure analysis with multi-swing comparison.
    
    Args:
        swing_points: List of swing points
        lookback_window: Number of recent swings to analyze
        
    Returns:
        Enhanced market structure analysis
    """
    if not swing_points:
        return {'structure': 'unknown', 'trend': 'unknown', 'confidence': 0.0}
    
    # Separate highs and lows
    highs = [sp for sp in swing_points if hasattr(sp, 'point_type') and sp.point_type == 'high']
    lows = [sp for sp in swing_points if hasattr(sp, 'point_type') and sp.point_type == 'low']
    
    if len(highs) < lookback_window or len(lows) < lookback_window:
        return {'structure': 'insufficient_data', 'trend': 'unknown', 'confidence': 0.0}
    
    # Analyze recent swings
    recent_highs = sorted(highs, key=lambda x: x.time)[-lookback_window:]
    recent_lows = sorted(lows, key=lambda x: x.time)[-lookback_window:]
    
    # Count trend patterns
    higher_highs = 0
    lower_highs = 0
    higher_lows = 0
    lower_lows = 0
    
    # Compare consecutive highs
    for i in range(1, len(recent_highs)):
        if recent_highs[i].price > recent_highs[i-1].price:
            higher_highs += 1
        else:
            lower_highs += 1
    
    # Compare consecutive lows
    for i in range(1, len(recent_lows)):
        if recent_lows[i].price > recent_lows[i-1].price:
            higher_lows += 1
        else:
            lower_lows += 1
    
    # Calculate trend confidence
    total_comparisons = (len(recent_highs) - 1) + (len(recent_lows) - 1)
    bullish_signals = higher_highs + higher_lows
    bearish_signals = lower_highs + lower_lows
    
    bullish_confidence = bullish_signals / total_comparisons if total_comparisons > 0 else 0
    bearish_confidence = bearish_signals / total_comparisons if total_comparisons > 0 else 0
    
    # Determine structure and trend
    structure = 'unknown'
    trend = 'unknown'
    confidence = 0.0
    
    if bullish_confidence > 0.7:
        structure = 'uptrend'
        trend = 'bullish'
        confidence = bullish_confidence
    elif bearish_confidence > 0.7:
        structure = 'downtrend'
        trend = 'bearish'
        confidence = bearish_confidence
    elif abs(bullish_confidence - bearish_confidence) < 0.2:
        structure = 'range'
        trend = 'sideways'
        confidence = 1 - abs(bullish_confidence - bearish_confidence)
    else:
        structure = 'transitional'
        trend = 'uncertain'
        confidence = max(bullish_confidence, bearish_confidence)
    
    return {
        'structure': structure,
        'trend': trend,
        'confidence': confidence,
        'higher_highs': higher_highs,
        'lower_highs': lower_highs,
        'higher_lows': higher_lows,
        'lower_lows': lower_lows,
        'bullish_confidence': bullish_confidence,
        'bearish_confidence': bearish_confidence,
        'analyzed_swings': len(recent_highs) + len(recent_lows)
    }


def calculate_liquidity_adjusted_return(expected_return: float, volume: float, 
                                     spread: float, position_size: float) -> Dict:
    """
    Stricter liquidity adjustment with fixed 0.5 pip slippage.
    
    Args:
        expected_return: Expected return in pips
        volume: Current volume
        spread: Current spread in pips
        position_size: Position size in lots
        
    Returns:
        Dictionary with liquidity-adjusted metrics
    """
    # Base transaction costs (Spread + Fixed Commission)
    fixed_costs = spread + 0.1  # Adding 0.1 pip for commission/fees
    
    # FIXED: Hard-code 0.5 pip slippage per trade (reflecting real-world ECN execution delays)
    fixed_slippage = 0.5  # 0.5 pips fixed slippage
    
    # Market Impact (Quadratic/Square Root Model)
    # Impact increases as a function of our size relative to available volume
    volume_participation = position_size / max(volume, 1)
    market_impact = 0.5 * (volume_participation ** 0.5)  # Square root impact law
    
    # Total Punitive Cost (including fixed slippage)
    total_cost = fixed_costs + fixed_slippage + market_impact
    
    # Margin of Safety: Signal must be at least 3x the cost to be viable
    is_viable = expected_return > (total_cost * 3.0)
    
    # Liquidity score (0-1, higher is better)
    liquidity_score = max(0, min(1, 1 - volume_participation))
    
    # Risk-adjusted return
    if total_cost > 0:
        risk_adjusted_return = (expected_return - total_cost) / total_cost
    else:
        risk_adjusted_return = expected_return
    
    return {
        'expected_return': expected_return,
        'net_alpha': expected_return - total_cost,
        'cost_basis': total_cost,
        'fixed_costs': fixed_costs,
        'fixed_slippage': fixed_slippage,
        'market_impact': market_impact,
        'spread_cost': spread,
        'liquidity_adjusted_return': expected_return - total_cost,
        'liquidity_score': liquidity_score,
        'risk_adjusted_return': risk_adjusted_return,
        'trade_viable': is_viable
    }


def create_enhanced_alpha_factory(config: Optional[Dict] = None) -> 'AlphaFactory':
    """
    Create an enhanced Alpha Factory with all improvements.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Enhanced AlphaFactory instance
    """
    from .alpha_factory import AlphaFactory
    
    # Enhanced default configuration
    enhanced_config = {
        'market_data': {
            'swing_lookback': 5,
            'strength_threshold': 0.3,
            'enhanced_structure_analysis': True,
            'structure_lookback_window': 5
        },
        'features': {
            'batch_processing': True,
            'max_lookback': 1050,
            'memory_optimization': True,
            'target_dtype': 'float32',
            'stationarity_check': True
        },
        'causality': {
            'target_col': 'close',
            'max_lag': 5,
            'top_n_features': 10,
            'enhanced_analysis': True,
            'transfer_entropy': True,
            'lookahead_bias_check': True
        },
        'decision': {
            'min_confidence': 0.6,
            'min_causal_score': 0.3,
            'trend_weight': 0.3,
            'support_resistance_weight': 0.2,
            'momentum_weight': 0.25,
            'causal_weight': 0.25,
            'liquidity_adjustment': True
        },
        'output': {
            'save_features': True,
            'save_causality': True,
            'save_decisions': True,
            'output_dir': 'alpha_output_enhanced'
        }
    }
    
    # Merge with user config
    if config:
        for key, value in config.items():
            if key in enhanced_config and isinstance(enhanced_config[key], dict):
                enhanced_config[key].update(value)
            else:
                enhanced_config[key] = value
    
    return AlphaFactory(enhanced_config)
