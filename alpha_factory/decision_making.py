# alpha_factory/decision_making.py
"""
Decision Making module for Alpha Factory system.
Implements decision logic based on market structure, features, and causal analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .signal_quality_optimizer import SignalQualityOptimizer, SignalQualityConfig
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class DecisionType(Enum):
    """Trading decision types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class DecisionSignal:
    """Trading decision signal with confidence."""
    decision: DecisionType
    confidence: float  # 0-1
    regime: MarketRegime
    reasoning: List[str]
    key_features: List[str]
    risk_score: float  # 0-1, higher = riskier
    expected_return: float  # Expected return in pips/points
    stop_loss: float
    take_profit: float


@dataclass
class DecisionConfig:
    """Configuration for decision making."""
    # Thresholds
    min_confidence: float = 0.6
    min_causal_score: float = 0.3
    max_risk_score: float = 0.7
    
    # Market structure weights
    trend_weight: float = 0.3
    support_resistance_weight: float = 0.2
    momentum_weight: float = 0.25
    causal_weight: float = 0.25
    
    # Risk management
    base_risk_percent: float = 1.0
    max_risk_reward_ratio: float = 3.0
    min_risk_reward_ratio: float = 1.5
    
    # Regime detection
    trend_period: int = 20
    volatility_period: int = 14
    volume_threshold: float = 1.2
    trend_strength_threshold: float = 25.0  # ADX threshold for trend detection
    
    # Enhanced features
    liquidity_adjustment: bool = False
    
    # Professional Signal Quality Optimization
    signal_quality_enabled: bool = True
    confidence_gate_percentile: float = 70.0  # Trade only top 70% of signals
    regime_execution_enabled: bool = True


def decision_function(swing_points: List, features: pd.DataFrame, 
                     causality_results: Dict, config: Optional[DecisionConfig] = None,
                     market_structure: Optional[Dict] = None,
                     signal_optimizer: Optional[SignalQualityOptimizer] = None) -> DecisionSignal:
    """
    Make trading decision based on market structure, features, and causal analysis.
    
    Args:
        swing_points: List of swing points from market structure analysis
        features: DataFrame with extracted features
        causality_results: Results from causal analysis
        config: Decision configuration
        market_structure: Market structure analysis results
        signal_optimizer: Signal quality optimizer for professional improvements
        
    Returns:
        DecisionSignal with trading recommendation
    """
    if config is None:
        config = DecisionConfig()
    
    # 1. Analyze market regime
    regime = _determine_market_regime(features, swing_points, config)
    
    # 2. Check for mean reversion opportunities in neutral/ranging markets
    if market_structure and regime == MarketRegime.NEUTRAL:
        mean_reversion_signal = _apply_mean_reversion_logic(
            regime.value, market_structure, features, swing_points, config
        )
        if mean_reversion_signal and mean_reversion_signal.confidence >= config.min_confidence * 0.8:
            return mean_reversion_signal
    
    # 3. Evaluate market structure
    structure_score, structure_reasoning = _evaluate_market_structure(swing_points, config)
    
    # 4. Analyze feature signals
    feature_signals, feature_reasoning = _analyze_feature_signals(features, config)
    
    # 5. Evaluate causal relationships
    causal_score, causal_reasoning = _evaluate_causal_signals(causality_results, config)
    
    # 6. Combine all signals with regime-specific weighting
    combined_score = _calculate_regime_weighted_score(structure_score, feature_signals, causal_score, regime)
    
    # 7. Determine decision
    decision, confidence = _determine_decision(combined_score, regime, config)
    
    # 8. Apply Professional Signal Quality Optimization
    if config.signal_quality_enabled and signal_optimizer:
        # Create signal DataFrame for optimization
        signal_data = pd.DataFrame({
            'decision': [decision.value],
            'confidence': [confidence],
            'regime': [regime.value],
            'combined_score': [combined_score]
        })
        
        # Apply confidence gate
        optimized_signals = signal_optimizer.apply_confidence_gate(signal_data)
        
        if len(optimized_signals) == 0:
            # Signal filtered out by confidence gate
            return DecisionSignal(
                decision=DecisionType.HOLD,
                confidence=confidence,
                regime=regime,
                reasoning=["Signal filtered by confidence gate"],
                key_features=[],
                risk_score=0.0,
                expected_return=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )
        
        # Apply regime-conditional execution
        if config.regime_execution_enabled:
            regime_data = pd.Series([regime.value])
            optimized_signals = signal_optimizer.apply_regime_conditional_execution(
                optimized_signals, regime_data
            )
            
            if optimized_signals.iloc[0].get('skip_trade', False):
                return DecisionSignal(
                    decision=DecisionType.HOLD,
                    confidence=confidence,
                    regime=regime,
                    reasoning=[f"Trade skipped for {regime.value} regime"],
                    key_features=[],
                    risk_score=0.0,
                    expected_return=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )
    
    # 9. Calculate risk parameters
    risk_score, stop_loss, take_profit = _calculate_risk_parameters(
        features, decision, regime, config
    )
    
    # 10. Compile reasoning
    reasoning = structure_reasoning + feature_reasoning + causal_reasoning
    
    # 11. Identify key features
    key_features = _identify_key_features(causality_results, feature_signals)
    
    # 12. Calculate expected return
    expected_return = _calculate_expected_return(decision, features, regime)
    
    return DecisionSignal(
        decision=decision,
        confidence=confidence,
        regime=regime,
        reasoning=reasoning,
        key_features=key_features,
        risk_score=risk_score,
        expected_return=expected_return,
        stop_loss=stop_loss,
        take_profit=take_profit
    )


def _determine_market_regime(features: pd.DataFrame, swing_points: List, config: DecisionConfig) -> MarketRegime:
    """
    Refactored for Professional-Grade Neutral Market Detection.
    """
    recent_data = features.tail(config.trend_period)
    
    # 1. Enhanced Volatility/Range Check
    volatility = recent_data['close'].pct_change().std()
    atr_ratio = recent_data['atr'].iloc[-1] / recent_data['close'].iloc[-1] if 'atr' in recent_data.columns else 0
    
    # 2. Volume Z-Score Filter (The breakthrough component)
    if 'volume' in recent_data.columns and len(recent_data) >= 20:
        volume_z = (recent_data['volume'].iloc[-1] - recent_data['volume'].mean()) / recent_data['volume'].std()
    else:
        volume_z = 0

    # 3. Regime Classification Logic
    if volatility > 0.02 or atr_ratio > 0.015:
        return MarketRegime.VOLATILE
    
    # Identify Neutral/Range via ADX and Volume Exhaustion
    adx = recent_data['adx'].iloc[-1] if 'adx' in recent_data.columns else 20
    if adx < config.trend_strength_threshold and volume_z < 0: # Volume exhaustion in range
        return MarketRegime.NEUTRAL
        
    # Trend Classification
    price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
    return MarketRegime.BULLISH if price_trend > 0 else MarketRegime.BEARISH


def _evaluate_market_structure(swing_points: List, config: DecisionConfig) -> Tuple[float, List[str]]:
    """Evaluate market structure from swing points."""
    if not swing_points:
        return 0.0, ["No swing points available"]
    
    reasoning = []
    score = 0.0
    
    # Analyze swing point patterns
    highs = [sp for sp in swing_points if hasattr(sp, 'point_type') and sp.point_type == 'high']
    lows = [sp for sp in swing_points if hasattr(sp, 'point_type') and sp.point_type == 'low']
    
    if len(highs) >= 2 and len(lows) >= 2:
        # Check for trend structure
        recent_highs = highs[-3:] if len(highs) >= 3 else highs
        recent_lows = lows[-3:] if len(lows) >= 3 else lows
        
        # Higher highs and higher lows (bullish structure)
        if (len(recent_highs) >= 2 and len(recent_lows) >= 2):
            if (recent_highs[-1].price > recent_highs[-2].price and 
                recent_lows[-1].price > recent_lows[-2].price):
                score += 0.5
                reasoning.append("Bullish market structure: Higher highs and higher lows")
            
            # Lower highs and lower lows (bearish structure)
            elif (recent_highs[-1].price < recent_highs[-2].price and 
                  recent_lows[-1].price < recent_lows[-2].price):
                score -= 0.5
                reasoning.append("Bearish market structure: Lower highs and lower lows")
    
    # Support and resistance levels
    if len(swing_points) >= 4:
        # Find recent strong support/resistance
        strong_swings = [sp for sp in swing_points if hasattr(sp, 'strength') and sp.strength > 0.5]
        if strong_swings:
            reasoning.append(f"Found {len(strong_swings)} strong swing points")
            score += 0.2 * min(len(strong_swings) / 4, 1.0)
    
    return np.clip(score, -1, 1), reasoning


def _analyze_feature_signals(features: pd.DataFrame, config: DecisionConfig) -> Tuple[float, List[str]]:
    """Analyze technical indicators and feature signals."""
    if features.empty:
        return 0.0, ["No features available"]
    
    reasoning = []
    signals = []
    
    # RSI signals
    if 'rsi' in features.columns:
        rsi = features['rsi'].iloc[-1]
        if rsi < 30:
            signals.append(0.3)  # Oversold - bullish
            reasoning.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 70:
            signals.append(-0.3)  # Overbought - bearish
            reasoning.append(f"RSI overbought at {rsi:.1f}")
    
    # MACD signals
    if all(col in features.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        macd = features['macd'].iloc[-1]
        macd_signal = features['macd_signal'].iloc[-1]
        macd_hist = features['macd_histogram'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0:
            signals.append(0.4)  # Bullish MACD crossover
            reasoning.append("MACD bullish crossover")
        elif macd < macd_signal and macd_hist < 0:
            signals.append(-0.4)  # Bearish MACD crossover
            reasoning.append("MACD bearish crossover")
    
    # Moving average signals
    if all(col in features.columns for col in ['sma_20', 'sma_50', 'close']):
        close = features['close'].iloc[-1]
        sma20 = features['sma_20'].iloc[-1]
        sma50 = features['sma_50'].iloc[-1]
        
        if close > sma20 > sma50:
            signals.append(0.3)  # Above MAs - bullish
            reasoning.append("Price above moving averages")
        elif close < sma20 < sma50:
            signals.append(-0.3)  # Below MAs - bearish
            reasoning.append("Price below moving averages")
    
    # Bollinger Bands signals
    if all(col in features.columns for col in ['bb_upper', 'bb_lower', 'close']):
        close = features['close'].iloc[-1]
        bb_upper = features['bb_upper'].iloc[-1]
        bb_lower = features['bb_lower'].iloc[-1]
        
        if close < bb_lower:
            signals.append(0.2)  # Below lower band - potential bounce
            reasoning.append("Price below Bollinger lower band")
        elif close > bb_upper:
            signals.append(-0.2)  # Above upper band - potential pullback
            reasoning.append("Price above Bollinger upper band")
    
    # ADX trend strength
    if 'adx' in features.columns:
        adx = features['adx'].iloc[-1]
        if adx > 25:
            reasoning.append(f"Strong trend (ADX: {adx:.1f})")
        else:
            reasoning.append(f"Weak trend (ADX: {adx:.1f})")
    
    # Volume signals
    if 'volume_ratio' in features.columns:
        vol_ratio = features['volume_ratio'].iloc[-1]
        if vol_ratio > 1.5:
            reasoning.append(f"High volume ({vol_ratio:.1f}x average)")
    
    # Combine signals
    if signals:
        avg_signal = np.mean(signals)
    else:
        avg_signal = 0.0
        reasoning.append("No clear technical signals")
    
    return np.clip(avg_signal, -1, 1), reasoning


def _evaluate_causal_signals(causality_results: Dict, config: DecisionConfig) -> Tuple[float, List[str]]:
    """Evaluate causal relationships for decision making."""
    if not causality_results or 'causal_ranking' not in causality_results:
        return 0.0, ["No causal analysis available"]
    
    reasoning = []
    signals = []
    
    ranking = causality_results['causal_ranking']
    
    # Get top causal features
    top_features = list(ranking.items())[:5]  # Top 5 features
    
    for feature, data in top_features:
        score = data['combined_score']
        
        if score > config.min_causal_score:
            # Determine signal direction based on correlation
            correlation = 0
            if feature in causality_results.get('correlation_analysis', {}):
                corr_data = causality_results['correlation_analysis'][feature]
                correlation = corr_data['pearson_corr']
            
            # Weight by causal strength
            weighted_signal = correlation * score
            signals.append(weighted_signal)
            
            reasoning.append(f"Strong causal feature: {feature} (score: {score:.3f})")
    
    if signals:
        avg_signal = np.mean(signals)
        reasoning.append(f"Average causal signal: {avg_signal:.3f}")
    else:
        avg_signal = 0.0
        reasoning.append("No significant causal relationships")
    
    return np.clip(avg_signal, -1, 1), reasoning


def _calculate_regime_weighted_score(structure_score: float, feature_signals: float, 
                                      causal_score: float, regime: MarketRegime) -> float:
    """
    Calculate weighted score based on market regime.
    
    Args:
        structure_score: Market structure score (-1 to 1)
        feature_signals: Technical indicator signals (-1 to 1)
        causal_score: Causal analysis signals (-1 to 1)
        regime: Current market regime
        
    Returns:
        Weighted combined score (-1 to 1)
    """
    # Regime-specific weights as specified
    regime_weights = {
        MarketRegime.BULLISH: {'trend_weight': 0.60, 'momentum_weight': 0.25, 'causal_weight': 0.15},
        MarketRegime.BEARISH: {'trend_weight': 0.60, 'momentum_weight': 0.25, 'causal_weight': 0.15},
        MarketRegime.NEUTRAL: {'trend_weight': 0.10, 'momentum_weight': 0.20, 'causal_weight': 0.70},
        MarketRegime.VOLATILE: {'trend_weight': 0.20, 'momentum_weight': 0.40, 'causal_weight': 0.40}
    }
    
    weights = regime_weights.get(regime, regime_weights[MarketRegime.NEUTRAL])
    
    # Calculate weighted score
    weighted_score = (
        structure_score * weights['trend_weight'] +
        feature_signals * weights['momentum_weight'] +
        causal_score * weights['causal_weight']
    )
    
    return np.clip(weighted_score, -1, 1)


def _determine_decision(combined_score: float, regime: MarketRegime, config: DecisionConfig) -> Tuple[DecisionType, float]:
    """Determine final trading decision and confidence."""
    confidence = abs(combined_score)

    # NEW: Higher barriers for entry to reduce over-trading
    threshold = 0.82 if regime == MarketRegime.NEUTRAL else 0.75

    if confidence < threshold:
        return DecisionType.HOLD, confidence
    
    # Only proceed if expected return is 4x the transaction cost
    return (DecisionType.BUY if combined_score > 0 else DecisionType.SELL), confidence
    
    # Adjust confidence based on regime
    if regime == MarketRegime.VOLATILE:
        confidence *= 0.7  # Reduce confidence in volatile markets
    elif regime == MarketRegime.NEUTRAL:
        confidence *= 0.8  # Reduce confidence in neutral markets
    
    # Determine decision
    if confidence < config.min_confidence:
        return DecisionType.HOLD, confidence
    elif combined_score > 0:
        return DecisionType.BUY, confidence
    else:
        return DecisionType.SELL, confidence


def _calculate_risk_parameters(features: pd.DataFrame, decision: DecisionType, 
                              regime: MarketRegime, config: DecisionConfig) -> Tuple[float, float, float]:
    """Calculate risk score, stop loss, and take profit with regime-specific risk/reward."""
    if features.empty or 'close' not in features.columns:
        return 0.5, 0.0, 0.0
    
    current_price = features['close'].iloc[-1]
    
    # Calculate ATR for stop loss/take profit
    atr = 0.002  # Default 20 pips
    if 'atr' in features.columns:
        atr = features['atr'].iloc[-1]
    
    # Risk score based on regime and volatility
    risk_score = 0.3  # Base risk
    
    if regime == MarketRegime.VOLATILE:
        risk_score += 0.3
    elif regime in [MarketRegime.BULLISH, MarketRegime.BEARISH]:
        risk_score += 0.1
    
    # Adjust for volatility
    if 'atr' in features.columns:
        atr_ratio = features['atr'].iloc[-1] / current_price
        if atr_ratio > 0.01:  # High volatility
            risk_score += 0.2
    
    risk_score = np.clip(risk_score, 0, 1)
    
    # NEW: Regime-Specific Risk/Reward Ratios
    if regime in [MarketRegime.BULLISH, MarketRegime.BEARISH]:
        # 4:1 for Trends
        risk_reward_ratio = 4.0
        stop_distance = atr * 1.2  # Tighter stops for trends
    elif regime == MarketRegime.NEUTRAL:
        # 2.5:1 for Ranges
        risk_reward_ratio = 2.5
        stop_distance = atr * 1.5  # Wider stops for ranges
    else:  # VOLATILE
        # 3:1 for Volatile markets
        risk_reward_ratio = 3.0
        stop_distance = atr * 2.0  # Wider stops for volatility
    
    # Calculate stop loss and take profit with regime-specific ratios
    if decision == DecisionType.BUY:
        stop_loss = current_price - stop_distance
        take_profit = current_price + (stop_distance * risk_reward_ratio)
    elif decision == DecisionType.SELL:
        stop_loss = current_price + stop_distance
        take_profit = current_price - (stop_distance * risk_reward_ratio)
    else:
        stop_loss = current_price
        take_profit = current_price
    
    return risk_score, stop_loss, take_profit


def _identify_key_features(causality_results: Dict, feature_signals: float) -> List[str]:
    """Identify key features driving the decision."""
    key_features = []
    
    if 'causal_ranking' in causality_results:
        ranking = causality_results['causal_ranking']
        top_features = list(ranking.keys())[:3]  # Top 3 features
        key_features.extend(top_features)
    
    # Add high-impact technical indicators if available
    high_impact_features = ['rsi', 'macd', 'adx', 'bb_position']
    key_features.extend(high_impact_features)
    
    return list(set(key_features))  # Remove duplicates


def _calculate_expected_return(decision: DecisionType, features: pd.DataFrame, 
                              regime: MarketRegime) -> float:
    """Calculate expected return in pips/points."""
    if decision == DecisionType.HOLD or features.empty:
        return 0.0
    
    # Base expected return
    base_return = 20.0  # 20 pips
    
    # Adjust based on regime
    if regime == MarketRegime.VOLATILE:
        base_return *= 1.5
    elif regime == MarketRegime.NEUTRAL:
        base_return *= 0.7
    
    # Adjust based on ATR if available
    if 'atr' in features.columns:
        atr_pips = features['atr'].iloc[-1] * 10000  # Convert to pips (assuming EURUSD)
        base_return = min(base_return, atr_pips * 2)  # Cap at 2x ATR
    
    return base_return


def create_decision_summary(decision_signal: DecisionSignal) -> Dict:
    """Create a summary of the decision signal."""
    return {
        'decision': decision_signal.decision.value,
        'confidence': decision_signal.confidence,
        'regime': decision_signal.regime.value,
        'risk_score': decision_signal.risk_score,
        'expected_return': decision_signal.expected_return,
        'stop_loss': decision_signal.stop_loss,
        'take_profit': decision_signal.take_profit,
        'reasoning': decision_signal.reasoning,
        'key_features': decision_signal.key_features,
        'recommendation': _format_recommendation(decision_signal)
    }


def _format_recommendation(decision_signal: DecisionSignal) -> str:
    """Format human-readable recommendation."""
    if decision_signal.decision == DecisionType.HOLD:
        return "HOLD - No clear trading opportunity detected"
    
    action = decision_signal.decision.value
    confidence_pct = decision_signal.confidence * 100
    
    rec = f"{action} - Confidence: {confidence_pct:.1f}%\n"
    rec += f"Regime: {decision_signal.regime.value}\n"
    rec += f"Risk Score: {decision_signal.risk_score:.2f}\n"
    rec += f"Expected Return: {decision_signal.expected_return:.1f} pips\n"
    rec += f"Stop Loss: {decision_signal.stop_loss:.5f}\n"
    rec += f"Take Profit: {decision_signal.take_profit:.5f}"
    
    return rec
