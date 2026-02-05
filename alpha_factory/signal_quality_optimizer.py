"""
Signal Quality Optimizer for Alpha Factory

This module implements professional alpha improvement techniques:
1. Confidence Gating - Trade only top X% of signals
2. Probability Calibration - Platt scaling and Brier score
3. Regime-Conditional Execution - Different rules per regime
4. Feature Pruning - Remove weak contributors
5. Signal Decay Model - Track alpha expiration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

@dataclass
class SignalQualityConfig:
    """Configuration for signal quality optimization."""
    # Confidence Gate Settings
    confidence_gate_enabled: bool = True
    confidence_percentile: float = 70.0  # Trade only top 70% of signals
    min_confidence_threshold: float = 0.75  # Minimum confidence to consider
    
    # Probability Calibration
    calibration_enabled: bool = True
    calibration_method: str = 'isotonic'  # 'isotonic' or 'platt'
    calibration_window: int = 1000  # Trades for calibration
    
    # Regime-Conditional Execution
    regime_execution_enabled: bool = True
    regime_multipliers: Dict[str, float] = None  # Size multipliers by regime
    
    # Feature Pruning
    feature_pruning_enabled: bool = False
    min_feature_contribution: float = 0.05  # 5% minimum contribution
    
    # Signal Decay
    signal_decay_enabled: bool = True
    decay_rate: float = 0.001  # Decay rate per trade
    max_signal_age: int = 100  # Maximum trades before retrain

class SignalQualityOptimizer:
    """Professional signal quality optimization for Alpha Factory."""
    
    def __init__(self, config: SignalQualityConfig = None):
        self.config = config or SignalQualityConfig()
        self.calibration_models = {}
        self.feature_importance = {}
        self.signal_history = []
        self.trade_count = 0
        
        # Initialize regime multipliers if not provided
        if self.config.regime_multipliers is None:
            self.config.regime_multipliers = {
                'bullish': 1.0,      # Full size in strong trends
                'bearish': 1.0,      # Full size in strong trends  
                'neutral': 0.0,      # No trades in neutral/range
                'volatile': 0.5      # Half size in volatile
            }
    
    def apply_confidence_gate(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply confidence gating - trade only top X% of signals.
        
        Args:
            signals: DataFrame with signal probabilities and confidences
            
        Returns:
            Filtered signals DataFrame
        """
        if not self.config.confidence_gate_enabled:
            return signals
        
        # Calculate confidence threshold
        if 'confidence' in signals.columns:
            confidence_values = signals['confidence'].values
            threshold = np.percentile(confidence_values, self.config.confidence_percentile)
            
            # Apply minimum threshold
            final_threshold = max(threshold, self.config.min_confidence_threshold)
            
            # Filter signals
            filtered_signals = signals[signals['confidence'] >= final_threshold].copy()
            
            logger.info(f"Confidence gate: {len(filtered_signals)}/{len(signals)} signals passed "
                       f"(threshold: {final_threshold:.3f})")
            
            return filtered_signals
        
        return signals
    
    def calibrate_probabilities(self, signals: pd.DataFrame, actual_outcomes: pd.Series) -> pd.DataFrame:
        """
        Calibrate signal probabilities using isotonic regression.
        
        Args:
            signals: DataFrame with predicted probabilities
            actual_outcomes: Series with actual trade outcomes (1=win, 0=loss)
            
        Returns:
            DataFrame with calibrated probabilities
        """
        if not self.config.calibration_enabled or len(signals) < self.config.calibration_window:
            return signals
        
        if 'predicted_probability' not in signals.columns:
            return signals
        
        try:
            # Prepare calibration data
            predicted = signals['predicted_probability'].values
            actual = actual_outcomes.values
            
            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predicted, actual)
            
            # Apply calibration
            calibrated_probabilities = iso_reg.transform(predicted)
            
            # Update signals
            signals = signals.copy()
            signals['calibrated_probability'] = calibrated_probabilities
            
            # Calculate Brier score
            brier_score = brier_score_loss(actual, calibrated_probabilities)
            original_brier = brier_score_loss(actual, predicted)
            
            logger.info(f"Probability calibration: Brier score {original_brier:.4f} → {brier_score:.4f}")
            
            # Store calibration model
            self.calibration_models['probability'] = iso_reg
            
            return signals
            
        except Exception as e:
            logger.error(f"Probability calibration failed: {e}")
            return signals
    
    def calculate_brier_score(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate Brier score for probability calibration assessment."""
        return brier_score_loss(actual_outcomes, predicted_probs)
    
    def plot_reliability_curve(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray):
        """Generate reliability curve for probability calibration."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_outcomes, predicted_probs, n_bins=10
            )
            
            # Store for plotting (would implement actual plotting in production)
            reliability_data = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
            
            logger.info("Reliability curve data generated for probability calibration")
            return reliability_data
            
        except Exception as e:
            logger.error(f"Reliability curve generation failed: {e}")
            return None
    
    def apply_regime_conditional_execution(self, signals: pd.DataFrame, regime_data: pd.Series) -> pd.DataFrame:
        """
        Apply regime-specific execution rules.
        
        Args:
            signals: DataFrame with trading signals
            regime_data: Series with regime classifications
            
        Returns:
            DataFrame with regime-adjusted position sizes
        """
        if not self.config.regime_execution_enabled:
            return signals
        
        signals = signals.copy()
        
        # Apply regime multipliers
        for idx, signal in signals.iterrows():
            regime = regime_data.iloc[idx] if idx < len(regime_data) else 'neutral'
            multiplier = self.config.regime_multipliers.get(regime, 0.5)
            
            # Adjust position size
            if 'position_size' in signal:
                signals.at[idx, 'adjusted_position_size'] = signal['position_size'] * multiplier
            else:
                signals.at[idx, 'adjusted_position_size'] = multiplier
            
            # Skip trades in neutral regime
            if regime == 'neutral' and multiplier == 0:
                signals.at[idx, 'skip_trade'] = True
        
        # Log regime distribution
        regime_counts = {}
        for regime in regime_data.unique():
            regime_signals = signals[signals.index.isin(regime_data[regime_data == regime].index)]
            regime_counts[regime] = len(regime_signals)
        
        logger.info(f"Regime distribution: {regime_counts}")
        
        return signals
    
    def prune_features(self, feature_importance: Dict[str, float]) -> List[str]:
        """
        Remove features that contribute less than minimum threshold.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            
        Returns:
            List of features to keep
        """
        if not self.config.feature_pruning_enabled:
            return list(feature_importance.keys())
        
        # Calculate total importance
        total_importance = sum(feature_importance.values())
        
        # Filter features by contribution
        features_to_keep = []
        for feature, importance in feature_importance.items():
            contribution = importance / total_importance
            if contribution >= self.config.min_feature_contribution:
                features_to_keep.append(feature)
            else:
                logger.info(f"Pruning feature '{feature}': contribution {contribution:.2%} < threshold")
        
        logger.info(f"Feature pruning: {len(features_to_keep)}/{len(feature_importance)} features kept")
        
        return features_to_keep
    
    def apply_signal_decay(self, signals: pd.DataFrame, current_trade_count: int) -> pd.DataFrame:
        """
        Apply signal decay model to account for alpha expiration.
        
        Args:
            signals: DataFrame with trading signals
            current_trade_count: Current number of trades since last retrain
            
        Returns:
            DataFrame with decay-adjusted probabilities
        """
        if not self.config.signal_decay_enabled:
            return signals
        
        signals = signals.copy()
        
        # Calculate decay factor
        age = current_trade_count
        decay_factor = np.exp(-self.config.decay_rate * age)
        
        # Apply decay to probabilities
        if 'predicted_probability' in signals.columns:
            signals['decay_adjusted_probability'] = signals['predicted_probability'] * decay_factor
            
            # Skip trades if decayed probability is too low
            signals['skip_due_to_decay'] = signals['decay_adjusted_probability'] < 0.5
        
        logger.info(f"Signal decay applied: factor {decay_factor:.3f} (age: {age} trades)")
        
        return signals
    
    def optimize_signal_quality(self, signals: pd.DataFrame, regime_data: pd.Series = None,
                               actual_outcomes: pd.Series = None, feature_importance: Dict = None) -> pd.DataFrame:
        """
        Apply complete signal quality optimization pipeline.
        
        Args:
            signals: DataFrame with raw trading signals
            regime_data: Series with regime classifications
            actual_outcomes: Series with actual outcomes (for calibration)
            feature_importance: Dict with feature importance scores
            
        Returns:
            Optimized signals DataFrame
        """
        logger.info("Starting signal quality optimization pipeline")
        
        optimized_signals = signals.copy()
        
        # Step 1: Apply confidence gate
        optimized_signals = self.apply_confidence_gate(optimized_signals)
        
        # Step 2: Calibrate probabilities (if outcomes available)
        if actual_outcomes is not None:
            optimized_signals = self.calibrate_probabilities(optimized_signals, actual_outcomes)
        
        # Step 3: Apply regime-conditional execution
        if regime_data is not None:
            optimized_signals = self.apply_regime_conditional_execution(optimized_signals, regime_data)
        
        # Step 4: Apply signal decay
        optimized_signals = self.apply_signal_decay(optimized_signals, self.trade_count)
        
        # Step 5: Feature pruning (returns feature list for downstream use)
        if feature_importance is not None:
            self.feature_importance = feature_importance
            features_to_keep = self.prune_features(feature_importance)
            logger.info(f"Features to keep: {features_to_keep}")
        
        # Update trade count
        self.trade_count += len(optimized_signals)
        
        logger.info(f"Signal quality optimization complete: {len(optimized_signals)} signals remaining")
        
        return optimized_signals
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization effects."""
        return {
            'trade_count': self.trade_count,
            'calibration_models': list(self.calibration_models.keys()),
            'feature_importance': self.feature_importance,
            'config': {
                'confidence_gate': self.config.confidence_gate_enabled,
                'calibration': self.config.calibration_enabled,
                'regime_execution': self.config.regime_execution_enabled,
                'feature_pruning': self.config.feature_pruning_enabled,
                'signal_decay': self.config.signal_decay_enabled
            }
        }
