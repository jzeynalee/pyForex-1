"""
Probability Calibration for Alpha Factory

This module implements professional probability calibration techniques:
1. Brier Score measurement for calibration assessment
2. Platt scaling (logistic regression) for probability calibration
3. Isotonic regression for non-parametric calibration
4. Reliability curve visualization and analysis
5. Walk-forward calibration to prevent overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""
    # Calibration method
    method: str = 'isotonic'  # 'isotonic', 'platt', or 'ensemble'
    
    # Walk-forward parameters
    calibration_window: int = 1000  # Number of trades for calibration
    min_calibration_samples: int = 100  # Minimum samples for calibration
    
    # Validation parameters
    validation_split: float = 0.2  # Fraction for validation
    brier_score_threshold: float = 0.25  # Maximum acceptable Brier score
    
    # Platt scaling parameters
    platt_regularization: float = 1.0  # L2 regularization for Platt scaling
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = None  # Weights for ensemble methods
    
    # Update frequency
    update_frequency: int = 100  # Update calibration every N trades

class ProbabilityCalibrator:
    """Professional probability calibration for Alpha Factory."""
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.calibration_models = {}
        self.calibration_history = []
        self.trade_count = 0
        self.last_update = 0
        
        # Initialize ensemble weights if not provided
        if self.config.ensemble_weights is None:
            self.config.ensemble_weights = {
                'isotonic': 0.5,
                'platt': 0.3,
                'raw': 0.2
            }
    
    def calculate_brier_score(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Calculate Brier score for probability calibration assessment.
        
        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Actual binary outcomes (1=win, 0=loss)
            
        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        return brier_score_loss(actual_outcomes, predicted_probs)
    
    def calculate_log_loss(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate log loss for probability assessment."""
        return log_loss(actual_outcomes, predicted_probs)
    
    def generate_reliability_curve(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray, 
                                 n_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate reliability curve for probability calibration visualization.
        
        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Actual binary outcomes
            n_bins: Number of bins for reliability curve
            
        Returns:
            Dictionary with reliability curve data
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_outcomes, predicted_probs, n_bins=n_bins, strategy='quantile'
            )
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'calibration_error': calibration_error,
                'n_bins': n_bins
            }
            
        except Exception as e:
            logger.error(f"Reliability curve generation failed: {e}")
            return None
    
    def fit_platt_scaling(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> LogisticRegression:
        """
        Fit Platt scaling (logistic regression) for probability calibration.
        
        Args:
            predicted_probs: Raw predicted probabilities
            actual_outcomes: Actual binary outcomes
            
        Returns:
            Fitted logistic regression model
        """
        try:
            # Platt scaling: logistic regression on log-odds
            log_odds = np.log(predicted_probs / (1 - predicted_probs + 1e-15))
            
            # Fit logistic regression with regularization
            platt_model = LogisticRegression(
                C=1/self.config.platt_regularization,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000
            )
            
            platt_model.fit(log_odds.reshape(-1, 1), actual_outcomes)
            
            logger.info(f"Platt scaling fitted with regularization: {self.config.platt_regularization}")
            
            return platt_model
            
        except Exception as e:
            logger.error(f"Platt scaling failed: {e}")
            return None
    
    def fit_isotonic_regression(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> IsotonicRegression:
        """
        Fit isotonic regression for non-parametric probability calibration.
        
        Args:
            predicted_probs: Raw predicted probabilities
            actual_outcomes: Actual binary outcomes
            
        Returns:
            Fitted isotonic regression model
        """
        try:
            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predicted_probs, actual_outcomes)
            
            logger.info("Isotonic regression fitted successfully")
            
            return iso_reg
            
        except Exception as e:
            logger.error(f"Isotonic regression failed: {e}")
            return None
    
    def calibrate_probabilities(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                               method: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calibrate probabilities using specified method.
        
        Args:
            predicted_probs: Raw predicted probabilities
            actual_outcomes: Actual binary outcomes
            method: Calibration method ('isotonic', 'platt', 'ensemble')
            
        Returns:
            Tuple of calibrated probabilities and calibration metrics
        """
        if method is None:
            method = self.config.method
        
        if len(predicted_probs) < self.config.min_calibration_samples:
            logger.warning(f"Insufficient samples for calibration: {len(predicted_probs)} < {self.config.min_calibration_samples}")
            return predicted_probs.copy(), {'method': 'raw', 'brier_score': self.calculate_brier_score(predicted_probs, actual_outcomes)}
        
        # Calculate original metrics
        original_brier = self.calculate_brier_score(predicted_probs, actual_outcomes)
        original_log_loss = self.calculate_log_loss(predicted_probs, actual_outcomes)
        
        calibrated_probs = predicted_probs.copy()
        calibration_metrics = {
            'method': method,
            'original_brier': original_brier,
            'original_log_loss': original_log_loss,
            'samples': len(predicted_probs)
        }
        
        if method == 'isotonic':
            # Fit isotonic regression
            iso_model = self.fit_isotonic_regression(predicted_probs, actual_outcomes)
            if iso_model is not None:
                calibrated_probs = iso_model.transform(predicted_probs)
                self.calibration_models['isotonic'] = iso_model
                
                calibrated_brier = self.calculate_brier_score(calibrated_probs, actual_outcomes)
                calibration_metrics['calibrated_brier'] = calibrated_brier
                calibration_metrics['brier_improvement'] = original_brier - calibrated_brier
                
        elif method == 'platt':
            # Fit Platt scaling
            platt_model = self.fit_platt_scaling(predicted_probs, actual_outcomes)
            if platt_model is not None:
                log_odds = np.log(predicted_probs / (1 - predicted_probs + 1e-15))
                calibrated_log_odds = platt_model.predict_proba(log_odds.reshape(-1, 1))[:, 1]
                calibrated_probs = calibrated_log_odds
                self.calibration_models['platt'] = platt_model
                
                calibrated_brier = self.calculate_brier_score(calibrated_probs, actual_outcomes)
                calibration_metrics['calibrated_brier'] = calibrated_brier
                calibration_metrics['brier_improvement'] = original_brier - calibrated_brier
                
        elif method == 'ensemble':
            # Ensemble method: combine multiple calibrations
            iso_model = self.fit_isotonic_regression(predicted_probs, actual_outcomes)
            platt_model = self.fit_platt_scaling(predicted_probs, actual_outcomes)
            
            ensemble_probs = np.zeros_like(predicted_probs)
            
            if iso_model is not None:
                iso_probs = iso_model.transform(predicted_probs)
                ensemble_probs += self.config.ensemble_weights['isotonic'] * iso_probs
                self.calibration_models['isotonic'] = iso_model
            
            if platt_model is not None:
                log_odds = np.log(predicted_probs / (1 - predicted_probs + 1e-15))
                platt_probs = platt_model.predict_proba(log_odds.reshape(-1, 1))[:, 1]
                ensemble_probs += self.config.ensemble_weights['platt'] * platt_probs
                self.calibration_models['platt'] = platt_model
            
            # Add raw probabilities
            ensemble_probs += self.config.ensemble_weights['raw'] * predicted_probs
            
            calibrated_probs = ensemble_probs
            calibrated_brier = self.calculate_brier_score(calibrated_probs, actual_outcomes)
            calibration_metrics['calibrated_brier'] = calibrated_brier
            calibration_metrics['brier_improvement'] = original_brier - calibrated_brier
            calibration_metrics['ensemble_weights'] = self.config.ensemble_weights
        
        # Ensure probabilities are in valid range
        calibrated_probs = np.clip(calibrated_probs, 0.01, 0.99)
        
        # Generate reliability curve
        reliability_data = self.generate_reliability_curve(predicted_probs, actual_outcomes)
        if reliability_data:
            calibration_metrics['reliability_curve'] = reliability_data
        
        logger.info(f"Probability calibration completed: {method}")
        logger.info(f"  Original Brier: {original_brier:.4f}")
        if 'calibrated_brier' in calibration_metrics:
            logger.info(f"  Calibrated Brier: {calibration_metrics['calibrated_brier']:.4f}")
            logger.info(f"  Improvement: {calibration_metrics['brier_improvement']:.4f}")
        
        return calibrated_probs, calibration_metrics
    
    def walk_forward_calibration(self, trade_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform walk-forward calibration to prevent overfitting.
        
        Args:
            trade_history: DataFrame with trade history containing 'predicted_prob' and 'outcome'
            
        Returns:
            Dictionary with calibration results
        """
        if len(trade_history) < self.config.calibration_window:
            logger.warning(f"Insufficient data for walk-forward calibration: {len(trade_history)}")
            return {'status': 'insufficient_data'}
        
        # Split data for walk-forward validation
        train_size = int(len(trade_history) * (1 - self.config.validation_split))
        train_data = trade_history.iloc[:train_size]
        val_data = trade_history.iloc[train_size:]
        
        # Extract probabilities and outcomes
        train_probs = train_data['predicted_prob'].values
        train_outcomes = train_data['outcome'].values
        val_probs = val_data['predicted_prob'].values
        val_outcomes = val_data['outcome'].values
        
        # Calibrate on training data
        calibrated_train_probs, train_metrics = self.calibrate_probabilities(
            train_probs, train_outcomes, self.config.method
        )
        
        # Apply calibration to validation data
        if self.config.method == 'isotonic' and 'isotonic' in self.calibration_models:
            calibrated_val_probs = self.calibration_models['isotonic'].transform(val_probs)
        elif self.config.method == 'platt' and 'platt' in self.calibration_models:
            log_odds = np.log(val_probs / (1 - val_probs + 1e-15))
            calibrated_val_probs = self.calibration_models['platt'].predict_proba(log_odds.reshape(-1, 1))[:, 1]
        else:
            calibrated_val_probs = val_probs.copy()
        
        # Calculate validation metrics
        val_brier_original = self.calculate_brier_score(val_probs, val_outcomes)
        val_brier_calibrated = self.calculate_brier_score(calibrated_val_probs, val_outcomes)
        
        results = {
            'status': 'success',
            'train_metrics': train_metrics,
            'validation_brier_original': val_brier_original,
            'validation_brier_calibrated': val_brier_calibrated,
            'validation_improvement': val_brier_original - val_brier_calibrated,
            'calibration_method': self.config.method
        }
        
        # Store calibration history
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'trade_count': len(trade_history),
            'results': results
        })
        
        logger.info(f"Walk-forward calibration completed:")
        logger.info(f"  Validation Brier improvement: {results['validation_improvement']:.4f}")
        
        return results
    
    def update_calibration(self, new_trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Update calibration with new trade data.
        
        Args:
            new_trades: DataFrame with new trades
            
        Returns:
            Update results
        """
        self.trade_count += len(new_trades)
        
        # Check if update is needed
        if (self.trade_count - self.last_update) < self.config.update_frequency:
            return {'status': 'no_update_needed', 'reason': 'update_frequency_not_met'}
        
        # Perform walk-forward calibration
        results = self.walk_forward_calibration(new_trades)
        self.last_update = self.trade_count
        
        return results
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration performance."""
        if not self.calibration_history:
            return {'status': 'no_calibration_history'}
        
        latest_calibration = self.calibration_history[-1]
        
        summary = {
            'total_calibrations': len(self.calibration_history),
            'latest_calibration': latest_calibration['timestamp'],
            'total_trades': self.trade_count,
            'calibration_method': self.config.method,
            'latest_results': latest_calibration['results']
        }
        
        # Calculate average improvement
        if len(self.calibration_history) > 1:
            improvements = [cal['results'].get('validation_improvement', 0) 
                          for cal in self.calibration_history 
                          if 'validation_improvement' in cal['results']]
            if improvements:
                summary['average_improvement'] = np.mean(improvements)
                summary['improvement_std'] = np.std(improvements)
        
        return summary
    
    def plot_reliability_diagram(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                                save_path: str = None) -> bool:
        """
        Plot reliability diagram for visualization.
        
        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Actual outcomes
            save_path: Path to save the plot
            
        Returns:
            True if successful, False otherwise
        """
        try:
            reliability_data = self.generate_reliability_curve(predicted_probs, actual_outcomes)
            
            if reliability_data is None:
                return False
            
            plt.figure(figsize=(10, 6))
            
            # Plot perfect calibration line
            plt.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
            
            # Plot reliability curve
            plt.plot(reliability_data['mean_predicted_value'], 
                    reliability_data['fraction_of_positives'], 
                    's-', label='Reliability curve')
            
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positive outcomes')
            plt.title('Reliability Diagram')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Reliability diagram saved to {save_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to plot reliability diagram: {e}")
            return False
