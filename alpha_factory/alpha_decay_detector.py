"""
Alpha Decay Detector for Alpha Factory

Phase 3: Live Alpha Decay Detection (Risk Preservation)

Goal: Reduce size before drawdown starts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class DecayConfig:
    """Configuration for alpha decay detection."""
    # Performance monitoring windows
    win_rate_window: int = 100  # Trades for win rate calculation
    ev_degradation_window: int = 50  # Window for EV degradation
    probability_calibration_window: int = 200  # Window for calibration drift
    
    # Decay thresholds
    win_rate_decay_threshold: float = 0.05  # 5% win rate decay triggers alert
    ev_degradation_threshold: float = 0.20  # 20% EV degradation triggers alert
    calibration_drift_threshold: float = 0.10  # 10% calibration drift triggers alert
    
    # Z-score thresholds
    win_rate_z_threshold: float = 2.0  # 2 standard deviations
    ev_z_threshold: float = 2.0
    calibration_z_threshold: float = 2.0
    
    # Response rules
    mild_decay_response: str = 'reduce_size'  # Options: 'reduce_size', 'raise_threshold', 'monitor'
    strong_decay_response: str = 'raise_threshold'
    severe_decay_response: str = 'pause_strategy'
    
    # Response parameters
    size_reduction_factor: float = 0.7  # Reduce to 70% of current size
    threshold_increase_factor: float = 1.2  # Increase thresholds by 20%
    pause_duration_hours: int = 24  # Pause for 24 hours
    
    # Regime-specific tracking
    regime_specific_decay: bool = True
    regime_decay_thresholds: Dict[str, float] = None

class DecayMetrics:
    """Container for decay metrics."""
    
    def __init__(self):
        self.win_rate_current = 0.0
        self.win_rate_baseline = 0.0
        self.win_rate_decay = 0.0
        self.win_rate_z_score = 0.0
        
        self.ev_current = 0.0
        self.ev_baseline = 0.0
        self.ev_degradation = 0.0
        self.ev_z_score = 0.0
        
        self.calibration_current = 0.0
        self.calibration_baseline = 0.0
        self.calibration_drift = 0.0
        self.calibration_z_score = 0.0
        
        self.overall_decay_score = 0.0
        self.decay_level = 'none'  # 'none', 'mild', 'strong', 'severe'
        self.regime_decay = {}

class AlphaDecayDetector:
    """Professional alpha decay detection for Alpha Factory."""
    
    def __init__(self, config: DecayConfig = None):
        self.config = config or DecayConfig()
        
        # Initialize regime-specific thresholds if not provided
        if self.config.regime_decay_thresholds is None:
            self.config.regime_decay_thresholds = {
                'bullish': 0.06,   # 6% threshold for trends
                'bearish': 0.06,   # 6% threshold for trends
                'neutral': 0.04,   # 4% threshold for ranges
                'volatile': 0.08    # 8% threshold for volatile
            }
        
        # Performance tracking
        self.trade_history = []
        self.ev_history = []
        self.probability_history = []
        self.regime_history = []
        
        # Baseline metrics
        self.baseline_metrics = {
            'win_rate': 0.65,
            'ev': 25.0,
            'calibration': 0.05
        }
        
        # Decay tracking
        self.decay_alerts = []
        self.current_decay_state = 'normal'
        self.last_decay_check = datetime.now()
        
        # Response tracking
        self.response_history = []
        self.active_responses = {}
        
        logger.info("Alpha decay detector initialized")
    
    def update_trade_result(self, pnl: float, probability: float, 
                          predicted_probability: float, regime: str, 
                          expected_value: float):
        """
        Update trade results for decay analysis.
        
        Args:
            pnl: Trade P&L result
            probability: Actual win probability (0 or 1)
            predicted_probability: Model predicted probability
            regime: Trade regime
            expected_value: Expected value calculation
        """
        trade_record = {
            'timestamp': datetime.now(),
            'pnl': pnl,
            'probability': probability,
            'predicted_probability': predicted_probability,
            'regime': regime,
            'expected_value': expected_value,
            'calibration_error': probability - predicted_probability
        }
        
        self.trade_history.append(trade_record)
        self.ev_history.append(expected_value)
        self.probability_history.append(predicted_probability)
        self.regime_history.append(regime)
        
        # Keep history manageable
        max_history = max(self.config.win_rate_window, 
                         self.config.ev_degradation_window,
                         self.config.probability_calibration_window) * 2
        
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]
            self.ev_history = self.ev_history[-max_history:]
            self.probability_history = self.probability_history[-max_history:]
            self.regime_history = self.regime_history[-max_history:]
        
        logger.debug(f"Trade result updated: P&L=${pnl:.2f}, EV=${expected_value:.2f}")
    
    def calculate_win_rate_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate current win rate, baseline, and decay.
        
        Returns:
            Tuple of (current_win_rate, baseline_win_rate, win_rate_decay)
        """
        if len(self.trade_history) < 20:
            return 0.0, self.baseline_metrics['win_rate'], 0.0
        
        # Current win rate
        recent_trades = self.trade_history[-self.config.win_rate_window:]
        wins = [t for t in recent_trades if t['pnl'] > 0]
        current_win_rate = len(wins) / len(recent_trades)
        
        # Baseline win rate (historical average)
        baseline_trades = self.trade_history[:-self.config.win_rate_window] if len(self.trade_history) > self.config.win_rate_window else self.trade_history[:-len(recent_trades)]
        
        if baseline_trades:
            baseline_wins = [t for t in baseline_trades if t['pnl'] > 0]
            baseline_win_rate = len(baseline_wins) / len(baseline_trades)
        else:
            baseline_win_rate = self.baseline_metrics['win_rate']
        
        # Calculate decay
        if baseline_win_rate > 0:
            win_rate_decay = (baseline_win_rate - current_win_rate) / baseline_win_rate
        else:
            win_rate_decay = 0.0
        
        return current_win_rate, baseline_win_rate, win_rate_decay
    
    def calculate_ev_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate current EV, baseline, and degradation.
        
        Returns:
            Tuple of (current_ev, baseline_ev, ev_degradation)
        """
        if len(self.ev_history) < 20:
            return 0.0, self.baseline_metrics['ev'], 0.0
        
        # Current EV
        recent_ev = self.ev_history[-self.config.ev_degradation_window:]
        current_ev = np.mean(recent_ev)
        
        # Baseline EV
        baseline_ev = self.ev_history[:-self.config.ev_degradation_window] if len(self.ev_history) > self.config.ev_degradation_window else self.ev_history[:-len(recent_ev)]
        
        if baseline_ev:
            baseline_ev = np.mean(baseline_ev)
        else:
            baseline_ev = self.baseline_metrics['ev']
        
        # Calculate degradation
        if baseline_ev > 0:
            ev_degradation = (baseline_ev - current_ev) / baseline_ev
        else:
            ev_degradation = 0.0
        
        return current_ev, baseline_ev, ev_degradation
    
    def calculate_calibration_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate current calibration, baseline, and drift.
        
        Returns:
            Tuple of (current_calibration, baseline_calibration, calibration_drift)
        """
        if len(self.trade_history) < 20:
            return 0.0, self.baseline_metrics['calibration'], 0.0
        
        # Current calibration (mean absolute error)
        recent_trades = self.trade_history[-self.config.probability_calibration_window:]
        calibration_errors = [abs(t['calibration_error']) for t in recent_trades]
        current_calibration = np.mean(calibration_errors)
        
        # Baseline calibration
        baseline_trades = self.trade_history[:-self.config.probability_calibration_window] if len(self.trade_history) > self.config.probability_calibration_window else self.trade_history[:-len(recent_trades)]
        
        if baseline_trades:
            baseline_errors = [abs(t['calibration_error']) for t in baseline_trades]
            baseline_calibration = np.mean(baseline_errors)
        else:
            baseline_calibration = self.baseline_metrics['calibration']
        
        # Calculate drift
        if baseline_calibration > 0:
            calibration_drift = (current_calibration - baseline_calibration) / baseline_calibration
        else:
            calibration_drift = 0.0
        
        return current_calibration, baseline_calibration, calibration_drift
    
    def calculate_z_scores(self, current: float, baseline: float, 
                          historical_values: List[float]) -> float:
        """
        Calculate Z-score for metric deviation.
        
        Args:
            current: Current metric value
            baseline: Baseline metric value
            historical_values: Historical values for std calculation
            
        Returns:
            Z-score
        """
        if len(historical_values) < 10:
            return 0.0
        
        # Calculate deviation from baseline
        deviation = current - baseline
        
        # Calculate standard deviation of historical values
        std_dev = np.std(historical_values)
        
        if std_dev == 0:
            return 0.0
        
        z_score = deviation / std_dev
        return z_score
    
    def calculate_regime_decay(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate decay metrics by regime.
        
        Returns:
            Dictionary of regime-specific decay metrics
        """
        if not self.config.regime_specific_decay:
            return {}
        
        regime_decay = {}
        
        for regime in ['bullish', 'bearish', 'neutral', 'volatile']:
            regime_trades = [t for t in self.trade_history if t.get('regime') == regime]
            
            if len(regime_trades) < 10:
                continue
            
            # Calculate regime-specific win rate
            wins = [t for t in regime_trades if t['pnl'] > 0]
            regime_win_rate = len(wins) / len(regime_trades)
            
            # Calculate regime-specific EV
            regime_evs = [t['expected_value'] for t in regime_trades]
            regime_ev = np.mean(regime_evs)
            
            # Compare to overall baseline
            win_rate_decay = max(0, (self.baseline_metrics['win_rate'] - regime_win_rate) / self.baseline_metrics['win_rate'])
            ev_degradation = max(0, (self.baseline_metrics['ev'] - regime_ev) / self.baseline_metrics['ev'])
            
            regime_decay[regime] = {
                'win_rate': regime_win_rate,
                'ev': regime_ev,
                'win_rate_decay': win_rate_decay,
                'ev_degradation': ev_degradation,
                'trade_count': len(regime_trades)
            }
        
        return regime_decay
    
    def detect_decay(self) -> DecayMetrics:
        """
        Detect alpha decay across all metrics.
        
        Returns:
            DecayMetrics object with current decay state
        """
        metrics = DecayMetrics()
        
        # Calculate basic metrics
        metrics.win_rate_current, metrics.win_rate_baseline, metrics.win_rate_decay = self.calculate_win_rate_metrics()
        metrics.ev_current, metrics.ev_baseline, metrics.ev_degradation = self.calculate_ev_metrics()
        metrics.calibration_current, metrics.calibration_baseline, metrics.calibration_drift = self.calculate_calibration_metrics()
        
        # Calculate Z-scores
        if len(self.trade_history) > 50:
            # Win rate Z-score
            win_rates = []
            for i in range(len(self.trade_history) - 20):
                window = self.trade_history[i:i+20]
                wins = [t for t in window if t['pnl'] > 0]
                win_rates.append(len(wins) / len(window))
            
            metrics.win_rate_z_score = self.calculate_z_scores(
                metrics.win_rate_current, metrics.win_rate_baseline, win_rates
            )
            
            # EV Z-score
            ev_windows = []
            for i in range(len(self.ev_history) - 20):
                ev_windows.append(np.mean(self.ev_history[i:i+20]))
            
            metrics.ev_z_score = self.calculate_z_scores(
                metrics.ev_current, metrics.ev_baseline, ev_windows
            )
            
            # Calibration Z-score
            cal_windows = []
            for i in range(len(self.trade_history) - 20):
                window = self.trade_history[i:i+20]
                cal_errors = [abs(t['calibration_error']) for t in window]
                cal_windows.append(np.mean(cal_errors))
            
            metrics.calibration_z_score = self.calculate_z_scores(
                metrics.calibration_current, metrics.calibration_baseline, cal_windows
            )
        
        # Calculate regime-specific decay
        metrics.regime_decay = self.calculate_regime_decay()
        
        # Determine overall decay level
        decay_indicators = []
        
        # Win rate decay
        if metrics.win_rate_decay > self.config.win_rate_decay_threshold:
            decay_indicators.append('win_rate')
        
        # EV degradation
        if metrics.ev_degradation > self.config.ev_degradation_threshold:
            decay_indicators.append('ev')
        
        # Calibration drift
        if metrics.calibration_drift > self.config.calibration_drift_threshold:
            decay_indicators.append('calibration')
        
        # Z-score indicators
        if abs(metrics.win_rate_z_score) > self.config.win_rate_z_threshold:
            decay_indicators.append('win_rate_z')
        
        if abs(metrics.ev_z_score) > self.config.ev_z_threshold:
            decay_indicators.append('ev_z')
        
        if abs(metrics.calibration_z_score) > self.config.calibration_z_threshold:
            decay_indicators.append('calibration_z')
        
        # Determine decay level
        num_indicators = len(decay_indicators)
        
        if num_indicators == 0:
            metrics.decay_level = 'none'
        elif num_indicators <= 2:
            metrics.decay_level = 'mild'
        elif num_indicators <= 4:
            metrics.decay_level = 'strong'
        else:
            metrics.decay_level = 'severe'
        
        # Calculate overall decay score (0-1)
        decay_score = 0.0
        decay_score += min(1.0, metrics.win_rate_decay / self.config.win_rate_decay_threshold)
        decay_score += min(1.0, metrics.ev_degradation / self.config.ev_degradation_threshold)
        decay_score += min(1.0, metrics.calibration_drift / self.config.calibration_drift_threshold)
        decay_score += min(1.0, abs(metrics.win_rate_z_score) / self.config.win_rate_z_threshold)
        decay_score += min(1.0, abs(metrics.ev_z_score) / self.config.ev_z_threshold)
        decay_score += min(1.0, abs(metrics.calibration_z_score) / self.config.calibration_z_threshold)
        
        metrics.overall_decay_score = decay_score / 6.0
        
        # Update decay state
        self.current_decay_state = metrics.decay_level
        self.last_decay_check = datetime.now()
        
        logger.info(f"Decay detection complete: Level={metrics.decay_level}, Score={metrics.overall_decay_score:.3f}")
        
        return metrics
    
    def generate_decay_response(self, decay_metrics: DecayMetrics) -> Dict[str, Any]:
        """
        Generate appropriate response to detected decay.
        
        Args:
            decay_metrics: Current decay metrics
            
        Returns:
            Dictionary with response actions
        """
        response = {
            'decay_level': decay_metrics.decay_level,
            'decay_score': decay_metrics.overall_decay_score,
            'timestamp': datetime.now(),
            'actions': [],
            'parameters': {}
        }
        
        if decay_metrics.decay_level == 'none':
            response['actions'].append('monitor')
            return response
        
        elif decay_metrics.decay_level == 'mild':
            if self.config.mild_decay_response == 'reduce_size':
                response['actions'].append('reduce_position_size')
                response['parameters']['size_factor'] = self.config.size_reduction_factor
            elif self.config.mild_decay_response == 'raise_threshold':
                response['actions'].append('raise_ev_threshold')
                response['parameters']['threshold_factor'] = self.config.threshold_increase_factor
            else:
                response['actions'].append('monitor')
        
        elif decay_metrics.decay_level == 'strong':
            if self.config.strong_decay_response == 'raise_threshold':
                response['actions'].append('raise_ev_threshold')
                response['parameters']['threshold_factor'] = self.config.threshold_increase_factor
                response['actions'].append('reduce_position_size')
                response['parameters']['size_factor'] = self.config.size_reduction_factor * 0.8
            else:
                response['actions'].append('pause_new_trades')
                response['parameters']['pause_duration'] = self.config.pause_duration_hours // 2
        
        elif decay_metrics.decay_level == 'severe':
            if self.config.severe_decay_response == 'pause_strategy':
                response['actions'].append('pause_all_trading')
                response['parameters']['pause_duration'] = self.config.pause_duration_hours
            else:
                response['actions'].append('emergency_stop')
        
        # Add regime-specific responses if enabled
        if self.config.regime_specific_decay and decay_metrics.regime_decay:
            for regime, regime_metrics in decay_metrics.regime_decay.items():
                threshold = self.config.regime_decay_thresholds.get(regime, 0.05)
                
                if regime_metrics['win_rate_decay'] > threshold:
                    response['actions'].append(f'reduce_{regime}_exposure')
                    response['parameters'][f'{regime}_factor'] = 0.5
        
        # Store response
        self.response_history.append(response)
        
        logger.warning(f"Decay response generated: Level={decay_metrics.decay_level}, Actions={response['actions']}")
        
        return response
    
    def get_decay_report(self) -> Dict[str, Any]:
        """Get comprehensive decay detection report."""
        current_metrics = self.detect_decay()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_decay_state': self.current_decay_state,
            'last_decay_check': self.last_decay_check.isoformat(),
            'metrics': {
                'win_rate': {
                    'current': current_metrics.win_rate_current,
                    'baseline': current_metrics.win_rate_baseline,
                    'decay': current_metrics.win_rate_decay,
                    'z_score': current_metrics.win_rate_z_score
                },
                'expected_value': {
                    'current': current_metrics.ev_current,
                    'baseline': current_metrics.ev_baseline,
                    'degradation': current_metrics.ev_degradation,
                    'z_score': current_metrics.ev_z_score
                },
                'calibration': {
                    'current': current_metrics.calibration_current,
                    'baseline': current_metrics.calibration_baseline,
                    'drift': current_metrics.calibration_drift,
                    'z_score': current_metrics.calibration_z_score
                }
            },
            'overall_decay_score': current_metrics.overall_decay_score,
            'decay_level': current_metrics.decay_level,
            'regime_decay': current_metrics.regime_decay,
            'decay_alerts': len(self.decay_alerts),
            'response_history': self.response_history[-5:],  # Last 5 responses
            'config': {
                'win_rate_threshold': self.config.win_rate_decay_threshold,
                'ev_threshold': self.config.ev_degradation_threshold,
                'calibration_threshold': self.config.calibration_drift_threshold,
                'regime_specific': self.config.regime_specific_decay
            }
        }
