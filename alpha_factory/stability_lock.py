"""
Stability Lock-In for Alpha Factory v2.0 Production Baseline

Phase 0: Ensure current edge cannot be accidentally destroyed
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Frozen production configuration."""
    # Confidence thresholds (LOCKED)
    confidence_gate_percentile: float = 0.70
    min_confidence_threshold: float = 0.75
    
    # Regime rules (LOCKED)
    regime_filters: Dict[str, bool] = None
    regime_multipliers: Dict[str, float] = None
    regime_confidence_thresholds: Dict[str, float] = None
    
    # Feature set (LOCKED)
    core_features: List[str] = None
    enabled_features: List[str] = None
    
    # Position sizing (LOCKED)
    kelly_fraction: float = 0.25
    max_position_size: float = 0.05
    min_position_size: float = 0.01
    
    # Exit logic (LOCKED)
    trailing_distance: float = 0.001
    trailing_activation: float = 0.002
    partial_exit_levels: List[float] = None
    
    # Performance thresholds (LOCKED)
    min_win_rate: float = 0.68
    min_expectancy: float = 55.0
    max_drawdown: float = 0.15
    max_regime_variance: float = 0.04
    
    def __post_init__(self):
        if self.regime_filters is None:
            self.regime_filters = {
                'bullish': True,
                'bearish': True,
                'neutral': False,
                'volatile': True
            }
        
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                'bullish': 1.0,
                'bearish': 1.0,
                'neutral': 0.0,
                'volatile': 0.5
            }
        
        if self.regime_confidence_thresholds is None:
            self.regime_confidence_thresholds = {
                'bullish': 0.75,
                'bearish': 0.75,
                'neutral': 0.82,
                'volatile': 0.70
            }
        
        if self.core_features is None:
            self.core_features = [
                'close', 'high', 'low', 'volume',
                'atr', 'rsi', 'macd', 'adx'
            ]
        
        if self.enabled_features is None:
            self.enabled_features = [
                'close', 'high', 'low', 'volume',
                'atr', 'rsi', 'macd', 'adx',
                'feature_1', 'feature_2', 'feature_3'
            ]
        
        if self.partial_exit_levels is None:
            self.partial_exit_levels = [0.5, 1.0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            'confidence_gate_percentile': self.confidence_gate_percentile,
            'min_confidence_threshold': self.min_confidence_threshold,
            'regime_filters': self.regime_filters,
            'regime_multipliers': self.regime_multipliers,
            'regime_confidence_thresholds': self.regime_confidence_thresholds,
            'core_features': sorted(self.core_features),
            'enabled_features': sorted(self.enabled_features),
            'kelly_fraction': self.kelly_fraction,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'trailing_distance': self.trailing_distance,
            'trailing_activation': self.trailing_activation,
            'partial_exit_levels': self.partial_exit_levels,
            'min_win_rate': self.min_win_rate,
            'min_expectancy': self.min_expectancy,
            'max_drawdown': self.max_drawdown,
            'max_regime_variance': self.max_regime_variance
        }
    
    def get_hash(self) -> str:
        """Get unique hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class StabilityLock:
    """Stability lock-in system for Alpha Factory v2.0."""
    
    def __init__(self):
        self.production_config = ProductionConfig()
        self.config_hash = self.production_config.get_hash()
        self.baseline_tag = "alpha_factory_v2.0_production_baseline"
        
        # Performance tracking
        self.performance_history = []
        self.invariant_violations = []
        
        # Decision logging
        self.decision_log = []
        
        logger.info(f"Stability lock initialized with config hash: {self.config_hash}")
        logger.info(f"Baseline tag: {self.baseline_tag}")
    
    def validate_configuration(self, test_config: ProductionConfig) -> bool:
        """
        Validate that configuration matches production baseline.
        
        Args:
            test_config: Configuration to validate
            
        Returns:
            True if configuration matches baseline
        """
        test_hash = test_config.get_hash()
        
        if test_hash != self.config_hash:
            logger.error(f"Configuration mismatch! Expected: {self.config_hash}, Got: {test_hash}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def check_performance_invariants(self, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check performance invariants against baseline thresholds.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary of invariant checks
        """
        invariants = {}
        
        # Win rate invariant
        win_rate = current_metrics.get('win_rate', 0)
        invariants['win_rate'] = win_rate >= self.production_config.min_win_rate
        
        # Expectancy invariant
        expectancy = current_metrics.get('expectancy', 0)
        invariants['expectancy'] = expectancy >= self.production_config.min_expectancy
        
        # Drawdown invariant
        max_drawdown = current_metrics.get('max_drawdown', 0)
        invariants['max_drawdown'] = max_drawdown <= self.production_config.max_drawdown
        
        # Regime variance invariant
        regime_variance = current_metrics.get('regime_variance', 0)
        invariants['regime_variance'] = regime_variance <= self.production_config.max_regime_variance
        
        # Log violations
        for invariant, passed in invariants.items():
            if not passed:
                violation = {
                    'timestamp': datetime.now(),
                    'invariant': invariant,
                    'current_value': current_metrics.get(invariant, 0),
                    'threshold': getattr(self.production_config, f'min_{invariant}' if 'min_' in invariant else f'max_{invariant}', 0)
                }
                self.invariant_violations.append(violation)
                logger.warning(f"Invariant violation: {invariant}")
        
        return invariants
    
    def log_decision_internals(self, decision_data: Dict[str, Any]) -> None:
        """
        Log detailed decision internals for analysis.
        
        Args:
            decision_data: Decision data with internals
        """
        log_entry = {
            'timestamp': datetime.now(),
            'p_regime': decision_data.get('p_regime', 0),
            'p_decision_pre_calibration': decision_data.get('p_decision_pre', 0),
            'p_decision_post_calibration': decision_data.get('p_decision_post', 0),
            'expected_value': decision_data.get('expected_value', 0),
            'position_size': decision_data.get('position_size', 0),
            'exit_reason': decision_data.get('exit_reason', ''),
            'regime': decision_data.get('regime', ''),
            'decision': decision_data.get('decision', ''),
            'confidence': decision_data.get('confidence', 0)
        }
        
        self.decision_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.decision_log) > 10000:
            self.decision_log = self.decision_log[-5000:]
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report."""
        return {
            'baseline_tag': self.baseline_tag,
            'config_hash': self.config_hash,
            'production_config': self.production_config.to_dict(),
            'total_decisions_logged': len(self.decision_log),
            'invariant_violations': len(self.invariant_violations),
            'recent_violations': self.invariant_violations[-10:] if self.invariant_violations else [],
            'performance_invariants': {
                'min_win_rate': self.production_config.min_win_rate,
                'min_expectancy': self.production_config.min_expectancy,
                'max_drawdown': self.production_config.max_drawdown,
                'max_regime_variance': self.production_config.max_regime_variance
            }
        }
    
    def export_baseline(self) -> Dict[str, Any]:
        """Export baseline configuration for version control."""
        return {
            'tag': self.baseline_tag,
            'config_hash': self.config_hash,
            'timestamp': datetime.now().isoformat(),
            'config': self.production_config.to_dict(),
            'invariants': {
                'min_win_rate': self.production_config.min_win_rate,
                'min_expectancy': self.production_config.min_expectancy,
                'max_drawdown': self.production_config.max_drawdown,
                'max_regime_variance': self.production_config.max_regime_variance
            }
        }

# Global stability lock instance
stability_lock = StabilityLock()
