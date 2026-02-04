# alpha_factory/alpha_factory.py
"""
Main Alpha Factory System Implementation.

This is the orchestrator class that combines all components:
1. Market structure identification
2. Feature engineering
3. Causal analysis
4. Decision making
5. Alpha generation and execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path

from .market_data import MarketData
from .causal_analysis import compute_causality, get_top_causal_features, create_causal_network
from .decision_making import DecisionConfig, decision_function, DecisionSignal, create_decision_summary
from .profitability_optimizer import ProfitabilityOptimizer, ProfitabilityConfig, optimize_alpha_factory_signal
from .features_engineering import FeatureEngineerOptimized

# Import enhanced features

from .enhancements import (
    enhanced_causal_analysis, check_lookahead_bias, optimize_memory_usage,
    enhanced_market_structure_analysis, calculate_liquidity_adjusted_return,
    create_enhanced_alpha_factory
)
HAS_ENHANCEMENTS = True

logger = logging.getLogger(__name__)


class AlphaFactory:
    """
    Alpha Factory System - Complete pipeline for market analysis and alpha generation.
    
    Pipeline:
    1. Load and validate market data
    2. Extract swing points and market structure
    3. Generate comprehensive features (220+ indicators)
    4. Perform causal analysis
    5. Make trading decisions
    6. Generate and save alpha strategies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Alpha Factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Components
        self.market_data: Optional[MarketData] = None
        self.feature_engineer = FeatureEngineerOptimized()
        self.decision_config = DecisionConfig(**self.config.get('decision', {}))
        self.profitability_optimizer = ProfitabilityOptimizer(
            ProfitabilityConfig(**self.config.get('profitability', {}))
        )
        
        # Results storage
        self.features: Optional[pd.DataFrame] = None
        self.swing_points: List = []
        self.causality_results: Dict = {}
        self.decision_signal: Optional[DecisionSignal] = None
        self.alpha_strategies: List[Dict] = []
        
        # Performance tracking
        self.processing_times = {}
        self.last_analysis_time: Optional[datetime] = None
        
        logger.info("Alpha Factory initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'market_data': {
                'swing_lookback': 5,
                'strength_threshold': 0.3,
                'enhanced_structure_analysis': False,
                'structure_lookback_window': 5
            },
            'features': {
                'batch_processing': True,
                'max_lookback': 1050,
                'memory_optimization': False,
                'target_dtype': 'float32',
                'stationarity_check': False
            },
            'causality': {
                'target_col': 'close',
                'max_lag': 5,
                'top_n_features': 10,
                'enhanced_analysis': False,
                'transfer_entropy': False,
                'lookahead_bias_check': False
            },
            'decision': {
                'min_confidence': 0.6,
                'min_causal_score': 0.3,
                'max_risk_score': 0.7,
                'trend_weight': 0.3,
                'support_resistance_weight': 0.2,
                'momentum_weight': 0.25,
                'causal_weight': 0.25,
                'liquidity_adjustment': False
            },
            'profitability': {
                'min_profit_target_pips': 15.0,  # Higher profit target
                'max_risk_per_trade': 1.5,  # Lower risk per trade
                'risk_reward_ratio': 2.5,  # Better risk:reward
                'min_volatility_threshold': 0.0008,
                'max_spread_threshold': 0.00015,
                'trend_strength_threshold': 25.0,
                'kelly_criterion': True,
                'max_position_size': 0.05,  # 5% max position
                'trailing_stop_enabled': True,
                'trailing_stop_distance': 0.0008,
                'profit_target_scaling': True,
                'trend_following_enabled': True,
                'mean_reversion_enabled': True,
                'breakout_enabled': True,
                'confidence_threshold_trend': 0.75,
                'confidence_threshold_range': 0.6,
                'adaptive_thresholds': True
            },
            'output': {
                'save_features': True,
                'save_causality': True,
                'save_decisions': True,
                'output_dir': 'alpha_output'
            }
        }
    
    def load_data(self, data: pd.DataFrame) -> 'AlphaFactory':
        """
        Load OHLCV market data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        
        try:
            self.market_data = MarketData(data)
            self.processing_times['data_loading'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Loaded {len(data)} bars of market data")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            raise
    
    def extract_market_structure(self) -> 'AlphaFactory':
        """
        Extract swing points and market structure.
        
        Returns:
            Self for method chaining
        """
        if self.market_data is None:
            raise ValueError("No market data loaded. Call load_data() first.")
        
        start_time = datetime.now()
        
        try:
            # Extract swing points
            lookback = self.config['market_data']['swing_lookback']
            strength_threshold = self.config['market_data']['strength_threshold']
            
            self.swing_points = self.market_data.extract_swings(
                lookback=lookback,
                strength_threshold=strength_threshold
            )
            
            # Get market structure analysis
            if (self.config['market_data'].get('enhanced_structure_analysis', False) and 
                HAS_ENHANCEMENTS):
                # Use enhanced analysis
                lookback_window = self.config['market_data'].get('structure_lookback_window', 5)
                market_structure = enhanced_market_structure_analysis(
                    self.swing_points, 
                    lookback_window=lookback_window
                )
            else:
                # Use standard analysis
                market_structure = self.market_data.get_market_structure()
            
            self.processing_times['structure_extraction'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Extracted {len(self.swing_points)} swing points")
            logger.info(f"Market structure: {market_structure['structure']}, Trend: {market_structure['trend']}")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to extract market structure: {e}")
            raise
    
    def generate_features(self) -> 'AlphaFactory':
        """
        Generate comprehensive features from market data.
        
        Returns:
            Self for method chaining
        """
        if self.market_data is None:
            raise ValueError("No market data loaded. Call load_data() first.")
        
        start_time = datetime.now()
        
        try:
            # Get recent data for feature generation
            recent_data = self.market_data.get_recent_data(n_bars=1000)
            
            # Generate features using the existing optimized feature engineer
            batch_processing = self.config['features']['batch_processing']
            self.features = self.feature_engineer.generate_features(
                recent_data, 
                batch_processing=batch_processing
            )
            
            # Apply memory optimization if enabled
            if (self.config['features'].get('memory_optimization', False) and 
                HAS_ENHANCEMENTS):
                target_dtype = self.config['features'].get('target_dtype', 'float32')
                self.features = optimize_memory_usage(self.features, target_dtype)
                logger.info("Applied memory optimization")
            
            # Check for look-ahead bias if enabled
            if (self.config['causality'].get('lookahead_bias_check', False) and 
                HAS_ENHANCEMENTS):
                bias_analysis = check_lookahead_bias(self.features)
                if bias_analysis['potential_lookahead_issues']:
                    logger.warning(f"Found {len(bias_analysis['potential_lookahead_issues'])} potential look-ahead bias issues")
                    for issue in bias_analysis['potential_lookahead_issues']:
                        logger.warning(f"  - {issue['feature']}: {issue['issue']}")
            
            self.processing_times['feature_generation'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generated {len(self.features.columns)} features from {len(self.features)} bars")
            
            # Save features if configured
            if self.config['output']['save_features']:
                self._save_features()
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to generate features: {e}")
            raise
    
    def analyze_causality(self) -> 'AlphaFactory':
        """
        Perform causal analysis on features.
        
        Returns:
            Self for method chaining
        """
        if self.features is None:
            raise ValueError("No features generated. Call generate_features() first.")
        
        start_time = datetime.now()
        
        try:
            # Use enhanced causal analysis if enabled
            if (self.config['causality'].get('enhanced_analysis', False) and 
                HAS_ENHANCEMENTS):
                target_col = self.config['causality']['target_col']
                self.causality_results = enhanced_causal_analysis(self.features, target_col)
                
                # Log enhanced analysis results
                if 'stationarity_analysis' in self.causality_results:
                    stationarity = self.causality_results['stationarity_analysis']
                    logger.info(f"Stationarity: {stationarity['stationary_features']}/{stationarity['features_checked']} features stationary")
                
                if 'transfer_entropy' in self.causality_results:
                    te_features = len(self.causality_results['transfer_entropy'])
                    logger.info(f"Transfer Entropy analysis completed for {te_features} features")
                
            else:
                # Use standard causal analysis
                target_col = self.config['causality']['target_col']
                max_lag = self.config['causality']['max_lag']
                
                self.causality_results = compute_causality(
                    self.features, 
                    target_col=target_col
                )
            
            # Get top causal features
            top_n = self.config['causality']['top_n_features']
            top_features = get_top_causal_features(self.causality_results, top_n)
            
            # Create causal network
            causal_network = create_causal_network(self.causality_results)
            
            self.processing_times['causality_analysis'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Analyzed causality for {len(self.features.columns)} features")
            logger.info(f"Top {len(top_features)} causal features identified")
            
            # Save causality results if configured
            if self.config['output']['save_causality']:
                self._save_causality_results()
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to analyze causality: {e}")
            raise
    
    def make_decision(self) -> 'AlphaFactory':
        """
        Make optimized trading decision based on all analysis.
        
        Returns:
            Self for method chaining
        """
        if self.features is None or self.causality_results is None:
            raise ValueError("Features and causality analysis required. Run generate_features() and analyze_causality() first.")
        
        start_time = datetime.now()
        
        try:
            # Create decision config
            decision_config = DecisionConfig(**self.config.get('decision', {}))
            
            # Get market structure for decision making
            market_structure = None
            if hasattr(self.market_data, 'get_market_structure'):
                market_structure = self.market_data.get_market_structure()
            
            # Make initial decision
            initial_decision = decision_function(
                swing_points=self.swing_points,
                features=self.features,
                causality_results=self.causality_results,
                config=decision_config,
                market_structure=market_structure
            )
            
            # Convert to dictionary for optimization
            decision_dict = {
                'decision': initial_decision.decision.value,
                'confidence': initial_decision.confidence,
                'regime': initial_decision.regime.value,
                'risk_score': initial_decision.risk_score,
                'expected_return': initial_decision.expected_return,
                'stop_loss': initial_decision.stop_loss,
                'take_profit': initial_decision.take_profit,
                'reasoning': initial_decision.reasoning,
                'key_features': initial_decision.key_features
            }
            
            # Get recent market data for optimization
            recent_data = self.market_data.get_recent_data(n_bars=100)
            
            # Apply profitability optimization
            optimized_decision = self.profitability_optimizer.optimize_decision_pipeline(
                market_data=recent_data,
                original_signal=decision_dict,
                market_structure=market_structure,
                recent_trades=[]  # Would be populated from trade history
            )
            
            # Convert back to DecisionSignal
            self.decision_signal = DecisionSignal(
                decision=initial_decision.decision.__class__(optimized_decision['decision']),
                confidence=optimized_decision['confidence'],
                regime=initial_decision.regime.__class__(optimized_decision['regime']),
                reasoning=optimized_decision.get('reasoning', []),
                key_features=optimized_decision.get('key_features', []),
                risk_score=optimized_decision.get('risk_score', 0.3),
                expected_return=optimized_decision.get('expected_return', 0.0),
                stop_loss=optimized_decision.get('optimized_stop_loss', optimized_decision.get('stop_loss')),
                take_profit=optimized_decision.get('optimized_take_profit', optimized_decision.get('take_profit'))
            )
            
            self.processing_times['decision_making'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Optimized Decision: {self.decision_signal.decision.value}")
            logger.info(f"Confidence: {self.decision_signal.confidence:.3f}")
            logger.info(f"Regime: {self.decision_signal.regime.value}")
            
            # Log optimization details
            if optimized_decision.get('optimization_applied'):
                logger.info("Profitability optimization applied")
                logger.info(f"Risk:Reward Ratio: {optimized_decision.get('risk_reward_ratio', 0):.2f}")
                logger.info(f"Position Size: {optimized_decision.get('position_size', 0):.2%}")
            
            # Save decision if configured
            if self.config['output']['save_decisions']:
                self._save_decision()
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to make decision: {e}")
            raise
    
    def generate_alpha(self) -> Dict:
        """
        Generate alpha strategy based on analysis.
        
        Returns:
            Dictionary containing alpha strategy
        """
        if self.decision_signal is None:
            raise ValueError("No decision made. Call make_decision() first.")
        
        # Create alpha strategy
        alpha_strategy = {
            'timestamp': datetime.now().isoformat(),
            'decision': create_decision_summary(self.decision_signal),
            'market_structure': {
                'swing_points_count': len(self.swing_points),
                'structure_analysis': self.market_data.get_market_structure() if self.market_data else {}
            },
            'top_features': get_top_causal_features(
                self.causality_results, 
                self.config['causality']['top_n_features']
            ) if self.causality_results else [],
            'causal_network': create_causal_network(self.causality_results) if self.causality_results else {},
            'performance_metrics': {
                'processing_times': self.processing_times,
                'total_processing_time': sum(self.processing_times.values())
            },
            'configuration': self.config
        }
        
        self.alpha_strategies.append(alpha_strategy)
        
        logger.info("Alpha strategy generated successfully")
        return alpha_strategy
    
    def execute_trade(self) -> Optional[Dict]:
        """
        Execute trade based on decision (placeholder for actual execution).
        
        Returns:
            Trade execution details or None if no trade
        """
        if self.decision_signal is None:
            return None
        
        if self.decision_signal.decision.value == 'HOLD':
            logger.info("No trade to execute (HOLD decision)")
            return None
        
        # Create trade execution details
        trade_details = {
            'action': self.decision_signal.decision.value,
            'confidence': self.decision_signal.confidence,
            'entry_price': self.market_data.data['close'].iloc[-1] if self.market_data else 0,
            'stop_loss': self.decision_signal.stop_loss,
            'take_profit': self.decision_signal.take_profit,
            'risk_score': self.decision_signal.risk_score,
            'expected_return': self.decision_signal.expected_return,
            'reasoning': self.decision_signal.reasoning,
            'key_features': self.decision_signal.key_features,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Trade execution prepared: {trade_details['action']}")
        return trade_details
    
    def save_alpha(self, filename: Optional[str] = None) -> str:
        """
        Save generated alpha strategy to file.
        
        Args:
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            Path to saved file
        """
        if not self.alpha_strategies:
            raise ValueError("No alpha strategies to save")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alpha_strategy_{timestamp}.json"
        
        # Create output directory
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save strategy
        filepath = output_dir / filename
        latest_strategy = self.alpha_strategies[-1]
        
        with open(filepath, 'w') as f:
            json.dump(latest_strategy, f, indent=2, default=str)
        
        logger.info(f"Alpha strategy saved to {filepath}")
        return str(filepath)
    
    def process_data(self, data: pd.DataFrame) -> Dict:
        """
        Complete pipeline: process data and generate alpha.
        
        Args:
            data: OHLCV market data
            
        Returns:
            Complete alpha strategy
        """
        logger.info("Starting Alpha Factory pipeline...")
        
        try:
            # Run complete pipeline
            self.load_data(data)
            self.extract_market_structure()
            self.generate_features()
            self.analyze_causality()
            self.make_decision()
            
            # Generate and return alpha
            alpha = self.generate_alpha()
            
            logger.info("Alpha Factory pipeline completed successfully")
            return alpha
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_summary(self) -> Dict:
        """
        Get summary of current analysis.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'status': 'completed' if self.decision_signal else 'incomplete',
            'data_loaded': self.market_data is not None,
            'swing_points': len(self.swing_points) if self.swing_points else 0,
            'features': len(self.features.columns) if self.features is not None else 0,
            'causal_analysis': bool(self.causality_results),
            'decision': self.decision_signal.decision.value if self.decision_signal else None,
            'confidence': self.decision_signal.confidence if self.decision_signal else 0,
            'regime': self.decision_signal.regime.value if self.decision_signal else None,
            'processing_times': self.processing_times,
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
        
        return summary
    
    def _save_features(self):
        """Save features to file."""
        if self.features is None:
            return
        
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"features_{timestamp}.parquet"
        
        self.features.to_parquet(filepath)
        logger.info(f"Features saved to {filepath}")
    
    def _save_causality_results(self):
        """Save causality results to file."""
        if not self.causality_results:
            return
        
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"causality_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.causality_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {k2: v2 if not isinstance(v2, np.ndarray) else v2.tolist() 
                                                     for k2, v2 in v.items()}
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Causality results saved to {filepath}")
    
    def _save_decision(self):
        """Save decision to file."""
        if self.decision_signal is None:
            return
        
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"decision_{timestamp}.json"
        
        decision_summary = create_decision_summary(self.decision_signal)
        
        with open(filepath, 'w') as f:
            json.dump(decision_summary, f, indent=2, default=str)
        
        logger.info(f"Decision saved to {filepath}")


# Convenience function for quick usage
def create_alpha_factory(config: Optional[Dict] = None) -> AlphaFactory:
    """
    Create Alpha Factory instance with optional configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AlphaFactory instance
    """
    return AlphaFactory(config)
