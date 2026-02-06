"""
Feature Pruning for Alpha Factory

This module implements professional feature pruning techniques:
1. Marginal contribution analysis for each feature
2. Feature decay rate measurement
3. Statistical significance testing
4. Correlation-based redundancy removal
5. Performance impact assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeaturePruningConfig:
    """Configuration for feature pruning."""
    enabled: bool = False
    # Pruning thresholds
    min_contribution_threshold: float = 0.05  # 5% minimum contribution
    max_correlation_threshold: float = 0.8   # 80% max correlation
    min_significance_threshold: float = 0.05 # 5% significance level
    
    # Performance metrics
    performance_window: int = 1000  # Trades for performance analysis
    decay_window: int = 500        # Trades for decay analysis
    
    # Analysis methods
    use_mutual_information: bool = True
    use_random_forest_importance: bool = True
    use_correlation_analysis: bool = True
    use_performance_impact: bool = True
    
    # Pruning strategy
    aggressive_pruning: bool = False  # Remove more features if True
    keep_core_features: List[str] = None  # Features to never remove

class FeaturePruner:
    """Professional feature pruning for Alpha Factory."""
    
    def __init__(self, config: FeaturePruningConfig = None):
        self.config = config or FeaturePruningConfig()
        
        # Initialize core features to keep
        if self.config.keep_core_features is None:
            self.config.keep_core_features = [
                'close', 'high', 'low', 'volume',  # Core price data
                'atr', 'rsi', 'macd', 'adx'        # Core indicators
            ]
        
        # Feature analysis history
        self.feature_importance_history = {}
        self.feature_performance_history = {}
        self.pruning_history = []
        
    def calculate_marginal_contribution(self, features: pd.DataFrame, targets: pd.Series,
                                      feature_name: str) -> float:
        """
        Calculate marginal contribution of a specific feature.
        
        Args:
            features: DataFrame with all features
            targets: Series with target values
            feature_name: Name of feature to analyze
            
        Returns:
            Marginal contribution score (0-1)
        """
        if feature_name not in features.columns:
            return 0.0
        
        try:
            # Method 1: Mutual Information
            if self.config.use_mutual_information:
                mi_score = mutual_info_regression(
                    features[[feature_name]], targets, random_state=42
                )[0]
                mi_contribution = mi_score / (mi_score + 1e-6)  # Normalize
            else:
                mi_contribution = 0.0
            
            # Method 2: Random Forest Importance
            if self.config.use_random_forest_importance:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(features, targets)
                importances = rf.feature_importances_
                feature_idx = list(features.columns).index(feature_name)
                rf_contribution = importances[feature_idx]
            else:
                rf_contribution = 0.0
            
            # Method 3: Correlation with target
            if self.config.use_correlation_analysis:
                correlation, p_value = pearsonr(features[feature_name], targets)
                if p_value < self.config.min_significance_threshold:
                    corr_contribution = abs(correlation)
                else:
                    corr_contribution = 0.0
            else:
                corr_contribution = 0.0
            
            # Combine contributions (weighted average)
            total_contribution = (
                0.4 * mi_contribution +
                0.4 * rf_contribution +
                0.2 * corr_contribution
            )
            
            return min(1.0, total_contribution)
            
        except Exception as e:
            logger.error(f"Error calculating marginal contribution for {feature_name}: {e}")
            return 0.0
    
    def calculate_feature_decay_rate(self, feature_values: pd.Series, 
                                   performance_window: int = None) -> float:
        """
        Calculate decay rate of feature effectiveness over time.
        
        Args:
            feature_values: Series with feature values over time
            performance_window: Window for decay analysis
            
        Returns:
            Decay rate (0-1, higher = faster decay)
        """
        if performance_window is None:
            performance_window = self.config.decay_window
        
        if len(feature_values) < performance_window * 2:
            return 0.0  # Insufficient data
        
        try:
            # Split into two halves
            mid_point = len(feature_values) // 2
            first_half = feature_values.iloc[:mid_point]
            second_half = feature_values.iloc[mid_point:mid_point + performance_window]
            
            # Calculate variance in each half
            first_var = first_half.var()
            second_var = second_half.var()
            
            # Calculate decay rate based on variance reduction
            if first_var > 0:
                decay_rate = max(0, (first_var - second_var) / first_var)
            else:
                decay_rate = 0.0
            
            return min(1.0, decay_rate)
            
        except Exception as e:
            logger.error(f"Error calculating feature decay rate: {e}")
            return 0.0
    
    def identify_redundant_features(self, features: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated features that might be redundant.
        
        Args:
            features: DataFrame with features
            
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        redundant_pairs = []
        
        if not self.config.use_correlation_analysis:
            return redundant_pairs
        
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find highly correlated pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if correlation > self.config.max_correlation_threshold:
                        redundant_pairs.append((feature1, feature2, correlation))
            
        except Exception as e:
            logger.error(f"Error identifying redundant features: {e}")
        
        return redundant_pairs
    
    def analyze_feature_performance_impact(self, features: pd.DataFrame, 
                                          feature_name: str, targets: pd.Series) -> Dict[str, float]:
        """
        Analyze the performance impact of including vs excluding a feature.
        
        Args:
            features: DataFrame with all features
            feature_name: Feature to analyze
            targets: Target values
            
        Returns:
            Dictionary with performance metrics
        """
        if feature_name not in features.columns:
            return {'with_feature': 0, 'without_feature': 0, 'impact': 0}
        
        try:
            # Performance with feature
            features_with = features.copy()
            rf_with = RandomForestRegressor(n_estimators=50, random_state=42)
            scores_with = cross_val_score(rf_with, features_with, targets, cv=3, scoring='neg_mean_squared_error')
            perf_with = -scores_with.mean()
            
            # Performance without feature
            features_without = features.drop(columns=[feature_name])
            if len(features_without.columns) == 0:
                perf_without = perf_with  # No features left
            else:
                rf_without = RandomForestRegressor(n_estimators=50, random_state=42)
                scores_without = cross_val_score(rf_without, features_without, targets, cv=3, scoring='neg_mean_squared_error')
                perf_without = -scores_without.mean()
            
            # Calculate impact
            impact = (perf_without - perf_with) / perf_with if perf_with > 0 else 0
            
            return {
                'with_feature': perf_with,
                'without_feature': perf_without,
                'impact': impact
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance impact for {feature_name}: {e}")
            return {'with_feature': 0, 'without_feature': 0, 'impact': 0}
    
    def prune_features(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive feature pruning.
        
        Args:
            features: DataFrame with all features
            targets: Target values
            
        Returns:
            Tuple of pruned features DataFrame and pruning report
        """
        if not bool(getattr(self.config, 'enabled', False)):
            report = {
                'original_features': len(features.columns),
                'pruned_features': [],
                'kept_features': list(features.columns),
                'feature_analysis': {},
                'redundant_pairs': [],
                'pruning_reasons': {},
                'final_features': len(features.columns)
            }
            logger.info("Feature pruning disabled; returning full feature set")
            return features.copy(), report

        logger.info("Starting feature pruning analysis")
        
        # Initialize pruning report
        report = {
            'original_features': len(features.columns),
            'pruned_features': [],
            'kept_features': [],
            'feature_analysis': {},
            'redundant_pairs': [],
            'pruning_reasons': {}
        }
        
        # Analyze each feature
        feature_analysis = {}
        
        for feature_name in features.columns:
            # Skip core features
            if feature_name in self.config.keep_core_features:
                feature_analysis[feature_name] = {
                    'contribution': 1.0,  # Keep core features
                    'decay_rate': 0.0,
                    'performance_impact': 0.0,
                    'action': 'kept_core'
                }
                continue
            
            # Calculate marginal contribution
            contribution = self.calculate_marginal_contribution(features, targets, feature_name)
            
            # Calculate decay rate
            decay_rate = self.calculate_feature_decay_rate(features[feature_name])
            
            # Calculate performance impact
            perf_impact = self.analyze_feature_performance_impact(features, feature_name, targets)
            
            feature_analysis[feature_name] = {
                'contribution': contribution,
                'decay_rate': decay_rate,
                'performance_impact': perf_impact['impact'],
                'action': 'pending'
            }
        
        # Identify redundant features
        redundant_pairs = self.identify_redundant_features(features)
        report['redundant_pairs'] = redundant_pairs
        
        # Make pruning decisions
        features_to_keep = list(self.config.keep_core_features)
        features_to_prune = []
        
        for feature_name, analysis in feature_analysis.items():
            if feature_name in self.config.keep_core_features:
                features_to_keep.append(feature_name)
                analysis['action'] = 'kept_core'
                continue
            
            # Pruning criteria
            should_prune = False
            reason = ""
            
            # Criterion 1: Low contribution
            if analysis['contribution'] < self.config.min_contribution_threshold:
                should_prune = True
                reason = f"Low contribution: {analysis['contribution']:.3f} < {self.config.min_contribution_threshold}"
            
            # Criterion 2: High decay rate
            elif analysis['decay_rate'] > 0.5:  # More than 50% decay
                should_prune = True
                reason = f"High decay rate: {analysis['decay_rate']:.3f}"
            
            # Criterion 3: Negative performance impact
            elif analysis['performance_impact'] < -0.1:  # More than 10% negative impact
                should_prune = True
                reason = f"Negative performance impact: {analysis['performance_impact']:.3f}"
            
            # Criterion 4: Redundancy (keep the one with higher contribution)
            elif self.config.aggressive_pruning:
                for feature1, feature2, correlation in redundant_pairs:
                    if feature_name == feature2:
                        other_analysis = feature_analysis.get(feature1, {})
                        if (other_analysis.get('contribution', 0) > analysis['contribution']):
                            should_prune = True
                            reason = f"Redundant with {feature1} (corr: {correlation:.3f})"
                        break
            
            if should_prune:
                features_to_prune.append(feature_name)
                analysis['action'] = 'pruned'
                report['pruning_reasons'][feature_name] = reason
            else:
                features_to_keep.append(feature_name)
                analysis['action'] = 'kept'
        
        # Create pruned features DataFrame
        pruned_features = features[features_to_keep].copy()
        
        # Update report
        report['feature_analysis'] = feature_analysis
        report['pruned_features'] = features_to_prune
        report['kept_features'] = features_to_keep
        report['final_features'] = len(pruned_features.columns)
        
        # Store pruning history
        self.pruning_history.append({
            'timestamp': datetime.now(),
            'original_count': len(features.columns),
            'final_count': len(pruned_features.columns),
            'pruned_count': len(features_to_prune),
            'features_pruned': features_to_prune
        })
        
        logger.info(f"Feature pruning completed: {len(features.columns)} → {len(pruned_features.columns)} features")
        
        return pruned_features, report
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning history and recommendations."""
        if not self.pruning_history:
            return {'status': 'no_pruning_history'}
        
        latest_pruning = self.pruning_history[-1]
        
        summary = {
            'total_pruning_operations': len(self.pruning_history),
            'latest_pruning': latest_pruning,
            'pruning_rate': 1 - (latest_pruning['final_count'] / latest_pruning['original_count']),
            'aggressive_mode': self.config.aggressive_pruning,
            'core_features_protected': len(self.config.keep_core_features)
        }
        
        # Calculate average pruning rate
        if len(self.pruning_history) > 1:
            pruning_rates = [1 - (p['final_count'] / p['original_count']) for p in self.pruning_history]
            summary['average_pruning_rate'] = np.mean(pruning_rates)
            summary['pruning_rate_std'] = np.std(pruning_rates)
        
        return summary
    
    def recommend_pruning_strategy(self, feature_analysis: Dict[str, Dict]) -> Dict[str, str]:
        """
        Recommend pruning strategy based on feature analysis.
        
        Args:
            feature_analysis: Analysis results for each feature
            
        Returns:
            Dictionary of feature recommendations
        """
        recommendations = {}
        
        for feature_name, analysis in feature_analysis.items():
            if feature_name in self.config.keep_core_features:
                recommendations[feature_name] = "KEEP (core feature)"
                continue
            
            contribution = analysis.get('contribution', 0)
            decay_rate = analysis.get('decay_rate', 0)
            perf_impact = analysis.get('performance_impact', 0)
            
            # Generate recommendation
            if contribution < 0.02:
                recommendations[feature_name] = "REMOVE (very low contribution)"
            elif contribution < 0.05:
                recommendations[feature_name] = "CONSIDER REMOVE (low contribution)"
            elif decay_rate > 0.7:
                recommendations[feature_name] = "REMOVE (high decay)"
            elif perf_impact < -0.2:
                recommendations[feature_name] = "REMOVE (negative impact)"
            elif contribution > 0.15:
                recommendations[feature_name] = "KEEP (high contribution)"
            else:
                recommendations[feature_name] = "MONITOR (moderate contribution)"
        
        return recommendations
