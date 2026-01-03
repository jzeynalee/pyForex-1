# alpha_factory/causal_analysis.py
"""
Causal Analysis module for Alpha Factory system.
Performs causality analysis between features using various statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
# Granger causality implementation (fallback if statsmodels not available)
def _granger_causality_fallback(data, max_lag=5):
    """Simple Granger causality implementation using linear regression."""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        feature, target = data.columns
        results = {}
        
        for lag in range(1, max_lag + 1):
            # Prepare data
            X = []
            y = []
            
            for i in range(lag, len(data)):
                # Lagged values of both variables
                lagged_values = []
                for l in range(1, lag + 1):
                    lagged_values.append(data[feature].iloc[i - l])
                    lagged_values.append(data[target].iloc[i - l])
                X.append(lagged_values)
                y.append(data[target].iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Full model (with feature lags)
            model_full = LinearRegression().fit(X, y)
            y_pred_full = model_full.predict(X)
            mse_full = mean_squared_error(y, y_pred_full)
            
            # Reduced model (only target lags)
            X_reduced = X[:, 1::2]  # Only target lags
            model_reduced = LinearRegression().fit(X_reduced, y)
            y_pred_reduced = model_reduced.predict(X_reduced)
            mse_reduced = mean_squared_error(y, y_pred_reduced)
            
            # F-statistic
            if mse_reduced > 0:
                f_stat = ((mse_reduced - mse_full) / lag) / (mse_full / (len(y) - 2 * lag))
                p_value = 1 - stats.f.cdf(f_stat, lag, len(y) - 2 * lag)
            else:
                f_stat = 0
                p_value = 1
            
            results[lag] = {
                'ssr_ftest': (f_stat, p_value, None, None),
                'lrtest': (f_stat, p_value, None, None),
                'params_ftest': (f_stat, p_value, None, None)
            }
        
        return results
    except ImportError:
        return None
from sklearn.feature_selection import mutual_info_regression
import warnings
import logging

logger = logging.getLogger(__name__)

# Global configuration
max_lag = 5


def _benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Benjamini-Hochberg procedure to control false discovery rate.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired false discovery rate (default 0.05)
        
    Returns:
        List of corrected p-values
    """
    if not p_values:
        return []
    
    # Convert to numpy array
    p_array = np.array(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    # Calculate BH critical values
    m = len(p_values)
    bh_critical = np.arange(1, m + 1) * alpha / m
    
    # Find the largest p-value that is less than its BH critical value
    significant_indices = np.where(sorted_p <= bh_critical)[0]
    
    if len(significant_indices) > 0:
        max_idx = significant_indices[-1]
        threshold = sorted_p[max_idx]
    else:
        threshold = 0
    
    # Apply correction (simple approach: use BH threshold)
    corrected_p = np.minimum(p_array / alpha * threshold, 1.0)
    
    return corrected_p.tolist()


def compute_causality(features: pd.DataFrame, target_col: str, max_lag: int = 5) -> Dict:
    """
    Compute causality between features and target variable.
    
    Args:
        features: DataFrame with features
        target_col: Target column name
        
    Returns:
        Dictionary with causality results
    """
    # Filter out non-trading features and metadata
    filtered_features = _filter_trading_features(features, target_col)
    
    logger.info(f"Analyzing causality for {len(filtered_features.columns)} features against {target_col}")
    
    # Get feature columns (excluding target)
    feature_cols = [col for col in filtered_features.columns if col != target_col]
    
    # Initialize results
    granger_results = {}
    correlation_results = {}
    mutual_info_results = {}
    lead_lag_results = {}
    
    # Granger causality analysis
    logger.info("Computing Granger causality...")
    all_p_values = []  # Collect all p-values for Benjamini-Hochberg correction
    
    for feature in feature_cols:
        try:
            # Prepare data for Granger test
            data = filtered_features[[feature, target_col]].dropna()
            
            if len(data) < max_lag * 2:
                continue
            
            # Try statsmodels first, then fallback
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                test_result = grangercausalitytests(data, max_lag, verbose=False)
            except ImportError:
                # Use fallback implementation
                test_result = _granger_causality_fallback(data, max_lag)
                if test_result is None:
                    continue
            
            # Extract F-statistic p-values for each lag
            p_values = []
            f_stats = []
            
            for lag in range(1, max_lag + 1):
                f_stat = test_result[lag][0]['ssr_ftest'][0]
                p_value = test_result[lag][0]['ssr_ftest'][1]
                f_stats.append(f_stat)
                p_values.append(p_value)
            
            # Use the best lag (lowest p-value)
            best_lag = np.argmin(p_values) + 1
            best_f_stat = f_stats[best_lag - 1]
            best_p_value = p_values[best_lag - 1]
            
            granger_results[feature] = {
                'best_lag': best_lag,
                'best_f_stat': best_f_stat,
                'best_p_value': best_p_value,
                'f_stats': f_stats,
                'p_values': p_values
            }
            
            # Collect p-value for correction
            all_p_values.append(best_p_value)
            
        except Exception as e:
            logger.debug(f"Granger causality failed for {feature}: {e}")
            continue
    
    # Apply Benjamini-Hochberg correction to control false discovery rate
    if all_p_values:
        corrected_p_values = _benjamini_hochberg_correction(all_p_values)
        
        # Update granger results with corrected p-values
        p_value_idx = 0
        for feature in feature_cols:
            if feature in granger_results:
                granger_results[feature]['best_p_value_corrected'] = corrected_p_values[p_value_idx]
                granger_results[feature]['significant_corrected'] = corrected_p_values[p_value_idx] < 0.05
                p_value_idx += 1
    
    # Correlation analysis
    logger.info("Computing correlations...")
    for feature in feature_cols:
        try:
            # Prepare data
            df = filtered_features[[feature, target_col]].dropna()
            
            if len(df) < 10:
                continue
            
            clean_feature = df[feature].values
            clean_target = df[target_col].values
            
            # Remove constant series
            if np.std(clean_feature) == 0 or np.std(clean_target) == 0:
                continue
            
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(clean_feature, clean_target)
            
            # Spearman correlation
            spearman_corr, spearman_p = stats.spearmanr(clean_feature, clean_target)
            
            # Kendall correlation
            kendall_corr, kendall_p = stats.kendalltau(clean_feature, clean_target)
            
            correlation_results[feature] = {
                'pearson_corr': str(pearson_corr),
                'pearson_p_value': pearson_p,
                'pearson_abs': str(abs(pearson_corr)),
                'spearman_corr': spearman_corr,
                'spearman_p_value': spearman_p,
                'spearman_abs': abs(spearman_corr),
                'kendall_corr': kendall_corr,
                'kendall_p_value': kendall_p,
                'kendall_abs': abs(kendall_corr),
                'significant_pearson': str(pearson_p < 0.05),
                'significant_spearman': str(spearman_p < 0.05),
                'significant_kendall': str(kendall_p < 0.05)
            }
            
        except Exception as e:
            logger.debug(f"Correlation analysis failed for {feature}: {e}")
            continue
    
    # Mutual information
    logger.info("Computing mutual information...")
    for feature in feature_cols:
        try:
            # Prepare data
            df = filtered_features[[feature, target_col]].dropna()
            
            if len(df) < 10:
                continue
            
            clean_feature = df[feature].values
            clean_target = df[target_col].values
            
            # Remove constant series
            if np.std(clean_feature) == 0 or np.std(clean_target) == 0:
                continue
            
            # Compute mutual information
            mi_score = mutual_info_regression(clean_feature.reshape(-1, 1), clean_target, random_state=42)
            
            # Normalize MI score
            max_possible_mi = min(np.log(len(df)), np.log(len(set(clean_feature))) + np.log(len(set(clean_target))))
            normalized_mi = mi_score / max_possible_mi if max_possible_mi > 0 else 0
            
            mutual_info_results[feature] = {
                'mi_score': mi_score,
                'normalized_mi': normalized_mi,
                'rank': None
            }
            
        except Exception as e:
            logger.debug(f"Mutual information failed for {feature}: {e}")
            continue
    
    # Lead-lag analysis
    logger.info("Computing lead-lag relationships...")
    for feature in feature_cols:
        try:
            # Prepare data
            df = filtered_features[[feature, target_col]].dropna()
            
            if len(df) < 20:
                continue
            
            clean_feature = df[feature].values
            clean_target = df[target_col].values
            
            # Remove constant series
            if np.std(clean_feature) == 0 or np.std(clean_target) == 0:
                continue
            
            # Compute correlations for different lags
            correlations = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(clean_feature, clean_target)[0, 1]
                elif lag > 0:
                    # Feature leads target
                    if len(clean_feature) > lag:
                        corr = np.corrcoef(clean_feature[:-lag], clean_target[lag:])[0, 1]
                    else:
                        corr = 0
                else:
                    # Feature lags target
                    if len(clean_target) > abs(lag):
                        corr = np.corrcoef(clean_feature[abs(lag):], clean_target[:-abs(lag)])[0, 1]
                    else:
                        corr = 0
                
                correlations.append(corr)
            
            # Find best correlation
            best_lag = np.argmax(np.abs(correlations)) - max_lag
            best_correlation = correlations[best_lag + max_lag]
            
            # Determine if feature leads or lags target
            leads_target = best_lag < 0
            lags_target = best_lag > 0
            
            lead_lag_results[feature] = {
                'best_lag': str(best_lag),
                'best_correlation': best_correlation,
                'best_abs_correlation': abs(best_correlation),
                'all_correlations': correlations,
                'leads_target': str(leads_target),
                'lags_target': str(lags_target),
                'lead_strength': abs(best_lag) if leads_target else 0,
                'lag_strength': best_lag if lags_target else 0
            }
            
        except Exception as e:
            logger.debug(f"Lead-lag analysis failed for {feature}: {e}")
            continue
    
    # Combine results into ranking
    causal_ranking = {}
    
    for feature in feature_cols:
        if feature in correlation_results or feature in mutual_info_results:
            score = 0
            methods = 0
            
            # Add Granger causality score
            if feature in granger_results:
                # Convert F-statistic to score (higher is better)
                f_stat = granger_results[feature]['best_f_stat']
                p_value = granger_results[feature]['best_p_value']
                if p_value < 0.05:
                    score += np.log1p(f_stat) * (1 - p_value)
                    methods += 1
            
            # Add correlation score
            if feature in correlation_results:
                # Use absolute Pearson correlation
                pearson_abs = float(correlation_results[feature]['pearson_abs'])
                score += pearson_abs * 3  # Weight correlation more heavily
                methods += 1
            
            # Add mutual information score
            if feature in mutual_info_results:
                mi_score = mutual_info_results[feature]['normalized_mi']
                score += mi_score * 2
                methods += 1
            
            # Add lead-lag score
            if feature in lead_lag_results:
                best_abs_corr = lead_lag_results[feature]['best_abs_correlation']
                score += best_abs_corr
                methods += 1
            
            if methods > 0:
                causal_ranking[feature] = {
                    'combined_score': score,
                    'num_methods': methods,
                    'granger_causality': granger_results.get(feature, {}),
                    'correlation_analysis': correlation_results.get(feature, {}),
                    'mutual_information': mutual_info_results.get(feature, {}),
                    'lead_lag_analysis': lead_lag_results.get(feature, {})
                }
    
    # Sort by combined score
    sorted_features = sorted(causal_ranking.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    
    # Assign ranks
    for rank, (feature, data) in enumerate(sorted_features, 1):
        data['rank'] = rank
    
    return {
        'causal_ranking': dict(sorted_features),
        'granger_causality': granger_results,
        'correlation_analysis': correlation_results,
        'mutual_information': mutual_info_results,
        'lead_lag_analysis': lead_lag_results,
        'total_features_analyzed': len(feature_cols),
        'target_column': target_col
    }


def _filter_trading_features(features: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Advanced filter for non-trading metadata and spurious features.
    """
    # 1. Standard metadata pattern exclusion
    exclude_patterns = ['timestamp', 'time', 'date', 'year', 'month', 'id', 'index']
    
    # 2. Strict Variance Thresholding: Remove features that don't move enough to provide alpha
    # Removes constants or near-constants that can cause math errors in Granger/MI
    numeric_df = features.select_dtypes(include=[np.number])
    
    # Apply variance filter - features with near-zero variance are useless
    variance_filter = numeric_df.std() > 1e-6
    filtered_cols = numeric_df.columns[variance_filter].tolist()
    
    logger.info(f"Variance filtering: {len(numeric_df.columns)} -> {len(filtered_cols)} features")
    
    # 3. Autocorrelation Filter: Remove 'Straight Line' features (like incremental indices)
    # Features with Lag-1 autocorrelation > 0.99 are usually metadata masquerading as data
    final_cols = []
    for col in filtered_cols:
        if col == target_col:
            final_cols.append(col)
            continue
        
        # Check autocorrelation
        try:
            series = features[col].dropna()
            if len(series) > 50:
                acf = series.autocorr(lag=1)
                if abs(acf) >= 0.99:
                    logger.debug(f"Filtering {col} - autocorrelation too high: {acf:.3f}")
                    continue
        except:
            # If we can't calculate autocorrelation, keep the feature
            pass
        
        # Filter metadata patterns
        if not any(p in col.lower() for p in exclude_patterns):
            final_cols.append(col)
    
    logger.info(f"Autocorrelation filtering: {len(filtered_cols)} -> {len(final_cols)} features")
    
    # 4. Apply advanced statistical filters
    if len(final_cols) > 0:
        valid_features = _apply_statistical_filters(features[final_cols], target_col)
        logger.info(f"Statistical filtering: {len(final_cols)} -> {len(valid_features)} features")
        return features[valid_features]
    else:
        logger.warning("No features passed initial filtering")
        return features[[target_col]]


def _apply_statistical_filters(features: pd.DataFrame, target_col: str) -> List[str]:
    """
    Apply statistical filters to remove noise and non-predictive features.
    
    Args:
        features: DataFrame with numeric features
        target_col: Target column name
        
    Returns:
        List of feature names that pass statistical filters
    """
    valid_features = []
    
    for col in features.columns:
        if col == target_col:
            valid_features.append(col)
            continue
        
        series = features[col].dropna()
        
        if len(series) < 50:  # Insufficient data
            continue
        
        # 1. Autocorrelation filter - remove highly autocorrelated features (>0.99)
        try:
            autocorr = series.autocorr(lag=1)
            if abs(autocorr) > 0.99:
                logger.debug(f"Filtering {col} - autocorrelation too high: {autocorr:.3f}")
                continue
        except:
            continue
        
        # 2. Variance filter - remove near-constant features
        if series.var() < 1e-10:  # Near-zero variance
            logger.debug(f"Filtering {col} - variance too low: {series.var():.2e}")
            continue
        
        # 3. Stationarity filter - prefer stationary features for causal analysis
        try:
            # Simple stationarity check using rolling mean/std
            rolling_mean = series.rolling(50).mean()
            rolling_std = series.rolling(50).std()
            
            # Check if mean and std are relatively stable
            mean_stability = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else 0
            std_stability = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 0
            
            # If feature is highly non-stationary, apply differencing
            if mean_stability > 0.1 or std_stability > 0.2:
                # Check if differencing makes it stationary
                diff_series = series.diff().dropna()
                if len(diff_series) > 50:
                    diff_autocorr = diff_series.autocorr(lag=1)
                    if abs(diff_autocorr) < 0.99 and diff_series.var() > 1e-10:
                        # Keep the feature but mark it as differenced
                        logger.debug(f"Keeping {col} - non-stationary but differenciable")
                        valid_features.append(col)
                        continue
                else:
                    logger.debug(f"Filtering {col} - non-stationary and not differenciable")
                    continue
        except:
            # If stationarity check fails, keep the feature but log warning
            logger.debug(f"Stationarity check failed for {col}, keeping feature")
        
        # 4. Correlation with target - ensure meaningful relationship
        if target_col in features.columns:
            try:
                target_series = features[target_col].dropna()
                aligned_data = pd.concat([series, target_series], axis=1).dropna()
                
                if len(aligned_data) > 50:
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    
                    # Filter out features with very low correlation to target
                    if abs(correlation) < 0.01:
                        logger.debug(f"Filtering {col} - low correlation with target: {correlation:.3f}")
                        continue
            except:
                pass
        
        # Feature passed all filters
        valid_features.append(col)
    
    logger.info(f"Statistical filtering: {len(features.columns)} -> {len(valid_features)} features")
    return valid_features


def get_top_causal_features(causality_results: Dict, top_n: int = 10) -> List[Dict]:
    """
    Get top N causal features based on combined ranking.
    
    Args:
        causality_results: Results from compute_causality
        top_n: Number of top features to return
        
    Returns:
        List of top features with detailed information
    """
    if 'causal_ranking' not in causality_results:
        return []
    
    top_features = []
    ranking = causality_results['causal_ranking']
    
    for feature, data in list(ranking.items())[:top_n]:
        feature_info = {
            'feature': feature,
            'rank': data['rank'],
            'combined_score': data['combined_score'],
            'num_methods': data['num_methods']
        }
        
        # Add method-specific information
        for method in ['granger_causality', 'mutual_information', 'correlation_analysis', 'lead_lag_analysis']:
            if feature in causality_results[method]:
                feature_info[method] = causality_results[method][feature]
        
        top_features.append(feature_info)
    
    return top_features


def create_causal_network(causality_results: Dict, threshold: float = 0.05) -> Dict:
    """
    Create a causal network representation from causality analysis.
    
    Args:
        causality_results: Results from compute_causality
        threshold: Threshold for including edges in the network
        
    Returns:
        Dictionary representing the causal network
    """
    network = {
        'nodes': [],
        'edges': [],
        'metadata': {
            'threshold': threshold,
            'total_nodes': 0,
            'total_edges': 0
        }
    }
    
    # Add nodes (features)
    if 'causal_ranking' in causality_results:
        for feature, data in causality_results['causal_ranking'].items():
            network['nodes'].append({
                'id': feature,
                'score': data['combined_score'],
                'rank': data['rank'],
                'methods': data['num_methods']
            })
    
    # Add edges based on significant relationships
    for feature, data in causality_results.get('granger_causality', {}).items():
        if data['significant'] and data['p_value'] < threshold:
            network['edges'].append({
                'source': feature,
                'target': 'close',
                'type': 'granger',
                'strength': data['f_statistic'],
                'lag': data['best_lag'],
                'p_value': data['p_value']
            })
    
    for feature, data in causality_results.get('lead_lag_analysis', {}).items():
        if data['best_abs_correlation'] > threshold:
            edge_type = 'leads' if data['leads_target'] else 'lags'
            network['edges'].append({
                'source': feature,
                'target': 'close',
                'type': edge_type,
                'strength': data['best_abs_correlation'],
                'lag': data['best_lag']
            })
    
    network['metadata']['total_nodes'] = len(network['nodes'])
    network['metadata']['total_edges'] = len(network['edges'])
    
    return network
