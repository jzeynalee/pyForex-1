"""
Phase 3: Meta-Labeling with Gradient Boosting

Meta-labeling (López de Prado) answers: "Given that my primary model says BUY/SELL,
should I actually take this trade?"

The meta-model predicts P(primary_prediction_correct) using:
- Primary model's features and confidence
- Market conditions (volatility, spread, regime)
- Time features
- Historical pattern features

Key benefits:
1. Filters out low-quality signals
2. Improves precision without retraining primary model
3. Can incorporate features the primary model doesn't see
4. Naturally handles regime changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)

# Optional imports - gracefully handle if not installed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Using sklearn GradientBoosting as fallback.")

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN_GB = True
except ImportError:
    HAS_SKLEARN_GB = False


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling model."""
    # Model parameters
    use_lightgbm: bool = True          # Use LightGBM (faster) or sklearn
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1             # L1 regularization
    reg_lambda: float = 0.1            # L2 regularization
    
    # Training parameters
    early_stopping_rounds: int = 20
    n_cv_splits: int = 5               # Time series CV splits
    
    # Prediction threshold
    default_threshold: float = 0.5
    
    # Feature configuration
    include_primary_features: bool = True
    include_primary_confidence: bool = True
    include_market_features: bool = True
    include_time_features: bool = True
    
    # Class weighting
    use_class_weights: bool = True
    
    # Feature groups for importance analysis
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'primary': [],      # From primary model
        'market': [],       # Market conditions
        'time': [],         # Time-based features
        'technical': []     # Technical indicators
    })


class MetaFeatureExtractor:
    """
    Extracts features for the meta-labeling model.
    
    Features include:
    - Primary model's outputs (direction probs, confidence, volatility)
    - Market microstructure (spread, liquidity)
    - Time features (hour, day, session)
    - Recent performance metrics
    """
    
    def __init__(self, config: MetaLabelingConfig):
        self.config = config
        self.feature_names = []
    
    def extract_features(
        self,
        primary_predictions: Dict[str, np.ndarray],
        market_data: pd.DataFrame,
        timestamps: Optional[pd.DatetimeIndex] = None,
        additional_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract all features for meta-labeling.
        
        Args:
            primary_predictions: Dict with 'direction_probs', 'volatility', 'quantiles', 'features'
            market_data: DataFrame with 'spread', 'atr', 'volume', etc.
            timestamps: Datetime index for time features
            additional_features: Any additional features to include
        
        Returns:
            Feature array (n_samples, n_features)
        """
        features_list = []
        self.feature_names = []
        
        n_samples = len(primary_predictions.get('direction_probs', []))
        
        # Primary model features
        if self.config.include_primary_features:
            # Direction probabilities
            dir_probs = primary_predictions.get('direction_probs')
            if dir_probs is not None:
                features_list.append(dir_probs)
                self.feature_names.extend(['prob_bear', 'prob_side', 'prob_bull'])
            
            # Volatility prediction
            volatility = primary_predictions.get('volatility')
            if volatility is not None:
                if volatility.ndim == 1:
                    volatility = volatility.reshape(-1, 1)
                features_list.append(volatility)
                self.feature_names.append('pred_volatility')
            
            # Quantile features (spread between quantiles)
            quantiles = primary_predictions.get('quantiles')
            if quantiles is not None:
                # Q95 - Q5 (prediction interval width)
                interval_width = quantiles[:, -1] - quantiles[:, 0]
                # Asymmetry (Q75-Q50 vs Q50-Q25)
                upper_range = quantiles[:, 3] - quantiles[:, 2]
                lower_range = quantiles[:, 2] - quantiles[:, 1]
                asymmetry = upper_range / (lower_range + 1e-8)
                
                features_list.append(interval_width.reshape(-1, 1))
                features_list.append(asymmetry.reshape(-1, 1))
                self.feature_names.extend(['interval_width', 'asymmetry'])
        
        # Confidence features
        if self.config.include_primary_confidence:
            dir_probs = primary_predictions.get('direction_probs')
            if dir_probs is not None:
                # Max confidence
                confidence = np.max(dir_probs, axis=1, keepdims=True)
                # Entropy (uncertainty)
                entropy = -np.sum(dir_probs * np.log(dir_probs + 1e-8), axis=1, keepdims=True)
                # Second highest probability
                sorted_probs = np.sort(dir_probs, axis=1)
                second_best = sorted_probs[:, -2].reshape(-1, 1)
                # Margin (difference between top two)
                margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
                
                features_list.extend([confidence, entropy, second_best, margin])
                self.feature_names.extend(['confidence', 'entropy', 'second_best_prob', 'margin'])
        
        # Market features
        if self.config.include_market_features and market_data is not None:
            market_features = []
            market_names = []
            
            if 'spread' in market_data.columns:
                market_features.append(market_data['spread'].values.reshape(-1, 1))
                market_names.append('spread')
            
            if 'atr' in market_data.columns:
                market_features.append(market_data['atr'].values.reshape(-1, 1))
                market_names.append('atr')
            
            if 'volume' in market_data.columns:
                vol = market_data['volume'].values.reshape(-1, 1)
                # Normalize volume
                vol_ma = pd.Series(vol.flatten()).rolling(20).mean().fillna(method='bfill').values
                rel_volume = vol.flatten() / (vol_ma + 1e-8)
                market_features.append(rel_volume.reshape(-1, 1))
                market_names.append('relative_volume')
            
            # Volatility regime (ATR percentile)
            if 'atr' in market_data.columns:
                atr = market_data['atr'].values
                atr_pct = pd.Series(atr).rolling(50).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
                ).fillna(0.5).values
                market_features.append(atr_pct.reshape(-1, 1))
                market_names.append('volatility_regime')
            
            if market_features:
                features_list.extend(market_features)
                self.feature_names.extend(market_names)
        
        # Time features
        if self.config.include_time_features and timestamps is not None:
            time_features = []
            
            # Hour of day (cyclical encoding)
            hour = timestamps.hour.values
            time_features.append(np.sin(2 * np.pi * hour / 24).reshape(-1, 1))
            time_features.append(np.cos(2 * np.pi * hour / 24).reshape(-1, 1))
            self.feature_names.extend(['hour_sin', 'hour_cos'])
            
            # Day of week (cyclical)
            day = timestamps.dayofweek.values
            time_features.append(np.sin(2 * np.pi * day / 5).reshape(-1, 1))
            time_features.append(np.cos(2 * np.pi * day / 5).reshape(-1, 1))
            self.feature_names.extend(['day_sin', 'day_cos'])
            
            # Session indicators
            hour_arr = timestamps.hour.values
            is_tokyo = ((hour_arr >= 0) & (hour_arr < 9)).astype(float).reshape(-1, 1)
            is_london = ((hour_arr >= 8) & (hour_arr < 17)).astype(float).reshape(-1, 1)
            is_ny = ((hour_arr >= 13) & (hour_arr < 22)).astype(float).reshape(-1, 1)
            is_overlap = ((hour_arr >= 13) & (hour_arr < 17)).astype(float).reshape(-1, 1)
            
            time_features.extend([is_tokyo, is_london, is_ny, is_overlap])
            self.feature_names.extend(['is_tokyo', 'is_london', 'is_ny', 'is_overlap'])
            
            features_list.extend(time_features)
        
        # Additional features
        if additional_features is not None:
            features_list.append(additional_features)
            n_additional = additional_features.shape[1] if additional_features.ndim > 1 else 1
            self.feature_names.extend([f'additional_{i}' for i in range(n_additional)])
        
        # Concatenate all features
        if features_list:
            return np.hstack(features_list)
        else:
            return np.zeros((n_samples, 1))
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


class MetaLabelingModel:
    """
    Gradient Boosting model for meta-labeling.
    
    Predicts: P(primary_model_correct | features)
    
    Usage:
    1. Train primary model, generate predictions
    2. Generate triple barrier labels (ground truth)
    3. Create meta-labels: 1 if primary prediction matches barrier outcome, 0 otherwise
    4. Train meta-model on meta-features -> meta-labels
    5. At inference: filter trades where P(correct) < threshold
    """
    
    def __init__(self, config: Optional[MetaLabelingConfig] = None):
        self.config = config or MetaLabelingConfig()
        self.model = None
        self.feature_extractor = MetaFeatureExtractor(self.config)
        self.threshold = self.config.default_threshold
        self.feature_importances_ = None
        self.cv_scores_ = None
    
    def _create_model(self):
        """Create the gradient boosting model."""
        if self.config.use_lightgbm and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif HAS_SKLEARN_GB:
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_child_samples,
                subsample=self.config.subsample,
                random_state=42
            )
        else:
            raise ImportError("Neither LightGBM nor sklearn GradientBoosting available")
    
    def create_meta_labels(
        self,
        primary_directions: np.ndarray,
        barrier_outcomes: np.ndarray
    ) -> np.ndarray:
        """
        Create meta-labels from primary predictions and barrier outcomes.
        
        Meta-label = 1 if primary prediction matches barrier outcome
        Meta-label = 0 if primary prediction doesn't match
        
        Args:
            primary_directions: Primary model's predicted directions (-1, 0, 1)
            barrier_outcomes: Triple barrier outcomes (-1 LOSS, 0 TIMEOUT, 1 WIN)
        
        Returns:
            Binary meta-labels array
        """
        # For BUY predictions (direction=1), WIN (outcome=1) is correct
        # For SELL predictions (direction=-1), WIN (outcome=1) is also correct
        # (because barrier is set appropriately for direction)
        
        # Meta-label is 1 if:
        # - Direction was non-zero (actual trade)
        # - Outcome was WIN
        
        took_trade = primary_directions != 0
        trade_won = barrier_outcomes == 1
        
        meta_labels = (took_trade & trade_won).astype(int)
        
        return meta_labels
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the meta-labeling model.
        
        Args:
            X: Meta-features
            y: Meta-labels (0 or 1)
            sample_weights: Optional sample weights
            validation_split: Fraction for validation
        
        Returns:
            Training metrics
        """
        # Time series split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if sample_weights is not None:
            sw_train = sample_weights[:split_idx]
        else:
            sw_train = None
        
        # Class weights
        class_weights = None
        if self.config.use_class_weights:
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            class_weights = {c: total / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
        
        # Create and train model
        self.model = self._create_model()
        
        if self.config.use_lightgbm and HAS_LIGHTGBM:
            # LightGBM with early stopping
            self.model.fit(
                X_train, y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
            )
        else:
            # Sklearn
            self.model.fit(X_train, y_train, sample_weight=sw_train)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            self.feature_extractor.get_feature_names(),
            self.model.feature_importances_
        ))
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        }
        
        logger.info(f"Meta-model trained: {metrics}")
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Time series cross-validation.
        
        Returns metrics for each fold.
        """
        n_splits = n_splits or self.config.n_cv_splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._create_model()
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
            if len(np.unique(y_val)) > 1:
                cv_metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            else:
                cv_metrics['roc_auc'].append(0.5)
        
        self.cv_scores_ = cv_metrics
        
        # Log summary
        logger.info(f"CV Results (mean ± std):")
        for metric, values in cv_metrics.items():
            logger.info(f"  {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        return cv_metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of primary model being correct.
        
        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def should_trade(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Decide whether to take each trade.
        
        Args:
            X: Meta-features
            threshold: Probability threshold (uses default if not specified)
        
        Returns:
            Boolean array (True = take trade)
        """
        threshold = threshold or self.threshold
        probs = self.predict_proba(X)
        return probs >= threshold
    
    def optimize_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'f1',
        thresholds: Optional[List[float]] = None
    ) -> float:
        """
        Find optimal threshold for a given metric.
        
        Args:
            X: Validation features
            y: Validation labels
            metric: 'f1', 'precision', or 'recall'
            thresholds: Thresholds to try
        
        Returns:
            Optimal threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.3, 0.8, 0.05)
        
        probs = self.predict_proba(X)
        
        best_score = 0
        best_threshold = 0.5
        
        for t in thresholds:
            y_pred = (probs >= t).astype(int)
            
            if metric == 'f1':
                score = f1_score(y, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = t
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}={best_score:.3f})")
        self.threshold = best_threshold
        
        return best_threshold
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features."""
        if self.feature_importances_ is None:
            return {}
        
        sorted_features = sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def save(self, path: str):
        """Save model to file."""
        import joblib
        
        save_dict = {
            'model': self.model,
            'config': self.config,
            'threshold': self.threshold,
            'feature_importances': self.feature_importances_,
            'feature_names': self.feature_extractor.get_feature_names()
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Meta-labeling model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MetaLabelingModel':
        """Load model from file."""
        import joblib
        
        save_dict = joblib.load(path)
        
        instance = cls(save_dict['config'])
        instance.model = save_dict['model']
        instance.threshold = save_dict['threshold']
        instance.feature_importances_ = save_dict['feature_importances']
        instance.feature_extractor.feature_names = save_dict['feature_names']
        
        logger.info(f"Meta-labeling model loaded from {path}")
        return instance


class TradeFilter:
    """
    High-level interface for filtering trades using meta-labeling.
    
    Combines feature extraction and prediction into a simple API.
    """
    
    def __init__(
        self,
        meta_model: MetaLabelingModel,
        min_confidence: float = 0.5,
        min_meta_score: float = 0.5
    ):
        """
        Args:
            meta_model: Trained MetaLabelingModel
            min_confidence: Minimum primary model confidence
            min_meta_score: Minimum meta-model score to trade
        """
        self.meta_model = meta_model
        self.min_confidence = min_confidence
        self.min_meta_score = min_meta_score
    
    def filter(
        self,
        signal: str,
        features: Dict[str, Any],
        direction_confidence: float
    ) -> 'FilterResult':
        """
        Filter a single trading signal.
        
        Args:
            signal: Direction signal ('BUY' or 'SELL')
            features: Meta-features dict
            direction_confidence: Primary model confidence
            
        Returns:
            FilterResult with should_trade and meta_score
        """
        from dataclasses import dataclass
        
        @dataclass
        class FilterResult:
            should_trade: bool
            meta_score: float
            reason: str = ""
        
        # Check confidence threshold
        if direction_confidence < self.min_confidence:
            return FilterResult(
                should_trade=False,
                meta_score=0.0,
                reason=f"Low confidence: {direction_confidence:.2f} < {self.min_confidence}"
            )
        
        # Get meta-model prediction if model is available
        meta_score = 0.5  # Default neutral score
        if self.meta_model is not None:
            try:
                # Convert features dict to array for prediction
                feature_values = []
                for key in sorted(features.keys()):
                    val = features[key]
                    if isinstance(val, (int, float)):
                        feature_values.append(val)
                    elif isinstance(val, np.ndarray):
                        feature_values.extend(val.flatten().tolist())
                
                if feature_values:
                    X = np.array([feature_values])
                    meta_score = self.meta_model.predict_proba(X)[0]
            except Exception as e:
                # If prediction fails, use neutral score
                meta_score = 0.5
        
        should_trade = meta_score >= self.min_meta_score
        
        return FilterResult(
            should_trade=should_trade,
            meta_score=meta_score,
            reason="" if should_trade else f"Meta score {meta_score:.2f} < {self.min_meta_score}"
        )
    
    def filter_signals(
        self,
        primary_predictions: Dict[str, np.ndarray],
        market_data: pd.DataFrame,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter trading signals using meta-labeling.
        
        Args:
            primary_predictions: Primary model outputs
            market_data: Market condition data
            timestamps: Datetime index
        
        Returns:
            (should_trade_mask, meta_scores)
        """
        # Extract meta-features
        meta_features = self.meta_model.feature_extractor.extract_features(
            primary_predictions=primary_predictions,
            market_data=market_data,
            timestamps=timestamps
        )
        
        # Get meta-model scores
        meta_scores = self.meta_model.predict_proba(meta_features)
        
        # Primary model confidence
        direction_probs = primary_predictions.get('direction_probs')
        if direction_probs is not None:
            confidence = np.max(direction_probs, axis=1)
        else:
            confidence = np.ones(len(meta_scores))
        
        # Combined filter
        should_trade = (
            (confidence >= self.min_confidence) &
            (meta_scores >= self.min_meta_score)
        )
        
        return should_trade, meta_scores
    
    def get_filter_stats(
        self,
        should_trade: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate filtering effectiveness statistics.
        
        Args:
            should_trade: Boolean mask from filter_signals
            actual_outcomes: Actual trade outcomes (1=win, 0=loss)
        
        Returns:
            Statistics about filtering effectiveness
        """
        total_signals = len(should_trade)
        passed_filter = should_trade.sum()
        filtered_out = total_signals - passed_filter
        
        # Win rates
        if passed_filter > 0:
            filtered_win_rate = actual_outcomes[should_trade].mean()
        else:
            filtered_win_rate = 0
        
        if filtered_out > 0:
            rejected_would_be_win_rate = actual_outcomes[~should_trade].mean()
        else:
            rejected_would_be_win_rate = 0
        
        overall_win_rate = actual_outcomes.mean()
        
        return {
            'total_signals': total_signals,
            'passed_filter': passed_filter,
            'filtered_out': filtered_out,
            'filter_rate': filtered_out / total_signals if total_signals > 0 else 0,
            'overall_win_rate': overall_win_rate,
            'filtered_win_rate': filtered_win_rate,
            'rejected_win_rate': rejected_would_be_win_rate,
            'win_rate_improvement': filtered_win_rate - overall_win_rate
        }
