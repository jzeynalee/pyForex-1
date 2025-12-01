# models/trend_classifier.py
"""
Lightweight XGBoost classifier for Trend Detector Step 4.

This model predicts trend direction from tabular features:
- Structural analysis scores
- Multi-timeframe confluence
- Technical indicators (ADX, DI, EMAs, ROC)

Designed to work with FusionFXTrendDetector._prepare_ml_input()
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrendClassifierConfig:
    """Configuration for trend classifier training."""
    # Feature configuration (must match _prepare_ml_input order)
    feature_names: Tuple[str, ...] = (
        'struct_score',
        'mtf_score', 
        'regime',
        'adx',
        'plus_di',
        'minus_di',
        'price_above_ema20',
        'price_above_ema50',
        'price_above_ema200',
        'ema_alignment',
        'vol_compression',
        'roc_5',
        'roc_10',
    )
    
    # Target classes
    target_classes: Tuple[str, ...] = ('BEARISH', 'SIDEWAYS', 'BULLISH')
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    use_time_series_split: bool = True
    n_splits: int = 5


class TrendClassifier:
    """
    Lightweight trend direction classifier for FTDM-V1 Step 4.
    
    Provides sklearn-compatible interface expected by FusionFXTrendDetector:
    - predict_proba(X) -> array of shape (n_samples, 3)
    
    Output probabilities: [P(BEARISH), P(SIDEWAYS), P(BULLISH)]
    """
    
    def __init__(self, config: Optional[TrendClassifierConfig] = None):
        self.config = config or TrendClassifierConfig()
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._model_type = 'xgboost' if XGB_AVAILABLE else 'sklearn'
    
    def _create_model(self):
        """Create the underlying classifier."""
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                random_state=self.config.random_state,
            )
        elif SKLEARN_AVAILABLE:
            logger.info("XGBoost not available, using GradientBoostingClassifier")
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                random_state=self.config.random_state,
            )
        else:
            raise RuntimeError("Neither XGBoost nor sklearn available")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validate: bool = True,
    ) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, 13)
            y: Target labels (-1=BEARISH, 0=SIDEWAYS, 1=BULLISH)
            validate: Whether to perform cross-validation
        
        Returns:
            Dict with training metrics
        """
        X = np.array(X)
        y = np.array(y)
        
        # Map labels to 0, 1, 2 for classifier
        y_mapped = y + 1  # -1,0,1 -> 0,1,2
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        self.model = self._create_model()
        
        metrics = {}
        
        if validate:
            # Cross-validation
            if self.config.use_time_series_split:
                cv = TimeSeriesSplit(n_splits=self.config.n_splits)
            else:
                cv = self.config.n_splits
            
            cv_scores = cross_val_score(
                self.model, X_scaled, y_mapped,
                cv=cv, scoring='accuracy'
            )
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train/test split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_mapped,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=not self.config.use_time_series_split,
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
        metrics['classification_report'] = classification_report(
            y_test, y_pred,
            target_names=list(self.config.target_classes),
            output_dict=True,
        )
        
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=list(self.config.target_classes),
        ))
        
        # Refit on all data for production
        self.model.fit(X_scaled, y_mapped)
        
        return metrics
    
    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix or single sample
        
        Returns:
            Array of shape (n_samples, 3) with [P(BEAR), P(SIDEWAYS), P(BULL)]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix or single sample
        
        Returns:
            Array of labels (-1=BEAR, 0=SIDEWAYS, 1=BULL)
        """
        probs = self.predict_proba(X)
        # Convert back: 0,1,2 -> -1,0,1
        return np.argmax(probs, axis=1) - 1
    
    def save(self, path: Union[str, Path]):
        """Save model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_type': self._model_type,
        }
        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrendClassifier':
        """Load model from disk."""
        path = Path(path)
        save_dict = joblib.load(path)
        
        instance = cls(config=save_dict['config'])
        instance.model = save_dict['model']
        instance.scaler = save_dict['scaler']
        instance._model_type = save_dict.get('model_type', 'unknown')
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model doesn't support feature importance")
        
        df = pd.DataFrame({
            'feature': self.config.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        return df


def generate_synthetic_training_data(
    n_samples: int = 5000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for initial model development.
    
    This creates realistic feature distributions and labels based on
    domain knowledge of how indicators correlate with trends.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Amount of noise to add
        random_state: Random seed
    
    Returns:
        X: Feature matrix (n_samples, 13)
        y: Labels (-1, 0, 1)
    """
    np.random.seed(random_state)
    
    # Generate base trend state
    # Roughly 30% bearish, 40% sideways, 30% bullish
    trend_probs = [0.30, 0.40, 0.30]
    y = np.random.choice([-1, 0, 1], size=n_samples, p=trend_probs)
    
    X = np.zeros((n_samples, 13))
    
    for i in range(n_samples):
        trend = y[i]
        noise = np.random.randn() * noise_level
        
        if trend == 1:  # BULLISH
            # struct_score: High for bullish
            X[i, 0] = np.clip(0.7 + np.random.randn() * 0.15, 0, 1)
            # mtf_score: High alignment
            X[i, 1] = np.clip(0.65 + np.random.randn() * 0.15, 0, 1)
            # regime: 1=trending
            X[i, 2] = 1 if np.random.random() > 0.2 else 0
            # adx: Strong trend
            X[i, 3] = np.clip(35 + np.random.randn() * 10, 0, 100)
            # plus_di > minus_di
            X[i, 4] = np.clip(30 + np.random.randn() * 8, 0, 100)  # plus_di
            X[i, 5] = np.clip(15 + np.random.randn() * 5, 0, 100)  # minus_di
            # Price above EMAs
            X[i, 6] = 1 if np.random.random() > 0.15 else 0  # above ema20
            X[i, 7] = 1 if np.random.random() > 0.2 else 0   # above ema50
            X[i, 8] = 1 if np.random.random() > 0.25 else 0  # above ema200
            # ema_alignment: positive
            X[i, 9] = np.clip(0.7 + np.random.randn() * 0.2, -1, 1)
            # vol_compression: low to moderate
            X[i, 10] = np.clip(0.3 + np.random.randn() * 0.15, 0, 1)
            # ROC: positive
            X[i, 11] = np.clip(2 + np.random.randn() * 1.5, -10, 10)  # roc_5
            X[i, 12] = np.clip(4 + np.random.randn() * 2, -15, 15)    # roc_10
            
        elif trend == -1:  # BEARISH
            X[i, 0] = np.clip(0.25 + np.random.randn() * 0.15, 0, 1)
            X[i, 1] = np.clip(0.3 + np.random.randn() * 0.15, 0, 1)
            X[i, 2] = 1 if np.random.random() > 0.2 else 0
            X[i, 3] = np.clip(32 + np.random.randn() * 10, 0, 100)
            X[i, 4] = np.clip(15 + np.random.randn() * 5, 0, 100)   # plus_di low
            X[i, 5] = np.clip(30 + np.random.randn() * 8, 0, 100)   # minus_di high
            X[i, 6] = 1 if np.random.random() > 0.85 else 0
            X[i, 7] = 1 if np.random.random() > 0.8 else 0
            X[i, 8] = 1 if np.random.random() > 0.75 else 0
            X[i, 9] = np.clip(-0.6 + np.random.randn() * 0.2, -1, 1)
            X[i, 10] = np.clip(0.35 + np.random.randn() * 0.15, 0, 1)
            X[i, 11] = np.clip(-2 + np.random.randn() * 1.5, -10, 10)
            X[i, 12] = np.clip(-4 + np.random.randn() * 2, -15, 15)
            
        else:  # SIDEWAYS
            X[i, 0] = np.clip(0.45 + np.random.randn() * 0.15, 0, 1)
            X[i, 1] = np.clip(0.45 + np.random.randn() * 0.15, 0, 1)
            X[i, 2] = 0  # ranging regime
            X[i, 3] = np.clip(18 + np.random.randn() * 6, 0, 100)  # low ADX
            X[i, 4] = np.clip(22 + np.random.randn() * 6, 0, 100)
            X[i, 5] = np.clip(22 + np.random.randn() * 6, 0, 100)
            X[i, 6] = 1 if np.random.random() > 0.5 else 0
            X[i, 7] = 1 if np.random.random() > 0.5 else 0
            X[i, 8] = 1 if np.random.random() > 0.5 else 0
            X[i, 9] = np.clip(0 + np.random.randn() * 0.3, -1, 1)
            X[i, 10] = np.clip(0.6 + np.random.randn() * 0.15, 0, 1)  # high compression
            X[i, 11] = np.clip(0 + np.random.randn() * 1, -10, 10)
            X[i, 12] = np.clip(0 + np.random.randn() * 1.5, -15, 15)
    
    return X, y


if __name__ == "__main__":
    # Demo: Train on synthetic data
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Trend Classifier Training Demo (Synthetic Data)")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = generate_synthetic_training_data(n_samples=5000)
    print(f"\nGenerated {len(X)} samples")
    print(f"Class distribution: BEAR={sum(y==-1)}, SIDEWAYS={sum(y==0)}, BULL={sum(y==1)}")
    
    # Train classifier
    classifier = TrendClassifier()
    metrics = classifier.fit(X, y, validate=True)
    
    # Feature importance
    print("\nFeature Importance:")
    print(classifier.get_feature_importance().to_string(index=False))
    
    # Save model
    model_path = Path("models/trend_classifier_v1.joblib")
    classifier.save(model_path)
    
    # Test loading and prediction
    loaded = TrendClassifier.load(model_path)
    test_sample = X[0:1]
    probs = loaded.predict_proba(test_sample)
    pred = loaded.predict(test_sample)
    
    print(f"\nTest prediction:")
    print(f"  Probabilities: BEAR={probs[0,0]:.3f}, SIDEWAYS={probs[0,1]:.3f}, BULL={probs[0,2]:.3f}")
    print(f"  Prediction: {pred[0]} (actual: {y[0]})")