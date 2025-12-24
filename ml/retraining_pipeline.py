"""
Retraining Pipeline Module for pyForex ML System.

Handles the end-to-end retraining process including data preparation,
model training, hyperparameter optimization, and validation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
from enum import Enum

from utils.feature_schema import get_feature_schema_version

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the retraining pipeline."""
    INITIALIZED = "initialized"
    DATA_COLLECTION = "data_collection"
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_VALIDATION = "data_validation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_COMPARISON = "model_comparison"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result of a retraining pipeline run."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: PipelineStage
    success: bool
    model_id: Optional[str]
    
    # Metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    comparison_result: Optional[Dict] = None
    
    # Data info
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    
    # Timing
    stage_durations: Dict[str, float] = field(default_factory=dict)
    
    # Error info
    error_message: Optional[str] = None
    error_stage: Optional[PipelineStage] = None
    
    # Model deployment
    deployed: bool = False
    previous_model_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'success': self.success,
            'model_id': self.model_id,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'comparison_result': self.comparison_result,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'feature_count': self.feature_count,
            'stage_durations': self.stage_durations,
            'error_message': self.error_message,
            'error_stage': self.error_stage.value if self.error_stage else None,
            'deployed': self.deployed,
            'previous_model_id': self.previous_model_id
        }


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    feature_names: List[str] = field(default_factory=list)
    
    @property
    def training_samples(self) -> int:
        return len(self.X_train)
    
    @property
    def validation_samples(self) -> int:
        return len(self.X_val)
    
    @property
    def test_samples(self) -> int:
        return len(self.X_test) if self.X_test is not None else 0


class RetrainingPipeline:
    """
    Orchestrates the complete model retraining process.
    
    Pipeline stages:
    1. Data Collection - Fetch raw data from providers
    2. Data Preparation - Clean, filter, align data
    3. Feature Engineering - Generate ML features
    4. Data Validation - Check data quality
    5. Hyperparameter Tuning - Optionally tune hyperparameters
    6. Model Training - Train the model
    7. Model Validation - Validate on holdout set
    8. Model Comparison - Compare against current model
    9. Deployment - Deploy if validation passes
    """
    
    def __init__(
        self,
        config: 'RetrainingConfig',
        model_manager: 'ModelManager',
        feature_builder: Optional['MTFFeatureBuilder'] = None,
        data_provider: Optional['MTFDataProvider'] = None
    ):
        self.config = config
        self.model_manager = model_manager
        self.feature_builder = feature_builder
        self.data_provider = data_provider
        
        # Pipeline state
        self.current_stage: PipelineStage = PipelineStage.INITIALIZED
        self.stage_times: Dict[str, datetime] = {}
        
        # Results
        self.run_history: List[PipelineResult] = []
        
        logger.info("RetrainingPipeline initialized")
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{hash_suffix}"
    
    def _set_stage(self, stage: PipelineStage) -> None:
        """Update current stage and record timing."""
        self.stage_times[stage.value] = datetime.now()
        self.current_stage = stage
        logger.info(f"Pipeline stage: {stage.value}")
    
    def _calc_stage_duration(self, stage: PipelineStage) -> float:
        """Calculate duration of a stage in seconds."""
        if stage.value not in self.stage_times:
            return 0.0
        
        stage_order = list(PipelineStage)
        stage_idx = stage_order.index(stage)
        
        start_time = self.stage_times[stage.value]
        
        # Find next stage that has a timestamp
        for next_stage in stage_order[stage_idx + 1:]:
            if next_stage.value in self.stage_times:
                return (self.stage_times[next_stage.value] - start_time).total_seconds()
        
        # If no next stage, calculate to now
        return (datetime.now() - start_time).total_seconds()
    
    def run(
        self,
        profile_name: str,
        symbols: List[str],
        force_retrain: bool = False,
        skip_validation: bool = False
    ) -> PipelineResult:
        """
        Execute the full retraining pipeline.
        
        Args:
            profile_name: MTF profile to use (SCALP, SWING)
            symbols: Currency pairs to train on
            force_retrain: Skip model comparison check
            skip_validation: Skip validation stage (not recommended)
        
        Returns:
            PipelineResult with status and metrics
        """
        run_id = self._generate_run_id()
        started_at = datetime.now()
        
        result = PipelineResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=None,
            status=PipelineStage.INITIALIZED,
            success=False,
            model_id=None
        )
        
        logger.info(f"Starting retraining pipeline: {run_id}")
        logger.info(f"Profile: {profile_name}, Symbols: {symbols}")
        
        try:
            # Stage 1: Data Collection
            self._set_stage(PipelineStage.DATA_COLLECTION)
            raw_data = self._collect_data(symbols, profile_name)
            
            if raw_data is None or len(raw_data) == 0:
                raise ValueError("No data collected")
            
            # Stage 2: Data Preparation
            self._set_stage(PipelineStage.DATA_PREPARATION)
            prepared_data = self._prepare_data(raw_data)
            
            # Stage 3: Feature Engineering
            self._set_stage(PipelineStage.FEATURE_ENGINEERING)
            features_df, target = self._engineer_features(prepared_data)
            result.feature_count = len(features_df.columns)
            
            # Stage 4: Data Validation
            self._set_stage(PipelineStage.DATA_VALIDATION)
            if not self._validate_data(features_df, target):
                raise ValueError("Data validation failed")
            
            # Stage 5: Split Data
            data_split = self._split_data(features_df, target)
            result.training_samples = data_split.training_samples
            result.validation_samples = data_split.validation_samples
            
            # Stage 6: Hyperparameter Tuning (optional)
            if self.config.model.tune_hyperparameters:
                self._set_stage(PipelineStage.HYPERPARAMETER_TUNING)
                best_params = self._tune_hyperparameters(data_split)
            else:
                best_params = self._get_default_params()
            
            # Stage 7: Model Training
            self._set_stage(PipelineStage.MODEL_TRAINING)
            model, training_metrics = self._train_model(data_split, best_params)
            result.training_metrics = training_metrics
            
            # Stage 8: Model Validation
            self._set_stage(PipelineStage.MODEL_VALIDATION)
            validation_metrics = self._validate_model(model, data_split)
            result.validation_metrics = validation_metrics
            
            # Stage 9: Model Comparison
            if not skip_validation:
                self._set_stage(PipelineStage.MODEL_COMPARISON)
                comparison_passed, comparison_result = self._compare_models(
                    validation_metrics, profile_name, force_retrain
                )
                result.comparison_result = comparison_result
                
                if not comparison_passed and not force_retrain:
                    raise ValueError(
                        f"Model comparison failed: {comparison_result.get('reason', 'unknown')}"
                    )
            else:
                comparison_passed = True
            
            # Stage 10: Deployment
            self._set_stage(PipelineStage.DEPLOYMENT)
            model_id = self._deploy_model(
                model=model,
                profile_name=profile_name,
                feature_names=list(features_df.columns),
                hyperparameters=best_params,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                training_samples=result.training_samples,
                training_data=prepared_data
            )
            
            result.model_id = model_id
            result.deployed = True
            
            # Get previous model ID before switching
            current_active = self.model_manager.active_models.get(profile_name)
            result.previous_model_id = current_active
            
            # Activate new model
            if comparison_passed or force_retrain:
                self.model_manager.activate_model(model_id, force=force_retrain)
            
            # Success
            self._set_stage(PipelineStage.COMPLETED)
            result.status = PipelineStage.COMPLETED
            result.success = True
            
            logger.info(f"Pipeline completed successfully: {model_id}")
            
        except Exception as e:
            logger.error(f"Pipeline failed at stage {self.current_stage.value}: {e}")
            result.status = PipelineStage.FAILED
            result.error_message = str(e)
            result.error_stage = self.current_stage
            result.success = False
        
        # Record completion time and stage durations
        result.completed_at = datetime.now()
        result.stage_durations = {
            stage.value: self._calc_stage_duration(stage)
            for stage in PipelineStage
            if stage.value in self.stage_times
        }
        
        # Add to history
        self.run_history.append(result)
        
        return result
    
    def _collect_data(
        self,
        symbols: List[str],
        profile_name: str
    ) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        """Collect raw data from data provider."""
        if self.data_provider is None:
            logger.warning("No data provider configured, generating mock data")
            return self._generate_mock_data(symbols, profile_name)
        
        try:
            all_data = {}
            for symbol in symbols:
                data = self.data_provider.fetch_for_profile(
                    symbol=symbol,
                    profile_name=profile_name
                )
                if data:
                    all_data[symbol] = data
            
            return all_data if all_data else None
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return None
    
    def _generate_mock_data(
        self,
        symbols: List[str],
        profile_name: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate mock data for testing."""
        # Get timeframes from profile
        from ..utils.mtf_config import SCALP_PROFILE, SWING_PROFILE
        
        profile = SCALP_PROFILE if profile_name == "SCALP" else SWING_PROFILE
        
        all_data = {}
        for symbol in symbols:
            symbol_data = {}
            for tf in profile.timeframes:
                n_bars = self.config.data.max_training_samples // len(profile.timeframes)
                dates = pd.date_range(
                    end=datetime.now(),
                    periods=n_bars,
                    freq='5min' if tf.value == 'M5' else '15min'
                )
                
                base_price = 1.1000 if 'EUR' in symbol else 150.0
                
                df = pd.DataFrame({
                    'time': dates,
                    'open': base_price + np.random.randn(n_bars).cumsum() * 0.001,
                    'high': 0,
                    'low': 0,
                    'close': 0,
                    'tick_volume': np.random.randint(100, 10000, n_bars),
                    'spread': np.random.randint(1, 5, n_bars),
                    'real_volume': np.random.randint(0, 1000, n_bars)
                })
                
                df['high'] = df['open'] + abs(np.random.randn(n_bars)) * 0.0005
                df['low'] = df['open'] - abs(np.random.randn(n_bars)) * 0.0005
                df['close'] = df['open'] + np.random.randn(n_bars) * 0.0003
                df.set_index('time', inplace=True)
                
                symbol_data[tf.value] = df
            
            all_data[symbol] = symbol_data
        
        return all_data
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Clean and prepare data for feature engineering."""
        prepared = {}
        
        for symbol, timeframes in raw_data.items():
            prepared[symbol] = {}
            
            for tf_name, df in timeframes.items():
                # Remove NaN rows
                df_clean = df.dropna()
                
                # Remove outliers (optional)
                if self.config.data.max_outlier_ratio > 0:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_clean.columns:
                            q1 = df_clean[col].quantile(0.001)
                            q99 = df_clean[col].quantile(0.999)
                            df_clean = df_clean[
                                (df_clean[col] >= q1) & (df_clean[col] <= q99)
                            ]
                
                # Ensure minimum samples
                if len(df_clean) >= 100:
                    prepared[symbol][tf_name] = df_clean
        
        return prepared
    
    def _engineer_features(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate features from prepared data."""
        if self.feature_builder is not None:
            # Use MTF feature builder
            features_list = []
            
            for symbol, timeframes in data.items():
                features = self.feature_builder.build_features(timeframes)
                if features is not None:
                    features['symbol'] = symbol
                    features_list.append(features)
            
            if features_list:
                features_df = pd.concat(features_list, axis=0)
            else:
                raise ValueError("Feature engineering produced no features")
        else:
            # Basic feature engineering fallback
            features_df = self._basic_feature_engineering(data)
        
        # Generate target variable
        target = self._generate_target(features_df, data)
        
        # Align features and target
        common_idx = features_df.index.intersection(target.index)
        features_df = features_df.loc[common_idx]
        target = target.loc[common_idx]
        
        # Drop rows with NaN
        mask = ~(features_df.isna().any(axis=1) | target.isna())
        features_df = features_df[mask]
        target = target[mask]
        
        return features_df, target
    
    def _basic_feature_engineering(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Basic feature engineering when MTFFeatureBuilder not available."""
        features_list = []
        
        for symbol, timeframes in data.items():
            # Use lowest timeframe as base
            base_tf = sorted(timeframes.keys())[0]
            df = timeframes[base_tf].copy()
            
            # Basic features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for period in [10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            df['atr'] = self._calc_atr(df, 14)
            
            # Momentum
            df['rsi'] = self._calc_rsi(df['close'], 14)
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            df['symbol'] = symbol
            features_list.append(df)
        
        return pd.concat(features_list)
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _generate_target(
        self,
        features_df: pd.DataFrame,
        data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> pd.Series:
        """Generate target variable (future returns classification)."""
        # Use lowest timeframe data for target
        targets = []
        
        for symbol, timeframes in data.items():
            base_tf = sorted(timeframes.keys())[0]
            df = timeframes[base_tf]
            
            # Future returns (e.g., next 5 bars)
            future_return = df['close'].shift(-5) / df['close'] - 1
            
            # Classify: -1 (sell), 0 (neutral), 1 (buy)
            threshold = 0.001  # 0.1%
            target = pd.Series(0, index=df.index)
            target[future_return > threshold] = 1
            target[future_return < -threshold] = -1
            
            targets.append(target)
        
        return pd.concat(targets)
    
    def _validate_data(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> bool:
        """Validate data quality before training."""
        issues = []
        
        # Check minimum samples
        if len(features_df) < self.config.data.min_training_samples:
            issues.append(
                f"Insufficient samples: {len(features_df)} < {self.config.data.min_training_samples}"
            )
        
        # Check for excessive NaN
        nan_ratio = features_df.isna().mean().mean()
        if nan_ratio > self.config.data.max_missing_ratio:
            issues.append(f"Too many NaN values: {nan_ratio:.2%}")
        
        # Check class balance
        class_counts = target.value_counts(normalize=True)
        if any(class_counts < 0.1):
            logger.warning(f"Class imbalance detected: {class_counts.to_dict()}")
        
        # Check feature variance
        low_var_features = features_df.columns[features_df.std() < 1e-10]
        if len(low_var_features) > 0:
            logger.warning(f"Low variance features: {list(low_var_features)[:5]}")
        
        if issues:
            for issue in issues:
                logger.error(f"Data validation issue: {issue}")
            return False
        
        logger.info(f"Data validation passed: {len(features_df)} samples, "
                   f"{len(features_df.columns)} features")
        return True
    
    def _split_data(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> DataSplit:
        """Split data into train/validation/test sets."""
        # Sort by index (time) for proper time-series split
        features_df = features_df.sort_index()
        target = target.sort_index()
        
        n = len(features_df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        X_train = features_df.iloc[:train_end]
        y_train = target.iloc[:train_end]
        X_val = features_df.iloc[train_end:val_end]
        y_val = target.iloc[train_end:val_end]
        X_test = features_df.iloc[val_end:]
        y_test = target.iloc[val_end:]
        
        # Drop non-numeric columns for training
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        return DataSplit(
            X_train=X_train[numeric_cols],
            y_train=y_train,
            X_val=X_val[numeric_cols],
            y_val=y_val,
            X_test=X_test[numeric_cols],
            y_test=y_test,
            feature_names=numeric_cols
        )
    
    def _tune_hyperparameters(self, data_split: DataSplit) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna or similar."""
        logger.info("Hyperparameter tuning starting...")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                
                model = self._create_model(params)
                model.fit(
                    data_split.X_train, data_split.y_train,
                    eval_set=[(data_split.X_val, data_split.y_val)],
                    verbose=False
                )
                
                preds = model.predict(data_split.X_val)
                accuracy = (preds == data_split.y_val).mean()
                return accuracy
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective,
                n_trials=self.config.model.tuning_trials,
                timeout=self.config.model.tuning_timeout_minutes * 60
            )
            
            logger.info(f"Best params: {study.best_params}")
            return study.best_params
            
        except ImportError:
            logger.warning("Optuna not available, using default params")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters."""
        return {
            'n_estimators': self.config.model.n_estimators,
            'learning_rate': self.config.model.learning_rate,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create model instance based on config."""
        model_type = self.config.model.model_type
        
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
            except ImportError:
                pass
        
        if model_type == 'lightgbm' or model_type == 'xgboost':
            try:
                import lightgbm as lgb
                lgb_params = {
                    'n_estimators': params.get('n_estimators', 500),
                    'learning_rate': params.get('learning_rate', 0.05),
                    'max_depth': params.get('max_depth', 6),
                    'num_leaves': 31,
                    'verbose': -1
                }
                return lgb.LGBMClassifier(**lgb_params)
            except ImportError:
                pass
        
        # Fallback to sklearn
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            random_state=42
        )
    
    def _train_model(
        self,
        data_split: DataSplit,
        params: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train the model."""
        logger.info("Training model...")
        
        model = self._create_model(params)
        
        # Fit with early stopping if available
        try:
            model.fit(
                data_split.X_train, data_split.y_train,
                eval_set=[(data_split.X_val, data_split.y_val)],
                callbacks=[self._early_stopping_callback()]
            )
        except (TypeError, AttributeError):
            model.fit(data_split.X_train, data_split.y_train)
        
        # Calculate training metrics
        train_preds = model.predict(data_split.X_train)
        train_metrics = {
            'train_accuracy': (train_preds == data_split.y_train).mean(),
        }
        
        logger.info(f"Training metrics: {train_metrics}")
        return model, train_metrics
    
    def _early_stopping_callback(self):
        """Create early stopping callback."""
        try:
            import lightgbm as lgb
            return lgb.early_stopping(self.config.model.early_stopping_rounds)
        except ImportError:
            return None
    
    def _validate_model(
        self,
        model: Any,
        data_split: DataSplit
    ) -> Dict[str, float]:
        """Validate model on holdout set."""
        logger.info("Validating model...")
        
        val_preds = model.predict(data_split.X_val)
        
        # Calculate metrics
        metrics = {
            'val_accuracy': (val_preds == data_split.y_val).mean(),
        }
        
        # Per-class metrics
        for cls in [-1, 0, 1]:
            mask = data_split.y_val == cls
            if mask.sum() > 0:
                class_acc = (val_preds[mask] == data_split.y_val[mask]).mean()
                metrics[f'val_accuracy_class_{cls}'] = class_acc
        
        # Test set metrics if available
        if data_split.X_test is not None and len(data_split.X_test) > 0:
            test_preds = model.predict(data_split.X_test)
            metrics['test_accuracy'] = (test_preds == data_split.y_test).mean()
        
        # Prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                val_proba = model.predict_proba(data_split.X_val)
                metrics['avg_confidence'] = val_proba.max(axis=1).mean()
            except:
                pass
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def _compare_models(
        self,
        new_metrics: Dict[str, float],
        profile_name: str,
        force: bool = False
    ) -> Tuple[bool, Dict]:
        """Compare new model against current active model."""
        current_id = self.model_manager.active_models.get(profile_name)
        
        if current_id is None:
            logger.info("No current model to compare against")
            return True, {'reason': 'no_baseline', 'passed': True}
        
        current_meta = self.model_manager.registry.get(current_id)
        if current_meta is None:
            return True, {'reason': 'baseline_not_found', 'passed': True}
        
        current_metrics = current_meta.validation_metrics
        
        # Compare primary metric
        primary = self.config.validation.min_improvement_pct
        new_val = new_metrics.get('val_accuracy', 0)
        old_val = current_metrics.get('val_accuracy', 0)
        
        if old_val > 0:
            improvement = (new_val - old_val) / old_val * 100
        else:
            improvement = 100 if new_val > 0 else 0
        
        passed = improvement >= -primary  # Allow small regression
        
        comparison = {
            'new_accuracy': new_val,
            'old_accuracy': old_val,
            'improvement_pct': improvement,
            'threshold_pct': primary,
            'passed': passed,
            'reason': 'sufficient_improvement' if passed else 'insufficient_improvement'
        }
        
        logger.info(f"Model comparison: {improvement:.2f}% improvement, "
                   f"passed={passed}")
        
        return passed, comparison
    
    def _deploy_model(
        self,
        model: Any,
        profile_name: str,
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        training_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        training_samples: int,
        training_data: Any
    ) -> str:
        """Deploy model through model manager."""
        version = datetime.now().strftime("v%Y%m%d_%H%M")
        
        model_id = self.model_manager.save_model(
            model=model,
            profile_name=profile_name,
            version=version,
            model_type=self.config.model.model_type,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            feature_schema_version=get_feature_schema_version(),
            training_data=training_data,
            training_start=self.stage_times.get(PipelineStage.MODEL_TRAINING.value, datetime.now()),
            training_end=datetime.now(),
            validation_metrics={**training_metrics, **validation_metrics},
            notes=f"Auto-trained by RetrainingPipeline"
        )
        
        logger.info(f"Model deployed: {model_id}")
        return model_id
    
    def get_run_summary(self, run_id: Optional[str] = None) -> Dict:
        """Get summary of a pipeline run."""
        if run_id:
            for result in self.run_history:
                if result.run_id == run_id:
                    return result.to_dict()
            return {}
        
        # Return latest run
        if self.run_history:
            return self.run_history[-1].to_dict()
        return {}
    
    def get_pipeline_stats(self) -> Dict:
        """Get overall pipeline statistics."""
        if not self.run_history:
            return {'total_runs': 0}
        
        successful = [r for r in self.run_history if r.success]
        failed = [r for r in self.run_history if not r.success]
        
        return {
            'total_runs': len(self.run_history),
            'successful_runs': len(successful),
            'failed_runs': len(failed),
            'success_rate': len(successful) / len(self.run_history) if self.run_history else 0,
            'avg_duration_seconds': np.mean([
                (r.completed_at - r.started_at).total_seconds()
                for r in self.run_history if r.completed_at
            ]) if self.run_history else 0,
            'last_run': self.run_history[-1].run_id if self.run_history else None,
            'last_success': successful[-1].run_id if successful else None
        }
