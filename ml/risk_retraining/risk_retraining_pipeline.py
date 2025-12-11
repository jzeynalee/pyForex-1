# ml/risk_retraining/risk_retraining_pipeline.py
"""
Retraining Pipeline for Risk Management Models.

Handles end-to-end retraining for:
- TCN Risk Model (direction + volatility + quantiles)
- GBM Meta-Labeling Model
- RL Exit Optimizer

Features:
- Multi-task loss training for TCN
- Feature engineering for meta-labeling
- Champion/challenger model comparison
- Automatic rollback on degradation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import shutil
import pickle

from .risk_retraining_config import (
    RiskModelType, RiskRetrainingConfig, RetrainingTriggerType
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Stages
# =============================================================================

class PipelineStage(Enum):
    """Stages of the retraining pipeline."""
    INITIALIZED = "initialized"
    DATA_COLLECTION = "data_collection"
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    model_type: RiskModelType
    trigger: RetrainingTriggerType
    success: bool
    stage_reached: PipelineStage
    duration_seconds: float
    metrics: Dict[str, float]
    champion_metrics: Optional[Dict[str, float]]
    challenger_metrics: Optional[Dict[str, float]]
    model_path: Optional[str]
    error: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type.name,
            'trigger': self.trigger.name,
            'success': self.success,
            'stage_reached': self.stage_reached.value,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics,
            'champion_metrics': self.champion_metrics,
            'challenger_metrics': self.challenger_metrics,
            'model_path': self.model_path,
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
        }


# =============================================================================
# Data Providers
# =============================================================================

class RiskDataProvider:
    """
    Provides data for risk model retraining.
    
    Handles data loading, splitting, and preprocessing.
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        validation_ratio: float = 0.2,
    ):
        self.data_dir = Path(data_dir)
        self.validation_ratio = validation_ratio
    
    def load_tcn_data(
        self,
        symbols: List[str],
        lookback_days: int = 90,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for TCN training."""
        all_data = []
        
        for symbol in symbols:
            data_path = self.data_dir / f"{symbol.lower()}_features.parquet"
            if data_path.exists():
                df = pd.read_parquet(data_path)
                df['symbol'] = symbol
                all_data.append(df)
            else:
                logger.warning(f"Data file not found: {data_path}")
        
        if not all_data:
            raise ValueError("No data files found")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Filter to recent data
        if 'timestamp' in combined.columns:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            combined = combined[combined['timestamp'] >= cutoff]
        
        # Split train/validation
        split_idx = int(len(combined) * (1 - self.validation_ratio))
        train_df = combined.iloc[:split_idx]
        val_df = combined.iloc[split_idx:]
        
        return train_df, val_df
    
    def load_meta_labeling_data(
        self,
        symbols: List[str],
        tcn_predictions_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data for GBM meta-labeling training."""
        # Load TCN predictions
        tcn_preds = pd.read_parquet(tcn_predictions_path)
        
        # Load trade outcomes
        outcomes_path = self.data_dir / "trade_outcomes.parquet"
        if outcomes_path.exists():
            outcomes = pd.read_parquet(outcomes_path)
        else:
            raise ValueError("Trade outcomes file not found")
        
        # Merge predictions with outcomes
        merged = tcn_preds.merge(
            outcomes,
            on=['timestamp', 'symbol'],
            how='inner'
        )
        
        # Create meta-labels (was the TCN prediction correct?)
        merged['meta_label'] = (
            (merged['tcn_prediction'] == merged['actual_direction']) &
            (merged['trade_profitable'] == True)
        ).astype(int)
        
        # Split
        split_idx = int(len(merged) * (1 - self.validation_ratio))
        train_df = merged.iloc[:split_idx]
        val_df = merged.iloc[split_idx:]
        
        return train_df, val_df
    
    def load_rl_data(
        self,
        symbols: List[str],
        episode_path: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Load episode data for RL training."""
        with open(episode_path, 'rb') as f:
            episodes = pickle.load(f)
        
        # Split episodes
        split_idx = int(len(episodes) * (1 - self.validation_ratio))
        train_episodes = episodes[:split_idx]
        val_episodes = episodes[split_idx:]
        
        return train_episodes, val_episodes


# =============================================================================
# TCN Training Pipeline
# =============================================================================

class TCNRiskTrainingPipeline:
    """Training pipeline for TCN Risk Model."""
    
    def __init__(
        self,
        config: RiskRetrainingConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.device = device
        self.data_provider = RiskDataProvider(
            data_dir=config.data_dir,
            validation_ratio=config.validation_holdout_ratio,
        )
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create TCN Risk Model instance."""
        # Import here to avoid circular dependency
        try:
            from models.tcn_risk import TCNRiskModel, RiskModelConfig
            
            model_config = RiskModelConfig(
                input_dim=input_dim,
                profile=self.config.profile_name,
            )
            return TCNRiskModel(model_config)
        except ImportError:
            logger.warning("TCNRiskModel not found, using placeholder")
            return self._create_placeholder_model(input_dim)
    
    def _create_placeholder_model(self, input_dim: int) -> nn.Module:
        """Create placeholder model for testing."""
        class PlaceholderTCN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = nn.Linear(input_dim, 64)
                self.direction = nn.Linear(64, 3)
                self.volatility = nn.Linear(64, 1)
                self.quantiles = nn.Linear(64, 5)
            
            def forward(self, x):
                h = torch.relu(self.fc(x[:, -1, :]))
                return {
                    'direction': self.direction(h),
                    'volatility': self.volatility(h),
                    'quantiles': self.quantiles(h),
                }
        
        return PlaceholderTCN(input_dim)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare data for TCN training."""
        # Extract feature columns (exclude targets and metadata)
        exclude_cols = ['timestamp', 'symbol', 'direction_target', 'volatility_target',
                       'quantile_targets', 'returns', 'close', 'open', 'high', 'low']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Create sequences
        X_list = []
        y_direction = []
        y_volatility = []
        y_quantiles = []
        
        for i in range(sequence_length, len(df)):
            X_list.append(df[feature_cols].iloc[i-sequence_length:i].values)
            
            if 'direction_target' in df.columns:
                y_direction.append(df['direction_target'].iloc[i])
            if 'volatility_target' in df.columns:
                y_volatility.append(df['volatility_target'].iloc[i])
            if 'quantile_targets' in df.columns:
                y_quantiles.append(df['quantile_targets'].iloc[i])
        
        X = torch.FloatTensor(np.array(X_list))
        
        targets = {}
        if y_direction:
            targets['direction'] = torch.LongTensor(y_direction)
        if y_volatility:
            targets['volatility'] = torch.FloatTensor(y_volatility)
        if y_quantiles:
            targets['quantiles'] = torch.FloatTensor(y_quantiles)
        
        return X, targets
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Dict[str, nn.Module],
        loss_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        epoch_losses = {k: 0.0 for k in criterion.keys()}
        epoch_losses['total'] = 0.0
        
        for batch_X, *batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            total_loss = 0.0
            
            # Direction loss
            if 'direction' in outputs and len(batch_y) > 0:
                direction_loss = criterion['direction'](
                    outputs['direction'],
                    batch_y[0].to(self.device)
                )
                total_loss += loss_weights.get('direction', 1.0) * direction_loss
                epoch_losses['direction'] += direction_loss.item()
            
            # Volatility loss
            if 'volatility' in outputs and len(batch_y) > 1:
                vol_loss = criterion['volatility'](
                    outputs['volatility'].squeeze(),
                    batch_y[1].to(self.device)
                )
                total_loss += loss_weights.get('volatility', 1.0) * vol_loss
                epoch_losses['volatility'] += vol_loss.item()
            
            # Quantile loss (pinball)
            if 'quantiles' in outputs and len(batch_y) > 2:
                quantile_loss = self._pinball_loss(
                    outputs['quantiles'],
                    batch_y[2].to(self.device),
                    [0.05, 0.25, 0.5, 0.75, 0.95]
                )
                total_loss += loss_weights.get('quantiles', 1.0) * quantile_loss
                epoch_losses['quantiles'] += quantile_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
        
        # Average losses
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def _pinball_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantiles: List[float],
    ) -> torch.Tensor:
        """Calculate pinball loss for quantile regression."""
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.mean(torch.max(q * errors, (q - 1) * errors)))
        return torch.stack(losses).mean()
    
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate model and compute metrics."""
        model.eval()
        
        all_direction_preds = []
        all_direction_targets = []
        all_vol_preds = []
        all_vol_targets = []
        all_quantile_preds = []
        all_quantile_targets = []
        
        with torch.no_grad():
            for batch_X, *batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                
                if 'direction' in outputs and len(batch_y) > 0:
                    all_direction_preds.append(outputs['direction'].cpu().numpy())
                    all_direction_targets.append(batch_y[0].numpy())
                
                if 'volatility' in outputs and len(batch_y) > 1:
                    all_vol_preds.append(outputs['volatility'].squeeze().cpu().numpy())
                    all_vol_targets.append(batch_y[1].numpy())
                
                if 'quantiles' in outputs and len(batch_y) > 2:
                    all_quantile_preds.append(outputs['quantiles'].cpu().numpy())
                    all_quantile_targets.append(batch_y[2].numpy())
        
        metrics = {}
        
        # Direction metrics
        if all_direction_preds:
            preds = np.concatenate(all_direction_preds)
            targets = np.concatenate(all_direction_targets)
            metrics['direction_accuracy'] = np.mean(np.argmax(preds, axis=1) == targets)
        
        # Volatility metrics
        if all_vol_preds:
            preds = np.concatenate(all_vol_preds)
            targets = np.concatenate(all_vol_targets)
            metrics['volatility_mae'] = np.mean(np.abs(preds - targets))
            metrics['volatility_correlation'] = np.corrcoef(preds, targets)[0, 1]
        
        # Quantile metrics
        if all_quantile_preds:
            preds = np.concatenate(all_quantile_preds)
            targets = np.concatenate(all_quantile_targets)
            metrics['quantile_pinball'] = self._calculate_pinball_numpy(
                preds, targets[:, 2], [0.05, 0.25, 0.5, 0.75, 0.95]  # Assuming Q50 target
            )
        
        return metrics
    
    def _calculate_pinball_numpy(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        quantiles: List[float],
    ) -> float:
        """Calculate pinball loss in numpy."""
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - preds[:, i]
            loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
            losses.append(loss)
        return np.mean(losses)
    
    def run(
        self,
        trigger: RetrainingTriggerType,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
    ) -> PipelineResult:
        """Run the full TCN training pipeline."""
        start_time = datetime.now()
        stage = PipelineStage.INITIALIZED
        
        try:
            # Data collection
            stage = PipelineStage.DATA_COLLECTION
            train_df, val_df = self.data_provider.load_tcn_data(
                symbols=self.config.symbols,
            )
            logger.info(f"Loaded {len(train_df)} training, {len(val_df)} validation samples")
            
            # Data preparation
            stage = PipelineStage.DATA_PREPARATION
            X_train, y_train = self.prepare_data(train_df)
            X_val, y_val = self.prepare_data(val_df)
            
            # Create datasets
            train_dataset = TensorDataset(
                X_train,
                y_train.get('direction', torch.zeros(len(X_train))),
                y_train.get('volatility', torch.zeros(len(X_train))),
                y_train.get('quantiles', torch.zeros(len(X_train), 5)),
            )
            val_dataset = TensorDataset(
                X_val,
                y_val.get('direction', torch.zeros(len(X_val))),
                y_val.get('volatility', torch.zeros(len(X_val))),
                y_val.get('quantiles', torch.zeros(len(X_val), 5)),
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training
            stage = PipelineStage.TRAINING
            input_dim = X_train.shape[-1]
            model = self.create_model(input_dim).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = {
                'direction': nn.CrossEntropyLoss(),
                'volatility': nn.MSELoss(),
            }
            loss_weights = {'direction': 1.0, 'volatility': 0.5, 'quantiles': 0.5}
            
            best_metrics = None
            best_model_state = None
            
            for epoch in range(epochs):
                train_losses = self.train_epoch(
                    model, train_loader, optimizer, criterion, loss_weights
                )
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: {train_losses}")
            
            # Validation
            stage = PipelineStage.VALIDATION
            challenger_metrics = self.validate(model, val_loader)
            logger.info(f"Challenger metrics: {challenger_metrics}")
            
            # Compare with champion (if exists)
            stage = PipelineStage.COMPARISON
            champion_metrics = None
            model_path = Path(self.config.models_dir) / "tcn_risk_best.pt"
            
            if model_path.exists() and self.config.champion_challenger_enabled:
                # Load and validate champion
                champion_state = torch.load(model_path, map_location=self.device)
                champion_model = self.create_model(input_dim).to(self.device)
                champion_model.load_state_dict(champion_state['model_state_dict'])
                champion_metrics = self.validate(champion_model, val_loader)
                
                # Compare
                challenger_better = self._is_challenger_better(
                    champion_metrics, challenger_metrics
                )
                
                if not challenger_better:
                    logger.info("Champion model is better, keeping current model")
                    return PipelineResult(
                        model_type=RiskModelType.TCN_RISK,
                        trigger=trigger,
                        success=True,
                        stage_reached=PipelineStage.COMPARISON,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        metrics=challenger_metrics,
                        champion_metrics=champion_metrics,
                        challenger_metrics=challenger_metrics,
                        model_path=str(model_path),
                        error=None,
                        timestamp=datetime.now(),
                    )
            
            # Deployment
            stage = PipelineStage.DEPLOYMENT
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config.__dict__,
                'metrics': challenger_metrics,
                'timestamp': datetime.now().isoformat(),
                'input_dim': input_dim,
            }, model_path)
            
            logger.info(f"Saved new model to {model_path}")
            
            stage = PipelineStage.COMPLETED
            return PipelineResult(
                model_type=RiskModelType.TCN_RISK,
                trigger=trigger,
                success=True,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics=challenger_metrics,
                champion_metrics=champion_metrics,
                challenger_metrics=challenger_metrics,
                model_path=str(model_path),
                error=None,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed at {stage}: {e}")
            return PipelineResult(
                model_type=RiskModelType.TCN_RISK,
                trigger=trigger,
                success=False,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics={},
                champion_metrics=None,
                challenger_metrics=None,
                model_path=None,
                error=str(e),
                timestamp=datetime.now(),
            )
    
    def _is_challenger_better(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
    ) -> bool:
        """Determine if challenger model is better than champion."""
        # Score based on key metrics
        champion_score = 0
        challenger_score = 0
        
        # Direction accuracy (higher is better)
        if 'direction_accuracy' in champion_metrics:
            if challenger_metrics.get('direction_accuracy', 0) > champion_metrics['direction_accuracy']:
                challenger_score += 2
            else:
                champion_score += 2
        
        # Volatility MAE (lower is better)
        if 'volatility_mae' in champion_metrics:
            if challenger_metrics.get('volatility_mae', float('inf')) < champion_metrics['volatility_mae']:
                challenger_score += 1
            else:
                champion_score += 1
        
        # Volatility correlation (higher is better)
        if 'volatility_correlation' in champion_metrics:
            if challenger_metrics.get('volatility_correlation', 0) > champion_metrics['volatility_correlation']:
                challenger_score += 1
            else:
                champion_score += 1
        
        return challenger_score > champion_score


# =============================================================================
# GBM Meta-Labeling Pipeline
# =============================================================================

class GBMMetaTrainingPipeline:
    """Training pipeline for GBM Meta-Labeling Model."""
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        self.data_provider = RiskDataProvider(
            data_dir=config.data_dir,
            validation_ratio=config.validation_holdout_ratio,
        )
    
    def create_model(self):
        """Create GBM model."""
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
    
    def prepare_meta_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for meta-labeling."""
        # Meta-features: TCN outputs + confidence + market features
        meta_feature_cols = [
            'tcn_direction_prob',
            'tcn_confidence',
            'tcn_volatility_pred',
            'volatility_regime',
            'trend_strength',
            'volume_ratio',
            'spread_pips',
            'hour_of_day',
            'day_of_week',
        ]
        
        available_cols = [c for c in meta_feature_cols if c in df.columns]
        
        X = df[available_cols].values
        y = df['meta_label'].values if 'meta_label' in df.columns else np.zeros(len(df))
        
        return X, y
    
    def validate(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        trade_results: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Validate GBM model."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        preds = model.predict_proba(X_val)[:, 1]
        pred_binary = (preds >= 0.5).astype(int)
        
        metrics = {
            'precision': precision_score(y_val, pred_binary, zero_division=0),
            'recall': recall_score(y_val, pred_binary, zero_division=0),
            'f1': f1_score(y_val, pred_binary, zero_division=0),
            'filter_rate': np.mean(preds < 0.5),
        }
        
        if trade_results is not None:
            mask = preds >= 0.5
            if mask.sum() > 0:
                metrics['filtered_win_rate'] = np.mean(trade_results[mask] > 0)
                baseline_win_rate = np.mean(trade_results > 0)
                metrics['filter_improvement'] = metrics['filtered_win_rate'] - baseline_win_rate
        
        return metrics
    
    def run(
        self,
        trigger: RetrainingTriggerType,
        tcn_predictions_path: str,
    ) -> PipelineResult:
        """Run GBM meta-labeling training pipeline."""
        start_time = datetime.now()
        stage = PipelineStage.INITIALIZED
        
        try:
            # Data collection
            stage = PipelineStage.DATA_COLLECTION
            train_df, val_df = self.data_provider.load_meta_labeling_data(
                symbols=self.config.symbols,
                tcn_predictions_path=tcn_predictions_path,
            )
            
            # Feature engineering
            stage = PipelineStage.FEATURE_ENGINEERING
            X_train, y_train = self.prepare_meta_features(train_df)
            X_val, y_val = self.prepare_meta_features(val_df)
            
            trade_results = val_df['trade_pnl'].values if 'trade_pnl' in val_df.columns else None
            
            # Training
            stage = PipelineStage.TRAINING
            model = self.create_model()
            model.fit(X_train, y_train)
            
            # Validation
            stage = PipelineStage.VALIDATION
            challenger_metrics = self.validate(model, X_val, y_val, trade_results)
            logger.info(f"GBM challenger metrics: {challenger_metrics}")
            
            # Comparison with champion
            stage = PipelineStage.COMPARISON
            model_path = Path(self.config.models_dir) / "gbm_meta_best.pkl"
            champion_metrics = None
            
            if model_path.exists() and self.config.champion_challenger_enabled:
                with open(model_path, 'rb') as f:
                    champion_data = pickle.load(f)
                champion_model = champion_data['model']
                champion_metrics = self.validate(champion_model, X_val, y_val, trade_results)
                
                if not self._is_challenger_better(champion_metrics, challenger_metrics):
                    logger.info("Champion GBM is better")
                    return PipelineResult(
                        model_type=RiskModelType.GBM_META,
                        trigger=trigger,
                        success=True,
                        stage_reached=stage,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        metrics=challenger_metrics,
                        champion_metrics=champion_metrics,
                        challenger_metrics=challenger_metrics,
                        model_path=str(model_path),
                        error=None,
                        timestamp=datetime.now(),
                    )
            
            # Deployment
            stage = PipelineStage.DEPLOYMENT
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metrics': challenger_metrics,
                    'timestamp': datetime.now().isoformat(),
                }, f)
            
            stage = PipelineStage.COMPLETED
            return PipelineResult(
                model_type=RiskModelType.GBM_META,
                trigger=trigger,
                success=True,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics=challenger_metrics,
                champion_metrics=champion_metrics,
                challenger_metrics=challenger_metrics,
                model_path=str(model_path),
                error=None,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"GBM pipeline failed at {stage}: {e}")
            return PipelineResult(
                model_type=RiskModelType.GBM_META,
                trigger=trigger,
                success=False,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics={},
                champion_metrics=None,
                challenger_metrics=None,
                model_path=None,
                error=str(e),
                timestamp=datetime.now(),
            )
    
    def _is_challenger_better(
        self,
        champion: Dict[str, float],
        challenger: Dict[str, float],
    ) -> bool:
        """Check if challenger is better."""
        score = 0
        
        if challenger.get('precision', 0) > champion.get('precision', 0):
            score += 2
        if challenger.get('f1', 0) > champion.get('f1', 0):
            score += 1
        if challenger.get('filter_improvement', 0) > champion.get('filter_improvement', 0):
            score += 2
        
        return score >= 3


# =============================================================================
# RL Exit Pipeline
# =============================================================================

class RLExitTrainingPipeline:
    """Training pipeline for RL Exit Optimizer."""
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
    
    def run(
        self,
        trigger: RetrainingTriggerType,
        env_config: Optional[Dict] = None,
    ) -> PipelineResult:
        """Run RL exit optimizer training pipeline."""
        start_time = datetime.now()
        stage = PipelineStage.INITIALIZED
        
        try:
            # Try to import stable_baselines3
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv
            except ImportError:
                logger.warning("stable_baselines3 not available, skipping RL training")
                return PipelineResult(
                    model_type=RiskModelType.RL_EXIT,
                    trigger=trigger,
                    success=False,
                    stage_reached=stage,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    metrics={},
                    champion_metrics=None,
                    challenger_metrics=None,
                    model_path=None,
                    error="stable_baselines3 not installed",
                    timestamp=datetime.now(),
                )
            
            stage = PipelineStage.TRAINING
            
            # Create environment
            try:
                from ml.rl.exit_environment import TradingExitEnv
                env = DummyVecEnv([lambda: TradingExitEnv(env_config or {})])
            except ImportError:
                logger.warning("TradingExitEnv not found")
                return PipelineResult(
                    model_type=RiskModelType.RL_EXIT,
                    trigger=trigger,
                    success=False,
                    stage_reached=stage,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    metrics={},
                    champion_metrics=None,
                    challenger_metrics=None,
                    model_path=None,
                    error="TradingExitEnv not implemented",
                    timestamp=datetime.now(),
                )
            
            # Train PPO
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1,
            )
            
            model.learn(total_timesteps=100000)
            
            # Validation
            stage = PipelineStage.VALIDATION
            challenger_metrics = self._evaluate_policy(model, env)
            
            # Deployment
            stage = PipelineStage.DEPLOYMENT
            model_path = Path(self.config.models_dir) / "rl_exit_best.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            
            stage = PipelineStage.COMPLETED
            return PipelineResult(
                model_type=RiskModelType.RL_EXIT,
                trigger=trigger,
                success=True,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics=challenger_metrics,
                champion_metrics=None,
                challenger_metrics=challenger_metrics,
                model_path=str(model_path),
                error=None,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"RL pipeline failed: {e}")
            return PipelineResult(
                model_type=RiskModelType.RL_EXIT,
                trigger=trigger,
                success=False,
                stage_reached=stage,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics={},
                champion_metrics=None,
                challenger_metrics=None,
                model_path=None,
                error=str(e),
                timestamp=datetime.now(),
            )
    
    def _evaluate_policy(self, model, env, n_episodes: int = 100) -> Dict[str, float]:
        """Evaluate trained policy."""
        rewards = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
            
            rewards.append(episode_reward)
        
        return {
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
        }


# =============================================================================
# Unified Pipeline Manager
# =============================================================================

class RiskRetrainingPipelineManager:
    """
    Manages all risk model retraining pipelines.
    
    Handles:
    - Pipeline selection based on model type
    - Dependency chain execution
    - Rollback coordination
    """
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        
        self.tcn_pipeline = TCNRiskTrainingPipeline(config)
        self.gbm_pipeline = GBMMetaTrainingPipeline(config)
        self.rl_pipeline = RLExitTrainingPipeline(config)
        
        self.pipeline_history: List[PipelineResult] = []
    
    def run_pipeline(
        self,
        model_type: RiskModelType,
        trigger: RetrainingTriggerType,
        **kwargs
    ) -> PipelineResult:
        """Run pipeline for a specific model type."""
        if model_type == RiskModelType.TCN_RISK:
            result = self.tcn_pipeline.run(trigger, **kwargs)
        elif model_type == RiskModelType.GBM_META:
            result = self.gbm_pipeline.run(trigger, **kwargs)
        elif model_type == RiskModelType.RL_EXIT:
            result = self.rl_pipeline.run(trigger, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.pipeline_history.append(result)
        return result
    
    def run_with_dependencies(
        self,
        model_type: RiskModelType,
        trigger: RetrainingTriggerType,
        **kwargs
    ) -> List[PipelineResult]:
        """Run pipeline with dependent models."""
        results = []
        
        # Run primary model
        primary_result = self.run_pipeline(model_type, trigger, **kwargs)
        results.append(primary_result)
        
        if not primary_result.success:
            return results
        
        # Check dependencies
        deps = self.config.dependencies
        
        if model_type == RiskModelType.TCN_RISK and deps.tcn_triggers_gbm:
            logger.info("TCN retrained, triggering GBM retraining")
            gbm_result = self.run_pipeline(
                RiskModelType.GBM_META,
                RetrainingTriggerType.DEPENDENCY,
                tcn_predictions_path=primary_result.model_path,
            )
            results.append(gbm_result)
            
            if deps.tcn_triggers_rl and gbm_result.success:
                logger.info("Triggering RL retraining")
                rl_result = self.run_pipeline(
                    RiskModelType.RL_EXIT,
                    RetrainingTriggerType.DEPENDENCY,
                )
                results.append(rl_result)
        
        return results
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of recent pipeline runs."""
        recent = self.pipeline_history[-20:]
        
        return {
            'total_runs': len(self.pipeline_history),
            'recent_runs': len(recent),
            'success_rate': sum(1 for r in recent if r.success) / len(recent) if recent else 0,
            'by_model': {
                model_type.name: [r.to_dict() for r in recent if r.model_type == model_type]
                for model_type in RiskModelType
            },
        }