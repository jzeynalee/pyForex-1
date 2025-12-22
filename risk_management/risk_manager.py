"""
Risk Management System - Integration Module

This module provides the unified RiskManager class that orchestrates
all three phases of the risk management pipeline:

Phase 1: Predictive Foundation (Multi-Head TCN)
    → Direction, Volatility, Quantile predictions
    → Trade outcome probabilities (p_long/p_short): TP-before-SL likelihood

Phase 2: Risk Calculations
    → SL/TP levels, Position sizing, Hard rules enforcement

Phase 3: Trade Filtering  
    → Triple barrier labeling, Meta-labeling for signal filtering

Usage:
    manager = RiskManager.create_for_profile('INTRADAY', input_features=64)
    
    # Training
    manager.train_predictive_model(train_data)
    manager.train_meta_labeler(train_data)
    
    # Inference
    decision = manager.evaluate_trade_opportunity(
        features=current_features,
        entry_price=1.1234,
        pair='EURUSD',
        account_balance=10000,
        current_spread=1.5
    )
    
    if decision['should_trade']:
        print(f"Position: {decision['position_size']} lots")
        print(f"SL: {decision['stop_loss']}, TP: {decision['take_profit']}")
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

# Phase 1 imports
from .phase1_predictive import (
    TCNConfig, TradingProfile, MultiHeadTCN, RiskPrediction,
    create_tcn_for_profile, TrainingConfig, RiskDataset, MultiHeadTCNTrainer
)

# Phase 2 imports
from .phase2_risk_calc import (
    SLTPConfig, SLTPCalculator, SLTPResult, MarketRegime, TradeDirection,
    PositionSizingConfig, PositionSizingCalculator, PositionSizeResult,
    HardRulesConfig, HardRulesEngine, TradeGatekeeper,
    calculate_sl_tp_from_predictions, calculate_position_from_predictions
)

# Phase 3 imports
from .phase3_filtering import (
    TripleBarrierConfig, TripleBarrierLabeler,
    MetaLabelingConfig, MetaLabelingModel, TradeFilter
)

# Utils
from .utils import (
    RegimeDetector, calculate_atr, normalize_features,
    create_direction_labels, create_volatility_labels, create_price_move_labels,
    generate_performance_report
)

logger = logging.getLogger(__name__)


@dataclass
class RiskManagerConfig:
    """Master configuration for the Risk Management System."""
    # Profile
    profile: str = 'INTRADAY'
    
    # Model configuration
    input_features: int = 64
    sequence_length: int = 60
    vision_features: Optional[int] = None
    
    # Phase 1: TCN Config
    tcn_hidden_channels: int = 128
    tcn_dropout: float = 0.2
    
    # Phase 2: Risk Calc Config
    base_risk_percent: float = 1.0
    min_risk_reward: float = 1.5
    max_leverage: float = 10.0
    
    # Phase 3: Filtering Config
    meta_labeling_threshold: float = 0.5
    min_direction_confidence: float = 0.5
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    model_dir: str = './models/risk_management'


@dataclass
class TradeDecision:
    """Complete trade decision from the risk management system."""
    # Decision
    should_trade: bool
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Direction
    direction: str = ''  # 'BUY', 'SELL', or ''
    direction_confidence: float = 0.0
    direction_probs: Dict[str, float] = field(default_factory=dict)
    
    # Risk parameters
    stop_loss: float = 0.0
    take_profit: float = 0.0
    sl_distance_pips: float = 0.0
    tp_distance_pips: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Position
    position_size: float = 0.0
    position_units: int = 0
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    
    # Meta-labeling
    meta_score: float = 0.0
    
    # Market context
    regime: str = ''
    volatility: float = 0.0
    
    # Validation
    rule_violations: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'should_trade': self.should_trade,
            'rejection_reasons': self.rejection_reasons,
            'direction': self.direction,
            'direction_confidence': self.direction_confidence,
            'direction_probs': self.direction_probs,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_distance_pips': self.sl_distance_pips,
            'tp_distance_pips': self.tp_distance_pips,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'position_units': self.position_units,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'meta_score': self.meta_score,
            'regime': self.regime,
            'volatility': self.volatility,
            'rule_violations': self.rule_violations
        }


class RiskManager:
    """
    Unified Risk Management System.
    
    Orchestrates the complete risk management pipeline:
    1. Get predictions from multi-head TCN
    2. Calculate SL/TP and position size
    3. Apply hard rules and filters
    4. Return final trade decision
    """
    
    def __init__(self, config: RiskManagerConfig):
        """
        Initialize Risk Manager.
        
        Args:
            config: Master configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize Phase 1: Predictive Model
        self._init_predictive_model()
        
        # Initialize Phase 2: Risk Calculators
        self._init_risk_calculators()
        
        # Initialize Phase 3: Filtering
        self._init_filtering()
        
        # Utilities
        self.regime_detector = RegimeDetector()
        
        # State tracking
        self._is_trained = False
        self._meta_model_trained = False
        
        logger.info(f"RiskManager initialized for {config.profile} profile")
    
    def _init_predictive_model(self):
        """Initialize Phase 1 multi-head TCN."""
        self.tcn_model = create_tcn_for_profile(
            profile=self.config.profile,
            input_features=self.config.input_features,
            vision_features=self.config.vision_features
        ).to(self.device)
    
    def _init_risk_calculators(self):
        """Initialize Phase 2 calculators."""
        # SL/TP Calculator
        self.sltp_config = SLTPConfig(
            min_risk_reward=self.config.min_risk_reward
        )
        self.sltp_calculator = SLTPCalculator(self.sltp_config)
        
        # Position Sizing
        self.position_config = PositionSizingConfig(
            base_risk_percent=self.config.base_risk_percent
        )
        self.position_calculator = PositionSizingCalculator(self.position_config)
        
        # Hard Rules
        self.rules_config = HardRulesConfig(
            max_leverage_default=self.config.max_leverage
        )
        self.rules_engine = HardRulesEngine(self.rules_config)
        self.gatekeeper = TradeGatekeeper(self.rules_config)
    
    def _init_filtering(self):
        """Initialize Phase 3 filtering components."""
        # Triple Barrier Labeler
        self.barrier_config = TripleBarrierConfig()
        self.barrier_labeler = TripleBarrierLabeler(self.barrier_config)
        
        # Meta-Labeling Model
        self.meta_config = MetaLabelingConfig(
            default_threshold=self.config.meta_labeling_threshold
        )
        self.meta_model = MetaLabelingModel(self.meta_config)
        self.trade_filter: Optional[TradeFilter] = None
    
    @classmethod
    def create_for_profile(
        cls,
        profile: str,
        input_features: int = 64,
        **kwargs
    ) -> 'RiskManager':
        """
        Factory method to create RiskManager for a trading profile.
        
        Args:
            profile: 'SCALP', 'INTRADAY', or 'SWING'
            input_features: Number of input features
            **kwargs: Additional config overrides
        
        Returns:
            Configured RiskManager instance
        """
        config = RiskManagerConfig(
            profile=profile.upper(),
            input_features=input_features,
            **kwargs
        )
        return cls(config)
    
    # =========================================================================
    # Training Methods
    # =========================================================================
    
    def train_predictive_model(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        horizon: int = 1,
        validation_split: float = 0.2,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, List[float]]:
        """Train the Phase 1 multi-head TCN model.

        In addition to direction/volatility/quantiles, this training step can also
        supervise the trade-objective probability heads:
        - p_long  ≈ P(TP hits before SL | enter long now)
        - p_short ≈ P(TP hits before SL | enter short now)

        Outcome targets are generated via leakage-safe ATR-based triple-barrier
        labeling (no model-derived SL/TP).
        
        Args:
            features: (n_samples, n_features) feature matrix
            prices: DataFrame with 'high', 'low', 'close' columns
            horizon: Prediction horizon in candles
            validation_split: Fraction for validation
            training_config: Optional training configuration
        
        Returns:
            Training history
        """
        logger.info("Training Phase 1: Multi-Head TCN...")
        
        # Create labels
        direction_labels = create_direction_labels(
            prices['close'].values, horizon=horizon, threshold=0.0001
        )
        volatility_labels = create_volatility_labels(
            prices['high'].values, prices['low'].values,
            prices['close'].values, horizon=horizon
        )
        price_move_labels = create_price_move_labels(
            prices['close'].values, horizon=horizon
        )

        # Leakage-safe dense outcome labels for p_long/p_short training
        atr = calculate_atr(
            prices['high'].values,
            prices['low'].values,
            prices['close'].values,
            period=14
        )
        outcome_labels = self.barrier_labeler.generate_outcome_labels(
            prices=prices,
            atr=atr,
            profile=self.config.profile
        )
        
        # Normalize features
        features_norm, self._norm_params = normalize_features(features)
        
        # Create dataset
        dataset = RiskDataset(
            features=features_norm,
            direction_labels=direction_labels,
            volatility_labels=volatility_labels,
            price_move_labels=price_move_labels,
            sequence_length=self.config.sequence_length,
            outcome_labels=outcome_labels
        )
        
        # Split
        split_idx = int(len(dataset) * (1 - validation_split))
        train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
        val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
        
        # Create loaders
        train_config = training_config or TrainingConfig()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config.batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=train_config.batch_size, shuffle=False
        )
        
        # Train
        trainer = MultiHeadTCNTrainer(
            self.tcn_model, train_config, device=self.config.device
        )
        history = trainer.train(train_loader, val_loader)
        
        self._is_trained = True
        logger.info("Phase 1 training complete")
        
        return history
    
    def train_meta_labeler(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, float]:
        """
        Train the Phase 3 meta-labeling model.
        
        Requires Phase 1 model to be trained first.
        
        Args:
            features: Feature matrix
            prices: OHLC prices
            market_data: Optional market condition data
            timestamps: Optional datetime index
        
        Returns:
            Training metrics
        """
        if not self._is_trained:
            raise ValueError("Must train predictive model first")
        
        logger.info("Training Phase 3: Meta-Labeling GBM...")
        
        # Generate predictions from Phase 1
        predictions = self._generate_batch_predictions(features)
        
        # Generate triple barrier labels
        labels, detailed = self.barrier_labeler.label_with_model_predictions(
            prices=prices,
            model_predictions=predictions,
            confidence_threshold=self.config.min_direction_confidence,
            profile=self.config.profile
        )
        
        # Create meta-labels
        pred_directions = np.argmax(predictions['direction_probs'], axis=1)
        directions = np.where(pred_directions == 2, 1, np.where(pred_directions == 0, -1, 0))
        meta_labels = self.meta_model.create_meta_labels(directions, labels)
        
        # Extract meta-features
        meta_features = self.meta_model.feature_extractor.extract_features(
            primary_predictions=predictions,
            market_data=market_data,
            timestamps=timestamps
        )
        
        # Filter to only entries with valid labels
        valid_mask = np.abs(directions) == 1  # Only actual trades
        if valid_mask.sum() == 0:
            logger.warning("No valid trade signals for meta-labeling training")
            return {}
        
        X_meta = meta_features[valid_mask]
        y_meta = meta_labels[valid_mask]
        
        # Train
        metrics = self.meta_model.train(X_meta, y_meta)
        
        # Create trade filter
        self.trade_filter = TradeFilter(
            meta_model=self.meta_model,
            min_confidence=self.config.min_direction_confidence,
            min_meta_score=self.config.meta_labeling_threshold
        )
        
        self._meta_model_trained = True
        logger.info("Phase 3 training complete")
        
        return metrics
    
    def _generate_batch_predictions(
        self,
        features: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """Generate predictions for entire feature matrix.

        Returns a dict compatible with Phase 3 feature extraction and labeling.
        Includes p_long/p_short outputs if the underlying model provides them.
        """
        self.tcn_model.eval()
        
        # Normalize
        features_norm, _ = normalize_features(features)
        
        n_samples = len(features_norm)
        all_preds = {
            'direction_probs': [],
            'volatility': [],
            'quantiles': [],
            'p_long': [],
            'p_short': []
        }
        
        with torch.no_grad():
            for start in range(0, n_samples - self.config.sequence_length, batch_size):
                end = min(start + batch_size, n_samples - self.config.sequence_length)
                
                batch_seqs = []
                for i in range(start, end):
                    seq = features_norm[i:i + self.config.sequence_length]
                    batch_seqs.append(seq)
                
                if not batch_seqs:
                    continue
                
                batch_tensor = torch.tensor(
                    np.array(batch_seqs), dtype=torch.float32
                ).to(self.device)
                
                output = self.tcn_model(batch_tensor, mode='all')
                
                all_preds['direction_probs'].append(output['direction'].cpu().numpy())
                all_preds['volatility'].append(output['volatility'].cpu().numpy())
                all_preds['quantiles'].append(output['quantiles'].cpu().numpy())

                if 'p_long' in output:
                    all_preds['p_long'].append(output['p_long'].cpu().numpy())
                if 'p_short' in output:
                    all_preds['p_short'].append(output['p_short'].cpu().numpy())
        
        # Concatenate and pad
        for key in all_preds:
            if all_preds[key]:
                arr = np.concatenate(all_preds[key], axis=0)
                # Pad beginning
                pad_shape = (self.config.sequence_length,) + arr.shape[1:]
                all_preds[key] = np.concatenate([
                    np.zeros(pad_shape), arr
                ], axis=0)[:n_samples]
            else:
                if key == 'direction_probs':
                    all_preds[key] = np.zeros((n_samples, 3))
                elif key == 'quantiles':
                    all_preds[key] = np.zeros((n_samples, 5))
                else:
                    all_preds[key] = np.zeros((n_samples,))
        
        return all_preds
    
    # =========================================================================
    # Inference Methods
    # =========================================================================
    
    def evaluate_trade_opportunity(
        self,
        features: np.ndarray,
        entry_price: float,
        pair: str,
        account_balance: float,
        current_spread: float,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        vision_features: Optional[np.ndarray] = None,
        current_time: Optional[datetime] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> TradeDecision:
        """
        Evaluate a potential trade opportunity.
        
        This is the main entry point for live trading decisions.
        
        Args:
            features: Recent feature sequence (seq_len, n_features)
            entry_price: Potential entry price
            pair: Currency pair (e.g., 'EURUSD')
            account_balance: Current account balance
            current_spread: Current spread in pips
            high, low, close: Optional price arrays for regime detection
            vision_features: Optional vision model features
            current_time: Current datetime (for session rules)
            market_data: Optional market condition data
        
        Returns:
            TradeDecision with complete trade parameters
        """
        decision = TradeDecision()
        
        if not self._is_trained:
            decision.rejection_reasons.append("Model not trained")
            return decision
        
        # =====================================================================
        # Phase 1: Get Predictions
        # =====================================================================
        predictions = self._get_predictions(features, vision_features)
        
        # Extract direction
        direction_probs = predictions['direction_probs']
        pred_direction_idx = int(np.argmax(direction_probs))
        confidence = float(np.max(direction_probs))
        
        decision.direction_probs = {
            'bear': float(direction_probs[0]),
            'sideways': float(direction_probs[1]),
            'bull': float(direction_probs[2])
        }
        decision.direction_confidence = confidence
        decision.volatility = float(predictions['volatility'])
        
        # Check minimum confidence
        if confidence < self.config.min_direction_confidence:
            decision.rejection_reasons.append(
                f"Low confidence: {confidence:.2f} < {self.config.min_direction_confidence}"
            )
        
        # Check for sideways prediction
        if pred_direction_idx == 1:
            decision.rejection_reasons.append("Sideways prediction - no clear direction")
            return decision
        
        # Set direction
        decision.direction = 'BUY' if pred_direction_idx == 2 else 'SELL'
        
        # =====================================================================
        # Detect Regime (if price data provided)
        # =====================================================================
        if high is not None and low is not None and close is not None:
            regime, _ = self.regime_detector.detect(high, low, close)
            decision.regime = regime.value
        
        # =====================================================================
        # Phase 2: Calculate Risk Parameters
        # =====================================================================
        
        # Calculate SL/TP
        sltp_result = calculate_sl_tp_from_predictions(
            entry_price=entry_price,
            direction=decision.direction,
            predictions={
                'quantiles': predictions['quantiles'],
                'volatility': predictions['volatility'],
                'direction_probs': predictions['direction_probs']
            },
            regime=decision.regime if decision.regime else None,
            atr=decision.volatility
        )
        
        decision.stop_loss = sltp_result.stop_loss
        decision.take_profit = sltp_result.take_profit
        decision.sl_distance_pips = sltp_result.sl_distance * (100 if 'JPY' in pair else 10000)
        decision.tp_distance_pips = sltp_result.tp_distance * (100 if 'JPY' in pair else 10000)
        decision.risk_reward_ratio = sltp_result.risk_reward_ratio
        
        # Calculate Position Size
        position_result = self.position_calculator.calculate(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=sltp_result.stop_loss,
            pair=pair,
            direction_confidence=confidence,
            volatility=decision.volatility
        )
        
        decision.position_size = position_result.position_size
        decision.position_units = position_result.units
        decision.risk_amount = position_result.risk_amount
        decision.risk_percent = position_result.risk_percent
        
        if position_result.warnings:
            for warning in position_result.warnings:
                decision.rejection_reasons.append(warning)
        
        # =====================================================================
        # Phase 2: Apply Hard Rules
        # =====================================================================
        validation = self.gatekeeper.validate_trade(
            pair=pair,
            direction=decision.direction,
            position_size=decision.position_size,
            entry_price=entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            account_balance=account_balance,
            current_spread=current_spread,
            current_time=current_time,
            regime=decision.regime
        )
        
        decision.rule_violations = validation['violations']
        
        if not validation['allowed']:
            for violation in validation['violations']:
                if violation['severity'] in ('block', 'critical'):
                    decision.rejection_reasons.append(violation['message'])
        
        # Apply any adjustments
        if 'position_size' in validation['adjustments']:
            decision.position_size = validation['adjustments']['position_size']
            decision.position_units = int(decision.position_size * 100000)
        
        # =====================================================================
        # Phase 3: Meta-Labeling Filter
        # =====================================================================
        if self._meta_model_trained and self.trade_filter is not None:
            # Extract meta-features
            meta_features = self.meta_model.feature_extractor.extract_features(
                primary_predictions={
                    'direction_probs': predictions['direction_probs'].reshape(1, -1),
                    'volatility': np.array([predictions['volatility']]),
                    'quantiles': predictions['quantiles'].reshape(1, -1)
                },
                market_data=market_data,
                timestamps=None
            )
            
            # Get meta-score
            meta_score = self.meta_model.predict_proba(meta_features)[0]
            decision.meta_score = float(meta_score)
            
            if meta_score < self.config.meta_labeling_threshold:
                decision.rejection_reasons.append(
                    f"Meta-model filter: {meta_score:.2f} < {self.config.meta_labeling_threshold}"
                )
        
        # =====================================================================
        # Final Decision
        # =====================================================================
        decision.should_trade = len(decision.rejection_reasons) == 0
        
        return decision
    
    def _get_predictions(
        self,
        features: np.ndarray,
        vision_features: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Get predictions from the TCN model.

        Returns:
            Dict with direction/volatility/quantiles/features and optionally
            p_long/p_short if the model provides them.
        """
        self.tcn_model.eval()
        
        # Ensure correct shape
        if features.ndim == 2:
            features = features[-self.config.sequence_length:]
            features = features.reshape(1, self.config.sequence_length, -1)
        
        # Normalize
        if hasattr(self, '_norm_params'):
            from .utils import apply_normalization
            features = apply_normalization(
                features.reshape(-1, features.shape[-1]),
                self._norm_params
            ).reshape(features.shape)
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        vision_tensor = None
        if vision_features is not None:
            vision_tensor = torch.tensor(
                vision_features, dtype=torch.float32
            ).to(self.device)
        
        with torch.no_grad():
            output = self.tcn_model(x, vision_tensor, mode='all')

        result = {
            'direction_probs': output['direction'].cpu().numpy()[0],
            'volatility': output['volatility'].cpu().numpy().item(),
            'quantiles': output['quantiles'].cpu().numpy()[0],
            'features': output['features'].cpu().numpy()[0]
        }

        if 'p_long' in output:
            result['p_long'] = output['p_long'].cpu().numpy().item()
        if 'p_short' in output:
            result['p_short'] = output['p_short'].cpu().numpy().item()

        return result
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self, path: Optional[str] = None):
        """Save the complete risk management system."""
        path = path or self.config.model_dir
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save TCN model
        torch.save({
            'model_state_dict': self.tcn_model.state_dict(),
            'config': self.tcn_model.config
        }, save_dir / 'tcn_model.pt')
        
        # Save meta-model
        if self._meta_model_trained:
            self.meta_model.save(str(save_dir / 'meta_model.joblib'))
        
        # Save normalization params
        if hasattr(self, '_norm_params'):
            np.save(save_dir / 'norm_params.npy', self._norm_params)
        
        # Save config
        config_dict = {
            'profile': self.config.profile,
            'input_features': self.config.input_features,
            'sequence_length': self.config.sequence_length,
            'is_trained': self._is_trained,
            'meta_model_trained': self._meta_model_trained
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Risk management system saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'RiskManager':
        """Load a saved risk management system."""
        load_dir = Path(path)
        
        # Load config
        with open(load_dir / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = RiskManagerConfig(
            profile=config_dict['profile'],
            input_features=config_dict['input_features'],
            sequence_length=config_dict['sequence_length'],
            device=device,
            model_dir=path
        )
        
        manager = cls(config)
        
        # Load TCN model
        checkpoint = torch.load(load_dir / 'tcn_model.pt', map_location=device)
        manager.tcn_model.load_state_dict(checkpoint['model_state_dict'])
        manager._is_trained = config_dict['is_trained']
        
        # Load meta-model
        if config_dict['meta_model_trained']:
            manager.meta_model = MetaLabelingModel.load(
                str(load_dir / 'meta_model.joblib')
            )
            manager.trade_filter = TradeFilter(
                meta_model=manager.meta_model,
                min_confidence=config.min_direction_confidence,
                min_meta_score=config.meta_labeling_threshold
            )
            manager._meta_model_trained = True
        
        # Load normalization params
        norm_path = load_dir / 'norm_params.npy'
        if norm_path.exists():
            manager._norm_params = np.load(norm_path, allow_pickle=True).item()
        
        logger.info(f"Risk management system loaded from {path}")
        return manager
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def update_positions(self, positions: Dict[str, Dict]):
        """Update tracked positions for exposure calculations."""
        self.rules_engine.update_positions(positions)
    
    def add_news_event(
        self,
        event_time: datetime,
        title: str,
        impact: str = 'high',
        currencies: Optional[List[str]] = None
    ):
        """Add a scheduled news event for blackout consideration."""
        self.rules_engine.add_news_event(event_time, title, impact, currencies)
    
    def get_model_summary(self) -> Dict:
        """Get summary of the risk management system."""
        return {
            'profile': self.config.profile,
            'input_features': self.config.input_features,
            'sequence_length': self.config.sequence_length,
            'tcn_trained': self._is_trained,
            'meta_model_trained': self._meta_model_trained,
            'device': str(self.device),
            'tcn_parameters': sum(p.numel() for p in self.tcn_model.parameters()),
            'receptive_field': self.tcn_model.config.receptive_field
        }
