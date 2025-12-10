# trading/decision_engine.py
"""
Enhanced Decision Engine with Full Risk Management Integration

Integrates:
- Phase 1: Multi-head TCN predictions (direction, volatility, quantiles)
- Phase 2: SL/TP calculation, position sizing, hard rules
- Phase 3: Meta-labeling filter for signal quality

This replaces the previous MTFDecisionEngine with comprehensive risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import logging

# Risk Management imports
from risk_management import (
    # Phase 1
    MultiHeadTCN, create_tcn_for_profile, RiskPrediction,
    # Phase 2
    SLTPCalculator, SLTPConfig, SLTPResult,
    PositionSizingCalculator, PositionSizingConfig, PositionSizeResult,
    TradeGatekeeper, HardRulesConfig, MarketRegime, TradeDirection,
    calculate_sl_tp_from_predictions, calculate_position_from_predictions,
    # Phase 3
    MetaLabelingModel, TradeFilter,
    # Utils
    RegimeDetector
)

# Existing pyForex imports
try:
    from utils.mtf_config import get_profile, MTFProfile
    from trend_detection.mtf_trend_detector import MTFTrendDetector
except ImportError:
    MTFProfile = None
    MTFTrendDetector = None

logger = logging.getLogger(__name__)


class Signal(IntEnum):
    """Trading signal enumeration."""
    BEAR = 0     # Sell
    SIDEWAYS = 1 # Hold
    BULL = 2     # Buy


@dataclass
class TradeDecision:
    """Complete trade decision with risk parameters."""
    # Signal
    signal: Signal
    signal_name: str
    should_trade: bool
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Direction
    direction: str = ''  # 'BUY', 'SELL', ''
    direction_confidence: float = 0.0
    direction_probs: Dict[str, float] = field(default_factory=dict)
    
    # Risk parameters (from Phase 2)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Position sizing
    position_size: float = 0.0
    position_units: int = 0
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    
    # Meta-labeling (Phase 3)
    meta_score: float = 0.0
    
    # Market context
    regime: str = ''
    volatility: float = 0.0
    atr: float = 0.0
    
    # MTF context (if available)
    mtf_alignment: float = 0.0
    mtf_trend: str = ''
    
    # Validation
    rule_violations: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'signal': self.signal.value,
            'signal_name': self.signal_name,
            'should_trade': self.should_trade,
            'rejection_reasons': self.rejection_reasons,
            'direction': self.direction,
            'direction_confidence': self.direction_confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'risk_percent': self.risk_percent,
            'meta_score': self.meta_score,
            'regime': self.regime,
            'mtf_alignment': self.mtf_alignment
        }


@dataclass
class DecisionEngineConfig:
    """Configuration for enhanced decision engine."""
    # Trading profile
    profile: str = 'INTRADAY'
    
    # Confidence thresholds
    min_direction_confidence: float = 0.55
    min_meta_score: float = 0.5
    min_mtf_alignment: float = 0.6
    
    # Risk parameters
    base_risk_percent: float = 1.0
    min_risk_reward: float = 1.5
    max_leverage: float = 10.0
    
    # Regime restrictions
    allow_volatile_regime: bool = False
    reduce_size_low_vol: bool = True


class EnhancedDecisionEngine:
    """
    Decision engine with full risk management integration.
    
    Combines:
    1. Direction prediction (TCN/existing model)
    2. MTF trend analysis (existing)
    3. Risk calculations (new Phase 2)
    4. Trade filtering (new Phase 3)
    5. Hard rules enforcement (new Phase 2)
    
    Usage:
        engine = EnhancedDecisionEngine(config)
        decision = engine.evaluate(
            predictions=model_predictions,
            entry_price=1.1234,
            pair='EURUSD',
            account_balance=10000,
            market_data=df
        )
        
        if decision.should_trade:
            execute_trade(decision)
    """
    
    def __init__(
        self,
        config: Optional[DecisionEngineConfig] = None,
        meta_model: Optional[MetaLabelingModel] = None
    ):
        self.config = config or DecisionEngineConfig()
        
        # Phase 2: Risk calculators
        self.sltp_calculator = SLTPCalculator(SLTPConfig(
            min_risk_reward=self.config.min_risk_reward
        ))
        
        self.position_calculator = PositionSizingCalculator(PositionSizingConfig(
            base_risk_percent=self.config.base_risk_percent
        ))
        
        self.gatekeeper = TradeGatekeeper(HardRulesConfig(
            max_leverage_default=self.config.max_leverage
        ))
        
        # Phase 3: Meta-labeling filter
        self.meta_model = meta_model
        self.trade_filter = TradeFilter(
            meta_model=meta_model,
            min_confidence=self.config.min_direction_confidence,
            min_meta_score=self.config.min_meta_score
        ) if meta_model else None
        
        # Utilities
        self.regime_detector = RegimeDetector()
        
        # MTF integration (if available)
        self.mtf_detector = None
        if MTFTrendDetector is not None:
            try:
                self.mtf_detector = MTFTrendDetector(profile=self.config.profile)
            except Exception as e:
                logger.warning(f"Could not initialize MTF detector: {e}")
        
        logger.info(f"EnhancedDecisionEngine initialized for {self.config.profile}")
    
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        entry_price: float,
        pair: str,
        account_balance: float,
        market_data: pd.DataFrame,
        current_spread: Optional[float] = None,
        current_time: Optional[datetime] = None,
        mtf_data: Optional[Dict[str, pd.DataFrame]] = None,
        vision_features: Optional[np.ndarray] = None
    ) -> TradeDecision:
        """
        Evaluate a trading opportunity with full risk management.
        
        Args:
            predictions: Dict with 'direction_probs', 'volatility', 'quantiles'
                        (from TCN or converted from existing model)
            entry_price: Current price / intended entry
            pair: Currency pair (e.g., 'EURUSD')
            account_balance: Current account balance
            market_data: DataFrame with OHLCV + indicators
            current_spread: Current spread in pips (optional)
            current_time: Current datetime (optional)
            mtf_data: Multi-timeframe data dict (optional)
            vision_features: Vision model features (optional)
        
        Returns:
            TradeDecision with all risk parameters
        """
        current_time = current_time or datetime.utcnow()
        current_spread = current_spread or self._estimate_spread(pair)
        
        decision = TradeDecision(
            signal=Signal.SIDEWAYS,
            signal_name='HOLD',
            should_trade=False
        )
        
        # =================================================================
        # Step 1: Extract direction from predictions
        # =================================================================
        direction_probs = predictions.get('direction_probs')
        if direction_probs is None:
            decision.rejection_reasons.append("No direction predictions available")
            return decision
        
        if isinstance(direction_probs, np.ndarray) and direction_probs.ndim > 1:
            direction_probs = direction_probs[0]
        
        predicted_class = int(np.argmax(direction_probs))
        confidence = float(np.max(direction_probs))
        
        decision.signal = Signal(predicted_class)
        decision.signal_name = ['BEAR', 'SIDEWAYS', 'BULL'][predicted_class]
        decision.direction_confidence = confidence
        decision.direction_probs = {
            'bear': float(direction_probs[0]),
            'sideways': float(direction_probs[1]),
            'bull': float(direction_probs[2])
        }
        
        # Determine trade direction
        if predicted_class == Signal.BULL:
            decision.direction = 'BUY'
        elif predicted_class == Signal.BEAR:
            decision.direction = 'SELL'
        else:
            decision.rejection_reasons.append("Sideways prediction - no trade")
            return decision
        
        # Check confidence threshold
        if confidence < self.config.min_direction_confidence:
            decision.rejection_reasons.append(
                f"Low confidence: {confidence:.2f} < {self.config.min_direction_confidence}"
            )
            return decision
        
        # =================================================================
        # Step 2: Detect market regime
        # =================================================================
        if 'high' in market_data.columns and 'low' in market_data.columns:
            regime_info = self.regime_detector.detect(
                market_data['high'].values,
                market_data['low'].values,
                market_data['close'].values
            )
            decision.regime = regime_info.regime.value
            
            # Block volatile regime if configured
            if decision.regime == 'volatile' and not self.config.allow_volatile_regime:
                decision.rejection_reasons.append("Volatile regime - trading blocked")
                return decision
        
        # =================================================================
        # Step 3: Extract volatility and quantiles
        # =================================================================
        volatility = predictions.get('volatility')
        if volatility is not None:
            if hasattr(volatility, 'item'):
                volatility = volatility.item()
            decision.volatility = float(volatility)
        
        # Calculate ATR if not provided
        if 'atr' in market_data.columns:
            decision.atr = float(market_data['atr'].iloc[-1])
        elif decision.volatility > 0:
            decision.atr = decision.volatility
        
        # =================================================================
        # Step 4: MTF Analysis (if available)
        # =================================================================
        if self.mtf_detector and mtf_data:
            try:
                mtf_result = self.mtf_detector.analyze(mtf_data)
                decision.mtf_alignment = mtf_result.get('alignment_score', 0)
                decision.mtf_trend = mtf_result.get('trend', 'unknown')
                
                # Check MTF alignment
                if decision.mtf_alignment < self.config.min_mtf_alignment:
                    decision.rejection_reasons.append(
                        f"Low MTF alignment: {decision.mtf_alignment:.2f}"
                    )
            except Exception as e:
                logger.warning(f"MTF analysis failed: {e}")
        
        # =================================================================
        # Step 5: Calculate SL/TP (Phase 2)
        # =================================================================
        quantiles = predictions.get('quantiles')
        if quantiles is not None:
            sltp_result = calculate_sl_tp_from_predictions(
                entry_price=entry_price,
                direction=decision.direction,
                predictions=predictions,
                regime=decision.regime,
                atr=decision.atr
            )
            
            decision.stop_loss = sltp_result.stop_loss
            decision.take_profit = sltp_result.take_profit
            decision.sl_pips = self._price_to_pips(sltp_result.sl_distance, pair)
            decision.tp_pips = self._price_to_pips(sltp_result.tp_distance, pair)
            decision.risk_reward_ratio = sltp_result.risk_reward_ratio
        else:
            # Fallback: ATR-based SL/TP
            atr = decision.atr or decision.volatility or entry_price * 0.001
            sl_distance = atr * 1.5
            tp_distance = atr * 2.5
            
            if decision.direction == 'BUY':
                decision.stop_loss = entry_price - sl_distance
                decision.take_profit = entry_price + tp_distance
            else:
                decision.stop_loss = entry_price + sl_distance
                decision.take_profit = entry_price - tp_distance
            
            decision.sl_pips = self._price_to_pips(sl_distance, pair)
            decision.tp_pips = self._price_to_pips(tp_distance, pair)
            decision.risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # =================================================================
        # Step 6: Position Sizing (Phase 2)
        # =================================================================
        position_result = self.position_calculator.calculate(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=decision.stop_loss,
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
        
        # =================================================================
        # Step 7: Hard Rules Check (Phase 2)
        # =================================================================
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
        
        # Apply adjustments
        if 'position_size' in validation['adjustments']:
            decision.position_size = validation['adjustments']['position_size']
            decision.position_units = int(decision.position_size * 100000)
        
        # =================================================================
        # Step 8: Meta-Labeling Filter (Phase 3)
        # =================================================================
        if self.trade_filter and self.meta_model:
            try:
                meta_features = self.meta_model.feature_extractor.extract_features(
                    primary_predictions={
                        'direction_probs': direction_probs.reshape(1, -1),
                        'volatility': np.array([decision.volatility]),
                        'quantiles': predictions.get('quantiles', np.zeros((1, 5)))
                    },
                    market_data=market_data,
                    timestamps=None
                )
                
                meta_score = self.meta_model.predict_proba(meta_features)[0]
                decision.meta_score = float(meta_score)
                
                if meta_score < self.config.min_meta_score:
                    decision.rejection_reasons.append(
                        f"Meta-filter rejected: {meta_score:.2f} < {self.config.min_meta_score}"
                    )
            except Exception as e:
                logger.warning(f"Meta-labeling failed: {e}")
        
        # =================================================================
        # Final Decision
        # =================================================================
        decision.should_trade = len(decision.rejection_reasons) == 0
        
        if decision.should_trade:
            logger.info(
                f"Trade approved: {pair} {decision.direction} | "
                f"Size: {decision.position_size:.2f} lots | "
                f"SL: {decision.sl_pips:.1f} pips | "
                f"TP: {decision.tp_pips:.1f} pips | "
                f"R:R: {decision.risk_reward_ratio:.2f}"
            )
        else:
            logger.info(f"Trade rejected: {decision.rejection_reasons}")
        
        return decision
    
    def _price_to_pips(self, price_diff: float, pair: str) -> float:
        """Convert price difference to pips."""
        if 'JPY' in pair.upper():
            return price_diff * 100
        return price_diff * 10000
    
    def _estimate_spread(self, pair: str) -> float:
        """Estimate typical spread for a pair."""
        spreads = {
            'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.0,
            'USDCHF': 1.5, 'AUDUSD': 1.5, 'USDCAD': 1.5,
            'NZDUSD': 2.0, 'EURJPY': 2.0, 'GBPJPY': 3.0
        }
        return spreads.get(pair.upper(), 2.0)
    
    def update_positions(self, positions: Dict[str, Dict]):
        """Update tracked positions for exposure calculations."""
        self.gatekeeper.rules_engine.update_positions(positions)
    
    def set_meta_model(self, meta_model: MetaLabelingModel):
        """Set or update the meta-labeling model."""
        self.meta_model = meta_model
        self.trade_filter = TradeFilter(
            meta_model=meta_model,
            min_confidence=self.config.min_direction_confidence,
            min_meta_score=self.config.min_meta_score
        )


# Backward compatibility alias
MTFDecisionEngine = EnhancedDecisionEngine


def convert_legacy_predictions(probs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert legacy prediction format to new format.
    
    Legacy format: [P(BUY), P(SELL), P(HOLD)]
    New format: [P(BEAR), P(SIDEWAYS), P(BULL)]
    """
    if probs.shape[-1] != 3:
        raise ValueError(f"Expected 3 classes, got {probs.shape[-1]}")
    
    # Reorder: BUY->BULL, SELL->BEAR, HOLD->SIDEWAYS
    # Legacy: [BUY, SELL, HOLD] -> New: [BEAR, SIDEWAYS, BULL]
    if probs.ndim == 1:
        new_probs = np.array([probs[1], probs[2], probs[0]])  # SELL, HOLD, BUY
    else:
        new_probs = np.stack([probs[:, 1], probs[:, 2], probs[:, 0]], axis=1)
    
    return {
        'direction_probs': new_probs,
        'volatility': None,
        'quantiles': None
    }