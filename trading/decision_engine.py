# trading/decision_engine.py
"""
Enhanced Decision Engine with Full Risk Management Integration (v2)

Integrates ALL 5 Phases:
- Phase 1: Multi-head TCN predictions (direction, volatility, quantiles)
- Phase 2: SL/TP calculation, position sizing, hard rules
- Phase 3: Meta-labeling filter for signal quality
- Phase 4: RL exit recommendations (via ExitAdvisor)
- Phase 5: Capital protection (final safety layer)

This replaces the previous MTFDecisionEngine with comprehensive risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import logging

# Risk Management imports - Phases 1-3
from risk_management import (
    # Phase 1
    MultiHeadTCN, create_tcn_for_profile, RiskPrediction,
    # Phase 2
    SLTPCalculator, SLTPConfig, SLTPResult,
    PositionSizingCalculator, PositionSizingConfig, PositionSizeResult,
    TradeGatekeeper, HardRulesConfig, MarketRegime, TradeDirection,
    # Phase 3
    MetaLabelingModel, TradeFilter,
    # Utils
    RegimeDetector
)

# Phase 5: Capital Protection
from risk_management import (
    CapitalProtector, ProtectionConfig, ProtectionManager,
    ProtectionLevel, ProtectionAction
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
    
    # Capital protection (Phase 5)
    protection_level: str = 'normal'
    protection_warnings: List[str] = field(default_factory=list)
    size_adjusted_by_protection: bool = False
    original_position_size: float = 0.0
    
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
            'direction': self.direction,
            'direction_confidence': self.direction_confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'position_units': self.position_units,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'meta_score': self.meta_score,
            'protection_level': self.protection_level,
            'protection_warnings': self.protection_warnings,
            'regime': self.regime,
            'volatility': self.volatility,
            'mtf_alignment': self.mtf_alignment,
            'rejection_reasons': self.rejection_reasons
        }


@dataclass
class DecisionEngineConfig:
    """Configuration for decision engine."""
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
    
    # Capital protection settings (Phase 5)
    enable_capital_protection: bool = True
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 6.0
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 30


class EnhancedDecisionEngine:
    """
    Decision engine with full risk management integration (v2).
    
    Combines ALL 5 Phases:
    1. Direction prediction (TCN/existing model)
    2. MTF trend analysis (existing)
    3. Risk calculations (Phase 2)
    4. Trade filtering (Phase 3)
    5. Hard rules enforcement (Phase 2)
    6. Capital protection (Phase 5) - FINAL SAFETY CHECK
    
    Usage:
        engine = EnhancedDecisionEngine(config)
        engine.initialize(starting_balance=10000)
        
        decision = engine.evaluate(
            predictions=model_predictions,
            entry_price=1.1234,
            pair='EURUSD',
            account_balance=10000,
            market_data=df
        )
        
        if decision.should_trade:
            execute_trade(decision)
        
        # After trade closes:
        engine.record_trade_result(pnl=150, is_win=True)
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
        
        # Phase 5: Capital Protection
        self.capital_protector: Optional[CapitalProtector] = None
        if self.config.enable_capital_protection:
            protection_config = ProtectionConfig(
                max_daily_loss_pct=self.config.max_daily_loss_pct,
                max_weekly_loss_pct=self.config.max_weekly_loss_pct,
                max_drawdown_pct=self.config.max_drawdown_pct,
                max_consecutive_losses=self.config.max_consecutive_losses,
                losing_streak_cooldown_minutes=self.config.cooldown_minutes
            )
            self.capital_protector = CapitalProtector(protection_config)
        
        # Utilities
        self.regime_detector = RegimeDetector()
        
        # MTF integration (if available)
        self.mtf_detector = None
        if MTFTrendDetector is not None:
            try:
                self.mtf_detector = MTFTrendDetector(profile=self.config.profile)
            except Exception as e:
                logger.warning(f"Could not initialize MTF detector: {e}")
        
        # Tracking
        self._initialized = False
        self._current_balance = 0.0
        
        logger.info(f"EnhancedDecisionEngine v2 initialized for {self.config.profile}")
    
    def initialize(self, starting_balance: float):
        """Initialize engine with starting balance (required for capital protection)."""
        self._current_balance = starting_balance
        if self.capital_protector:
            self.capital_protector.initialize(starting_balance)
        self._initialized = True
        logger.info(f"Engine initialized with balance: {starting_balance:.2f}")
    
    def record_trade_result(self, pnl: float, is_win: bool, size: float = 0.0):
        """Record trade result for capital protection tracking."""
        self._current_balance += pnl
        if self.capital_protector:
            self.capital_protector.record_trade(pnl, is_win, size)
    
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
        # Phase 5 PRE-CHECK: Capital Protection
        # =================================================================
        if self.capital_protector:
            # Early check - is trading even allowed?
            pre_check = self.capital_protector.check_trade(
                proposed_size=0.01,  # Minimal check
                account_balance=account_balance
            )
            
            if not pre_check['allowed']:
                decision.rejection_reasons.append(
                    f"Capital protection: {pre_check['reason']}"
                )
                decision.protection_level = pre_check.get('protection_level', 'critical')
                return decision
            
            decision.protection_level = pre_check.get('protection_level', 'normal')
        
        # =================================================================
        # Step 1: Extract direction from predictions
        # =================================================================
        direction_probs = predictions.get('direction_probs')
        if direction_probs is None:
            decision.rejection_reasons.append("No direction predictions available")
            return decision
        
        # Handle different formats
        if isinstance(direction_probs, np.ndarray):
            if direction_probs.ndim > 1:
                direction_probs = direction_probs.flatten()
            probs = {
                'BEAR': float(direction_probs[0]),
                'SIDEWAYS': float(direction_probs[1]),
                'BULL': float(direction_probs[2])
            }
        else:
            probs = direction_probs
        
        decision.direction_probs = probs
        
        # Get predicted direction and confidence
        direction_idx = np.argmax(list(probs.values()))
        direction_names = ['BEAR', 'SIDEWAYS', 'BULL']
        predicted_direction = direction_names[direction_idx]
        confidence = list(probs.values())[direction_idx]
        
        decision.signal = Signal(direction_idx)
        decision.signal_name = predicted_direction
        decision.direction_confidence = confidence
        
        # Check confidence threshold
        if confidence < self.config.min_direction_confidence:
            decision.rejection_reasons.append(
                f"Low confidence: {confidence:.2%} < {self.config.min_direction_confidence:.2%}"
            )
            return decision
        
        # Skip if sideways
        if predicted_direction == 'SIDEWAYS':
            decision.rejection_reasons.append("Sideways prediction - no trade")
            return decision
        
        decision.direction = 'BUY' if predicted_direction == 'BULL' else 'SELL'
        
        # =================================================================
        # Step 2: Market Regime Detection
        # =================================================================
        regime = self._detect_regime(market_data)
        decision.regime = regime.value if hasattr(regime, 'value') else str(regime)
        
        # Extract volatility
        volatility = predictions.get('volatility')
        if volatility is not None:
            if isinstance(volatility, np.ndarray):
                volatility = float(volatility.flatten()[0])
            decision.volatility = volatility
        
        # Get ATR if available
        if 'atr' in market_data.columns:
            decision.atr = float(market_data['atr'].iloc[-1])
        
        # =================================================================
        # Step 3: Calculate SL/TP (Phase 2)
        # =================================================================
        quantiles = predictions.get('quantiles')
        trade_dir = TradeDirection.BUY if decision.direction == 'BUY' else TradeDirection.SELL
        
        sltp_result = self.sltp_calculator.calculate(
            entry_price=entry_price,
            direction=trade_dir,
            quantiles=quantiles,
            volatility=volatility,
            regime=regime,
            atr=decision.atr if decision.atr > 0 else None
        )
        
        decision.stop_loss = sltp_result.stop_loss
        decision.take_profit = sltp_result.take_profit
        pip_size = 0.01 if 'JPY' in pair.upper() else 0.0001
        decision.sl_pips = float(sltp_result.sl_distance / pip_size) if pip_size > 0 else 0.0
        decision.tp_pips = float(sltp_result.tp_distance / pip_size) if pip_size > 0 else 0.0
        decision.risk_reward_ratio = sltp_result.risk_reward_ratio
        
        # Check risk-reward
        if sltp_result.risk_reward_ratio < self.config.min_risk_reward:
            decision.rejection_reasons.append(
                f"Low R:R ratio: {sltp_result.risk_reward_ratio:.2f} < {self.config.min_risk_reward}"
            )
            return decision
        
        # =================================================================
        # Step 4: Position Sizing (Phase 2)
        # =================================================================
        size_result = self.position_calculator.calculate(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=decision.stop_loss,
            volatility=volatility,
            regime=regime,
            direction_confidence=confidence
        )
        
        decision.position_size = size_result.position_size
        decision.position_units = size_result.position_units
        decision.risk_amount = size_result.risk_amount
        decision.risk_percent = size_result.risk_percent
        decision.original_position_size = size_result.position_size
        
        # =================================================================
        # Step 5: Hard Rules Check (Phase 2)
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
            current_time=current_time
        )
        
        if not validation['allowed']:
            decision.rule_violations = validation['violations']
            decision.rejection_reasons.extend(
                [v['message'] for v in validation['violations']]
            )
            return decision
        
        # =================================================================
        # Step 6: Meta-labeling Filter (Phase 3)
        # =================================================================
        if self.trade_filter:
            meta_features = self._build_meta_features(
                predictions, market_data, decision
            )
            
            filter_result = self.trade_filter.filter(
                signal=decision.direction,
                features=meta_features,
                direction_confidence=confidence
            )
            
            decision.meta_score = filter_result.meta_score
            
            if not filter_result.should_trade:
                decision.rejection_reasons.append(
                    f"Meta-filter rejected: score={filter_result.meta_score:.2f}"
                )
                return decision
        
        # =================================================================
        # Step 7: MTF Alignment Check (Optional)
        # =================================================================
        if self.mtf_detector and mtf_data:
            mtf_result = self._check_mtf_alignment(decision.direction, mtf_data)
            decision.mtf_alignment = mtf_result['alignment']
            decision.mtf_trend = mtf_result['trend']
            
            if mtf_result['alignment'] < self.config.min_mtf_alignment:
                decision.rejection_reasons.append(
                    f"Low MTF alignment: {mtf_result['alignment']:.2%}"
                )
                return decision
        
        # =================================================================
        # Phase 5 FINAL CHECK: Capital Protection Size Adjustment
        # =================================================================
        if self.capital_protector:
            final_check = self.capital_protector.check_trade(
                proposed_size=decision.position_size,
                account_balance=account_balance
            )
            
            if not final_check['allowed']:
                decision.rejection_reasons.append(
                    f"Capital protection final check: {final_check['reason']}"
                )
                return decision
            
            # Apply size adjustment if needed
            adjusted_size = final_check.get('adjusted_size', decision.position_size)
            if adjusted_size != decision.position_size:
                decision.size_adjusted_by_protection = True
                decision.position_size = adjusted_size
                decision.position_units = int(adjusted_size * 100000)  # Standard lot
            
            # Record warnings
            decision.protection_warnings = final_check.get('warnings', [])
            decision.protection_level = final_check.get('protection_level', 'normal')
        
        # =================================================================
        # All checks passed - approve trade
        # =================================================================
        decision.should_trade = True
        
        logger.info(
            f"Trade approved: {decision.direction} {pair} | "
            f"Size: {decision.position_size:.4f} | "
            f"SL: {decision.sl_pips:.1f} pips | "
            f"TP: {decision.tp_pips:.1f} pips | "
            f"R:R: {decision.risk_reward_ratio:.2f} | "
            f"Protection: {decision.protection_level}"
        )
        
        return decision
    
    def get_protection_status(self) -> Dict:
        """Get current capital protection status."""
        if not self.capital_protector:
            return {'enabled': False}
        
        state = self.capital_protector.get_state()
        metrics = self.capital_protector.get_metrics()
        
        return {
            'enabled': True,
            'level': state.level.value,
            'action': state.action.value,
            'size_multiplier': state.size_multiplier,
            'trigger_reason': state.trigger_reason,
            'metrics': {
                'balance': metrics.current_balance,
                'peak_balance': metrics.peak_balance,
                'drawdown_pct': metrics.current_drawdown_pct,
                'daily_pnl': metrics.daily_pnl,
                'weekly_pnl': metrics.weekly_pnl,
                'consecutive_losses': metrics.consecutive_losses,
                'win_rate': metrics.recent_win_rate
            }
        }
    
    def reset_daily_protection(self):
        """Reset daily protection counters."""
        if self.capital_protector:
            self.capital_protector.reset_daily()
    
    def reset_weekly_protection(self):
        """Reset weekly protection counters."""
        if self.capital_protector:
            self.capital_protector.reset_weekly()
    
    def _detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        try:
            return self.regime_detector.detect(market_data)
        except Exception:
            return MarketRegime.VOLATILE
    
    def _estimate_spread(self, pair: str) -> float:
        """Estimate spread for a pair."""
        spreads = {
            'EURUSD': 1.0,
            'GBPUSD': 1.5,
            'USDJPY': 1.0,
            'USDCHF': 1.5,
            'AUDUSD': 1.2,
            'NZDUSD': 1.5,
            'USDCAD': 1.5,
        }
        return spreads.get(pair.upper(), 2.0)
    
    def _build_meta_features(
        self,
        predictions: Dict,
        market_data: pd.DataFrame,
        decision: TradeDecision
    ) -> np.ndarray:
        """Build feature vector for meta-labeling."""
        features = []
        
        # Direction confidence
        features.append(decision.direction_confidence)
        
        # Volatility
        features.append(decision.volatility if decision.volatility else 0.0)
        
        # ATR
        features.append(decision.atr if decision.atr else 0.0)
        
        # Risk-reward
        features.append(decision.risk_reward_ratio)
        
        # Quantiles spread
        quantiles = predictions.get('quantiles')
        if quantiles is not None:
            if isinstance(quantiles, np.ndarray):
                features.append(float(quantiles.flatten()[-1] - quantiles.flatten()[0]))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Recent returns
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().iloc[-5:]
            features.extend([
                returns.mean() if len(returns) > 0 else 0.0,
                returns.std() if len(returns) > 0 else 0.0
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _check_mtf_alignment(
        self,
        direction: str,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Check multi-timeframe trend alignment."""
        if not self.mtf_detector:
            return {'alignment': 1.0, 'trend': 'unknown'}
        
        try:
            analysis = self.mtf_detector.analyze(mtf_data)
            
            # Calculate alignment with intended direction
            aligned = 0
            total = len(analysis.get('timeframes', {}))
            
            for tf, tf_data in analysis.get('timeframes', {}).items():
                tf_direction = tf_data.get('direction', 'SIDEWAYS')
                if (direction == 'BUY' and tf_direction == 'BULL') or \
                   (direction == 'SELL' and tf_direction == 'BEAR'):
                    aligned += 1
            
            alignment = aligned / total if total > 0 else 0.0
            
            return {
                'alignment': alignment,
                'trend': analysis.get('overall_trend', 'unknown')
            }
        except Exception as e:
            logger.warning(f"MTF analysis error: {e}")
            return {'alignment': 1.0, 'trend': 'unknown'}


def convert_legacy_predictions(
    direction_probs: np.ndarray,
    volatility: Optional[float] = None,
    entry_price: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Convert legacy prediction format to new format.
    
    Args:
        direction_probs: [P(bear), P(sideways), P(bull)]
        volatility: Optional volatility estimate
        entry_price: Current price for quantile estimation
    
    Returns:
        Dict with 'direction_probs', 'volatility', 'quantiles'
    """
    result = {
        'direction_probs': direction_probs
    }
    
    # Estimate volatility if not provided
    if volatility is None:
        volatility = 0.001  # Default 10 pips
    result['volatility'] = np.array([volatility])
    
    # Generate approximate quantiles
    q5 = -2.0 * volatility
    q25 = -0.67 * volatility
    q50 = 0.0
    q75 = 0.67 * volatility
    q95 = 2.0 * volatility
    
    result['quantiles'] = np.array([q5, q25, q50, q75, q95])
    
    return result


# Backward compatibility alias
MTFDecisionEngine = EnhancedDecisionEngine
