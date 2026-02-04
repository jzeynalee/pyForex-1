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
    
    # Confidence thresholds - RAISED for better trade quality
    min_direction_confidence: float = 0.65  # Was 0.55
    min_meta_score: float = 0.60            # Was 0.5
    min_mtf_alignment: float = 0.70         # Was 0.6

    use_meta_labeling: bool = True
    
    # Risk parameters - TIGHTENED
    base_risk_percent: float = 0.5   # Was 1.0
    min_risk_reward: float = 2.0     # Was 1.5
    max_leverage: float = 10.0
    
    # Regime restrictions
    allow_volatile_regime: bool = False
    reduce_size_low_vol: bool = True
    
    # Capital protection settings (Phase 5)
    enable_capital_protection: bool = True
    max_daily_loss_pct: float = 2.0   # Was 3.0 - tighter daily limit
    max_weekly_loss_pct: float = 5.0  # Was 6.0 - tighter weekly limit
    max_drawdown_pct: float = 8.0     # Was 10.0 - tighter drawdown limit
    max_consecutive_losses: int = 4   # Was 5
    cooldown_minutes: int = 60        # Was 30 - longer cooldown

    # Hard-rules controls
    avoid_rollover: bool = True


class EnhancedDecisionEngine:
    """
    Decision engine with full risk management integration (v2).
    
    Combines ALL 5 Phases:
    1. Direction prediction (TCN/existing model)
       - If available, uses TP-before-SL probabilities (p_long/p_short) for side selection
         and confidence gating; otherwise falls back to direction probabilities.
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
            max_leverage_default=self.config.max_leverage,
            # Increase exposure limits for realistic forex trading
            # Standard forex uses 50:1 to 100:1 leverage
            max_single_pair_exposure=500.0,  # Allow up to 500% notional per pair
            max_total_exposure=1000.0,  # Allow up to 1000% total notional (10:1 effective)
            max_single_direction_exposure=500.0,  # Allow up to 500% in one direction
            max_correlated_group_exposure=500.0,  # Allow up to 500% in correlated pairs
            # Skip time-based checks for backtesting (historical data may have gaps)
            skip_weekend_check=True,
            skip_session_check=True,
            avoid_rollover=bool(getattr(self.config, 'avoid_rollover', True)),
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
                self.mtf_detector = MTFTrendDetector(profile=get_profile(self.config.profile))
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
    
    def record_trade_result(
        self,
        pnl: float,
        is_win: bool,
        size: float = 0.0,
        timestamp: Optional[datetime] = None,
    ):
        """Record trade result for capital protection tracking."""
        self._current_balance += pnl
        if self.capital_protector:
            try:
                self.capital_protector.record_trade(pnl, is_win, size, timestamp=timestamp)
            except TypeError:
                # Backward compatibility if protector signature differs
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
                Optional:
                    - p_long: P(TP before SL | long)
                    - p_short: P(TP before SL | short)
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
        # Step 1: Extract direction / trade side from predictions
        # =================================================================
        p_long = predictions.get('p_long')
        p_short = predictions.get('p_short')

        if p_long is not None and p_short is not None:
            try:
                p_long_f = float(np.array(p_long).flatten()[0])
                p_short_f = float(np.array(p_short).flatten()[0])
            except Exception:
                p_long_f = None
                p_short_f = None

            if p_long_f is not None and p_short_f is not None:
                confidence = max(p_long_f, p_short_f)
                predicted_direction = 'BULL' if p_long_f >= p_short_f else 'BEAR'

                decision.direction_probs = {
                    'BEAR': float(p_short_f),
                    'SIDEWAYS': float(max(0.0, 1.0 - confidence)),
                    'BULL': float(p_long_f)
                }

                decision.signal = Signal.BULL if predicted_direction == 'BULL' else Signal.BEAR
                decision.signal_name = predicted_direction
                decision.direction_confidence = confidence

                if confidence < self.config.min_direction_confidence:
                    # Outcome head may be untrained or poorly calibrated in older checkpoints.
                    # Fall back to direction_probs instead of hard rejecting the bar.
                    p_long = None
                    p_short = None
                    p_long_f = None
                    p_short_f = None

                decision.direction = 'BUY' if predicted_direction == 'BULL' else 'SELL'
            else:
                p_long = None
                p_short = None

        if p_long is None or p_short is None:
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
        
        # Extract volatility from model
        model_volatility = predictions.get('volatility')
        if model_volatility is not None:
            if isinstance(model_volatility, np.ndarray):
                model_volatility = float(model_volatility.flatten()[0])
        
        # Get ATR if available (more reliable for SL/TP)
        atr_value = None
        if 'atr' in market_data.columns:
            atr_value = float(market_data['atr'].iloc[-1])
        elif 'atr_14' in market_data.columns:
            atr_value = float(market_data['atr_14'].iloc[-1])
        
        # Use ATR as volatility if model volatility is unrealistic
        # Model volatility should be in the same scale as price movements (e.g., 0.001-0.003 for forex H1)
        # If model volatility > 0.01 (100 pips), it's likely garbage from untrained model
        if atr_value is not None and atr_value > 0 and atr_value < 0.1:
            # ATR is available and realistic - use it
            volatility = atr_value
        elif model_volatility is not None and 0.0001 < model_volatility < 0.01:
            # Model volatility is in realistic range
            volatility = model_volatility
        else:
            # Fallback: estimate volatility as ~20 pips for forex
            volatility = 0.002
        
        decision.volatility = volatility
        decision.atr = atr_value if atr_value else volatility
        
        # =================================================================
        # Step 3: Calculate SL/TP (Phase 2)
        # =================================================================
        quantiles = predictions.get('quantiles')
        trade_dir = TradeDirection.BUY if decision.direction == 'BUY' else TradeDirection.SELL
        
        # Validate quantiles - they should be small price movements (typically < 0.1 for forex)
        # If quantiles are unrealistic, use None to force volatility-based calculation
        if quantiles is not None:
            max_quantile = np.max(np.abs(quantiles))
            if max_quantile > 1.0:  # Unrealistic quantiles from untrained model
                logger.debug(f"Invalid quantiles detected (max={max_quantile:.2f}), using ATR-based SL/TP")
                quantiles = None
        
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
        rr_eps = 1e-9
        if (float(sltp_result.risk_reward_ratio) + rr_eps) < float(self.config.min_risk_reward):
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
            pair=pair,
            volatility=volatility,
            direction_confidence=confidence
        )
        
        decision.position_size = size_result.position_size
        decision.position_units = size_result.units
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
            meta_features_dict = self._build_meta_features_dict(predictions, market_data, decision)
            
            filter_result = self.trade_filter.filter(
                signal=decision.direction,
                features=meta_features_dict,
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

            # SCALP profile hierarchy:
            # - H1 is HTF main trend (hard filter)
            # - M15 is entry TF (handled in strategy.on_bar)
            # - M5 is feature TF (model input)
            if self.config.profile == 'SCALP':
                if (not bool(mtf_result.get('higher_tf_aligned', False))) and float(mtf_result.get('alignment', 0.0) or 0.0) < 0.95:
                    decision.rejection_reasons.append("HTF not aligned (higher_tf_aligned=False)")
                    return decision

                tf_dirs = mtf_result.get('timeframe_directions') or {}
                h1_dir = tf_dirs.get('H1')
                if h1_dir is not None:
                    try:
                        h1_dir = int(h1_dir)
                    except Exception:
                        h1_dir = None

                if h1_dir is not None:
                    if decision.direction == 'BUY' and h1_dir < 0:
                        decision.rejection_reasons.append(
                            f"HTF(H1) counter-trend: dir={h1_dir}"
                        )
                        return decision
                    if decision.direction == 'SELL' and h1_dir > 0:
                        decision.rejection_reasons.append(
                            f"HTF(H1) counter-trend: dir={h1_dir}"
                        )
                        return decision

                    # If H1 is sideways/neutral, allow only when overall MTF alignment is very strong.
                    if h1_dir == 0 and float(mtf_result.get('alignment', 0.0) or 0.0) < 0.90:
                        decision.rejection_reasons.append(
                            f"HTF(H1) neutral: dir=0"
                        )
                        return decision
        
        # =================================================================
        # Phase 5 FINAL CHECK: Capital Protection Size Adjustment
        # =================================================================
        if self.capital_protector:
            final_check = self.capital_protector.check_trade(
                proposed_size=decision.position_size,
                account_balance=account_balance,
                current_time=current_time,
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
            f"MTF: {decision.mtf_alignment:.2f} | "
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
        """Build feature dictionary for meta-labeling.

        The TradeFilter API expects a dict-like structure. This method returns
        scalar features plus optional TP-before-SL probabilities when available.
        """
        feat = self._build_meta_features_dict(predictions, market_data, decision)

        # Stable numeric vector (tests rely on index 0 being direction_confidence)
        order = [
            'direction_confidence',
            'volatility',
            'atr',
            'risk_reward_ratio',
            'quantile_width',
            'recent_return_mean',
            'recent_return_std',
            'p_long',
            'p_short',
        ]
        values = [float(feat.get(k, 0.0) or 0.0) for k in order]
        return np.array(values, dtype=np.float32)

    def _build_meta_features_dict(
        self,
        predictions: Dict,
        market_data: pd.DataFrame,
        decision: TradeDecision
    ) -> Dict[str, float]:
        feat: Dict[str, float] = {}

        feat['direction_confidence'] = float(decision.direction_confidence)
        feat['volatility'] = float(decision.volatility if decision.volatility else 0.0)
        feat['atr'] = float(decision.atr if decision.atr else 0.0)
        feat['risk_reward_ratio'] = float(decision.risk_reward_ratio)

        # Trade-objective probabilities (if present)
        p_long = predictions.get('p_long')
        p_short = predictions.get('p_short')
        if p_long is not None:
            try:
                feat['p_long'] = float(np.array(p_long).flatten()[0])
            except Exception:
                feat['p_long'] = 0.0
        if p_short is not None:
            try:
                feat['p_short'] = float(np.array(p_short).flatten()[0])
            except Exception:
                feat['p_short'] = 0.0

        # Quantiles spread
        quantiles = predictions.get('quantiles')
        if quantiles is not None and isinstance(quantiles, np.ndarray):
            q_flat = quantiles.flatten()
            if q_flat.size >= 2:
                feat['quantile_width'] = float(q_flat[-1] - q_flat[0])
            else:
                feat['quantile_width'] = 0.0
        else:
            feat['quantile_width'] = 0.0

        # Recent returns
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().iloc[-5:]
            feat['recent_return_mean'] = float(returns.mean() if len(returns) > 0 else 0.0)
            feat['recent_return_std'] = float(returns.std() if len(returns) > 0 else 0.0)
        else:
            feat['recent_return_mean'] = 0.0
            feat['recent_return_std'] = 0.0

        return feat
    
    def _check_mtf_alignment(
        self,
        direction: str,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Check multi-timeframe trend alignment."""
        if not self.mtf_detector or not mtf_data:
            return {'alignment': 1.0, 'trend': 'unknown'}
        
        try:
            result = self.mtf_detector.detect(mtf_data, compute_ml_features=False)
            alignment = float(getattr(result, 'mtf_alignment', 1.0) or 0.0)
            trend = str(getattr(result, 'direction', '') or getattr(result, 'trend_name', '') or 'unknown')
            tf_dirs = getattr(result, 'timeframe_directions', None)
            if tf_dirs is None:
                tf_dirs = {}
            higher_tf_aligned = bool(getattr(result, 'higher_tf_aligned', False))
            return {
                'alignment': alignment,
                'trend': trend,
                'timeframe_directions': tf_dirs,
                'higher_tf_aligned': higher_tf_aligned,
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
