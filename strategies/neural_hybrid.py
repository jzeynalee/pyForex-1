# strategies/neural_hybrid.py
"""
Neural Hybrid Strategy with Full Risk Management (v2)

Integrates ALL 5 PHASES:
- Phase 1: TCN/ViT/YOLO prediction pipeline
- Phase 2: Risk calculations (via decision engine)
- Phase 3: Meta-labeling (via decision engine)
- Phase 4: Exit advisor integration hooks
- Phase 5: Capital protection integration

This is the main trading strategy for pyForex.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import logging

from utils.candle_to_image import candle_image

# Core imports
from inference.predictor import (
    RiskAwareTCNPredictor, HybridPredictor, PredictionResult,
    create_predictor
)
from trading.decision_engine import (
    EnhancedDecisionEngine, DecisionEngineConfig, TradeDecision
)

# Risk management imports - Phases 1-3
try:
    from risk_management import (
        MetaLabelingModel, TradeFilter,
        TripleBarrierLabeler, TripleBarrierConfig
    )
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False

# Phase 4: Exit Advisor
try:
    from risk_management import ExitAdvisor, ExitAction, Position
    HAS_EXIT_ADVISOR = True
except ImportError:
    HAS_EXIT_ADVISOR = False
    ExitAdvisor = None

# Phase 5: Capital Protection
try:
    from risk_management import (
        CapitalProtector, ProtectionConfig, ProtectionManager,
        ProtectionLevel, ProtectionAction
    )
    HAS_CAPITAL_PROTECTION = True
except ImportError:
    HAS_CAPITAL_PROTECTION = False
    CapitalProtector = None

# MTF imports
try:
    from utils.mtf_config import get_profile, MTFProfile
    from trend_detection.mtf_trend_detector import MTFTrendDetector
    HAS_MTF = True
except ImportError:
    HAS_MTF = False

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Order:
    """Trade order structure."""
    symbol: str
    order_type: OrderType
    direction: str           # 'BUY' or 'SELL'
    volume: float           # Lots
    price: float            # Entry price (0 for market)
    stop_loss: float
    take_profit: float
    comment: str = ""
    magic_number: int = 123456
    ticket: Optional[str] = None
    
    # Risk info
    risk_percent: float = 0.0
    risk_reward: float = 0.0
    confidence: float = 0.0
    meta_score: float = 0.0
    protection_level: str = 'normal'


@dataclass
class OpenPosition:
    """Tracked open position."""
    ticket: str
    direction: int          # 1 = long, -1 = short
    entry_price: float
    entry_time: datetime
    volume: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0


@dataclass
class StrategyConfig:
    """Configuration for neural hybrid strategy."""
    # Trading profile
    profile: str = 'INTRADAY'
    symbol: str = 'EURUSD'
    
    # Model paths
    tcn_weights: str = 'models/weights/tcn_best.pt'
    vit_weights: Optional[str] = 'models/weights/vit_best.pt'
    yolo_weights: Optional[str] = 'models/weights/yolo_patterns.pt'
    fusion_weights: Optional[str] = 'models/weights/fusion_best.pt'
    meta_model_path: Optional[str] = 'models/weights/meta_model.joblib'
    exit_model_path: Optional[str] = None  # NEW: Phase 4
    
    # Feature settings
    sequence_length: int = 60
    use_vision: bool = True
    use_yolo: bool = True
    
    # Risk settings
    base_risk_percent: float = 1.0
    min_risk_reward: float = 1.5
    max_open_trades: int = 3
    
    # Confidence thresholds
    min_direction_confidence: float = 0.55
    min_meta_score: float = 0.5
    min_mtf_alignment: float = 0.6

    # SCALP-only controls (TTF=M15)
    cooldown_after_loss_m15: int = 0
    max_holding_m15: int = 0
    time_exit_profit_pips: float = 0.0
    time_exit_cut_loss_pips: float = 0.0
    
    # Exit advisor settings (Phase 4)
    enable_exit_advisor: bool = True
    exit_confidence_threshold: float = 0.6
    
    # Capital protection settings (Phase 5)
    enable_capital_protection: bool = True
    max_daily_loss_percent: float = 3.0
    max_weekly_loss_percent: float = 6.0
    max_drawdown_percent: float = 10.0
    
    # Limits
    max_daily_trades: int = 10
    max_daily_loss: float = 500.0

    # Hard rules controls
    avoid_rollover: bool = True

    def __post_init__(self):
        self.profile = str(self.profile or 'INTRADAY').upper()

        # SCALP profile tightening defaults
        if self.profile == 'SCALP':
            # Stronger MTF requirement
            self.min_mtf_alignment = max(float(self.min_mtf_alignment), 0.70)

            # Reduce trade frequency and exposure
            self.max_open_trades = min(int(self.max_open_trades), 1)
            self.max_daily_trades = min(int(self.max_daily_trades), 10)

            # Cooldown after a losing trade (in M15 candles)
            self.cooldown_after_loss_m15 = max(int(self.cooldown_after_loss_m15), 1)

            try:
                self.max_holding_m15 = int(self.max_holding_m15)
            except Exception:
                self.max_holding_m15 = 0
            if self.max_holding_m15 <= 0:
                self.max_holding_m15 = 8

            try:
                self.time_exit_profit_pips = float(self.time_exit_profit_pips)
            except Exception:
                self.time_exit_profit_pips = 0.0
            if self.time_exit_profit_pips <= 0.0:
                self.time_exit_profit_pips = 6.0

            try:
                self.time_exit_cut_loss_pips = float(self.time_exit_cut_loss_pips)
            except Exception:
                self.time_exit_cut_loss_pips = 0.0
            if self.time_exit_cut_loss_pips <= 0.0:
                self.time_exit_cut_loss_pips = 8.0

            try:
                self.base_risk_percent = min(float(self.base_risk_percent), 0.5)
            except Exception:
                self.base_risk_percent = 0.5


class NeuralHybridStrategy:
    """
    Neural hybrid strategy with full risk management (v2).
    
    New in v2:
    - Phase 4: Exit advisor for active position management
    - Phase 5: Capital protection integration
    
    Pipeline:
        1. Fetch market data
        2. Generate features
        3. Get predictions (TCN + Vision)
        4. Evaluate with decision engine (Phases 1-3)
           - If available, include TP-before-SL probabilities (p_long/p_short)
        5. Apply capital protection (Phase 5)
        6. Create and execute order
        7. Monitor positions with exit advisor (Phase 4)
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        data_provider = None,
        executor = None,
        **kwargs
    ):
        self.config = config or StrategyConfig()
        self.data_provider = data_provider
        self.executor = executor
        
        # Handle extra args like risk_manager if passed
        if 'risk_manager' in kwargs:
            # We currently use internal decision_engine which wraps risk logic
            # but we could store this if needed
            pass
            
        self.name = "NeuralHybridStrategy"
        
        # Models
        self.predictor: Optional[HybridPredictor] = None
        self.decision_engine: Optional[EnhancedDecisionEngine] = None
        self.meta_model: Optional[MetaLabelingModel] = None
        self.exit_advisor: Optional[ExitAdvisor] = None  # Phase 4
        self.capital_protector: Optional[CapitalProtector] = None  # Phase 5
        self.mtf_detector = None
        
        # State
        self._initialized = False
        self._starting_balance = 0.0
        self._current_balance = 0.0
        self._open_positions: Dict[str, OpenPosition] = {}
        
        # Daily tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_wins = 0
        self._daily_losses = 0

        self._last_daily_reset_day: Optional[date] = None
        
        # Trade history
        self._trade_history: List[Dict] = []

        # Backtest adapter state
        self._last_processed_trade_count = 0

        # SCALP entry gating (TTF=M15)
        self._last_ttf_close_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        
        logger.info(f"NeuralHybridStrategy v2 created for {self.config.symbol}")

    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Process a closed bar in an event-driven runner.

        This is a thin adapter around :meth:`evaluate` for compatibility with
        the project-wide :class:`~strategies.base.Strategy` interface used by
        :class:`~trading.bot.BacktestBot`.

        Args:
            df: Recent OHLCV window ending at the latest closed bar.

        Returns:
            Signal string ('BUY' or 'SELL') if a trade is executed, otherwise None.
        """
        if df is None or df.empty:
            return None

        if not self._initialized:
            starting_balance = 10000.0
            if self.executor is not None:
                starting_balance = float(getattr(self.executor, 'balance', starting_balance))
            if not self.initialize(starting_balance=starting_balance):
                return None

        self._process_new_closed_trades()
        self._sync_open_positions_from_executor()

        try:
            current_time = None
            if 'time' in df.columns:
                try:
                    current_time = pd.to_datetime(df['time'].iloc[-1]).to_pydatetime()
                except Exception:
                    current_time = None

            if current_time is not None:
                try:
                    bar_day = current_time.date()
                except Exception:
                    bar_day = None
                if bar_day is not None and bar_day != self._last_daily_reset_day:
                    self.reset_daily_stats()
                    self._last_daily_reset_day = bar_day

            if (
                self.config.profile == 'SCALP'
                and current_time is not None
                and (float(getattr(self.config, 'time_exit_cut_loss_pips', 0.0) or 0.0) > 0.0
                     or (int(getattr(self.config, 'max_holding_m15', 0) or 0) > 0
                         and float(getattr(self.config, 'time_exit_profit_pips', 0.0) or 0.0) > 0.0))
                and self.executor is not None
                and hasattr(self.executor, 'close_position')
                and len(self._open_positions) > 0
            ):
                x_pips = float(getattr(self.config, 'time_exit_profit_pips', 0.0) or 0.0)
                y_pips = float(getattr(self.config, 'time_exit_cut_loss_pips', 0.0) or 0.0)
                max_minutes = 15 * int(getattr(self.config, 'max_holding_m15', 0) or 0)

                bar_high = None
                bar_low = None
                try:
                    bar_high = float(df['high'].iloc[-1])
                    bar_low = float(df['low'].iloc[-1])
                except Exception:
                    bar_high = None
                    bar_low = None

                current_price = None
                try:
                    current_price = float(df['close'].iloc[-1])
                except Exception:
                    current_price = None
                for ticket, p in list(self._open_positions.items()):
                    try:
                        held_minutes = (current_time - p.entry_time).total_seconds() / 60.0
                    except Exception:
                        held_minutes = 0.0
                    pips = 0.0
                    if current_price is not None:
                        try:
                            if int(p.direction) > 0:
                                pips = (current_price - float(p.entry_price)) / 0.0001
                            else:
                                pips = (float(p.entry_price) - current_price) / 0.0001
                        except Exception:
                            pips = 0.0

                    # Cut-loss: apply at any time.
                    if y_pips > 0.0:
                        y = abs(float(y_pips))
                        stop_price = None
                        try:
                            if int(p.direction) > 0:
                                stop_price = float(p.entry_price) - (y * 0.0001)
                            else:
                                stop_price = float(p.entry_price) + (y * 0.0001)
                        except Exception:
                            stop_price = None

                        hit_stop = False
                        if stop_price is not None and bar_high is not None and bar_low is not None:
                            if int(p.direction) > 0:
                                hit_stop = bool(bar_low <= stop_price)
                            else:
                                hit_stop = bool(bar_high >= stop_price)
                        else:
                            hit_stop = bool(pips <= -y)

                        if hit_stop:
                            try:
                                if stop_price is not None and hasattr(self.executor, 'close_position_at'):
                                    self.executor.close_position_at(int(ticket), stop_price, reason='CLOSED_MANUAL')
                                else:
                                    self.executor.close_position(int(ticket), reason='CLOSED_MANUAL')
                            except Exception:
                                pass
                            continue

                    # Profit-aware time exit: only after max hold.
                    if max_minutes > 0 and held_minutes >= float(max_minutes):
                        if x_pips > 0.0:
                            x = abs(float(x_pips))
                            if pips >= x:
                                try:
                                    self.executor.close_position(int(ticket), reason='CLOSED_MANUAL')
                                except Exception:
                                    pass

                # Process any newly-closed positions immediately (cooldown, stats, etc.)
                self._process_new_closed_trades()
                self._sync_open_positions_from_executor()

            # SCALP plan: M5=LTF features, but evaluate every base bar (e.g., M5).
            if self.config.profile == 'SCALP' and current_time is not None:
                if self._cooldown_until is not None and current_time < self._cooldown_until:
                    return None
                if self._last_ttf_close_time is not None and current_time <= self._last_ttf_close_time:
                    return None
                self._last_ttf_close_time = current_time

            # Enforce daily/open-position limits in the backtest on_bar path.
            if not self._check_daily_limits():
                return None
            if len(self._open_positions) >= self.config.max_open_trades:
                return None

            # LTF features always come from the incoming window (typically M5 in backtests)
            ltf_data = df
            if len(ltf_data) < self.config.sequence_length:
                return None

            # Trading timeframe context (TTF) drives decision engine metrics (ATR/vol/SLTP etc.)
            market_data = ltf_data
            if self.config.profile == 'SCALP' and self.data_provider is not None:
                try:
                    market_data = self.data_provider.get_ohlcv(
                        self.config.symbol,
                        timeframe='M15',
                        count=self.config.sequence_length + 50,
                    )
                except Exception:
                    market_data = ltf_data

            if market_data is None or len(market_data) < self.config.sequence_length:
                return None

            entry_price = float(market_data['close'].iloc[-1])

            features = self._prepare_features(ltf_data)
            if features is None or np.isnan(features).any():
                return None

            chart_image = None
            if self.config.use_vision:
                chart_image = self._generate_chart_image(market_data)

            if isinstance(self.predictor, HybridPredictor):
                prediction = self.predictor.predict(features, chart_image)
            else:
                prediction = self.predictor.predict(features)

            predictions = {
                'direction_probs': prediction.probabilities,
                'volatility': prediction.volatility,
                'quantiles': prediction.quantiles,
                'features': prediction.features
            }

            if getattr(prediction, 'p_long', None) is not None:
                predictions['p_long'] = prediction.p_long
            if getattr(prediction, 'p_short', None) is not None:
                predictions['p_short'] = prediction.p_short

            mtf_data = None
            if self.mtf_detector:
                mtf_data = self._fetch_mtf_data()

            decision = self.decision_engine.evaluate(
                predictions=predictions,
                entry_price=entry_price,
                pair=self.config.symbol,
                account_balance=self._get_account_balance(),
                market_data=market_data,
                current_spread=None,
                current_time=current_time,
                mtf_data=mtf_data,
            )

            order = self.create_order(decision)
            if order is None:
                return None

            if self.execute(order):
                return order.direction

            return None

        except Exception as e:
            logger.error(f"on_bar error: {e}", exc_info=True)
            return None
    
    def initialize(self, starting_balance: Optional[float] = None) -> bool:
        """Initialize strategy components."""
        try:
            # Get starting balance
            if starting_balance is not None:
                self._starting_balance = starting_balance
            elif self.executor:
                self._starting_balance = self.executor.get_account_balance()
            else:
                self._starting_balance = 10000.0
            
            self._current_balance = self._starting_balance
            
            # Initialize predictor
            self.predictor = create_predictor(
                profile=self.config.profile,
                weights_path=self.config.tcn_weights,
                use_vision=self.config.use_vision,
                use_yolo=self.config.use_yolo
            )
            
            # Load meta-model if available
            self.meta_model = None
            if HAS_RISK_MANAGEMENT and self.config.meta_model_path:
                try:
                    self.meta_model = MetaLabelingModel.load(self.config.meta_model_path)
                    logger.info("Meta-labeling model loaded")
                except Exception as e:
                    logger.warning(f"Could not load meta-model: {e}")
            
            # Initialize decision engine with Phase 5 config
            engine_config = DecisionEngineConfig(
                profile=self.config.profile,
                min_direction_confidence=self.config.min_direction_confidence,
                min_meta_score=self.config.min_meta_score,
                min_mtf_alignment=self.config.min_mtf_alignment,
                base_risk_percent=self.config.base_risk_percent,
                min_risk_reward=self.config.min_risk_reward,
                enable_capital_protection=self.config.enable_capital_protection,
                max_daily_loss_pct=self.config.max_daily_loss_percent,
                max_weekly_loss_pct=self.config.max_weekly_loss_percent,
                max_drawdown_pct=self.config.max_drawdown_percent,
                avoid_rollover=bool(getattr(self.config, 'avoid_rollover', True)),
            )
            
            self.decision_engine = EnhancedDecisionEngine(
                config=engine_config,
                meta_model=self.meta_model
            )
            
            # Initialize decision engine with balance
            self.decision_engine.initialize(self._starting_balance)
            
            # Initialize Phase 4: Exit Advisor
            if self.config.enable_exit_advisor and HAS_EXIT_ADVISOR:
                if self.config.exit_model_path:
                    try:
                        self.exit_advisor = ExitAdvisor.load(self.config.exit_model_path)
                        logger.info("Phase 4: Exit Advisor loaded")
                    except Exception as e:
                        logger.warning(f"Could not load Exit Advisor: {e}")
            
            # Reference to decision engine's capital protector (Phase 5)
            self.capital_protector = self.decision_engine.capital_protector
            
            # Initialize MTF detector
            if HAS_MTF:
                try:
                    self.mtf_detector = MTFTrendDetector(profile=get_profile(self.config.profile))
                    logger.info("MTF detector initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize MTF detector: {e}")
            
            self._initialized = True
            logger.info(
                f"Strategy v2 initialized for {self.config.symbol} ({self.config.profile}) "
                f"with balance: {self._starting_balance:.2f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            return False
    
    def evaluate(
        self,
        current_time: Optional[datetime] = None
    ) -> Optional[TradeDecision]:
        """
        Evaluate current market conditions for trading opportunity.
        
        Returns:
            TradeDecision if opportunity found, None otherwise
        """
        if not self._initialized:
            logger.error("Strategy not initialized")
            return None
        
        current_time = current_time or datetime.utcnow()
        
        # Check daily limits
        if not self._check_daily_limits():
            return None
        
        # Check max open trades
        if len(self._open_positions) >= self.config.max_open_trades:
            logger.debug("Max open trades reached")
            return None
        
        try:
            # Fetch data
            market_data = self._fetch_data()
            if market_data is None or len(market_data) < self.config.sequence_length:
                logger.warning("Insufficient data")
                return None

            try:
                window = market_data.tail(int(self.config.sequence_length)).copy()
                window.columns = [str(c).lower() for c in window.columns]
                required = ['open', 'high', 'low', 'close']
                if set(required).issubset(set(window.columns)):
                    cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 'real_volume'] if c in window.columns]
                    if cols and window[cols].isna().any().any():
                        return None
            except Exception:
                pass
            
            # Get current price and spread
            entry_price = float(market_data['close'].iloc[-1])
            current_spread = self._get_spread()
            
            # Generate features
            features = self._prepare_features(market_data)

            if features is None or np.isnan(features).any():
                logger.warning("Invalid features (NaNs) - skipping inference")
                return None
            
            # Get chart image if using vision
            chart_image = None
            if self.config.use_vision:
                chart_image = self._generate_chart_image(market_data)
            
            # Get predictions
            if isinstance(self.predictor, HybridPredictor):
                prediction = self.predictor.predict(features, chart_image)
            else:
                prediction = self.predictor.predict(features)
            
            # Convert to dict format for decision engine
            predictions = {
                'direction_probs': prediction.probabilities,
                'volatility': prediction.volatility,
                'quantiles': prediction.quantiles,
                'features': prediction.features
            }

            if getattr(prediction, 'p_long', None) is not None:
                predictions['p_long'] = prediction.p_long
            if getattr(prediction, 'p_short', None) is not None:
                predictions['p_short'] = prediction.p_short
            
            # Fetch MTF data if available
            mtf_data = None
            if self.mtf_detector:
                mtf_data = self._fetch_mtf_data()
            
            # Get account balance
            account_balance = self._get_account_balance()
            
            # Evaluate with decision engine (includes Phase 5 protection)
            decision = self.decision_engine.evaluate(
                predictions=predictions,
                entry_price=entry_price,
                pair=self.config.symbol,
                account_balance=account_balance,
                market_data=market_data,
                current_spread=current_spread,
                current_time=current_time,
                mtf_data=mtf_data
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)
            return None
    
    def check_position_for_exit(
        self,
        ticket: str,
        market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Check if a position should be exited (Phase 4).
        
        Args:
            ticket: Position ticket
            market_data: Recent market data
        
        Returns:
            Exit recommendation dict or None
        """
        if not self.exit_advisor:
            return None
        
        if ticket not in self._open_positions:
            return None
        
        pos = self._open_positions[ticket]
        current_price = float(market_data['close'].iloc[-1])
        
        # Create Position object for advisor
        rl_position = Position(
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_time=0,
            initial_size=pos.volume,
            current_size=pos.volume,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            initial_sl=pos.stop_loss,
            initial_tp=pos.take_profit
        )
        
        # Get recommendation
        recommendation = self.exit_advisor.get_recommendation(
            rl_position,
            market_data,
            deterministic=True
        )
        
        # Check if exit is recommended
        if recommendation['action'] == ExitAction.EXIT:
            if recommendation['confidence'] >= self.config.exit_confidence_threshold:
                return {
                    'ticket': ticket,
                    'action': 'EXIT',
                    'confidence': recommendation['confidence'],
                    'reason': 'rl_advisor'
                }
        
        elif recommendation['action'] == ExitAction.TRAIL_STOP:
            return {
                'ticket': ticket,
                'action': 'TRAIL_STOP',
                'confidence': recommendation['confidence']
            }
        
        elif recommendation['action'] in [ExitAction.PARTIAL_25, ExitAction.PARTIAL_50, ExitAction.PARTIAL_75]:
            if recommendation['confidence'] >= self.config.exit_confidence_threshold:
                fraction = {
                    ExitAction.PARTIAL_25: 0.25,
                    ExitAction.PARTIAL_50: 0.50,
                    ExitAction.PARTIAL_75: 0.75
                }.get(recommendation['action'], 0.5)
                return {
                    'ticket': ticket,
                    'action': 'PARTIAL',
                    'fraction': fraction,
                    'confidence': recommendation['confidence']
                }
        
        return None
    
    def check_all_positions(self, market_data: pd.DataFrame) -> List[Dict]:
        """Check all open positions for exit recommendations."""
        recommendations = []
        for ticket in self._open_positions:
            rec = self.check_position_for_exit(ticket, market_data)
            if rec:
                recommendations.append(rec)
        return recommendations
    
    def create_order(self, decision: TradeDecision) -> Optional[Order]:
        """Create order from trade decision."""
        if not decision.should_trade:
            return None
        
        return Order(
            symbol=self.config.symbol,
            order_type=OrderType.MARKET,
            direction=decision.direction,
            volume=decision.position_size,
            price=0.0,  # Market order
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            comment=f"NeuralHybrid_{self.config.profile}",
            risk_percent=decision.risk_percent,
            risk_reward=decision.risk_reward_ratio,
            confidence=decision.direction_confidence,
            meta_score=decision.meta_score,
            protection_level=decision.protection_level
        )
    
    def execute(self, order: Order) -> bool:
        """Execute order through executor.

        Supports both live executors (MT5-style ``execute_order``) and the
        backtesting executor (``entry``).
        """
        if not self.executor:
            logger.warning("No executor configured")
            return False
        
        try:
            if hasattr(self.executor, 'execute_order'):
                result = self.executor.execute_order(
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    direction=order.direction,
                    volume=order.volume,
                    price=order.price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    comment=order.comment,
                    magic_number=order.magic_number
                )
            elif hasattr(self.executor, 'entry'):
                result = self.executor.entry(
                    signal=order.direction,
                    volume=order.volume,
                    sl=order.stop_loss,
                    tp=order.take_profit,
                )
            else:
                logger.warning("Executor does not support execute_order or entry")
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def on_trade_closed(self, ticket: str, pnl: float, close_time: Optional[datetime] = None):
        """Handle trade close event."""
        # Remove from open positions
        pos = self._open_positions.pop(ticket, None)
        
        # Update daily stats
        self._daily_pnl += pnl
        self._current_balance += pnl
        if pnl >= 0:
            self._daily_wins += 1
        else:
            self._daily_losses += 1

        # SCALP cooldown after losses (measured in M15 candles)
        if pnl < 0 and self.config.profile == 'SCALP' and self.config.cooldown_after_loss_m15 > 0:
            ct = close_time or datetime.utcnow()
            self._cooldown_until = ct + timedelta(minutes=15 * int(self.config.cooldown_after_loss_m15))
        
        # Update decision engine's capital protection
        if self.decision_engine:
            try:
                self.decision_engine.record_trade_result(
                    pnl=float(pnl),
                    is_win=bool(pnl > 0),
                    timestamp=close_time,
                )
            except Exception:
                pass
        
        # Record in history
        self._trade_history.append({
            'ticket': ticket,
            'pnl': pnl,
            'is_win': pnl >= 0,
            'close_time': datetime.utcnow().isoformat(),
            'balance': self._current_balance
        })
        
        logger.info(f"Trade closed: {ticket} | PnL: {pnl:.2f} | Balance: {self._current_balance:.2f}")

    def _process_new_closed_trades(self):
        """Process newly closed trades from the configured executor (backtest mode)."""
        if self.executor is None or not hasattr(self.executor, 'get_trade_history'):
            return

        try:
            history = self.executor.get_trade_history() or []
        except Exception:
            return

        new_trades = history[self._last_processed_trade_count:]
        self._last_processed_trade_count = len(history)

        for t in new_trades:
            try:
                ticket = str(t.get('ticket'))
                pnl = t.get('pnl')
                if pnl is None:
                    continue
                ct = None
                for k in ('exit_time', 'close_time', 'time'):
                    if k in t and t.get(k) is not None:
                        try:
                            ct = pd.to_datetime(t.get(k)).to_pydatetime()
                            break
                        except Exception:
                            ct = None
                self.on_trade_closed(ticket=ticket, pnl=float(pnl), close_time=ct)
            except Exception:
                continue

    def _sync_open_positions_from_executor(self):
        """Sync internal open-position tracking from executor state (backtest mode)."""
        if self.executor is None or not hasattr(self.executor, 'get_open_positions'):
            return

        try:
            open_positions = self.executor.get_open_positions() or []
        except Exception:
            return

        next_positions: Dict[str, OpenPosition] = {}
        for p in open_positions:
            try:
                ticket = str(p.get('ticket'))
                direction_str = str(p.get('type', '')).upper()
                direction = 1 if direction_str == 'BUY' else -1
                et = None
                if p.get('entry_time') is not None:
                    try:
                        et = pd.to_datetime(p.get('entry_time')).to_pydatetime()
                    except Exception:
                        et = None
                next_positions[ticket] = OpenPosition(
                    ticket=ticket,
                    direction=direction,
                    entry_price=float(p.get('price_open', 0.0) or 0.0),
                    entry_time=et or datetime.utcnow(),
                    volume=float(p.get('volume', 0.0) or 0.0),
                    stop_loss=float(p.get('sl', 0.0) or 0.0),
                    take_profit=float(p.get('tp', 0.0) or 0.0),
                    unrealized_pnl=float(p.get('profit', 0.0) or 0.0),
                )
            except Exception:
                continue

        self._open_positions = next_positions
    
    def get_protection_status(self) -> Dict:
        """Get capital protection status."""
        if self.decision_engine:
            return self.decision_engine.get_protection_status()
        return {'enabled': False}
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_wins = 0
        self._daily_losses = 0
        
        if self.decision_engine:
            self.decision_engine.reset_daily_protection()
        
        logger.info("Daily stats reset")
    
    def get_stats(self) -> Dict:
        """Get current strategy statistics."""
        total_trades = self._daily_wins + self._daily_losses
        win_rate = self._daily_wins / total_trades if total_trades > 0 else 0
        
        return {
            'symbol': self.config.symbol,
            'profile': self.config.profile,
            'initialized': self._initialized,
            'starting_balance': self._starting_balance,
            'current_balance': self._current_balance,
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'daily_wins': self._daily_wins,
            'daily_losses': self._daily_losses,
            'daily_win_rate': win_rate,
            'win_rate': win_rate,
            'open_positions': len(self._open_positions),
            'protection_status': self.get_protection_status()
        }

    def get_open_positions(self) -> List[Dict]:
        """Get list of currently tracked open positions."""
        positions: List[Dict] = []
        for ticket, p in self._open_positions.items():
            positions.append({
                'ticket': str(ticket),
                'direction': int(p.direction),
                'entry_price': float(p.entry_price),
                'entry_time': p.entry_time,
                'volume': float(p.volume),
                'sl': float(p.stop_loss),
                'tp': float(p.take_profit),
                'profit': float(p.unrealized_pnl),
            })
        return positions
    
    def _check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        if self._daily_trades >= self.config.max_daily_trades:
            logger.debug("Daily trade limit reached")
            return False
        
        if self._daily_pnl <= -self.config.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
        
        return True
    
    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch market data."""
        if not self.data_provider:
            logger.warning("No data provider")
            return None
        
        try:
            return self.data_provider.get_ohlcv(
                self.config.symbol,
                count=self.config.sequence_length + 50
            )
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None
    
    def _fetch_mtf_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch multi-timeframe data."""
        if not self.data_provider:
            return None
        
        try:
            if HAS_MTF:
                profile = get_profile(self.config.profile)
                timeframes = list(profile.timeframe_strings)
                candle_counts = dict(getattr(profile, 'candle_counts', {}) or {})
            else:
                timeframes = ['M5', 'M15', 'H1']
                candle_counts = {}

            return {
                tf: self.data_provider.get_ohlcv(
                    self.config.symbol,
                    timeframe=tf,
                    count=int(candle_counts.get(tf, 200))
                )
                for tf in timeframes
            }
        except Exception as e:
            logger.error(f"MTF data fetch error: {e}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction using FeatureEngineer.

        If the loaded TCN expects a 5-feature input (legacy/simple checkpoints),
        this function uses raw OHLCV as the feature tensor. If the input CSV does
        not contain a 'volume' column, it will fall back to 'tick_volume' or
        'real_volume' (or zeros if none are present).
        """
        # Get expected feature columns and count from predictor.
        expected_feature_cols = None
        expected_feature_count = 5  # Default

        if hasattr(self.predictor, '_feature_names') and getattr(self.predictor, '_feature_names', None):
            expected_feature_cols = self.predictor._feature_names
            expected_feature_count = len(expected_feature_cols)

        # If feature names are not available, try to infer the expected input
        # dimensionality from the loaded TCN model.
        if expected_feature_cols is None:
            try:
                tcn_pred = self.predictor
                if hasattr(self.predictor, 'tcn_predictor'):
                    tcn_pred = self.predictor.tcn_predictor

                if hasattr(tcn_pred, 'model'):
                    model = tcn_pred.model
                    if hasattr(model, 'config') and hasattr(model.config, 'input_channels'):
                        expected_feature_count = int(model.config.input_channels)
                    elif hasattr(model, 'tcn') and hasattr(model.tcn, 'input_dim'):
                        expected_feature_count = int(model.tcn.input_dim)
            except Exception:
                pass

        if expected_feature_count == 5:
            vol_col = None
            for c in ('volume', 'tick_volume', 'real_volume'):
                if c in data.columns:
                    vol_col = c
                    break

            if vol_col is None:
                vol = np.zeros((len(data),), dtype=np.float32)
            else:
                vol = data[vol_col].to_numpy(dtype=np.float32, copy=False)

            ohlc = data[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32, copy=False)
            ohlcv = np.column_stack([ohlc, vol])[-self.config.sequence_length:]
            return np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Always use FeatureEngineer to generate features (same as training)
        try:
            from utils.features_engineering import FeatureEngineer
            fe = FeatureEngineer()
            data_with_features = fe.generate_features(data.copy())
            
            # Select features based on checkpoint or generate matching set
            if expected_feature_cols:
                # Use exact feature columns from checkpoint
                available_cols = [c for c in expected_feature_cols if c in data_with_features.columns]
                missing_cols = [c for c in expected_feature_cols if c not in data_with_features.columns]
                
                if missing_cols:
                    logger.debug(f"Missing {len(missing_cols)} features, will pad with zeros")
                
                # Get available features
                features = data_with_features[available_cols].values[-self.config.sequence_length:]
                
                # Pad missing features with zeros to match expected count
                if len(available_cols) < expected_feature_count:
                    padding = np.zeros((features.shape[0], expected_feature_count - len(available_cols)))
                    features = np.hstack([features, padding])
            else:
                # No checkpoint feature list - use all numeric features
                exclude_cols = ['time', 'timestamp', 'date', 'tick_volume', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']
                numeric_cols = data_with_features.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                # Limit to expected count
                if len(feature_cols) > expected_feature_count:
                    feature_cols = feature_cols[:expected_feature_count]
                
                features = data_with_features[feature_cols].values[-self.config.sequence_length:]
                
                # Pad if needed
                if features.shape[1] < expected_feature_count:
                    padding = np.zeros((features.shape[0], expected_feature_count - features.shape[1]))
                    features = np.hstack([features, padding])
            
            # Handle NaNs
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}, using OHLCV fallback")
        
        # Fallback to OHLCV with padding to expected features
        vol_col = None
        for c in ('volume', 'tick_volume', 'real_volume'):
            if c in data.columns:
                vol_col = c
                break
        if vol_col is None:
            vol = np.zeros((len(data),), dtype=np.float32)
        else:
            vol = data[vol_col].to_numpy(dtype=np.float32, copy=False)

        ohlc = data[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32, copy=False)
        ohlcv = np.column_stack([ohlc, vol])[-self.config.sequence_length:]
        padding = np.zeros((ohlcv.shape[0], expected_feature_count - 5))
        return np.hstack([ohlcv, padding]).astype(np.float32)
    
    def _generate_chart_image(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate chart image for vision models."""
        try:
            if data is None or data.empty:
                return None

            cols = {'open', 'high', 'low', 'close'}
            if not cols.issubset(set(map(str.lower, data.columns))):
                return None

            window = data.tail(int(self.config.sequence_length)).copy()
            window.columns = [str(c).lower() for c in window.columns]

            img = candle_image(window, target_size=224, include_volume=False)
            return img
        except Exception:
            return None
    
    def _get_spread(self) -> float:
        """Get current spread."""
        if self.data_provider:
            try:
                return self.data_provider.get_spread(self.config.symbol)
            except:
                pass
        
        # Default spreads
        spreads = {
            'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.0,
            'USDCHF': 1.5, 'AUDUSD': 1.2
        }
        return spreads.get(self.config.symbol, 2.0)
    
    def _get_account_balance(self) -> float:
        """Get current account balance.

        In backtesting, some executors expose the balance as an attribute instead
        of a method; this function supports both.
        """
        if self.executor:
            try:
                return self.executor.get_account_balance()
            except:
                pass
            try:
                return float(getattr(self.executor, 'balance'))
            except Exception:
                pass
        return self._current_balance


def create_strategy(
    profile: str = 'INTRADAY',
    symbol: str = 'EURUSD',
    data_provider = None,
    executor = None,
    config: Optional[StrategyConfig] = None,
    tcn_weights: str = 'models/weights/tcn_best.pt',
    meta_model_path: Optional[str] = None,
    exit_model_path: Optional[str] = None,
    starting_balance: Optional[float] = None,
    **kwargs
) -> NeuralHybridStrategy:
    """Factory function to create configured strategy."""
    if config is None:
        config = StrategyConfig(
            profile=profile,
            symbol=symbol,
            tcn_weights=tcn_weights,
            meta_model_path=meta_model_path,
            exit_model_path=exit_model_path,
            **kwargs
        )
    
    strategy = NeuralHybridStrategy(
        config=config,
        data_provider=data_provider,
        executor=executor
    )
    
    strategy.initialize(starting_balance)
    
    return strategy
