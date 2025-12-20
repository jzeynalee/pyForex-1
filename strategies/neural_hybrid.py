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
from datetime import datetime, timedelta
from enum import Enum
import logging

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
        
        # Trade history
        self._trade_history: List[Dict] = []
        
        logger.info(f"NeuralHybridStrategy v2 created for {self.config.symbol}")
    
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
                tcn_weights=self.config.tcn_weights,
                vit_weights=self.config.vit_weights if self.config.use_vision else None,
                yolo_weights=self.config.yolo_weights if self.config.use_yolo else None,
                fusion_weights=self.config.fusion_weights
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
                max_drawdown_pct=self.config.max_drawdown_percent
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
                    self.mtf_detector = MTFTrendDetector(profile=self.config.profile)
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
        """Execute order through executor."""
        if not self.executor:
            logger.warning("No executor configured")
            return False
        
        try:
            result = self.executor.execute_order(
                symbol=order.symbol,
                order_type=order.order_type.value,
                direction=order.direction,
                volume=order.volume,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                comment=order.comment,
                magic_number=order.magic_number
            )
            
            if result.get('success'):
                ticket = result.get('ticket', str(datetime.utcnow().timestamp()))
                order.ticket = ticket
                order.price = float(result.get('price', order.price) or order.price)
                
                # Track position
                self._open_positions[ticket] = OpenPosition(
                    ticket=ticket,
                    direction=1 if order.direction == 'BUY' else -1,
                    entry_price=result.get('price', order.price),
                    entry_time=datetime.utcnow(),
                    volume=order.volume,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit
                )
                
                self._daily_trades += 1
                
                logger.info(
                    f"Order executed: {order.direction} {order.volume} {order.symbol} | "
                    f"Ticket: {ticket} | Protection: {order.protection_level}"
                )
                return True
            else:
                logger.warning(f"Order execution failed: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def on_trade_closed(self, ticket: str, pnl: float):
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
        
        # Update decision engine's capital protection
        if self.decision_engine:
            self.decision_engine.record_trade_result(
                pnl=pnl,
                is_win=pnl >= 0,
                size=pos.volume if pos else 0.0
            )
        
        # Record in history
        self._trade_history.append({
            'ticket': ticket,
            'pnl': pnl,
            'is_win': pnl >= 0,
            'close_time': datetime.utcnow().isoformat(),
            'balance': self._current_balance
        })
        
        logger.info(f"Trade closed: {ticket} | PnL: {pnl:.2f} | Balance: {self._current_balance:.2f}")
    
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
            'open_positions': len(self._open_positions),
            'protection_status': self.get_protection_status()
        }
    
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
            timeframes = ['M5', 'M15', 'H1', 'H4']
            return {
                tf: self.data_provider.get_ohlcv(self.config.symbol, timeframe=tf)
                for tf in timeframes
            }
        except Exception as e:
            logger.error(f"MTF data fetch error: {e}")
            return None
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        # Get feature columns from predictor
        if hasattr(self.predictor, 'feature_columns'):
            feature_cols = self.predictor.feature_columns
            if all(col in data.columns for col in feature_cols):
                return data[feature_cols].values[-self.config.sequence_length:]
        
        # Fallback to OHLCV
        return data[['open', 'high', 'low', 'close', 'volume']].values[-self.config.sequence_length:]
    
    def _generate_chart_image(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate chart image for vision models."""
        # Placeholder - actual implementation would render candlestick chart
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
        """Get current account balance."""
        if self.executor:
            try:
                return self.executor.get_account_balance()
            except:
                pass
        return self._current_balance


def create_strategy(
    profile: str,
    symbol: str,
    data_provider = None,
    executor = None,
    tcn_weights: str = 'models/weights/tcn_best.pt',
    meta_model_path: Optional[str] = None,
    exit_model_path: Optional[str] = None,
    starting_balance: Optional[float] = None,
    **kwargs
) -> NeuralHybridStrategy:
    """Factory function to create configured strategy."""
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
