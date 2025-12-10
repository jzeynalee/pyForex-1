# strategies/neural_hybrid.py
"""
Neural Hybrid Strategy with Full Risk Management

Integrates:
- TCN/ViT/YOLO prediction pipeline
- MTF trend analysis
- Phase 1-3 risk management
- Automated position sizing and SL/TP

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

# Risk management imports
try:
    from risk_management import (
        MetaLabelingModel, TradeFilter,
        TripleBarrierLabeler, TripleBarrierConfig,
        generate_performance_report
    )
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False

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
    entry_price: float
    stop_loss: float
    take_profit: float
    comment: str = ""
    magic_number: int = 123456
    
    # Risk info
    risk_percent: float = 0.0
    risk_reward: float = 0.0
    confidence: float = 0.0
    meta_score: float = 0.0


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
    
    # Feature settings
    sequence_length: int = 60
    use_vision: bool = True
    use_yolo: bool = True
    
    # Risk settings
    base_risk_percent: float = 1.0
    max_risk_percent: float = 2.0
    min_risk_reward: float = 1.5
    max_daily_loss_percent: float = 3.0
    max_open_trades: int = 3
    
    # Signal thresholds
    min_confidence: float = 0.55
    min_meta_score: float = 0.5
    min_mtf_alignment: float = 0.6
    
    # Session settings
    allowed_sessions: List[str] = field(default_factory=lambda: ['london', 'new_york'])
    avoid_news_minutes: int = 30


class NeuralHybridStrategy:
    """
    Production trading strategy combining neural networks with risk management.
    
    Pipeline:
    1. Fetch multi-timeframe data
    2. Generate features
    3. Get predictions from TCN (+ ViT + YOLO)
    4. Analyze MTF trend alignment
    5. Calculate SL/TP from quantile predictions
    6. Calculate position size from volatility
    7. Apply meta-labeling filter
    8. Validate against hard rules
    9. Execute or reject trade
    
    Usage:
        strategy = NeuralHybridStrategy(config, data_provider, executor)
        strategy.initialize()
        
        # In trading loop
        signal = strategy.evaluate()
        if signal and signal.should_trade:
            order = strategy.create_order(signal)
            strategy.execute(order)
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider,  # Your data provider class
        executor,       # Your order executor class
        risk_manager=None  # Optional legacy risk manager
    ):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.legacy_risk_manager = risk_manager
        
        # Components (initialized in initialize())
        self.predictor = None
        self.decision_engine = None
        self.mtf_detector = None
        self.meta_model = None
        
        # State
        self._initialized = False
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._open_positions: Dict[str, Dict] = {}
        
        # Performance tracking
        self._trade_history: List[Dict] = []
    
    def initialize(self) -> bool:
        """
        Initialize all strategy components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize predictor
            logger.info("Initializing predictor...")
            self.predictor = create_predictor(
                profile=self.config.profile,
                weights_path=self.config.tcn_weights,
                use_vision=self.config.use_vision,
                use_yolo=self.config.use_yolo
            )
            
            # Load additional vision weights if hybrid
            if isinstance(self.predictor, HybridPredictor):
                if self.config.vit_weights:
                    self.predictor._init_vision(self.config.vit_weights)
                if self.config.yolo_weights:
                    self.predictor._init_yolo(self.config.yolo_weights)
                if self.config.fusion_weights:
                    self.predictor._init_fusion(self.config.fusion_weights)
            
            # Initialize decision engine with risk management
            logger.info("Initializing decision engine...")
            engine_config = DecisionEngineConfig(
                profile=self.config.profile,
                min_direction_confidence=self.config.min_confidence,
                min_meta_score=self.config.min_meta_score,
                min_mtf_alignment=self.config.min_mtf_alignment,
                base_risk_percent=self.config.base_risk_percent,
                min_risk_reward=self.config.min_risk_reward
            )
            
            # Load meta-model if available
            if HAS_RISK_MANAGEMENT and self.config.meta_model_path:
                try:
                    self.meta_model = MetaLabelingModel.load(self.config.meta_model_path)
                    logger.info("Meta-labeling model loaded")
                except Exception as e:
                    logger.warning(f"Could not load meta-model: {e}")
                    self.meta_model = None
            
            self.decision_engine = EnhancedDecisionEngine(
                config=engine_config,
                meta_model=self.meta_model
            )
            
            # Initialize MTF detector
            if HAS_MTF:
                try:
                    self.mtf_detector = MTFTrendDetector(profile=self.config.profile)
                    logger.info("MTF detector initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize MTF detector: {e}")
            
            self._initialized = True
            logger.info(f"Strategy initialized for {self.config.symbol} ({self.config.profile})")
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
            
            # Evaluate with decision engine
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
    
    def create_order(self, decision: TradeDecision) -> Optional[Order]:
        """
        Create order from trade decision.
        
        Args:
            decision: TradeDecision from decision engine
        
        Returns:
            Order object ready for execution
        """
        if not decision.should_trade:
            return None
        
        return Order(
            symbol=self.config.symbol,
            order_type=OrderType.MARKET,
            direction=decision.direction,
            volume=decision.position_size,
            entry_price=0.0,  # Market order
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            comment=f"NeuralHybrid_{self.config.profile}",
            risk_percent=decision.risk_percent,
            risk_reward=decision.risk_reward_ratio,
            confidence=decision.direction_confidence,
            meta_score=decision.meta_score
        )
    
    def execute(self, order: Order) -> bool:
        """
        Execute order through executor.
        
        Args:
            order: Order to execute
        
        Returns:
            True if execution successful
        """
        try:
            # Execute through your executor
            result = self.executor.execute_order(
                symbol=order.symbol,
                order_type=order.order_type.value,
                direction=order.direction,
                volume=order.volume,
                sl=order.stop_loss,
                tp=order.take_profit,
                comment=order.comment,
                magic=order.magic_number
            )
            
            if result.get('success', False):
                # Track position
                ticket = result.get('ticket', 0)
                self._open_positions[str(ticket)] = {
                    'symbol': order.symbol,
                    'direction': order.direction,
                    'volume': order.volume,
                    'entry_price': result.get('price', 0),
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                    'open_time': datetime.utcnow()
                }
                
                self._daily_trades += 1
                
                logger.info(
                    f"Order executed: {order.direction} {order.volume} {order.symbol} | "
                    f"SL: {order.stop_loss:.5f} | TP: {order.take_profit:.5f}"
                )
                return True
            else:
                logger.warning(f"Order execution failed: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def on_trade_closed(self, ticket: str, pnl: float):
        """
        Handle closed trade notification.
        
        Args:
            ticket: Trade ticket number
            pnl: Profit/loss amount
        """
        if ticket in self._open_positions:
            position = self._open_positions.pop(ticket)
            
            # Update daily P&L
            self._daily_pnl += pnl
            
            # Record trade
            self._trade_history.append({
                'ticket': ticket,
                'symbol': position['symbol'],
                'direction': position['direction'],
                'volume': position['volume'],
                'entry_price': position['entry_price'],
                'pnl': pnl,
                'close_time': datetime.utcnow(),
                'holding_period': (datetime.utcnow() - position['open_time']).total_seconds() / 60
            })
            
            # Update decision engine positions
            self.decision_engine.update_positions(self._open_positions)
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of each day)."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
    
    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics."""
        if not self._trade_history:
            return {'total_trades': 0}
        
        pnls = [t['pnl'] for t in self._trade_history]
        
        if HAS_RISK_MANAGEMENT:
            from risk_management.utils import generate_performance_report
            report = generate_performance_report(self._trade_history)
            return {
                'total_trades': report.total_trades,
                'win_rate': report.win_rate,
                'profit_factor': report.profit_factor,
                'sharpe_ratio': report.sharpe_ratio,
                'max_drawdown': report.max_drawdown,
                'total_return': report.total_return
            }
        
        # Basic stats if risk management not available
        wins = sum(1 for p in pnls if p > 0)
        return {
            'total_trades': len(pnls),
            'wins': wins,
            'losses': len(pnls) - wins,
            'win_rate': wins / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0
        }
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _check_daily_limits(self) -> bool:
        """Check if daily loss limit reached."""
        account_balance = self._get_account_balance()
        max_loss = account_balance * (self.config.max_daily_loss_percent / 100)
        
        if self._daily_pnl <= -max_loss:
            logger.warning(f"Daily loss limit reached: {self._daily_pnl:.2f}")
            return False
        
        return True
    
    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch market data from data provider."""
        try:
            return self.data_provider.get_data(
                symbol=self.config.symbol,
                timeframe=self._get_primary_timeframe(),
                bars=self.config.sequence_length + 100
            )
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None
    
    def _fetch_mtf_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch multi-timeframe data."""
        if not HAS_MTF:
            return None
        
        try:
            profile = get_profile(self.config.profile)
            mtf_data = {}
            
            for tf in profile.timeframe_strings:
                data = self.data_provider.get_data(
                    symbol=self.config.symbol,
                    timeframe=tf,
                    bars=200
                )
                if data is not None:
                    mtf_data[tf] = data
            
            return mtf_data if mtf_data else None
            
        except Exception as e:
            logger.warning(f"MTF data fetch error: {e}")
            return None
    
    def _get_primary_timeframe(self) -> str:
        """Get primary timeframe for profile."""
        tf_map = {
            'SCALP': 'M5',
            'INTRADAY': 'M15',
            'SWING': 'H1'
        }
        return tf_map.get(self.config.profile, 'M15')
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features from market data."""
        # This should use your existing feature engineering
        # For now, return OHLCV + basic indicators
        features = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add basic indicators if not present
        if 'atr' not in data.columns:
            features['atr'] = self._calculate_atr(data)
        else:
            features['atr'] = data['atr']
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features.values[-self.config.sequence_length:]
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    def _generate_chart_image(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate chart image for vision models."""
        try:
            # Use your chart generator
            from utils.candle_to_image import CandlestickRenderer
            return CandlestickRenderer(data, size=224)
        except ImportError:
            return None
    
    def _get_spread(self) -> float:
        """Get current spread from data provider or estimate."""
        try:
            return self.data_provider.get_spread(self.config.symbol)
        except:
            # Estimate based on pair
            spreads = {
                'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.0,
                'USDCHF': 1.5, 'AUDUSD': 1.5, 'NZDUSD': 2.0
            }
            return spreads.get(self.config.symbol, 2.0)
    
    def _get_account_balance(self) -> float:
        """Get current account balance."""
        try:
            return self.executor.get_account_balance()
        except:
            return 10000.0  # Default for testing


# Factory function
def create_strategy(
    profile: str = 'INTRADAY',
    symbol: str = 'EURUSD',
    data_provider=None,
    executor=None,
    **kwargs
) -> NeuralHybridStrategy:
    """
    Create configured strategy instance.
    
    Args:
        profile: Trading profile
        symbol: Trading symbol
        data_provider: Data provider instance
        executor: Order executor instance
        **kwargs: Additional config options
    
    Returns:
        Configured NeuralHybridStrategy
    """
    config = StrategyConfig(
        profile=profile,
        symbol=symbol,
        **kwargs
    )
    
    return NeuralHybridStrategy(config, data_provider, executor)