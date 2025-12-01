# strategies/neural_hybrid.py
"""
Neural network hybrid strategy combining multiple ML models.
"""
import logging
from typing import Optional
import pandas as pd

from .base import Strategy, DataProvider, Executor
from inference.predictor import HybridPredictor, SimpleLSTMPredictor
from trading.signal_engine import generate_signal, Signal, SignalConfig
from trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class NeuralHybridStrategy(Strategy):
    """
    Multi-modal neural network trading strategy.
    
    Combines:
    - LSTM for time-series analysis
    - ViT for visual pattern recognition
    - YOLO for candlestick pattern detection
    - Fusion network for signal generation
    
    With proper risk management integration.
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
        signal_config: Optional[SignalConfig] = None,
        min_bars: int = 100,
    ):
        super().__init__(data_provider, executor, name="NeuralHybrid")
        
        self.risk_manager = risk_manager
        self.predictor = predictor or HybridPredictor()
        self.signal_config = signal_config or SignalConfig()
        self.min_bars = min_bars
        
        # State tracking
        self.last_signal: Optional[str] = None
        self.signals_generated = 0
        self.trades_executed = 0
    
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Process new bar and potentially generate trade.
        
        Args:
            df: DataFrame with recent OHLCV data
        
        Returns:
            Signal generated or None
        """
        if not self.is_active:
            return None
        
        # Validate data
        if len(df) < self.min_bars:
            logger.debug(f"Insufficient data: {len(df)} < {self.min_bars}")
            return None
        
        # Check risk limits before prediction (saves compute)
        account_info = self._get_account_info()
        if account_info:
            allowed, reason = self.risk_manager.check_risk_limits(
                current_balance=account_info.get('balance', 0),
                current_equity=account_info.get('equity', 0),
                open_positions=len(self.executor.get_open_positions()),
            )
            if not allowed:
                logger.info(f"[STRATEGY] Trading blocked: {reason}")
                return None
        
        # Generate prediction
        try:
            result = self.predictor.predict(df)
            logger.debug(
                f"Prediction: {result.probabilities} "
                f"(conf: {result.confidence:.2%}, gates: {result.gate_weights})"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
        
        # Generate signal from probabilities
        signal_result = generate_signal(result.probabilities, self.signal_config)
        self.signals_generated += 1
        
        logger.info(f"[STRATEGY] {signal_result.signal.value}: {signal_result.reason}")
        
        # Only execute on actionable signals
        if signal_result.signal not in (Signal.BUY, Signal.SELL):
            self.last_signal = signal_result.signal.value
            return signal_result.signal.value
        
        # Check for existing position in same direction (avoid doubling up)
        if self._has_position_in_direction(signal_result.signal.value):
            logger.info(f"[STRATEGY] Already have {signal_result.signal.value} position")
            return signal_result.signal.value
        
        # Calculate trade parameters
        params = self.risk_manager.get_params(
            df=df,
            signal=signal_result.signal.value,
            symbol_info=self._get_symbol_info(),
        )
        
        logger.info(
            f"[STRATEGY] Trade params: vol={params.volume}, "
            f"SL={params.stop_loss}, TP={params.take_profit}, ATR={params.atr:.5f}"
        )
        
        # Execute trade
        try:
            order_result = self.executor.entry(
                signal=signal_result.signal.value,
                volume=params.volume,
                sl=params.stop_loss,
                tp=params.take_profit,
            )
            
            if order_result.success:
                self.trades_executed += 1
                self.risk_manager.record_trade()
                logger.info(f"[STRATEGY] Order executed: ticket={order_result.ticket}")
            else:
                logger.warning(f"[STRATEGY] Order failed: {order_result.error}")
                
        except Exception as e:
            logger.error(f"[STRATEGY] Execution error: {e}")
        
        self.last_signal = signal_result.signal.value
        return signal_result.signal.value
    
    def _get_account_info(self) -> Optional[dict]:
        """Get account info from executor if available."""
        if hasattr(self.executor, 'get_account_info'):
            info = self.executor.get_account_info()
            if info:
                return {
                    'balance': info.balance,
                    'equity': info.equity,
                }
        return None
    
    def _get_symbol_info(self) -> Optional[dict]:
        """Get symbol info from executor if available."""
        if hasattr(self.executor, 'get_symbol_info'):
            return self.executor.get_symbol_info()
        return None
    
    def _has_position_in_direction(self, direction: str) -> bool:
        """Check if already have position in given direction."""
        try:
            positions = self.executor.get_open_positions()
            for pos in positions:
                if pos.get('type', '').upper() == direction.upper():
                    return True
        except Exception:
            pass
        return False
    
    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'last_signal': self.last_signal,
        }


class SimpleLSTMStrategy(Strategy):
    """
    Simplified strategy using only LSTM model.
    Faster inference, useful for testing.
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        confidence_threshold: float = 0.6,
    ):
        super().__init__(data_provider, executor, name="SimpleLSTM")
        
        self.risk_manager = risk_manager
        self.predictor = SimpleLSTMPredictor()
        self.confidence_threshold = confidence_threshold
    
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """Process bar with LSTM-only prediction."""
        if not self.is_active or len(df) < 60:
            return None
        
        try:
            result = self.predictor.predict(df)
            
            if result.confidence < self.confidence_threshold:
                return "HOLD"
            
            signal = result.CLASS_NAMES[result.predicted_class]
            
            if signal in ('BUY', 'SELL'):
                params = self.risk_manager.get_params(df, signal)
                self.executor.entry(signal, params.volume, params.stop_loss, params.take_profit)
            
            return signal
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
