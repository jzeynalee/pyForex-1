# strategies/neural_hybrid.py
"""
Neural network hybrid strategy combining multiple ML models.
"""
import logging
from typing import Optional, Dict
import pandas as pd

from .base import Strategy, DataProvider, Executor
from inference.predictor import HybridPredictor, SimpleLSTMPredictor
from trading.signal_engine import generate_signal, Signal, SignalConfig
from trading.risk_manager import RiskManager
from trend_detection.fusion_trend_detector import FusionFXTrendDetector
from trading.decision_engine import DecisionEngine

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

        # Initialize Components
        self.predictor = HybridPredictor()
        
        # Initialize "Brain 2": Trend Engine
        # Note: Requires ML model path or None for Step 4
        self.trend_detector = FusionFXTrendDetector(ml_model=None) 
        
        # Initialize "Cortex": Decision Engine
        self.decision_engine = DecisionEngine(threshold=0.65)
        
        # Buffer for MTF data (Backtesting needs to simulate MTF)
        self._history_buffer = []
    
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
        
        # 1. BRAIN 1: Deep Learning Prediction
        try:
            pred_result = self.predictor.predict(df)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
        
        # 2. BRAIN 2: Trend Analysis
        # Note: In backtesting, getting MTF data is tricky. 
        # For now, we simulate using the current H1 data resampled, 
        # or simplified structural analysis if H4/M15 aren't available.
        
        # Ideally, data_provider should support .get_data(tf='H4')
        # Here we attempt to build the dict if possible, or fall back to single TF
        dfs_dict = self._prepare_mtf_data(df)
        
        trend_analysis = self.trend_detector.detect_trend(dfs_dict)

        # 3. DECISION: Gate the signal
        decision = self.decision_engine.decide(
            pattern_probs=pred_result.probabilities,
            trend_analysis=trend_analysis
        )

        logger.info(f"[DECISION] {decision.signal} | {decision.reason} | Conf: {decision.confidence:.2f}")

        if decision.signal == "NO_TRADE":
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
    
    def _prepare_mtf_data(self, df_h1: pd.DataFrame) -> Dict:
        """
        Helper to approximate MTF data from H1 for backtesting 
        if separate streams aren't available.
        """
        # Simplistic resampling for H4
        df_h4 = df_h1.resample('4H', on='time').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna()
        
        # For M15, we can't invent data, so we might just use H1 as fallback 
        # or require the data_provider to supply it.
        return {
            'H1': df_h1,
            'H4': df_h4 if not df_h4.empty else df_h1,
            'M15': df_h1 # Fallback if M15 not available in backtest stream
        }
    
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
