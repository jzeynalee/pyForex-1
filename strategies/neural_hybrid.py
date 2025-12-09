# strategies/neural_hybrid.py
"""
Neural network hybrid strategy combining multiple ML models.

UPDATED:
- Uses TCN instead of LSTM for sequence modeling
- Integrates with MTFTrendDetector for multi-timeframe context
- Uses new DecisionEngine with MTF support
"""

import logging
from typing import Optional, Dict
import pandas as pd

from strategies.base import Strategy, DataProvider, Executor
from inference.predictor import HybridPredictor, TCNPredictor, PredictionResult
from trading.signal_engine import generate_signal, Signal, SignalConfig
from trading.risk_manager import RiskManager
from trading.decision_engine import DecisionEngine, MTFDecisionEngine, DecisionConfig

logger = logging.getLogger(__name__)


class NeuralHybridStrategy(Strategy):
    """
    Multi-modal neural network trading strategy.
    
    Combines:
    - TCN for time-series analysis (replaces LSTM)
    - ViT for visual pattern recognition
    - YOLO for candlestick pattern detection
    - Fusion network for signal generation
    - MTF trend analysis for context
    
    With proper risk management integration.
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
        signal_config: Optional[SignalConfig] = None,
        decision_config: Optional[DecisionConfig] = None,
        mtf_profile: str = "SWING",
        min_bars: int = 100,
    ):
        super().__init__(data_provider, executor, name="NeuralHybrid")
        
        self.risk_manager = risk_manager
        self.signal_config = signal_config or SignalConfig()
        self.min_bars = min_bars
        self.mtf_profile = mtf_profile
        
        # Initialize predictor (uses TCN by default now)
        self.predictor = predictor or HybridPredictor()
        
        # Initialize MTF-enabled decision engine
        self.decision_engine = MTFDecisionEngine(
            profile=mtf_profile,
            config=decision_config,
        )
        
        # State tracking
        self.last_signal: Optional[str] = None
        self.last_decision: Optional[Dict] = None
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
        
        # 1. Generate pattern prediction (TCN + ViT + YOLO + Fusion)
        try:
            pred_result = self.predictor.predict(df)
            logger.debug(
                f"Prediction: {pred_result.probabilities} "
                f"(conf: {pred_result.confidence:.2%}, gates: {pred_result.gate_weights})"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
        
        # 2. Build MTF data dict for trend analysis
        # In live trading, we'd fetch multiple timeframes
        # Here we use the primary TF data as fallback
        dfs_dict = self._prepare_mtf_data(df)
        
        # 3. Get decision from MTF-enabled decision engine
        recommendation = self.decision_engine.get_recommendation(
            pattern_probs=pred_result.probabilities.tolist(),
            dfs_dict=dfs_dict,
        )
        
        self.last_decision = recommendation
        self.signals_generated += 1
        
        signal = recommendation['signal']
        confidence = recommendation['confidence']
        reason = recommendation['reason']
        
        logger.info(
            f"[STRATEGY] {signal} | Conf: {confidence:.2%} | "
            f"Trend: {recommendation['trend']} | {reason}"
        )
        
        # Only execute on actionable signals
        if signal not in ('BUY', 'SELL'):
            self.last_signal = signal
            return signal
        
        # Check for existing position in same direction
        if self._has_position_in_direction(signal):
            logger.info(f"[STRATEGY] Already have {signal} position")
            return signal
        
        # Calculate trade parameters
        params = self.risk_manager.get_params(
            df=df,
            signal=signal,
            symbol_info=self._get_symbol_info(),
        )
        
        # Adjust volume based on confidence
        volume = params.volume
        if confidence > 0.80:
            volume = min(volume * 1.2, 1.0)  # Cap at 1 lot
        elif confidence < 0.60:
            volume = volume * 0.8
        
        logger.info(
            f"[STRATEGY] Trade params: vol={volume:.2f}, "
            f"SL={params.stop_loss:.5f}, TP={params.take_profit:.5f}"
        )
        
        # Execute trade
        try:
            order_result = self.executor.entry(
                signal=signal,
                volume=volume,
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
        
        self.last_signal = signal
        return signal
    
    def _prepare_mtf_data(self, df_primary: pd.DataFrame) -> Dict:
        """
        Prepare MTF data dict from primary timeframe.
        
        In production, this should fetch data for multiple timeframes
        from the data_provider. Here we use resampling as fallback.
        """
        # Check if data_provider supports MTF
        if hasattr(self.data_provider, 'get_mtf_data'):
            return self.data_provider.get_mtf_data()
        
        # Check if data_provider has MTF cache
        if hasattr(self.data_provider, 'fetch_for_profile'):
            from utils.mtf_config import get_profile
            profile = get_profile(self.mtf_profile)
            return self.data_provider.fetch_for_profile(profile)
        
        # Fallback: Use primary TF for all (not ideal but works for testing)
        # In production, you should properly fetch multi-TF data
        if self.mtf_profile == "SCALP":
            return {
                'M5': df_primary,
                'M15': df_primary,
                'H1': df_primary,
            }
        elif self.mtf_profile == "SWING":
            return {
                'H1': df_primary,
                'H4': df_primary,
                'D1': df_primary,
            }
        else:  # INTRADAY
            return {
                'M15': df_primary,
                'H1': df_primary,
                'H4': df_primary,
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
            'mtf_profile': self.mtf_profile,
            'last_decision': self.last_decision,
        }


class SimpleTCNStrategy(Strategy):
    """
    Simplified strategy using only TCN model.
    
    Faster inference, useful for:
    - Testing and debugging
    - Scalping where latency matters
    - When visual features aren't reliable
    
    NOTE: Replaces old SimpleLSTMStrategy
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        profile: str = "INTRADAY",
        confidence_threshold: float = 0.6,
    ):
        super().__init__(data_provider, executor, name="SimpleTCN")
        
        self.risk_manager = risk_manager
        self.predictor = TCNPredictor(profile=profile)
        self.confidence_threshold = confidence_threshold
        self.profile = profile
        
        # State
        self.signals_generated = 0
        self.trades_executed = 0
    
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """Process bar with TCN-only prediction."""
        if not self.is_active or len(df) < 60:
            return None
        
        try:
            result = self.predictor.predict(df)
            self.signals_generated += 1
            
            if result.confidence < self.confidence_threshold:
                return "HOLD"
            
            signal = result.signal_name
            
            if signal in ('BUY', 'SELL'):
                params = self.risk_manager.get_params(df, signal)
                
                order_result = self.executor.entry(
                    signal=signal,
                    volume=params.volume,
                    sl=params.stop_loss,
                    tp=params.take_profit,
                )
                
                if order_result.success:
                    self.trades_executed += 1
                    self.risk_manager.record_trade()
            
            return signal
            
        except Exception as e:
            logger.error(f"TCN prediction error: {e}")
            return None
    
    def get_stats(self) -> dict:
        return {
            'name': self.name,
            'profile': self.profile,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
        }


# =============================================================================
# Backward Compatibility
# =============================================================================

# DEPRECATED: Use SimpleTCNStrategy instead
SimpleLSTMStrategy = SimpleTCNStrategy