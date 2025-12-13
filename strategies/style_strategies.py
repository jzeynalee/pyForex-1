# strategies/style_strategies.py
"""
Style-Specific Strategy Implementations.

Each strategy adapts the neural hybrid model to its trading style:
- ScalpingStrategy: Fast entries, tight stops, quick profits
- IntradayStrategy: Balanced approach, standard parameters
- SwingStrategy: Patient entries, wide stops, large targets
"""

import logging
from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from strategies.base import Strategy, DataProvider, Executor
from trading.style_config import StyleConfig, TradingStyle, SCALP_CONFIG, INTRADAY_CONFIG, SWING_CONFIG
from trading.risk_manager import RiskManager
from inference.predictor import HybridPredictor, RiskAwareTCNPredictor, PredictionResult

logger = logging.getLogger(__name__)


class StyleStrategy(Strategy):
    """
    Base class for style-specific strategies.
    
    Adapts neural network predictions to style-specific rules:
    - Confidence thresholds
    - Risk/reward requirements
    - Session filtering
    - Position sizing
    """
    
    def __init__(
        self,
        style_config: StyleConfig,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
    ):
        super().__init__(data_provider, executor, name=style_config.name)
        
        self.style_config = style_config
        self.style = style_config.style
        self.risk_manager = risk_manager
        self.predictor = predictor or HybridPredictor()
        
        # Tracking
        self.last_signal: Optional[str] = None
        self.last_prediction: Optional[PredictionResult] = None
        self.signals_generated = 0
        self.signals_filtered = 0
        self.trades_executed = 0
        
        # Cooldown
        self.last_trade_time: Optional[datetime] = None
        self.min_trade_interval = self._get_min_trade_interval()
    
    def _get_min_trade_interval(self) -> int:
        """Get minimum seconds between trades for this style."""
        intervals = {
            TradingStyle.SCALP: 60,      # 1 minute
            TradingStyle.INTRADAY: 300,  # 5 minutes
            TradingStyle.SWING: 3600,    # 1 hour
        }
        return intervals.get(self.style, 300)
    
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Process new bar for this style.
        
        Applies style-specific filters before signaling.
        """
        if not self.is_active:
            return None
        
        # Data validation
        if len(df) < self.style_config.min_bars_required:
            logger.debug(f"[{self.name}] Insufficient data: {len(df)}")
            return None
        
        # Session filter
        if not self._is_valid_session():
            logger.debug(f"[{self.name}] Outside trading session")
            return None
        
        # Cooldown check
        if not self._check_cooldown():
            logger.debug(f"[{self.name}] In cooldown period")
            return None
        
        # Get prediction
        try:
            result = self.predictor.predict(df)
            self.last_prediction = result
        except Exception as e:
            logger.error(f"[{self.name}] Prediction failed: {e}")
            return None
        
        # Apply style-specific signal logic
        signal, confidence, reason = self._evaluate_signal(result, df)
        self.signals_generated += 1
        
        if signal == 'NO_TRADE':
            self.signals_filtered += 1
            logger.debug(f"[{self.name}] Filtered: {reason}")
            return None
        
        logger.info(f"[{self.name}] Signal: {signal} ({confidence:.2%}) - {reason}")
        
        self.last_signal = signal
        return signal
    
    def _evaluate_signal(
        self, 
        result: PredictionResult,
        df: pd.DataFrame,
    ) -> Tuple[str, float, str]:
        """
        Evaluate prediction against style rules.
        
        Returns:
            (signal, confidence, reason)
        """
        probs = result.probabilities
        p_buy, p_sell, p_hold = probs[0], probs[1], probs[2]
        
        # Determine raw signal
        if p_buy > p_sell and p_buy > p_hold:
            raw_signal = 'BUY'
            raw_conf = p_buy
        elif p_sell > p_buy and p_sell > p_hold:
            raw_signal = 'SELL'
            raw_conf = p_sell
        else:
            return 'NO_TRADE', p_hold, "HOLD dominant"
        
        # Filter 1: Minimum confidence
        if raw_conf < self.style_config.min_confidence:
            return 'NO_TRADE', raw_conf, f"Low confidence ({raw_conf:.2%} < {self.style_config.min_confidence:.2%})"
        
        # Filter 2: Trend alignment (if required)
        if self.style_config.require_trend_confirmation:
            trend_aligned, trend_conf = self._check_trend_alignment(df, raw_signal)
            
            if not trend_aligned:
                # Check if counter-trend allowed
                if self.style_config.allow_counter_trend:
                    if raw_conf >= self.style_config.counter_trend_confidence:
                        return raw_signal, raw_conf * 0.7, "Counter-trend (high confidence)"
                return 'NO_TRADE', raw_conf, f"Trend misalignment ({trend_conf:.2%})"
        
        # Filter 3: Risk/Reward check
        rr = self._estimate_risk_reward(df, raw_signal)
        if rr < self.style_config.min_risk_reward:
            return 'NO_TRADE', raw_conf, f"Poor R:R ({rr:.1f} < {self.style_config.min_risk_reward:.1f})"
        
        # Filter 4: Volatility filter (avoid extremes)
        vol_ok, vol_reason = self._check_volatility(df)
        if not vol_ok:
            return 'NO_TRADE', raw_conf, vol_reason
        
        return raw_signal, raw_conf, "All filters passed"
    
    def _check_trend_alignment(
        self, 
        df: pd.DataFrame, 
        signal: str
    ) -> Tuple[bool, float]:
        """Check if signal aligns with trend on higher timeframe."""
        # Simple EMA-based trend check
        close = df['close']
        
        ema_fast = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema_trend = close.ewm(span=200, adjust=False).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Calculate trend score
        bullish_score = 0
        if ema_fast > ema_slow:
            bullish_score += 0.33
        if ema_slow > ema_trend:
            bullish_score += 0.33
        if current_price > ema_fast:
            bullish_score += 0.34
        
        if signal == 'BUY':
            aligned = bullish_score >= self.style_config.min_trend_alignment
            return aligned, bullish_score
        else:  # SELL
            bearish_score = 1.0 - bullish_score
            aligned = bearish_score >= self.style_config.min_trend_alignment
            return aligned, bearish_score
    
    def _estimate_risk_reward(self, df: pd.DataFrame, signal: str) -> float:
        """Estimate R:R ratio for potential trade."""
        atr = self._calculate_atr(df)
        
        sl_dist = atr * self.style_config.sl_atr_multiplier
        tp_dist = atr * self.style_config.tp_atr_multiplier
        
        if sl_dist == 0:
            return 0.0
        
        return tp_dist / sl_dist
    
    def _check_volatility(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if volatility is appropriate for this style."""
        atr = self._calculate_atr(df)
        avg_atr = self._calculate_atr(df, period=50)  # Longer-term ATR
        
        if avg_atr == 0:
            return True, ""
        
        vol_ratio = atr / avg_atr
        
        # Scalping: Avoid very low volatility
        if self.style == TradingStyle.SCALP:
            if vol_ratio < 0.5:
                return False, "Volatility too low for scalp"
            if vol_ratio > 2.5:
                return False, "Volatility too high for scalp"
        
        # Swing: More tolerant of high volatility
        elif self.style == TradingStyle.SWING:
            if vol_ratio < 0.3:
                return False, "Volatility too low for swing"
        
        return True, ""
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if len(df) < period + 1:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(true_range[-period:])
        
        return float(atr)
    
    def _is_valid_session(self) -> bool:
        """Check if current time is within allowed trading sessions."""
        if not self.style_config.allowed_sessions:
            return True  # No restriction
        
        now = datetime.utcnow()
        current_hour = now.hour
        
        for start_hour, end_hour in self.style_config.allowed_sessions:
            if start_hour <= current_hour < end_hour:
                return True
        
        return False
    
    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since last trade."""
        if self.last_trade_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= self.min_trade_interval
    
    def record_trade(self):
        """Record that a trade was executed."""
        self.trades_executed += 1
        self.last_trade_time = datetime.now()
    
    def get_trade_params(self, df: pd.DataFrame, signal: str) -> Dict:
        """
        Get style-specific trade parameters.
        
        Returns:
            Dict with volume, sl, tp, and metadata
        """
        atr = self._calculate_atr(df)
        current_price = df['close'].iloc[-1]
        
        # Calculate SL/TP based on style config
        sl_dist = atr * self.style_config.sl_atr_multiplier
        tp_dist = atr * self.style_config.tp_atr_multiplier
        
        if signal == 'BUY':
            sl = current_price - sl_dist
            tp = current_price + tp_dist
        else:
            sl = current_price + sl_dist
            tp = current_price - tp_dist
        
        return {
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': round(sl, 5),
            'take_profit': round(tp, 5),
            'atr': atr,
            'sl_pips': int(sl_dist / 0.0001),
            'tp_pips': int(tp_dist / 0.0001),
            'risk_reward': tp_dist / sl_dist if sl_dist > 0 else 0,
            'style': self.style.value,
        }
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            'name': self.name,
            'style': self.style.value,
            'is_active': self.is_active,
            'signals_generated': self.signals_generated,
            'signals_filtered': self.signals_filtered,
            'filter_rate': self.signals_filtered / max(1, self.signals_generated) * 100,
            'trades_executed': self.trades_executed,
            'last_signal': self.last_signal,
            'last_prediction': {
                'probs': self.last_prediction.probabilities.tolist() if self.last_prediction else None,
                'conf': self.last_prediction.confidence if self.last_prediction else None,
            } if self.last_prediction else None,
        }


# =============================================================================
# CONCRETE STYLE IMPLEMENTATIONS
# =============================================================================

class ScalpingStrategy(StyleStrategy):
    """
    Scalping Strategy (M5/M15).
    
    Characteristics:
    - High confidence threshold (0.75+)
    - Tight stops (1.0 ATR)
    - Quick targets (1.5 ATR)
    - Session-restricted (London/NY)
    - No counter-trend trades
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
        config: Optional[StyleConfig] = None,
    ):
        super().__init__(
            style_config=config or SCALP_CONFIG,
            data_provider=data_provider,
            executor=executor,
            risk_manager=risk_manager,
            predictor=predictor,
        )
    
    def _evaluate_signal(
        self, 
        result: PredictionResult,
        df: pd.DataFrame,
    ) -> Tuple[str, float, str]:
        """Scalp-specific signal evaluation with stricter rules."""
        signal, conf, reason = super()._evaluate_signal(result, df)
        
        if signal == 'NO_TRADE':
            return signal, conf, reason
        
        # Additional scalp filters
        
        # Momentum confirmation (price action)
        if not self._check_momentum(df, signal):
            return 'NO_TRADE', conf, "No momentum confirmation"
        
        # Spread check (would need real spread data)
        # For now, skip during likely high-spread times
        hour = datetime.utcnow().hour
        if hour in [22, 23, 0, 1]:  # Asian session low liquidity
            return 'NO_TRADE', conf, "Low liquidity period"
        
        return signal, conf, reason
    
    def _check_momentum(self, df: pd.DataFrame, signal: str) -> bool:
        """Check for momentum in signal direction."""
        if len(df) < 5:
            return True
        
        recent = df.tail(5)
        closes = recent['close'].values
        
        if signal == 'BUY':
            # At least 3 of last 5 bars should be bullish
            bullish_count = sum(1 for i in range(len(closes)-1) if closes[i+1] > closes[i])
            return bullish_count >= 2
        else:
            bearish_count = sum(1 for i in range(len(closes)-1) if closes[i+1] < closes[i])
            return bearish_count >= 2


class IntradayStrategy(StyleStrategy):
    """
    Intraday Strategy (M30/H1).
    
    Characteristics:
    - Standard confidence (0.65+)
    - Balanced stops (1.5 ATR)
    - 1:2+ reward targets
    - Main session trading
    - Limited counter-trend (0.80+ conf)
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
        config: Optional[StyleConfig] = None,
    ):
        super().__init__(
            style_config=config or INTRADAY_CONFIG,
            data_provider=data_provider,
            executor=executor,
            risk_manager=risk_manager,
            predictor=predictor,
        )


class SwingStrategy(StyleStrategy):
    """
    Swing Strategy (H4/D1).
    
    Characteristics:
    - Lower confidence (0.60+)
    - Wide stops (2.0 ATR)
    - Large targets (4.0 ATR, 1:2+ RR)
    - Any session
    - Counter-trend allowed with high confidence
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        risk_manager: RiskManager,
        predictor: Optional[HybridPredictor] = None,
        config: Optional[StyleConfig] = None,
    ):
        super().__init__(
            style_config=config or SWING_CONFIG,
            data_provider=data_provider,
            executor=executor,
            risk_manager=risk_manager,
            predictor=predictor,
        )
    
    def _evaluate_signal(
        self, 
        result: PredictionResult,
        df: pd.DataFrame,
    ) -> Tuple[str, float, str]:
        """Swing-specific evaluation with structural focus."""
        signal, conf, reason = super()._evaluate_signal(result, df)
        
        if signal == 'NO_TRADE':
            return signal, conf, reason
        
        # Additional swing filters
        
        # Check for key level proximity
        if self._near_key_level(df, signal):
            # Boost confidence near support/resistance
            return signal, min(conf * 1.1, 1.0), f"{reason} (near key level)"
        
        return signal, conf, reason
    
    def _near_key_level(self, df: pd.DataFrame, signal: str) -> bool:
        """Check if price is near a key support/resistance level."""
        if len(df) < 50:
            return False
        
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()
        current = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        # Near resistance (for sells) or support (for buys)
        if signal == 'SELL' and abs(current - high_20) < atr:
            return True
        if signal == 'BUY' and abs(current - low_20) < atr:
            return True
        
        return False


# =============================================================================
# FACTORY
# =============================================================================

def create_style_strategy(
    style: TradingStyle,
    data_provider: DataProvider,
    executor: Executor,
    risk_manager: RiskManager,
    predictor: Optional[HybridPredictor] = None,
    config: Optional[StyleConfig] = None,
) -> StyleStrategy:
    """Factory function to create style-specific strategy."""
    
    strategy_classes = {
        TradingStyle.SCALP: ScalpingStrategy,
        TradingStyle.INTRADAY: IntradayStrategy,
        TradingStyle.SWING: SwingStrategy,
    }
    
    cls = strategy_classes.get(style, IntradayStrategy)
    return cls(
        data_provider=data_provider,
        executor=executor,
        risk_manager=risk_manager,
        predictor=predictor,
        config=config,
    )
