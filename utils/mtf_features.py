# utils/mtf_features.py
"""
Multi-Timeframe Feature Engineering

Generates comprehensive features from MTF data for:
- ML model input
- Signal generation
- Trend analysis

Features include:
- Per-timeframe indicators (EMA, ADX, RSI, etc.)
- Cross-timeframe confluence measures
- Momentum and volatility metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MTFFeatureSet:
    """Container for MTF features."""
    features: Dict[str, float]
    feature_names: List[str]
    primary_tf: str
    timestamp: Optional[pd.Timestamp] = None
    
    def to_array(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert to numpy array in specified order."""
        order = feature_order or self.feature_names
        return np.array([self.features.get(f, 0.0) for f in order])
    
    def to_dict(self) -> Dict[str, float]:
        """Return features as dict."""
        return self.features.copy()


class MTFFeatureBuilder:
    """
    Builds feature vectors from multi-timeframe data.
    
    Features generated per timeframe:
    - EMA positions and slopes
    - ADX, +DI, -DI
    - RSI
    - ATR (normalized)
    - Price position in range
    - Momentum (ROC)
    
    Cross-timeframe features:
    - Direction alignment
    - Weighted direction score
    - Confluence strength
    """
    
    def __init__(
        self,
        ema_periods: Tuple[int, ...] = (20, 50, 200),
        adx_period: int = 14,
        rsi_period: int = 14,
        atr_period: int = 14,
        roc_periods: Tuple[int, ...] = (5, 10, 20),
        lookback: int = 20,
    ):
        self.ema_periods = ema_periods
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.roc_periods = roc_periods
        self.lookback = lookback
    
    def build_features(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        primary_tf: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> MTFFeatureSet:
        """
        Build comprehensive feature set from MTF data.
        
        Args:
            dfs_dict: Dict mapping timeframe to DataFrame
            primary_tf: Primary timeframe for reference
            weights: Optional weights for cross-TF features
        
        Returns:
            MTFFeatureSet with all computed features
        """
        features = {}
        feature_names = []
        
        # Per-timeframe features
        for tf, df in dfs_dict.items():
            if df is None or df.empty:
                continue
            
            tf_features = self._compute_tf_features(df, tf)
            features.update(tf_features)
            feature_names.extend(tf_features.keys())
        
        # Cross-timeframe features
        cross_features = self._compute_cross_tf_features(dfs_dict, weights)
        features.update(cross_features)
        feature_names.extend(cross_features.keys())
        
        # Get timestamp
        timestamp = None
        if primary_tf in dfs_dict and 'time' in dfs_dict[primary_tf].columns:
            timestamp = dfs_dict[primary_tf]['time'].iloc[-1]
        
        return MTFFeatureSet(
            features=features,
            feature_names=list(features.keys()),
            primary_tf=primary_tf,
            timestamp=timestamp,
        )
    
    def _compute_tf_features(
        self,
        df: pd.DataFrame,
        tf: str,
    ) -> Dict[str, float]:
        """Compute features for a single timeframe."""
        features = {}
        prefix = f"{tf}_"
        
        close = df['close']
        high = df['high']
        low = df['low']
        current_price = close.iloc[-1]
        
        # EMA features
        for period in self.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()
            ema_val = ema.iloc[-1]
            
            # Price vs EMA (-1 to 1)
            features[f"{prefix}price_vs_ema{period}"] = 1 if current_price > ema_val else -1
            
            # EMA slope (normalized)
            if len(ema) > self.lookback:
                slope = (ema.iloc[-1] - ema.iloc[-self.lookback]) / ema.iloc[-self.lookback]
                features[f"{prefix}ema{period}_slope"] = np.tanh(slope * 50)
            else:
                features[f"{prefix}ema{period}_slope"] = 0.0
        
        # EMA alignment score
        emas = [close.ewm(span=p, adjust=False).mean().iloc[-1] for p in self.ema_periods]
        if emas[0] > emas[1] > emas[2]:
            features[f"{prefix}ema_alignment"] = 1.0  # Bullish alignment
        elif emas[0] < emas[1] < emas[2]:
            features[f"{prefix}ema_alignment"] = -1.0  # Bearish alignment
        else:
            features[f"{prefix}ema_alignment"] = 0.0  # Mixed
        
        # ADX features
        adx, plus_di, minus_di = self._calculate_adx(df)
        features[f"{prefix}adx"] = adx
        features[f"{prefix}plus_di"] = plus_di
        features[f"{prefix}minus_di"] = minus_di
        features[f"{prefix}di_diff"] = plus_di - minus_di
        
        # ADX trend strength category
        if adx > 40:
            features[f"{prefix}trend_strength"] = 1.0
        elif adx > 25:
            features[f"{prefix}trend_strength"] = 0.5
        else:
            features[f"{prefix}trend_strength"] = 0.0
        
        # RSI
        rsi = self._calculate_rsi(close)
        features[f"{prefix}rsi"] = rsi
        features[f"{prefix}rsi_zone"] = self._rsi_zone(rsi)
        
        # ATR (normalized by price)
        atr = self._calculate_atr(df)
        features[f"{prefix}atr_pct"] = (atr / current_price) * 100
        
        # Volatility compression
        features[f"{prefix}vol_compression"] = self._volatility_compression(df)
        
        # Price position in range
        high_20 = high.tail(20).max()
        low_20 = low.tail(20).min()
        range_size = high_20 - low_20
        if range_size > 0:
            features[f"{prefix}range_position"] = (current_price - low_20) / range_size
        else:
            features[f"{prefix}range_position"] = 0.5
        
        # Momentum (ROC)
        for period in self.roc_periods:
            if len(close) > period:
                roc = ((close.iloc[-1] / close.iloc[-period-1]) - 1) * 100
                features[f"{prefix}roc_{period}"] = roc
            else:
                features[f"{prefix}roc_{period}"] = 0.0
        
        # Candle features
        features[f"{prefix}body_ratio"] = self._body_ratio(df)
        features[f"{prefix}upper_shadow"] = self._upper_shadow_ratio(df)
        features[f"{prefix}lower_shadow"] = self._lower_shadow_ratio(df)
        
        return features
    
    def _compute_cross_tf_features(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute cross-timeframe features."""
        features = {}
        
        if not dfs_dict:
            return features
        
        # Default equal weights
        if weights is None:
            weights = {tf: 1.0 / len(dfs_dict) for tf in dfs_dict}
        
        # Direction per TF
        directions = {}
        strengths = {}
        
        for tf, df in dfs_dict.items():
            if df is None or df.empty:
                continue
            
            # Simple direction from EMA
            close = df['close']
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            
            if close.iloc[-1] > ema20 > ema50:
                directions[tf] = 1
            elif close.iloc[-1] < ema20 < ema50:
                directions[tf] = -1
            else:
                directions[tf] = 0
            
            # Strength from ADX
            adx, _, _ = self._calculate_adx(df)
            strengths[tf] = min(adx / 50, 1.0)
        
        # Weighted direction
        weighted_dir = sum(
            directions.get(tf, 0) * weights.get(tf, 0) * strengths.get(tf, 0.5)
            for tf in dfs_dict
        )
        features['mtf_weighted_direction'] = np.tanh(weighted_dir * 2)
        
        # Direction alignment (what % agree)
        if directions:
            dir_values = list(directions.values())
            bullish = sum(1 for d in dir_values if d > 0)
            bearish = sum(1 for d in dir_values if d < 0)
            features['mtf_direction_alignment'] = max(bullish, bearish) / len(dir_values)
            features['mtf_bullish_count'] = bullish
            features['mtf_bearish_count'] = bearish
        else:
            features['mtf_direction_alignment'] = 0.0
            features['mtf_bullish_count'] = 0
            features['mtf_bearish_count'] = 0
        
        # Average strength
        if strengths:
            features['mtf_avg_strength'] = np.mean(list(strengths.values()))
        else:
            features['mtf_avg_strength'] = 0.0
        
        # Confluence score (direction agreement * strength)
        features['mtf_confluence'] = (
            features['mtf_direction_alignment'] * features['mtf_avg_strength']
        )
        
        return features
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']
        period = self.adx_period
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return (
            float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0,
            float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0,
            float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0,
        )
    
    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    def _rsi_zone(self, rsi: float) -> float:
        """Map RSI to zone: -1 (oversold), 0 (neutral), 1 (overbought)."""
        if rsi > 70:
            return 1.0
        elif rsi < 30:
            return -1.0
        else:
            return 0.0
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
    
    def _volatility_compression(self, df: pd.DataFrame) -> float:
        """Calculate volatility compression ratio."""
        atr = self._calculate_atr(df)
        
        # Long-term ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_long = tr.rolling(window=50).mean()
        
        atr_long_val = float(atr_long.iloc[-1]) if not pd.isna(atr_long.iloc[-1]) else atr
        
        if atr_long_val > 0:
            return atr / atr_long_val
        return 1.0
    
    def _body_ratio(self, df: pd.DataFrame) -> float:
        """Calculate body to range ratio of last candle."""
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        range_size = h - l
        if range_size > 0:
            return abs(c - o) / range_size
        return 0.5
    
    def _upper_shadow_ratio(self, df: pd.DataFrame) -> float:
        """Calculate upper shadow ratio."""
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        range_size = h - l
        body_top = max(o, c)
        if range_size > 0:
            return (h - body_top) / range_size
        return 0.0
    
    def _lower_shadow_ratio(self, df: pd.DataFrame) -> float:
        """Calculate lower shadow ratio."""
        o, h, l, c = df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        range_size = h - l
        body_bottom = min(o, c)
        if range_size > 0:
            return (body_bottom - l) / range_size
        return 0.0


def build_ml_features(
    dfs_dict: Dict[str, pd.DataFrame],
    primary_tf: str = "H1",
) -> Dict[str, float]:
    """
    Convenience function to build ML features.
    
    Args:
        dfs_dict: Dict mapping timeframe to DataFrame
        primary_tf: Primary timeframe
    
    Returns:
        Dict of feature name to value
    """
    builder = MTFFeatureBuilder()
    feature_set = builder.build_features(dfs_dict, primary_tf)
    return feature_set.features