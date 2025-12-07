# utils/mtf_features.py
"""
Multi-Timeframe Feature Engineering

Generates features from MTF data for ML models:
- Per-timeframe technical indicators
- Cross-timeframe relationships
- Confluence features
- Time-based features
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MTFFeatureSet:
    """Container for MTF features."""
    features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    timeframes_used: List[str] = field(default_factory=list)
    timestamp: Optional[pd.Timestamp] = None
    
    def to_array(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert to numpy array."""
        if feature_order:
            return np.array([self.features.get(f, 0) for f in feature_order])
        return np.array(list(self.features.values()))
    
    def to_dict(self) -> Dict[str, float]:
        """Return features as dict."""
        return self.features.copy()


class MTFFeatureBuilder:
    """
    Builds comprehensive feature set from multi-timeframe data.
    
    Features generated:
    1. Per-TF indicators (EMA, RSI, ADX, etc.)
    2. Cross-TF features (alignment, agreement)
    3. Structural features (swing patterns)
    4. Volatility features
    """
    
    def __init__(
        self,
        ema_periods: Tuple[int, ...] = (10, 20, 50, 200),
        rsi_period: int = 14,
        adx_period: int = 14,
        atr_period: int = 14,
        roc_periods: Tuple[int, ...] = (5, 10, 20),
        bb_period: int = 20,
    ):
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.roc_periods = roc_periods
        self.bb_period = bb_period
    
    def build_features(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        primary_tf: str = "H1",
    ) -> MTFFeatureSet:
        """
        Build comprehensive feature set from MTF data.
        
        Args:
            dfs_dict: Dict mapping timeframe to DataFrame
            primary_tf: Primary timeframe for relative features
        
        Returns:
            MTFFeatureSet with all computed features
        """
        features = {}
        timeframes_used = list(dfs_dict.keys())
        
        # 1. Per-timeframe features
        for tf, df in dfs_dict.items():
            if df is None or df.empty:
                continue
            
            tf_features = self._compute_tf_features(df, tf)
            features.update(tf_features)
        
        # 2. Cross-timeframe features
        if len(dfs_dict) > 1:
            cross_features = self._compute_cross_tf_features(dfs_dict, primary_tf)
            features.update(cross_features)
        
        # 3. Confluence features
        confluence_features = self._compute_confluence_features(dfs_dict)
        features.update(confluence_features)
        
        # Get timestamp from primary TF
        timestamp = None
        if primary_tf in dfs_dict and 'time' in dfs_dict[primary_tf].columns:
            timestamp = dfs_dict[primary_tf]['time'].iloc[-1]
        
        return MTFFeatureSet(
            features=features,
            feature_names=list(features.keys()),
            timeframes_used=timeframes_used,
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
        
        # EMAs and EMA relationships
        emas = {}
        for period in self.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()
            emas[period] = ema
            features[f"{prefix}ema{period}"] = float(ema.iloc[-1])
            
            # Price position relative to EMA
            features[f"{prefix}price_vs_ema{period}"] = (
                1.0 if close.iloc[-1] > ema.iloc[-1] else -1.0
            )
        
        # EMA slopes (normalized)
        for period in [20, 50]:
            if period in emas:
                ema = emas[period]
                if len(ema) > 5:
                    slope = (ema.iloc[-1] - ema.iloc[-5]) / (ema.iloc[-5] + 1e-10) * 100
                    features[f"{prefix}ema{period}_slope"] = float(slope)
        
        # EMA alignment score
        if all(p in emas for p in [20, 50, 200]):
            e20, e50, e200 = emas[20].iloc[-1], emas[50].iloc[-1], emas[200].iloc[-1]
            if e20 > e50 > e200:
                features[f"{prefix}ema_alignment"] = 1.0
            elif e20 < e50 < e200:
                features[f"{prefix}ema_alignment"] = -1.0
            else:
                features[f"{prefix}ema_alignment"] = 0.0
        
        # RSI
        rsi = self._compute_rsi(close)
        features[f"{prefix}rsi"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        
        # RSI zones
        rsi_val = features[f"{prefix}rsi"]
        features[f"{prefix}rsi_zone"] = (
            1 if rsi_val > 70 else (-1 if rsi_val < 30 else 0)
        )
        
        # ADX and DI
        adx, plus_di, minus_di = self._compute_adx(df)
        features[f"{prefix}adx"] = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
        features[f"{prefix}plus_di"] = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0
        features[f"{prefix}minus_di"] = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0
        
        # DI direction
        features[f"{prefix}di_direction"] = (
            1 if features[f"{prefix}plus_di"] > features[f"{prefix}minus_di"] else -1
        )
        
        # Trend strength category
        adx_val = features[f"{prefix}adx"]
        features[f"{prefix}trend_strength"] = (
            1 if adx_val > 25 else 0
        )
        
        # ATR (normalized)
        atr = self._compute_atr(df)
        features[f"{prefix}atr"] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
        features[f"{prefix}atr_pct"] = features[f"{prefix}atr"] / close.iloc[-1] * 100
        
        # ROC (Rate of Change)
        for period in self.roc_periods:
            if len(close) > period:
                roc = (close.iloc[-1] / close.iloc[-period] - 1) * 100
                features[f"{prefix}roc_{period}"] = float(roc)
        
        # Bollinger Bands
        bb_sma = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        
        if not pd.isna(bb_sma.iloc[-1]) and not pd.isna(bb_std.iloc[-1]):
            upper_bb = bb_sma.iloc[-1] + 2 * bb_std.iloc[-1]
            lower_bb = bb_sma.iloc[-1] - 2 * bb_std.iloc[-1]
            bb_width = (upper_bb - lower_bb) / bb_sma.iloc[-1] * 100
            
            features[f"{prefix}bb_width"] = float(bb_width)
            features[f"{prefix}bb_position"] = (
                (close.iloc[-1] - lower_bb) / (upper_bb - lower_bb + 1e-10)
            )
        
        # Volatility compression
        if len(atr) > 50:
            atr_ma = atr.rolling(50).mean()
            if not pd.isna(atr_ma.iloc[-1]):
                vol_compression = atr.iloc[-1] / (atr_ma.iloc[-1] + 1e-10)
                features[f"{prefix}vol_compression"] = float(vol_compression)
        
        # Price position in recent range
        if len(high) >= 20:
            high_20 = high.tail(20).max()
            low_20 = low.tail(20).min()
            range_position = (close.iloc[-1] - low_20) / (high_20 - low_20 + 1e-10)
            features[f"{prefix}range_position"] = float(range_position)
        
        # Candle features (current candle)
        body = abs(close.iloc[-1] - df['open'].iloc[-1])
        total_range = high.iloc[-1] - low.iloc[-1]
        
        features[f"{prefix}body_ratio"] = body / (total_range + 1e-10)
        features[f"{prefix}is_bullish"] = 1 if close.iloc[-1] > df['open'].iloc[-1] else 0
        
        return features
    
    def _compute_cross_tf_features(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        primary_tf: str,
    ) -> Dict[str, float]:
        """Compute features that compare across timeframes."""
        features = {}
        
        # Get list of timeframes sorted by duration
        tf_minutes = {'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
        sorted_tfs = sorted(
            [tf for tf in dfs_dict.keys() if tf in tf_minutes],
            key=lambda x: tf_minutes[x]
        )
        
        if len(sorted_tfs) < 2:
            return features
        
        # Compare primary to higher TF
        primary_idx = sorted_tfs.index(primary_tf) if primary_tf in sorted_tfs else 0
        
        if primary_idx < len(sorted_tfs) - 1:
            higher_tf = sorted_tfs[primary_idx + 1]
            
            # Compare EMA alignments
            primary_df = dfs_dict[primary_tf]
            higher_df = dfs_dict[higher_tf]
            
            # Get EMA directions
            primary_ema20 = primary_df['close'].ewm(span=20).mean()
            higher_ema20 = higher_df['close'].ewm(span=20).mean()
            
            primary_slope = (primary_ema20.iloc[-1] - primary_ema20.iloc[-5]) if len(primary_ema20) > 5 else 0
            higher_slope = (higher_ema20.iloc[-1] - higher_ema20.iloc[-5]) if len(higher_ema20) > 5 else 0
            
            # Same direction?
            features['cross_ema_agreement'] = 1 if (primary_slope * higher_slope) > 0 else 0
            
            # Relative volatility
            primary_atr = self._compute_atr(primary_df)
            higher_atr = self._compute_atr(higher_df)
            
            if not pd.isna(primary_atr.iloc[-1]) and not pd.isna(higher_atr.iloc[-1]):
                features['cross_vol_ratio'] = (
                    primary_atr.iloc[-1] / (higher_atr.iloc[-1] + 1e-10)
                )
        
        # Compare all TFs for trend alignment
        directions = []
        for tf in sorted_tfs:
            df = dfs_dict[tf]
            ema20 = df['close'].ewm(span=20).mean()
            ema50 = df['close'].ewm(span=50).mean()
            
            if ema20.iloc[-1] > ema50.iloc[-1]:
                directions.append(1)
            elif ema20.iloc[-1] < ema50.iloc[-1]:
                directions.append(-1)
            else:
                directions.append(0)
        
        # Alignment score: 1 if all same direction, 0 if mixed
        if all(d == directions[0] for d in directions):
            features['mtf_alignment'] = abs(directions[0])
        else:
            features['mtf_alignment'] = 0
        
        # Weighted direction
        weights = [tf_minutes[tf] for tf in sorted_tfs]
        total_weight = sum(weights)
        weighted_dir = sum(d * w for d, w in zip(directions, weights)) / total_weight
        features['mtf_weighted_direction'] = weighted_dir
        
        return features
    
    def _compute_confluence_features(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compute confluence/agreement features."""
        features = {}
        
        if len(dfs_dict) < 2:
            return features
        
        # RSI agreement
        rsi_values = []
        for tf, df in dfs_dict.items():
            rsi = self._compute_rsi(df['close'])
            if not pd.isna(rsi.iloc[-1]):
                rsi_values.append(rsi.iloc[-1])
        
        if rsi_values:
            features['confluence_rsi_mean'] = np.mean(rsi_values)
            features['confluence_rsi_std'] = np.std(rsi_values)
            
            # RSI zone agreement
            zones = [1 if r > 70 else (-1 if r < 30 else 0) for r in rsi_values]
            features['confluence_rsi_zone_agree'] = 1 if len(set(zones)) == 1 else 0
        
        # ADX agreement
        adx_values = []
        for tf, df in dfs_dict.items():
            adx, _, _ = self._compute_adx(df)
            if not pd.isna(adx.iloc[-1]):
                adx_values.append(adx.iloc[-1])
        
        if adx_values:
            features['confluence_adx_mean'] = np.mean(adx_values)
            # All trending or all ranging?
            trending = [1 if a > 25 else 0 for a in adx_values]
            features['confluence_trend_agree'] = 1 if len(set(trending)) == 1 else 0
        
        return features
    
    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """Compute RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _compute_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']
        period = self.adx_period
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=self.atr_period).mean()
    
    def get_feature_names(
        self,
        timeframes: List[str],
    ) -> List[str]:
        """Get list of all feature names that would be generated."""
        names = []
        
        # Per-TF features
        for tf in timeframes:
            prefix = f"{tf}_"
            
            for period in self.ema_periods:
                names.append(f"{prefix}ema{period}")
                names.append(f"{prefix}price_vs_ema{period}")
            
            names.extend([
                f"{prefix}ema20_slope",
                f"{prefix}ema50_slope",
                f"{prefix}ema_alignment",
                f"{prefix}rsi",
                f"{prefix}rsi_zone",
                f"{prefix}adx",
                f"{prefix}plus_di",
                f"{prefix}minus_di",
                f"{prefix}di_direction",
                f"{prefix}trend_strength",
                f"{prefix}atr",
                f"{prefix}atr_pct",
            ])
            
            for period in self.roc_periods:
                names.append(f"{prefix}roc_{period}")
            
            names.extend([
                f"{prefix}bb_width",
                f"{prefix}bb_position",
                f"{prefix}vol_compression",
                f"{prefix}range_position",
                f"{prefix}body_ratio",
                f"{prefix}is_bullish",
            ])
        
        # Cross-TF features
        names.extend([
            'cross_ema_agreement',
            'cross_vol_ratio',
            'mtf_alignment',
            'mtf_weighted_direction',
        ])
        
        # Confluence features
        names.extend([
            'confluence_rsi_mean',
            'confluence_rsi_std',
            'confluence_rsi_zone_agree',
            'confluence_adx_mean',
            'confluence_trend_agree',
        ])
        
        return names


def build_mtf_features_for_training(
    dfs_dict: Dict[str, pd.DataFrame],
    lookback_window: int = 60,
    stride: int = 1,
    primary_tf: str = "H1",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix for training from historical data.
    
    Args:
        dfs_dict: Dict mapping timeframe to historical DataFrame
        lookback_window: Window size for each sample
        stride: Step between samples
        primary_tf: Primary timeframe for labeling
    
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    builder = MTFFeatureBuilder()
    
    # Get the shortest DataFrame length
    min_len = min(len(df) for df in dfs_dict.values() if df is not None and not df.empty)
    
    features_list = []
    
    for i in range(lookback_window, min_len, stride):
        # Create window for each timeframe
        window_dict = {}
        for tf, df in dfs_dict.items():
            window_dict[tf] = df.iloc[i - lookback_window:i].copy()
        
        # Build features
        feature_set = builder.build_features(window_dict, primary_tf)
        features_list.append(feature_set.to_dict())
    
    if not features_list:
        return np.array([]), []
    
    # Convert to matrix
    feature_names = list(features_list[0].keys())
    matrix = np.array([
        [f.get(name, 0) for name in feature_names]
        for f in features_list
    ])
    
    return matrix, feature_names
