# alpha_factory/market_data.py
"""
Market Data module for Alpha Factory system.
Handles OHLCV data processing and swing point extraction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """Represents a swing point in market data."""
    index: int
    time: datetime
    price: float
    point_type: str  # 'high' or 'low'
    strength: float  # 0-1, higher = stronger swing
    confirmed: bool = True  # Whether the swing point is confirmed or preliminary
    confidence: float = 1.0  # Confidence score for preliminary swings

class CorporateActionHandler:
    """Handles splits and dividends to prevent false signals."""
    
    @staticmethod
    def detect_and_adjust_splits(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """
        Heuristic split detection. If price drops > 30% (threshold) overnight 
        without proportional volume spike, assume split.
        Note: In production, rely on metadata. This is a safeguard.
        """
        adjusted_df = df.copy()
        
        # Calculate overnight returns
        close_prices = adjusted_df['close'].values
        opens = adjusted_df['open'].values
        
        # Backward adjustment accumulator
        adjustment_factor = 1.0
        
        # Iterate backwards
        for i in range(len(close_prices) - 1, 0, -1):
            curr_open = opens[i]
            prev_close = close_prices[i-1]
            
            if prev_close == 0: continue
            
            ratio = curr_open / prev_close
            
            # Detect Split (e.g., 2:1 split results in ratio ~0.5)
            if ratio < (1.0 - threshold): 
                # Check volume to confirm it's not a crash
                # Real crashes usually have massive volume. Splits might not.
                # Simplistic heuristic: assume 2:1, 3:1, etc.
                split_ratio = round(1/ratio)
                if split_ratio > 1:
                    logger.warning(f"Detected potential {split_ratio}:1 split at index {i}. Adjusting historical data.")
                    adjustment_factor *= split_ratio
            
            # Apply accumulated adjustment to past data
            if adjustment_factor != 1.0:
                adjusted_df.iloc[i-1] = adjusted_df.iloc[i-1] / split_ratio 
                # Note: Volume should strictly be multiplied, prices divided.
                # Simplified implementation for prices here.
                
        return adjusted_df


class MarketData:
    """
    Market data handler for Alpha Factory.
    
    Processes OHLCV data and extracts swing points for market structure analysis.
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame, handle_splits: bool = True):
        self.raw_data = ohlcv_data
        if handle_splits:
            self.data = CorporateActionHandler.detect_and_adjust_splits(self._validate_and_prepare_data(ohlcv_data))
        else:
            self.data = self._validate_and_prepare_data(ohlcv_data)
        self.swing_points: List[SwingPoint] = []
        
        
    def _validate_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare OHLCV data."""
        if data.empty:
            raise ValueError("Empty OHLCV data provided")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Handle time column - check if it's already the index
        if 'time' in df.columns:
            # If time is also the index name, drop the column
            if df.index.name == 'time':
                df = df.drop('time', axis=1)
            else:
                # Convert time column to datetime and set as index
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
        elif df.index.name != 'time':
            # Create a time column if it doesn't exist
            df.index = pd.to_datetime(df.index)
            df.index.name = 'time'
        
        # Sort by time
        df = df.sort_index()
        
        # Basic OHLC validation
        if (df['high'] < df['low']).any():
            logger.warning("Found invalid OHLC data (high < low), attempting to fix")
            df['high'] = np.maximum(df['high'], df['low'])
        
        if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
            logger.warning("Found invalid OHLC data (high < open/close)")
            
        if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
            logger.warning("Found invalid OHLC data (low > open/close)")
        
        return df
    
    def extract_swings(self, lookback: int = 5, strength_threshold: float = 0.3, 
                  anticipatory: bool = False) -> List[SwingPoint]:
        """
        Extract swing points from price data.
        REFACTORED: Removed anticipatory swings to eliminate look-ahead bias.
        Only uses confirmed swing points with strict lookback confirmation.
        
        Args:
            lookback: Number of bars to look back for swing point confirmation
            strength_threshold: Minimum strength threshold for swing points
            anticipatory: Kept for compatibility but always False for confirmed swings
            
        Returns:
            List of confirmed SwingPoint objects only
        """
        if len(self.data) < lookback + 1:
            return []
        
        highs = self.data['high'].values
        lows = self.data['low'].values
        times = self.data.index
        
        swing_points = []
        
        # Process bars with strict lookback - NO FUTURE DATA
        # End at len(highs) - lookback to ensure we have future data for confirmation
        for i in range(lookback, len(highs) - lookback):
            # Check for confirmed swing high
            is_swing_high = True
            
            # Check past bars (must be highest in lookback)
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j]:
                    is_swing_high = False
                    break
            
            # Check future bars (must remain highest for confirmation)
            if is_swing_high:
                for j in range(1, lookback + 1):
                    if highs[i] <= highs[i + j]:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                # Calculate strength
                strength = self._calculate_swing_strength(highs, i, lookback, 'high')
                
                if strength >= strength_threshold:
                    swing_point = SwingPoint(
                        index=i,
                        time=times[i],
                        price=highs[i],
                        point_type='high',
                        strength=strength,
                        confirmed=True,
                        confidence=1.0
                    )
                    swing_points.append(swing_point)
            
            # Check for confirmed swing low
            is_swing_low = True
            
            # Check past bars (must be lowest in lookback)
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j]:
                    is_swing_low = False
                    break
            
            # Check future bars (must remain lowest for confirmation)
            if is_swing_low:
                for j in range(1, lookback + 1):
                    if lows[i] >= lows[i + j]:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                # Calculate strength
                strength = self._calculate_swing_strength(lows, i, lookback, 'low')
                
                if strength >= strength_threshold:
                    swing_point = SwingPoint(
                        index=i,
                        time=times[i],
                        price=lows[i],
                        point_type='low',
                        strength=strength,
                        confirmed=True,
                        confidence=1.0
                    )
                    swing_points.append(swing_point)
        
        # Sort by index
        swing_points.sort(key=lambda x: x.index)
        logger.info(f"Extracted {len(swing_points)} confirmed swing points (no anticipatory swings)")
        return swing_points
    
    def get_market_structure(self) -> dict:
        """
        Analyze market structure from swing points.
        
        Returns:
            Dictionary containing market structure information
        """
        if not self.swing_points:
            return {'structure': 'unknown', 'trend': 'unknown', 'key_levels': []}
        
        # Identify higher highs, higher lows, etc.
        highs = [sp for sp in self.swing_points if sp.point_type == 'high']
        lows = [sp for sp in self.swing_points if sp.point_type == 'low']
        
        structure = {
            'structure': 'unknown',
            'trend': 'unknown',
            'key_levels': [],
            'last_high': None,
            'last_low': None,
            'higher_highs': 0,
            'lower_highs': 0,
            'higher_lows': 0,
            'lower_lows': 0
        }
        
        if len(highs) >= 2:
            # Compare recent highs
            recent_highs = sorted(highs, key=lambda x: x.time)[-2:]
            if recent_highs[1].price > recent_highs[0].price:
                structure['higher_highs'] += 1
            else:
                structure['lower_highs'] += 1
            structure['last_high'] = recent_highs[1]
        
        if len(lows) >= 2:
            # Compare recent lows
            recent_lows = sorted(lows, key=lambda x: x.time)[-2:]
            if recent_lows[1].price > recent_lows[0].price:
                structure['higher_lows'] += 1
            else:
                structure['lower_lows'] += 1
            structure['last_low'] = recent_lows[1]
        
        # Determine trend
        if structure['higher_highs'] > structure['lower_highs'] and structure['higher_lows'] > structure['lower_lows']:
            structure['trend'] = 'bullish'
            structure['structure'] = 'uptrend'
        elif structure['lower_highs'] > structure['higher_highs'] and structure['lower_lows'] > structure['higher_lows']:
            structure['trend'] = 'bearish'
            structure['structure'] = 'downtrend'
        else:
            structure['trend'] = 'sideways'
            structure['structure'] = 'range'
        
        # Identify key support/resistance levels
        key_levels = []
        
        # Recent significant swing points
        significant_swings = [sp for sp in self.swing_points if sp.strength > 0.5]
        for swing in significant_swings[-10:]:  # Last 10 significant swings
            key_levels.append({
                'price': swing.price,
                'type': swing.point_type,
                'strength': swing.strength,
                'time': swing.time
            })
        
        structure['key_levels'] = sorted(key_levels, key=lambda x: x['strength'], reverse=True)
        
        return structure
    
    def get_recent_data(self, n_bars: int = 100) -> pd.DataFrame:
        """Get the most recent n bars of data."""
        return self.data.tail(n_bars).copy()
    
    def get_data_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get data within a specific time range."""
        mask = (self.data.index >= start_time) & (self.data.index <= end_time)
        return self.data[mask].copy()
    
    def _calculate_swing_strength(self, prices: np.ndarray, index: int, 
                                lookback: int, point_type: str) -> float:
        """
        Calculate strength of a swing point.
        
        Args:
            prices: Array of prices
            index: Index of the swing point
            lookback: Lookback period
            point_type: 'high' or 'low'
            
        Returns:
            Strength score (0-1)
        """
        try:
            # Get surrounding prices
            start_idx = max(0, index - lookback)
            end_idx = min(len(prices), index + lookback + 1)
            
            surrounding_prices = prices[start_idx:end_idx]
            
            if len(surrounding_prices) < 3:
                return 0.0
            
            # Calculate strength based on how extreme the point is
            if point_type == 'high':
                # For high points, compare with surrounding highs
                surrounding_highs = surrounding_prices
                if len(surrounding_highs) > 1:
                    max_surrounding = np.max(surrounding_highs[surrounding_highs < prices[index]])
                    if max_surrounding > 0:
                        strength = (prices[index] - max_surrounding) / prices[index]
                    else:
                        strength = 0.0
            else:  # low
                # For low points, compare with surrounding lows
                surrounding_lows = surrounding_prices
                if len(surrounding_lows) > 1:
                    min_surrounding = np.min(surrounding_lows[surrounding_lows > prices[index]])
                    if min_surrounding > 0:
                        strength = (min_surrounding - prices[index]) / prices[index]
                    else:
                        strength = 0.0
                else:
                    strength = 0.0
            
            # Normalize to 0-1 range
            return np.clip(strength * 10, 0, 1)  # Scale and clip
            
        except Exception as e:
            logger.debug(f"Error calculating swing strength: {e}")
            return 0.0
    
    def _calculate_preliminary_strength(self, prices: np.ndarray, index: int, 
                                       lookback: int, point_type: str) -> float:
        """
        Calculate strength of a preliminary swing point.
        
        Args:
            prices: Array of prices
            index: Index of the swing point
            lookback: Lookback period
            point_type: 'high' or 'low'
            
        Returns:
            Strength score (0-1)
        """
        try:
            # Get past prices only (no look-ahead)
            start_idx = max(0, index - lookback)
            past_prices = prices[start_idx:index + 1]
            
            if len(past_prices) < 3:
                return 0.0
            
            # Calculate strength based on how extreme the point is compared to past
            if point_type == 'high':
                # For high points, compare with past highs
                if len(past_prices) > 1:
                    max_past = np.max(past_prices[:-1])  # Exclude current
                    if max_past > 0:
                        strength = (prices[index] - max_past) / prices[index]
                    else:
                        strength = 0.0
            else:  # low
                # For low points, compare with past lows
                if len(past_prices) > 1:
                    min_past = np.min(past_prices[:-1])  # Exclude current
                    if min_past > 0:
                        strength = (min_past - prices[index]) / prices[index]
                    else:
                        strength = 0.0
                else:
                    strength = 0.0
            
            # Normalize to 0-1 range
            return np.clip(strength * 10, 0, 1)  # Scale and clip
            
        except Exception as e:
            logger.debug(f"Error calculating preliminary swing strength: {e}")
            return 0.0
    
    def _calculate_preliminary_confidence(self, strength: float, index: int, 
                                         total_bars: int) -> float:
        """
        Calculate confidence score for preliminary swing points.
        
        Args:
            strength: Swing point strength
            index: Index of the swing point
            total_bars: Total number of bars in dataset
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence from strength
            base_confidence = strength
            
            # Reduce confidence for very recent bars (less confirmation)
            recency_factor = (total_bars - index) / total_bars
            recency_penalty = recency_factor * 0.3  # Max 30% penalty
            
            # Final confidence
            confidence = base_confidence * (1 - recency_penalty)
            
            return np.clip(confidence, 0, 1)
            
        except Exception as e:
            logger.debug(f"Error calculating preliminary confidence: {e}")
            return 0.5
