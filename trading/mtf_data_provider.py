# trading/mtf_data_provider.py
"""
Multi-Timeframe Data Provider

Handles fetching, caching, and synchronizing data across multiple timeframes.
Works with both live MT5 connection and backtest data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import conditionally to avoid circular imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


@dataclass
class MTFDataCache:
    """Cache for multi-timeframe data."""
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    last_update: Dict[str, datetime] = field(default_factory=dict)
    update_intervals: Dict[str, int] = field(default_factory=dict)  # Minutes
    
    def is_stale(self, tf: str) -> bool:
        """Check if data for timeframe is stale."""
        if tf not in self.last_update:
            return True
        
        interval = self.update_intervals.get(tf, 1)
        age = (datetime.now() - self.last_update[tf]).total_seconds() / 60
        return age >= interval
    
    def update(self, tf: str, df: pd.DataFrame):
        """Update cached data for timeframe."""
        self.data[tf] = df
        self.last_update[tf] = datetime.now()
    
    def get(self, tf: str) -> Optional[pd.DataFrame]:
        """Get cached data if not stale."""
        if tf in self.data and not self.is_stale(tf):
            return self.data[tf]
        return None
    
    def clear(self):
        """Clear all cached data."""
        self.data.clear()
        self.last_update.clear()


class MTFDataProvider:
    """
    Provides synchronized multi-timeframe data for analysis.
    
    Features:
    - Fetches data for multiple timeframes
    - Caches data to avoid excessive API calls
    - Aligns timestamps across timeframes
    - Supports both live and backtest modes
    """
    
    # MT5 timeframe mapping
    TF_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440
    }
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        connector: Optional[Any] = None,
        cache_enabled: bool = True,
    ):
        self.symbol = symbol
        self.connector = connector
        self.cache_enabled = cache_enabled
        
        # Initialize cache with update intervals based on timeframe
        self.cache = MTFDataCache(
            update_intervals={
                "M1": 1, "M5": 1, "M15": 1, "M30": 5,
                "H1": 5, "H4": 15, "D1": 60
            }
        )
        
        self._mt5_initialized = False
    
    def _ensure_mt5(self) -> bool:
        """Ensure MT5 is initialized."""
        if self.connector is not None:
            return self.connector.ensure_connected()
        
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
            return False
        
        if not self._mt5_initialized:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            self._mt5_initialized = True
        
        return True
    
    def _get_mt5_timeframe(self, tf: str) -> int:
        """Convert string timeframe to MT5 constant."""
        if not MT5_AVAILABLE:
            return self.TF_MAP.get(tf.upper(), 60)
        
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(tf.upper(), mt5.TIMEFRAME_H1)
    
    def fetch_single_timeframe(
        self,
        timeframe: str,
        n_candles: int = 200,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data for a single timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., "M15", "H1")
            n_candles: Number of candles to fetch
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        tf = timeframe.upper()
        
        # Check cache
        if use_cache and self.cache_enabled:
            cached = self.cache.get(tf)
            if cached is not None and len(cached) >= n_candles:
                return cached.tail(n_candles).copy()
        
        # Use connector if available
        if self.connector is not None:
            df = self.connector.get_data(n=n_candles, timeframe=tf)
        else:
            df = self._fetch_from_mt5(tf, n_candles)
        
        if df is not None and not df.empty:
            # Normalize columns
            df = self._normalize_dataframe(df)
            
            # Update cache
            if self.cache_enabled:
                self.cache.update(tf, df)
        
        return df if df is not None else pd.DataFrame()
    
    def _fetch_from_mt5(self, tf: str, n_candles: int) -> Optional[pd.DataFrame]:
        """Fetch data directly from MT5."""
        if not self._ensure_mt5():
            return None
        
        mt5_tf = self._get_mt5_timeframe(tf)
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, n_candles)
        
        if rates is None:
            logger.warning(f"Failed to fetch {tf} data: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        return df
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and types."""
        df = df.copy()
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Handle time column
        if 'time' in df.columns:
            if df['time'].dtype in ['int64', 'float64']:
                df['time'] = pd.to_datetime(df['time'], unit='s')
            else:
                df['time'] = pd.to_datetime(df['time'])
        
        # Rename volume if needed
        if 'volume' in df.columns and 'tick_volume' not in df.columns:
            df.rename(columns={'volume': 'tick_volume'}, inplace=True)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def fetch_mtf_data(
        self,
        timeframes: List[str],
        candle_counts: Optional[Dict[str, int]] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.
        
        Args:
            timeframes: List of timeframe strings
            candle_counts: Dict mapping timeframe to candle count
            use_cache: Whether to use cached data
        
        Returns:
            Dict mapping timeframe to DataFrame
        """
        candle_counts = candle_counts or {}
        result = {}
        
        for tf in timeframes:
            tf = tf.upper()
            n_candles = candle_counts.get(tf, 200)
            
            df = self.fetch_single_timeframe(tf, n_candles, use_cache)
            
            if df.empty:
                logger.warning(f"Empty data for {tf}")
            
            result[tf] = df
        
        return result
    
    def fetch_for_profile(
        self,
        profile: "MTFProfile",
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data according to MTF profile configuration.
        
        Args:
            profile: MTFProfile configuration
            use_cache: Whether to use cached data
        
        Returns:
            Dict mapping timeframe to DataFrame
        """
        return self.fetch_mtf_data(
            timeframes=profile.timeframe_strings,
            candle_counts=profile.candle_counts,
            use_cache=use_cache,
        )
    
    def get_aligned_data(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        align_to: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Align data across timeframes to common time range.
        
        Args:
            dfs_dict: Dict of timeframe DataFrames
            align_to: Timeframe to align to (uses longest TF if None)
        
        Returns:
            Dict of aligned DataFrames
        """
        if not dfs_dict:
            return {}
        
        # Find common time range
        starts = []
        ends = []
        
        for tf, df in dfs_dict.items():
            if not df.empty and 'time' in df.columns:
                starts.append(df['time'].min())
                ends.append(df['time'].max())
        
        if not starts or not ends:
            return dfs_dict
        
        common_start = max(starts)
        common_end = min(ends)
        
        # Filter each DataFrame to common range
        result = {}
        for tf, df in dfs_dict.items():
            if not df.empty and 'time' in df.columns:
                mask = (df['time'] >= common_start) & (df['time'] <= common_end)
                result[tf] = df[mask].copy().reset_index(drop=True)
            else:
                result[tf] = df
        
        return result
    
    def validate_data(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        min_bars: int = 50,
    ) -> Tuple[bool, List[str]]:
        """
        Validate fetched data meets requirements.
        
        Args:
            dfs_dict: Dict of timeframe DataFrames
            min_bars: Minimum bars required per timeframe
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        for tf, df in dfs_dict.items():
            if df is None or df.empty:
                errors.append(f"{tf}: No data")
                continue
            
            if len(df) < min_bars:
                errors.append(f"{tf}: Only {len(df)} bars (need {min_bars})")
            
            required = ['open', 'high', 'low', 'close']
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"{tf}: Missing columns {missing}")
            
            # Check for NaN
            nan_counts = df[required].isna().sum()
            if nan_counts.any():
                errors.append(f"{tf}: NaN values in {nan_counts[nan_counts > 0].to_dict()}")
        
        return len(errors) == 0, errors
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("MTF data cache cleared")
    
    def fetch_multiple_timeframes(self, timeframes: list, n_candles: int = 200, use_cache: bool = True) -> dict:
        """Fetch data for multiple timeframes."""
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_single_timeframe(tf, n_candles, use_cache)
        return result
    
    def get_timeframe_data(self, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
        """Get timeframe data (alias for fetch_single_timeframe)."""
        return self.fetch_single_timeframe(timeframe, n_candles)
    
    def refresh_timeframe(self, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
        """Refresh specific timeframe data."""
        return self.fetch_single_timeframe(timeframe, n_candles, use_cache=False)
    
    def refresh_all_timeframes(self, timeframes: list = None, n_candles: int = 200) -> dict:
        """Refresh all cached timeframes or specific timeframes."""
        if timeframes is None:
            timeframes = list(self.cache.keys()) if self.cache_enabled else []
        result = {}
        for tf in timeframes:
            result[tf] = self.refresh_timeframe(tf, n_candles)
        return result
    
    def get_latest_bar_time(self, timeframe: str) -> datetime:
        """Get the latest bar time for a timeframe."""
        df = self.fetch_single_timeframe(timeframe, 1)
        if not df.empty:
            return df['time'].iloc[-1]
        return datetime.now()


class BacktestMTFDataProvider(MTFDataProvider):
    """
    MTF Data Provider for backtesting.
    
    Uses pre-loaded historical data instead of live MT5 connection.
    Simulates real-time data access by sliding through history.
    """
    
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        symbol: str = "EURUSD",
    ):
        """
        Args:
            historical_data: Dict mapping timeframe to full historical DataFrame
            symbol: Symbol name
        """
        super().__init__(symbol=symbol, connector=None, cache_enabled=False)
        
        self.historical_data = {}
        for tf, df in historical_data.items():
            self.historical_data[tf.upper()] = self._normalize_dataframe(df)
        
        # Current position in each timeframe
        self.current_idx: Dict[str, int] = {}
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize current index to start of each timeframe."""
        for tf in self.historical_data:
            self.current_idx[tf] = 0
    
    def set_time(self, target_time: datetime):
        """
        Set the current time position for all timeframes.
        
        Args:
            target_time: Target datetime to seek to
        """
        for tf, df in self.historical_data.items():
            if 'time' not in df.columns:
                continue
            
            # Find index of bar at or before target_time
            mask = df['time'] <= target_time
            if mask.any():
                self.current_idx[tf] = mask.sum() - 1
            else:
                self.current_idx[tf] = 0
    
    def advance(self, timeframe: str = None):
        """
        Advance to next bar.
        
        Args:
            timeframe: Specific timeframe to advance (all if None)
        """
        if timeframe:
            tfs = [timeframe.upper()]
        else:
            tfs = list(self.current_idx.keys())
        
        for tf in tfs:
            if tf in self.current_idx:
                max_idx = len(self.historical_data[tf]) - 1
                self.current_idx[tf] = min(self.current_idx[tf] + 1, max_idx)
    
    def fetch_single_timeframe(
        self,
        timeframe: str,
        n_candles: int = 200,
        use_cache: bool = False,  # Ignored in backtest
    ) -> pd.DataFrame:
        """Fetch historical data up to current position."""
        tf = timeframe.upper()
        
        if tf not in self.historical_data:
            logger.warning(f"No historical data for {tf}")
            return pd.DataFrame()
        
        current_idx = self.current_idx.get(tf, 0)
        start_idx = max(0, current_idx - n_candles + 1)
        
        df = self.historical_data[tf].iloc[start_idx:current_idx + 1].copy()
        return df.reset_index(drop=True)
    
    def get_current_time(self, timeframe: str) -> Optional[datetime]:
        """Get current bar time for a timeframe."""
        tf = timeframe.upper()
        
        if tf not in self.historical_data:
            return None
        
        idx = self.current_idx.get(tf, 0)
        df = self.historical_data[tf]
        
        if 'time' in df.columns and idx < len(df):
            return df['time'].iloc[idx]
        
        return None
    
    def is_end_of_data(self, timeframe: str = None) -> bool:
        """Check if reached end of historical data."""
        if timeframe:
            tf = timeframe.upper()
            return self.current_idx.get(tf, 0) >= len(self.historical_data.get(tf, [])) - 1
        
        # Check all timeframes
        return all(
            self.current_idx.get(tf, 0) >= len(df) - 1
            for tf, df in self.historical_data.items()
        )


def create_mock_mtf_data(
    n_candles: int = 500,
    timeframes: List[str] = ["M15", "H1", "H4"],
    base_price: float = 1.1000,
    volatility: float = 0.0010,
) -> Dict[str, pd.DataFrame]:
    """
    Create mock MTF data for testing.
    
    Args:
        n_candles: Number of candles per timeframe
        timeframes: List of timeframes to generate
        base_price: Starting price
        volatility: Price volatility
    
    Returns:
        Dict mapping timeframe to DataFrame
    """
    np.random.seed(42)
    result = {}
    
    for tf in timeframes:
        prices = [base_price]
        
        # Simulate price movement
        for _ in range(n_candles - 1):
            change = np.random.randn() * volatility
            prices.append(prices[-1] + change)
        
        prices = np.array(prices)
        
        # Generate OHLCV
        data = []
        tf_minutes = MTFDataProvider.TF_MAP.get(tf.upper(), 60)
        start_time = datetime.now() - timedelta(minutes=tf_minutes * n_candles)
        
        for i, price in enumerate(prices):
            high = price + abs(np.random.randn() * volatility * 0.5)
            low = price - abs(np.random.randn() * volatility * 0.5)
            close = price + np.random.randn() * volatility * 0.3
            
            data.append({
                'time': start_time + timedelta(minutes=tf_minutes * i),
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'tick_volume': np.random.randint(100, 2000),
            })
        
        result[tf.upper()] = pd.DataFrame(data)
