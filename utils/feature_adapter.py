# utils/feature_adapter.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Import your unified Engine
try:
    from utils.features_engineering import FeatureEngineerOptimized
except ImportError:
    raise ImportError("Could not find 'utils/features_engineering.py'.")

logger = logging.getLogger(__name__)

class EnhancedDataLoaderV2:
    def __init__(self, sequence_length: int = 30, trend_threshold: float = 0.05):
        self.sequence_length = sequence_length
        self.trend_threshold = trend_threshold
        self.feature_columns = []
        # Initialize Engine (No DB needed for training)
        self.engine = FeatureEngineerOptimized(db_connector=None)

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Smart Loader: Checks for a cached 'engineered' version before calculating.
        """
        raw_path = Path(filepath)
        
        # Define Cache Path: data/raw/eurusd.csv -> data/processed/eurusd_engineered.csv
        # We assume project structure: root/data/raw -> root/data/processed
        processed_dir = raw_path.parent.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        cache_filename = f"{raw_path.stem}_engineered.csv"
        cache_path = processed_dir / cache_filename
        
        # 1. CHECK CACHE
        if cache_path.exists():
            # Compare modification times
            raw_mtime = raw_path.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            
            # If Raw file is OLDER than Cache, the Cache is valid
            if raw_mtime < cache_mtime:
                logger.info(f"⚡ CACHE HIT: Loading pre-calculated features from {cache_path}")
                try:
                    df = pd.read_csv(cache_path)
                    # Populate feature columns
                    self._identify_feature_columns(df)
                    return df
                except Exception as e:
                    logger.warning(f"⚠️ Cache file corrupted ({e}). Recalculating...")

        # 2. CALCULATE (If no cache or stale)
        logger.info(f"📂 Loading Raw Data: {filepath}")
        df = pd.read_csv(filepath)
        
        # Standardize basic columns
        df.columns = df.columns.str.lower()
        
        logger.info("🚀 Running Batched Feature Engineering (This happens once)...")
        df = self.engine.generate_features(df)
        
        # Clean NaNs - SMART MODE
        initial_len = len(df)
        
        # Identify "Broken" Columns (Columns with > 20% NaNs)
        nan_counts = df.isna().sum()
        broken_cols = nan_counts[nan_counts > (0.2 * len(df))].index.tolist()
        
        if broken_cols:
            logger.warning(f"⚠️ Dropping {len(broken_cols)} broken columns: {broken_cols}")
            df.drop(columns=broken_cols, inplace=True)
            
        # Drop Rows for minor NaNs (warmup period)
        df.dropna(inplace=True)
        
        dropped = initial_len - len(df)
        logger.info(f"✅ Engineering Complete. Dropped {dropped} rows. Final: {len(df)}")
        
        # 3. SAVE CACHE
        logger.info(f"💾 Saving to cache: {cache_path}")
        df.to_csv(cache_path, index=False)
        
        self._identify_feature_columns(df)
        return df

    def _identify_feature_columns(self, df):
        """Helper to identify numeric feature columns"""
        exclude = ['time', 'date', 'timestamp', 'open', 'high', 'low', 'close', 
                   'tick_volume', 'volume', 'spread', 'real_volume', 'target', 'custom_target']
        
        self.feature_columns = [
            c for c in df.columns 
            if c not in exclude 
            and pd.api.types.is_numeric_dtype(df[c])
        ]