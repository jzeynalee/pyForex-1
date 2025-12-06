# utils/feature_adapter.py
import pandas as pd
import numpy as np
import logging

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
        logger.info(f"📂 Loading {filepath}...")
        df = pd.read_csv(filepath)
        
        # Standardize basic columns
        df.columns = df.columns.str.lower()
        
        # USE THE BATCHED ENGINE
        logger.info("🚀 Running Batched Feature Engineering...")
        # This will now use _process_in_batches automatically for 500k rows
        df = self.engine.generate_features(df)
        
        # Clean NaNs
        initial = len(df)
        df.dropna(inplace=True)
        dropped = initial - len(df)
        logger.info(f"✅ Engineering Complete. Dropped {dropped} rows. Final: {len(df)}")
        
        # Store feature names
        exclude = ['time', 'date', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 'spread', 'real_volume', 'target', 'custom_target']
        self.feature_columns = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32]]
        
        return df