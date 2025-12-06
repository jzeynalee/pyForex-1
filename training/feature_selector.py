# training/feature_selector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class DynamicFeatureSelector:
    def __init__(self, n_features=20, sample_size=50000):
        self.n_features = n_features
        self.sample_size = sample_size
        
    def select(self, df: pd.DataFrame, target_col: str, exclude_cols: list) -> list:
        """
        Selects top N features based on Random Forest importance.
        """
        logger.info(f"🔍 Running Dynamic Feature Selection (Target: Top {self.n_features})...")
        
        # 1. Prepare Candidates
        feature_candidates = [c for c in df.columns if c not in exclude_cols]
        
        # 2. Sample Data (Use recent data for relevance)
        if len(df) > self.sample_size:
            sample_df = df.iloc[-self.sample_size:]
        else:
            sample_df = df
            
        X = sample_df[feature_candidates].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = sample_df[target_col]
        
        # 3. Train fast RF
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # 4. Rank
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        selected = []
        logger.info("🏆 TOP SELECTED FEATURES:")
        for i in range(self.n_features):
            idx = indices[i]
            feat = feature_candidates[idx]
            selected.append(feat)
            logger.info(f"   {i+1:2d}. {feat:<25} ({importances[idx]:.4f})")
            
        return selected