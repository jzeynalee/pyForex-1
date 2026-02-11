# trend_detection/trend_features.py
"""
Feature engineering for ML-based trend confirmation (Step 4)
Prepares features for your existing Fusion model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from utils.indicators_extended import TrendIndicators

class TrendFeatureBuilder:
    """
    Builds comprehensive feature set for ML trend confirmation
    These features augment your existing TCN/Price Action features
    """
    
    def __init__(self):
        self.indicators = TrendIndicators()
    
    def build_features(self, df, structural_result, mtf_result, regime_result):
        """
        Build feature vector for ML model
        Returns: dict of features suitable for your Fusion model
        """
        features = {}
        
        # Structural features
        features['struct_direction'] = structural_result['direction']
        features['struct_score'] = structural_result['score']
        
        # MTF features
        features['mtf_score'] = mtf_result['mtf_score']
        for tf, score in mtf_result['individual_scores'].items():
            features[f'mtf_{tf.lower()}_score'] = score
        
        # Regime features
        regime_map = {'TRENDING': 1, 'RANGING': 0, 'VOLATILE': 2, 'TRANSITIONAL': 3}
        features['regime'] = regime_map.get(regime_result['regime'], 0)
        features['regime_adx'] = regime_result['adx']
        features['regime_volatility'] = regime_result['volatility']
        
        # Additional context features
        adx, plus_di, minus_di = self.indicators.calculate_adx(df)
        features['adx'] = adx.iloc[-1]
        features['plus_di'] = plus_di.iloc[-1]
        features['minus_di'] = minus_di.iloc[-1]
        
        # EMA alignment
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['close'].ewm(span=200).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        features['price_above_ema20'] = 1 if current_price > ema20 else 0
        features['price_above_ema50'] = 1 if current_price > ema50 else 0
        features['price_above_ema200'] = 1 if current_price > ema200 else 0
        features['ema_alignment'] = 1 if ema20 > ema50 > ema200 else (-1 if ema20 < ema50 < ema200 else 0)
        
        # Volatility features
        vol_compression = self.indicators.calculate_volatility_compression(df)
        features['vol_compression'] = vol_compression.iloc[-1]
        
        # Momentum features
        roc_5 = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100
        roc_10 = ((df['close'].iloc[-1] / df['close'].iloc[-11]) - 1) * 100
        features['roc_5'] = roc_5
        features['roc_10'] = roc_10
        
        return features