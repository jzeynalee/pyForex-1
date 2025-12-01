
# trend_detection/regime_classifier.py
"""
Step 3: Regime Classification & Filtering
Determines market regime and applies trend validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from utils.indicators_extended import TrendIndicators

class RegimeClassifier:
    """
    Market Regime Classifier
    Implements Step 3 of FTDM
    """
    
    def __init__(self):
        self.indicators = TrendIndicators()
    
    def classify_regime(self, df):
        """
        Classify market regime into:
          - TRENDING
          - RANGING
          - VOLATILE
          - TRANSITIONAL
        """
        # Calculate ADX for trend strength
        adx, plus_di, minus_di = self.indicators.calculate_adx(df)
        adx_current = adx.iloc[-1]
        
        # Calculate volatility compression
        vol_compression = self.indicators.calculate_volatility_compression(df)
        vol_current = vol_compression.iloc[-1]
        
        # Calculate Bollinger Band width
        bb_period = 20
        sma = df['close'].rolling(window=bb_period).mean()
        std = df['close'].rolling(window=bb_period).std()
        bb_width = (std.iloc[-1] / sma.iloc[-1]) * 100
        
        # Regime logic
        if adx_current > 25:
            regime = 'TRENDING'
        elif adx_current < 20 and bb_width < 2.0:
            regime = 'RANGING'
        elif vol_current > 1.3:  # High relative volatility
            regime = 'VOLATILE'
        else:
            regime = 'TRANSITIONAL'
        
        return {
            'regime': regime,
            'adx': adx_current,
            'volatility': vol_current,
            'bb_width': bb_width
        }
    
    def apply_regime_filter(self, regime_info, mtf_score):
        """
        Apply regime-based filtering to trend score
        Args:
            regime_info: dict from classify_regime()
            mtf_score: 0-1 trend score from MTF analysis
        Returns:
            adjusted_score: 0-1 filtered score
        """
        regime = regime_info['regime']
        
        # Regime adjustment rules from FTDM spec
        if regime in ['TRANSITIONAL', 'VOLATILE']:
            if mtf_score < 0.65:
                # Downweight weak trends in unstable regimes
                adjusted_score = mtf_score * 0.5
            else:
                adjusted_score = mtf_score
        elif regime == 'RANGING':
            # Strongly penalize trend signals in ranging markets
            adjusted_score = mtf_score * 0.3
        else:  # TRENDING
            # Allow trends with lower threshold
            if mtf_score > 0.55:
                adjusted_score = mtf_score
            else:
                adjusted_score = mtf_score * 0.7
        
        return {
            'adjusted_score': adjusted_score,
            'regime': regime,
            'adjustment_factor': adjusted_score / mtf_score if mtf_score > 0 else 1.0
        }