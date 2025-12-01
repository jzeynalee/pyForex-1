# trend_detection/mtf_analyzer.py

"""
Step 2: Multi-Timeframe Analysis
Combines H4, H1, M15 trend scores with weighted confluence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from utils.indicators_extended import TrendIndicators

class MTFAnalyzer:
    """
    Multi-Timeframe Trend Analyzer
    Implements Step 2 of FTDM
    """
    
    def __init__(self):
        self.indicators = TrendIndicators()
        self.weights = {'H4': 0.4, 'H1': 0.4, 'M15': 0.2}
    
    def compute_tf_score(self, df, structural_score):
        """
        Compute trend score for a single timeframe
        Components:
          - EMA slope (30%)
          - Structural score (40%)
          - ADX (20%)
          - Price location (10%)
        """
        # EMA slopes
        ema20, slope20 = self.indicators.calculate_ema_slope(df, period=20)
        ema50, slope50 = self.indicators.calculate_ema_slope(df, period=50)
        ema200, slope200 = self.indicators.calculate_ema_slope(df, period=200)
        
        # Normalize slopes to -1 to 1
        avg_slope = (slope20.iloc[-1] + slope50.iloc[-1] + slope200.iloc[-1]) / 3
        ema_component = np.tanh(avg_slope * 10)  # Squash to [-1, 1]
        
        # ADX
        adx, plus_di, minus_di = self.indicators.calculate_adx(df)
        adx_value = adx.iloc[-1]
        
        # ADX component: direction from DI, strength from ADX
        di_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1
        adx_normalized = min(adx_value / 50, 1.0)  # Normalize to 0-1
        adx_component = di_direction * adx_normalized
        
        # Price location (relative to EMAs)
        current_price = df['close'].iloc[-1]
        price_above_20 = 1 if current_price > ema20.iloc[-1] else -1
        price_above_50 = 1 if current_price > ema50.iloc[-1] else -1
        price_above_200 = 1 if current_price > ema200.iloc[-1] else -1
        price_component = (price_above_20 + price_above_50 + price_above_200) / 3
        
        # Weighted combination
        tf_score = (
            0.30 * ema_component +
            0.40 * (structural_score * 2 - 1) +  # Convert 0-1 to -1 to 1
            0.20 * adx_component +
            0.10 * price_component
        )
        
        # Normalize to 0-1
        tf_score_normalized = (tf_score + 1) / 2
        
        return {
            'score': tf_score_normalized,
            'ema_slope': avg_slope,
            'adx': adx_value,
            'price_position': price_component
        }
    
    def analyze(self, dfs_dict, structural_scores):
        """
        Main MTF analysis method
        Args:
            dfs_dict: {'H4': df_h4, 'H1': df_h1, 'M15': df_m15}
            structural_scores: {'H4': score, 'H1': score, 'M15': score}
        Returns:
            mtf_score: 0.0 to 1.0
        """
        tf_scores = {}
        
        for tf, df in dfs_dict.items():
            if tf not in structural_scores:
                continue
            
            result = self.compute_tf_score(df, structural_scores[tf])
            tf_scores[tf] = result['score']
        
        # Weighted average
        mtf_final_score = sum(
            tf_scores.get(tf, 0.5) * weight 
            for tf, weight in self.weights.items()
        )
        
        return {
            'mtf_score': mtf_final_score,
            'individual_scores': tf_scores
        }