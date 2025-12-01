# trend_detection/structural_analyzer.py

"""
Step 1: Structural Trend Detection
Deterministic, swing-based trend classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.swing_detector import SwingDetector

class StructuralAnalyzer:
    """
    Implements Step 1 of FTDM: Structural Trend Detection
    Non-ML, swing-point based trend identification
    """
    
    def __init__(self, atr_multiplier=3.5, confirmation_candles=2):
        self.swing_detector = SwingDetector(
            atr_multiplier=atr_multiplier,
            confirmation_candles=confirmation_candles
        )
    
    def analyze(self, df):
        """
        Main analysis method
        Returns: (direction, structure_score)
          direction: -1 (bearish), 0 (sideways), 1 (bullish)
          structure_score: 0.0 to 1.0 (confidence)
        """
        # Detect swing points
        df_swings = self.swing_detector.detect_swings(df)
        
        # Classify structure
        structure_type, structure_score = self.swing_detector.classify_structure(df_swings)
        
        # Map to direction
        if structure_type == 'bullish' and structure_score > 0.6:
            direction = 1
        elif structure_type == 'bearish' and structure_score > 0.6:
            direction = -1
        else:
            direction = 0
            structure_score = max(0.0, structure_score - 0.3)  # Penalize mixed
        
        return {
            'direction': direction,
            'score': structure_score,
            'type': structure_type,
            'swings_df': df_swings
        }