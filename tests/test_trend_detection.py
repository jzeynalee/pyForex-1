# tests/test_trend_detection.py
"""
Test script for trend detection system
Run this before live trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from trend_detection.fusion_trend_detector import FusionFXTrendDetector
from trading.mt5_connector import get_candles

def test_trend_detection():
    """Test trend detection on historical data"""
    
    print("🧪 Testing FusionFX Trend Detection...")
    
    # Get test data
    h4_candles = get_candles("EURUSD", "H4", n=200)
    h1_candles = get_candles("EURUSD", "H1", n=200)
    m15_candles = get_candles("EURUSD", "M15", n=200)
    
    dfs = {
        'H4': pd.DataFrame(h4_candles),
        'H1': pd.DataFrame(h1_candles),
        'M15': pd.DataFrame(m15_candles)
    }
    
    # Initialize detector
    detector = FusionFXTrendDetector(ml_model=None)
    
    # Run detection
    result = detector.detect_trend(dfs)
    
    # Display results
    print("\n" + "="*60)
    print("TREND DETECTION RESULTS")
    print("="*60)
    print(f"Trend Class: {result['trend_class']} - {result['trend_name']}")
    print(f"Direction: {result['direction']}")
    print(f"Trend Strength: {result['trend_strength']:.1f}/100")
    print(f"Overall Confidence: {result['confidence']:.2f}")
    print("\nDetails:")
    print(f"  Structural Score: {result['details']['structural']['H1']['score']:.2f}")
    print(f"  MTF Score: {result['details']['mtf']['mtf_score']:.2f}")
    print(f"  Regime: {result['details']['regime']['regime']}")
    print(f"  Regime ADX: {result['details']['regime']['adx']:.1f}")
    print("="*60)
    
    return result

if __name__ == "__main__":
    test_trend_detection()