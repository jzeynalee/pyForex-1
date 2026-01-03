# test_price_action_replacement.py
"""
Test script to verify Price Action Pattern Extractor replacement for YOLO.

This script tests:
1. PriceActionPatternExtractor initialization and pattern detection
2. Integration with the predictor pipeline
3. Fusion network compatibility
4. Strategy configuration updates
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_price_action_extractor():
    """Test the PriceActionPatternExtractor class."""
    print("\n=== Testing PriceActionPatternExtractor ===")
    
    try:
        from models.price_action_pattern import PriceActionPatternExtractor
        
        # Initialize extractor
        extractor = PriceActionPatternExtractor(
            num_classes=25,
            include_extended_patterns=True,
            include_confidence=False
        )
        
        print(f"✅ Extractor initialized successfully")
        print(f"   Feature dimension: {extractor.get_feature_dim()}")
        
        # Create test OHLCV data
        np.random.seed(42)
        n_bars = 100
        
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')
        base_price = 1.1000
        
        # Generate realistic price data with some patterns
        price_changes = np.random.normal(0, 0.0001, n_bars)
        prices = base_price + np.cumsum(price_changes)
        
        test_df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 0.0002, n_bars)),
            'low': prices - np.abs(np.random.normal(0, 0.0002, n_bars)),
            'close': prices + np.random.normal(0, 0.0001, n_bars),
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure OHLC consistency
        test_df['high'] = np.maximum(test_df['high'], test_df[['open', 'close']].max(axis=1))
        test_df['low'] = np.minimum(test_df['low'], test_df[['open', 'close']].min(axis=1))
        
        # Extract patterns
        patterns = extractor.extract(test_df)
        
        print(f"✅ Pattern extraction successful")
        print(f"   Pattern vector shape: {patterns.shape}")
        print(f"   Non-zero patterns: {np.count_nonzero(patterns)}")
        print(f"   Pattern vector sample: {patterns[:10]}")
        
        # Test with details
        patterns_with_details, detections = extractor.extract_with_details(test_df)
        
        print(f"✅ Detailed extraction successful")
        print(f"   Detections found: {len(detections)}")
        for detection in detections[:3]:  # Show first 3 detections
            print(f"   - {detection['class_name']}: {detection['confidence']:.3f}")
        
        # Test pattern summary
        summary = extractor.get_pattern_summary(test_df)
        if 'error' not in summary:
            print(f"✅ Pattern summary generated")
            print(f"   Primary patterns: {len(summary.get('primary_patterns', {}))}")
            print(f"   Extended patterns: {len(summary.get('extended_patterns', {}))}")
            print(f"   Market context: RSI={summary.get('market_context', {}).get('rsi', 0):.2f}")
        else:
            print(f"⚠️ Pattern summary error: {summary['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ PriceActionPatternExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_integration():
    """Test predictor integration with price action."""
    print("\n=== Testing Predictor Integration ===")
    
    try:
        from inference.predictor import create_predictor, PredictorConfig
        
        # Test with price action enabled
        config = PredictorConfig(
            profile='SCALP',
            use_price_action=True,
            use_vision=False  # Disable vision for this test
        )
        
        predictor = create_predictor(
            profile='SCALP',
            use_price_action=True,
            use_vision=False
        )
        
        print(f"✅ Predictor created with price action")
        print(f"   Predictor type: {type(predictor).__name__}")
        
        # Create test features
        np.random.seed(42)
        features = np.random.randn(60, 20).astype(np.float32)
        
        # Create test OHLCV data for price action
        n_bars = 60
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, n_bars)
        prices = base_price + np.cumsum(price_changes)
        
        market_data = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 0.0002, n_bars)),
            'low': prices - np.abs(np.random.normal(0, 0.0002, n_bars)),
            'close': prices + np.random.normal(0, 0.0001, n_bars),
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure OHLC consistency
        market_data['high'] = np.maximum(market_data['high'], market_data[['open', 'close']].max(axis=1))
        market_data['low'] = np.minimum(market_data['low'], market_data[['open', 'close']].min(axis=1))
        
        # Test prediction
        prediction = predictor.predict(features, market_data)
        
        print(f"✅ Prediction successful")
        print(f"   Signal: {prediction.signal_name}")
        print(f"   Confidence: {prediction.confidence:.3f}")
        print(f"   Probabilities: {prediction.probabilities}")
        
        if hasattr(prediction, 'gate_weights'):
            print(f"   Gate weights: {prediction.gate_weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictor integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_compatibility():
    """Test fusion network compatibility."""
    print("\n=== Testing Fusion Network Compatibility ===")
    
    try:
        from models.fusion import FusionNet
        import torch
        
        # Test with price action dimension
        fusion_net = FusionNet(
            seq_dim=64,
            vit_dim=768,
            yolo_dim=50,  # Price action patterns dimension
            hidden_dim=256,
            num_classes=3
        )
        
        print(f"✅ Fusion network created with price action dimension")
        
        # Create test tensors
        batch_size = 2
        seq_feat = torch.randn(batch_size, 64)
        vit_feat = torch.randn(batch_size, 768)
        price_action_feat = torch.randn(batch_size, 50)
        
        # Test forward pass
        with torch.no_grad():
            logits = fusion_net(seq_feat, vit_feat, price_action_feat)
        
        print(f"✅ Fusion forward pass successful")
        print(f"   Output shape: {logits.shape}")
        
        # Test with gates
        logits, gates = fusion_net.forward_with_gates(seq_feat, vit_feat, price_action_feat)
        
        print(f"✅ Fusion with gates successful")
        print(f"   Gate weights: {gates[0].tolist()}")
        
        # Test modality importance
        importance = fusion_net.get_modality_importance(seq_feat, vit_feat, price_action_feat)
        
        print(f"✅ Modality importance calculated")
        print(f"   Importance: {importance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fusion compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_configuration():
    """Test strategy configuration updates."""
    print("\n=== Testing Strategy Configuration ===")
    
    try:
        from strategies.neural_hybrid import StrategyConfig
        
        # Test default configuration
        config = StrategyConfig()
        
        print(f"✅ Strategy config created")
        print(f"   Use price action: {config.use_price_action}")
        print(f"   Use vision: {config.use_vision}")
        print(f"   Profile: {config.profile}")
        
        # Test SCALP profile
        scalp_config = StrategyConfig(profile='SCALP')
        
        print(f"✅ SCALP profile config created")
        print(f"   Max open trades: {scalp_config.max_open_trades}")
        print(f"   Max daily trades: {scalp_config.max_daily_trades}")
        print(f"   Use price action: {scalp_config.use_price_action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Price Action Replacement for YOLO")
    print("=" * 60)
    
    tests = [
        ("Price Action Extractor", test_price_action_extractor),
        ("Predictor Integration", test_predictor_integration),
        ("Fusion Compatibility", test_fusion_compatibility),
        ("Strategy Configuration", test_strategy_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Price Action replacement is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
