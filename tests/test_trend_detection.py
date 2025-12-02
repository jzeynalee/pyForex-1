# tests/test_trend_detection.py
"""
Unit tests for trend detection modules.
"""
import pytest
import numpy as np
import pandas as pd

from trend_detection.structural_analyzer import StructuralAnalyzer
from trend_detection.mtf_analyzer import MTFAnalyzer
from trend_detection.regime_classifier import RegimeClassifier
from trend_detection.trend_features import TrendFeatureBuilder
from trend_detection.fusion_trend_detector import FusionFXTrendDetector

@pytest.mark.unit
class TestStructuralAnalyzer:
    """Test suite for structural trend analyzer."""
    
    def test_init(self):
        """Test StructuralAnalyzer initialization."""
        analyzer = StructuralAnalyzer()
        assert analyzer.swing_detector is not None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        analyzer = StructuralAnalyzer(atr_multiplier=4.0, confirmation_candles=3)
        assert analyzer.swing_detector.atr_mult == 4.0
        assert analyzer.swing_detector.confirm_candles == 3
    
    def test_analyze_returns_dict(self, sample_ohlcv_data):
        """Test that analyze returns expected structure."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(sample_ohlcv_data)
        
        assert 'direction' in result
        assert 'score' in result
        assert 'type' in result
        assert 'swings_df' in result
    
    def test_analyze_direction_values(self, sample_ohlcv_data):
        """Test direction is valid value."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(sample_ohlcv_data)
        
        assert result['direction'] in [-1, 0, 1]
    
    def test_analyze_score_range(self, sample_ohlcv_data):
        """Test score is in valid range."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(sample_ohlcv_data)
        
        assert 0.0 <= result['score'] <= 1.0
    
    def test_analyze_bullish_trend(self, bullish_trend_data):
        """Test analysis of bullish trend data."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(bullish_trend_data)
        
        # Should detect bullish or neutral, not bearish
        assert result['direction'] >= 0 or result['type'] in ['bullish', 'mixed']
    
    def test_analyze_bearish_trend(self, bearish_trend_data):
        """Test analysis of bearish trend data."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(bearish_trend_data)
        
        # Should detect bearish or neutral, not bullish
        assert result['direction'] <= 0 or result['type'] in ['bearish', 'mixed']

    def test_analyze_insufficient_data(self):
        """Test behavior with too few candles."""
        analyzer = StructuralAnalyzer()
        small_df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='h'),  # Fixed 'H' -> 'h'
            'open': [1.1] * 5,
            'high': [1.11] * 5,
            'low': [1.09] * 5,
            'close': [1.1] * 5,
            'volume': [100] * 5,
        })
        
        # Should handle gracefully, not crash
        result = analyzer.analyze(small_df)
        assert 'direction' in result
        assert 'score' in result
    
    def test_analyze_with_nan_values(self, sample_ohlcv_data):
        """Test handling of NaN in data."""
        analyzer = StructuralAnalyzer()
        df = sample_ohlcv_data.copy()
        df.loc[10:15, 'close'] = np.nan
        
        # Code handles NaN gracefully - test that it returns valid structure
        result = analyzer.analyze(df)
        
        # Should still return expected structure
        assert 'direction' in result
        assert 'score' in result
        assert result['direction'] in [-1, 0, 1]
        # Score might be degraded due to NaN but should still be valid
        assert 0.0 <= result['score'] <= 1.0

@pytest.mark.unit
class TestMTFAnalyzer:
    """Test suite for multi-timeframe analyzer."""
    
    def test_init(self):
        """Test MTFAnalyzer initialization."""
        analyzer = MTFAnalyzer()
        assert analyzer.weights == {'H4': 0.4, 'H1': 0.4, 'M15': 0.2}
    
    def test_compute_tf_score(self, sample_ohlcv_data):
        """Test single timeframe score computation."""
        analyzer = MTFAnalyzer()
        result = analyzer.compute_tf_score(sample_ohlcv_data, structural_score=0.6)
        
        assert 'score' in result
        assert 'ema_slope' in result
        assert 'adx' in result
        assert 'price_position' in result
        
        assert 0.0 <= result['score'] <= 1.0
    
    def test_analyze_mtf(self, mtf_data):
        """Test multi-timeframe analysis."""
        analyzer = MTFAnalyzer()
        
        structural_scores = {'H4': 0.6, 'H1': 0.65, 'M15': 0.55}
        result = analyzer.analyze(mtf_data, structural_scores)
        
        assert 'mtf_score' in result
        assert 'individual_scores' in result
        
        assert 0.0 <= result['mtf_score'] <= 1.0
    
    def test_analyze_missing_timeframe(self, sample_ohlcv_data):
        """Test analysis with missing timeframe data."""
        analyzer = MTFAnalyzer()
        
        dfs_dict = {'H1': sample_ohlcv_data}  # Missing H4 and M15
        structural_scores = {'H1': 0.6}
        
        result = analyzer.analyze(dfs_dict, structural_scores)
        
        assert 'mtf_score' in result
        assert 'H1' in result['individual_scores']

@pytest.mark.unit
class TestRegimeClassifier:
    """Test suite for regime classifier."""
    
    def test_init(self):
        """Test RegimeClassifier initialization."""
        classifier = RegimeClassifier()
        assert classifier.indicators is not None
    
    def test_classify_regime_returns_dict(self, sample_ohlcv_data):
        """Test that classify_regime returns expected structure."""
        classifier = RegimeClassifier()
        result = classifier.classify_regime(sample_ohlcv_data)
        
        assert 'regime' in result
        assert 'adx' in result
        assert 'volatility' in result
        assert 'bb_width' in result
    
    def test_classify_regime_valid_regime(self, sample_ohlcv_data):
        """Test that regime is a valid value."""
        classifier = RegimeClassifier()
        result = classifier.classify_regime(sample_ohlcv_data)
        
        valid_regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL']
        assert result['regime'] in valid_regimes
    
    def test_apply_regime_filter_trending(self):
        """Test regime filter for trending market."""
        classifier = RegimeClassifier()
        
        regime_info = {'regime': 'TRENDING', 'adx': 30, 'volatility': 1.0, 'bb_width': 2.5}
        result = classifier.apply_regime_filter(regime_info, mtf_score=0.7)
        
        assert 'adjusted_score' in result
        assert 'adjustment_factor' in result
        assert result['adjusted_score'] == 0.7  # No adjustment for trending + high score
    
    def test_apply_regime_filter_ranging(self):
        """Test regime filter for ranging market."""
        classifier = RegimeClassifier()
        
        regime_info = {'regime': 'RANGING', 'adx': 15, 'volatility': 0.8, 'bb_width': 1.5}
        result = classifier.apply_regime_filter(regime_info, mtf_score=0.7)
        
        # Should heavily penalize in ranging
        assert result['adjusted_score'] < 0.7
        assert result['adjustment_factor'] < 1.0
    
    def test_apply_regime_filter_volatile(self):
        """Test regime filter for volatile market."""
        classifier = RegimeClassifier()
        
        regime_info = {'regime': 'VOLATILE', 'adx': 22, 'volatility': 1.5, 'bb_width': 3.0}
        result = classifier.apply_regime_filter(regime_info, mtf_score=0.5)
        
        # Should penalize weak signals in volatile regime
        assert result['adjusted_score'] < 0.5


@pytest.mark.unit
class TestTrendFeatureBuilder:
    """Test suite for trend feature builder."""
    
    def test_init(self):
        """Test TrendFeatureBuilder initialization."""
        builder = TrendFeatureBuilder()
        assert builder.indicators is not None
    
    def test_build_features(self, sample_ohlcv_data):
        """Test feature building."""
        builder = TrendFeatureBuilder()
        
        # Create mock intermediate results
        structural_result = {'direction': 1, 'score': 0.7, 'type': 'bullish'}
        mtf_result = {'mtf_score': 0.65, 'individual_scores': {'H1': 0.65, 'H4': 0.6}}
        regime_result = {'regime': 'TRENDING', 'adx': 28, 'volatility': 1.0, 'bb_width': 2.0}
        
        features = builder.build_features(
            sample_ohlcv_data,
            structural_result,
            mtf_result,
            regime_result
        )
        
        # Check expected features exist
        assert 'struct_score' in features
        assert 'mtf_score' in features
        assert 'regime' in features
        assert 'adx' in features
        assert 'plus_di' in features
        assert 'minus_di' in features
        assert 'price_above_ema20' in features
        assert 'ema_alignment' in features
        assert 'vol_compression' in features
        assert 'roc_5' in features
        assert 'roc_10' in features
    
    def test_feature_values_valid(self, sample_ohlcv_data):
        """Test that feature values are valid."""
        builder = TrendFeatureBuilder()
        
        structural_result = {'direction': 1, 'score': 0.7, 'type': 'bullish'}
        mtf_result = {'mtf_score': 0.65, 'individual_scores': {'H1': 0.65}}
        regime_result = {'regime': 'TRENDING', 'adx': 28, 'volatility': 1.0, 'bb_width': 2.0}
        
        features = builder.build_features(
            sample_ohlcv_data,
            structural_result,
            mtf_result,
            regime_result
        )
        
        # Binary features should be 0 or 1
        assert features['price_above_ema20'] in [0, 1]
        assert features['price_above_ema50'] in [0, 1]
        assert features['price_above_ema200'] in [0, 1]
        
        # EMA alignment should be -1, 0, or 1
        assert features['ema_alignment'] in [-1, 0, 1]


@pytest.mark.unit
class TestFusionFXTrendDetector:
    """Test suite for the main trend detector."""
    
    def test_init_without_ml(self):
        """Test initialization without ML model."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        assert detector.ml_model is None
        assert detector.structural_analyzer is not None
        assert detector.mtf_analyzer is not None
        assert detector.regime_classifier is not None
    
    def test_detect_trend_returns_dict(self, mtf_data):
        """Test that detect_trend returns expected structure."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        assert 'trend_class' in result
        assert 'trend_name' in result
        assert 'trend_strength' in result
        assert 'direction' in result
        assert 'confidence' in result
        assert 'details' in result
    
    def test_detect_trend_class_valid(self, mtf_data):
        """Test that trend class is valid."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        assert result['trend_class'] in [0, 1, 2, 3, 4]
    
    def test_detect_trend_direction_valid(self, mtf_data):
        """Test that direction is valid."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        assert result['direction'] in ['BULLISH', 'BEARISH', 'SIDEWAYS']
    
    def test_detect_trend_strength_range(self, mtf_data):
        """Test that trend strength is in valid range."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        assert 0 <= result['trend_strength'] <= 100
    
    def test_detect_trend_confidence_range(self, mtf_data):
        """Test that confidence is in valid range."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        assert 0 <= result['confidence'] <= 1
    
    def test_detect_trend_details_structure(self, mtf_data):
        """Test that details contains expected components."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        details = result['details']
        assert 'structural' in details
        assert 'mtf' in details
        assert 'regime' in details
        assert 'regime_filter' in details
        assert 'ml_direction' in details
        assert 'ml_confidence' in details
    
    def test_classify_trend_sideways(self):
        """Test trend classification for sideways."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=20, struct_direction=0, regime='RANGING'
        )
        
        assert trend_class == 0
        assert direction == 'SIDEWAYS'
    
    def test_classify_trend_early_bull(self):
        """Test trend classification for early bullish."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=45, struct_direction=1, regime='TRENDING'
        )
        
        assert trend_class == 1
        assert direction == 'BULLISH'
        assert 'Early' in trend_name
    
    def test_classify_trend_mature_bull(self):
        """Test trend classification for mature bullish."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=70, struct_direction=1, regime='TRENDING'
        )
        
        assert trend_class == 2
        assert direction == 'BULLISH'
        assert 'Mature' in trend_name
    
    def test_classify_trend_early_bear(self):
        """Test trend classification for early bearish."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=45, struct_direction=-1, regime='TRENDING'
        )
        
        assert trend_class == 3
        assert direction == 'BEARISH'
    
    def test_classify_trend_mature_bear(self):
        """Test trend classification for mature bearish."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=70, struct_direction=-1, regime='TRENDING'
        )
        
        assert trend_class == 4
        assert direction == 'BEARISH'
    
    def test_prepare_ml_input(self):
        """Test ML input preparation."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        features = {
            'struct_score': 0.7,
            'mtf_score': 0.65,
            'regime': 1,
            'adx': 28,
            'plus_di': 30,
            'minus_di': 15,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 1,
            'ema_alignment': 1,
            'vol_compression': 0.8,
            'roc_5': 1.5,
            'roc_10': 2.5,
        }
        
        ml_input = detector._prepare_ml_input(features)
        
        assert len(ml_input) == 13
        assert ml_input[0] == 0.7  # struct_score
        assert ml_input[1] == 0.65  # mtf_score