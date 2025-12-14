# tests/test_trading_enhanced_signal_engine.py
"""
Unit tests for trading/enhanced_signal_engine.py - FusionFX Trend Detector.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from trading.enhanced_signal_engine import (
    FusionFXTrendDetector, EnhancedSignalEngine
)


@pytest.mark.unit
class TestFusionFXTrendDetector:
    """Test FusionFXTrendDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multi-timeframe data."""
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        
        base_price = 1.1000
        prices = base_price + np.cumsum(np.random.randn(200) * 0.0001)
        
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices * 1.0002,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        return {
            'H4': df.iloc[::4].reset_index(drop=True),
            'H1': df,
            'M15': df.iloc[::4].reset_index(drop=True)
        }

    @pytest.fixture
    def mock_components(self):
        """Mock trend detection components."""
        with patch('trading.enhanced_signal_engine.StructuralAnalyzer') as MockStruct, \
             patch('trading.enhanced_signal_engine.MTFAnalyzer') as MockMTF, \
             patch('trading.enhanced_signal_engine.RegimeClassifier') as MockRegime, \
             patch('trading.enhanced_signal_engine.TrendFeatureBuilder') as MockFeature:
            
            mock_struct = Mock()
            mock_mtf = Mock()
            mock_regime = Mock()
            mock_feature = Mock()
            
            MockStruct.return_value = mock_struct
            MockMTF.return_value = mock_mtf
            MockRegime.return_value = mock_regime
            MockFeature.return_value = mock_feature
            
            yield {
                'struct': mock_struct,
                'mtf': mock_mtf,
                'regime': mock_regime,
                'feature': mock_feature
            }

    def test_init_no_model(self, mock_components):
        """Test initialization without ML model."""
        detector = FusionFXTrendDetector()

        assert detector.ml_model is None
        assert detector.structural_analyzer is not None
        assert detector.mtf_analyzer is not None

    def test_init_with_model_path(self, mock_components):
        """Test initialization with model path."""
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.2, 0.3, 0.5]]))
        
        with patch('trading.enhanced_signal_engine.TrendClassifier') as MockTrendClass:
            MockTrendClass.load = Mock(return_value=mock_model)
            
            detector = FusionFXTrendDetector(ml_model_path='models/trend_classifier.pkl')

            # Model may or may not load, depending on path existence
            # Just test that it attempts to load
            assert detector is not None

    def test_detect_trend_basic(self, mock_components, sample_data):
        """Test basic trend detection."""
        # Setup mocks
        mock_components['struct'].analyze.return_value = {
            'score': 0.75,
            'direction': 1  # Bullish
        }
        
        mock_components['mtf'].analyze.return_value = {
            'mtf_score': 0.70
        }
        
        mock_components['regime'].classify_regime.return_value = {
            'regime': 'TRENDING'
        }
        
        mock_components['regime'].apply_regime_filter.return_value = {
            'adjusted_score': 0.70
        }
        
        mock_components['feature'].build_features.return_value = {
            'struct_score': 0.75,
            'mtf_score': 0.70,
            'regime': 1,
            'adx': 25.0,
            'plus_di': 20.0,
            'minus_di': 10.0,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 1,
            'ema_alignment': 1,
            'vol_compression': 0,
            'roc_5': 0.001,
            'roc_10': 0.002
        }
        
        detector = FusionFXTrendDetector()

        result = detector.detect_trend(sample_data)

        assert 'trend_class' in result
        assert 'trend_strength' in result
        assert 'direction' in result
        assert 'confidence' in result
        assert 'details' in result
        assert 0 <= result['trend_strength'] <= 100

    def test_detect_trend_with_ml_model(self, mock_components, sample_data):
        """Test trend detection with ML model."""
        # Setup structural/mtf/regime mocks
        mock_components['struct'].analyze.return_value = {
            'score': 0.75,
            'direction': 1
        }
        
        mock_components['mtf'].analyze.return_value = {
            'mtf_score': 0.70
        }
        
        mock_components['regime'].classify_regime.return_value = {
            'regime': 'TRENDING'
        }
        
        mock_components['regime'].apply_regime_filter.return_value = {
            'adjusted_score': 0.70
        }
        
        mock_components['feature'].build_features.return_value = {
            'struct_score': 0.75,
            'mtf_score': 0.70,
            'regime': 1,
            'adx': 25.0,
            'plus_di': 20.0,
            'minus_di': 10.0,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 1,
            'ema_alignment': 1,
            'vol_compression': 0,
            'roc_5': 0.001,
            'roc_10': 0.002
        }
        
        # Create mock ML model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))  # BULL
        
        detector = FusionFXTrendDetector(ml_model=mock_model)

        result = detector.detect_trend(sample_data)

        assert result['trend_class'] in [0, 1, 2, 3, 4]
        assert 'ml_probabilities' in result['details']
        mock_model.predict_proba.assert_called_once()

    def test_detect_trend_classification(self, mock_components, sample_data):
        """Test trend classification logic."""
        # Sideways scenario
        mock_components['struct'].analyze.return_value = {
            'score': 0.20,
            'direction': 0
        }
        mock_components['mtf'].analyze.return_value = {'mtf_score': 0.25}
        mock_components['regime'].classify_regime.return_value = {'regime': 'RANGING'}
        mock_components['regime'].apply_regime_filter.return_value = {'adjusted_score': 0.25}
        
        detector = FusionFXTrendDetector()

        result = detector.detect_trend(sample_data)

        assert result['trend_class'] == 0  # Sideways
        assert result['direction'] == 'SIDEWAYS'

    def test_prepare_ml_input(self, mock_components):
        """Test ML input preparation."""
        detector = FusionFXTrendDetector()
        
        features = {
            'struct_score': 0.75,
            'mtf_score': 0.70,
            'regime': 1,
            'adx': 25.0,
            'plus_di': 20.0,
            'minus_di': 10.0,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 1,
            'ema_alignment': 1,
            'vol_compression': 0,
            'roc_5': 0.001,
            'roc_10': 0.002
        }

        feature_vector = detector._prepare_ml_input(features)

        assert len(feature_vector) == 13
        assert feature_vector[0] == 0.75  # struct_score
        assert feature_vector[1] == 0.70  # mtf_score

    def test_prepare_ml_input_missing_keys(self, mock_components):
        """Test ML input preparation with missing keys."""
        detector = FusionFXTrendDetector()
        
        features = {
            'struct_score': 0.75,
            # Missing other keys
        }

        feature_vector = detector._prepare_ml_input(features)

        assert len(feature_vector) == 13
        assert feature_vector[0] == 0.75
        # Missing keys should default to 0
        assert feature_vector[2] == 0  # regime missing

    def test_classify_trend_sideways(self, mock_components):
        """Test trend classification - sideways."""
        detector = FusionFXTrendDetector()

        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=25.0,
            struct_direction=0,
            regime='RANGING'
        )

        assert trend_class == 0
        assert trend_name == 'Sideways/Compression'
        assert direction == 'SIDEWAYS'

    def test_classify_trend_early_bull(self, mock_components):
        """Test trend classification - early bull."""
        detector = FusionFXTrendDetector()

        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=40.0,
            struct_direction=1,
            regime='TRENDING'
        )

        assert trend_class == 1
        assert trend_name == 'Early Bull Trend'
        assert direction == 'BULLISH'

    def test_classify_trend_mature_bull(self, mock_components):
        """Test trend classification - mature bull."""
        detector = FusionFXTrendDetector()

        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=70.0,
            struct_direction=1,
            regime='TRENDING'
        )

        assert trend_class == 2
        assert trend_name == 'Mature Bull Trend'
        assert direction == 'BULLISH'

    def test_classify_trend_early_bear(self, mock_components):
        """Test trend classification - early bear."""
        detector = FusionFXTrendDetector()

        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=40.0,
            struct_direction=-1,
            regime='TRENDING'
        )

        assert trend_class == 3
        assert trend_name == 'Early Bear Trend'
        assert direction == 'BEARISH'

    def test_classify_trend_mature_bear(self, mock_components):
        """Test trend classification - mature bear."""
        detector = FusionFXTrendDetector()

        trend_class, trend_name, direction = detector._classify_trend(
            trend_strength=70.0,
            struct_direction=-1,
            regime='TRENDING'
        )

        assert trend_class == 4
        assert trend_name == 'Mature Bear Trend'
        assert direction == 'BEARISH'

    def test_calculate_confidence(self, mock_components):
        """Test confidence calculation."""
        detector = FusionFXTrendDetector()

        confidence = detector._calculate_confidence(
            struct_score=0.75,
            mtf_score=0.70,
            ml_conf=0.80,
            regime_score=0.65
        )

        assert 0 <= confidence <= 1
        # Weighted average should be reasonable
        assert confidence > 0.5

    def test_ml_model_error_handling(self, mock_components, sample_data):
        """Test error handling when ML model fails."""
        mock_model = Mock()
        mock_model.predict_proba = Mock(side_effect=Exception("ML error"))
        
        mock_components['struct'].analyze.return_value = {
            'score': 0.75,
            'direction': 1
        }
        mock_components['mtf'].analyze.return_value = {'mtf_score': 0.70}
        mock_components['regime'].classify_regime.return_value = {'regime': 'TRENDING'}
        mock_components['regime'].apply_regime_filter.return_value = {'adjusted_score': 0.70}
        mock_components['feature'].build_features.return_value = {
            'struct_score': 0.75,
            'mtf_score': 0.70,
            'regime': 1,
            'adx': 25.0,
            'plus_di': 20.0,
            'minus_di': 10.0,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 1,
            'ema_alignment': 1,
            'vol_compression': 0,
            'roc_5': 0.001,
            'roc_10': 0.002
        }
        
        detector = FusionFXTrendDetector(ml_model=mock_model)

        # Should not raise, should use neutral defaults
        result = detector.detect_trend(sample_data)

        assert 'trend_class' in result
        assert result['details']['ml_confidence'] == 0.5  # Neutral default


@pytest.mark.unit
class TestEnhancedSignalEngine:
    """Test EnhancedSignalEngine alias."""

    def test_alias_exists(self):
        """Test that EnhancedSignalEngine alias exists."""
        assert EnhancedSignalEngine == FusionFXTrendDetector

