import numpy as np
import pytest
from types import SimpleNamespace

from trend_detection.fusion_trend_detector import FusionFXTrendDetector, TrendFeatureBuilder


class DummyAnalyzer:
    def __init__(self, score=0.5, direction=0):
        self._score = score
        self._direction = direction

    def analyze(self, df):
        return {'score': self._score, 'direction': self._direction}


class DummyMTF:
    def __init__(self, mtf_score=0.5):
        self._mtf_score = mtf_score

    def analyze(self, dfs_dict, structural_scores):
        return {'mtf_score': self._mtf_score}


class DummyRegime:
    def __init__(self, regime='TRENDING', adjusted=0.5):
        self._regime = regime
        self._adjusted = adjusted

    def classify_regime(self, df):
        return {'regime': self._regime}

    def apply_regime_filter(self, regime_info, mtf_score):
        return {'adjusted_score': self._adjusted}


class DummyFeatureBuilder:
    def build_features(self, df, primary_structure, mtf_result, regime_info):
        return {
            'struct_score': primary_structure['score'],
            'mtf_score': mtf_result['mtf_score'],
            'regime': 1,
            'adx': 10,
            'plus_di': 12,
            'minus_di': 8,
            'price_above_ema20': 1,
            'price_above_ema50': 1,
            'price_above_ema200': 0,
            'ema_alignment': 1,
            'vol_compression': 0.1,
            'roc_5': 0.02,
            'roc_10': 0.05,
        }


class DummyML:
    def __init__(self, probs=None, raise_exc=False):
        self.probs = probs if probs is not None else [0.1, 0.2, 0.7]
        self.raise_exc = raise_exc

    def predict_proba(self, X):
        if self.raise_exc:
            raise RuntimeError("model error")
        return [self.probs]


def test_classify_trend_sideways_and_ranging():
    det = FusionFXTrendDetector()

    # Ranging overrides strength
    cls, name, direction = det._classify_trend(80, 1, 'RANGING')
    assert cls == 0
    assert 'Sideways' in name
    assert direction == 'SIDEWAYS'

    # Low strength -> sideways
    cls, name, direction = det._classify_trend(10, 1, 'TRENDING')
    assert cls == 0


def test_classify_trend_bull_early_and_mature():
    det = FusionFXTrendDetector()

    # Early bull
    cls, name, direction = det._classify_trend(35, 1, 'TRENDING')
    assert cls == 1
    assert direction == 'BULLISH'

    # Mature bull
    cls, name, direction = det._classify_trend(80, 1, 'TRENDING')
    assert cls == 2


def test_classify_trend_bear_early_and_mature():
    det = FusionFXTrendDetector()

    # Early bear (struct_direction <= 0)
    cls, name, direction = det._classify_trend(35, -1, 'TRENDING')
    assert cls == 3
    assert direction == 'BEARISH'

    # Mature bear
    cls, name, direction = det._classify_trend(80, -1, 'TRENDING')
    assert cls == 4


def test_calculate_confidence_weights():
    det = FusionFXTrendDetector()
    conf = det._calculate_confidence(0.4, 0.6, 0.8, 0.2)
    expected = 0.35*0.4 + 0.35*0.6 + 0.20*0.8 + 0.10*0.2
    assert pytest.approx(conf, rel=1e-6) == expected


def test_prepare_ml_input_order_and_defaults():
    det = FusionFXTrendDetector()
    features = {'struct_score': 0.5, 'mtf_score': 0.3, 'regime': 1, 'adx': 10}

    vec = det._prepare_ml_input(features)
    # Ensure length equals expected feature_order
    assert isinstance(vec, list)
    assert len(vec) == 13
    # Check known positions
    assert vec[0] == 0.5
    assert vec[1] == 0.3
    assert vec[2] == 1
    # Missing keys become 0
    assert vec[3] == 10


def test_detect_trend_without_ml_model(monkeypatch):
    det = FusionFXTrendDetector(ml_model=None)

    # Patch analyzers with deterministic output
    det.structural_analyzer = DummyAnalyzer(score=0.6, direction=1)
    det.mtf_analyzer = DummyMTF(mtf_score=0.7)
    det.regime_classifier = DummyRegime(regime='TRENDING', adjusted=0.5)

    # Prepare fake dfs_dict (contents not used by dummies)
    dfs = {'H4': None, 'H1': None, 'M15': None}

    out = det.detect_trend(dfs)

    # Validate keys
    assert 'trend_class' in out
    assert 'trend_strength' in out
    assert 'confidence' in out
    assert 0 <= out['confidence'] <= 1
    # Without ML, ml_confidence default 0.5
    assert out['details']['ml_confidence'] == 0.5


def test_detect_trend_with_ml_model_success(monkeypatch):
    # ML returns strong bull probability
    ml = DummyML(probs=[0.05, 0.1, 0.85])
    det = FusionFXTrendDetector(ml_model=ml)

    det.structural_analyzer = DummyAnalyzer(score=0.7, direction=1)
    det.mtf_analyzer = DummyMTF(mtf_score=0.6)
    det.regime_classifier = DummyRegime(regime='TRENDING', adjusted=0.4)
    det.feature_builder = DummyFeatureBuilder()

    dfs = {'H4': None, 'H1': None, 'M15': None}

    out = det.detect_trend(dfs)

    # ML confidence should be the max prob
    assert pytest.approx(out['details']['ml_confidence'], rel=1e-6) == 0.85
    # ml_direction normalized used in final score -> check it's non-negative
    assert out['trend_strength'] >= 0
    assert out['direction'] in ('BULLISH', 'BEARISH', 'SIDEWAYS')


def test_detect_trend_with_ml_model_failure(monkeypatch):
    # ML predict raises exception
    ml = DummyML(raise_exc=True)
    det = FusionFXTrendDetector(ml_model=ml)

    det.structural_analyzer = DummyAnalyzer(score=0.5, direction=0)
    det.mtf_analyzer = DummyMTF(mtf_score=0.5)
    det.regime_classifier = DummyRegime(regime='TRENDING', adjusted=0.5)
    det.feature_builder = DummyFeatureBuilder()

    dfs = {'H4': None, 'H1': None, 'M15': None}

    out = det.detect_trend(dfs)

    # ML failure should fallback to defaults
    assert out['details']['ml_confidence'] == 0.5
    assert out['details']['ml_direction'] == 0
    assert 0 <= out['confidence'] <= 1
