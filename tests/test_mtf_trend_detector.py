import pytest
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

from trend_detection.mtf_trend_detector import MTFTrendDetector


class DummyProfile:
    def __init__(self):
        self.primary_tf = SimpleNamespace(value='H1')
        self.timeframe_strings = ['H1', 'M15']
        self.weights = {'H1': 0.6, 'M15': 0.4}
        self.name = 'TEST_PROFILE'
        self.min_confluence_score = 0.5
        self.require_higher_tf_alignment = True
        self.min_bars = 10
        self.adjust_risk_by_confluence = False


class DummyStructural:
    def analyze(self, df):
        return {'direction': 1, 'score': 0.7, 'type': 'swing'}


class DummyRegime:
    def classify_regime(self, df):
        return {'regime': 'TRENDING', 'confidence': 0.8}


class DummyFeatureSet:
    def __init__(self, features):
        self.features = features


class DummyFeatureBuilder:
    def build_features(self, dfs_dict, primary_tf):
        # return a wrapper with .features
        f = {
            'H1_range_position': 0.6,
            'mtf_weighted_direction': 0.2,
            'H1_adx': 30,
            'H1_plus_di': 20,
            'H1_minus_di': 10,
            'H1_price_vs_ema20': 1,
            'H1_price_vs_ema50': 1,
            'H1_price_vs_ema200': 0,
            'H1_ema_alignment': 1,
            'H1_vol_compression': 0.9,
            'H1_roc_5': 0.01,
            'H1_roc_10': 0.02,
            'mtf_weighted_direction': 0.2,
        }
        return DummyFeatureSet(f)


class DummyML:
    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, X):
        return [self._probs]


def make_df_with_time(n=50):
    df = pd.DataFrame({'open': [1.1 + i*0.0001 for i in range(n)],
                       'high': [1.1 + i*0.0001 + 0.0002 for i in range(n)],
                       'low': [1.1 + i*0.0001 - 0.0002 for i in range(n)],
                       'close': [1.1 + i*0.0001 for i in range(n)],
                       'time': pd.date_range('2025-01-01', periods=n)})
    return df


def test_prepare_ml_input_mapping():
    profile = DummyProfile()
    det = MTFTrendDetector(profile=profile)

    features = {
        'H1_range_position': 0.8,
        'mtf_weighted_direction': 0.1,
        'H1_adx': 40,
        'H1_plus_di': 25,
        'H1_minus_di': 5,
        'H1_price_vs_ema20': 1,
        'H1_price_vs_ema50': 0,
        'H1_price_vs_ema200': 1,
        'H1_ema_alignment': 1,
        'H1_vol_compression': 0.7,
        'H1_roc_5': 0.02,
        'H1_roc_10': 0.03,
    }

    vec = det._prepare_ml_input(features)
    # Expected order: struct_score, mtf_score, regime, adx, plus_di, minus_di, price_above_ema20, price_above_ema50, price_above_ema200, ema_alignment, vol_compression, roc_5, roc_10
    assert isinstance(vec, list)
    assert len(vec) == 13
    assert vec[0] == pytest.approx(0.8)
    # mtf_weighted_direction of 0.1 -> mtf_score = 0.1*0.5 + 0.5 = 0.55
    assert vec[1] == pytest.approx(0.55)
    # regime 1 because adx 40 > 25
    assert vec[2] == 1
    assert vec[3] == 40
    assert vec[4] == 25
    assert vec[5] == 5


def test_classify_trend_sideways_and_with_ml_boost():
    profile = DummyProfile()
    det = MTFTrendDetector(profile=profile)

    # create mtf_result simple namespace
    mtf_result = SimpleNamespace(mtf_score=0.2, direction=0)
    struct = {'direction': 0}
    regime = {'regime': 'RANGING'}

    cls, name, direction = det._classify_trend(mtf_result, struct, regime, 0, 0.5)
    assert cls == 0
    assert direction == 'SIDEWAYS'

    # ML boost when ML agrees
    mtf_result = SimpleNamespace(mtf_score=0.6, direction=1)
    struct = {'direction': 1}
    # ml_direction 1, ml_confidence >0.7 boosts strength by 1.1
    cls2, name2, direction2 = det._classify_trend(mtf_result, struct, {'regime':'TRENDING'}, 1, 0.8)
    assert direction2 == 'BULLISH'
    assert cls2 in (1,2)

    # ML disagreement reduces strength
    cls3, name3, direction3 = det._classify_trend(mtf_result, struct, {'regime':'TRENDING'}, -1, 0.9)
    assert direction3 == 'BULLISH' or direction3 == 'BEARISH'  # classification still returns a direction


def test_generate_signal_various_conditions():
    profile = DummyProfile()
    det = MTFTrendDetector(profile=profile)

    # SIDEWAYS
    sig, conf = det._generate_signal('SIDEWAYS', 0.6, 0.9, 'TRENDING', True)
    assert sig == 'NO_TRADE'
    assert conf == 0.0

    # low confluence
    sig2, conf2 = det._generate_signal('BULLISH', 0.6, 0.4, 'TRENDING', True)
    assert sig2 == 'NO_TRADE'
    assert conf2 == pytest.approx(0.4)

    # RANGING with low confluence
    sig3, conf3 = det._generate_signal('BULLISH', 0.6, 0.7, 'RANGING', True)
    # regime 'RANGING' and confluence <0.75 -> NO_TRADE with penalty 0.8
    assert sig3 == 'NO_TRADE'
    assert conf3 == pytest.approx(0.7 * 0.8)

    # higher TF not aligned and threshold
    profile.require_higher_tf_alignment = True
    sig4, conf4 = det._generate_signal('BEARISH', 0.6, 0.75, 'TRENDING', False)
    # confluence <0.8 and not aligned -> NO_TRADE
    assert sig4 == 'NO_TRADE'

    # valid BUY condition
    sig5, conf5 = det._generate_signal('BULLISH', 0.8, 0.9, 'TRENDING', True)
    assert sig5 == 'BUY'
    assert conf5 <= 1.0


def test_compute_confidence_weights():
    profile = DummyProfile()
    det = MTFTrendDetector(profile=profile)

    conf = det._compute_confidence(0.6, 0.7, 0.8)
    expected = 0.40*0.6 + 0.35*0.7 + 0.25*0.8
    assert pytest.approx(conf, rel=1e-6) == expected


def test_detect_flow_with_mocks(monkeypatch):
    profile = DummyProfile()
    structural = DummyStructural()
    regime = DummyRegime()
    ml = DummyML([0.1, 0.2, 0.7])

    det = MTFTrendDetector(profile=profile, structural_analyzer=structural, regime_classifier=regime, ml_model=ml)
    # override feature builder
    det._feature_builder = DummyFeatureBuilder()

    # prepare dfs_dict with H1 containing time
    dfs = {'H1': make_df_with_time(30), 'M15': make_df_with_time(30)}

    # patch mtf_analyzer.analyze to return simple object
    mtf_res = SimpleNamespace(
        mtf_score=0.7,
        direction=1,
        direction_label='BULLISH',
        confidence=0.8,
        individual_scores={'H1':0.7,'M15':0.6},
        individual_analysis={'H1': SimpleNamespace(direction=1), 'M15': SimpleNamespace(direction=1)},
        alignment_score=1.0,
        higher_tf_aligned=True,
    )

    monkeypatch.setattr(det.mtf_analyzer, 'analyze', lambda dfs_dict, structural_scores, weights: mtf_res)
    # patch confluence scorer
    monkeypatch.setattr(det.confluence_scorer, 'compute_confluence', lambda res, require_higher_tf: {'total_score':0.8})

    result = det.detect(dfs, compute_ml_features=True)

    assert result.trend_class in (1,2)
    assert result.signal in ('BUY','NO_TRADE')
    assert result.mtf_score == pytest.approx(0.7)
    assert result.primary_tf == 'H1'
    assert result.timestamp == dfs['H1']['time'].iloc[-1]
