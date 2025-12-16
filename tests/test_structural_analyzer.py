import pandas as pd
import pytest
from types import SimpleNamespace

import trend_detection.structural_analyzer as sa_module
from trend_detection.structural_analyzer import StructuralAnalyzer


class DummySwingDetector:
    def __init__(self, atr_multiplier=3.5, confirmation_candles=2):
        self.atr_multiplier = atr_multiplier
        self.confirmation_candles = confirmation_candles
        self.detected = False
        self.last_df = None
        # values to return
        self.swings_df = pd.DataFrame({'x': []})
        self.structure = ('unknown', 0.0)

    def detect_swings(self, df):
        self.detected = True
        self.last_df = df
        return self.swings_df

    def classify_structure(self, swings_df):
        return self.structure


def make_dummy_df():
    return pd.DataFrame({'open': [1,2,3], 'high': [1,2,3], 'low':[1,2,3], 'close':[1,2,3]})


def test_init_passes_parameters(monkeypatch):
    # Patch the SwingDetector in module to our dummy
    monkeypatch.setattr(sa_module, 'SwingDetector', DummySwingDetector)

    analyzer = StructuralAnalyzer(atr_multiplier=4.2, confirmation_candles=5)
    # The StructuralAnalyzer should have created a DummySwingDetector with same params
    assert isinstance(analyzer.swing_detector, DummySwingDetector)
    assert analyzer.swing_detector.atr_multiplier == 4.2
    assert analyzer.swing_detector.confirmation_candles == 5


def test_analyze_bullish(monkeypatch):
    dummy = DummySwingDetector()
    dummy.swings_df = pd.DataFrame({'swing': [1,2,3]})
    dummy.structure = ('bullish', 0.8)

    monkeypatch.setattr(sa_module, 'SwingDetector', lambda *a, **k: dummy)

    analyzer = StructuralAnalyzer()
    df = make_dummy_df()
    out = analyzer.analyze(df)

    assert out['direction'] == 1
    assert out['score'] == pytest.approx(0.8)
    assert out['type'] == 'bullish'
    assert out['swings_df'] is dummy.swings_df


def test_analyze_bearish(monkeypatch):
    dummy = DummySwingDetector()
    dummy.swings_df = pd.DataFrame({'swing': [1,2]})
    dummy.structure = ('bearish', 0.9)

    monkeypatch.setattr(sa_module, 'SwingDetector', lambda *a, **k: dummy)

    analyzer = StructuralAnalyzer()
    df = make_dummy_df()
    out = analyzer.analyze(df)

    assert out['direction'] == -1
    assert out['score'] == pytest.approx(0.9)
    assert out['type'] == 'bearish'


def test_analyze_mixed_penalty(monkeypatch):
    dummy = DummySwingDetector()
    dummy.swings_df = pd.DataFrame({'swing': [1]})
    dummy.structure = ('bullish', 0.5)  # below 0.6 threshold

    monkeypatch.setattr(sa_module, 'SwingDetector', lambda *a, **k: dummy)

    analyzer = StructuralAnalyzer()
    df = make_dummy_df()
    out = analyzer.analyze(df)

    # direction should be 0 and score penalized by 0.3 (floor at 0)
    assert out['direction'] == 0
    assert out['score'] == pytest.approx(max(0.0, 0.5 - 0.3))
    assert out['type'] == 'bullish'


def test_analyze_unknown_low_score_floor(monkeypatch):
    dummy = DummySwingDetector()
    dummy.swings_df = pd.DataFrame({'swing': []})
    dummy.structure = ('unknown', 0.2)

    monkeypatch.setattr(sa_module, 'SwingDetector', lambda *a, **k: dummy)

    analyzer = StructuralAnalyzer()
    df = make_dummy_df()
    out = analyzer.analyze(df)

    assert out['direction'] == 0
    assert out['score'] == pytest.approx(0.0)
    assert out['type'] == 'unknown'
