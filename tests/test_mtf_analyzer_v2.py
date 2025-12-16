import numpy as np
import pandas as pd
import pytest

from trend_detection.mtf_analyzer_v2 import (
    MTFAnalyzerV2,
    TimeframeAnalysis,
    MTFAnalysisResult,
    MTFConfluenceScorer,
)


def make_ohlcv(n=100, start=1.1000, step=0.0002):
    price = start
    rows = []
    for i in range(n):
        o = price
        c = price + step
        h = max(o, c) + 0.0001
        l = min(o, c) - 0.0001
        rows.append({'open': o, 'high': h, 'low': l, 'close': c})
        price = c
    return pd.DataFrame(rows)


def test_empty_dfs_returns_empty_result():
    analyzer = MTFAnalyzerV2()
    result = analyzer.analyze({})

    assert isinstance(result, MTFAnalysisResult)
    assert result.mtf_score == pytest.approx(0.5)
    assert result.direction == 0
    assert result.direction_label == 'SIDEWAYS'
    assert result.confidence == pytest.approx(0.0)
    assert result.individual_scores == {}


def test_compute_weighted_score_basic():
    analyzer = MTFAnalyzerV2()
    scores = {'H1': 0.8, 'M15': 0.6}
    weights = {'H1': 0.7, 'M15': 0.3}

    mtf = analyzer._compute_weighted_score(scores, weights)
    expected = (0.8*0.7 + 0.6*0.3) / (0.7 + 0.3)
    assert mtf == pytest.approx(expected)


def test_determine_direction_weighted_majority():
    analyzer = MTFAnalyzerV2()

    a1 = TimeframeAnalysis('H4', 0.8, 1, 0.1, 30, 1, 1.0, 'STRONG')
    a2 = TimeframeAnalysis('H1', 0.7, 1, 0.05, 20, 1, 1.0, 'MODERATE')
    a3 = TimeframeAnalysis('M15', 0.3, -1, -0.02, 10, -1, -1.0, 'WEAK')

    analysis = {'H4': a1, 'H1': a2, 'M15': a3}
    weights = {'H4': 0.5, 'H1': 0.3, 'M15': 0.2}

    direction, label = analyzer._determine_direction(analysis, weights)
    assert direction == 1
    assert label == 'BULLISH'

    # Flip to bearish majority
    a1.direction = -1
    a2.direction = -1
    direction, label = analyzer._determine_direction(analysis, weights)
    assert direction == -1
    assert label == 'BEARISH'

    # Mixed -> sideways
    a1.direction = 1
    a2.direction = -1
    a3.direction = 0
    direction, label = analyzer._determine_direction(analysis, weights)
    assert direction == 0
    assert label == 'SIDEWAYS'


def test_compute_alignment():
    analyzer = MTFAnalyzerV2()

    a1 = TimeframeAnalysis('H4', 0.8, 1, 0.1, 30, 1, 1.0, 'STRONG')
    a2 = TimeframeAnalysis('H1', 0.7, 1, 0.05, 20, 1, 1.0, 'MODERATE')
    analysis = {'H4': a1, 'H1': a2}

    alignment = analyzer._compute_alignment(analysis)
    assert alignment == pytest.approx(1.0)

    # Disagreement
    a2.direction = -1
    alignment = analyzer._compute_alignment(analysis)
    # When TFs disagree evenly there is no majority => 0.0
    assert alignment == pytest.approx(0.0)

    # Single timeframe -> perfect alignment
    alignment = analyzer._compute_alignment({'H4': a1})
    assert alignment == pytest.approx(1.0)


def test_check_higher_tf_alignment_orders_and_logic():
    analyzer = MTFAnalyzerV2()

    # Higher TF H4 present and matches overall
    a_h4 = TimeframeAnalysis('H4', 0.8, 1, 0.1, 30, 1, 1.0, 'STRONG')
    a_h1 = TimeframeAnalysis('H1', 0.6, 1, 0.05, 20, 1, 1.0, 'MODERATE')
    analysis = {'H4': a_h4, 'H1': a_h1}

    # dfs_dict only used for TF ordering, pass dummy dataframes
    dfs = {'H4': make_ohlcv(100), 'H1': make_ohlcv(100)}

    aligned = analyzer._check_higher_tf_alignment(analysis, overall_direction=1, dfs_dict=dfs)
    assert aligned is True

    # If higher tf disagrees
    a_h4.direction = -1
    aligned = analyzer._check_higher_tf_alignment(analysis, overall_direction=1, dfs_dict=dfs)
    assert aligned is False

    # No analysis -> True
    assert analyzer._check_higher_tf_alignment({}, 1, {}) is True


def test_analyze_uses_analyze_timeframe_and_computes_result(monkeypatch):
    analyzer = MTFAnalyzerV2(weights={'H1': 0.6, 'M15': 0.4})

    # Patch _analyze_timeframe to return deterministic TimeframeAnalysis
    def fake_analyze_timeframe(df, tf, struct_score):
        return TimeframeAnalysis(tf, 0.7, 1, 0.05, 30, 1, 1.0, 'MODERATE')

    monkeypatch.setattr(MTFAnalyzerV2, '_analyze_timeframe', lambda self, df, tf, s: fake_analyze_timeframe(df, tf, s))

    dfs = {'H1': make_ohlcv(100), 'M15': make_ohlcv(100)}
    result = analyzer.analyze(dfs)

    assert isinstance(result, MTFAnalysisResult)
    assert 0.0 <= result.mtf_score <= 1.0
    assert result.direction in (-1, 0, 1)
    assert 'H1' in result.individual_scores
    assert 'M15' in result.individual_scores


def test_mtf_confluence_scorer_insufficient_data():
    scorer = MTFConfluenceScorer()
    # Build minimal analysis result
    res = MTFAnalysisResult(
        mtf_score=0.5,
        direction=0,
        direction_label='SIDEWAYS',
        confidence=0.0,
        individual_scores={},
        individual_analysis={},
        alignment_score=0.0,
        higher_tf_aligned=True,
    )

    out = scorer.compute_confluence(res)
    assert out['reason'] == 'Insufficient data'
    assert out['trade_recommended'] is False


def test_mtf_confluence_scorer_good_confluence():
    scorer = MTFConfluenceScorer()

    # Create three timeframe analyses all bullish with high ADX and positive slopes
    a1 = TimeframeAnalysis('H4', 0.8, 1, 0.05, 40, 1, 1.0, 'STRONG')
    a2 = TimeframeAnalysis('H1', 0.7, 1, 0.04, 35, 1, 1.0, 'STRONG')
    a3 = TimeframeAnalysis('M15', 0.6, 1, 0.03, 30, 1, 1.0, 'MODERATE')

    individual_analysis = {'H4': a1, 'H1': a2, 'M15': a3}

    res = MTFAnalysisResult(
        mtf_score=0.75,
        direction=1,
        direction_label='BULLISH',
        confidence=0.8,
        individual_scores={k: v.score for k, v in individual_analysis.items()},
        individual_analysis=individual_analysis,
        alignment_score=1.0,
        higher_tf_aligned=True,
    )

    out = scorer.compute_confluence(res)
    assert out['trade_recommended'] is True
    assert 'Good confluence' in out['reason']
    assert out['direction_agreement'] == pytest.approx(1.0)
    assert 0.0 <= out['total_score'] <= 1.0
