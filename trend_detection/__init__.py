# trend_detection/__init__.py
"""
Trend Detection package for pyForex.
"""

from .mtf_analyzer_v2 import (
    MTFAnalyzerV2,
    MTFAnalysisResult,
    TimeframeAnalysis,
    MTFConfluenceScorer,
)

from .mtf_trend_detector import (
    MTFTrendDetector,
    MTFTrendResult,
    MTFTradingEngine,
)

__all__ = [
    'MTFAnalyzerV2',
    'MTFAnalysisResult',
    'TimeframeAnalysis',
    'MTFConfluenceScorer',
    'MTFTrendDetector',
    'MTFTrendResult',
    'MTFTradingEngine',
]
