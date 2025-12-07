# trend_detection/mtf_analyzer_v2.py
"""
Multi-Timeframe Trend Analyzer V2

Flexible MTF analysis supporting configurable timeframe profiles.
Works with both (M5, M15, H1) and (M15, H1, H4) combinations.

Features:
- Configurable timeframe weights
- EMA slope confluence
- ADX trend strength
- Price position analysis
- Structural alignment scoring
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: str
    score: float              # 0-1 trend score
    direction: int            # -1, 0, 1
    ema_slope: float          # Normalized EMA slope
    adx: float                # ADX value
    adx_direction: int        # DI direction (-1 or 1)
    price_position: float     # -1 to 1 (position vs EMAs)
    trend_strength: str       # 'STRONG', 'MODERATE', 'WEAK', 'NONE'


@dataclass  
class MTFAnalysisResult:
    """Complete MTF analysis results."""
    mtf_score: float                          # Weighted confluence score (0-1)
    direction: int                            # Overall direction (-1, 0, 1)
    direction_label: str                      # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    confidence: float                         # Confidence in signal (0-1)
    individual_scores: Dict[str, float]       # Per-timeframe scores
    individual_analysis: Dict[str, TimeframeAnalysis]  # Detailed per-TF
    alignment_score: float                    # How well TFs agree (0-1)
    higher_tf_aligned: bool                   # Is higher TF in same direction
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'mtf_score': self.mtf_score,
            'direction': self.direction,
            'direction_label': self.direction_label,
            'confidence': self.confidence,
            'individual_scores': self.individual_scores,
            'alignment_score': self.alignment_score,
            'higher_tf_aligned': self.higher_tf_aligned,
        }


class MTFAnalyzerV2:
    """
    Multi-Timeframe Trend Analyzer with configurable profiles.
    
    Supports any combination of timeframes with custom weights.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ema_periods: Tuple[int, int, int] = (20, 50, 200),
        adx_period: int = 14,
        slope_lookback: int = 5,
    ):
        """
        Args:
            weights: Dict mapping timeframe to weight (should sum to 1.0)
            ema_periods: EMA periods to calculate
            adx_period: ADX calculation period
            slope_lookback: Bars to look back for slope calculation
        """
        self.weights = weights or {}
        self.ema_periods = ema_periods
        self.adx_period = adx_period
        self.slope_lookback = slope_lookback
    
    def analyze(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        structural_scores: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> MTFAnalysisResult:
        """
        Main MTF analysis method.
        
        Args:
            dfs_dict: Dict mapping timeframe to DataFrame
            structural_scores: Optional pre-computed structural scores per TF
            weights: Optional override for timeframe weights
        
        Returns:
            MTFAnalysisResult with comprehensive analysis
        """
        weights = weights or self.weights
        structural_scores = structural_scores or {}
        
        # Analyze each timeframe
        individual_analysis = {}
        individual_scores = {}
        
        for tf, df in dfs_dict.items():
            if df is None or df.empty:
                logger.warning(f"Empty data for {tf}")
                continue
            
            struct_score = structural_scores.get(tf, 0.5)
            analysis = self._analyze_timeframe(df, tf, struct_score)
            
            individual_analysis[tf] = analysis
            individual_scores[tf] = analysis.score
        
        if not individual_scores:
            return self._empty_result()
        
        # Compute weighted MTF score
        mtf_score = self._compute_weighted_score(individual_scores, weights)
        
        # Determine overall direction
        direction, direction_label = self._determine_direction(
            individual_analysis, weights
        )
        
        # Compute alignment score
        alignment_score = self._compute_alignment(individual_analysis)
        
        # Check higher TF alignment
        higher_tf_aligned = self._check_higher_tf_alignment(
            individual_analysis, direction, dfs_dict
        )
        
        # Compute confidence
        confidence = self._compute_confidence(
            mtf_score, alignment_score, higher_tf_aligned
        )
        
        return MTFAnalysisResult(
            mtf_score=mtf_score,
            direction=direction,
            direction_label=direction_label,
            confidence=confidence,
            individual_scores=individual_scores,
            individual_analysis=individual_analysis,
            alignment_score=alignment_score,
            higher_tf_aligned=higher_tf_aligned,
        )
    
    def _analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        structural_score: float,
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe."""
        close = df['close']
        
        # Calculate EMAs
        emas = {}
        for period in self.ema_periods:
            emas[period] = close.ewm(span=period, adjust=False).mean()
        
        # EMA slopes
        slopes = {}
        for period, ema in emas.items():
            if len(ema) > self.slope_lookback:
                slope = (ema.iloc[-1] - ema.iloc[-self.slope_lookback]) / ema.iloc[-self.slope_lookback]
                slopes[period] = slope * 100  # As percentage
            else:
                slopes[period] = 0
        
        avg_slope = np.mean(list(slopes.values()))
        ema_component = np.tanh(avg_slope * 20)  # Squash to [-1, 1]
        
        # Calculate ADX
        adx_value, plus_di, minus_di = self._calculate_adx(df)
        di_direction = 1 if plus_di > minus_di else -1
        adx_normalized = min(adx_value / 50, 1.0)
        adx_component = di_direction * adx_normalized
        
        # Price position relative to EMAs
        current_price = close.iloc[-1]
        positions = []
        for period, ema in emas.items():
            positions.append(1 if current_price > ema.iloc[-1] else -1)
        price_position = np.mean(positions)
        
        # Combine components
        # Weights: EMA slope=30%, Structural=40%, ADX=20%, Price position=10%
        tf_score = (
            0.30 * (ema_component + 1) / 2 +  # Normalize to 0-1
            0.40 * structural_score +
            0.20 * (adx_component + 1) / 2 +
            0.10 * (price_position + 1) / 2
        )
        
        # Determine direction
        if tf_score > 0.6:
            direction = 1
        elif tf_score < 0.4:
            direction = -1
        else:
            direction = 0
        
        # Trend strength label
        if adx_value > 40:
            strength = 'STRONG'
        elif adx_value > 25:
            strength = 'MODERATE'
        elif adx_value > 15:
            strength = 'WEAK'
        else:
            strength = 'NONE'
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            score=tf_score,
            direction=direction,
            ema_slope=avg_slope,
            adx=adx_value,
            adx_direction=di_direction,
            price_position=price_position,
            trend_strength=strength,
        )
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']
        period = self.adx_period
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # DI calculations
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return (
            float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0,
            float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0,
            float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0,
        )
    
    def _compute_weighted_score(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute weighted average of scores."""
        if not scores:
            return 0.5
        
        total_weight = 0
        weighted_sum = 0
        
        for tf, score in scores.items():
            weight = weights.get(tf, 1.0 / len(scores))
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _determine_direction(
        self,
        analysis: Dict[str, TimeframeAnalysis],
        weights: Dict[str, float],
    ) -> Tuple[int, str]:
        """Determine overall direction from individual analyses."""
        if not analysis:
            return 0, 'SIDEWAYS'
        
        # Weighted vote
        bullish_weight = 0
        bearish_weight = 0
        
        for tf, a in analysis.items():
            weight = weights.get(tf, 1.0 / len(analysis))
            if a.direction > 0:
                bullish_weight += weight
            elif a.direction < 0:
                bearish_weight += weight
        
        if bullish_weight > 0.5:
            return 1, 'BULLISH'
        elif bearish_weight > 0.5:
            return -1, 'BEARISH'
        else:
            return 0, 'SIDEWAYS'
    
    def _compute_alignment(
        self,
        analysis: Dict[str, TimeframeAnalysis],
    ) -> float:
        """
        Compute how well timeframes agree.
        1.0 = all agree, 0.0 = complete disagreement
        """
        if len(analysis) < 2:
            return 1.0
        
        directions = [a.direction for a in analysis.values()]
        
        # Count agreement with majority
        if sum(d > 0 for d in directions) > len(directions) / 2:
            majority = 1
        elif sum(d < 0 for d in directions) > len(directions) / 2:
            majority = -1
        else:
            majority = 0
        
        agreement_count = sum(1 for d in directions if d == majority)
        return agreement_count / len(directions)
    
    def _check_higher_tf_alignment(
        self,
        analysis: Dict[str, TimeframeAnalysis],
        overall_direction: int,
        dfs_dict: Dict[str, pd.DataFrame],
    ) -> bool:
        """Check if higher timeframe agrees with signal direction."""
        # TF ordering by minutes
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        
        # Find highest TF
        sorted_tfs = sorted(
            analysis.keys(),
            key=lambda x: tf_minutes.get(x, 0),
            reverse=True
        )
        
        if not sorted_tfs:
            return True
        
        higher_tf = sorted_tfs[0]
        higher_analysis = analysis.get(higher_tf)
        
        if higher_analysis is None:
            return True
        
        # Check alignment
        return higher_analysis.direction == overall_direction or overall_direction == 0
    
    def _compute_confidence(
        self,
        mtf_score: float,
        alignment_score: float,
        higher_tf_aligned: bool,
    ) -> float:
        """Compute confidence in the signal."""
        # Base confidence from MTF score distance from neutral
        base_conf = abs(mtf_score - 0.5) * 2
        
        # Adjust for alignment
        conf = base_conf * alignment_score
        
        # Penalty for higher TF disagreement
        if not higher_tf_aligned:
            conf *= 0.7
        
        return min(1.0, conf)
    
    def _empty_result(self) -> MTFAnalysisResult:
        """Return empty/neutral result."""
        return MTFAnalysisResult(
            mtf_score=0.5,
            direction=0,
            direction_label='SIDEWAYS',
            confidence=0.0,
            individual_scores={},
            individual_analysis={},
            alignment_score=0.0,
            higher_tf_aligned=True,
        )
    
    @classmethod
    def from_profile(cls, profile: "MTFProfile") -> "MTFAnalyzerV2":
        """Create analyzer from MTF profile configuration."""
        return cls(weights=profile.weights)


class MTFConfluenceScorer:
    """
    Specialized confluence scoring for MTF analysis.
    
    Computes detailed confluence metrics between timeframes.
    """
    
    def __init__(self):
        pass
    
    def compute_confluence(
        self,
        analysis_result: MTFAnalysisResult,
        require_higher_tf: bool = True,
    ) -> Dict:
        """
        Compute detailed confluence metrics.
        
        Returns:
            Dict with confluence scores and breakdown
        """
        scores = analysis_result.individual_analysis
        
        if len(scores) < 2:
            return {
                'total_score': 0.5,
                'direction_agreement': 1.0,
                'strength_agreement': 1.0,
                'trade_recommended': False,
                'reason': 'Insufficient data'
            }
        
        # Direction agreement
        directions = [a.direction for a in scores.values()]
        bullish = sum(1 for d in directions if d > 0)
        bearish = sum(1 for d in directions if d < 0)
        
        direction_agreement = max(bullish, bearish) / len(directions)
        
        # Strength agreement
        strengths = [a.adx for a in scores.values()]
        avg_strength = np.mean(strengths)
        strength_std = np.std(strengths)
        strength_agreement = 1.0 - min(strength_std / 20, 1.0)
        
        # EMA slope agreement
        slopes = [a.ema_slope for a in scores.values()]
        slope_signs = [1 if s > 0 else -1 for s in slopes]
        slope_agreement = abs(sum(slope_signs)) / len(slope_signs)
        
        # Total confluence score
        total_score = (
            0.40 * direction_agreement +
            0.30 * slope_agreement +
            0.20 * strength_agreement +
            0.10 * (1.0 if analysis_result.higher_tf_aligned else 0.5)
        )
        
        # Trade recommendation
        trade_recommended = (
            total_score >= 0.65 and
            direction_agreement >= 0.67 and
            (not require_higher_tf or analysis_result.higher_tf_aligned)
        )
        
        # Reason
        if trade_recommended:
            reason = f"Good confluence ({total_score:.2f})"
        elif direction_agreement < 0.67:
            reason = f"TFs disagree on direction ({direction_agreement:.2f})"
        elif not analysis_result.higher_tf_aligned:
            reason = "Higher TF not aligned"
        else:
            reason = f"Low confluence ({total_score:.2f})"
        
        return {
            'total_score': total_score,
            'direction_agreement': direction_agreement,
            'strength_agreement': strength_agreement,
            'slope_agreement': slope_agreement,
            'avg_adx': avg_strength,
            'trade_recommended': trade_recommended,
            'reason': reason,
        }
