# trend_detection/mtf_trend_detector.py
"""
Integrated Multi-Timeframe Trend Detector

Combines all MTF components:
- MTF Data Provider (data fetching)
- MTF Analyzer V2 (trend analysis)
- MTF Feature Builder (ML features)
- Structural Analyzer (swing detection)
- Regime Classifier (market regime)

Supports configurable timeframe profiles:
- SCALP: M5, M15, H1
- SWING: M15, H1, H4
"""

import logging
from typing import Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# MTF analyzer implementations are imported at module level so they are
# defined for use during initialization and by other methods. Importing
# here avoids undefined name errors and keeps lazy imports for analyzers
# unnecessary while still preserving circular-import safety via
# TYPE_CHECKING for heavier type hints.
from trend_detection.mtf_analyzer_v2 import MTFAnalyzerV2, MTFConfluenceScorer, MTFAnalysisResult

if TYPE_CHECKING:
    # Imported only for type hints to avoid runtime circular imports
    from utils.mtf_config import MTFProfile
    from trend_detection.structural_analyzer import StructuralAnalyzer
    from trend_detection.regime_classifier import RegimeClassifier


@dataclass
class MTFTrendResult:
    """Complete MTF trend detection result."""
    # Main outputs
    trend_class: int              # 0-4 (Sideways, Early Bull, Mature Bull, Early Bear, Mature Bear)
    trend_name: str               # Human-readable trend name
    trend_strength: float         # 0-100 trend strength score
    direction: str                # 'BULLISH', 'BEARISH', 'SIDEWAYS'
    confidence: float             # 0-1 confidence score
    
    # Trading signal
    signal: str                   # 'BUY', 'SELL', 'NO_TRADE'
    signal_confidence: float      # Signal-specific confidence
    
    # MTF Analysis
    mtf_score: float              # Weighted MTF score
    mtf_alignment: float          # How well TFs agree
    higher_tf_aligned: bool       # Higher TF in same direction
    
    # Per-timeframe details
    timeframe_scores: Dict[str, float]
    timeframe_directions: Dict[str, int]
    
    # Regime info
    regime: str                   # 'TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL'
    regime_confidence: float
    
    # ML features (for further processing)
    ml_features: Optional[Dict[str, float]] = None
    
    # Metadata
    profile_name: str = ""
    primary_tf: str = ""
    timestamp: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trend_class': self.trend_class,
            'trend_name': self.trend_name,
            'trend_strength': self.trend_strength,
            'direction': self.direction,
            'confidence': self.confidence,
            'signal': self.signal,
            'signal_confidence': self.signal_confidence,
            'mtf_score': self.mtf_score,
            'mtf_alignment': self.mtf_alignment,
            'higher_tf_aligned': self.higher_tf_aligned,
            'timeframe_scores': self.timeframe_scores,
            'timeframe_directions': self.timeframe_directions,
            'regime': self.regime,
            'profile_name': self.profile_name,
            'primary_tf': self.primary_tf,
        }


class MTFTrendDetector:
    """
    Complete Multi-Timeframe Trend Detection System.
    
    Pipeline:
    1. Fetch MTF data
    2. Structural analysis per TF
    3. MTF confluence analysis
    4. Regime classification
    5. ML feature generation
    6. Signal generation
    
    Usage:
        detector = MTFTrendDetector.from_profile("SWING")
        result = detector.detect(data_provider)
    """
    
    TREND_CLASSES = {
        0: 'Sideways/Compression',
        1: 'Early Bull Trend',
        2: 'Mature Bull Trend',
        3: 'Early Bear Trend',
        4: 'Mature Bear Trend',
    }
    
    def __init__(
        self,
        profile: "MTFProfile",
        ml_model: Optional[Any] = None,
        structural_analyzer: Optional["StructuralAnalyzer"] = None,
        regime_classifier: Optional["RegimeClassifier"] = None,
    ):
        """
        Args:
            profile: MTF profile configuration
            ml_model: Optional ML model for trend confirmation
            structural_analyzer: Optional custom structural analyzer
            regime_classifier: Optional custom regime classifier
        """
        self.profile = profile
        self.ml_model = ml_model
        
        # Initialize components
        self.mtf_analyzer = MTFAnalyzerV2.from_profile(profile)
        self.confluence_scorer = MTFConfluenceScorer()
        
        # Lazy load these to avoid circular imports
        self._structural_analyzer = structural_analyzer
        self._regime_classifier = regime_classifier
        self._feature_builder = None
    
    @property
    def structural_analyzer(self):
        """Lazy load structural analyzer."""
        if self._structural_analyzer is None:
            try:
                from trend_detection.structural_analyzer import StructuralAnalyzer
                self._structural_analyzer = StructuralAnalyzer()
            except ImportError:
                logger.warning("StructuralAnalyzer not available")
        return self._structural_analyzer
    
    @property
    def regime_classifier(self):
        """Lazy load regime classifier."""
        if self._regime_classifier is None:
            try:
                from trend_detection.regime_classifier import RegimeClassifier
                self._regime_classifier = RegimeClassifier()
            except ImportError:
                logger.warning("RegimeClassifier not available")
        return self._regime_classifier
    
    @property
    def feature_builder(self):
        """Lazy load feature builder."""
        if self._feature_builder is None:
            from utils.mtf_features import MTFFeatureBuilder
            self._feature_builder = MTFFeatureBuilder()
        return self._feature_builder
    
    def detect(
        self,
        dfs_dict: Dict[str, pd.DataFrame],
        compute_ml_features: bool = True,
    ) -> MTFTrendResult:
        """
        Run complete trend detection pipeline.
        
        Args:
            dfs_dict: Dict mapping timeframe to DataFrame
            compute_ml_features: Whether to compute ML features
        
        Returns:
            MTFTrendResult with comprehensive analysis
        """
        primary_tf = self.profile.primary_tf.value
        
        # Validate data
        for tf in self.profile.timeframe_strings:
            if tf not in dfs_dict or dfs_dict[tf] is None or dfs_dict[tf].empty:
                logger.warning(f"Missing or empty data for {tf}")
        
        # Step 1: Structural analysis per timeframe
        structural_results = {}
        for tf, df in dfs_dict.items():
            if df is not None and not df.empty and self.structural_analyzer:
                structural_results[tf] = self.structural_analyzer.analyze(df)
            else:
                structural_results[tf] = {'direction': 0, 'score': 0.5, 'type': 'unknown'}
        
        # Extract structural scores
        structural_scores = {
            tf: result['score'] for tf, result in structural_results.items()
        }
        
        # Step 2: MTF confluence analysis
        mtf_result = self.mtf_analyzer.analyze(
            dfs_dict,
            structural_scores=structural_scores,
            weights=self.profile.weights,
        )
        
        # Step 3: Regime classification (on primary TF)
        regime_info = {'regime': 'UNKNOWN', 'adx': 0, 'volatility': 0}
        if primary_tf in dfs_dict and self.regime_classifier:
            regime_info = self.regime_classifier.classify_regime(dfs_dict[primary_tf])
        
        # Step 4: Confluence scoring
        confluence = self.confluence_scorer.compute_confluence(
            mtf_result,
            require_higher_tf=self.profile.require_higher_tf_alignment,
        )
        
        # Step 5: ML features (optional)
        ml_features = None
        ml_confidence = 0.5
        ml_direction = 0
        
        if compute_ml_features:
            feature_set = self.feature_builder.build_features(dfs_dict, primary_tf)
            ml_features = feature_set.features
            
            # If ML model available, get prediction
            if self.ml_model is not None:
                try:
                    feature_vector = self._prepare_ml_input(ml_features)
                    probs = self.ml_model.predict_proba([feature_vector])[0]
                    ml_confidence = float(np.max(probs))
                    ml_direction = int(np.argmax(probs)) - 1  # -1, 0, 1
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
        
        # Step 6: Final trend classification
        trend_class, trend_name, direction = self._classify_trend(
            mtf_result,
            structural_results.get(primary_tf, {}),
            regime_info,
            ml_direction,
            ml_confidence,
        )
        
        # Step 7: Signal generation
        signal, signal_confidence = self._generate_signal(
            direction=direction,
            mtf_score=mtf_result.mtf_score,
            confluence_score=confluence['total_score'],
            regime=regime_info['regime'],
            higher_tf_aligned=mtf_result.higher_tf_aligned,
        )
        
        # Compute overall confidence
        confidence = self._compute_confidence(
            mtf_result.confidence,
            confluence['total_score'],
            ml_confidence if self.ml_model else 0.5,
        )
        
        # Build per-TF info
        tf_scores = mtf_result.individual_scores
        tf_directions = {
            tf: a.direction
            for tf, a in mtf_result.individual_analysis.items()
        }
        
        # Get timestamp
        timestamp = None
        if primary_tf in dfs_dict and 'time' in dfs_dict[primary_tf].columns:
            timestamp = dfs_dict[primary_tf]['time'].iloc[-1]
        
        return MTFTrendResult(
            trend_class=trend_class,
            trend_name=trend_name,
            trend_strength=mtf_result.mtf_score * 100,
            direction=direction,
            confidence=confidence,
            signal=signal,
            signal_confidence=signal_confidence,
            mtf_score=mtf_result.mtf_score,
            mtf_alignment=mtf_result.alignment_score,
            higher_tf_aligned=mtf_result.higher_tf_aligned,
            timeframe_scores=tf_scores,
            timeframe_directions=tf_directions,
            regime=regime_info['regime'],
            regime_confidence=regime_info.get('confidence', 0.5),
            ml_features=ml_features,
            profile_name=self.profile.name,
            primary_tf=primary_tf,
            timestamp=timestamp,
        )
    
    def _classify_trend(
        self,
        mtf_result: "MTFAnalysisResult",
        structural_result: Dict,
        regime_info: Dict,
        ml_direction: int,
        ml_confidence: float,
    ) -> Tuple[int, str, str]:
        """Classify trend into one of 5 classes."""
        mtf_score = mtf_result.mtf_score
        struct_direction = structural_result.get('direction', 0)
        regime = regime_info.get('regime', 'UNKNOWN')
        
        # Scale MTF score to 0-100
        strength = mtf_score * 100
        
        # Adjust for ML prediction if confident
        if ml_confidence > 0.7:
            if ml_direction != 0:
                # Boost/reduce strength based on ML agreement
                if (ml_direction > 0 and mtf_result.direction > 0) or \
                   (ml_direction < 0 and mtf_result.direction < 0):
                    strength *= 1.1
                else:
                    strength *= 0.9
        
        # Classification logic
        if strength < 30 or regime == 'RANGING':
            return 0, self.TREND_CLASSES[0], 'SIDEWAYS'
        
        # Use MTF direction as primary, fallback to structural
        direction = mtf_result.direction if mtf_result.direction != 0 else struct_direction
        
        if direction > 0:  # Bullish
            if 30 <= strength < 55:
                return 1, self.TREND_CLASSES[1], 'BULLISH'
            else:
                return 2, self.TREND_CLASSES[2], 'BULLISH'
        else:  # Bearish
            if 30 <= strength < 55:
                return 3, self.TREND_CLASSES[3], 'BEARISH'
            else:
                return 4, self.TREND_CLASSES[4], 'BEARISH'
    
    def _generate_signal(
        self,
        direction: str,
        mtf_score: float,
        confluence_score: float,
        regime: str,
        higher_tf_aligned: bool,
    ) -> Tuple[str, float]:
        """Generate trading signal."""
        threshold = self.profile.min_confluence_score
        
        # No signal conditions
        if direction == 'SIDEWAYS':
            return 'NO_TRADE', 0.0
        
        if confluence_score < threshold:
            return 'NO_TRADE', confluence_score
        
        # Regime filter
        if regime == 'RANGING' and confluence_score < 0.75:
            return 'NO_TRADE', confluence_score * 0.8
        
        # Higher TF filter
        if self.profile.require_higher_tf_alignment and not higher_tf_aligned:
            # Allow with penalty if confluence is strong
            if confluence_score < 0.80:
                return 'NO_TRADE', confluence_score * 0.7
        
        # Generate signal
        if direction == 'BULLISH':
            signal = 'BUY'
        elif direction == 'BEARISH':
            signal = 'SELL'
        else:
            signal = 'NO_TRADE'
        
        # Calculate signal confidence
        signal_conf = confluence_score
        if higher_tf_aligned:
            signal_conf *= 1.1
        if regime == 'TRENDING':
            signal_conf *= 1.1
        
        return signal, min(1.0, signal_conf)
    
    def _compute_confidence(
        self,
        mtf_confidence: float,
        confluence_score: float,
        ml_confidence: float,
    ) -> float:
        """Compute overall confidence."""
        # Weighted average
        conf = (
            0.40 * mtf_confidence +
            0.35 * confluence_score +
            0.25 * ml_confidence
        )
        return min(1.0, conf)
    
    def _prepare_ml_input(self, features: Dict) -> list:
        """Prepare features for ML model."""
        # Expected feature order for TrendClassifier
        feature_order = [
            'struct_score', 'mtf_score', 'regime', 'adx',
            'plus_di', 'minus_di', 'price_above_ema20',
            'price_above_ema50', 'price_above_ema200',
            'ema_alignment', 'vol_compression', 'roc_5', 'roc_10'
        ]
        
        # Map features to expected format
        primary_tf = self.profile.primary_tf.value
        
        mapped = {}
        mapped['struct_score'] = features.get(f'{primary_tf}_range_position', 0.5)
        mapped['mtf_score'] = features.get('mtf_weighted_direction', 0) * 0.5 + 0.5
        mapped['regime'] = 1 if features.get(f'{primary_tf}_adx', 0) > 25 else 0
        mapped['adx'] = features.get(f'{primary_tf}_adx', 0)
        mapped['plus_di'] = features.get(f'{primary_tf}_plus_di', 0)
        mapped['minus_di'] = features.get(f'{primary_tf}_minus_di', 0)
        mapped['price_above_ema20'] = features.get(f'{primary_tf}_price_vs_ema20', 0)
        mapped['price_above_ema50'] = features.get(f'{primary_tf}_price_vs_ema50', 0)
        mapped['price_above_ema200'] = features.get(f'{primary_tf}_price_vs_ema200', 0)
        mapped['ema_alignment'] = features.get(f'{primary_tf}_ema_alignment', 0)
        mapped['vol_compression'] = features.get(f'{primary_tf}_vol_compression', 1)
        mapped['roc_5'] = features.get(f'{primary_tf}_roc_5', 0)
        mapped['roc_10'] = features.get(f'{primary_tf}_roc_10', 0)
        
        return [mapped.get(f, 0) for f in feature_order]
    
    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        ml_model: Optional[Any] = None,
    ) -> "MTFTrendDetector":
        """
        Create detector from profile name.
        
        Args:
            profile_name: 'SCALP', 'SWING', or 'INTRADAY'
            ml_model: Optional ML model
        
        Returns:
            Configured MTFTrendDetector
        """
        from utils.mtf_config import get_profile
        profile = get_profile(profile_name)
        return cls(profile=profile, ml_model=ml_model)
    
    @classmethod
    def create_scalp_detector(cls, ml_model: Optional[Any] = None) -> "MTFTrendDetector":
        """Create detector for M5/M15/H1 scalping."""
        return cls.from_profile("SCALP", ml_model)
    
    @classmethod
    def create_swing_detector(cls, ml_model: Optional[Any] = None) -> "MTFTrendDetector":
        """Create detector for M15/H1/H4 swing trading."""
        return cls.from_profile("SWING", ml_model)


class MTFTradingEngine:
    """
    High-level trading engine using MTF analysis.
    
    Integrates:
    - MTF Trend Detector
    - MTF Data Provider
    - Risk Management
    - Signal Generation
    """
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        profile_name: str = "SWING",
        connector: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        ml_model: Optional[Any] = None,
    ):
        self.symbol = symbol
        
        # Initialize components
        from trading.mtf_data_provider import MTFDataProvider
        from utils.mtf_config import get_profile
        
        self.profile = get_profile(profile_name)
        self.data_provider = MTFDataProvider(symbol=symbol, connector=connector)
        self.trend_detector = MTFTrendDetector(profile=self.profile, ml_model=ml_model)
        self.risk_manager = risk_manager
        
        # State
        self.last_signal: Optional[str] = None
        self.last_analysis: Optional[MTFTrendResult] = None
    
    def analyze(self) -> MTFTrendResult:
        """
        Fetch data and run analysis.
        
        Returns:
            MTFTrendResult with analysis and signal
        """
        # Fetch MTF data
        dfs_dict = self.data_provider.fetch_for_profile(self.profile)
        
        # Validate
        is_valid, errors = self.data_provider.validate_data(
            dfs_dict, 
            min_bars=self.profile.min_bars
        )
        
        if not is_valid:
            logger.warning(f"Data validation failed: {errors}")
        
        # Run analysis
        result = self.trend_detector.detect(dfs_dict)
        
        self.last_analysis = result
        self.last_signal = result.signal
        
        return result
    
    def get_trade_params(
        self,
        result: Optional[MTFTrendResult] = None,
    ) -> Optional[Dict]:
        """
        Get trade parameters from analysis.
        
        Args:
            result: MTFTrendResult (uses last analysis if None)
        
        Returns:
            Dict with volume, sl, tp or None if no trade
        """
        result = result or self.last_analysis
        
        if result is None or result.signal == 'NO_TRADE':
            return None
        
        if self.risk_manager is None:
            return {
                'signal': result.signal,
                'confidence': result.signal_confidence,
            }
        
        # Get data for ATR calculation
        dfs_dict = self.data_provider.fetch_for_profile(self.profile)
        primary_df = dfs_dict.get(self.profile.primary_tf.value)
        
        if primary_df is None:
            return None
        
        # Calculate trade params
        params = self.risk_manager.get_params(
            df=primary_df,
            signal=result.signal,
        )
        
        # Adjust for confluence
        if self.profile.adjust_risk_by_confluence if hasattr(self.profile, 'adjust_risk_by_confluence') else False:
            if result.confidence > 0.75:
                # Increase size for high confidence
                params = params._replace(volume=params.volume * 1.2)
            elif result.confidence < 0.55:
                # Decrease size for low confidence
                params = params._replace(volume=params.volume * 0.8)
        
        return {
            'signal': result.signal,
            'volume': params.volume,
            'sl': params.stop_loss,
            'tp': params.take_profit,
            'confidence': result.signal_confidence,
            'trend': result.trend_name,
            'regime': result.regime,
        }
