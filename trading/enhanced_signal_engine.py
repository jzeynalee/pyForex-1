# trading/enhanced_signal_engine.py
"""
Main FusionFX Trend Detector (FTDM-V1)
Orchestrates all 5 steps of trend detection

Updated to integrate TrendClassifier for Step 4.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union

from trend_detection.structural_analyzer import StructuralAnalyzer
from trend_detection.mtf_analyzer import MTFAnalyzer
from trend_detection.regime_classifier import RegimeClassifier
from trend_detection.trend_features import TrendFeatureBuilder

logger = logging.getLogger(__name__)


class FusionFXTrendDetector:
    """
    Complete FTDM-V1 Implementation
    5-step hybrid trend detection system
    
    Step 4 now supports:
    - TrendClassifier (XGBoost/sklearn): Takes 13 tabular features
    - None: Uses neutral defaults (direction=0, confidence=0.5)
    """
    
    # Expected feature order for Step 4 ML model
    FEATURE_ORDER = [
        'struct_score', 'mtf_score', 'regime', 'adx',
        'plus_di', 'minus_di', 'price_above_ema20',
        'price_above_ema50', 'price_above_ema200',
        'ema_alignment', 'vol_compression', 'roc_5', 'roc_10'
    ]
    
    def __init__(
        self,
        ml_model: Optional[Any] = None,
        ml_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            ml_model: Pre-loaded TrendClassifier instance (optional)
            ml_model_path: Path to saved TrendClassifier (optional)
            
        If neither is provided, Step 4 uses neutral defaults.
        """
        self.structural_analyzer = StructuralAnalyzer()
        self.mtf_analyzer = MTFAnalyzer()
        self.regime_classifier = RegimeClassifier()
        self.feature_builder = TrendFeatureBuilder()
        
        # Load ML model
        self.ml_model = ml_model
        if self.ml_model is None and ml_model_path is not None:
            self.ml_model = self._load_model(ml_model_path)
        
        if self.ml_model is not None:
            logger.info("FusionFXTrendDetector: ML model loaded for Step 4")
        else:
            logger.info("FusionFXTrendDetector: Step 4 disabled (no ML model)")
    
    def _load_model(self, path: Union[str, Path]) -> Optional[Any]:
        """Load TrendClassifier from disk."""
        try:
            from models.trend_classifier import TrendClassifier
            return TrendClassifier.load(path)
        except Exception as e:
            logger.warning(f"Failed to load ML model from {path}: {e}")
            return None
    
    def detect_trend(self, dfs_dict: Dict[str, "pd.DataFrame"]) -> Dict:
        """
        Main trend detection pipeline
        
        Args:
            dfs_dict: {
                'H4': DataFrame with H4 data,
                'H1': DataFrame with H1 data,
                'M15': DataFrame with M15 data
            }
        
        Returns:
            {
                'trend_class': int (0-4),
                'trend_strength': float (0-100),
                'direction': str ('BULLISH', 'BEARISH', 'SIDEWAYS'),
                'confidence': float (0-1),
                'details': dict with all intermediate results
            }
        """
        
        # ========================================
        # STEP 1: STRUCTURAL TREND DETECTION
        # ========================================
        structural_results = {}
        for tf, df in dfs_dict.items():
            structural_results[tf] = self.structural_analyzer.analyze(df)
        
        # Primary timeframe (H1) structure
        primary_structure = structural_results['H1']
        
        # ========================================
        # STEP 2: MULTI-TIMEFRAME CONFLUENCE
        # ========================================
        structural_scores = {
            tf: result['score'] for tf, result in structural_results.items()
        }
        
        mtf_result = self.mtf_analyzer.analyze(dfs_dict, structural_scores)
        mtf_score = mtf_result['mtf_score']
        
        # ========================================
        # STEP 3: REGIME CONDITIONING
        # ========================================
        regime_info = self.regime_classifier.classify_regime(dfs_dict['H1'])
        regime_filter = self.regime_classifier.apply_regime_filter(
            regime_info, 
            mtf_score
        )
        regime_adjusted_score = regime_filter['adjusted_score']
        
        # ========================================
        # STEP 4: ML PROBABILISTIC CONFIRMATION
        # ========================================
        ml_direction = 0
        ml_confidence = 0.5
        ml_probs = None
        
        if self.ml_model is not None:
            try:
                # Build features for ML model
                features = self.feature_builder.build_features(
                    dfs_dict['H1'],
                    primary_structure,
                    mtf_result,
                    regime_info
                )
                
                # Prepare input vector
                feature_vector = self._prepare_ml_input(features)
                
                # Get ML prediction
                # TrendClassifier.predict_proba returns [P(BEAR), P(SIDEWAYS), P(BULL)]
                ml_probs = self.ml_model.predict_proba([feature_vector])[0]
                
                ml_confidence = float(np.max(ml_probs))
                
                if ml_confidence >= 0.60:
                    # Map argmax to direction: 0->-1 (BEAR), 1->0 (SIDEWAYS), 2->1 (BULL)
                    ml_direction = int(np.argmax(ml_probs)) - 1
                else:
                    ml_direction = 0  # Low confidence -> sideways
                    
                logger.debug(
                    f"Step 4 ML: probs={ml_probs}, direction={ml_direction}, "
                    f"confidence={ml_confidence:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                ml_direction = 0
                ml_confidence = 0.5
        
        # ========================================
        # STEP 5: TREND FUSION & CLASSIFICATION
        # ========================================
        
        # Normalize structural direction to 0-1
        struct_normalized = (primary_structure['direction'] + 1) / 2
        
        # Normalize ML direction to 0-1
        ml_normalized = (ml_direction + 1) / 2
        
        # Final fusion score
        # Weights: struct=35%, mtf=35%, ml=20%, regime=10%
        final_score = (
            0.35 * primary_structure['score'] * struct_normalized +
            0.35 * mtf_score +
            0.20 * ml_confidence * ml_normalized +
            0.10 * regime_adjusted_score
        )
        
        # Scale to 0-100
        trend_strength = final_score * 100
        
        # Classify trend
        trend_class, trend_name, direction = self._classify_trend(
            trend_strength,
            primary_structure['direction'],
            regime_info['regime']
        )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            primary_structure['score'],
            mtf_score,
            ml_confidence,
            regime_adjusted_score
        )
        
        return {
            'trend_class': trend_class,
            'trend_name': trend_name,
            'trend_strength': trend_strength,
            'direction': direction,
            'confidence': confidence,
            'details': {
                'structural': structural_results,
                'mtf': mtf_result,
                'regime': regime_info,
                'regime_filter': regime_filter,
                'ml_direction': ml_direction,
                'ml_confidence': ml_confidence,
                'ml_probabilities': ml_probs.tolist() if ml_probs is not None else None,
                'final_score': final_score
            }
        }
    
    def _classify_trend(
        self,
        trend_strength: float,
        struct_direction: int,
        regime: str,
    ) -> tuple:
        """
        Map trend strength to one of 5 classes:
        0: Sideways/Compression
        1: Early Bull Trend
        2: Mature Bull Trend
        3: Early Bear Trend
        4: Mature Bear Trend
        """
        if trend_strength < 30 or regime == 'RANGING':
            return 0, 'Sideways/Compression', 'SIDEWAYS'
        
        if struct_direction > 0:  # Bullish
            if 30 <= trend_strength < 55:
                return 1, 'Early Bull Trend', 'BULLISH'
            else:
                return 2, 'Mature Bull Trend', 'BULLISH'
        else:  # Bearish
            if 30 <= trend_strength < 55:
                return 3, 'Early Bear Trend', 'BEARISH'
            else:
                return 4, 'Mature Bear Trend', 'BEARISH'
    
    def _calculate_confidence(
        self,
        struct_score: float,
        mtf_score: float,
        ml_conf: float,
        regime_score: float,
    ) -> float:
        """Calculate overall confidence in trend classification."""
        # Weighted average of all confidence components
        confidence = (
            0.35 * struct_score +
            0.35 * mtf_score +
            0.20 * ml_conf +
            0.10 * regime_score
        )
        return float(confidence)
    
    def _prepare_ml_input(self, features: Dict) -> list:
        """
        Convert feature dict to format expected by TrendClassifier.
        
        Args:
            features: Dict from TrendFeatureBuilder.build_features()
        
        Returns:
            List of 13 features in correct order
        """
        return [features.get(k, 0) for k in self.FEATURE_ORDER]


# Backwards compatibility alias
EnhancedSignalEngine = FusionFXTrendDetector