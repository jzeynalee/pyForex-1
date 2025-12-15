# trend_detection/fusion_trend_detector.py
"""
Main FusionFX Trend Detector (FTDM-V1)
Orchestrates all 5 steps of trend detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from trend_detection.structural_analyzer import StructuralAnalyzer
from trend_detection.mtf_analyzer_v2 import MTFAnalyzerV2
from trend_detection.regime_classifier import RegimeClassifier
from trend_detection.trend_features import TrendFeatureBuilder

class FusionFXTrendDetector:
    """
    Complete FTDM-V1 Implementation
    5-step hybrid trend detection system
    """
    
    def __init__(self, ml_model=None):
        """
        Args:
            ml_model: Your trained Fusion model (optional for Step 4)
        """
        self.structural_analyzer = StructuralAnalyzer()
        self.mtf_analyzer = MTFAnalyzerV2()
        self.regime_classifier = RegimeClassifier()
        self.feature_builder = TrendFeatureBuilder()
        self.ml_model = ml_model
    
    def detect_trend(self, dfs_dict):
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
        
        mtf_result_obj = self.mtf_analyzer.analyze(dfs_dict=dfs_dict, structural_scores=structural_scores)
        mtf_result = mtf_result_obj.to_dict()
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
        
        if self.ml_model is not None:
            # Build features for ML model
            features = self.feature_builder.build_features(
                dfs_dict['H1'],
                primary_structure,
                mtf_result,
                regime_info
            )
            
            # Get ML prediction
            # Assuming your Fusion model outputs probabilities: [P(bear), P(flat), P(bull)]
            try:
                feature_vector = self._prepare_ml_input(features)
                ml_probs = self.ml_model.predict_proba([feature_vector])[0]
                
                ml_confidence = max(ml_probs)
                if ml_confidence >= 0.60:
                    ml_direction = np.argmax(ml_probs) - 1  # -1, 0, 1
                else:
                    ml_direction = 0  # Low confidence -> sideways
            except Exception as e:
                print(f"ML prediction failed: {e}")
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
                'final_score': final_score
            }
        }
    
    def _classify_trend(self, trend_strength, struct_direction, regime):
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
    
    def _calculate_confidence(self, struct_score, mtf_score, ml_conf, regime_score):
        """Calculate overall confidence in trend classification"""
        # Weighted average of all confidence components
        confidence = (
            0.35 * struct_score +
            0.35 * mtf_score +
            0.20 * ml_conf +
            0.10 * regime_score
        )
        return confidence
    
    def _prepare_ml_input(self, features):
        """
        Convert feature dict to format expected by ML model
        Adjust this based on your actual model's input format
        """
        # Return as list in consistent order
        feature_order = [
            'struct_score', 'mtf_score', 'regime', 'adx',
            'plus_di', 'minus_di', 'price_above_ema20',
            'price_above_ema50', 'price_above_ema200',
            'ema_alignment', 'vol_compression', 'roc_5', 'roc_10'
        ]
        
        return [features.get(k, 0) for k in feature_order]