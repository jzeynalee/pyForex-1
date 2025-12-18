# models/yolo_pattern.py
"""
YOLO pattern extractor module.
Provides YOLOPatternExtractor class for candlestick pattern detection.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. YOLO extractor will use fallback.")


# Standard candlestick pattern classes
PATTERN_CLASSES = [
    "doji",
    "hammer", 
    "inverted_hammer",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "bullish_harami",
    "bearish_harami",
    "shooting_star",
    "hanging_man",
    "piercing_line",
    "dark_cloud_cover",
    "tweezer_top",
    "tweezer_bottom",
    "spinning_top",
    "marubozu_bull",
    "marubozu_bear",
]


class YOLOPatternExtractor:
    """
    Extracts candlestick patterns from chart images using YOLO.
    
    This is the class that the predictor expects to import from models.yolo_pattern.
    """
    
    def __init__(
        self,
        model_path: str = "models/weights/yolo_patterns.pt",
        num_classes: int = 20,
        confidence_threshold: float = 0.5,
        include_confidence: bool = False,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.include_confidence = include_confidence
        
        # Feature dimension: binary presence + optional confidence
        self.feature_dim = num_classes * 2 if include_confidence else num_classes
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.enabled = True
                logger.info(f"YOLO model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Using fallback.")
                self.model = None
                self.enabled = False
        else:
            self.model = None
            self.enabled = False
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract patterns from image.
        
        Args:
            img: Image array in HWC format (height, width, channels)
                 Should be RGB uint8 [0, 255]
        
        Returns:
            feature_vector: numpy array of shape (feature_dim,)
        """
        if not self.enabled or self.model is None:
            return self._fallback_features()
        
        try:
            # Run inference
            results = self.model(img, verbose=False)[0]
            
            # Build feature vector
            if self.include_confidence:
                vector = np.zeros(self.num_classes * 2, dtype=np.float32)
                
                for box in results.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    
                    if cls_id < self.num_classes and conf >= self.confidence_threshold:
                        # Binary presence
                        vector[cls_id] = 1.0
                        # Max confidence for this class
                        vector[self.num_classes + cls_id] = max(
                            vector[self.num_classes + cls_id], 
                            conf
                        )
            else:
                vector = np.zeros(self.num_classes, dtype=np.float32)
                
                for box in results.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    
                    if cls_id < self.num_classes and conf >= self.confidence_threshold:
                        vector[cls_id] = 1.0
            
            return vector
            
        except Exception as e:
            logger.error(f"YOLO extraction error: {e}")
            return self._fallback_features()
    
    def _fallback_features(self) -> np.ndarray:
        """Returns zero vector when YOLO is unavailable."""
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def extract_with_details(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract patterns and return both feature vector and detection details.
        
        Returns:
            feature_vector: numpy array
            detections: list of dicts with {class_id, class_name, confidence, bbox}
        """
        feature_vector = self.extract(img)
        detections = []
        
        if self.enabled and self.model is not None:
            try:
                results = self.model(img, verbose=False)[0]
                
                for box in results.boxes:
                    cls_id = int(box.cls)
                    detections.append({
                        'class_id': cls_id,
                        'class_name': PATTERN_CLASSES[cls_id] if cls_id < len(PATTERN_CLASSES) else f"class_{cls_id}",
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    })
            except Exception as e:
                logger.error(f"Detection with details error: {e}")
        
        return feature_vector, detections
    
    def get_feature_dim(self) -> int:
        """Returns output feature dimension for fusion layer."""
        return self.feature_dim


class MockYOLOPatternExtractor:
    """
    Mock extractor for testing without trained YOLO model.
    Generates random but consistent features based on image hash.
    """
    
    def __init__(self, num_classes: int = 20):
        self.num_classes = num_classes
        self.feature_dim = num_classes
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        # Use image mean as seed for reproducibility
        seed = int(img.mean() * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Sparse binary vector (most patterns not present)
        vector = np.zeros(self.num_classes, dtype=np.float32)
        num_detections = rng.randint(0, 3)  # 0-2 patterns
        
        if num_detections > 0:
            indices = rng.choice(self.num_classes, num_detections, replace=False)
            vector[indices] = 1.0
        
        return vector
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
