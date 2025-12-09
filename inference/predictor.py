"""
Hybrid Predictor for pyForex Trading System.

Combines multiple models for robust trading predictions:
- TCN (Temporal Convolutional Network) for time-series analysis [UPDATED from LSTM]
- ViT (Vision Transformer) for visual chart patterns
- YOLO for candlestick pattern detection
- Fusion network for combining modalities

Author: pyForex Team
Updated: TCN replaces LSTM for better parallelization and gradient stability
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple, List
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class Signal(IntEnum):
    """Trading signal enumeration."""
    BUY = 0
    SELL = 1
    HOLD = 2


class PredictionResult(NamedTuple):
    """Structured prediction output."""
    probabilities: np.ndarray  # [P(BUY), P(SELL), P(HOLD)]
    predicted_class: int       # Signal enum value
    confidence: float          # Max probability
    signal_name: str          # 'BUY', 'SELL', 'HOLD'
    gate_weights: Optional[np.ndarray] = None  # Modality weights [seq, vit, yolo]


@dataclass
class ModelDimensions:
    """Model dimension configuration."""
    seq_dim: int = 64        # TCN output dimension
    vit_dim: int = 768       # ViT feature dimension
    yolo_dim: int = 20       # YOLO pattern vector size
    num_classes: int = 3     # BUY, SELL, HOLD


@dataclass 
class PredictorConfig:
    """Configuration for HybridPredictor."""
    weights_dir: str = "./weights"
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    sequence_length: int = 60
    image_size: int = 224
    
    # Model selection - UPDATED: TCN is now default
    seq_model_type: str = "tcn"  # 'tcn' (recommended) or 'lstm' (legacy)
    fusion_type: str = "gated"   # 'gated', 'simple', 'attention'
    
    # TCN-specific
    tcn_profile: str = "INTRADAY"  # 'SCALP', 'INTRADAY', 'SWING'
    
    # Inference settings
    confidence_threshold: float = 0.6
    use_ensemble: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =============================================================================
# Main Predictor Class
# =============================================================================

class HybridPredictor:
    """
    Multi-modal predictor combining:
    - TCN for time-series patterns (replaces LSTM)
    - ViT for visual chart patterns
    - YOLO for candlestick pattern detection
    - Fusion network for combining modalities
    
    Example:
        predictor = HybridPredictor(weights_dir="./weights")
        result = predictor.predict(df_window)
        print(f"Signal: {result.signal_name}, Confidence: {result.confidence:.2%}")
    """
    
    SIGNAL_NAMES = ['BUY', 'SELL', 'HOLD']
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        weights_dir: Optional[str] = None,
        device: Optional[str] = None,
        dims: Optional[ModelDimensions] = None,
    ):
        """
        Initialize the hybrid predictor.
        
        Args:
            config: Full configuration object
            weights_dir: Path to model weights (overrides config)
            device: Device string (overrides config)
            dims: Model dimensions (overrides config defaults)
        """
        # Handle config
        self.config = config or PredictorConfig()
        if weights_dir:
            self.config.weights_dir = weights_dir
        if device:
            self.config.device = device
        
        self.weights_dir = Path(self.config.weights_dir)
        self.device = get_device(self.config.device)
        self.dims = dims or ModelDimensions()
        
        self._models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Initialize and load all model components."""
        logger.info(f"Loading models from {self.weights_dir} on {self.device}")
        
        try:
            self._load_sequence_model()
            self._load_vision_model()
            self._load_yolo_model()
            self._load_fusion_model()
            
            # Load trained weights
            self._load_weights()
            
            # Set to evaluation mode
            self.seq_model.eval()
            self.vit.eval()
            self.fusion.eval()
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_sequence_model(self):
        """Load TCN (preferred) or LSTM (legacy) sequence model."""
        # UPDATED: TCN is now the default and preferred model
        if self.config.seq_model_type == "tcn":
            from models.tcn import TCNModel
            self.seq_model = TCNModel.from_profile(
                self.config.tcn_profile
            ).to(self.device)
            logger.info(f"Loaded TCN model (profile: {self.config.tcn_profile})")
        elif self.config.seq_model_type == "lstm":
            # Legacy LSTM support
            logger.warning("LSTM is deprecated. Consider switching to TCN for better performance.")
            from models.lstm import LSTMModel
            self.seq_model = LSTMModel(
                input_dim=5,
                hidden_dim=self.dims.seq_dim,
                num_classes=self.dims.num_classes,
            ).to(self.device)
            logger.info("Loaded LSTM model (legacy)")
        else:
            # Default to TCN
            from models.tcn import TCNModel
            self.seq_model = TCNModel.from_profile("INTRADAY").to(self.device)
            logger.info("Loaded TCN model (default)")
    
    def _load_vision_model(self):
        """Load Vision Transformer model."""
        try:
            from models.vit import ViTExtractor
            self.vit = ViTExtractor(
                model_name="vit_base_patch16_224",
                pretrained=True,
            ).to(self.device)
        except ImportError:
            # Fallback to vit_extractor
            from models.vit_extractor import ViTExtractor
            self.vit = ViTExtractor().to(self.device)
    
    def _load_yolo_model(self):
        """Load YOLO pattern detector."""
        from models.yolo_detector import YOLOPatternDetector
        yolo_path = self.weights_dir / "yolo_best.pt"
        self.yolo = YOLOPatternDetector(
            model_path=str(yolo_path) if yolo_path.exists() else None,
            num_classes=self.dims.yolo_dim,
        )
    
    def _load_fusion_model(self):
        """Load fusion network."""
        from models.fusion import create_fusion_model
        self.fusion = create_fusion_model(
            fusion_type=self.config.fusion_type,
            seq_dim=self.dims.seq_dim,
            vit_dim=self.dims.vit_dim,
            yolo_dim=self.dims.yolo_dim,
            num_classes=self.dims.num_classes,
        ).to(self.device)
    
    def _load_weights(self):
        """Load saved model weights."""
        # UPDATED: TCN weights path
        if self.config.seq_model_type == "tcn":
            seq_weight_name = "tcn_best.pt"
        else:
            seq_weight_name = "lstm_best.pt"
        
        weight_files = {
            'seq_model': self.weights_dir / seq_weight_name,
            'vit': self.weights_dir / "vit_best.pt",
            'fusion': self.weights_dir / "fusion_best.pt",
        }
        
        for name, path in weight_files.items():
            if path.exists():
                model = getattr(self, name)
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights: {path}")
            else:
                logger.warning(f"Weight file not found: {path}")
    
    def predict(self, df_window: pd.DataFrame) -> PredictionResult:
        """
        Generate prediction from raw DataFrame.
        
        Args:
            df_window: DataFrame with at least seq_len rows of OHLCV data
                      Required columns: open, high, low, close, tick_volume
        
        Returns:
            PredictionResult with probabilities, class, confidence, and signal name
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        seq_len = self.config.sequence_length
        if len(df_window) < seq_len:
            raise ValueError(f"Need at least {seq_len} rows, got {len(df_window)}")
        
        # Use last seq_len rows
        df_input = df_window.tail(seq_len)
        
        # Prepare inputs
        seq_input = self._prepare_seq_input(df_input)
        vit_input = self._prepare_vit_input(df_input)
        
        # Run inference
        with torch.no_grad():
            # Extract features
            seq_feat = self.seq_model(seq_input, mode='features')
            vit_feat = self.vit(vit_input)
            
            # YOLO detection
            yolo_vec = self._get_yolo_features(df_input)
            yolo_feat = torch.tensor(yolo_vec).float().unsqueeze(0).to(self.device)
            
            # Fusion and classification
            logits, gate_weights = self.fusion.forward_with_gates(
                seq_feat, vit_feat, yolo_feat
            )
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            gates = gate_weights.cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionResult(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_name=self.SIGNAL_NAMES[predicted_class],
            gate_weights=gates,
        )
    
    def _prepare_seq_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare sequence model input tensor."""
        cols = ['open', 'high', 'low', 'close', 'tick_volume']
        data = df[cols].values.astype(np.float32)
        
        # Z-score normalization within window
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_norm = (data - mean) / std
        
        # Shape: (1, seq_len, features)
        tensor = torch.tensor(data_norm).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def _prepare_vit_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare ViT input tensor from candlestick image."""
        try:
            from utils.candle_to_image import candle_image, normalize_for_model
            
            img_array = candle_image(df, target_size=self.config.image_size)
            img_norm = normalize_for_model(img_array, use_imagenet_stats=True)
            
            # Shape: (1, 3, H, W)
            tensor = torch.tensor(img_norm).float().unsqueeze(0)
            return tensor.to(self.device)
        except ImportError:
            # Fallback: return dummy tensor
            logger.warning("candle_to_image not available, using dummy ViT input")
            return torch.zeros(1, 3, 224, 224).to(self.device)
    
    def _get_yolo_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get YOLO pattern detection features."""
        try:
            from utils.candle_to_image import candle_image
            
            img_array = candle_image(df, target_size=self.config.image_size)
            return self.yolo.detect(img_array)
        except ImportError:
            # Fallback: return zeros
            return np.zeros(self.dims.yolo_dim, dtype=np.float32)
    
    def predict_batch(
        self, 
        df_windows: List[pd.DataFrame]
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple windows.
        
        Args:
            df_windows: List of DataFrames to predict
        
        Returns:
            List of PredictionResults
        """
        # TODO: Implement proper batching for efficiency
        return [self.predict(df) for df in df_windows]
    
    def get_signal(
        self, 
        df_window: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Get trading signal with confidence threshold.
        
        Args:
            df_window: Input DataFrame
            threshold: Confidence threshold (default from config)
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        threshold = threshold or self.config.confidence_threshold
        result = self.predict(df_window)
        
        # Only trade if confident enough
        if result.confidence < threshold:
            return 'HOLD'
        
        return result.signal_name
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'sequence_model': self.config.seq_model_type,
            'tcn_profile': self.config.tcn_profile,
            'fusion_type': self.config.fusion_type,
            'device': str(self.device),
            'weights_dir': str(self.weights_dir),
            'models_loaded': self._models_loaded,
            'dimensions': {
                'seq_dim': self.dims.seq_dim,
                'vit_dim': self.dims.vit_dim,
                'yolo_dim': self.dims.yolo_dim,
            }
        }


# =============================================================================
# TCN-only Predictor (Replaces SimpleLSTMPredictor)
# =============================================================================

class TCNPredictor:
    """
    Simplified predictor using only TCN.
    
    Useful for:
    - Testing and debugging
    - When visual features aren't available
    - Lower latency requirements
    
    NOTE: This replaces the old SimpleLSTMPredictor
    """
    
    SIGNAL_NAMES = ['BUY', 'SELL', 'HOLD']
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        profile: str = "INTRADAY",
        device: Optional[str] = None,
        sequence_length: int = 60,
    ):
        self.device = get_device(device or "auto")
        self.seq_len = sequence_length
        self.profile = profile
        
        from models.tcn import TCNModel
        self.model = TCNModel.from_profile(profile).to(self.device)
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded TCN weights from {weights_path}")
        
        self.model.eval()
    
    def predict(self, df_window: pd.DataFrame) -> PredictionResult:
        """Generate prediction from DataFrame."""
        df_input = df_window.tail(self.seq_len)
        
        # Prepare input
        cols = ['open', 'high', 'low', 'close', 'tick_volume']
        data = df_input[cols].values.astype(np.float32)
        
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_norm = (data - mean) / std
        
        tensor = torch.tensor(data_norm).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor, mode='classify')
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionResult(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_name=self.SIGNAL_NAMES[predicted_class],
            gate_weights=None,
        )
    
    def get_signal(
        self, 
        df_window: pd.DataFrame,
        threshold: float = 0.6,
    ) -> str:
        """Get trading signal with threshold."""
        result = self.predict(df_window)
        if result.confidence < threshold:
            return 'HOLD'
        return result.signal_name


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# DEPRECATED: Use TCNPredictor instead
SimpleLSTMPredictor = TCNPredictor  # Alias for backward compatibility


# =============================================================================
# Factory Functions
# =============================================================================

def create_predictor(
    predictor_type: str = "hybrid",
    **kwargs
) -> HybridPredictor:
    """
    Factory function for creating predictors.
    
    Args:
        predictor_type: 'hybrid', 'tcn', or 'simple' (legacy alias)
        **kwargs: Arguments passed to predictor constructor
    """
    if predictor_type in ("tcn", "simple"):
        return TCNPredictor(**kwargs)
    elif predictor_type == "hybrid":
        return HybridPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")