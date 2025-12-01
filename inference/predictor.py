# inference/predictor.py
"""
Hybrid predictor combining LSTM, ViT, and YOLO models.
"""
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.fusion import FusionNet
from models.yolo_detector import YOLOPatternDetector
from utils.candle_to_image import candle_image, normalize_for_model
from utils.config import settings

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Structured prediction output."""
    probabilities: np.ndarray  # [P(BUY), P(SELL), P(HOLD)]
    predicted_class: int       # 0=BUY, 1=SELL, 2=HOLD
    confidence: float          # Max probability
    gate_weights: Optional[np.ndarray] = None  # Modality weights if available


@dataclass
class ModelDimensions:
    """Model dimension configuration."""
    lstm_dim: int = 64
    vit_dim: int = 768
    yolo_dim: int = 20
    num_classes: int = 3


class HybridPredictor:
    """
    Multi-modal predictor combining:
    - LSTM for time-series patterns
    - ViT for visual chart patterns
    - YOLO for candlestick pattern detection
    - Fusion network for combining modalities
    """
    
    CLASS_NAMES = ['BUY', 'SELL', 'HOLD']
    
    def __init__(
        self,
        weights_dir: Optional[str] = None,
        device: Optional[str] = None,
        dims: Optional[ModelDimensions] = None,
    ):
        self.weights_dir = Path(weights_dir or settings.WEIGHTS_DIR)
        self.device = torch.device(device or settings.DEVICE)
        self.dims = dims or ModelDimensions()
        self.seq_len = settings.SEQUENCE_LENGTH
        self.img_size = settings.IMAGE_SIZE
        
        self._models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Initialize and load all model components."""
        logger.info(f"Loading models from {self.weights_dir} on {self.device}")
        
        try:
            # Initialize architectures
            self.lstm = LSTMModel(
                input_dim=5,
                hidden_dim=self.dims.lstm_dim,
                num_classes=self.dims.num_classes,
            ).to(self.device)
            
            self.vit = ViTExtractor(
                model_name="vit_base_patch16_224",
                pretrained=True,
            ).to(self.device)
            
            self.fusion = FusionNet(
                lstm_dim=self.dims.lstm_dim,
                vit_dim=self.dims.vit_dim,
                yolo_dim=self.dims.yolo_dim,
                num_classes=self.dims.num_classes,
            ).to(self.device)
            
            # YOLO handles device internally
            self.yolo = YOLOPatternDetector(
                model_path=str(self.weights_dir / "yolo_best.pt"),
                num_classes=self.dims.yolo_dim,
            )
            
            # Load trained weights (if available)
            self._load_weights()
            
            # Set to eval mode
            self.lstm.eval()
            self.vit.eval()
            self.fusion.eval()
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_weights(self):
        """Load saved model weights."""
        weight_files = {
            'lstm': self.weights_dir / "lstm_best.pt",
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
                logger.warning(f"Weight file not found: {path} (using random/pretrained)")
    
    def predict(self, df_window: pd.DataFrame) -> PredictionResult:
        """
        Generate prediction from raw DataFrame.
        
        Args:
            df_window: DataFrame with at least `seq_len` rows of OHLCV data
        
        Returns:
            PredictionResult with probabilities and metadata
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded")
        
        if len(df_window) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows, got {len(df_window)}")
        
        # Use last seq_len rows
        df_input = df_window.tail(self.seq_len)
        
        # Prepare inputs
        lstm_input = self._prepare_lstm_input(df_input)
        vit_input = self._prepare_vit_input(df_input)
        
        # Run inference
        with torch.no_grad():
            # Extract features (not classifications!)
            lstm_feat = self.lstm(lstm_input, mode='features')
            vit_feat = self.vit(vit_input)
            
            # YOLO detection
            img_array = candle_image(df_input, target_size=self.img_size)
            yolo_vec = self.yolo.detect(img_array)
            yolo_feat = torch.tensor(yolo_vec).float().unsqueeze(0).to(self.device)
            
            # Fusion and classification
            logits, gate_weights = self.fusion.forward_with_gates(
                lstm_feat, vit_feat, yolo_feat
            )
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            gates = gate_weights.cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionResult(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            gate_weights=gates,
        )
    
    def _prepare_lstm_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare LSTM input tensor."""
        # Extract OHLCV
        cols = ['open', 'high', 'low', 'close', 'tick_volume']
        data = df[cols].values.astype(np.float32)
        
        # Simple normalization (z-score within window)
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data_norm = (data - mean) / std
        
        # To tensor: (1, seq_len, 5)
        tensor = torch.tensor(data_norm).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def _prepare_vit_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare ViT input tensor from candlestick image."""
        # Generate candlestick image
        img_array = candle_image(df, target_size=self.img_size)
        
        # Normalize for ViT
        img_norm = normalize_for_model(img_array, use_imagenet_stats=True)
        
        # To tensor: (1, 3, 224, 224)
        tensor = torch.tensor(img_norm).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_batch(
        self, 
        df_windows: list[pd.DataFrame]
    ) -> list[PredictionResult]:
        """
        Batch prediction for multiple windows.
        More efficient than calling predict() repeatedly.
        """
        # TODO: Implement proper batching for efficiency
        return [self.predict(df) for df in df_windows]
    
    def get_signal(
        self, 
        df_window: pd.DataFrame,
        threshold: float = 0.6,
    ) -> str:
        """
        Convenience method to get trading signal.
        
        Args:
            df_window: Input DataFrame
            threshold: Minimum confidence for BUY/SELL signal
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        result = self.predict(df_window)
        
        # Only trade if confident enough
        if result.confidence < threshold:
            return 'HOLD'
        
        return self.CLASS_NAMES[result.predicted_class]


class SimpleLSTMPredictor:
    """
    Simplified predictor using only LSTM.
    Useful for testing or when visual features aren't needed.
    """
    
    CLASS_NAMES = ['BUY', 'SELL', 'HOLD']
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or settings.DEVICE)
        self.seq_len = settings.SEQUENCE_LENGTH
        
        self.model = LSTMModel().to(self.device)
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
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
        )
