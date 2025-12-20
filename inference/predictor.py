# inference/predictor.py
"""
Enhanced Hybrid Predictor with Risk Management Integration

Combines:
- TCN for time-series predictions (direction, volatility, quantiles)
- ViT for visual chart patterns
- YOLO for candlestick pattern detection
- Fusion network for combining modalities
- Risk management predictions (Phase 1 multi-head outputs)

This replaces the previous predictor.py with full risk management support.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple, List, Union
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class Signal(IntEnum):
    """Trading signal enumeration (matches Phase 1)."""
    BEAR = 0      # Sell
    SIDEWAYS = 1  # Hold
    BULL = 2      # Buy


class PredictionResult(NamedTuple):
    """Structured prediction output with risk parameters."""
    # Direction
    probabilities: np.ndarray      # [P(BEAR), P(SIDEWAYS), P(BULL)]
    predicted_class: int           # Signal enum value
    confidence: float              # Max probability
    signal_name: str               # 'BEAR', 'SIDEWAYS', 'BULL'
    
    # Risk parameters (from Phase 1 heads)
    volatility: float              # Predicted σ
    quantiles: np.ndarray          # [Q5, Q25, Q50, Q75, Q95]
    
    # Modality info
    gate_weights: Optional[np.ndarray] = None  # [seq, vit, yolo]
    features: Optional[np.ndarray] = None      # Hidden features


@dataclass
class PredictorConfig:
    """Configuration for predictors."""
    weights_dir: str = "models/weights"
    device: str = "auto"
    sequence_length: int = 60

    # Model selection
    model_type: str = "tcn"         # 'tcn' (Temporal Convolutional Network)
    profile: str = "INTRADAY"       # 'SCALP', 'INTRADAY', 'SWING'
    use_risk_heads: bool = True     # Enable volatility/quantile heads
    
    # Vision
    use_vision: bool = True
    image_size: int = 224
    
    # YOLO
    use_yolo: bool = True
    
    # Fusion
    fusion_type: str = "gated"      # 'gated', 'simple', 'attention'
    
    # Inference
    confidence_threshold: float = 0.55


def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class RiskAwareTCNPredictor:
    """
    TCN predictor with multi-head risk outputs.
    
    Outputs:
    - Direction probabilities: P(Bear), P(Sideways), P(Bull)
    - Volatility prediction: σ (standard deviation)
    - Quantile predictions: [Q5, Q25, Q50, Q75, Q95] for price distribution
    
    These outputs feed directly into Phase 2 risk calculations.
    """
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        weights_path: Optional[str] = None
    ):
        self.config = config or PredictorConfig()
        self.device = get_device(self.config.device)
        
        # Initialize model
        self._init_model()
        
        # Load weights if provided
        if weights_path:
            self.load_weights(weights_path)
        
        # Feature scaler (loaded from checkpoint)
        self._scaler = None
        self._feature_names = None
        
        logger.info(f"RiskAwareTCNPredictor initialized on {self.device}")
    
    def _init_model(self):
        """Initialize the multi-head TCN model."""
        try:
            # Try to import from risk_management (new)
            from risk_management import create_tcn_for_profile, MultiHeadTCN
            
            self.model = create_tcn_for_profile(
                profile=self.config.profile,
                input_features=64  # Will be updated when loading weights
            ).to(self.device)
            
            self._use_risk_heads = True
            logger.info("Using MultiHeadTCN with risk outputs")
            
        except ImportError:
            # Fallback to standard TCN
            try:
                from models.tcn import TCNModel
                self.model = TCNModel.from_profile(self.config.profile).to(self.device)
                self._use_risk_heads = False
                logger.info("Using standard TCN (no risk heads)")
            except ImportError:
                raise ImportError("No TCN model available. Install risk_management or models.tcn")
    
    def load_weights(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Load feature info if available
        if 'feature_names' in checkpoint:
            self._feature_names = checkpoint['feature_names']
        if 'scaler_params' in checkpoint:
            self._scaler = checkpoint['scaler_params']
        if 'config' in checkpoint:
            # Update config from checkpoint
            ckpt_config = checkpoint['config']
            if hasattr(ckpt_config, 'input_channels'):
                self.config.input_features = ckpt_config.input_channels
        
        self.model.eval()
        logger.info(f"Loaded weights from {path}")
    
    def predict(
        self,
        features: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        return_features: bool = False
    ) -> PredictionResult:
        """
        Make prediction with risk parameters.
        """
        import time
        import hashlib
        
        start_time = time.time()
        
        # Prepare input
        x = self._prepare_input(features)
        
        # Calculate input snapshot hash for integrity
        input_hash = hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()[:8]
        
        self.model.eval()
        with torch.no_grad():
            if self._use_risk_heads:
                # Multi-head TCN outputs
                outputs = self.model(x, mode='all')
                
                direction_probs = outputs['direction'].cpu().numpy()[0]
                volatility = outputs['volatility'].cpu().numpy().item()
                quantiles = outputs['quantiles'].cpu().numpy()[0]
                hidden_features = outputs.get('features')
                if hidden_features is not None:
                    hidden_features = hidden_features.cpu().numpy()[0]
            else:
                # Standard TCN (direction only)
                logits = self.model(x)
                direction_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                volatility = 0.0
                quantiles = np.zeros(5)
                hidden_features = None
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract prediction info
        predicted_class = int(np.argmax(direction_probs))
        confidence = float(np.max(direction_probs))
        signal_name = ['BEAR', 'SIDEWAYS', 'BULL'][predicted_class]
        
        # Log model execution integrity
        logger.info(
            f"[TCN] Hash:{input_hash} | Latency:{latency_ms:.2f}ms | "
            f"Out:{signal_name}({confidence:.2f}) | Vol:{volatility:.5f}"
        )
        
        return PredictionResult(
            probabilities=direction_probs,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_name=signal_name,
            volatility=volatility,
            quantiles=quantiles,
            gate_weights=None,
            features=hidden_features if return_features else None
        )
    
    def predict_batch(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Batch prediction for multiple samples.
        
        Args:
            features: (batch, seq_len, n_features)
        
        Returns:
            Dict with batched predictions
        """
        x = self._prepare_input(features)
        
        self.model.eval()
        with torch.no_grad():
            if self._use_risk_heads:
                outputs = self.model(x, mode='all')
                return {
                    'direction_probs': outputs['direction'].cpu().numpy(),
                    'volatility': outputs['volatility'].cpu().numpy(),
                    'quantiles': outputs['quantiles'].cpu().numpy(),
                    'features': outputs.get('features', torch.zeros(1)).cpu().numpy()
                }
            else:
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                return {
                    'direction_probs': probs,
                    'volatility': np.zeros(len(probs)),
                    'quantiles': np.zeros((len(probs), 5))
                }
    
    def _prepare_input(
        self,
        features: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> torch.Tensor:
        """Prepare input tensor from various formats."""
        # Convert DataFrame to numpy
        if isinstance(features, pd.DataFrame):
            # Select features if we know which ones to use
            if self._feature_names:
                available = [f for f in self._feature_names if f in features.columns]
                features = features[available].values
            else:
                # Use all numeric columns
                features = features.select_dtypes(include=[np.number]).values
        
        # Convert to numpy if tensor
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Ensure correct shape (batch, seq_len, features)
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # Add batch dimension
        
        # Apply scaling if available
        if self._scaler is not None:
            features = self._apply_scaling(features)
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        return x
    
    def _apply_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply saved scaling parameters."""
        if self._scaler is None:
            return features
        
        original_shape = features.shape
        flat = features.reshape(-1, features.shape[-1])
        
        if 'mean' in self._scaler and 'std' in self._scaler:
            flat = (flat - self._scaler['mean']) / (self._scaler['std'] + 1e-8)
        elif 'min' in self._scaler and 'max' in self._scaler:
            flat = (flat - self._scaler['min']) / (self._scaler['max'] - self._scaler['min'] + 1e-8)
        
        return flat.reshape(original_shape)
    
    def to_dict(self, result: PredictionResult) -> Dict:
        """Convert prediction result to dictionary format for decision engine."""
        return {
            'direction_probs': result.probabilities,
            'volatility': result.volatility,
            'quantiles': result.quantiles,
            'features': result.features
        }


class HybridPredictor:
    """
    Full hybrid predictor combining TCN + ViT + YOLO + Fusion.
    
    This is the production predictor that combines all modalities
    with risk management outputs.
    """
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        tcn_weights: Optional[str] = None,
        vit_weights: Optional[str] = None,
        yolo_weights: Optional[str] = None,
        fusion_weights: Optional[str] = None
    ):
        self.config = config or PredictorConfig()
        self.device = get_device(self.config.device)
        
        # Initialize components
        self.tcn_predictor = RiskAwareTCNPredictor(config)
        if tcn_weights:
            self.tcn_predictor.load_weights(tcn_weights)
        
        # Vision components (optional)
        self.vit_model = None
        self.yolo_model = None
        self.fusion_model = None
        
        if self.config.use_vision:
            self._init_vision(vit_weights)
        
        if self.config.use_yolo:
            self._init_yolo(yolo_weights)
        
        if self.vit_model or self.yolo_model:
            self._init_fusion(fusion_weights)
        
        logger.info("HybridPredictor initialized")
    
    def _init_vision(self, weights_path: Optional[str]):
        """Initialize ViT model."""
        try:
            from models.vit import ViTChartClassifier
            self.vit_model = ViTChartClassifier().to(self.device)
            if weights_path:
                self.vit_model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
            self.vit_model.eval()
            logger.info("ViT model loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load ViT: {e}")
            self.vit_model = None
    
    def _init_yolo(self, weights_path: Optional[str]):
        """Initialize YOLO pattern detector."""
        try:
            from models.yolo_pattern import YOLOPatternExtractor
            self.yolo_model = YOLOPatternExtractor(weights_path)
            logger.info("YOLO model loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load YOLO: {e}")
            self.yolo_model = None
    
    def _init_fusion(self, weights_path: Optional[str]):
        """Initialize fusion network."""
        try:
            from models.fusion import FusionNet
            
            seq_dim = 64  # TCN output dimension
            vit_dim = 768 if self.vit_model else 0
            yolo_dim = 20 if self.yolo_model else 0
            
            self.fusion_model = FusionNet(
                seq_dim=seq_dim,
                vit_dim=vit_dim,
                yolo_dim=yolo_dim
            ).to(self.device)
            
            if weights_path:
                self.fusion_model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
            self.fusion_model.eval()
            logger.info("Fusion model loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load fusion model: {e}")
            self.fusion_model = None
    
    def predict(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        chart_image: Optional[np.ndarray] = None,
        return_all: bool = False
    ) -> PredictionResult:
        """
        Make hybrid prediction with all modalities.
        
        Args:
            features: Time-series features for TCN
            chart_image: Optional chart image for ViT/YOLO
            return_all: Return all intermediate outputs
        
        Returns:
            PredictionResult with fused predictions
        """
        # Get TCN prediction with risk parameters
        tcn_result = self.tcn_predictor.predict(features, return_features=True)
        
        # If no vision or no image, return TCN result
        if not self.config.use_vision or chart_image is None:
            return tcn_result
        
        # Get vision features
        vit_features = None
        yolo_features = None
        
        if self.vit_model and chart_image is not None:
            vit_features = self._get_vit_features(chart_image)
        
        if self.yolo_model and chart_image is not None:
            yolo_features = self._get_yolo_features(chart_image)
        
        # Fuse if fusion model available
        if self.fusion_model and tcn_result.features is not None:
            fused_probs, gate_weights = self._fuse_predictions(
                tcn_result.features,
                vit_features,
                yolo_features
            )
            
            predicted_class = int(np.argmax(fused_probs))
            confidence = float(np.max(fused_probs))
            signal_name = ['BEAR', 'SIDEWAYS', 'BULL'][predicted_class]
            
            return PredictionResult(
                probabilities=fused_probs,
                predicted_class=predicted_class,
                confidence=confidence,
                signal_name=signal_name,
                volatility=tcn_result.volatility,
                quantiles=tcn_result.quantiles,
                gate_weights=gate_weights,
                features=tcn_result.features
            )
        
        return tcn_result
    
    def _get_vit_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from ViT."""
        # Preprocess image
        if image.max() > 1:
            image = image / 255.0
        
        # Add batch and channel dimensions if needed
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=0)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        
        x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.vit_model(x, mode='features')
        
        return features.cpu().numpy()[0]
    
    def _get_yolo_features(self, image: np.ndarray) -> np.ndarray:
        """Extract pattern features from YOLO."""
        patterns = self.yolo_model.detect(image)
        return self.yolo_model.to_feature_vector(patterns)
    
    def _fuse_predictions(
        self,
        seq_features: np.ndarray,
        vit_features: Optional[np.ndarray],
        yolo_features: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse features from all modalities."""
        # Prepare inputs
        seq_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        vit_tensor = None
        if vit_features is not None:
            vit_tensor = torch.tensor(vit_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        yolo_tensor = None
        if yolo_features is not None:
            yolo_tensor = torch.tensor(yolo_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, gate_weights = self.fusion_model(
                seq_tensor, vit_tensor, yolo_tensor,
                return_gate_weights=True
            )
            probs = F.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()[0], gate_weights.cpu().numpy()[0]


# Alias for convenience
TCNPredictor = RiskAwareTCNPredictor


def create_predictor(
    profile: str = 'INTRADAY',
    weights_path: Optional[str] = None,
    use_vision: bool = False,
    use_yolo: bool = False
) -> Union[RiskAwareTCNPredictor, HybridPredictor]:
    """
    Factory function to create appropriate predictor.
    
    Args:
        profile: Trading profile ('SCALP', 'INTRADAY', 'SWING')
        weights_path: Path to model weights
        use_vision: Enable ViT
        use_yolo: Enable YOLO
    
    Returns:
        Configured predictor instance
    """
    config = PredictorConfig(
        profile=profile,
        use_vision=use_vision,
        use_yolo=use_yolo
    )
    
    if use_vision or use_yolo:
        return HybridPredictor(config, tcn_weights=weights_path)
    else:
        return RiskAwareTCNPredictor(config, weights_path=weights_path)
