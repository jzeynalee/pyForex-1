# inference/predictor.py
"""
Enhanced Hybrid Predictor with Risk Management Integration

Combines:
- TCN for time-series predictions (direction, volatility, quantiles, outcomes)
- ViT for visual chart patterns
- Price Action for rule-based pattern detection
- Fusion network for combining modalities
- Risk management predictions (Phase 1 multi-head outputs)

This replaces the previous predictor.py with full risk management support.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, NamedTuple, Union
from dataclasses import dataclass, field
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class Signal(IntEnum):
    """Trading signal enumeration (matches Phase 1)."""
    BEAR = 0      # Sell
    SIDEWAYS = 1  # Hold
    BULL = 2      # Buy


class PredictionResult(NamedTuple):
    """Structured prediction output with risk parameters.

    In addition to direction/volatility/quantiles, this result can optionally
    include TP-before-SL probabilities:
    - p_long:  P(TP hit before SL | enter long now)
    - p_short: P(TP hit before SL | enter short now)
    """
    # Direction
    probabilities: np.ndarray      # [P(BEAR), P(SIDEWAYS), P(BULL)]
    predicted_class: int           # Signal enum value
    confidence: float              # Max probability
    signal_name: str               # 'BEAR', 'SIDEWAYS', 'BULL'
    
    # Risk parameters (from Phase 1 heads)
    volatility: float              # Predicted σ
    quantiles: np.ndarray          # [Q5, Q25, Q50, Q75, Q95]

    # Trade-objective probabilities (optional)
    p_long: Optional[float] = None
    p_short: Optional[float] = None
    
    # Modality info
    gate_weights: Optional[np.ndarray] = None  # [seq, vit, price_action]
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
    
    # Vision - DISABLED by default (weak signal, adds latency)
    use_vision: bool = False
    image_size: int = 224
    
    # Price Action
    use_price_action: bool = True
    
    # Fusion
    fusion_type: str = "gated"      # 'gated', 'simple', 'attention'
    
    # Inference - RAISED thresholds for better trade quality
    confidence_threshold: float = 0.65  # Was 0.55


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

    Optional:
    - Outcome probabilities: [p_long, p_short] as TP-before-SL likelihood
    
    These outputs feed directly into Phase 2 risk calculations.
    """
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        weights_path: Optional[str] = None
    ):
        self.config = config or PredictorConfig()
        self.device = get_device(self.config.device)
        self.model = None
        self._use_risk_heads = bool(getattr(self.config, 'use_risk_heads', False))
        
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
            raise ImportError("MultiHeadTCN not available. Ensure risk_management.phase1_predictive.tcn_backbone is installed.")
    
    def load_weights(self, path: str):
        """Load model weights from checkpoint.

        This loader supports both MultiHeadTCN checkpoints (risk heads) and
        legacy/simple TCN checkpoints.

        If a legacy checkpoint is detected while the predictor is currently
        initialized with MultiHeadTCN, the predictor will switch to the simple
        TCN architecture and attempt to infer the required input dimensionality
        from the checkpoint metadata and/or the first convolution weight.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        state_dict = None
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif isinstance(checkpoint, dict) and not any(k.startswith('backbone') or k.startswith('direction') for k in checkpoint.keys()):
            # Checkpoint is a wrapper dict, not direct state_dict
            for key in ['model', 'net', 'network']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                logger.warning(f"Could not find model state in checkpoint. Keys: {list(checkpoint.keys())[:10]}")
                logger.warning("Skipping weight loading - model will use random initialization")
                self.model.eval()
                return
        else:
            state_dict = checkpoint
        
        # Detect checkpoint format and switch model if needed
        state_keys = list(state_dict.keys())
        is_simple_tcn = any(k.startswith('tcn.') for k in state_keys)
        is_multihead_tcn = any(k.startswith('backbone.') for k in state_keys)
        
        # Get config from checkpoint to reinitialize model with correct dimensions
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            ckpt_config = checkpoint['config']
            input_dim = ckpt_config.get('input_dim', ckpt_config.get('input_channels', ckpt_config.get('input_features', 64)))
            hidden_dim = ckpt_config.get('hidden_dim', ckpt_config.get('hidden_channels', 128))
            num_layers = ckpt_config.get('num_layers', 4)
            
            # Reinitialize MultiHeadTCN with correct dimensions from checkpoint
            if is_multihead_tcn or (not is_simple_tcn and self._use_risk_heads):
                try:
                    from risk_management.phase1_predictive.tcn_backbone import MultiHeadTCN
                    self.model = MultiHeadTCN(
                        input_channels=input_dim,
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        num_directions=3,
                        num_quantiles=5
                    ).to(self.device)
                    self._use_risk_heads = True
                    logger.info(f"Reinitialized MultiHeadTCN with input_dim={input_dim}, hidden_dim={hidden_dim}")
                except Exception as e:
                    logger.warning(f"Could not reinitialize MultiHeadTCN: {e}")
        
        if is_simple_tcn and self._use_risk_heads:
            # Legacy checkpoint from old TCN - cannot load, warn user
            logger.warning("Checkpoint is from legacy TCN format which is no longer supported.")
            logger.warning("Please retrain using walk-forward training: python main.py train walk-forward --data <path>")
            logger.warning("Model will use random initialization.")
        
        try:
            self.model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {path}")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            try:
                self.model.load_state_dict(state_dict, strict=False)
                logger.warning("Loaded weights with strict=False (some keys may be missing)")
            except Exception as e2:
                logger.error(f"Could not load weights even with strict=False: {e2}")
                logger.warning("Model will use random initialization")
        
        # Load feature info if available
        if 'feature_names' in checkpoint:
            self._feature_names = checkpoint['feature_names']
        elif 'feature_columns' in checkpoint:
            self._feature_names = checkpoint['feature_columns']
        if 'scaler_params' in checkpoint:
            self._scaler = checkpoint['scaler_params']
        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                input_features = ckpt_config.get('input_channels', ckpt_config.get('input_dim', ckpt_config.get('input_features')))
                if input_features is not None:
                    self.config.input_features = int(input_features)
            else:
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
                # Disable p_long/p_short as they are not currently trained
                p_long = None # float(outputs['p_long'].cpu().numpy().item()) if 'p_long' in outputs else None
                p_short = None # float(outputs['p_short'].cpu().numpy().item()) if 'p_short' in outputs else None
                hidden_features = outputs.get('features')
                if hidden_features is not None:
                    hidden_features = hidden_features.cpu().numpy()[0]
            else:
                # Standard TCN (direction only)
                try:
                    logits = self.model(x, mode='classify')
                except TypeError:
                    # Fallback for models that don't implement a 'mode' argument.
                    logits = self.model(x)
                # Handle different output shapes
                if logits.dim() == 3:
                    # Shape: (batch, seq_len, num_classes) - take last timestep
                    logits = logits[:, -1, :]
                elif logits.dim() == 1:
                    # Shape: (num_classes,) - add batch dim
                    logits = logits.unsqueeze(0)
                direction_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                # Ensure we have exactly 3 classes
                if len(direction_probs) != 3:
                    # Pad or truncate to 3 classes
                    if len(direction_probs) < 3:
                        direction_probs = np.pad(direction_probs, (0, 3 - len(direction_probs)))
                    else:
                        direction_probs = direction_probs[:3]
                volatility = 0.0
                quantiles = np.zeros(5)
                p_long = None
                p_short = None
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
            p_long=p_long,
            p_short=p_short,
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
                    'p_long': outputs.get('p_long', torch.zeros(outputs['direction'].shape[0], device=outputs['direction'].device)).cpu().numpy(),
                    'p_short': outputs.get('p_short', torch.zeros(outputs['direction'].shape[0], device=outputs['direction'].device)).cpu().numpy(),
                    'features': outputs.get('features', torch.zeros(1)).cpu().numpy()
                }
            else:
                try:
                    logits = self.model(x, mode='classify')
                except TypeError:
                    logits = self.model(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                return {
                    'direction_probs': probs,
                    'volatility': np.zeros(len(probs)),
                    'quantiles': np.zeros((len(probs), 5)),
                    'p_long': np.zeros(len(probs)),
                    'p_short': np.zeros(len(probs))
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

        # Align feature dimension to the currently loaded model.
        expected_dim = None
        try:
            if getattr(self, '_use_risk_heads', False) and hasattr(self.model, 'config'):
                expected_dim = int(getattr(self.model.config, 'input_channels', 0) or 0)
            elif hasattr(self.model, 'tcn') and hasattr(self.model.tcn, 'input_dim'):
                expected_dim = int(getattr(self.model.tcn, 'input_dim'))
        except Exception:
            expected_dim = None

        if expected_dim is not None and expected_dim > 0:
            current_dim = int(features.shape[-1])
            if current_dim > expected_dim:
                features = features[..., :expected_dim]
            elif current_dim < expected_dim:
                pad = np.zeros((features.shape[0], features.shape[1], expected_dim - current_dim), dtype=features.dtype)
                features = np.concatenate([features, pad], axis=-1)
        
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
        elif 'center' in self._scaler and 'scale' in self._scaler:
            # RobustScaler support
            center = np.array(self._scaler['center'])
            scale = np.array(self._scaler['scale'])
            # Handle mismatch in feature dimensions if necessary
            if flat.shape[1] == len(center):
                flat = (flat - center) / (scale + 1e-8)
            else:
                # Try to scale matching columns or broadcast if possible
                # For safety, if dims don't match, we might skip or trim
                common_dim = min(flat.shape[1], len(center))
                flat[:, :common_dim] = (flat[:, :common_dim] - center[:common_dim]) / (scale[:common_dim] + 1e-8)
        
        return flat.reshape(original_shape)
    
    def to_dict(self, result: PredictionResult) -> Dict:
        """Convert prediction result to dictionary format for decision engine."""
        out = {
            'direction_probs': result.probabilities,
            'volatility': result.volatility,
            'quantiles': result.quantiles,
            'features': result.features
        }

        if result.p_long is not None:
            out['p_long'] = result.p_long
        if result.p_short is not None:
            out['p_short'] = result.p_short

        return out


class HybridPredictor:
    """
    Full hybrid predictor combining TCN + ViT + Price Action + Fusion.
    
    This is the production predictor that combines all modalities
    with risk management outputs.
    """
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        tcn_weights: Optional[str] = None,
        vit_weights: Optional[str] = None,
        price_action_weights: Optional[str] = None,
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
        self.price_action_model = None
        self.fusion_model = None
        
        if self.config.use_vision:
            self._init_vision(vit_weights)
        
        if self.config.use_price_action:
            self._init_price_action(price_action_weights)

        if (self.vit_model or self.price_action_model) and fusion_weights and Path(fusion_weights).exists():
            self._init_fusion(fusion_weights)
        
        logger.info("HybridPredictor initialized")
    
    def _init_vision(self, weights_path: Optional[str]):
        """Initialize ViT model - DISABLED in v2.0."""
        # ViT has been removed in v2.0 due to weak performance
        logger.info("ViT model disabled in v2.0 - using MH-TCN + Price Action only")
        self.vit_model = None
    
    def _init_price_action(self, weights_path: Optional[str]):
        """Initialize Price Action pattern detector."""
        try:
            from models.price_action_pattern import PriceActionPatternExtractor
            # Price action doesn't need weights, but we pass the parameter for compatibility
            self.price_action_model = PriceActionPatternExtractor(
                include_extended_patterns=True,
                include_confidence=False
            )
            logger.info("Price Action model loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load Price Action: {e}")
            self.price_action_model = None
    
    def _init_fusion(self, weights_path: Optional[str]):
        """Initialize fusion network."""
        try:
            from models.fusion import FusionNet
            
            seq_dim = 64  # TCN output dimension
            vit_dim = 0
            if self.vit_model:
                try:
                    vit_dim = int(getattr(self.vit_model, 'feature_dim', 0) or 0)
                except Exception:
                    vit_dim = 0
                if vit_dim <= 0:
                    try:
                        vit_backbone = getattr(self.vit_model, 'vit', None)
                        vit_dim = int(getattr(vit_backbone, 'num_features', 0) or 0)
                    except Exception:
                        vit_dim = 0
                if vit_dim <= 0:
                    vit_dim = 768
            price_action_dim = 0
            if self.price_action_model:
                try:
                    if hasattr(self.price_action_model, 'get_feature_dim'):
                        price_action_dim = int(self.price_action_model.get_feature_dim() or 0)
                    else:
                        price_action_dim = int(getattr(self.price_action_model, 'feature_dim', 0) or 0)
                except Exception:
                    price_action_dim = 0
                if price_action_dim <= 0:
                    price_action_dim = 50  # Default for extended patterns
            
            self.fusion_model = FusionNet(
                seq_dim=seq_dim,
                vit_dim=vit_dim,
                yolo_dim=price_action_dim  # Reuse yolo_dim parameter for price action
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
        price_action_features = None
        
        if self.vit_model and chart_image is not None:
            vit_features = self._get_vit_features(chart_image)
        
        if self.price_action_model and chart_image is not None:
            price_action_features = self._get_price_action_features(chart_image)
        
        # Fuse if fusion model available
        if self.fusion_model and tcn_result.features is not None:
            fused_probs, gate_weights = self._fuse_predictions(
                tcn_result.features,
                vit_features,
                price_action_features
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
            # Prefer timm-style forward_features() to get embeddings.
            try:
                vit_backbone = getattr(self.vit_model, 'vit', None)
                if vit_backbone is not None and hasattr(vit_backbone, 'forward_features'):
                    feats = vit_backbone.forward_features(x)
                elif hasattr(self.vit_model, 'forward_features'):
                    feats = self.vit_model.forward_features(x)
                else:
                    feats = None
            except Exception:
                feats = None

            if feats is None:
                # Fallback: use the model forward output (may be logits; still better than crashing).
                feats = self.vit_model(x)

            # Normalize common timm output shapes into [B, D].
            if isinstance(feats, dict):
                if 'cls_token' in feats:
                    feats = feats['cls_token']
                elif 'x' in feats:
                    feats = feats['x']
            if hasattr(feats, 'dim') and feats.dim() == 3:
                feats = feats[:, 0, :]

        return feats.cpu().numpy()[0]
    
    def _get_price_action_features(self, image: np.ndarray) -> np.ndarray:
        """Extract pattern features from Price Action analysis."""
        try:
            if self.price_action_model is None:
                return None

            # Price action extractor expects OHLCV DataFrame, but we receive image
            # For compatibility, we'll need to pass the OHLCV data instead
            # This method will be called with OHLCV data in the updated interface
            if hasattr(self.price_action_model, 'extract'):
                # If it's a DataFrame (OHLCV data), extract directly
                if hasattr(image, 'columns'):  # It's a DataFrame
                    return self.price_action_model.extract(image)
                else:  # It's an image, return zero vector for now
                    logger.warning("Price action needs OHLCV data, not image. Returning zero vector.")
                    return np.zeros(self.price_action_model.get_feature_dim(), dtype=np.float32)
        except Exception:
            return None
    
    def _fuse_predictions(
        self,
        seq_features: np.ndarray,
        vit_features: Optional[np.ndarray],
        price_action_features: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse features from all modalities."""
        # Prepare inputs
        seq_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        vit_tensor = None
        if vit_features is not None:
            vit_tensor = torch.tensor(vit_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        price_action_tensor = None
        if price_action_features is not None:
            price_action_tensor = torch.tensor(price_action_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, gate_weights = self.fusion_model(
                seq_tensor, vit_tensor, price_action_tensor,
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
    use_price_action: bool = False
) -> Union[RiskAwareTCNPredictor, HybridPredictor]:
    """
    Factory function to create appropriate predictor.
    
    Args:
        profile: Trading profile ('SCALP', 'INTRADAY', 'SWING')
        weights_path: Path to model weights
        use_vision: Enable ViT
        use_price_action: Enable Price Action patterns
    
    Returns:
        Configured predictor instance
    """
    config = PredictorConfig(
        profile=profile,
        use_vision=use_vision,
        use_price_action=use_price_action
    )
    
    if use_vision or use_price_action:
        return HybridPredictor(config, tcn_weights=weights_path)
    else:
        return RiskAwareTCNPredictor(config, weights_path=weights_path)
