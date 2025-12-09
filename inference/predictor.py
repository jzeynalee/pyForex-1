# inference/predictor.py
"""
Hybrid Predictor for pyForex Trading System.

UPDATED: Now uses checkpoint_loader for loading features from checkpoints,
eliminating hardcoded TOP_FEATURES imports.

Combines multiple models for robust trading predictions:
- TCN (Temporal Convolutional Network) for time-series analysis
- ViT (Vision Transformer) for visual chart patterns
- YOLO for candlestick pattern detection
- Fusion network for combining modalities
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


# =============================================================================
# Type Definitions
# =============================================================================

class Signal(IntEnum):
    """Trading signal enumeration."""
    BEAR = 0    # Sell
    SIDEWAYS = 1  # Hold
    BULL = 2    # Buy


class PredictionResult(NamedTuple):
    """Structured prediction output."""
    probabilities: np.ndarray  # [P(BEAR), P(SIDEWAYS), P(BULL)]
    predicted_class: int       # Signal enum value
    confidence: float          # Max probability
    signal_name: str           # 'BEAR', 'SIDEWAYS', 'BULL'
    gate_weights: Optional[np.ndarray] = None  # Modality weights [seq, vit, yolo]


@dataclass
class PredictorConfig:
    """Configuration for predictors."""
    weights_dir: str = "models/weights"
    device: str = "auto"
    sequence_length: int = 30
    
    # Model selection
    model_type: str = "tcn"  # 'tcn' or 'lstm'
    profile: str = "INTRADAY"  # 'SCALP', 'INTRADAY', 'SWING'
    
    # Inference settings
    confidence_threshold: float = 0.6


# =============================================================================
# Helper Functions
# =============================================================================

def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =============================================================================
# TCN Predictor (Primary)
# =============================================================================

class TCNPredictor:
    """
    TCN-based predictor that loads features from checkpoint.
    
    This is the recommended predictor for production use.
    Features are automatically loaded from the model checkpoint,
    ensuring consistency between training and inference.
    
    Usage:
        predictor = TCNPredictor(
            checkpoint_path="models/weights/tcn_enhanced_best.pt"
        )
        
        result = predictor.predict(df_window)
        print(f"Signal: {result.signal_name}, Confidence: {result.confidence:.2%}")
    """
    
    SIGNAL_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
    
    def __init__(
        self,
        checkpoint_path: str = "models/weights/tcn_enhanced_best.pt",
        device: str = "auto",
        seq_len: int = 30,
    ):
        """
        Initialize TCN predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint (contains model + features)
            device: Device to run inference on
            seq_len: Sequence length for input
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(device)
        self.seq_len = seq_len
        
        # Load model and features from checkpoint
        self.model = None
        self.feature_columns: List[str] = []
        self.scaler = None
        self._checkpoint_data = None
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load model and features from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Use checkpoint_loader if available, otherwise manual loading
        try:
            from utils.checkpoint_loader import ModelLoader
            
            loader = ModelLoader(str(self.checkpoint_path), device=str(self.device))
            self.model = loader.get_model()
            self.feature_columns = loader.get_features()
            self._checkpoint_data = loader.checkpoint
            
            logger.info(f"✅ Loaded model via checkpoint_loader")
            logger.info(f"   Features: {len(self.feature_columns)}")
            
        except ImportError:
            # Fallback: manual loading
            logger.warning("checkpoint_loader not available, using manual loading")
            self._load_checkpoint_manual()
    
    def _load_checkpoint_manual(self):
        """Manual checkpoint loading (fallback)."""
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location='cpu',
            weights_only=False
        )
        self._checkpoint_data = checkpoint
        
        # Get features
        if 'feature_columns' in checkpoint:
            self.feature_columns = checkpoint['feature_columns']
        else:
            raise ValueError(
                "Checkpoint doesn't contain feature_columns. "
                "Please retrain with train_tcn_enhanced.py"
            )
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Build model
        from training.train_tcn_enhanced import EnhancedTCN
        
        self.model = EnhancedTCN(
            input_dim=model_config.get('input_dim', len(self.feature_columns)),
            hidden_dim=model_config.get('hidden_dim', 64),
            num_classes=training_config.get('num_classes', 3),
            dropout=training_config.get('dropout', 0.2),
        )
        
        # Load weights
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Loaded model manually")
        logger.info(f"   Features: {len(self.feature_columns)}")
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from DataFrame.
        
        Adds any missing technical indicators and selects the
        features used during training.
        """
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure required OHLCV columns
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        
        # Add technical features if missing
        df = self._add_technical_features(df, close, high, low)
        
        # Select training features (handle missing gracefully)
        available = set(df.columns)
        features_to_use = []
        
        for f in self.feature_columns:
            if f in available:
                features_to_use.append(f)
            else:
                logger.warning(f"Feature '{f}' not found, using zeros")
                df[f] = 0
                features_to_use.append(f)
        
        return df[features_to_use].values
    
    def _add_technical_features(
        self, 
        df: pd.DataFrame, 
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> pd.DataFrame:
        """Add commonly needed technical indicators."""
        # RSI
        if 'rsi_14' not in df.columns:
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = pd.Series(gains).rolling(14).mean()
            avg_loss = pd.Series(losses).rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df['rsi_14'] = np.concatenate([[50], rsi.values])
        
        # ATR
        if 'atr_14' not in df.columns:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
            )
            df['atr_14'] = pd.Series(tr).rolling(14).mean().fillna(0)
        
        # EMAs
        for period in [9, 20, 50, 200]:
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).ewm(span=period, adjust=False).mean()
        
        # MACD
        if 'macd' not in df.columns:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if 'bb_position' not in df.columns:
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Stochastic
        if 'stoch_k' not in df.columns:
            low_14 = pd.Series(low).rolling(14).min()
            high_14 = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ROC
        for period in [5, 10, 20]:
            col = f'roc_{period}'
            if col not in df.columns:
                df[col] = (close - np.roll(close, period)) / (np.roll(close, period) + 1e-10) * 100
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using robust scaling."""
        # Simple robust scaling (median/IQR based)
        median = np.median(features, axis=0)
        q75, q25 = np.percentile(features, [75, 25], axis=0)
        iqr = q75 - q25 + 1e-10
        
        return (features - median) / iqr
    
    def predict(self, df_window: pd.DataFrame) -> PredictionResult:
        """
        Make prediction on DataFrame window.
        
        Args:
            df_window: DataFrame with at least seq_len rows
                      Must contain OHLC columns at minimum
        
        Returns:
            PredictionResult with probabilities, class, confidence
        """
        if len(df_window) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows, got {len(df_window)}")
        
        # Take last seq_len rows
        df_input = df_window.tail(self.seq_len).copy()
        
        # Prepare and scale features
        features = self._prepare_features(df_input)
        features_scaled = self._scale_features(features)
        
        # Create tensor
        tensor = torch.tensor(features_scaled, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionResult(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_name=self.SIGNAL_NAMES[predicted_class],
            gate_weights=None,
        )
    
    def predict_batch(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple sequences.
        
        Args:
            sequences: Array of shape (batch, seq_len, features)
        
        Returns:
            (probabilities, predictions) tuple
        """
        tensor = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        
        predictions = np.argmax(probs, axis=1)
        
        return probs, predictions
    
    def get_signal(
        self,
        df_window: pd.DataFrame,
        threshold: float = 0.6,
    ) -> str:
        """
        Get trading signal with confidence threshold.
        
        Args:
            df_window: Input DataFrame
            threshold: Minimum confidence to return signal
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        result = self.predict(df_window)
        
        if result.confidence < threshold:
            return 'HOLD'
        
        # Map to trading signals
        if result.signal_name == 'BULL':
            return 'BUY'
        elif result.signal_name == 'BEAR':
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_feature_columns(self) -> List[str]:
        """Get the feature columns used by this model."""
        return self.feature_columns.copy()
    
    def get_checkpoint_info(self) -> Dict:
        """Get information about the loaded checkpoint."""
        if self._checkpoint_data is None:
            return {}
        
        return {
            'profile': self._checkpoint_data.get('profile'),
            'created_at': self._checkpoint_data.get('created_at'),
            'metrics': self._checkpoint_data.get('metrics', {}),
            'feature_count': len(self.feature_columns),
        }


# =============================================================================
# Hybrid Predictor (Multi-Modal)
# =============================================================================

class HybridPredictor:
    """
    Multi-modal predictor combining TCN, ViT, and YOLO.
    
    Uses FusionNet to combine predictions from:
    - TCN: Time-series patterns
    - ViT: Visual chart patterns
    - YOLO: Candlestick pattern detection
    """
    
    SIGNAL_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
    
    def __init__(
        self,
        tcn_checkpoint: str = "models/weights/tcn_enhanced_best.pt",
        fusion_checkpoint: Optional[str] = "models/weights/fusion_best.pt",
        vit_checkpoint: Optional[str] = "models/weights/vit_best.pt",
        yolo_checkpoint: Optional[str] = "models/weights/yolo_best.pt",
        device: str = "auto",
        seq_len: int = 30,
    ):
        """
        Initialize hybrid predictor.
        
        If only tcn_checkpoint is provided, falls back to TCN-only mode.
        """
        self.device = get_device(device)
        self.seq_len = seq_len
        
        # Load TCN (required)
        self.tcn_predictor = TCNPredictor(
            checkpoint_path=tcn_checkpoint,
            device=str(self.device),
            seq_len=seq_len,
        )
        
        # Track which components are loaded
        self.has_fusion = False
        self.has_vit = False
        self.has_yolo = False
        
        # Load optional components
        self._load_optional_components(fusion_checkpoint, vit_checkpoint, yolo_checkpoint)
    
    def _load_optional_components(
        self,
        fusion_path: Optional[str],
        vit_path: Optional[str],
        yolo_path: Optional[str],
    ):
        """Load optional multi-modal components."""
        # Fusion network
        if fusion_path and Path(fusion_path).exists():
            try:
                from models.fusion import FusionNet
                
                checkpoint = torch.load(fusion_path, map_location=self.device)
                self.fusion = FusionNet()
                
                if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                    self.fusion.load_state_dict(checkpoint['model_state'])
                else:
                    self.fusion.load_state_dict(checkpoint)
                
                self.fusion = self.fusion.to(self.device)
                self.fusion.eval()
                self.has_fusion = True
                logger.info("✅ Loaded FusionNet")
                
            except Exception as e:
                logger.warning(f"Could not load fusion model: {e}")
        
        # ViT
        if vit_path and Path(vit_path).exists():
            try:
                from models.vit import ViTExtractor
                
                self.vit = ViTExtractor(pretrained=False)
                checkpoint = torch.load(vit_path, map_location=self.device)
                self.vit.load_state_dict(checkpoint)
                self.vit = self.vit.to(self.device)
                self.vit.eval()
                self.has_vit = True
                logger.info("✅ Loaded ViT")
                
            except Exception as e:
                logger.warning(f"Could not load ViT: {e}")
        
        # YOLO
        if yolo_path and Path(yolo_path).exists():
            try:
                from models.yolo_detector import YOLOPatternDetector
                
                self.yolo = YOLOPatternDetector(model_path=yolo_path)
                self.has_yolo = self.yolo.enabled
                if self.has_yolo:
                    logger.info("✅ Loaded YOLO detector")
                    
            except Exception as e:
                logger.warning(f"Could not load YOLO: {e}")
    
    def predict(
        self,
        df_window: pd.DataFrame,
        chart_image: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """
        Make multi-modal prediction.
        
        Args:
            df_window: DataFrame with price data
            chart_image: Optional chart image for ViT/YOLO
        
        Returns:
            PredictionResult with combined prediction
        """
        # If no multi-modal components, use TCN only
        if not self.has_fusion:
            return self.tcn_predictor.predict(df_window)
        
        # Get TCN features
        tcn_result = self.tcn_predictor.predict(df_window)
        
        # Get ViT features (if available and image provided)
        if self.has_vit and chart_image is not None:
            vit_features = self._get_vit_features(chart_image)
        else:
            vit_features = torch.zeros(1, 768).to(self.device)
        
        # Get YOLO features
        if self.has_yolo and chart_image is not None:
            yolo_features = self._get_yolo_features(chart_image)
        else:
            yolo_features = torch.zeros(1, 20).to(self.device)
        
        # TCN features (use hidden representation)
        # For now, use probability as proxy
        tcn_features = torch.tensor(
            tcn_result.probabilities, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        tcn_features = F.pad(tcn_features, (0, 64 - 3))  # Pad to expected dim
        
        # Fusion
        with torch.no_grad():
            logits, gates = self.fusion.forward_with_gates(
                tcn_features, vit_features, yolo_features
            )
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            gate_weights = gates.cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionResult(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_name=self.SIGNAL_NAMES[predicted_class],
            gate_weights=gate_weights,
        )
    
    def _get_vit_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract ViT features from chart image."""
        # Preprocess image
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Resize to 224x224
        from PIL import Image
        img = Image.fromarray(image.astype(np.uint8))
        img = img.resize((224, 224))
        
        # Normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        
        # To tensor (B, C, H, W)
        tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.vit(tensor)
        
        return features
    
    def _get_yolo_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract YOLO pattern features from chart image."""
        pattern_vector = self.yolo.detect(image)
        return torch.tensor(pattern_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_signal(
        self,
        df_window: pd.DataFrame,
        chart_image: Optional[np.ndarray] = None,
        threshold: float = 0.6,
    ) -> str:
        """Get trading signal with threshold."""
        result = self.predict(df_window, chart_image)
        
        if result.confidence < threshold:
            return 'HOLD'
        
        if result.signal_name == 'BULL':
            return 'BUY'
        elif result.signal_name == 'BEAR':
            return 'SELL'
        else:
            return 'HOLD'


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# DEPRECATED: Use TCNPredictor instead
SimpleLSTMPredictor = TCNPredictor


# =============================================================================
# Factory Functions
# =============================================================================

def create_predictor(
    predictor_type: str = "tcn",
    checkpoint_path: str = "models/weights/tcn_enhanced_best.pt",
    **kwargs
) -> Union[TCNPredictor, HybridPredictor]:
    """
    Factory function for creating predictors.
    
    Args:
        predictor_type: 'tcn', 'hybrid', or 'simple' (legacy alias)
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional arguments
    
    Returns:
        Predictor instance
    """
    if predictor_type in ("tcn", "simple"):
        return TCNPredictor(checkpoint_path=checkpoint_path, **kwargs)
    elif predictor_type == "hybrid":
        return HybridPredictor(tcn_checkpoint=checkpoint_path, **kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


def load_predictor_from_checkpoint(
    checkpoint_path: str,
    device: str = "auto",
) -> TCNPredictor:
    """
    Load predictor directly from checkpoint.
    
    This is the recommended way to load a predictor as it
    automatically handles feature loading.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device for inference
    
    Returns:
        Configured TCNPredictor
    """
    return TCNPredictor(checkpoint_path=checkpoint_path, device=device)