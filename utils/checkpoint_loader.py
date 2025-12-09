# utils/checkpoint_loader.py
"""
Unified checkpoint loading utilities for pyForex models.

Provides consistent interface for loading models trained with different scripts,
handling backward compatibility with older checkpoint formats.

Usage:
    from utils.checkpoint_loader import load_model, load_features, ModelLoader
    
    # Simple loading
    model, features = load_model("models/weights/tcn_enhanced_best.pt")
    
    # Full checkpoint info
    loader = ModelLoader("models/weights/tcn_enhanced_best.pt")
    model = loader.get_model()
    features = loader.get_features()
    config = loader.get_config()
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_type: str  # 'tcn', 'lstm', 'enhanced_tcn', etc.
    input_dim: int
    hidden_dim: int
    num_classes: int
    profile: Optional[str]
    receptive_field: Optional[int]
    feature_count: int
    created_at: Optional[str]


class CheckpointFormatError(Exception):
    """Raised when checkpoint format is unrecognized."""
    pass


class ModelLoader:
    """
    Unified loader for pyForex model checkpoints.
    
    Handles multiple checkpoint formats:
    - New format (train_tcn_enhanced.py): Contains feature_columns, config, etc.
    - Old format (train_lstm_enhanced.py): May have model_state only
    - Legacy format: Direct state dict
    
    Example:
        loader = ModelLoader("models/weights/tcn_enhanced_best.pt")
        
        # Get model ready for inference
        model = loader.get_model(device='cuda')
        
        # Get features used during training
        features = loader.get_features()
        
        # Get full configuration
        config = loader.get_config()
        
        # Get training history
        history = loader.get_training_history()
    """
    
    def __init__(self, checkpoint_path: Union[str, Path], device: str = 'auto'):
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model to ('auto', 'cpu', 'cuda')
        """
        self.path = Path(checkpoint_path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=False)
        
        # Detect format
        self._format = self._detect_format()
        
        # Cache
        self._model: Optional[nn.Module] = None
        self._features: Optional[List[str]] = None
    
    def _detect_format(self) -> str:
        """Detect checkpoint format."""
        if isinstance(self.checkpoint, dict):
            if 'feature_columns' in self.checkpoint:
                return 'enhanced_v3'  # New train_tcn_enhanced.py format
            elif 'model_state' in self.checkpoint:
                if 'config' in self.checkpoint:
                    return 'enhanced_v2'  # Intermediate format
                return 'enhanced_v1'  # Old format with model_state
            elif any(k.startswith('tcn.') or k.startswith('lstm.') for k in self.checkpoint.keys()):
                return 'state_dict'  # Direct state dict
        
        return 'unknown'
    
    def get_model(self, device: Optional[str] = None) -> nn.Module:
        """
        Get loaded model ready for inference.
        
        Args:
            device: Override device (uses loader's device if None)
        
        Returns:
            Model in eval mode
        """
        if self._model is not None:
            return self._model
        
        device = torch.device(device) if device else self.device
        
        # Get model architecture info
        info = self.get_model_info()
        
        # Build model based on detected type
        if info.model_type in ['tcn', 'enhanced_tcn']:
            from training.train_tcn_enhanced import EnhancedTCN
            
            model = EnhancedTCN(
                input_dim=info.input_dim,
                hidden_dim=info.hidden_dim,
                num_classes=info.num_classes,
            )
        elif info.model_type in ['lstm', 'enhanced_lstm']:
            # Try to import EnhancedLSTM
            try:
                from training.train_lstm_enhanced import EnhancedLSTM
                model = EnhancedLSTM(
                    input_dim=info.input_dim,
                    hidden_dim=info.hidden_dim,
                    num_classes=info.num_classes,
                )
            except ImportError:
                from models.lstm import LSTMModel
                model = LSTMModel(
                    input_dim=info.input_dim,
                    hidden_dim=info.hidden_dim,
                    num_classes=info.num_classes,
                )
        else:
            raise CheckpointFormatError(f"Unknown model type: {info.model_type}")
        
        # Load state dict
        state_dict = self._get_state_dict()
        
        # Handle module prefix from DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        self._model = model
        return model
    
    def _get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract state dict from checkpoint."""
        if self._format == 'state_dict':
            return self.checkpoint
        elif 'model_state' in self.checkpoint:
            return self.checkpoint['model_state']
        else:
            raise CheckpointFormatError("Cannot find model state dict in checkpoint")
    
    def get_features(self) -> List[str]:
        """
        Get feature columns used during training.
        
        Returns:
            List of feature column names
        
        Raises:
            CheckpointFormatError if features not found
        """
        if self._features is not None:
            return self._features
        
        if 'feature_columns' in self.checkpoint:
            self._features = self.checkpoint['feature_columns']
        elif 'config' in self.checkpoint:
            config = self.checkpoint['config']
            if 'feature_columns' in config:
                self._features = config['feature_columns']
            elif 'features' in config:
                self._features = config['features']
        
        if self._features is None:
            raise CheckpointFormatError(
                "Checkpoint doesn't contain feature columns. "
                "This model may have been trained with an older script. "
                "Consider retraining with train_tcn_enhanced.py or providing features manually."
            )
        
        return self._features
    
    def get_features_safe(self, fallback: Optional[List[str]] = None) -> Optional[List[str]]:
        """
        Get features with fallback instead of raising exception.
        
        Args:
            fallback: Features to return if not found in checkpoint
        
        Returns:
            Feature list or fallback
        """
        try:
            return self.get_features()
        except CheckpointFormatError:
            return fallback
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration from checkpoint."""
        if 'config' in self.checkpoint:
            return self.checkpoint['config']
        return {}
    
    def get_training_history(self) -> Dict[str, List]:
        """Get training history (losses, accuracies)."""
        if 'training_history' in self.checkpoint:
            return self.checkpoint['training_history']
        return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics from checkpoint."""
        if 'metrics' in self.checkpoint:
            return self.checkpoint['metrics']
        return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available."""
        if 'feature_importance' in self.checkpoint:
            return self.checkpoint['feature_importance']
        return {}
    
    def get_model_info(self) -> ModelInfo:
        """Get structured information about the model."""
        # Determine model type
        config = self.get_config()
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Infer type from checkpoint structure or config
        if 'tcn' in str(self.path).lower() or 'receptive_field' in model_config:
            model_type = 'enhanced_tcn'
        elif 'lstm' in str(self.path).lower():
            model_type = 'enhanced_lstm'
        else:
            # Try to detect from state dict keys
            state_dict = self._get_state_dict()
            if any('tcn' in k.lower() for k in state_dict.keys()):
                model_type = 'tcn'
            elif any('lstm' in k.lower() for k in state_dict.keys()):
                model_type = 'lstm'
            else:
                model_type = 'unknown'
        
        # Get dimensions
        try:
            features = self.get_features()
            input_dim = len(features)
        except CheckpointFormatError:
            input_dim = model_config.get('input_dim', 0)
        
        return ModelInfo(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=model_config.get('hidden_dim', 64),
            num_classes=training_config.get('num_classes', 3),
            profile=self.checkpoint.get('profile'),
            receptive_field=model_config.get('receptive_field'),
            feature_count=input_dim,
            created_at=self.checkpoint.get('created_at'),
        )
    
    def summary(self) -> str:
        """Get human-readable summary of checkpoint."""
        info = self.get_model_info()
        lines = [
            f"Checkpoint: {self.path.name}",
            f"  Format: {self._format}",
            f"  Model Type: {info.model_type}",
            f"  Input Dim: {info.input_dim}",
            f"  Hidden Dim: {info.hidden_dim}",
            f"  Classes: {info.num_classes}",
        ]
        
        if info.profile:
            lines.append(f"  Profile: {info.profile}")
        if info.receptive_field:
            lines.append(f"  Receptive Field: {info.receptive_field}")
        if info.created_at:
            lines.append(f"  Created: {info.created_at}")
        
        # Features preview
        try:
            features = self.get_features()
            lines.append(f"  Features: {len(features)}")
            if len(features) <= 5:
                lines.append(f"    {features}")
            else:
                lines.append(f"    {features[:3]} ... {features[-2:]}")
        except CheckpointFormatError:
            lines.append("  Features: Not stored in checkpoint")
        
        # Metrics
        metrics = self.get_metrics()
        if metrics:
            if 'best_val_acc' in metrics:
                lines.append(f"  Best Val Acc: {metrics['best_val_acc']:.2%}")
            if 'test_accuracy' in metrics:
                lines.append(f"  Test Acc: {metrics['test_accuracy']:.2%}")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(
    checkpoint_path: Union[str, Path],
    device: str = 'auto',
) -> Tuple[nn.Module, List[str]]:
    """
    Quick function to load model and features.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to
    
    Returns:
        (model, features) tuple
    """
    loader = ModelLoader(checkpoint_path, device=device)
    model = loader.get_model()
    features = loader.get_features()
    return model, features


def load_features(checkpoint_path: Union[str, Path]) -> List[str]:
    """
    Quick function to load just features from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
    
    Returns:
        List of feature names
    """
    loader = ModelLoader(checkpoint_path)
    return loader.get_features()


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> ModelInfo:
    """
    Quick function to get model info without loading weights.
    
    Args:
        checkpoint_path: Path to checkpoint
    
    Returns:
        ModelInfo dataclass
    """
    loader = ModelLoader(checkpoint_path)
    return loader.get_model_info()


def print_checkpoint_summary(checkpoint_path: Union[str, Path]):
    """Print human-readable checkpoint summary."""
    loader = ModelLoader(checkpoint_path)
    print(loader.summary())


# =============================================================================
# Backward Compatibility
# =============================================================================

# For code that imported TOP_FEATURES from train_lstm_enhanced
# This provides a way to get features from a default checkpoint
_DEFAULT_CHECKPOINT = "models/weights/tcn_enhanced_best.pt"

def get_default_features() -> List[str]:
    """
    Get features from default checkpoint.
    
    This is a compatibility shim for code that used:
        from training.train_lstm_enhanced import TOP_FEATURES
    
    Replace with:
        from utils.checkpoint_loader import get_default_features
        TOP_FEATURES = get_default_features()
    """
    try:
        return load_features(_DEFAULT_CHECKPOINT)
    except (FileNotFoundError, CheckpointFormatError):
        logger.warning(
            f"Could not load features from {_DEFAULT_CHECKPOINT}. "
            "Returning empty list. Train a model first."
        )
        return []


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_loader.py <checkpoint_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print_checkpoint_summary(path)