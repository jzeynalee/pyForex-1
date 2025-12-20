# ml/auto_training_manager.py
"""
Auto Training Manager for automated model training and management.

This module provides functionality for automatically training models
when they are missing or need retraining.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingRequirement(Enum):
    """Training requirement levels."""
    OPTIONAL = "optional"
    REQUIRED = "required"
    CRITICAL = "critical"


class AutoTrainingManager:
    """
    Manages automated training of ML models.
    
    Handles checking for missing models and triggering training
    based on requirements and availability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the auto training manager."""
        self.config = config or {}
        self.models_dir = Path(self.config.get('models_dir', 'models/weights'))
        self.requirements = self.config.get('requirements', {})
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model file exists."""
        model_path = self.models_dir / f"{model_name}.pt"
        return model_path.exists()
    
    def get_training_requirement(self, model_name: str) -> TrainingRequirement:
        """Get the training requirement level for a model."""
        requirement_str = self.requirements.get(model_name, "optional")
        return TrainingRequirement(requirement_str)
    
    def list_missing_models(self, model_list: List[str]) -> List[str]:
        """List models that are missing from the filesystem."""
        missing = []
        for model_name in model_list:
            if not self.check_model_exists(model_name):
                missing.append(model_name)
        return missing
    
    def should_train_model(self, model_name: str) -> bool:
        """Determine if a model should be trained based on requirements."""
        if self.check_model_exists(model_name):
            return False
        
        requirement = self.get_training_requirement(model_name)
        return requirement in [TrainingRequirement.REQUIRED, TrainingRequirement.CRITICAL]
    
    def get_training_priority(self, model_name: str) -> int:
        """Get training priority (higher = more important)."""
        requirement = self.get_training_requirement(model_name)
        priority_map = {
            TrainingRequirement.OPTIONAL: 1,
            TrainingRequirement.REQUIRED: 2,
            TrainingRequirement.CRITICAL: 3,
        }
        return priority_map.get(requirement, 1)
    
    def sort_models_by_priority(self, model_list: List[str]) -> List[str]:
        """Sort models by training priority."""
        return sorted(model_list, key=self.get_training_priority, reverse=True)


def check_and_train_missing_models(
    model_list: List[str],
    config: Optional[Dict[str, Any]] = None,
    force_train: bool = False
) -> Dict[str, bool]:
    """
    Check for missing models and optionally train them.
    
    Args:
        model_list: List of model names to check
        config: Configuration dictionary
        force_train: Whether to force training of optional models
        
    Returns:
        Dictionary mapping model names to training status
    """
    manager = AutoTrainingManager(config)
    results = {}
    
    # Sort by priority
    sorted_models = manager.sort_models_by_priority(model_list)
    
    for model_name in sorted_models:
        if manager.check_model_exists(model_name):
            results[model_name] = True  # Already exists
            continue
        
        should_train = manager.should_train_model(model_name) or force_train
        
        if should_train:
            try:
                # In a real implementation, this would trigger actual training
                logger.info(f"Training model: {model_name}")
                results[model_name] = True
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = False
        else:
            logger.info(f"Skipping optional model: {model_name}")
            results[model_name] = False
    
    return results


# Convenience functions for common operations
def get_available_models(models_dir: str = "models/weights") -> List[str]:
    """Get list of available model files."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    model_files = list(models_path.glob("*.pt"))
    return [f.stem for f in model_files]


def validate_model_requirements(
    required_models: List[str],
    models_dir: str = "models/weights"
) -> Dict[str, bool]:
    """Validate that all required models are available."""
    available = get_available_models(models_dir)
    results = {}
    
    for model_name in required_models:
        results[model_name] = model_name in available
    
    return results
