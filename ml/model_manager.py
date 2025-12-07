"""
Model Manager Module for pyForex ML System.

Handles model versioning, storage, validation, deployment, and rollback.
"""

import os
import json
import pickle
import hashlib
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    version: str
    created_at: datetime
    training_start: datetime
    training_end: datetime
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    training_samples: int
    validation_metrics: Dict[str, float]
    profile_name: str  # SCALP, SWING, etc.
    data_hash: str  # Hash of training data
    notes: str = ""
    is_active: bool = False
    is_validated: bool = False
    validation_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'training_start': self.training_start.isoformat(),
            'training_end': self.training_end.isoformat(),
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'training_samples': self.training_samples,
            'validation_metrics': self.validation_metrics,
            'profile_name': self.profile_name,
            'data_hash': self.data_hash,
            'notes': self.notes,
            'is_active': self.is_active,
            'is_validated': self.is_validated,
            'validation_timestamp': self.validation_timestamp.isoformat() if self.validation_timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        return cls(
            model_id=data['model_id'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            training_start=datetime.fromisoformat(data['training_start']),
            training_end=datetime.fromisoformat(data['training_end']),
            model_type=data['model_type'],
            hyperparameters=data['hyperparameters'],
            feature_names=data['feature_names'],
            training_samples=data['training_samples'],
            validation_metrics=data['validation_metrics'],
            profile_name=data['profile_name'],
            data_hash=data['data_hash'],
            notes=data.get('notes', ''),
            is_active=data.get('is_active', False),
            is_validated=data.get('is_validated', False),
            validation_timestamp=datetime.fromisoformat(data['validation_timestamp']) if data.get('validation_timestamp') else None
        )


@dataclass 
class ValidationResult:
    """Result of model validation against current model."""
    timestamp: datetime
    candidate_id: str
    baseline_id: Optional[str]
    passed: bool
    metrics_comparison: Dict[str, Dict[str, float]]  # metric -> {candidate, baseline, diff}
    improvement_pct: float
    validation_samples: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'candidate_id': self.candidate_id,
            'baseline_id': self.baseline_id,
            'passed': self.passed,
            'metrics_comparison': self.metrics_comparison,
            'improvement_pct': self.improvement_pct,
            'validation_samples': self.validation_samples,
            'details': self.details,
            'recommendation': self.recommendation
        }


@dataclass
class ManagerConfig:
    """Configuration for model manager."""
    models_dir: str = "./models"
    max_versions: int = 10  # Keep last N versions
    
    # Validation settings
    validation_metric: str = "sharpe_ratio"  # Primary metric for comparison
    min_improvement_pct: float = 5.0  # Candidate must be X% better
    min_validation_samples: int = 100
    
    # Additional validation criteria
    secondary_metrics: List[str] = field(default_factory=lambda: ["win_rate", "profit_factor"])
    max_degradation_pct: float = 10.0  # Max allowed degradation in secondary metrics
    
    # Deployment settings
    require_validation: bool = True  # Must pass validation before activation
    auto_rollback_threshold: float = 0.2  # Rollback if performance drops by this %


class ModelManager:
    """
    Manages model lifecycle: versioning, storage, validation, and deployment.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of all models
        self.registry: Dict[str, ModelMetadata] = {}
        self.registry_file = self.models_dir / "registry.json"
        
        # Active models per profile
        self.active_models: Dict[str, str] = {}  # profile -> model_id
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"ModelManager initialized with {len(self.registry)} models")
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)
                
                self.registry = {
                    k: ModelMetadata.from_dict(v) 
                    for k, v in data.get('models', {}).items()
                }
                self.active_models = data.get('active_models', {})
                
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.registry = {}
                self.active_models = {}
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            data = {
                'models': {k: v.to_dict() for k, v in self.registry.items()},
                'active_models': self.active_models,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _generate_model_id(self, profile: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{profile}_{version}_{timestamp}"
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of training data for reproducibility tracking."""
        try:
            if hasattr(data, 'to_json'):
                data_str = data.to_json()
            else:
                data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()[:12]
        except:
            return "unknown"
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get path for model storage."""
        return self.models_dir / model_id
    
    def save_model(
        self,
        model: Any,
        profile_name: str,
        version: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        feature_names: List[str],
        training_data: Any,
        training_start: datetime,
        training_end: datetime,
        validation_metrics: Dict[str, float],
        notes: str = ""
    ) -> str:
        """
        Save a trained model with metadata.
        Returns the model_id.
        """
        model_id = self._generate_model_id(profile_name, version)
        model_path = self._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            created_at=datetime.now(),
            training_start=training_start,
            training_end=training_end,
            model_type=model_type,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            training_samples=len(training_data) if hasattr(training_data, '__len__') else 0,
            validation_metrics=validation_metrics,
            profile_name=profile_name,
            data_hash=self._compute_data_hash(training_data),
            notes=notes
        )
        
        # Save metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Register model
        self.registry[model_id] = metadata
        self._save_registry()
        
        # Cleanup old versions
        self._cleanup_old_versions(profile_name)
        
        logger.info(f"Model saved: {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> tuple[Any, ModelMetadata]:
        """Load a model and its metadata."""
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self._get_model_path(model_id)
        model_file = model_path / "model.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        metadata = self.registry[model_id]
        
        logger.info(f"Model loaded: {model_id}")
        return model, metadata
    
    def get_active_model(self, profile_name: str) -> Optional[tuple[Any, ModelMetadata]]:
        """Get the currently active model for a profile."""
        model_id = self.active_models.get(profile_name)
        if model_id:
            return self.load_model(model_id)
        return None
    
    def validate_model(
        self,
        candidate_id: str,
        validation_data: Any,
        validation_targets: Any,
        predict_fn: Callable,
        metric_calculators: Dict[str, Callable]
    ) -> ValidationResult:
        """
        Validate a candidate model against the current active model.
        
        Args:
            candidate_id: ID of model to validate
            validation_data: Validation features
            validation_targets: True labels/values
            predict_fn: Function to generate predictions: fn(model, data) -> predictions
            metric_calculators: Dict of metric_name -> fn(predictions, targets) -> float
        """
        if candidate_id not in self.registry:
            raise ValueError(f"Candidate model {candidate_id} not found")
        
        candidate_meta = self.registry[candidate_id]
        profile = candidate_meta.profile_name
        baseline_id = self.active_models.get(profile)
        
        # Load candidate model
        candidate_model, _ = self.load_model(candidate_id)
        
        # Generate candidate predictions
        candidate_preds = predict_fn(candidate_model, validation_data)
        
        # Calculate candidate metrics
        candidate_metrics = {}
        for name, calc_fn in metric_calculators.items():
            try:
                candidate_metrics[name] = calc_fn(candidate_preds, validation_targets)
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
                candidate_metrics[name] = 0.0
        
        # Compare with baseline if exists
        metrics_comparison = {}
        baseline_metrics = {}
        
        if baseline_id and baseline_id in self.registry:
            baseline_model, _ = self.load_model(baseline_id)
            baseline_preds = predict_fn(baseline_model, validation_data)
            
            for name, calc_fn in metric_calculators.items():
                try:
                    baseline_metrics[name] = calc_fn(baseline_preds, validation_targets)
                except:
                    baseline_metrics[name] = 0.0
                
                cand_val = candidate_metrics.get(name, 0)
                base_val = baseline_metrics.get(name, 0)
                diff = cand_val - base_val
                diff_pct = (diff / base_val * 100) if base_val != 0 else 0
                
                metrics_comparison[name] = {
                    'candidate': cand_val,
                    'baseline': base_val,
                    'diff': diff,
                    'diff_pct': diff_pct
                }
        else:
            # No baseline - just record candidate metrics
            for name, value in candidate_metrics.items():
                metrics_comparison[name] = {
                    'candidate': value,
                    'baseline': None,
                    'diff': None,
                    'diff_pct': None
                }
        
        # Determine if validation passed
        passed, improvement_pct, recommendation = self._evaluate_validation(
            metrics_comparison, baseline_id is not None
        )
        
        result = ValidationResult(
            timestamp=datetime.now(),
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            passed=passed,
            metrics_comparison=metrics_comparison,
            improvement_pct=improvement_pct,
            validation_samples=len(validation_data) if hasattr(validation_data, '__len__') else 0,
            details={
                'primary_metric': self.config.validation_metric,
                'min_improvement_required': self.config.min_improvement_pct
            },
            recommendation=recommendation
        )
        
        # Update model metadata
        self.registry[candidate_id].is_validated = passed
        self.registry[candidate_id].validation_timestamp = datetime.now()
        self._save_registry()
        
        self.validation_history.append(result)
        
        logger.info(
            f"Validation {'PASSED' if passed else 'FAILED'} for {candidate_id}. "
            f"Improvement: {improvement_pct:.2f}%"
        )
        
        return result
    
    def _evaluate_validation(
        self,
        metrics_comparison: Dict,
        has_baseline: bool
    ) -> tuple[bool, float, str]:
        """Evaluate validation results."""
        if not has_baseline:
            # No baseline to compare - pass by default
            return True, 0.0, "No baseline model. Candidate can be deployed."
        
        primary_metric = self.config.validation_metric
        
        if primary_metric not in metrics_comparison:
            return False, 0.0, f"Primary metric {primary_metric} not found."
        
        primary_comp = metrics_comparison[primary_metric]
        improvement_pct = primary_comp.get('diff_pct', 0) or 0
        
        # Check primary metric improvement
        if improvement_pct < self.config.min_improvement_pct:
            return False, improvement_pct, (
                f"Insufficient improvement in {primary_metric}: "
                f"{improvement_pct:.2f}% < {self.config.min_improvement_pct}% required"
            )
        
        # Check secondary metrics don't degrade too much
        for metric in self.config.secondary_metrics:
            if metric in metrics_comparison:
                comp = metrics_comparison[metric]
                diff_pct = comp.get('diff_pct', 0) or 0
                if diff_pct < -self.config.max_degradation_pct:
                    return False, improvement_pct, (
                        f"Too much degradation in {metric}: "
                        f"{diff_pct:.2f}% (max allowed: -{self.config.max_degradation_pct}%)"
                    )
        
        return True, improvement_pct, (
            f"Validation passed. {primary_metric} improved by {improvement_pct:.2f}%"
        )
    
    def activate_model(self, model_id: str, force: bool = False) -> bool:
        """
        Activate a model for live use.
        
        Args:
            model_id: Model to activate
            force: Skip validation requirement
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.registry[model_id]
        
        # Check validation
        if self.config.require_validation and not force:
            if not metadata.is_validated:
                logger.error(f"Model {model_id} not validated. Use force=True to override.")
                return False
        
        # Deactivate current active model
        profile = metadata.profile_name
        current_active = self.active_models.get(profile)
        if current_active and current_active in self.registry:
            self.registry[current_active].is_active = False
        
        # Activate new model
        self.active_models[profile] = model_id
        metadata.is_active = True
        
        self._save_registry()
        
        logger.info(f"Model {model_id} activated for profile {profile}")
        return True
    
    def rollback(self, profile_name: str, steps: int = 1) -> Optional[str]:
        """
        Rollback to a previous model version.
        
        Args:
            profile_name: Profile to rollback
            steps: Number of versions to rollback
        
        Returns:
            ID of activated model, or None if rollback failed
        """
        # Get models for this profile, sorted by creation date
        profile_models = [
            (mid, meta) for mid, meta in self.registry.items()
            if meta.profile_name == profile_name
        ]
        profile_models.sort(key=lambda x: x[1].created_at, reverse=True)
        
        if len(profile_models) <= steps:
            logger.error(f"Not enough versions to rollback {steps} steps")
            return None
        
        # Get target model
        target_id = profile_models[steps][0]
        
        # Deactivate current
        current_active = self.active_models.get(profile_name)
        if current_active and current_active in self.registry:
            self.registry[current_active].is_active = False
        
        # Activate rollback target
        self.active_models[profile_name] = target_id
        self.registry[target_id].is_active = True
        
        self._save_registry()
        
        logger.info(f"Rolled back to model {target_id} for profile {profile_name}")
        return target_id
    
    def _cleanup_old_versions(self, profile_name: str) -> None:
        """Remove old model versions exceeding max_versions."""
        profile_models = [
            (mid, meta) for mid, meta in self.registry.items()
            if meta.profile_name == profile_name
        ]
        profile_models.sort(key=lambda x: x[1].created_at, reverse=True)
        
        # Keep active model + last N versions
        to_remove = []
        for i, (mid, meta) in enumerate(profile_models):
            if i >= self.config.max_versions and not meta.is_active:
                to_remove.append(mid)
        
        for model_id in to_remove:
            self._remove_model(model_id)
    
    def _remove_model(self, model_id: str) -> None:
        """Remove a model from storage."""
        if model_id in self.registry:
            del self.registry[model_id]
        
        model_path = self._get_model_path(model_id)
        if model_path.exists():
            shutil.rmtree(model_path)
        
        logger.info(f"Removed model: {model_id}")
    
    def list_models(
        self,
        profile_name: Optional[str] = None,
        only_active: bool = False
    ) -> List[ModelMetadata]:
        """List models, optionally filtered."""
        models = list(self.registry.values())
        
        if profile_name:
            models = [m for m in models if m.profile_name == profile_name]
        
        if only_active:
            models = [m for m in models if m.is_active]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed info about a model."""
        if model_id not in self.registry:
            return None
        
        metadata = self.registry[model_id]
        return {
            **metadata.to_dict(),
            'model_path': str(self._get_model_path(model_id)),
            'model_exists': (self._get_model_path(model_id) / "model.pkl").exists()
        }
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Dict]:
        """Compare multiple models side by side."""
        comparison = {}
        
        for model_id in model_ids:
            if model_id in self.registry:
                meta = self.registry[model_id]
                comparison[model_id] = {
                    'version': meta.version,
                    'created_at': meta.created_at.isoformat(),
                    'training_samples': meta.training_samples,
                    'validation_metrics': meta.validation_metrics,
                    'is_active': meta.is_active,
                    'is_validated': meta.is_validated
                }
        
        return comparison
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """Export a model with all metadata to a zip file."""
        if model_id not in self.registry:
            return False
        
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            return False
        
        shutil.make_archive(export_path.replace('.zip', ''), 'zip', model_path)
        logger.info(f"Model {model_id} exported to {export_path}")
        return True
    
    def import_model(self, import_path: str) -> Optional[str]:
        """Import a model from a zip file."""
        if not os.path.exists(import_path):
            return None
        
        # Extract to temp location
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(import_path, temp_dir)
            
            # Load metadata
            metadata_file = os.path.join(temp_dir, "metadata.json")
            if not os.path.exists(metadata_file):
                return None
            
            with open(metadata_file) as f:
                meta_dict = json.load(f)
            
            metadata = ModelMetadata.from_dict(meta_dict)
            model_id = metadata.model_id
            
            # Copy to models directory
            dest_path = self._get_model_path(model_id)
            if dest_path.exists():
                logger.warning(f"Model {model_id} already exists, overwriting")
                shutil.rmtree(dest_path)
            
            shutil.copytree(temp_dir, dest_path)
            
            # Register
            self.registry[model_id] = metadata
            self._save_registry()
            
            logger.info(f"Model imported: {model_id}")
            return model_id
