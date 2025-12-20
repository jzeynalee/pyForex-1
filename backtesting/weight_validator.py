"""
Model Weight Validator for Backtesting
=======================================

Validates that all required ML model weights exist before running backtest.
Ensures no runtime failures due to missing model files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WeightValidationResult:
    """Result of weight validation."""
    is_valid: bool
    existing_weights: List[str] = field(default_factory=list)
    missing_weights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [f"Weight Validation: {status}"]
        lines.append(f"  Existing: {len(self.existing_weights)}")
        lines.append(f"  Missing: {len(self.missing_weights)}")
        if self.missing_weights:
            lines.append("  Missing files:")
            for w in self.missing_weights:
                lines.append(f"    - {w}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠️ {w}")
        return "\n".join(lines)


class WeightValidator:
    """
    Validates ML model weights for backtesting.
    
    Checks:
    - TCN weights per profile and timeframe
    - Fusion model weights
    - ViT weights (optional)
    - YOLO weights (optional)
    - Meta-labeling model (optional)
    - Exit advisor model (optional)
    """
    
    # Weight file definitions
    WEIGHT_DEFINITIONS = {
        # TCN by profile and timeframe
        'tcn': {
            'SCALP': {
                'M5': 'scalp_m5_best.pt',
                'M15': 'scalp_m15_best.pt',
                'H1': 'scalp_h1_best.pt',
            },
            'INTRADAY': {
                'M15': 'intraday_m15_best.pt',
                'H1': 'intraday_h1_best.pt',
                'H4': 'intraday_h4_best.pt',
            },
            'SWING': {
                'H1': 'swing_h1_best.pt',
                'H4': 'swing_h4_best.pt',
                'D1': 'swing_d1_best.pt',
            },
        },
        # Generic/shared models
        'generic': {
            'tcn_generic': 'tcn_best.pt',
            'tcn_enhanced': 'tcn_enhanced_best.pt',
            'fusion': 'fusion_best.pt',
        },
        # Optional models
        'optional': {
            'vit': 'vit_best.pt',
            'yolo': 'yolo_patterns.pt',
            'meta_model': 'meta_model.joblib',
            'exit_model': 'exit_model.pt',
        }
    }
    
    def __init__(self, weights_dir: str = "models/weights"):
        """
        Initialize validator.
        
        Args:
            weights_dir: Path to weights directory (relative to project root)
        """
        self.weights_dir = Path(weights_dir)
        
    def validate_all(self, project_root: Optional[Path] = None) -> WeightValidationResult:
        """
        Validate all model weights.
        
        Args:
            project_root: Project root directory
            
        Returns:
            WeightValidationResult with validation status
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
            
        weights_path = project_root / self.weights_dir
        
        existing = []
        missing = []
        warnings = []
        
        # Check TCN weights for all profiles
        for profile, timeframes in self.WEIGHT_DEFINITIONS['tcn'].items():
            for tf, filename in timeframes.items():
                filepath = weights_path / filename
                if filepath.exists():
                    existing.append(f"tcn/{profile}/{tf}: {filename}")
                else:
                    missing.append(f"tcn/{profile}/{tf}: {filename}")
        
        # Check generic models
        for name, filename in self.WEIGHT_DEFINITIONS['generic'].items():
            filepath = weights_path / filename
            if filepath.exists():
                existing.append(f"generic/{name}: {filename}")
            else:
                missing.append(f"generic/{name}: {filename}")
        
        # Check optional models (warnings only)
        for name, filename in self.WEIGHT_DEFINITIONS['optional'].items():
            filepath = weights_path / filename
            if not filepath.exists():
                warnings.append(f"Optional model missing: {name} ({filename})")
        
        # Determine validity (required weights must exist)
        # At minimum, we need tcn_generic or profile-specific TCN
        has_tcn = any('tcn' in w for w in existing)
        is_valid = has_tcn and len(missing) == 0
        
        # If we have at least one TCN, we can proceed with warnings
        if has_tcn and len(missing) > 0:
            warnings.append("Some profile-specific TCN weights missing, will use generic TCN")
            is_valid = True  # Can still run with generic TCN
        
        return WeightValidationResult(
            is_valid=is_valid,
            existing_weights=existing,
            missing_weights=missing,
            warnings=warnings
        )
    
    def validate_for_profile(
        self, 
        profile: str, 
        project_root: Optional[Path] = None
    ) -> WeightValidationResult:
        """
        Validate weights for a specific trading profile.
        
        Args:
            profile: Trading profile (SCALP, INTRADAY, SWING)
            project_root: Project root directory
            
        Returns:
            WeightValidationResult
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
            
        weights_path = project_root / self.weights_dir
        
        existing = []
        missing = []
        warnings = []
        
        profile = profile.upper()
        
        # Check profile-specific TCN weights
        if profile in self.WEIGHT_DEFINITIONS['tcn']:
            for tf, filename in self.WEIGHT_DEFINITIONS['tcn'][profile].items():
                filepath = weights_path / filename
                if filepath.exists():
                    existing.append(f"tcn/{profile}/{tf}: {filename}")
                else:
                    missing.append(f"tcn/{profile}/{tf}: {filename}")
        else:
            warnings.append(f"Unknown profile: {profile}")
        
        # Check generic TCN as fallback
        generic_tcn = weights_path / self.WEIGHT_DEFINITIONS['generic']['tcn_generic']
        if generic_tcn.exists():
            existing.append(f"generic/tcn: {self.WEIGHT_DEFINITIONS['generic']['tcn_generic']}")
        
        # Check fusion
        fusion_path = weights_path / self.WEIGHT_DEFINITIONS['generic']['fusion']
        if fusion_path.exists():
            existing.append(f"generic/fusion: {self.WEIGHT_DEFINITIONS['generic']['fusion']}")
        else:
            warnings.append("Fusion model missing - will use TCN-only predictions")
        
        # Determine validity
        has_tcn = any('tcn' in w for w in existing)
        is_valid = has_tcn
        
        if not has_tcn:
            missing.append("No TCN weights found - cannot run backtest")
        
        return WeightValidationResult(
            is_valid=is_valid,
            existing_weights=existing,
            missing_weights=missing,
            warnings=warnings
        )
    
    def get_weight_path(
        self, 
        model_type: str, 
        profile: Optional[str] = None,
        timeframe: Optional[str] = None,
        project_root: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Get path to a specific weight file.
        
        Args:
            model_type: Type of model (tcn, fusion, vit, yolo, meta_model, exit_model)
            profile: Trading profile (for TCN)
            timeframe: Timeframe (for TCN)
            project_root: Project root directory
            
        Returns:
            Path to weight file if exists, None otherwise
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
            
        weights_path = project_root / self.weights_dir
        
        # TCN with profile/timeframe
        if model_type == 'tcn' and profile and timeframe:
            profile = profile.upper()
            timeframe = timeframe.upper()
            if profile in self.WEIGHT_DEFINITIONS['tcn']:
                if timeframe in self.WEIGHT_DEFINITIONS['tcn'][profile]:
                    filename = self.WEIGHT_DEFINITIONS['tcn'][profile][timeframe]
                    filepath = weights_path / filename
                    if filepath.exists():
                        return filepath
            
            # Fallback to generic TCN
            generic_path = weights_path / self.WEIGHT_DEFINITIONS['generic']['tcn_generic']
            if generic_path.exists():
                return generic_path
            return None
        
        # Generic models
        if model_type in self.WEIGHT_DEFINITIONS['generic']:
            filepath = weights_path / self.WEIGHT_DEFINITIONS['generic'][model_type]
            return filepath if filepath.exists() else None
        
        # Optional models
        if model_type in self.WEIGHT_DEFINITIONS['optional']:
            filepath = weights_path / self.WEIGHT_DEFINITIONS['optional'][model_type]
            return filepath if filepath.exists() else None
        
        return None
    
    def list_available_weights(self, project_root: Optional[Path] = None) -> Dict[str, List[str]]:
        """
        List all available weight files.
        
        Returns:
            Dict mapping category to list of available weights
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
            
        weights_path = project_root / self.weights_dir
        
        available = {
            'tcn_profiles': [],
            'generic': [],
            'optional': []
        }
        
        # Check TCN by profile
        for profile, timeframes in self.WEIGHT_DEFINITIONS['tcn'].items():
            for tf, filename in timeframes.items():
                if (weights_path / filename).exists():
                    available['tcn_profiles'].append(f"{profile}/{tf}")
        
        # Check generic
        for name, filename in self.WEIGHT_DEFINITIONS['generic'].items():
            if (weights_path / filename).exists():
                available['generic'].append(name)
        
        # Check optional
        for name, filename in self.WEIGHT_DEFINITIONS['optional'].items():
            if (weights_path / filename).exists():
                available['optional'].append(name)
        
        return available


def validate_weights_for_backtest(
    profile: str = "INTRADAY",
    project_root: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Convenience function to validate weights before backtest.
    
    Args:
        profile: Trading profile
        project_root: Project root directory
        
    Returns:
        Tuple of (is_valid, message)
    """
    validator = WeightValidator()
    result = validator.validate_for_profile(profile, project_root)
    
    logger.info(str(result))
    
    return result.is_valid, str(result)


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)
    
    validator = WeightValidator()
    
    print("=" * 60)
    print("FULL VALIDATION")
    print("=" * 60)
    result = validator.validate_all()
    print(result)
    
    print("\n" + "=" * 60)
    print("AVAILABLE WEIGHTS")
    print("=" * 60)
    available = validator.list_available_weights()
    for category, weights in available.items():
        print(f"\n{category}:")
        for w in weights:
            print(f"  - {w}")
    
    print("\n" + "=" * 60)
    print("PROFILE-SPECIFIC VALIDATION")
    print("=" * 60)
    for profile in ['SCALP', 'INTRADAY', 'SWING']:
        print(f"\n{profile}:")
        result = validator.validate_for_profile(profile)
        print(result)
