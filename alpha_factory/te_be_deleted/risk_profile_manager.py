"""
Risk Profile Manager for Alpha Factory

Separates research logic from production logic.
"""

import yaml
import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class RiskProfile(Enum):
    """Available risk profiles."""
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    PRODUCTION = "PRODUCTION"

class RiskProfileManager:
    """Manages risk profiles and provides configuration."""
    
    def __init__(self, config_file: str = "alpha_factory/risk_profiles.yaml"):
        self.config_file = config_file
        self.profiles = self._load_profiles()
        self.current_profile = RiskProfile.RESEARCH
        logger.info("Risk profile manager initialized")
    
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load risk profiles from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                profiles = yaml.safe_load(f)
            logger.info(f"Loaded {len(profiles)} risk profiles")
            return profiles
        except FileNotFoundError:
            logger.error(f"Risk profiles file not found: {self.config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing risk profiles: {e}")
            raise
    
    def set_profile(self, profile: RiskProfile) -> None:
        """Set the current risk profile."""
        self.current_profile = profile
        logger.info(f"Risk profile set to: {profile.value}")
    
    def get_config(self, profile: RiskProfile = None) -> Dict[str, Any]:
        """Get configuration for specified profile."""
        if profile is None:
            profile = self.current_profile
        
        config = self.profiles.get(profile.value, {})
        if not config:
            logger.error(f"Profile not found: {profile.value}")
            raise ValueError(f"Risk profile not found: {profile.value}")
        
        return config
    
    def get_ev_threshold(self, profile: RiskProfile = None) -> float:
        """Get EV threshold for profile."""
        config = self.get_config(profile)
        return config.get('min_ev_threshold', 0.0)
    
    def get_position_multiplier(self, profile: RiskProfile = None) -> float:
        """Get position size multiplier for profile."""
        config = self.get_config(profile)
        return config.get('position_size_multiplier', 1.0)
    
    def get_safeguard_severity(self, profile: RiskProfile = None) -> float:
        """Get safeguard severity scale for profile."""
        config = self.get_config(profile)
        return config.get('safeguard_severity_scale', 1.0)
    
    def get_max_concurrent_trades(self, profile: RiskProfile = None) -> int:
        """Get max concurrent trades for profile."""
        config = self.get_config(profile)
        return config.get('max_concurrent_trades', 10)
    
    def get_drawdown_limits(self, profile: RiskProfile = None) -> tuple:
        """Get drawdown limits for profile."""
        config = self.get_config(profile)
        soft = config.get('drawdown_soft_limit', 0.20)
        hard = config.get('drawdown_hard_limit', 0.35)
        return soft, hard
    
    def is_catastrophic_block_enabled(self, profile: RiskProfile = None) -> bool:
        """Check if catastrophic block is enabled for profile."""
        config = self.get_config(profile)
        return config.get('catastrophic_block_enabled', True)
    
    def is_emergency_stop_enabled(self, profile: RiskProfile = None) -> bool:
        """Check if emergency stop is enabled for profile."""
        config = self.get_config(profile)
        return config.get('emergency_stop_enabled', True)
    
    def get_decay_penalties(self, profile: RiskProfile = None) -> tuple:
        """Get decay response penalties for profile."""
        config = self.get_config(profile)
        ev_penalty = config.get('decay_ev_penalty', 0.0)
        size_reduction = config.get('decay_size_reduction', 0.0)
        return ev_penalty, size_reduction
    
    def get_release_conditions(self, profile: RiskProfile = None) -> tuple:
        """Get safeguard release conditions for profile."""
        config = self.get_config(profile)
        min_duration = config.get('min_active_duration', 3)
        max_duration = config.get('max_active_duration', 25)
        return min_duration, max_duration
    
    def validate_profile(self, profile: RiskProfile) -> bool:
        """Validate that profile has all required fields."""
        config = self.get_config(profile)
        
        required_fields = [
            'min_ev_threshold',
            'position_size_multiplier',
            'safeguard_severity_scale',
            'max_concurrent_trades',
            'drawdown_soft_limit',
            'drawdown_hard_limit'
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            logger.error(f"Profile {profile.value} missing required fields: {missing_fields}")
            return False
        
        return True
    
    def get_profile_summary(self, profile: RiskProfile = None) -> str:
        """Get human-readable summary of profile."""
        if profile is None:
            profile = self.current_profile
        
        config = self.get_config(profile)
        
        summary = f"""
{profile.value} PROFILE SUMMARY:
- EV Threshold: ${config.get('min_ev_threshold', 0.0):.2f}
- Position Multiplier: {config.get('position_size_multiplier', 1.0):.1f}x
- Safeguard Severity: {config.get('safeguard_severity_scale', 1.0):.1f}x
- Max Concurrent Trades: {config.get('max_concurrent_trades', 10)}
- Drawdown Limits: {config.get('drawdown_soft_limit', 0.20):.1%} / {config.get('drawdown_hard_limit', 0.35):.1%}
- Catastrophic Block: {'Enabled' if config.get('catastrophic_block_enabled', True) else 'Disabled'}
- Emergency Stop: {'Enabled' if config.get('emergency_stop_enabled', True) else 'Disabled'}
"""
        return summary.strip()
