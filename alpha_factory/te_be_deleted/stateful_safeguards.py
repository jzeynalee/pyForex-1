"""
Stateful Safeguards for Alpha Factory

Phase 6.1: Stateful Safeguards (Critical Fix)

Replaces binary blocking with graduated risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"

class SafeguardState(Enum):
    """Safeguard operational states."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    RESTRICTED = "restricted"
    EMERGENCY = "emergency"
    CATASTROPHIC = "catastrophic"

@dataclass
class SafeguardThresholds:
    """Stateful safeguard thresholds."""
    # Activation thresholds
    volatility_explosion_activation: float = 3.0
    liquidity_shock_activation: float = 0.3
    spread_explosion_activation: float = 5.0
    regime_break_activation: float = 0.6
    anomaly_z_activation: float = 3.0
    
    # Release thresholds (hysteresis)
    volatility_explosion_release: float = 2.0
    liquidity_shock_release: float = 0.5
    spread_explosion_release: float = 3.0
    regime_break_release: float = 0.7
    anomaly_z_release: float = 2.0
    
    # Duration controls
    min_active_duration: int = 5  # Minimum periods to stay active
    max_active_duration: int = 50  # Maximum periods before forced release
    
    # Risk multipliers
    low_risk_multiplier: float = 1.0
    medium_risk_multiplier: float = 0.8
    high_risk_multiplier: float = 0.5
    critical_risk_multiplier: float = 0.2
    catastrophic_risk_multiplier: float = 0.0  # Only catastrophic blocks completely

@dataclass
class SafeguardTrigger:
    """Stateful safeguard trigger."""
    def __init__(self, trigger_type: str, risk_level: RiskLevel, severity_score: float, 
                 description: str, trigger_data: Dict[str, Any]):
        self.trigger_type = trigger_type
        self.risk_level = risk_level
        self.severity_score = severity_score
        self.description = description
        self.trigger_data = trigger_data
        self.timestamp = datetime.now()
        self.active_periods = 0
        self.state = SafeguardState.NORMAL
        self.release_threshold_met = False

class StatefulSafeguards:
    """Stateful safeguards with graduated risk management."""
    
    def __init__(self, config: SafeguardThresholds = None):
        self.config = config or SafeguardThresholds()
        
        # State management
        self.active_triggers = {}
        self.trigger_history = []
        self.current_state = SafeguardState.NORMAL
        self.risk_multiplier = 1.0
        
        # Market baselines
        self.baseline_volatility = 0.0001
        self.baseline_spread = 0.0002
        self.baseline_volume = 1000000
        self.baseline_regime = 'neutral'
        
        # Risk adjustments
        self.position_size_multiplier = 1.0
        self.ev_threshold_adjustment = 0.0
        self.entry_delay_periods = 0
        
        logger.info("Stateful safeguards initialized")
    
    def update_baselines(self, market_data: Dict[str, Any]):
        """Update market baselines."""
        if 'volatility' in market_data:
            self.baseline_volatility = 0.9 * self.baseline_volatility + 0.1 * market_data['volatility']
        if 'spread' in market_data:
            self.baseline_spread = 0.9 * self.baseline_spread + 0.1 * market_data['spread']
        if 'volume' in market_data:
            self.baseline_volume = 0.9 * self.baseline_volume + 0.1 * market_data['volume']
        if 'regime' in market_data:
            self.baseline_regime = market_data['regime']
    
    def assess_volatility_explosion(self, market_data: Dict[str, Any]) -> Optional[SafeguardTrigger]:
        """Assess volatility with stateful logic."""
        if 'volatility' not in market_data:
            return None
        
        current_vol = market_data['volatility']
        volatility_ratio = current_vol / self.baseline_volatility
        
        # Check activation
        if volatility_ratio > self.config.volatility_explosion_activation:
            risk_level = self._determine_risk_level(volatility_ratio, 
                                                   [3.0, 4.0, 5.0, 7.0],
                                                   [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CATASTROPHIC])
            
            trigger = SafeguardTrigger(
                'volatility_explosion',
                risk_level,
                min(1.0, volatility_ratio / 7.0),
                f"Volatility explosion: {volatility_ratio:.1f}x normal",
                {'current_volatility': current_vol, 'baseline_volatility': self.baseline_volatility, 'ratio': volatility_ratio}
            )
            
            return trigger
        
        return None
    
    def assess_liquidity_shock(self, market_data: Dict[str, Any]) -> Optional[SafeguardTrigger]:
        """Assess liquidity with stateful logic."""
        volume_ratio = market_data.get('volume', self.baseline_volume) / self.baseline_volume
        spread_ratio = market_data.get('spread', self.baseline_spread) / self.baseline_spread
        
        # Liquidity shock detection
        shock_severity = 0
        reasons = []
        
        if volume_ratio < self.config.liquidity_shock_activation:
            shock_severity += 0.5
            reasons.append(f"Low volume: {volume_ratio:.1%} of normal")
        
        if spread_ratio > self.config.liquidity_shock_activation:
            shock_severity += 0.5
            reasons.append(f"High spread: {spread_ratio:.1f}x normal")
        
        if shock_severity > 0:
            risk_level = RiskLevel.MEDIUM if shock_severity < 1.0 else RiskLevel.HIGH
            if spread_ratio > 3.0:
                risk_level = RiskLevel.CRITICAL
            if spread_ratio > 5.0:
                risk_level = RiskLevel.CATASTROPHIC
            
            trigger = SafeguardTrigger(
                'liquidity_shock',
                risk_level,
                min(1.0, shock_severity + spread_ratio / 10.0),
                f"Liquidity shock: {'; '.join(reasons)}",
                {'volume_ratio': volume_ratio, 'spread_ratio': spread_ratio, 'reasons': reasons}
            )
            
            return trigger
        
        return None
    
    def assess_regime_break(self, market_data: Dict[str, Any]) -> Optional[SafeguardTrigger]:
        """Assess regime stability with stateful logic."""
        # This would need regime history - simplified for now
        current_regime = market_data.get('regime', 'unknown')
        
        # Simple regime break detection (would need rolling window in real implementation)
        if current_regime == 'unknown':
            trigger = SafeguardTrigger(
                'regime_break',
                RiskLevel.MEDIUM,
                0.6,
                f"Regime uncertainty: {current_regime}",
                {'current_regime': current_regime, 'baseline_regime': self.baseline_regime}
            )
            return trigger
        
        return None
    
    def assess_system_anomaly(self, market_data: Dict[str, Any]) -> Optional[SafeguardTrigger]:
        """Assess system anomalies with stateful logic."""
        anomalies = []
        z_scores = []
        
        # Volatility anomaly
        if 'volatility' in market_data:
            vol_z = abs(market_data['volatility'] - self.baseline_volatility) / (self.baseline_volatility * 0.2)
            if vol_z > self.config.anomaly_z_activation:
                anomalies.append(f"Volatility anomaly: Z-score {vol_z:.1f}")
                z_scores.append(vol_z)
        
        # Volume anomaly
        if 'volume' in market_data:
            vol_z = abs(market_data['volume'] - self.baseline_volume) / (self.baseline_volume * 0.3)
            if vol_z > self.config.anomaly_z_activation:
                anomalies.append(f"Volume anomaly: Z-score {vol_z:.1f}")
                z_scores.append(vol_z)
        
        if anomalies:
            max_z = max(z_scores)
            risk_level = self._determine_risk_level(max_z, [3.0, 4.0, 5.0, 7.0],
                                                   [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CATASTROPHIC])
            
            trigger = SafeguardTrigger(
                'system_anomaly',
                risk_level,
                min(1.0, max_z / 7.0),
                f"System anomalies: {'; '.join(anomalies)}",
                {'anomalies': anomalies, 'max_z_score': max_z}
            )
            
            return trigger
        
        return None
    
    def _determine_risk_level(self, value: float, thresholds: List[float], levels: List[RiskLevel]) -> RiskLevel:
        """Determine risk level based on value and thresholds."""
        for threshold, level in zip(thresholds, levels):
            if value >= threshold:
                return level
        return levels[0]
    
    def update_safeguard_states(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update all safeguard states and calculate risk adjustments."""
        # Update baselines
        self.update_baselines(market_data)
        
        # Assess all triggers
        assessors = [
            self.assess_volatility_explosion,
            self.assess_liquidity_shock,
            self.assess_regime_break,
            self.assess_system_anomaly
        ]
        
        new_triggers = []
        for assessor in assessors:
            trigger = assessor(market_data)
            if trigger:
                new_triggers.append(trigger)
        
        # Update existing triggers
        for trigger_type in list(self.active_triggers.keys()):
            # Check if trigger should be released
            existing_trigger = self.active_triggers[trigger_type]
            existing_trigger.active_periods += 1
            
            # Check release conditions
            should_release = self._check_release_conditions(existing_trigger, market_data)
            
            if should_release:
                existing_trigger.release_threshold_met = True
                logger.info(f"Safeguard {trigger_type} release conditions met")
        
        # Add new triggers
        for trigger in new_triggers:
            if trigger.trigger_type not in self.active_triggers:
                self.active_triggers[trigger.trigger_type] = trigger
                logger.warning(f"Safeguard activated: {trigger.trigger_type} - {trigger.description}")
        
        # Remove released triggers (with minimum duration)
        released_triggers = []
        for trigger_type, trigger in list(self.active_triggers.items()):
            if trigger.release_threshold_met and trigger.active_periods >= self.config.min_active_duration:
                released_triggers.append(trigger_type)
                del self.active_triggers[trigger_type]
                logger.info(f"Safeguard released: {trigger_type}")
        
        # Force release triggers that exceed max duration
        for trigger_type, trigger in list(self.active_triggers.items()):
            if trigger.active_periods >= self.config.max_active_duration:
                del self.active_triggers[trigger_type]
                logger.warning(f"Safeguard force-released (max duration): {trigger_type}")
        
        # Calculate overall risk state
        self._calculate_risk_state()
        
        # Apply risk adjustments
        self._apply_risk_adjustments()
        
        return {
            'state': self.current_state.value,
            'risk_multiplier': self.risk_multiplier,
            'position_size_multiplier': self.position_size_multiplier,
            'ev_threshold_adjustment': self.ev_threshold_adjustment,
            'entry_delay_periods': self.entry_delay_periods,
            'active_triggers': len(self.active_triggers),
            'trigger_details': {t.trigger_type: t.risk_level.value for t in self.active_triggers.values()}
        }
    
    def _check_release_conditions(self, trigger: SafeguardTrigger, market_data: Dict[str, Any]) -> bool:
        """Check if trigger release conditions are met."""
        if trigger.trigger_type == 'volatility_explosion':
            current_ratio = market_data.get('volatility', self.baseline_volatility) / self.baseline_volatility
            return current_ratio < self.config.volatility_explosion_release
        
        elif trigger.trigger_type == 'liquidity_shock':
            volume_ratio = market_data.get('volume', self.baseline_volume) / self.baseline_volume
            spread_ratio = market_data.get('spread', self.baseline_spread) / self.baseline_spread
            return volume_ratio > self.config.liquidity_shock_release and spread_ratio < self.config.liquidity_shock_release
        
        elif trigger.trigger_type == 'regime_break':
            current_regime = market_data.get('regime', 'unknown')
            return current_regime != 'unknown'
        
        elif trigger.trigger_type == 'system_anomaly':
            # Simplified - would need rolling calculations
            return trigger.active_periods > 10  # Auto-release after 10 periods
        
        return False
    
    def _calculate_risk_state(self):
        """Calculate overall risk state from active triggers."""
        if not self.active_triggers:
            self.current_state = SafeguardState.NORMAL
            return
        
        # Determine highest risk level
        highest_risk = RiskLevel.LOW
        for trigger in self.active_triggers.values():
            if trigger.risk_level.value > highest_risk.value:
                highest_risk = trigger.risk_level
        
        # Map to safeguard state
        if highest_risk == RiskLevel.LOW:
            self.current_state = SafeguardState.NORMAL
        elif highest_risk == RiskLevel.MEDIUM:
            self.current_state = SafeguardState.ELEVATED
        elif highest_risk == RiskLevel.HIGH:
            self.current_state = SafeguardState.RESTRICTED
        elif highest_risk == RiskLevel.CRITICAL:
            self.current_state = SafeguardState.EMERGENCY
        else:
            self.current_state = SafeguardState.CATASTROPHIC
    
    def _apply_risk_adjustments(self):
        """Apply risk adjustments based on current state."""
        if self.current_state == SafeguardState.NORMAL:
            self.risk_multiplier = self.config.low_risk_multiplier
            self.position_size_multiplier = 1.0
            self.ev_threshold_adjustment = 0.0
            self.entry_delay_periods = 0
        
        elif self.current_state == SafeguardState.ELEVATED:
            self.risk_multiplier = self.config.medium_risk_multiplier
            self.position_size_multiplier = 0.8
            self.ev_threshold_adjustment = 5.0  # Raise EV threshold by $5
            self.entry_delay_periods = 1
        
        elif self.current_state == SafeguardState.RESTRICTED:
            self.risk_multiplier = self.config.high_risk_multiplier
            self.position_size_multiplier = 0.5
            self.ev_threshold_adjustment = 10.0  # Raise EV threshold by $10
            self.entry_delay_periods = 3
        
        elif self.current_state == SafeguardState.EMERGENCY:
            self.risk_multiplier = self.config.critical_risk_multiplier
            self.position_size_multiplier = 0.2
            self.ev_threshold_adjustment = 20.0  # Raise EV threshold by $20
            self.entry_delay_periods = 5
        
        elif self.current_state == SafeguardState.CATASTROPHIC:
            self.risk_multiplier = self.config.catastrophic_risk_multiplier
            self.position_size_multiplier = 0.0  # Complete block
            self.ev_threshold_adjustment = 100.0  # Effectively block all trades
            self.entry_delay_periods = 999
    
    def get_safeguard_report(self) -> Dict[str, Any]:
        """Get comprehensive safeguard report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_state': self.current_state.value,
            'risk_multiplier': self.risk_multiplier,
            'position_size_multiplier': self.position_size_multiplier,
            'ev_threshold_adjustment': self.ev_threshold_adjustment,
            'entry_delay_periods': self.entry_delay_periods,
            'active_triggers': len(self.active_triggers),
            'trigger_details': {
                t.trigger_type: {
                    'risk_level': t.risk_level.value,
                    'active_periods': t.active_periods,
                    'severity': t.severity_score,
                    'description': t.description
                }
                for t in self.active_triggers.values()
            },
            'baselines': {
                'volatility': self.baseline_volatility,
                'spread': self.baseline_spread,
                'volume': self.baseline_volume,
                'regime': self.baseline_regime
            }
        }
