"""
Advanced Safeguards for Alpha Factory

Phase 6: Advanced Safeguards (Optional but Professional)

Goal: Survive rare but deadly conditions
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
    EMERGENCY = "emergency"

class TriggerType(Enum):
    """Types of safeguard triggers."""
    VOLATILITY_EXPLOSION = "volatility_explosion"
    LIQUIDITY_SHOCK = "liquidity_shock"
    SPREAD_EXPLOSION = "spread_explosion"
    REGIME_BREAK = "regime_break"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EXTREME_DRAWDOWN = "extreme_drawdown"
    SYSTEM_ANOMALY = "system_anomaly"

@dataclass
class SafeguardConfig:
    """Configuration for advanced safeguards."""
    # Volatility thresholds
    volatility_explosion_threshold: float = 3.0  # 3x normal volatility
    volatility_window: int = 20  # Periods for volatility calculation
    
    # Liquidity thresholds
    liquidity_volume_threshold: float = 0.3  # 30% of normal volume
    liquidity_spread_threshold: float = 2.0   # 2x normal spread
    
    # Spread thresholds
    spread_explosion_threshold: float = 5.0   # 5x normal spread
    max_acceptable_spread: float = 0.0010    # 10 pips
    
    # Regime detection
    regime_confidence_threshold: float = 0.4   # 40% min regime confidence
    regime_stability_window: int = 50         # Periods for regime stability
    
    # Correlation monitoring
    correlation_breakdown_threshold: float = 0.9  # 90% correlation indicates breakdown
    correlation_window: int = 100                 # Periods for correlation
    
    # Drawdown thresholds
    extreme_drawdown_threshold: float = 0.20     # 20% drawdown triggers emergency
    drawdown_recovery_window: int = 20           # Periods for recovery check
    
    # System anomaly detection
    anomaly_detection_window: int = 10          # Periods for anomaly detection
    anomaly_z_threshold: float = 3.0             # 3 standard deviations
    
    # Response actions
    emergency_stop_enabled: bool = True
    manual_override_required: bool = True
    audit_trail_enabled: bool = True

class SafeguardTrigger:
    """Represents a safeguard trigger event."""
    
    def __init__(self, trigger_type: TriggerType, risk_level: RiskLevel, 
                 severity_score: float, description: str, 
                 trigger_data: Dict[str, Any]):
        self.trigger_type = trigger_type
        self.risk_level = risk_level
        self.severity_score = severity_score
        self.description = description
        self.trigger_data = trigger_data
        self.timestamp = datetime.now()
        self.resolved = False
        self.resolution_action = ""
        self.resolution_timestamp = None

class AdvancedSafeguards:
    """Professional advanced safeguards for Alpha Factory."""
    
    def __init__(self, config: SafeguardConfig = None):
        self.config = config or SafeguardConfig()
        
        # Monitoring data
        self.market_data_history = []
        self.volatility_history = []
        self.spread_history = []
        self.volume_history = []
        self.regime_history = []
        self.correlation_history = []
        
        # Trigger tracking
        self.active_triggers = []
        self.trigger_history = []
        self.audit_trail = []
        
        # System state
        self.emergency_stop_active = False
        self.manual_override_active = False
        self.last_system_check = datetime.now()
        
        # Baseline values
        self.baseline_volatility = 0.0001
        self.baseline_spread = 0.0002
        self.baseline_volume = 1000000
        self.baseline_correlations = {}
        
        logger.info("Advanced safeguards initialized")
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Update market data for safeguard monitoring.
        
        Args:
            market_data: Dictionary with current market data
        """
        # Add timestamp
        market_data['timestamp'] = datetime.now()
        
        # Store in history
        self.market_data_history.append(market_data)
        
        # Extract specific metrics
        if 'volatility' in market_data:
            self.volatility_history.append(market_data['volatility'])
        
        if 'spread' in market_data:
            self.spread_history.append(market_data['spread'])
        
        if 'volume' in market_data:
            self.volume_history.append(market_data['volume'])
        
        if 'regime' in market_data:
            self.regime_history.append(market_data['regime'])
        
        # Keep histories manageable
        max_history = max(self.config.volatility_window, 
                         self.config.liquidity_volume_threshold * 10,
                         self.config.correlation_window) * 2
        
        if len(self.market_data_history) > max_history:
            self.market_data_history = self.market_data_history[-max_history//2:]
        
        if len(self.volatility_history) > max_history:
            self.volatility_history = self.volatility_history[-max_history//2:]
        
        if len(self.spread_history) > max_history:
            self.spread_history = self.spread_history[-max_history//2:]
        
        if len(self.volume_history) > max_history:
            self.volume_history = self.volume_history[-max_history//2:]
        
        # Update baselines periodically
        if len(self.market_data_history) % 100 == 0:
            self._update_baselines()
        
        logger.debug(f"Market data updated: {len(self.market_data_history)} records")
    
    def _update_baselines(self) -> None:
        """Update baseline values from recent history."""
        if len(self.volatility_history) >= 50:
            self.baseline_volatility = np.mean(self.volatility_history[-50:])
        
        if len(self.spread_history) >= 50:
            self.baseline_spread = np.mean(self.spread_history[-50:])
        
        if len(self.volume_history) >= 50:
            self.baseline_volume = np.mean(self.volume_history[-50:])
        
        logger.info("Baselines updated")
    
    def detect_volatility_explosion(self) -> Optional[SafeguardTrigger]:
        """
        Detect volatility explosion conditions.
        
        Returns:
            SafeguardTrigger if volatility explosion detected
        """
        if len(self.volatility_history) < self.config.volatility_window:
            return None
        
        recent_volatility = np.mean(self.volatility_history[-self.config.volatility_window:])
        volatility_ratio = recent_volatility / self.baseline_volatility
        
        if volatility_ratio > self.config.volatility_explosion_threshold:
            # Determine risk level
            if volatility_ratio > 5.0:
                risk_level = RiskLevel.EMERGENCY
                severity = 1.0
            elif volatility_ratio > 4.0:
                risk_level = RiskLevel.CRITICAL
                severity = 0.8
            elif volatility_ratio > 3.0:
                risk_level = RiskLevel.HIGH
                severity = 0.6
            else:
                risk_level = RiskLevel.MEDIUM
                severity = 0.4
            
            trigger = SafeguardTrigger(
                TriggerType.VOLATILITY_EXPLOSION,
                risk_level,
                severity,
                f"Volatility explosion: {volatility_ratio:.1f}x normal",
                {
                    'current_volatility': recent_volatility,
                    'baseline_volatility': self.baseline_volatility,
                    'ratio': volatility_ratio
                }
            )
            
            return trigger
        
        return None
    
    def detect_liquidity_shock(self) -> Optional[SafeguardTrigger]:
        """
        Detect liquidity shock conditions.
        
        Returns:
            SafeguardTrigger if liquidity shock detected
        """
        if len(self.volume_history) < 10 or len(self.spread_history) < 10:
            return None
        
        recent_volume = np.mean(self.volume_history[-10:])
        recent_spread = np.mean(self.spread_history[-10:])
        
        volume_ratio = recent_volume / self.baseline_volume
        spread_ratio = recent_spread / self.baseline_spread
        
        # Check for liquidity shock
        liquidity_shock = False
        shock_reasons = []
        
        if volume_ratio < self.config.liquidity_volume_threshold:
            liquidity_shock = True
            shock_reasons.append(f"Low volume: {volume_ratio:.1%} of normal")
        
        if spread_ratio > self.config.liquidity_spread_threshold:
            liquidity_shock = True
            shock_reasons.append(f"High spread: {spread_ratio:.1f}x normal")
        
        if liquidity_shock:
            # Determine risk level
            if volume_ratio < 0.1 or spread_ratio > 5.0:
                risk_level = RiskLevel.EMERGENCY
                severity = 1.0
            elif volume_ratio < 0.2 or spread_ratio > 3.0:
                risk_level = RiskLevel.CRITICAL
                severity = 0.8
            else:
                risk_level = RiskLevel.HIGH
                severity = 0.6
            
            trigger = SafeguardTrigger(
                TriggerType.LIQUIDITY_SHOCK,
                risk_level,
                severity,
                f"Liquidity shock: {'; '.join(shock_reasons)}",
                {
                    'current_volume': recent_volume,
                    'baseline_volume': self.baseline_volume,
                    'volume_ratio': volume_ratio,
                    'current_spread': recent_spread,
                    'baseline_spread': self.baseline_spread,
                    'spread_ratio': spread_ratio,
                    'reasons': shock_reasons
                }
            )
            
            return trigger
        
        return None
    
    def detect_spread_explosion(self) -> Optional[SafeguardTrigger]:
        """
        Detect spread explosion conditions.
        
        Returns:
            SafeguardTrigger if spread explosion detected
        """
        if len(self.spread_history) < 10:
            return None
        
        recent_spread = np.mean(self.spread_history[-10:])
        spread_ratio = recent_spread / self.baseline_spread
        
        if spread_ratio > self.config.spread_explosion_threshold:
            # Determine risk level
            if recent_spread > self.config.max_acceptable_spread * 2:
                risk_level = RiskLevel.EMERGENCY
                severity = 1.0
            elif recent_spread > self.config.max_acceptable_spread * 1.5:
                risk_level = RiskLevel.CRITICAL
                severity = 0.8
            else:
                risk_level = RiskLevel.HIGH
                severity = 0.6
            
            trigger = SafeguardTrigger(
                TriggerType.SPREAD_EXPLOSION,
                risk_level,
                severity,
                f"Spread explosion: {spread_ratio:.1f}x normal, {recent_spread*10000:.1f} pips",
                {
                    'current_spread': recent_spread,
                    'baseline_spread': self.baseline_spread,
                    'ratio': spread_ratio,
                    'spread_pips': recent_spread * 10000
                }
            )
            
            return trigger
        
        return None
    
    def detect_regime_break(self) -> Optional[SafeguardTrigger]:
        """
        Detect regime break conditions.
        
        Returns:
            SafeguardTrigger if regime break detected
        """
        if len(self.regime_history) < self.config.regime_stability_window:
            return None
        
        recent_regimes = self.regime_history[-self.config.regime_stability_window:]
        
        # Check regime stability
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Find dominant regime
        dominant_regime = max(regime_counts, key=regime_counts.get)
        dominant_ratio = regime_counts[dominant_regime] / len(recent_regimes)
        
        # Check for regime instability
        if dominant_ratio < 0.6:  # Less than 60% stability
            # Determine risk level
            if dominant_ratio < 0.3:
                risk_level = RiskLevel.EMERGENCY
                severity = 1.0
            elif dominant_ratio < 0.5:
                risk_level = RiskLevel.CRITICAL
                severity = 0.8
            else:
                risk_level = RiskLevel.HIGH
                severity = 0.6
            
            trigger = SafeguardTrigger(
                TriggerType.REGIME_BREAK,
                risk_level,
                severity,
                f"Regime instability: {dominant_ratio:.1%} stability for {dominant_regime}",
                {
                    'dominant_regime': dominant_regime,
                    'stability_ratio': dominant_ratio,
                    'regime_counts': regime_counts,
                    'window_size': len(recent_regimes)
                }
            )
            
            return trigger
        
        return None
    
    def detect_system_anomaly(self) -> Optional[SafeguardTrigger]:
        """
        Detect system anomalies using statistical methods.
        
        Returns:
            SafeguardTrigger if system anomaly detected
        """
        if len(self.market_data_history) < self.config.anomaly_detection_window:
            return None
        
        # Check for anomalies in various metrics
        anomalies = []
        
        # Volatility anomaly
        if len(self.volatility_history) >= self.config.anomaly_detection_window:
            recent_vol = self.volatility_history[-self.config.anomaly_detection_window:]
            vol_mean = np.mean(recent_vol[:-1])
            vol_std = np.std(recent_vol[:-1])
            current_vol = recent_vol[-1]
            
            if vol_std > 0:
                vol_z = abs(current_vol - vol_mean) / vol_std
                if vol_z > self.config.anomaly_z_threshold:
                    anomalies.append(f"Volatility anomaly: Z-score {vol_z:.1f}")
        
        # Spread anomaly
        if len(self.spread_history) >= self.config.anomaly_detection_window:
            recent_spread = self.spread_history[-self.config.anomaly_detection_window:]
            spread_mean = np.mean(recent_spread[:-1])
            spread_std = np.std(recent_spread[:-1])
            current_spread = recent_spread[-1]
            
            if spread_std > 0:
                spread_z = abs(current_spread - spread_mean) / spread_std
                if spread_z > self.config.anomaly_z_threshold:
                    anomalies.append(f"Spread anomaly: Z-score {spread_z:.1f}")
        
        # Volume anomaly
        if len(self.volume_history) >= self.config.anomaly_detection_window:
            recent_volume = self.volume_history[-self.config.anomaly_detection_window:]
            volume_mean = np.mean(recent_volume[:-1])
            volume_std = np.std(recent_volume[:-1])
            current_volume = recent_volume[-1]
            
            if volume_std > 0:
                volume_z = abs(current_volume - volume_mean) / volume_std
                if volume_z > self.config.anomaly_z_threshold:
                    anomalies.append(f"Volume anomaly: Z-score {volume_z:.1f}")
        
        if anomalies:
            # Determine risk level based on number and severity of anomalies
            if len(anomalies) >= 3:
                risk_level = RiskLevel.EMERGENCY
                severity = 1.0
            elif len(anomalies) >= 2:
                risk_level = RiskLevel.CRITICAL
                severity = 0.8
            else:
                risk_level = RiskLevel.HIGH
                severity = 0.6
            
            trigger = SafeguardTrigger(
                TriggerType.SYSTEM_ANOMALY,
                risk_level,
                severity,
                f"System anomalies detected: {'; '.join(anomalies)}",
                {
                    'anomalies': anomalies,
                    'anomaly_count': len(anomalies),
                    'detection_window': self.config.anomaly_detection_window
                }
            )
            
            return trigger
        
        return None
    
    def check_all_safeguards(self) -> List[SafeguardTrigger]:
        """
        Check all safeguard conditions.
        
        Returns:
            List of active safeguard triggers
        """
        triggers = []
        
        # Check each safeguard type
        safeguard_checks = [
            self.detect_volatility_explosion,
            self.detect_liquidity_shock,
            self.detect_spread_explosion,
            self.detect_regime_break,
            self.detect_system_anomaly
        ]
        
        for check_func in safeguard_checks:
            try:
                trigger = check_func()
                if trigger:
                    triggers.append(trigger)
                    self.active_triggers.append(trigger)
                    self.trigger_history.append(trigger)
                    
                    # Log the trigger
                    logger.warning(f"Safeguard triggered: {trigger.trigger_type.value} - {trigger.description}")
                    
                    # Add to audit trail
                    if self.config.audit_trail_enabled:
                        self.audit_trail.append({
                            'timestamp': trigger.timestamp,
                            'action': 'trigger',
                            'trigger_type': trigger.trigger_type.value,
                            'risk_level': trigger.risk_level.value,
                            'description': trigger.description
                        })
                    
                    # Check for emergency stop
                    if trigger.risk_level == RiskLevel.EMERGENCY and self.config.emergency_stop_enabled:
                        self.activate_emergency_stop(trigger)
                    
            except Exception as e:
                logger.error(f"Error in safeguard check {check_func.__name__}: {e}")
        
        self.last_system_check = datetime.now()
        
        return triggers
    
    def activate_emergency_stop(self, trigger: SafeguardTrigger) -> None:
        """
        Activate emergency stop due to critical condition.
        
        Args:
            trigger: The trigger that caused emergency stop
        """
        self.emergency_stop_active = True
        
        # Log emergency stop
        logger.critical(f"EMERGENCY STOP ACTIVATED: {trigger.description}")
        
        # Add to audit trail
        if self.config.audit_trail_enabled:
            self.audit_trail.append({
                'timestamp': datetime.now(),
                'action': 'emergency_stop',
                'trigger_type': trigger.trigger_type.value,
                'risk_level': trigger.risk_level.value,
                'description': trigger.description,
                'manual_override_required': self.config.manual_override_required
            })
    
    def deactivate_emergency_stop(self, override_reason: str, manual: bool = False) -> bool:
        """
        Deactivate emergency stop.
        
        Args:
            override_reason: Reason for deactivation
            manual: Whether this is a manual override
            
        Returns:
            True if deactivation successful
        """
        if self.config.manual_override_required and not manual:
            logger.error("Manual override required for emergency stop deactivation")
            return False
        
        self.emergency_stop_active = False
        self.manual_override_active = manual
        
        # Log deactivation
        logger.info(f"EMERGENCY STOP DEACTIVATED: {override_reason} (Manual: {manual})")
        
        # Add to audit trail
        if self.config.audit_trail_enabled:
            self.audit_trail.append({
                'timestamp': datetime.now(),
                'action': 'emergency_stop_deactivated',
                'reason': override_reason,
                'manual_override': manual
            })
        
        return True
    
    def get_safeguard_report(self) -> Dict[str, Any]:
        """Get comprehensive safeguard report."""
        
        # Calculate trigger statistics
        trigger_stats = {}
        for trigger_type in TriggerType:
            type_triggers = [t for t in self.trigger_history if t.trigger_type == trigger_type]
            trigger_stats[trigger_type.value] = {
                'total': len(type_triggers),
                'active': len([t for t in self.active_triggers if t.trigger_type == trigger_type]),
                'last_triggered': type_triggers[-1].timestamp.isoformat() if type_triggers else None
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'emergency_stop_active': self.emergency_stop_active,
            'manual_override_active': self.manual_override_active,
            'last_system_check': self.last_system_check.isoformat(),
            'active_triggers': len(self.active_triggers),
            'total_triggers': len(self.trigger_history),
            'trigger_statistics': trigger_stats,
            'baseline_values': {
                'volatility': self.baseline_volatility,
                'spread': self.baseline_spread,
                'volume': self.baseline_volume
            },
            'config': {
                'volatility_threshold': self.config.volatility_explosion_threshold,
                'spread_threshold': self.config.spread_explosion_threshold,
                'emergency_stop_enabled': self.config.emergency_stop_enabled,
                'manual_override_required': self.config.manual_override_required
            },
            'recent_audit_trail': self.audit_trail[-10:] if self.config.audit_trail_enabled else []
        }
