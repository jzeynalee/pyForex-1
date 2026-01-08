# alpha_factory/three_tf_system.py
"""
3TF (Three-Timeframe) Institutional Architecture - Profile Aware

Core Principles:
1. HTF -> Participation Governor (Bias & Risk Scale)
2. MTF -> Structure Filter (Validation)
3. LTF -> Execution Timing (Trigger)

Invariants:
- Logic is identical across all profiles (Scalping/Intraday/Swing).
- Only the data feeding the snapshots changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import numpy as np
from .trading_profiles import TradingProfile

logger = logging.getLogger(__name__)

# ==========================================
# 1. DATA STRUCTURES (IMMUTABLE INTERFACES)
# ==========================================

@dataclass(frozen=True)
class FeatureSnapshot:
    """
    Standardized interface for feature intelligence.
    Produced by Feature Council/Pools offline.
    """
    timestamp: datetime
    timeframe: str            # Added for audit trail (e.g., "4h", "15m")
    directional_score: float  # Normalized [-1.0, +1.0]
    confidence: float         # [0.0, 1.0]
    stability: float          # [0.0, 1.0]
    regime_flags: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HTFDecision:
    allow: bool
    bias: Literal["LONG", "SHORT", "BOTH", "NONE"]
    risk_scale: float
    reason: str

@dataclass(frozen=True)
class MTFDecision:
    pass_structure: bool
    quality_boost: float
    reason: str

@dataclass(frozen=True)
class LTFSignal:
    trigger: bool
    final_confidence: float
    reason: str

@dataclass
class TradeInstruction:
    symbol: str
    profile: str
    direction: str
    size_multiplier: float
    confidence: float
    timestamp: datetime
    logic_path: str

# ==========================================
# 2. LOGIC ENGINES (PURE FUNCTIONS)
# ==========================================

class ThreeTFLogic:
    """Stateless logic containers verifying the architecture contracts."""

    @staticmethod
    def htf_decide(snapshot: FeatureSnapshot) -> HTFDecision:
        """HTF (Governor): Should we participate?"""
        if snapshot.stability < 0.4:
            return HTFDecision(False, "NONE", 0.0, f"HTF({snapshot.timeframe}) Stability < 0.4")

        if snapshot.confidence < 0.5:
            return HTFDecision(False, "NONE", 0.0, f"HTF({snapshot.timeframe}) Confidence < 0.5")

        if snapshot.directional_score > 0.25:
            return HTFDecision(True, "LONG", 1.0, "Strong Bullish Bias")
        
        if snapshot.directional_score < -0.25:
            return HTFDecision(True, "SHORT", 1.0, "Strong Bearish Bias")

        return HTFDecision(True, "BOTH", 0.6, "Neutral/Range Mode")

    @staticmethod
    def mtf_validate(snapshot: FeatureSnapshot, htf: HTFDecision) -> MTFDecision:
        """MTF (Structure): Is structure aligned?"""
        if not htf.allow:
            return MTFDecision(False, 0.0, "Blocked by HTF")

        if snapshot.confidence < 0.6:
            return MTFDecision(False, 0.0, f"MTF({snapshot.timeframe}) Confidence < 0.6")

        if abs(snapshot.directional_score) < 0.2:
            return MTFDecision(False, 0.0, "MTF Structure too weak")

        if htf.bias != "BOTH":
            if np.sign(snapshot.directional_score) != (1.0 if htf.bias == "LONG" else -1.0):
                return MTFDecision(False, 0.0, f"Structure contradicts HTF {htf.bias}")

        return MTFDecision(True, 0.25, "Structure Aligned")

    @staticmethod
    def ltf_trigger(snapshot: FeatureSnapshot, htf: HTFDecision, mtf: MTFDecision) -> LTFSignal:
        """LTF (Timing): Is timing favorable now?"""
        if not mtf.pass_structure:
            return LTFSignal(False, 0.0, "Blocked by MTF Structure")

        if snapshot.confidence < 0.65:
            return LTFSignal(False, 0.0, f"LTF({snapshot.timeframe}) Confidence < 0.65")

        if snapshot.stability < 0.5:
            return LTFSignal(False, 0.0, "LTF Micro-instability")

        final_conf = min(1.0, snapshot.confidence + mtf.quality_boost)
        return LTFSignal(True, final_conf, "Execution Triggered")

# ==========================================
# 3. FEATURE ADAPTER
# ==========================================

class FeatureAdapter:
    @staticmethod
    def create_snapshot(
        timestamp: datetime,
        timeframe: str,
        decision_signal: Dict[str, Any], 
        causality_results: Dict[str, Any],
        market_regime: str
    ) -> FeatureSnapshot:
        """Converts raw AlphaFactory output into a strict FeatureSnapshot."""
        
        base_score = decision_signal.get('confidence', 0.5)
        direction = decision_signal.get('decision', 'HOLD')
        
        if direction == 'BUY':
            dir_score = base_score
        elif direction == 'SELL':
            dir_score = -base_score
        else:
            dir_score = 0.0
            
        stability = 0.8
        if 'stationarity_analysis' in causality_results:
            stat_data = causality_results['stationarity_analysis']
            if stat_data.get('features_checked', 0) > 0:
                stability = stat_data['stationary_features'] / stat_data['features_checked']

        return FeatureSnapshot(
            timestamp=timestamp,
            timeframe=timeframe,
            directional_score=dir_score,
            confidence=decision_signal.get('confidence', 0.0),
            stability=stability,
            regime_flags={'regime': market_regime}
        )

# ==========================================
# 4. ORCHESTRATOR
# ==========================================

class ThreeTFOrchestrator:
    """Manages the lifecycle of a 3TF decision process for a specific profile."""
    
    def __init__(self, symbol: str, profile: TradingProfile):
        self.symbol = symbol
        self.profile = profile
        self.logic = ThreeTFLogic()
    
    def process_3tf(self, 
                    snapshot_htf: FeatureSnapshot, 
                    snapshot_mtf: FeatureSnapshot, 
                    snapshot_ltf: FeatureSnapshot) -> Optional[TradeInstruction]:
        """Execute the 3TF pipeline using the assigned profile's timeframes."""
        
        # Validation: Ensure snapshots match the profile
        if snapshot_htf.timeframe != self.profile.htf.value:
            logger.warning(f"HTF Mismatch: Expected {self.profile.htf.value}, got {snapshot_htf.timeframe}")
        
        logger.info(f"--- 3TF Pipeline ({self.profile.type.value}) Start: {self.symbol} ---")
        
        # 1. HTF Decision
        htf = self.logic.htf_decide(snapshot_htf)
        if not htf.allow:
            logger.info(f"HTF ({self.profile.htf.value}) Veto: {htf.reason}")
            return None
        logger.info(f"HTF Pass: {htf.bias} (Risk: {htf.risk_scale})")

        # 2. MTF Validation
        mtf = self.logic.mtf_validate(snapshot_mtf, htf)
        if not mtf.pass_structure:
            logger.info(f"MTF ({self.profile.mtf.value}) Veto: {mtf.reason}")
            return None
        logger.info(f"MTF Pass: Boost {mtf.quality_boost}")

        # 3. LTF Trigger
        ltf = self.logic.ltf_trigger(snapshot_ltf, htf, mtf)
        if not ltf.trigger:
            logger.info(f"LTF ({self.profile.ltf.value}) Veto: {ltf.reason}")
            return None
        
        # 4. Instruction Assembly
        final_direction = htf.bias
        if final_direction == "BOTH":
            final_direction = "LONG" if snapshot_ltf.directional_score > 0 else "SHORT"

        instruction = TradeInstruction(
            symbol=self.symbol,
            profile=self.profile.type.value,
            direction=final_direction,
            size_multiplier=htf.risk_scale,
            confidence=ltf.final_confidence,
            timestamp=snapshot_ltf.timestamp,
            logic_path=f"[{self.profile.htf.value}]{htf.reason} -> [{self.profile.mtf.value}]{mtf.reason} -> [{self.profile.ltf.value}]{ltf.reason}"
        )
        
        return instruction