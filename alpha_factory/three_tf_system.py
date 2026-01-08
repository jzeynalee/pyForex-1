# alpha_factory/three_tf_system.py
"""
3TF (Three-Timeframe) Institutional Architecture - Strict Mode

Core Principles:
1. HTF (Governor)  -> { allow, bias, risk_scale }
2. MTF (Validator) -> { pass, quality_boost }
3. LTF (Trigger)   -> { trigger, confidence }

Invariants:
- Snapshots are immutable and versioned.
- TFs never compute indicators.
- No composite scores (Sum(HTF+MTF+LTF) is forbidden).
- Logic is deterministic and profile-agnostic.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import numpy as np
import hashlib
from .trading_profiles import TradingProfile

logger = logging.getLogger(__name__)

# ==========================================
# 1. IMMUTABLE DATA STRUCTURES
# ==========================================

@dataclass(frozen=True)
class FeatureSnapshot:
    """
    First-class object representing raw intelligence at a specific time.
    Immutable. Versioned. Feature-agnostic.
    """
    timestamp: datetime
    timeframe: str
    directional_score: float  # ∈ [-1.0, +1.0]
    confidence: float         # ∈ [0.0, 1.0]
    stability: float          # ∈ [0.0, 1.0] feature consistency
    regime_flags: Dict[str, Any] = field(default_factory=dict)
    version: str = field(default="v0") # Hash of feature set + window

    def __post_init__(self):
        # Invariant checks on data integrity
        if not (-1.0 <= self.directional_score <= 1.0):
            raise ValueError(f"Directional score {self.directional_score} out of bounds")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence {self.confidence} out of bounds")

@dataclass(frozen=True)
class HTFDecision:
    """Output of H1 Participation Governor."""
    allow: bool
    bias: Literal["LONG", "SHORT", "BOTH", "NONE"]
    risk_scale: float
    reason: str

@dataclass(frozen=True)
class MTFDecision:
    """Output of M15 Structure Filter."""
    pass_structure: bool
    quality_boost: float
    reason: str

@dataclass(frozen=True)
class LTFSignal:
    """Output of M5 Execution Timing."""
    trigger: bool
    final_confidence: float
    reason: str

@dataclass(frozen=True)
class TradeInstruction:
    """Final executable instruction."""
    symbol: str
    profile: str
    direction: str
    size_multiplier: float
    confidence: float
    timestamp: datetime
    logic_path: str
    feature_version: str

# ==========================================
# 2. PURE LOGIC ENGINES
# ==========================================

class ThreeTFLogic:
    """
    Stateless logic containers. 
    Strictly implements the user's logic requirements.
    """

    @staticmethod
    def htf_decide(snapshot: FeatureSnapshot) -> HTFDecision:
        """
        HTF (Governor): Should we participate?
        INVARIANT: Returns { allow, bias, risk } ONLY. No timing.
        """
        # 1. Stability Gate
        if snapshot.stability < 0.4:
            return HTFDecision(False, "NONE", 0.0, f"HTF Stability {snapshot.stability:.2f} < 0.4")

        # 2. Confidence Gate
        if snapshot.confidence < 0.5:
            return HTFDecision(False, "NONE", 0.0, f"HTF Confidence {snapshot.confidence:.2f} < 0.5")

        # 3. Directional Bias Determination
        if snapshot.directional_score > 0.25:
            return HTFDecision(True, "LONG", 1.0, "Strong Bullish Bias")
        
        if snapshot.directional_score < -0.25:
            return HTFDecision(True, "SHORT", 1.0, "Strong Bearish Bias")

        # 4. Neutral/Range Mode (Allow Both)
        return HTFDecision(True, "BOTH", 0.6, "Neutral/Range Mode")

    @staticmethod
    def mtf_validate(snapshot: FeatureSnapshot, htf: HTFDecision) -> MTFDecision:
        """
        MTF (Structure): Is structure aligned?
        INVARIANT: Cannot create direction. Can only Veto or Boost.
        """
        # 1. HTF Veto Check (Gate)
        if not htf.allow:
            return MTFDecision(False, 0.0, "Blocked by HTF")

        # 2. MTF Confidence Gate
        if snapshot.confidence < 0.6:
            return MTFDecision(False, 0.0, f"MTF Confidence {snapshot.confidence:.2f} < 0.6")

        # 3. Structural Significance
        if abs(snapshot.directional_score) < 0.2:
            return MTFDecision(False, 0.0, "MTF Structure too weak")

        # 4. Bias Alignment Check
        if htf.bias != "BOTH":
            snapshot_sign = np.sign(snapshot.directional_score)
            bias_sign = 1.0 if htf.bias == "LONG" else -1.0
            
            if snapshot_sign != 0 and snapshot_sign != bias_sign:
                return MTFDecision(False, 0.0, f"Structure ({snapshot_sign}) contradicts HTF {htf.bias}")

        return MTFDecision(True, 0.25, "Structure Aligned")

    @staticmethod
    def ltf_trigger(snapshot: FeatureSnapshot, htf: HTFDecision, mtf: MTFDecision) -> LTFSignal:
        """
        LTF (Timing): Is timing favorable now?
        INVARIANT: Pure timing. No logic changes.
        """
        # 1. MTF Veto Check (Gate)
        if not mtf.pass_structure:
            return LTFSignal(False, 0.0, "Blocked by MTF Structure")

        # 2. LTF Confidence Gate
        if snapshot.confidence < 0.65:
            return LTFSignal(False, 0.0, f"LTF Confidence {snapshot.confidence:.2f} < 0.65")

        # 3. Micro-Stability Check
        if snapshot.stability < 0.5:
            return LTFSignal(False, 0.0, f"LTF Micro-instability {snapshot.stability:.2f}")

        # 4. Final Confidence Assembly
        # Base confidence + Structure quality boost (No composite score blending!)
        final_conf = min(1.0, snapshot.confidence + mtf.quality_boost)
        
        return LTFSignal(True, final_conf, "Execution Triggered")

# ==========================================
# 3. FEATURE ADAPTER (BRIDGE)
# ==========================================

class FeatureAdapter:
    @staticmethod
    def create_snapshot(
        timestamp: datetime,
        timeframe: str,
        decision_signal: Dict[str, Any], 
        causality_results: Dict[str, Any],
        market_regime: str,
        feature_version: str = "v1.0"
    ) -> FeatureSnapshot:
        """
        Converts raw AlphaFactory output into a strict FeatureSnapshot.
        Enforces decoupling of feature calculation from execution logic.
        """
        
        # Normalize direction score from raw signal
        base_score = decision_signal.get('confidence', 0.5)
        direction = decision_signal.get('decision', 'HOLD')
        
        if direction == 'BUY':
            dir_score = base_score
        elif direction == 'SELL':
            dir_score = -base_score
        else:
            dir_score = 0.0
            
        # Calculate stability from causality metadata
        stability = 0.8
        if 'stationarity_analysis' in causality_results:
            stat_data = causality_results['stationarity_analysis']
            if stat_data.get('features_checked', 0) > 0:
                stability = stat_data['stationary_features'] / stat_data['features_checked']

        # Generate version hash if not provided (ensures reproducibility)
        if feature_version == "v1.0":
            content_str = f"{timeframe}_{market_regime}_{len(causality_results)}"
            feature_version = hashlib.md5(content_str.encode()).hexdigest()[:8]

        return FeatureSnapshot(
            timestamp=timestamp,
            timeframe=timeframe,
            directional_score=dir_score,
            confidence=decision_signal.get('confidence', 0.0),
            stability=stability,
            regime_flags={'regime': market_regime},
            version=feature_version
        )

# ==========================================
# 4. ORCHESTRATOR WITH INVARIANTS
# ==========================================

class ThreeTFOrchestrator:
    """
    Manages the lifecycle of a 3TF decision process.
    Enforces architectural invariants via assertions.
    """
    
    def __init__(self, symbol: str, profile: TradingProfile):
        self.symbol = symbol
        self.profile = profile
        self.logic = ThreeTFLogic()
    
    def process_3tf(self, 
                    snapshot_htf: FeatureSnapshot, 
                    snapshot_mtf: FeatureSnapshot, 
                    snapshot_ltf: FeatureSnapshot) -> Optional[TradeInstruction]:
        """
        Execute the 3TF pipeline with strict invariant checks.
        """
        
        # --- Pre-Condition Invariants ---
        assert snapshot_htf.timeframe == self.profile.htf.value, "HTF Snapshot mismatch"
        assert snapshot_mtf.timeframe == self.profile.mtf.value, "MTF Snapshot mismatch"
        assert snapshot_ltf.timeframe == self.profile.ltf.value, "LTF Snapshot mismatch"
        assert snapshot_htf.version == snapshot_mtf.version == snapshot_ltf.version, "Feature Version Mismatch"

        logger.info(f"--- 3TF Pipeline ({self.profile.type.value}) Start: {self.symbol} ---")
        
        # 1. HTF Decision
        htf = self.logic.htf_decide(snapshot_htf)
        
        # HTF Invariants
        assert isinstance(htf.allow, bool), "HTF allow must be bool"
        assert htf.bias in ["LONG", "SHORT", "BOTH", "NONE"], "Invalid HTF Bias"
        
        if not htf.allow:
            logger.info(f"HTF Veto: {htf.reason}")
            return None
        logger.info(f"HTF Pass: {htf.bias} (Risk: {htf.risk_scale})")

        # 2. MTF Validation
        mtf = self.logic.mtf_validate(snapshot_mtf, htf)
        
        # MTF Invariants
        assert not (not htf.allow and mtf.pass_structure), "MTF passed despite HTF veto"
        
        if not mtf.pass_structure:
            logger.info(f"MTF Veto: {mtf.reason}")
            return None
        logger.info(f"MTF Pass: Boost {mtf.quality_boost}")

        # 3. LTF Trigger
        ltf = self.logic.ltf_trigger(snapshot_ltf, htf, mtf)
        
        # LTF Invariants
        assert not (not mtf.pass_structure and ltf.trigger), "LTF triggered despite MTF veto"
        
        if not ltf.trigger:
            logger.info(f"LTF Veto: {ltf.reason}")
            return None
        
        # 4. Instruction Assembly
        # Resolve Direction: Originates at HTF. If BOTH, collapses to LTF sign.
        final_direction = htf.bias
        if final_direction == "BOTH":
            # If HTF permitted both, we use the LTF directional score sign to resolve immediate execution side
            if snapshot_ltf.directional_score > 0:
                final_direction = "LONG"
            elif snapshot_ltf.directional_score < 0:
                final_direction = "SHORT"
            else:
                logger.warning("HTF=BOTH but LTF neutral. No trade.")
                return None

        instruction = TradeInstruction(
            symbol=self.symbol,
            profile=self.profile.type.value,
            direction=final_direction,
            size_multiplier=htf.risk_scale,
            confidence=ltf.final_confidence,
            timestamp=snapshot_ltf.timestamp,
            logic_path=f"[{self.profile.htf.value}]{htf.reason} -> [{self.profile.mtf.value}]{mtf.reason} -> [{self.profile.ltf.value}]{ltf.reason}",
            feature_version=snapshot_htf.version
        )
        
        return instruction