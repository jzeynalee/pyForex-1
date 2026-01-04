"""
Cross-Signal Intelligence for Alpha Factory

Phase 4: Cross-Signal Intelligence (Professional Scaling)

Goal: Extract hidden alpha from interactions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)

@dataclass
class SignalIntelligenceConfig:
    """Configuration for cross-signal intelligence."""
    # Signal concurrence detection
    concurrence_window: int = 5  # Minutes
    min_concurrence_signals: int = 2  # Minimum signals for boost
    concurrence_ev_boost: float = 0.20  # 20% EV boost
    
    # Contradiction detection
    contradiction_threshold: float = 0.3  # 30% EV penalty for contradictions
    max_contradictory_signals: int = 2
    
    # Evidence consistency scoring
    consistency_window: int = 10  # Signals for consistency check
    min_consistency_score: float = 0.6  # Minimum consistency
    consistency_ev_penalty: float = 0.15  # 15% penalty for inconsistency
    
    # Signal interaction weights
    bullish_bullish_boost: float = 0.10
    bearish_bearish_boost: float = 0.10
    bullish_bearish_penalty: float = 0.25
    
    # Temporal weighting
    recency_weight: float = 0.7  # Weight for recent signals
    historical_weight: float = 0.3  # Weight for historical patterns

class SignalInteraction:
    """Represents interaction between multiple signals."""
    
    def __init__(self, signals: List[Dict[str, Any]], interaction_type: str):
        self.signals = signals
        self.interaction_type = interaction_type  # 'concurrence', 'contradiction', 'consistency'
        self.timestamp = datetime.now()
        self.ev_adjustment = 0.0
        self.confidence_adjustment = 0.0
        self.interaction_score = 0.0
        self.reasoning = ''

class CrossSignalIntelligence:
    """Professional cross-signal intelligence for Alpha Factory."""
    
    def __init__(self, config: SignalIntelligenceConfig = None):
        self.config = config or SignalIntelligenceConfig()
        
        # Signal tracking
        self.signal_history = []
        self.interaction_history = []
        self.consistency_patterns = {}
        
        # Performance tracking
        self.interaction_performance = defaultdict(list)
        self.concurrence_success_rate = {}
        self.contradiction_impact = {}
        
        # Evidence consistency
        self.evidence_scores = defaultdict(float)
        self.consistency_history = []
        
        logger.info("Cross-signal intelligence initialized")
    
    def add_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Add a new signal to the intelligence system.
        
        Args:
            signal_data: Dictionary with signal information
        """
        signal = {
            'timestamp': signal_data.get('timestamp', datetime.now()),
            'symbol': signal_data.get('symbol', 'UNKNOWN'),
            'direction': signal_data.get('direction', 'HOLD'),
            'probability': signal_data.get('probability', 0.5),
            'expected_value': signal_data.get('expected_value', 0),
            'regime': signal_data.get('regime', 'neutral'),
            'confidence': signal_data.get('confidence', 0.5),
            'signal_id': len(self.signal_history)
        }
        
        self.signal_history.append(signal)
        
        # Keep history manageable
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]
        
        logger.debug(f"Signal added: {signal['symbol']} {signal['direction']} EV ${signal['expected_value']:.2f}")
    
    def detect_signal_concurrence(self, current_signal: Dict[str, Any]) -> List[SignalInteraction]:
        """
        Detect concurrent signals that boost EV.
        
        Args:
            current_signal: Current signal to check for concurrence
            
        Returns:
            List of concurrent signal interactions
        """
        interactions = []
        current_time = current_signal.get('timestamp', datetime.now())
        
        # Find recent signals for same symbol or correlated symbols
        recent_signals = []
        for signal in self.signal_history[-50:]:  # Check last 50 signals
            signal_time = signal['timestamp']
            time_diff = abs((current_time - signal_time).total_seconds() / 60)  # Minutes
            
            if time_diff <= self.config.concurrence_window and signal['signal_id'] != current_signal.get('signal_id', -1):
                # Check for same symbol or related signals
                if (signal['symbol'] == current_signal.get('symbol') or
                    self._are_symbols_related(signal['symbol'], current_signal.get('symbol'))):
                    recent_signals.append(signal)
        
        # Check for concurrence patterns
        if len(recent_signals) >= self.config.min_concurrence_signals:
            # Analyze signal alignment
            aligned_signals = [s for s in recent_signals if s['direction'] == current_signal.get('direction')]
            
            if len(aligned_signals) >= self.config.min_concurrence_signals:
                # Create concurrence interaction
                interaction = SignalInteraction(
                    [current_signal] + aligned_signals,
                    'concurrence'
                )
                
                # Calculate EV boost
                avg_ev = np.mean([s['expected_value'] for s in aligned_signals])
                ev_boost = avg_ev * self.config.concurrence_ev_boost
                interaction.ev_adjustment = ev_boost
                
                # Calculate confidence boost
                avg_confidence = np.mean([s['confidence'] for s in aligned_signals])
                confidence_boost = (avg_confidence - current_signal.get('confidence', 0.5)) * 0.5
                interaction.confidence_adjustment = confidence_boost
                
                # Calculate interaction score
                interaction.interaction_score = min(1.0, len(aligned_signals) / 5.0)  # Normalize to 0-1
                interaction.reasoning = f"Concurrence: {len(aligned_signals)} aligned signals in {self.config.concurrence_window}min window"
                
                interactions.append(interaction)
        
        return interactions
    
    def detect_structural_contradictions(self, current_signal: Dict[str, Any]) -> List[SignalInteraction]:
        """
        Detect structural contradictions between signals.
        
        Args:
            current_signal: Current signal to check for contradictions
            
        Returns:
            List of contradiction interactions
        """
        interactions = []
        current_time = current_signal.get('timestamp', datetime.now())
        
        # Find recent contradictory signals
        recent_signals = []
        for signal in self.signal_history[-20:]:  # Check last 20 signals
            signal_time = signal['timestamp']
            time_diff = abs((current_time - signal_time).total_seconds() / 60)  # Minutes
            
            if time_diff <= self.config.concurrence_window and signal['signal_id'] != current_signal.get('signal_id', -1):
                # Check for same symbol with opposite direction
                if (signal['symbol'] == current_signal.get('symbol') and
                    signal['direction'] != current_signal.get('direction')):
                    recent_signals.append(signal)
        
        # Check for contradictions
        if recent_signals:
            # Create contradiction interaction
            interaction = SignalInteraction(
                [current_signal] + recent_signals,
                'contradiction'
            )
            
            # Calculate EV penalty
            avg_ev = np.mean([s['expected_value'] for s in recent_signals])
            ev_penalty = avg_ev * self.config.contradiction_threshold
            interaction.ev_adjustment = -ev_penalty
            
            # Calculate confidence penalty
            avg_confidence = np.mean([s['confidence'] for s in recent_signals])
            confidence_penalty = (current_signal.get('confidence', 0.5) - avg_confidence) * 0.3
            interaction.confidence_adjustment = -abs(confidence_penalty)
            
            # Calculate interaction score (negative for contradictions)
            interaction.interaction_score = -min(1.0, len(recent_signals) / 3.0)
            interaction.reasoning = f"Contradiction: {len(recent_signals)} opposing signals for {current_signal.get('symbol')}"
            
            interactions.append(interaction)
        
        return interactions
    
    def calculate_evidence_consistency(self, current_signal: Dict[str, Any]) -> float:
        """
        Calculate evidence consistency score.
        
        Args:
            current_signal: Current signal
            
        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        symbol = current_signal.get('symbol', 'UNKNOWN')
        direction = current_signal.get('direction', 'HOLD')
        
        # Get recent signals for this symbol
        symbol_signals = [s for s in self.signal_history[-self.config.consistency_window:] 
                         if s['symbol'] == symbol]
        
        if len(symbol_signals) < 3:
            return 0.8  # Default consistency for insufficient data
        
        # Calculate direction consistency
        same_direction_signals = [s for s in symbol_signals if s['direction'] == direction]
        direction_consistency = len(same_direction_signals) / len(symbol_signals)
        
        # Calculate probability consistency
        probabilities = [s['probability'] for s in symbol_signals]
        current_prob = current_signal.get('probability', 0.5)
        
        if probabilities:
            prob_std = np.std(probabilities)
            prob_consistency = max(0, 1 - prob_std)  # Lower std = higher consistency
        else:
            prob_consistency = 0.5
        
        # Calculate EV consistency
        evs = [s['expected_value'] for s in symbol_signals]
        current_ev = current_signal.get('expected_value', 0)
        
        if evs:
            ev_std = np.std(evs)
            ev_consistency = max(0, 1 - (ev_std / np.mean(evs))) if np.mean(evs) > 0 else 0.5
        else:
            ev_consistency = 0.5
        
        # Combine consistency scores
        overall_consistency = (
            0.4 * direction_consistency +
            0.3 * prob_consistency +
            0.3 * ev_consistency
        )
        
        return overall_consistency
    
    def apply_interaction_adjustments(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all interaction adjustments to a signal.
        
        Args:
            signal: Original signal
            
        Returns:
            Adjusted signal with interaction effects
        """
        adjusted_signal = signal.copy()
        
        # Detect all interactions
        concurrence_interactions = self.detect_signal_concurrence(signal)
        contradiction_interactions = self.detect_structural_contradictions(signal)
        
        # Calculate evidence consistency
        consistency_score = self.calculate_evidence_consistency(signal)
        
        # Apply concurrence adjustments
        total_ev_boost = 0.0
        total_confidence_boost = 0.0
        interaction_reasons = []
        
        for interaction in concurrence_interactions:
            total_ev_boost += interaction.ev_adjustment
            total_confidence_boost += interaction.confidence_adjustment
            interaction_reasons.append(interaction.reasoning)
            self.interaction_history.append(interaction)
        
        # Apply contradiction adjustments
        total_ev_penalty = 0.0
        total_confidence_penalty = 0.0
        
        for interaction in contradiction_interactions:
            total_ev_penalty += abs(interaction.ev_adjustment)
            total_confidence_penalty += abs(interaction.confidence_adjustment)
            interaction_reasons.append(interaction.reasoning)
            self.interaction_history.append(interaction)
        
        # Apply consistency penalty if needed
        consistency_penalty = 0.0
        if consistency_score < self.config.min_consistency_score:
            consistency_penalty = signal.get('expected_value', 0) * self.config.consistency_ev_penalty
            interaction_reasons.append(f"Low consistency: {consistency_score:.2f}")
        
        # Calculate final adjustments
        net_ev_adjustment = total_ev_boost - total_ev_penalty - consistency_penalty
        net_confidence_adjustment = total_confidence_boost - total_confidence_penalty
        
        # Apply adjustments
        adjusted_signal['expected_value'] = max(0, signal.get('expected_value', 0) + net_ev_adjustment)
        adjusted_signal['confidence'] = max(0, min(1, signal.get('confidence', 0.5) + net_confidence_adjustment))
        
        # Add interaction metadata
        adjusted_signal['interaction_metadata'] = {
            'concurrence_boost': total_ev_boost,
            'contradiction_penalty': total_ev_penalty,
            'consistency_penalty': consistency_penalty,
            'net_ev_adjustment': net_ev_adjustment,
            'consistency_score': consistency_score,
            'interaction_reasons': interaction_reasons,
            'interactions_detected': len(concurrence_interactions) + len(contradiction_interactions)
        }
        
        logger.info(f"Signal adjusted: EV ${signal.get('expected_value', 0):.2f} → ${adjusted_signal['expected_value']:.2f} "
                   f"({net_ev_adjustment:+.2f}), Consistency: {consistency_score:.2f}")
        
        return adjusted_signal
    
    def _are_symbols_related(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are related (e.g., correlated currency pairs)."""
        # Simple correlation mapping for major pairs
        correlations = {
            'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],
            'GBPUSD': ['EURUSD', 'EURGBP'],
            'USDJPY': ['EURJPY', 'GBPJPY'],
            'AUDUSD': ['EURUSD', 'NZDUSD'],
            'USDCAD': ['EURCAD'],
        }
        
        return symbol2 in correlations.get(symbol1, [])
    
    def analyze_interaction_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of signal interactions.
        
        Returns:
            Dictionary with interaction performance analysis
        """
        if not self.interaction_history:
            return {'status': 'no_interaction_history'}
        
        # Analyze concurrence performance
        concurrence_interactions = [i for i in self.interaction_history if i.interaction_type == 'concurrence']
        contradiction_interactions = [i for i in self.interaction_history if i.interaction_type == 'contradiction']
        
        # Calculate success rates
        concurrence_success = len([i for i in concurrence_interactions if i.ev_adjustment > 0]) / len(concurrence_interactions) if concurrence_interactions else 0
        contradiction_impact = np.mean([abs(i.ev_adjustment) for i in contradiction_interactions]) if contradiction_interactions else 0
        
        # Analyze interaction frequency
        interaction_frequency = len(self.interaction_history) / len(self.signal_history) if self.signal_history else 0
        
        # Analyze consistency patterns
        avg_consistency = np.mean(list(self.evidence_scores.values())) if self.evidence_scores else 0.5
        
        return {
            'total_interactions': len(self.interaction_history),
            'concurrence_interactions': len(concurrence_interactions),
            'contradiction_interactions': len(contradiction_interactions),
            'concurrence_success_rate': concurrence_success,
            'contradiction_impact': contradiction_impact,
            'interaction_frequency': interaction_frequency,
            'average_consistency': avg_consistency,
            'config': {
                'concurrence_boost': self.config.concurrence_ev_boost,
                'contradiction_penalty': self.config.contradiction_threshold,
                'consistency_threshold': self.config.min_consistency_score
            }
        }
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive cross-signal intelligence report."""
        performance_analysis = self.analyze_interaction_performance()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_signals': len(self.signal_history),
            'total_interactions': len(self.interaction_history),
            'evidence_scores': dict(self.evidence_scores),
            'performance_analysis': performance_analysis,
            'recent_interactions': [
                {
                    'type': i.interaction_type,
                    'score': i.interaction_score,
                    'ev_adjustment': i.ev_adjustment,
                    'reasoning': i.reasoning,
                    'timestamp': i.timestamp.isoformat()
                }
                for i in self.interaction_history[-10:]  # Last 10 interactions
            ],
            'config': {
                'concurrence_window': self.config.concurrence_window,
                'min_concurrence_signals': self.config.min_concurrence_signals,
                'consistency_threshold': self.config.min_consistency_score
            }
        }
