"""
Capacity Controller for Alpha Factory

Phase 2: Signal Ranking & Capacity Control

Goal: Take the best trades first, not all allowed trades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)

@dataclass
class CapacityConfig:
    """Configuration for capacity control."""
    # Capacity limits
    max_trades_per_session: int = 10
    max_trades_per_symbol: int = 3
    max_trades_per_hour: int = 5
    max_exposure_per_session: float = 0.15  # 15% of capital
    
    # Ranking and selection
    ranking_method: str = 'ev'  # 'ev', 'probability', 'hybrid'
    min_rank_threshold: float = 0.3  # Minimum rank score

    ev_history_window: int = 500
    ev_norm_min_samples: int = 30
    ev_percentile_low: float = 50.0
    ev_percentile_high: float = 90.0
    
    # Time windows
    session_duration_hours: int = 8
    hour_window_minutes: int = 60
    
    # Exposure management
    max_correlated_exposure: float = 0.08  # 8% for correlated trades
    correlation_threshold: float = 0.7
    
    # Defer/drop logic
    defer_threshold: float = 0.5  # Below this rank, defer instead of drop
    max_deferred_trades: int = 20
    defer_timeout_minutes: int = 30

class TradeCandidate:
    """Trade candidate with ranking and capacity metadata."""
    
    def __init__(self, trade_data: Dict[str, Any]):
        self.symbol = trade_data.get('symbol', 'UNKNOWN')
        self.direction = trade_data.get('direction', 'HOLD')
        self.probability = trade_data.get('probability', 0.5)
        self.expected_value = trade_data.get('expected_value', 0)
        self.regime = trade_data.get('regime', 'neutral')
        self.confidence = trade_data.get('confidence', 0.5)
        self.position_size = trade_data.get('position_size', 0.02)
        self.timestamp = trade_data.get('timestamp', datetime.now())
        
        # Ranking metadata
        self.rank_score = 0.0
        self.rank_position = 0
        self.selection_reason = ''
        
        # Capacity metadata
        self.session_id = self._get_session_id()
        self.hour_bucket = self._get_hour_bucket()
        
        # Status tracking
        self.status = 'candidate'  # 'candidate', 'selected', 'deferred', 'dropped'
        self.defer_count = 0
        self.defer_timestamp = None
    
    def _get_session_id(self) -> str:
        """Get session identifier based on timestamp."""
        session_start = self.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return f"session_{session_start.strftime('%Y%m%d')}"
    
    def _get_hour_bucket(self) -> str:
        """Get hour bucket for time-based capacity."""
        hour = self.timestamp.hour
        return f"hour_{hour}"
    
    def calculate_rank_score(self, method: str = 'ev', ev_normalizer: Optional[Callable[[float], float]] = None) -> float:
        """Calculate ranking score based on method."""
        if method == 'ev':
            if ev_normalizer is not None:
                return ev_normalizer(float(self.expected_value))
            return min(1.0, self.expected_value / 50.0)
        elif method == 'probability':
            return self.probability
        elif method == 'hybrid':
            # Weighted combination
            ev_weight = 0.7
            prob_weight = 0.3
            if ev_normalizer is not None:
                ev_score = ev_normalizer(float(self.expected_value))
            else:
                ev_score = min(1.0, self.expected_value / 50.0)
            return ev_weight * ev_score + prob_weight * self.probability
        else:
            return 0.0
    
    def __lt__(self, other):
        """For heap sorting (highest priority first)."""
        return self.rank_score > other.rank_score  # Reverse for max-heap

class CapacityController:
    """Professional capacity control for Alpha Factory."""
    
    def __init__(self, config: CapacityConfig = None):
        self.config = config or CapacityConfig()
        
        # Capacity tracking
        self.session_trades = defaultdict(list)
        self.hourly_trades = defaultdict(list)
        self.symbol_trades = defaultdict(list)
        self.current_exposure = 0.0
        
        # Deferred trades queue
        self.deferred_trades = deque(maxlen=self.config.max_deferred_trades)

        self.ev_history = deque(maxlen=self.config.ev_history_window)
        
        # Correlation tracking
        self.symbol_correlations = {}
        
        # Performance tracking
        self.capacity_stats = {
            'total_candidates': 0,
            'selected_trades': 0,
            'deferred_trades': 0,
            'dropped_trades': 0,
            'capacity_rejections': 0
        }
        
        logger.info("Capacity controller initialized")

    def _ingest_ev_history(self, candidates: List[TradeCandidate]) -> None:
        for c in candidates:
            try:
                ev = float(c.expected_value)
            except Exception:
                continue

            if not np.isfinite(ev):
                continue
            self.ev_history.append(ev)

    def _normalize_ev(self, ev: float) -> float:
        if not np.isfinite(ev):
            return 0.0

        if len(self.ev_history) < self.config.ev_norm_min_samples:
            return float(np.clip(ev / 50.0, 0.0, 1.0))

        ev_values = np.asarray(self.ev_history, dtype=np.float64)
        p_low = float(np.percentile(ev_values, self.config.ev_percentile_low))
        p_high = float(np.percentile(ev_values, self.config.ev_percentile_high))

        denom = p_high - p_low
        if denom <= 0:
            return float(np.clip(ev / 50.0, 0.0, 1.0))

        score = (ev - p_low) / denom
        return float(np.clip(score, 0.0, 1.0))
    
    def update_symbol_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]):
        """Update symbol correlation matrix."""
        self.symbol_correlations = correlation_matrix
        logger.info(f"Updated correlations for {len(correlation_matrix)} symbols")
    
    def check_session_capacity(self, candidate: TradeCandidate) -> bool:
        """Check if candidate fits session capacity."""
        session_trades = self.session_trades[candidate.session_id]
        
        # Check trade count limit
        if len(session_trades) >= self.config.max_trades_per_session:
            logger.debug(f"Session capacity reached: {len(session_trades)} >= {self.config.max_trades_per_session}")
            return False
        
        # Check exposure limit
        if self.current_exposure + candidate.position_size > self.config.max_exposure_per_session:
            logger.debug(f"Exposure limit reached: {self.current_exposure + candidate.position_size:.3f} > {self.config.max_exposure_per_session}")
            return False
        
        return True
    
    def check_symbol_capacity(self, candidate: TradeCandidate) -> bool:
        """Check if candidate fits symbol capacity."""
        symbol_trades = self.symbol_trades[candidate.symbol]
        
        if len(symbol_trades) >= self.config.max_trades_per_symbol:
            logger.debug(f"Symbol capacity reached for {candidate.symbol}: {len(symbol_trades)} >= {self.config.max_trades_per_symbol}")
            return False
        
        return True
    
    def check_hourly_capacity(self, candidate: TradeCandidate) -> bool:
        """Check if candidate fits hourly capacity."""
        hourly_trades = self.hourly_trades[candidate.hour_bucket]
        
        if len(hourly_trades) >= self.config.max_trades_per_hour:
            logger.debug(f"Hourly capacity reached: {len(hourly_trades)} >= {self.config.max_trades_per_hour}")
            return False
        
        return True
    
    def check_correlation_capacity(self, candidate: TradeCandidate) -> bool:
        """Check correlation-based capacity constraints."""
        if not self.symbol_correlations or candidate.symbol not in self.symbol_correlations:
            return True
        
        # Calculate current correlated exposure
        correlated_exposure = 0.0
        candidate_correlations = self.symbol_correlations[candidate.symbol]
        
        for symbol, trades in self.symbol_trades.items():
            if symbol == candidate.symbol:
                correlated_exposure += sum(t.position_size for t in trades)
            elif symbol in candidate_correlations:
                correlation = candidate_correlations[symbol]
                if correlation >= self.config.correlation_threshold:
                    symbol_exposure = sum(t.position_size for t in trades)
                    correlated_exposure += symbol_exposure * correlation
        
        if correlated_exposure + candidate.position_size > self.config.max_correlated_exposure:
            logger.debug(f"Correlation limit reached: {correlated_exposure + candidate.position_size:.3f} > {self.config.max_correlated_exposure}")
            return False
        
        return True
    
    def rank_candidates(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """Rank trade candidates by configured method."""
        # Calculate rank scores
        for candidate in candidates:
            candidate.rank_score = candidate.calculate_rank_score(self.config.ranking_method, ev_normalizer=self._normalize_ev)
        
        # Sort by rank score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x.rank_score, reverse=True)
        
        # Assign rank positions
        for i, candidate in enumerate(ranked_candidates):
            candidate.rank_position = i + 1
        
        return ranked_candidates
    
    def process_deferred_trades(self) -> List[TradeCandidate]:
        """Process deferred trades that may now be eligible."""
        current_time = datetime.now()
        eligible_trades = []
        
        # Remove expired deferred trades
        while self.deferred_trades:
            deferred = self.deferred_trades[0]
            
            # Check timeout
            if (current_time - deferred.defer_timestamp).total_seconds() > self.config.defer_timeout_minutes * 60:
                self.deferred_trades.popleft()
                deferred.status = 'dropped'
                deferred.selection_reason = 'defer_timeout'
                self.capacity_stats['dropped_trades'] += 1
                continue
            
            # Check if still below threshold
            if deferred.rank_score >= self.config.min_rank_threshold:
                eligible_trades.append(deferred)
                self.deferred_trades.popleft()
            else:
                break
        
        return eligible_trades
    
    def select_trades(self, candidates: List[Dict[str, Any]], ev_optimizer: Optional[Any] = None) -> Dict[str, Any]:
        """
        Select optimal trades with capacity constraints.
        
        Args:
            candidates: List of trade candidate dictionaries
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Processing {len(candidates)} trade candidates")

        if ev_optimizer is not None:
            try:
                needs_ev = any(('expected_value' not in c or c.get('expected_value') is None) for c in candidates)
                if needs_ev and hasattr(ev_optimizer, 'rank_trades_by_ev'):
                    candidates = ev_optimizer.rank_trades_by_ev(candidates)
            except Exception as e:
                logger.error(f"EV optimizer integration error: {e}")
        
        # Convert to TradeCandidate objects
        trade_candidates = [TradeCandidate(c) for c in candidates]
        self.capacity_stats['total_candidates'] += len(trade_candidates)
        
        # Process any deferred trades first
        deferred_candidates = self.process_deferred_trades()
        trade_candidates.extend(deferred_candidates)

        self._ingest_ev_history(trade_candidates)
        
        # Rank all candidates
        ranked_candidates = self.rank_candidates(trade_candidates)
        
        # Filter by minimum threshold
        filtered_candidates = [c for c in ranked_candidates if c.rank_score >= self.config.min_rank_threshold]
        
        # Select trades with capacity constraints
        selected_trades = []
        deferred_trades = []
        dropped_trades = []
        
        for candidate in filtered_candidates:
            # Check all capacity constraints
            if (self.check_session_capacity(candidate) and
                self.check_symbol_capacity(candidate) and
                self.check_hourly_capacity(candidate) and
                self.check_correlation_capacity(candidate)):
                
                # Select trade
                candidate.status = 'selected'
                candidate.selection_reason = 'capacity_available'
                selected_trades.append(candidate)
                
                # Update capacity tracking
                self.session_trades[candidate.session_id].append(candidate)
                self.hourly_trades[candidate.hour_bucket].append(candidate)
                self.symbol_trades[candidate.symbol].append(candidate)
                self.current_exposure += candidate.position_size
                
                self.capacity_stats['selected_trades'] += 1
                
            else:
                # Decide to defer or drop
                if candidate.rank_score >= self.config.defer_threshold:
                    # Defer the trade
                    candidate.status = 'deferred'
                    candidate.selection_reason = 'capacity_full_deferred'
                    candidate.defer_count += 1
                    candidate.defer_timestamp = datetime.now()
                    deferred_trades.append(candidate)
                    self.deferred_trades.append(candidate)
                    self.capacity_stats['deferred_trades'] += 1
                else:
                    # Drop the trade
                    candidate.status = 'dropped'
                    candidate.selection_reason = 'capacity_full_dropped'
                    dropped_trades.append(candidate)
                    self.capacity_stats['dropped_trades'] += 1
        
        # Update capacity rejections
        capacity_rejections = len([c for c in (deferred_trades + dropped_trades) if 'capacity_full' in c.selection_reason])
        self.capacity_stats['capacity_rejections'] += capacity_rejections
        
        # Prepare results
        results = {
            'total_candidates': len(candidates),
            'ranked_candidates': len(ranked_candidates),
            'filtered_candidates': len(filtered_candidates),
            'selected_trades': len(selected_trades),
            'deferred_trades': len(deferred_trades),
            'dropped_trades': len(dropped_trades),
            'capacity_rejections': capacity_rejections,
            'selection_efficiency': len(selected_trades) / len(candidates) if candidates else 0,
            'selected_trades_data': [self._candidate_to_dict(t) for t in selected_trades],
            'deferred_trades_data': [self._candidate_to_dict(t) for t in deferred_trades],
            'capacity_utilization': {
                'session': len(self.session_trades.get(next(iter(self.session_trades.keys()), ''), [])),
                'max_session': self.config.max_trades_per_session,
                'exposure': self.current_exposure,
                'max_exposure': self.config.max_exposure_per_session
            }
        }
        
        logger.info(f"Selection complete: {len(selected_trades)} selected, {len(deferred_trades)} deferred, {len(dropped_trades)} dropped")
        
        return results
    
    def _candidate_to_dict(self, candidate: TradeCandidate) -> Dict[str, Any]:
        """Convert TradeCandidate back to dictionary."""
        return {
            'symbol': candidate.symbol,
            'direction': candidate.direction,
            'probability': candidate.probability,
            'expected_value': candidate.expected_value,
            'regime': candidate.regime,
            'confidence': candidate.confidence,
            'position_size': candidate.position_size,
            'rank_score': candidate.rank_score,
            'rank_position': candidate.rank_position,
            'status': candidate.status,
            'selection_reason': candidate.selection_reason,
            'defer_count': candidate.defer_count
        }
    
    def get_capacity_report(self) -> Dict[str, Any]:
        """Get comprehensive capacity utilization report."""
        current_time = datetime.now()
        
        # Calculate current utilization
        session_utilization = {}
        for session_id, trades in self.session_trades.items():
            session_utilization[session_id] = {
                'trade_count': len(trades),
                'max_trades': self.config.max_trades_per_session,
                'utilization': len(trades) / self.config.max_trades_per_session,
                'total_exposure': sum(t.position_size for t in trades)
            }
        
        # Symbol utilization
        symbol_utilization = {}
        for symbol, trades in self.symbol_trades.items():
            symbol_utilization[symbol] = {
                'trade_count': len(trades),
                'max_trades': self.config.max_trades_per_symbol,
                'utilization': len(trades) / self.config.max_trades_per_symbol
            }
        
        # Hourly utilization
        hourly_utilization = {}
        for hour_bucket, trades in self.hourly_trades.items():
            hourly_utilization[hour_bucket] = {
                'trade_count': len(trades),
                'max_trades': self.config.max_trades_per_hour,
                'utilization': len(trades) / self.config.max_trades_per_hour
            }
        
        return {
            'timestamp': current_time.isoformat(),
            'capacity_stats': self.capacity_stats,
            'current_exposure': self.current_exposure,
            'max_exposure': self.config.max_exposure_per_session,
            'exposure_utilization': self.current_exposure / self.config.max_exposure_per_session,
            'deferred_queue_size': len(self.deferred_trades),
            'session_utilization': session_utilization,
            'symbol_utilization': symbol_utilization,
            'hourly_utilization': hourly_utilization,
            'config': {
                'max_trades_per_session': self.config.max_trades_per_session,
                'max_trades_per_symbol': self.config.max_trades_per_symbol,
                'max_trades_per_hour': self.config.max_trades_per_hour,
                'max_exposure_per_session': self.config.max_exposure_per_session,
                'ranking_method': self.config.ranking_method
            }
        }
    
    def reset_session_capacity(self):
        """Reset capacity tracking for new session."""
        current_session = f"session_{datetime.now().strftime('%Y%m%d')}"
        
        # Clear old session data
        old_sessions = [s for s in self.session_trades.keys() if s != current_session]
        for session in old_sessions:
            del self.session_trades[session]
        
        # Reset exposure
        self.current_exposure = 0.0
        
        # Reset EV history to prevent stale data affecting new session rankings
        self.ev_history.clear()
        
        logger.info(f"Session capacity reset for {current_session}")
    
    def update_correlation_matrix(self, returns_data: pd.DataFrame):
        """Update correlation matrix from returns data."""
        try:
            correlation_matrix = returns_data.corr()
            
            # Convert to dictionary format
            corr_dict = {}
            for symbol in correlation_matrix.columns:
                corr_dict[symbol] = correlation_matrix[symbol].to_dict()
            
            self.update_symbol_correlations(corr_dict)
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
