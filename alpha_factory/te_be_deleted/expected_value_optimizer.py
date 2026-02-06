"""
Expected Value Optimizer for Alpha Factory

Phase 1: Replace Probability with Expected Value (Highest ROI)

Goal: Trade by money expectation, not confidence
EV = P(win) × AvgWin − (1 − P(win)) × AvgLoss
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class EVConfig:
    """Configuration for Expected Value optimization."""
    # EV calculation parameters
    ev_window: int = 100  # Trades for EV calculation
    min_ev_threshold: float = 5.0  # Minimum EV in dollars
    regime_conditioned_averages: bool = True
    
    # EV decay monitoring
    ev_decay_window: int = 50  # Window for decay detection
    ev_decay_threshold: float = 0.3  # 30% decay triggers alert
    
    # Ranking and selection
    max_trades_per_session: int = 10
    max_trades_per_symbol: int = 3
    ev_ranking_enabled: bool = True
    
    # Z-score monitoring
    z_score_window: int = 200  # Window for Z-score calculation
    z_score_threshold: float = 2.0  # 2 standard deviations

class ExpectedValueOptimizer:
    """Expected Value optimization for Alpha Factory."""
    
    def __init__(self, config: EVConfig = None):
        self.config = config or EVConfig()
        
        # Historical data storage
        self.trade_history = []
        self.ev_history = []
        self.regime_performance = {}
        
        # EV tracking
        self.current_ev = 0.0
        self.ev_decay_score = 0.0
        self.ev_z_score = 0.0
        
        # Regime-specific averages
        self.regime_averages = {
            'bullish': {'avg_win': 45, 'avg_loss': -18, 'win_rate': 0.62},
            'bearish': {'avg_win': 47, 'avg_loss': -19, 'win_rate': 0.60},
            'neutral': {'avg_win': 42, 'avg_loss': -17, 'win_rate': 0.58},
            'volatile': {'avg_win': 40, 'avg_loss': -20, 'win_rate': 0.55}
        }
        
        logger.info("Expected Value optimizer initialized")
    
    def calculate_regime_conditioned_averages(self, regime: str) -> Tuple[float, float, float]:
        """
        Calculate regime-conditioned win/loss averages and win rate.
        
        Args:
            regime: Current market regime
            
        Returns:
            Tuple of (avg_win, avg_loss, win_rate)
        """
        if not self.config.regime_conditioned_averages:
            # Use global averages
            recent_trades = self.trade_history[-self.config.ev_window:]
            if len(recent_trades) < 10:
                return 45.0, -18.0, 0.58  # Default values
            
            wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in recent_trades if t['pnl'] < 0]
            
            avg_win = np.mean(wins) if wins else 45.0
            avg_loss = np.mean(losses) if losses else -18.0
            win_rate = len(wins) / len(recent_trades) if recent_trades else 0.58
            
            return avg_win, avg_loss, win_rate
        
        # Use regime-specific averages
        regime_data = self.regime_averages.get(regime, self.regime_averages['neutral'])
        
        # Update with recent data if available
        regime_trades = [t for t in self.trade_history[-self.config.ev_window:] 
                        if t.get('regime') == regime]
        
        if len(regime_trades) >= 10:
            wins = [t['pnl'] for t in regime_trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in regime_trades if t['pnl'] < 0]
            
            if wins and losses:
                # Blend historical with recent (70/30)
                recent_avg_win = np.mean(wins)
                recent_avg_loss = np.mean(losses)
                recent_win_rate = len(wins) / len(regime_trades)
                
                avg_win = 0.7 * regime_data['avg_win'] + 0.3 * recent_avg_win
                avg_loss = 0.7 * regime_data['avg_loss'] + 0.3 * recent_avg_loss
                win_rate = 0.7 * regime_data['win_rate'] + 0.3 * recent_win_rate
                
                # Update stored averages
                self.regime_averages[regime] = {
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'win_rate': win_rate
                }
                
                return avg_win, avg_loss, win_rate
        
        return regime_data['avg_win'], regime_data['avg_loss'], regime_data['win_rate']
    
    def calculate_expected_value(self, probability: float, regime: str) -> float:
        """
        Calculate Expected Value for a trade.
        
        Args:
            probability: Win probability (calibrated)
            regime: Current market regime
            
        Returns:
            Expected Value in dollars
        """
        # Get regime-conditioned averages
        avg_win, avg_loss, historical_win_rate = self.calculate_regime_conditioned_averages(regime)
        
        # Calculate EV: EV = P(win) × AvgWin − (1 − P(win)) × AvgLoss
        ev = probability * avg_win - (1 - probability) * abs(avg_loss)
        
        return ev
    
    def calculate_ev_decay(self) -> float:
        """
        Calculate EV decay score over time.
        
        Returns:
            Decay score (0-1, higher = more decay)
        """
        if len(self.ev_history) < self.config.ev_decay_window:
            return 0.0
        
        # Compare recent EV to historical average
        recent_ev = self.ev_history[-self.config.ev_decay_window:]
        historical_ev = self.ev_history[:-self.config.ev_decay_window]
        
        if len(historical_ev) == 0:
            return 0.0
        
        recent_avg = np.mean(recent_ev)
        historical_avg = np.mean(historical_ev)
        
        if historical_avg <= 0:
            return 0.0
        
        decay = (historical_avg - recent_avg) / historical_avg
        return max(0, min(1, decay))
    
    def calculate_ev_z_score(self) -> float:
        """
        Calculate rolling Z-score of EV.
        
        Returns:
            Z-score (standard deviations from mean)
        """
        if len(self.ev_history) < self.config.z_score_window:
            return 0.0
        
        recent_ev = self.ev_history[-self.config.z_score_window:]
        current_ev = recent_ev[-1]
        
        mean_ev = np.mean(recent_ev[:-1])  # Exclude current value
        std_ev = np.std(recent_ev[:-1])
        
        if std_ev == 0:
            return 0.0
        
        z_score = (current_ev - mean_ev) / std_ev
        return z_score
    
    def rank_trades_by_ev(self, trade_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank trade candidates by Expected Value.
        
        Args:
            trade_candidates: List of trade candidates with probabilities
            
        Returns:
            Ranked list of trades by EV
        """
        if not self.config.ev_ranking_enabled:
            return trade_candidates
        
        # Calculate EV for each candidate
        for candidate in trade_candidates:
            probability = candidate.get('probability', 0.5)
            regime = candidate.get('regime', 'neutral')
            
            ev = self.calculate_expected_value(probability, regime)
            candidate['expected_value'] = ev
            
            # Add EV decay and Z-score context
            candidate['ev_decay_score'] = self.ev_decay_score
            candidate['ev_z_score'] = self.ev_z_score
        
        # Sort by EV (descending)
        ranked_trades = sorted(trade_candidates, key=lambda x: x.get('expected_value', 0), reverse=True)
        
        return ranked_trades
    
    def filter_trades_by_ev_threshold(self, trade_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter trades by minimum EV threshold.
        
        Args:
            trade_candidates: List of trade candidates
            
        Returns:
            Filtered list of trades
        """
        filtered_trades = []
        
        for candidate in trade_candidates:
            ev = candidate.get('expected_value', 0)
            
            if ev >= self.config.min_ev_threshold:
                filtered_trades.append(candidate)
            else:
                logger.info(f"Trade filtered by EV threshold: {ev:.2f} < {self.config.min_ev_threshold}")
        
        return filtered_trades
    
    def apply_capacity_constraints(self, trade_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply capacity constraints to trade selection.
        
        Args:
            trade_candidates: List of ranked trade candidates
            
        Returns:
            Final selection of trades
        """
        selected_trades = []
        symbol_counts = {}
        
        for candidate in trade_candidates:
            symbol = candidate.get('symbol', 'UNKNOWN')
            
            # Check per-symbol limit
            if symbol_counts.get(symbol, 0) >= self.config.max_trades_per_symbol:
                logger.info(f"Trade skipped due to symbol limit: {symbol}")
                continue
            
            # Check total limit
            if len(selected_trades) >= self.config.max_trades_per_session:
                logger.info(f"Trade skipped due to session limit: {len(selected_trades)}")
                break
            
            # Select trade
            selected_trades.append(candidate)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        return selected_trades
    
    def resolve_signal_collisions(self, trade_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve signal collisions and contradictions.
        
        Args:
            trade_candidates: List of trade candidates
            
        Returns:
            Resolved list of trades
        """
        resolved_trades = []
        symbol_signals = {}
        
        # Group by symbol
        for candidate in trade_candidates:
            symbol = candidate.get('symbol', 'UNKNOWN')
            direction = candidate.get('direction', 'HOLD')
            
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            
            symbol_signals[symbol].append(candidate)
        
        # Resolve each symbol
        for symbol, signals in symbol_signals.items():
            if len(signals) == 1:
                # Single signal, keep it
                resolved_trades.extend(signals)
            else:
                # Multiple signals for same symbol
                buy_signals = [s for s in signals if s.get('direction') == 'BUY']
                sell_signals = [s for s in signals if s.get('direction') == 'SELL']
                
                if buy_signals and sell_signals:
                    # Opposing signals - suppress both or downgrade
                    logger.warning(f"Opposing signals detected for {symbol}, suppressing")
                    continue  # Suppress both
                elif buy_signals:
                    # Multiple BUY signals - keep highest EV
                    best_buy = max(buy_signals, key=lambda x: x.get('expected_value', 0))
                    resolved_trades.append(best_buy)
                elif sell_signals:
                    # Multiple SELL signals - keep highest EV
                    best_sell = max(sell_signals, key=lambda x: x.get('expected_value', 0))
                    resolved_trades.append(best_sell)
        
        return resolved_trades
    
    def optimize_trade_selection(self, trade_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete trade selection optimization using EV.
        
        Args:
            trade_candidates: List of trade candidates
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing {len(trade_candidates)} trade candidates by EV")
        
        # Step 1: Calculate EV for all candidates
        ranked_trades = self.rank_trades_by_ev(trade_candidates)
        
        # Step 2: Filter by EV threshold
        ev_filtered = self.filter_trades_by_ev_threshold(ranked_trades)
        
        # Step 3: Resolve signal collisions
        collision_resolved = self.resolve_signal_collisions(ev_filtered)
        
        # Step 4: Apply capacity constraints
        final_selection = self.apply_capacity_constraints(collision_resolved)
        
        # Calculate selection metrics
        original_count = len(trade_candidates)
        filtered_count = len(ev_filtered)
        resolved_count = len(collision_resolved)
        final_count = len(final_selection)
        
        # Calculate average EV of selected trades
        avg_ev = np.mean([t.get('expected_value', 0) for t in final_selection]) if final_selection else 0
        
        results = {
            'original_candidates': original_count,
            'ev_filtered': filtered_count,
            'collision_resolved': resolved_count,
            'final_selection': final_count,
            'selected_trades': final_selection,
            'avg_expected_value': avg_ev,
            'ev_threshold': self.config.min_ev_threshold,
            'selection_efficiency': final_count / original_count if original_count > 0 else 0,
            'current_ev_decay': self.ev_decay_score,
            'current_ev_z_score': self.ev_z_score
        }
        
        logger.info(f"EV optimization complete: {final_count}/{original_count} trades selected")
        
        return results
    
    def update_trade_result(self, pnl: float, probability: float, regime: str, expected_value: float):
        """
        Update trade results for EV calculation.
        
        Args:
            pnl: Trade P&L result
            probability: Original win probability
            regime: Trade regime
            expected_value: Original EV calculation
        """
        trade_record = {
            'timestamp': datetime.now(),
            'pnl': pnl,
            'probability': probability,
            'regime': regime,
            'expected_value': expected_value,
            'ev_error': pnl - expected_value  # Difference between actual and expected
        }
        
        self.trade_history.append(trade_record)
        self.ev_history.append(expected_value)
        
        # Update EV metrics
        self.ev_decay_score = self.calculate_ev_decay()
        self.ev_z_score = self.calculate_ev_z_score()
        
        logger.info(f"Trade result updated: P&L=${pnl:.2f}, EV=${expected_value:.2f}, Error=${pnl - expected_value:.2f}")
    
    def get_ev_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive EV performance report."""
        if not self.trade_history:
            return {'status': 'no_trade_history'}
        
        # Calculate EV accuracy
        ev_errors = [t['ev_error'] for t in self.trade_history]
        actual_pnls = [t['pnl'] for t in self.trade_history]
        expected_values = [t['expected_value'] for t in self.trade_history]
        
        # Performance metrics
        total_actual = np.sum(actual_pnls)
        total_expected = np.sum(expected_values)
        ev_accuracy = 1 - abs(total_actual - total_expected) / abs(total_expected) if total_expected != 0 else 0
        
        # Regime performance
        regime_performance = {}
        for regime in ['bullish', 'bearish', 'neutral', 'volatile']:
            regime_trades = [t for t in self.trade_history if t.get('regime') == regime]
            if regime_trades:
                regime_pnls = [t['pnl'] for t in regime_trades]
                regime_evs = [t['expected_value'] for t in regime_trades]
                
                regime_performance[regime] = {
                    'trades': len(regime_trades),
                    'avg_pnl': np.mean(regime_pnls),
                    'avg_ev': np.mean(regime_evs),
                    'ev_accuracy': 1 - abs(np.sum(regime_pnls) - np.sum(regime_evs)) / abs(np.sum(regime_evs)) if np.sum(regime_evs) != 0 else 0
                }
        
        return {
            'total_trades': len(self.trade_history),
            'total_actual_pnl': total_actual,
            'total_expected_ev': total_expected,
            'ev_accuracy': ev_accuracy,
            'avg_ev_error': np.mean(ev_errors),
            'ev_error_std': np.std(ev_errors),
            'current_ev_decay': self.ev_decay_score,
            'current_ev_z_score': self.ev_z_score,
            'regime_performance': regime_performance,
            'regime_averages': self.regime_averages,
            'config': {
                'ev_threshold': self.config.min_ev_threshold,
                'ev_window': self.config.ev_window,
                'max_trades_per_session': self.config.max_trades_per_session
            }
        }
