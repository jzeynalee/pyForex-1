# trading/position_coordinator.py
"""
Position Coordinator for Multi-Style Trading.

Manages:
- Position tracking across all styles
- Exposure limits and correlation checks
- Opposing position prevention
- Aggregate P&L tracking
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from trading.style_config import TradingStyle, StyleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """Position tracked by coordinator."""
    ticket: int
    style: TradingStyle
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    volume: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    magic_number: int
    
    # Dynamic fields
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0  # For trailing
    lowest_price: float = 0.0
    
    # Status
    is_open: bool = True
    modified_sl: bool = False  # Moved to BE
    partial_closed: bool = False


@dataclass 
class StyleExposure:
    """Exposure metrics for a single style."""
    style: TradingStyle
    position_count: int = 0
    total_volume: float = 0.0
    long_volume: float = 0.0
    short_volume: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    trades_today: int = 0


@dataclass
class DailyStats:
    """Daily statistics for a style."""
    date: datetime
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0


class PositionCoordinator:
    """
    Coordinates positions across all trading styles.
    
    Responsibilities:
    - Track all open positions by style
    - Enforce exposure limits
    - Prevent opposing positions if configured
    - Calculate aggregate risk metrics
    - Manage daily trade counts
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        
        # Position tracking
        self.positions: Dict[int, TrackedPosition] = {}  # ticket -> position
        self.positions_by_style: Dict[TradingStyle, Set[int]] = {
            style: set() for style in TradingStyle
        }
        
        # Daily stats
        self.daily_stats: Dict[TradingStyle, DailyStats] = {}
        self._reset_daily_stats()
        
        # State
        self.last_reset_date = datetime.now().date()
        self.starting_balance: float = 0.0
        self.current_balance: float = 0.0
        self.peak_balance: float = 0.0
    
    def initialize(self, balance: float):
        """Initialize with account balance."""
        self.starting_balance = balance
        self.current_balance = balance
        self.peak_balance = balance
        logger.info(f"[COORDINATOR] Initialized with balance: ${balance:,.2f}")
    
    def _reset_daily_stats(self):
        """Reset daily statistics."""
        today = datetime.now()
        for style in TradingStyle:
            self.daily_stats[style] = DailyStats(date=today)
        self.last_reset_date = today.date()
    
    def _check_day_reset(self):
        """Check if day has changed and reset stats."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info("[COORDINATOR] New trading day - resetting daily stats")
            self._reset_daily_stats()
    
    # =========================================================================
    # POSITION REGISTRATION
    # =========================================================================
    
    def register_position(self, position: TrackedPosition) -> bool:
        """
        Register a new position.
        
        Returns:
            True if registered successfully
        """
        self._check_day_reset()
        
        if position.ticket in self.positions:
            logger.warning(f"Position {position.ticket} already registered")
            return False
        
        self.positions[position.ticket] = position
        self.positions_by_style[position.style].add(position.ticket)
        
        # Update daily stats
        self.daily_stats[position.style].trades_opened += 1
        
        logger.info(
            f"[COORDINATOR] Registered {position.style.value} position: "
            f"#{position.ticket} {position.direction} {position.volume} @ {position.entry_price:.5f}"
        )
        
        return True
    
    def close_position(
        self, 
        ticket: int, 
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
    ) -> Optional[TrackedPosition]:
        """
        Mark position as closed and update stats.
        
        Returns:
            The closed position or None
        """
        if ticket not in self.positions:
            logger.warning(f"Position {ticket} not found")
            return None
        
        position = self.positions[ticket]
        position.is_open = False
        if exit_price is None:
            exit_price = float(position.current_price or position.entry_price)
        position.current_price = float(exit_price)
        position.unrealized_pnl = pnl
        
        # Update style stats
        style = position.style
        stats = self.daily_stats[style]
        stats.trades_closed += 1
        stats.realized_pnl += pnl
        
        if pnl > 0:
            stats.wins += 1
        else:
            stats.losses += 1
        
        # Update balance
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Remove from tracking
        self.positions_by_style[style].discard(ticket)
        del self.positions[ticket]
        
        logger.info(
            f"[COORDINATOR] Closed {style.value} position #{ticket}: "
            f"P&L ${pnl:+.2f}"
        )
        
        return position
    
    def update_position(
        self, 
        ticket: int, 
        current_price: float,
        unrealized_pnl: float,
    ):
        """Update position with current market data."""
        if ticket not in self.positions:
            return
        
        pos = self.positions[ticket]
        pos.current_price = current_price
        pos.unrealized_pnl = unrealized_pnl
        
        # Track high/low for trailing stops
        if pos.direction == 'BUY':
            pos.highest_price = max(pos.highest_price or pos.entry_price, current_price)
        else:
            pos.lowest_price = min(pos.lowest_price or pos.entry_price, current_price)
    
    # =========================================================================
    # EXPOSURE CHECKS
    # =========================================================================
    
    def get_style_exposure(self, style: TradingStyle) -> StyleExposure:
        """Get current exposure for a style."""
        exposure = StyleExposure(style=style)
        
        for ticket in self.positions_by_style[style]:
            if ticket not in self.positions:
                continue
            pos = self.positions[ticket]
            if not pos.is_open:
                continue
            
            exposure.position_count += 1
            exposure.total_volume += pos.volume
            exposure.unrealized_pnl += pos.unrealized_pnl
            
            if pos.direction == 'BUY':
                exposure.long_volume += pos.volume
            else:
                exposure.short_volume += pos.volume
        
        # Add daily stats
        stats = self.daily_stats[style]
        exposure.realized_pnl_today = stats.realized_pnl
        exposure.trades_today = stats.trades_opened
        
        return exposure
    
    def get_total_exposure(self) -> Dict[str, float]:
        """Get aggregate exposure across all styles."""
        total = {
            'position_count': 0,
            'total_positions': 0,
            'total_volume': 0.0,
            'long_volume': 0.0,
            'short_volume': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_today': 0.0,
        }
        
        for style in TradingStyle:
            exp = self.get_style_exposure(style)
            total['position_count'] += exp.position_count
            total['total_positions'] += exp.position_count
            total['total_volume'] += exp.total_volume
            total['long_volume'] += exp.long_volume
            total['short_volume'] += exp.short_volume
            total['unrealized_pnl'] += exp.unrealized_pnl
            total['realized_pnl_today'] += exp.realized_pnl_today
        
        return total
    
    def can_open_position(
        self,
        style: TradingStyle,
        direction: str,
        volume: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Returns:
            (allowed, reason)
        """
        style_config = self.config.get_style_config(style)
        exposure = self.get_style_exposure(style)
        total_exposure = self.get_total_exposure()
        
        # Check 1: Style position limit
        if exposure.position_count >= style_config.max_positions:
            return False, f"limit: {style.value} max positions ({style_config.max_positions}) reached"
        
        # Check 2: Style daily trade limit
        if exposure.trades_today >= style_config.max_daily_trades:
            return False, f"{style.value} daily trade limit ({style_config.max_daily_trades}) reached"
        
        # Check 3: Total position limit
        if total_exposure['position_count'] >= self.config.max_total_positions:
            return False, f"Total position limit ({self.config.max_total_positions}) reached"
        
        # Check 4: Daily loss limit
        total_pnl = total_exposure['realized_pnl_today'] + total_exposure['unrealized_pnl']
        max_loss = self.starting_balance * self.config.max_daily_loss_pct
        if total_pnl < -max_loss:
            return False, f"Daily loss limit (${max_loss:.2f}) exceeded"
        
        # Check 5: Max drawdown
        current_equity = self.current_balance + total_exposure['unrealized_pnl']
        drawdown = (self.peak_balance - current_equity) / self.peak_balance
        if drawdown >= self.config.max_drawdown_pct:
            return False, f"Max drawdown ({self.config.max_drawdown_pct:.0%}) exceeded"
        
        # Check 6: Opposing positions
        if self.config.prevent_opposing_positions:
            opposing = self._has_opposing_position(direction)
            if opposing:
                return False, f"Opposing {opposing} position already open"
        
        return True, "OK"
    
    def _has_opposing_position(self, direction: str) -> Optional[str]:
        """Check if there's an opposing position in any style."""
        opposite = 'SELL' if direction == 'BUY' else 'BUY'
        
        for ticket, pos in self.positions.items():
            if pos.is_open and pos.direction == opposite:
                return f"{pos.style.value}"
        
        return None
    
    # =========================================================================
    # AGGREGATE SIGNALS
    # =========================================================================
    
    def get_aggregate_direction(self) -> Tuple[str, float]:
        """
        Get aggregate direction across all positions.
        
        Returns:
            (direction, strength) where strength is 0-1
        """
        long_vol = 0.0
        short_vol = 0.0
        
        for pos in self.positions.values():
            if not pos.is_open:
                continue
            if pos.direction == 'BUY':
                long_vol += pos.volume
            else:
                short_vol += pos.volume
        
        total = long_vol + short_vol
        if total == 0:
            return 'NEUTRAL', 0.0
        
        if long_vol > short_vol:
            strength = (long_vol - short_vol) / total
            return 'LONG', strength
        elif short_vol > long_vol:
            strength = (short_vol - long_vol) / total
            return 'SHORT', strength
        else:
            return 'NEUTRAL', 0.0
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def get_positions_needing_sl_update(self) -> List[Tuple[TrackedPosition, float]]:
        """
        Get positions that need SL moved to breakeven or trailing.
        
        Returns:
            List of (position, new_sl_price)
        """
        updates = []
        
        for pos in self.positions.values():
            if not pos.is_open:
                continue
            
            style_config = self.config.get_style_config(pos.style)
            
            # Calculate current R multiple
            if pos.direction == 'BUY':
                risk = pos.entry_price - pos.stop_loss
                reward = pos.current_price - pos.entry_price
            else:
                risk = pos.stop_loss - pos.entry_price
                reward = pos.entry_price - pos.current_price
            
            if risk <= 0:
                continue
            
            r_multiple = reward / risk
            
            # Check for breakeven move
            if not pos.modified_sl and r_multiple >= style_config.break_even_at_rr:
                if pos.direction == 'BUY':
                    new_sl = pos.entry_price + (risk * 0.1)  # Slightly above entry
                else:
                    new_sl = pos.entry_price - (risk * 0.1)
                
                updates.append((pos, new_sl))
                continue
            
            # Check for trailing stop
            if style_config.use_trailing_stop and pos.modified_sl:
                # Simple ATR-based trail (would need ATR passed in for full impl)
                trail_distance = risk * style_config.trailing_stop_atr
                
                if pos.direction == 'BUY' and pos.highest_price:
                    new_sl = pos.highest_price - trail_distance
                    if new_sl > pos.stop_loss:
                        updates.append((pos, new_sl))
                
                elif pos.direction == 'SELL' and pos.lowest_price:
                    new_sl = pos.lowest_price + trail_distance
                    if new_sl < pos.stop_loss:
                        updates.append((pos, new_sl))
        
        return updates
    
    def get_positions_for_partial_close(self) -> List[Tuple[TrackedPosition, float]]:
        """
        Get positions ready for partial close.
        
        Returns:
            List of (position, close_volume)
        """
        partials = []
        
        for pos in self.positions.values():
            if not pos.is_open or pos.partial_closed:
                continue
            
            style_config = self.config.get_style_config(pos.style)
            
            if style_config.partial_close_at_rr <= 0:
                continue
            
            # Calculate R multiple
            if pos.direction == 'BUY':
                risk = pos.entry_price - pos.stop_loss
                reward = pos.current_price - pos.entry_price
            else:
                risk = pos.stop_loss - pos.entry_price
                reward = pos.entry_price - pos.current_price
            
            if risk <= 0:
                continue
            
            r_multiple = reward / risk
            
            if r_multiple >= style_config.partial_close_at_rr:
                close_volume = pos.volume * style_config.partial_close_pct
                partials.append((pos, close_volume))
        
        return partials
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_status_report(self) -> Dict:
        """Generate comprehensive status report."""
        total = self.get_total_exposure()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'account': {
                'starting_balance': self.starting_balance,
                'current_balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'drawdown_pct': (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0,
            },
            'total_exposure': total,
            'by_style': {},
            'open_positions': [],
        }
        
        for style in TradingStyle:
            exp = self.get_style_exposure(style)
            stats = self.daily_stats[style]
            
            report['by_style'][style.value] = {
                'position_count': exp.position_count,
                'volume': exp.total_volume,
                'unrealized_pnl': exp.unrealized_pnl,
                'realized_pnl_today': exp.realized_pnl_today,
                'trades_today': exp.trades_today,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': stats.wins / (stats.wins + stats.losses) * 100 if (stats.wins + stats.losses) > 0 else 0,
            }
        
        for pos in self.positions.values():
            if pos.is_open:
                report['open_positions'].append({
                    'ticket': pos.ticket,
                    'style': pos.style.value,
                    'direction': pos.direction,
                    'volume': pos.volume,
                    'entry': pos.entry_price,
                    'current': pos.current_price,
                    'pnl': pos.unrealized_pnl,
                })
        
        return report
    
    def print_status(self):
        """Print formatted status to console."""
        report = self.get_status_report()
        
        print("\n" + "=" * 70)
        print("  POSITION COORDINATOR STATUS")
        print("=" * 70)
        
        acct = report['account']
        print(f"\n  Account:")
        print(f"    Balance: ${acct['current_balance']:,.2f} (Peak: ${acct['peak_balance']:,.2f})")
        print(f"    Drawdown: {acct['drawdown_pct']:.2f}%")
        
        total = report['total_exposure']
        print(f"\n  Total Exposure:")
        print(f"    Positions: {total['position_count']}")
        print(f"    Volume: {total['total_volume']:.2f} (L: {total['long_volume']:.2f}, S: {total['short_volume']:.2f})")
        print(f"    Unrealized P&L: ${total['unrealized_pnl']:+,.2f}")
        print(f"    Realized Today: ${total['realized_pnl_today']:+,.2f}")
        
        print(f"\n  By Style:")
        for style_name, data in report['by_style'].items():
            if data['position_count'] > 0 or data['trades_today'] > 0:
                print(f"    {style_name.upper()}:")
                print(f"      Positions: {data['position_count']}, Trades: {data['trades_today']}")
                print(f"      P&L: ${data['unrealized_pnl']:+,.2f} (realized: ${data['realized_pnl_today']:+,.2f})")
                print(f"      Win Rate: {data['win_rate']:.1f}%")
        
        if report['open_positions']:
            print(f"\n  Open Positions:")
            for pos in report['open_positions']:
                print(f"    #{pos['ticket']} [{pos['style']}] {pos['direction']} {pos['volume']} @ {pos['entry']:.5f} -> ${pos['pnl']:+.2f}")
    
    def get_positions_by_style(self, style: TradingStyle) -> List[TrackedPosition]:
        """Get all positions for a specific style."""
        return [pos for pos in self.positions.values() if pos.style == style]
    
    def get_daily_stats(self, style: TradingStyle = None):
        """Get daily statistics for a style or all styles."""
        if style:
            return self.daily_stats.get(style, DailyStats(date=datetime.now()))
        else:
            return {style.value: self.get_daily_stats(style) for style in TradingStyle}
        
        print("\n" + "=" * 70 + "\n")
