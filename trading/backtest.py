# trading/backtest.py
"""
Backtesting execution engine.
"""
import logging
from typing import List, Dict, Optional, NamedTuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BacktestTrade(NamedTuple):
    """Record of a backtested trade."""
    ticket: int
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str
    volume: float
    entry_price: float
    exit_price: Optional[float]
    sl: float
    tp: float
    pnl: Optional[float]
    status: str  # 'OPEN', 'CLOSED_SL', 'CLOSED_TP', 'CLOSED_MANUAL'


@dataclass
class Position:
    """Open position in backtest."""
    ticket: int
    direction: str
    volume: float
    entry_price: float
    entry_time: datetime
    sl: float
    tp: float


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0  # $7 per lot round trip
    spread_pips: float = 1.0         # Simulated spread
    pip_value: float = 10.0          # $10 per pip per lot


class BacktestExecutor:
    """
    Simulated trade executor for backtesting.
    
    Tracks:
    - Virtual balance
    - Open positions
    - Trade history
    - Performance metrics
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        
        self.positions: List[Position] = []
        self.trade_history: List[BacktestTrade] = []
        
        self.ticket_counter = 1000
        self.current_price = 0.0
        self.current_time = datetime.now()
    
    def connect(self) -> bool:
        """
        Stub connect method for compatibility with MT5Connector interface.
        BacktestExecutor is always connected by definition.
        
        Returns:
            bool: Always True
        """
        return True
    
    def disconnect(self) -> bool:
        """
        Stub disconnect method for compatibility with MT5Connector interface.
        
        Returns:
            bool: Always True
        """
        return True
    
    def entry(
        self,
        signal: str,
        volume: float,
        sl: float,
        tp: float,
    ) -> Dict:
        """
        Simulate order entry.
        
        Returns:
            Dict with success status and details
        """
        if signal.upper() not in ('BUY', 'SELL'):
            return {'success': False, 'error': f'Invalid signal: {signal}'}
        
        self.ticket_counter += 1
        
        # Apply spread to entry
        if signal.upper() == 'BUY':
            entry_price = self.current_price + (self.config.spread_pips * 0.0001)
        else:
            entry_price = self.current_price - (self.config.spread_pips * 0.0001)
        
        position = Position(
            ticket=self.ticket_counter,
            direction=signal.upper(),
            volume=volume,
            entry_price=entry_price,
            entry_time=self.current_time,
            sl=sl,
            tp=tp,
        )
        
        self.positions.append(position)
        
        # Deduct commission
        commission = self.config.commission_per_lot * volume
        self.balance -= commission
        
        logger.info(
            f"[BACKTEST] Opened {signal} {volume} @ {entry_price:.5f} "
            f"(SL: {sl:.5f}, TP: {tp:.5f})"
        )
        
        return {
            'success': True,
            'ticket': self.ticket_counter,
            'price': entry_price,
            'volume': volume,
        }
    
    def update_price(self, price: float, time: Optional[datetime] = None):
        """
        Update current price and check for SL/TP hits.
        
        Args:
            price: Current market price
            time: Current time (uses now if not provided)
        """
        self.current_price = price
        if time:
            self.current_time = time
        
        # Check each position for SL/TP
        positions_to_close = []
        
        for pos in self.positions:
            close_reason = None
            exit_price = price
            
            if pos.direction == 'BUY':
                if price <= pos.sl:
                    close_reason = 'CLOSED_SL'
                    exit_price = pos.sl
                elif price >= pos.tp:
                    close_reason = 'CLOSED_TP'
                    exit_price = pos.tp
            else:  # SELL
                if price >= pos.sl:
                    close_reason = 'CLOSED_SL'
                    exit_price = pos.sl
                elif price <= pos.tp:
                    close_reason = 'CLOSED_TP'
                    exit_price = pos.tp
            
            if close_reason:
                positions_to_close.append((pos, close_reason, exit_price))
        
        # Close triggered positions
        for pos, reason, exit_price in positions_to_close:
            self._close_position(pos, reason, exit_price)
        
        # Update equity
        self._update_equity()
    
    def _close_position(self, pos: Position, reason: str, exit_price: float):
        """Close a position and record trade."""
        # Calculate P&L
        if pos.direction == 'BUY':
            pips = (exit_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - exit_price) / 0.0001
        
        pnl = pips * self.config.pip_value * pos.volume
        self.balance += pnl
        
        # Record trade
        trade = BacktestTrade(
            ticket=pos.ticket,
            entry_time=pos.entry_time,
            exit_time=self.current_time,
            direction=pos.direction,
            volume=pos.volume,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            sl=pos.sl,
            tp=pos.tp,
            pnl=pnl,
            status=reason,
        )
        self.trade_history.append(trade)
        
        # Remove from open positions
        self.positions.remove(pos)
        
        logger.info(
            f"[BACKTEST] Closed {pos.direction} @ {exit_price:.5f} "
            f"({reason}) P&L: {pnl:.2f}"
        )
    
    def _update_equity(self):
        """Calculate current equity including unrealized P&L."""
        unrealized_pnl = 0.0
        
        for pos in self.positions:
            if pos.direction == 'BUY':
                pips = (self.current_price - pos.entry_price) / 0.0001
            else:
                pips = (pos.entry_price - self.current_price) / 0.0001
            
            unrealized_pnl += pips * self.config.pip_value * pos.volume
        
        self.equity = self.balance + unrealized_pnl
    
    def close_all_positions(self):
        """Close all open positions at current price."""
        for pos in list(self.positions):
            self._close_position(pos, 'CLOSED_MANUAL', self.current_price)

    def close_position(self, ticket: int, reason: str = 'CLOSED_MANUAL') -> bool:
        """Close a single open position by ticket at current price."""
        try:
            ticket_int = int(ticket)
        except Exception:
            return False

        for pos in list(self.positions):
            if int(pos.ticket) == ticket_int:
                self._close_position(pos, reason, self.current_price)
                return True
        return False

    def close_position_at(self, ticket: int, exit_price: float, reason: str = 'CLOSED_MANUAL') -> bool:
        """Close a single open position by ticket at a specific price."""
        try:
            ticket_int = int(ticket)
            exit_price_f = float(exit_price)
        except Exception:
            return False

        for pos in list(self.positions):
            if int(pos.ticket) == ticket_int:
                self._close_position(pos, reason, exit_price_f)
                return True
        return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [
            {
                'ticket': p.ticket,
                'type': p.direction,
                'volume': p.volume,
                'price_open': p.entry_price,
                'entry_time': p.entry_time,
                'price_current': self.current_price,
                'sl': p.sl,
                'tp': p.tp,
                'profit': self._calc_unrealized_pnl(p),
            }
            for p in self.positions
        ]
    
    def _calc_unrealized_pnl(self, pos: Position) -> float:
        """Calculate unrealized P&L for position."""
        if pos.direction == 'BUY':
            pips = (self.current_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - self.current_price) / 0.0001
        return pips * self.config.pip_value * pos.volume
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history as list of dicts."""
        return [
            {
                'ticket': t.ticket,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'volume': t.volume,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'status': t.status,
            }
            for t in self.trade_history
        ]
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
            }
        
        trades = self.trade_history
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        total_wins = sum(winners) if winners else 0
        total_losses = abs(sum(losers)) if losers else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) if trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'total_pnl': sum(pnls),
            'average_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'largest_win': max(winners) if winners else 0,
            'largest_loss': min(losers) if losers else 0,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.config.initial_balance) / self.config.initial_balance * 100,
        }
    
    def get_account_info(self):
        """Mock account info for compatibility."""
        from trading.mt5_connector import AccountInfo
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0,
            free_margin=self.equity,
            profit=self.equity - self.config.initial_balance,
        )
    
    def get_symbol_info(self) -> Dict:
        """Mock symbol info for compatibility."""
        return {
            'point': 0.00001,
            'trade_tick_value': 1.0,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
        }
