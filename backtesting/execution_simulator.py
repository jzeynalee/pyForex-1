"""
Realistic MT5 Execution Simulator
==================================

Simulates realistic order execution with:
- Market latency
- Slippage distributions
- Requotes
- Partial fills
- Broker constraints
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage distribution models."""
    NONE = "none"
    FIXED = "fixed"
    NORMAL = "normal"
    VOLATILITY_BASED = "volatility_based"
    REALISTIC = "realistic"  # Combines multiple factors


class LatencyModel(Enum):
    """Latency simulation models."""
    NONE = "none"
    FIXED = "fixed"
    NORMAL = "normal"
    REALISTIC = "realistic"  # Time-of-day dependent


class OrderStatus(Enum):
    """Order execution status."""
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    REJECTED = "rejected"
    REQUOTED = "requoted"


@dataclass
class ExecutionConfig:
    """Configuration for execution simulator."""
    # Initial capital
    initial_balance: float = 10000.0
    
    # Costs
    commission_per_lot: float = 7.0
    base_spread_pips: float = 1.0
    
    # Slippage
    slippage_model: SlippageModel = SlippageModel.REALISTIC
    slippage_mean_pips: float = 0.3
    slippage_std_pips: float = 0.5
    slippage_max_pips: float = 5.0
    
    # Latency
    latency_model: LatencyModel = LatencyModel.REALISTIC
    latency_mean_ms: float = 50
    latency_std_ms: float = 30
    latency_max_ms: float = 500
    
    # Execution quality
    requote_probability: float = 0.02  # 2% chance
    partial_fill_probability: float = 0.01  # 1% chance
    rejection_probability: float = 0.005  # 0.5% chance
    
    # Broker constraints
    min_lot_size: float = 0.01
    max_lot_size: float = 100.0
    lot_step: float = 0.01
    min_stop_distance_pips: float = 5.0
    max_leverage: float = 100.0
    
    # Market impact (for large orders)
    enable_market_impact: bool = True
    market_impact_factor: float = 0.1  # pips per lot
    
    # Spread widening
    enable_spread_widening: bool = True
    spread_widening_news_factor: float = 3.0
    spread_widening_low_liquidity_factor: float = 2.0


class ExecutionResult(NamedTuple):
    """Result of order execution attempt."""
    success: bool
    status: OrderStatus
    ticket: Optional[int]
    filled_price: Optional[float]
    filled_volume: float
    requested_volume: float
    slippage_pips: float
    latency_ms: float
    spread_pips: float
    commission: float
    error_message: Optional[str]
    metadata: Dict = {}


@dataclass
class Position:
    """Open position."""
    ticket: int
    direction: str
    volume: float
    entry_price: float
    entry_time: datetime
    sl: float
    tp: float
    commission_paid: float = 0.0
    
    def unrealized_pnl(self, current_price: float, pip_value: float = 10.0) -> float:
        """Calculate unrealized P&L."""
        if self.direction == 'BUY':
            pips = (current_price - self.entry_price) / 0.0001
        else:
            pips = (self.entry_price - current_price) / 0.0001
        return pips * pip_value * self.volume


@dataclass
class Trade:
    """Closed trade record."""
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    commission: float
    slippage_entry_pips: float
    slippage_exit_pips: float
    exit_reason: str  # 'SL', 'TP', 'MANUAL'


class RealisticExecutionSimulator:
    """
    Realistic execution simulator for backtesting.
    
    Features:
    - Realistic slippage based on volatility and liquidity
    - Market latency simulation
    - Requotes and rejections
    - Partial fills for large orders
    - Spread widening during news/low liquidity
    - Market impact for large orders
    - Broker constraint enforcement
    
    Usage:
        config = ExecutionConfig(
            initial_balance=10000,
            slippage_model=SlippageModel.REALISTIC
        )
        
        simulator = RealisticExecutionSimulator(config)
        simulator.initialize(10000)
        
        result = simulator.entry('BUY', 0.1, sl=1.0950, tp=1.1050)
        
        simulator.update_price(1.1000, datetime.now())
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        
        # Account state
        self.balance: float = 0.0
        self.equity: float = 0.0
        self.initial_balance: float = 0.0
        
        # Positions and trades
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        
        # Market state
        self.current_price: float = 0.0
        self.current_time: datetime = datetime.now()
        self.current_spread_pips: float = self.config.base_spread_pips
        self.current_volatility: float = 0.001  # ATR or similar
        
        # Execution tracking
        self.ticket_counter: int = 10000
        self.total_commission_paid: float = 0.0
        self.total_slippage_pips: float = 0.0
        self.execution_stats: Dict = {
            'total_orders': 0,
            'filled': 0,
            'rejected': 0,
            'requoted': 0,
            'partial_fills': 0
        }
        
        logger.info("RealisticExecutionSimulator initialized")
    
    def initialize(self, initial_balance: float):
        """Initialize simulator with starting balance."""
        self.balance = initial_balance
        self.equity = initial_balance
        self.initial_balance = initial_balance
        logger.info(f"Simulator initialized with balance: ${initial_balance:,.2f}")
    
    def connect(self) -> bool:
        """Compatibility method."""
        return True
    
    def disconnect(self) -> bool:
        """Compatibility method."""
        return True
    
    def entry(
        self,
        signal: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = ""
    ) -> Dict:
        """
        Execute market order entry.
        
        Args:
            signal: 'BUY' or 'SELL'
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
        
        Returns:
            Dict with execution result
        """
        self.execution_stats['total_orders'] += 1
        
        # Validate inputs
        validation_error = self._validate_order(signal, volume, sl, tp)
        if validation_error:
            self.execution_stats['rejected'] += 1
            return {
                'success': False,
                'error': validation_error,
                'ticket': None,
                'price': None,
                'volume': volume
            }
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        
        # Simulate execution quality issues
        if random.random() < self.config.rejection_probability:
            self.execution_stats['rejected'] += 1
            return {
                'success': False,
                'error': 'Order rejected by broker',
                'ticket': None,
                'price': None,
                'volume': volume
            }
        
        if random.random() < self.config.requote_probability:
            self.execution_stats['requoted'] += 1
            # Requote with slightly worse price
            requote_slippage = random.uniform(0.5, 2.0)
            logger.warning(f"Order requoted with {requote_slippage:.1f} pips slippage")
        
        # Calculate execution price with slippage
        slippage_pips = self._calculate_slippage(volume, signal)
        execution_price = self._apply_slippage(
            self.current_price,
            signal,
            slippage_pips
        )
        
        # Check for partial fill
        filled_volume = volume
        if volume > 1.0 and random.random() < self.config.partial_fill_probability:
            filled_volume = volume * random.uniform(0.5, 0.9)
            filled_volume = round(filled_volume / self.config.lot_step) * self.config.lot_step
            self.execution_stats['partial_fills'] += 1
            logger.warning(f"Partial fill: {filled_volume}/{volume} lots")
        
        # Calculate commission
        commission = self.config.commission_per_lot * filled_volume
        self.balance -= commission
        self.total_commission_paid += commission
        
        # Create position
        self.ticket_counter += 1
        position = Position(
            ticket=self.ticket_counter,
            direction=signal.upper(),
            volume=filled_volume,
            entry_price=execution_price,
            entry_time=self.current_time,
            sl=sl,
            tp=tp,
            commission_paid=commission
        )
        
        self.positions.append(position)
        self.execution_stats['filled'] += 1
        self.total_slippage_pips += abs(slippage_pips)
        
        logger.info(
            f"[EXEC] {signal} {filled_volume:.2f} @ {execution_price:.5f} "
            f"(slippage: {slippage_pips:.1f} pips, latency: {latency_ms:.0f}ms)"
        )
        
        return {
            'success': True,
            'ticket': self.ticket_counter,
            'price': execution_price,
            'volume': filled_volume,
            'slippage_pips': slippage_pips,
            'commission': commission
        }
    
    def update_price(self, price: float, time: Optional[datetime] = None):
        """
        Update current market price and check for SL/TP hits.
        
        Args:
            price: Current market price
            time: Current time
        """
        self.current_price = price
        if time:
            self.current_time = time
        
        # Update spread based on time and volatility
        self._update_spread()
        
        # Check positions for SL/TP
        positions_to_close = []
        
        for pos in self.positions:
            close_reason = None
            exit_price = price
            
            if pos.direction == 'BUY':
                if price <= pos.sl:
                    close_reason = 'SL'
                    exit_price = pos.sl
                elif price >= pos.tp:
                    close_reason = 'TP'
                    exit_price = pos.tp
            else:  # SELL
                if price >= pos.sl:
                    close_reason = 'SL'
                    exit_price = pos.sl
                elif price <= pos.tp:
                    close_reason = 'TP'
                    exit_price = pos.tp
            
            if close_reason:
                positions_to_close.append((pos, close_reason, exit_price))
        
        # Close triggered positions
        for pos, reason, exit_price in positions_to_close:
            self._close_position(pos, reason, exit_price)
        
        # Update equity
        self._update_equity()
    
    def close_all_positions(self):
        """Close all open positions at current price."""
        for pos in list(self.positions):
            self._close_position(pos, 'MANUAL', self.current_price)
    
    def _validate_order(
        self,
        signal: str,
        volume: float,
        sl: float,
        tp: float
    ) -> Optional[str]:
        """Validate order parameters."""
        # Check signal
        if signal.upper() not in ('BUY', 'SELL'):
            return f"Invalid signal: {signal}"
        
        # Check volume
        if volume < self.config.min_lot_size:
            return f"Volume {volume} below minimum {self.config.min_lot_size}"
        if volume > self.config.max_lot_size:
            return f"Volume {volume} above maximum {self.config.max_lot_size}"
        
        # Check lot step
        if round(volume / self.config.lot_step) * self.config.lot_step != volume:
            return f"Volume {volume} not multiple of lot step {self.config.lot_step}"
        
        # Check stop distance
        if signal.upper() == 'BUY':
            sl_distance_pips = (self.current_price - sl) / 0.0001
            tp_distance_pips = (tp - self.current_price) / 0.0001
        else:
            sl_distance_pips = (sl - self.current_price) / 0.0001
            tp_distance_pips = (self.current_price - tp) / 0.0001
        
        if sl_distance_pips < self.config.min_stop_distance_pips:
            return f"SL too close: {sl_distance_pips:.1f} pips < {self.config.min_stop_distance_pips}"
        
        # Check margin
        required_margin = self._calculate_required_margin(volume)
        if required_margin > self.equity:
            return f"Insufficient margin: required {required_margin:.2f}, available {self.equity:.2f}"
        
        return None
    
    def _calculate_required_margin(self, volume: float) -> float:
        """Calculate required margin for position."""
        contract_size = 100000  # Standard lot
        position_value = volume * contract_size * self.current_price
        return position_value / self.config.max_leverage
    
    def _simulate_latency(self) -> float:
        """Simulate order execution latency."""
        if self.config.latency_model == LatencyModel.NONE:
            return 0.0
        elif self.config.latency_model == LatencyModel.FIXED:
            return self.config.latency_mean_ms
        elif self.config.latency_model == LatencyModel.NORMAL:
            latency = np.random.normal(
                self.config.latency_mean_ms,
                self.config.latency_std_ms
            )
            return max(0, min(latency, self.config.latency_max_ms))
        else:  # REALISTIC
            # Higher latency during high volatility or news
            base_latency = self.config.latency_mean_ms
            
            # Time of day effect (higher during session opens)
            hour = self.current_time.hour
            if hour in [8, 9, 13, 14]:  # London/NY opens
                base_latency *= 1.5
            
            latency = np.random.normal(base_latency, self.config.latency_std_ms)
            return max(10, min(latency, self.config.latency_max_ms))
    
    def _calculate_slippage(self, volume: float, direction: str) -> float:
        """Calculate slippage in pips."""
        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0
        elif self.config.slippage_model == SlippageModel.FIXED:
            return self.config.slippage_mean_pips
        elif self.config.slippage_model == SlippageModel.NORMAL:
            slippage = np.random.normal(
                self.config.slippage_mean_pips,
                self.config.slippage_std_pips
            )
            return max(0, min(abs(slippage), self.config.slippage_max_pips))
        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Scale slippage by volatility
            vol_factor = self.current_volatility / 0.001  # Normalize
            base_slippage = self.config.slippage_mean_pips * vol_factor
            slippage = np.random.normal(base_slippage, self.config.slippage_std_pips)
            return max(0, min(abs(slippage), self.config.slippage_max_pips))
        else:  # REALISTIC
            # Combine multiple factors
            base_slippage = self.config.slippage_mean_pips
            
            # Volatility factor
            vol_factor = max(0.5, min(3.0, self.current_volatility / 0.001))
            base_slippage *= vol_factor
            
            # Market impact (larger orders get worse fills)
            if self.config.enable_market_impact and volume > 0.1:
                impact = (volume - 0.1) * self.config.market_impact_factor
                base_slippage += impact
            
            # Time of day (worse during low liquidity)
            hour = self.current_time.hour
            if hour < 6 or hour > 22:  # Low liquidity hours
                base_slippage *= 1.5
            
            # Add randomness
            slippage = np.random.normal(base_slippage, self.config.slippage_std_pips)
            return max(0, min(abs(slippage), self.config.slippage_max_pips))
    
    def _apply_slippage(
        self,
        price: float,
        direction: str,
        slippage_pips: float
    ) -> float:
        """Apply slippage to execution price."""
        slippage_price = slippage_pips * 0.0001
        
        if direction.upper() == 'BUY':
            # Buying - slippage makes price worse (higher)
            return price + slippage_price + (self.current_spread_pips * 0.0001)
        else:
            # Selling - slippage makes price worse (lower)
            return price - slippage_price - (self.current_spread_pips * 0.0001)
    
    def _update_spread(self):
        """Update current spread based on market conditions."""
        if not self.config.enable_spread_widening:
            self.current_spread_pips = self.config.base_spread_pips
            return
        
        spread = self.config.base_spread_pips
        
        # Widen during high volatility
        if self.current_volatility > 0.002:
            spread *= 1.5
        
        # Widen during low liquidity hours
        hour = self.current_time.hour
        if hour < 6 or hour > 22:
            spread *= self.config.spread_widening_low_liquidity_factor
        
        # Widen during session opens (simulating news)
        if hour in [8, 9, 13, 14]:
            if random.random() < 0.1:  # 10% chance of news event
                spread *= self.config.spread_widening_news_factor
        
        self.current_spread_pips = min(spread, self.config.base_spread_pips * 5)
    
    def _close_position(self, pos: Position, reason: str, exit_price: float):
        """Close position and record trade."""
        # Apply exit slippage
        exit_slippage_pips = self._calculate_slippage(pos.volume, pos.direction)
        actual_exit_price = self._apply_slippage(exit_price, pos.direction, exit_slippage_pips)
        
        # Calculate P&L
        if pos.direction == 'BUY':
            pips = (actual_exit_price - pos.entry_price) / 0.0001
        else:
            pips = (pos.entry_price - actual_exit_price) / 0.0001
        
        pip_value = 10.0  # $10 per pip per lot for standard account
        pnl = pips * pip_value * pos.volume
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade = Trade(
            ticket=pos.ticket,
            entry_time=pos.entry_time,
            exit_time=self.current_time,
            direction=pos.direction,
            volume=pos.volume,
            entry_price=pos.entry_price,
            exit_price=actual_exit_price,
            sl=pos.sl,
            tp=pos.tp,
            pnl=pnl,
            commission=pos.commission_paid,
            slippage_entry_pips=0,  # Would need to track from entry
            slippage_exit_pips=exit_slippage_pips,
            exit_reason=reason
        )
        
        self.closed_trades.append(trade)
        self.positions.remove(pos)
        
        logger.info(
            f"[CLOSE] {pos.direction} {pos.volume:.2f} @ {actual_exit_price:.5f} "
            f"({reason}) P&L: ${pnl:.2f}"
        )
    
    def _update_equity(self):
        """Update equity with unrealized P&L."""
        unrealized_pnl = sum(
            pos.unrealized_pnl(self.current_price)
            for pos in self.positions
        )
        self.equity = self.balance + unrealized_pnl
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [
            {
                'ticket': p.ticket,
                'type': p.direction,
                'volume': p.volume,
                'price_open': p.entry_price,
                'price_current': self.current_price,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.unrealized_pnl(self.current_price),
                'time': p.entry_time
            }
            for p in self.positions
        ]
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
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
                'commission': t.commission,
                'exit_reason': t.exit_reason,
                'slippage_pips': t.slippage_entry_pips + t.slippage_exit_pips
            }
            for t in self.closed_trades
        ]
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'final_balance': self.balance
            }
        
        pnls = [t.pnl for t in self.closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        total_wins = sum(winners) if winners else 0
        total_losses = abs(sum(losers)) if losers else 0
        
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(self.closed_trades) if self.closed_trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'total_pnl': sum(pnls),
            'average_pnl': np.mean(pnls),
            'largest_win': max(winners) if winners else 0,
            'largest_loss': min(losers) if losers else 0,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_commission': self.total_commission_paid,
            'total_slippage_pips': self.total_slippage_pips,
            'execution_stats': self.execution_stats
        }
    
    def get_account_info(self):
        """Get account information (compatibility)."""
        from trading.mt5_connector import AccountInfo
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0,
            free_margin=self.equity,
            profit=self.equity - self.initial_balance
        )
    
    def get_symbol_info(self) -> Dict:
        """Get symbol info (compatibility)."""
        return {
            'point': 0.00001,
            'trade_tick_value': 1.0,
            'volume_min': self.config.min_lot_size,
            'volume_max': self.config.max_lot_size,
            'volume_step': self.config.lot_step
        }
