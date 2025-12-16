# trading/live_trading_bot.py
"""
Live Trading Bot with Full Risk Management Integration (v2)

Production-ready trading bot integrating ALL 5 PHASES:
- Phase 1-3: TCN predictions, risk calculations, meta-labeling (via strategy)
- Phase 4: RL Exit Advisor for active position management
- Phase 5: Capital Protection with kill switch

Features:
- Continuous trading loop with ML signal generation
- Active position monitoring with RL exit recommendations
- Multi-level capital protection (daily/weekly/drawdown limits)
- Real-time performance monitoring
- Graceful shutdown handling

Usage:
    bot = LiveTradingBot(config)
    bot.initialize(starting_balance=10000)
    bot.run()  # Blocking
    # or
    bot.start()  # Non-blocking (thread)
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Strategy imports
from strategies.neural_hybrid import (
    NeuralHybridStrategy, StrategyConfig, create_strategy, Order
)
from trading.decision_engine import TradeDecision

# Risk management imports - Phases 1-3
try:
    from risk_management import (
        TradeGatekeeper, HardRulesConfig, TradingSession,
        RegimeDetector
    )
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False

# Phase 4: Exit Advisor
try:
    from risk_management import ExitAdvisor, ExitAction, Position
    HAS_EXIT_ADVISOR = True
except ImportError:
    HAS_EXIT_ADVISOR = False
    ExitAdvisor = None
    ExitAction = None
    Position = None

# Phase 5: Capital Protection
try:
    from risk_management import (
        CapitalProtector, ProtectionConfig, ProtectionManager,
        ProtectionLevel, ProtectionAction
    )
    HAS_CAPITAL_PROTECTION = True
except ImportError:
    HAS_CAPITAL_PROTECTION = False
    CapitalProtector = None
    ProtectionManager = None

logger = logging.getLogger(__name__)


class BotState(Enum):
    """Bot operational states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PROTECTION_HALT = "protection_halt"  # New: halted by capital protection


@dataclass
class OpenPosition:
    """Represents an open trading position for exit management."""
    ticket: str
    symbol: str
    direction: int          # 1 = long, -1 = short
    entry_price: float
    entry_time: datetime
    volume: float
    stop_loss: float
    take_profit: float
    current_pnl: float = 0.0
    last_checked: Optional[datetime] = None


@dataclass
class BotConfig:
    """Configuration for live trading bot."""
    # Trading settings
    symbol: str = 'EURUSD'
    profile: str = 'INTRADAY'
    
    # Timing
    check_interval_seconds: int = 60     # How often to check for signals
    position_check_seconds: int = 30     # How often to check open positions
    market_open_hour: int = 0            # UTC hour market opens (Sunday)
    market_close_hour: int = 22          # UTC hour market closes (Friday)
    
    # Model paths
    tcn_weights: str = 'models/weights/tcn_best.pt'
    vit_weights: Optional[str] = None
    yolo_weights: Optional[str] = None
    meta_model_path: Optional[str] = None
    exit_model_path: Optional[str] = None  # NEW: Phase 4 exit model
    
    # Risk settings
    base_risk_percent: float = 1.0
    max_daily_loss_percent: float = 3.0
    max_weekly_loss_percent: float = 6.0
    max_monthly_loss_percent: float = 10.0
    max_drawdown_percent: float = 10.0
    max_open_trades: int = 3
    
    # Exit advisor settings (Phase 4)
    enable_exit_advisor: bool = True
    exit_confidence_threshold: float = 0.6
    
    # Capital protection settings (Phase 5)
    enable_capital_protection: bool = True
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 30
    
    # Notifications
    enable_notifications: bool = True
    telegram_config: Optional[Dict] = None
    
    # Logging
    log_trades: bool = True
    trades_log_file: str = 'logs/trades.json'
    performance_log_file: str = 'logs/performance.json'

    persist_open_positions: bool = True
    open_positions_file: str = 'logs/open_positions.json'
    
    # Safety
    dry_run: bool = False               # Paper trading mode
    require_confirmation: bool = False   # Require manual confirmation
    halt_on_disconnect: bool = True
    min_order_interval_seconds: int = 2


class LiveTradingBot:
    """
    Production live trading bot with full risk management (v2).
    
    NEW in v2:
    - Phase 4: Active exit management using RL advisor
    - Phase 5: Multi-level capital protection with kill switch
    
    Example:
        from trading.mt5_executor import MT5Executor
        from data.mt5_provider import MT5DataProvider
        
        bot = LiveTradingBot(
            config=BotConfig(symbol='EURUSD', profile='INTRADAY'),
            data_provider=MT5DataProvider(),
            executor=MT5Executor()
        )
        
        bot.initialize(starting_balance=10000)
        bot.run()
    """
    
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        data_provider = None,
        executor = None,
        strategy: Optional[NeuralHybridStrategy] = None
    ):
        self.config = config or BotConfig()
        self.data_provider = data_provider
        self.executor = executor
        
        # State
        self.state = BotState.STOPPED
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Strategy (created in initialize if not provided)
        self.strategy = strategy
        
        # Phase 4: Exit Advisor
        self.exit_advisor: Optional[ExitAdvisor] = None
        
        # Phase 5: Capital Protection
        self.capital_protector: Optional[CapitalProtector] = None
        
        # Position tracking
        self._open_positions: Dict[str, OpenPosition] = {}
        
        # Performance tracking
        self._starting_balance = 0.0
        self._current_balance = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._trade_count = 0
        self._win_count = 0
        self._consecutive_losses = 0
        
        # Statistics
        self._signals_generated = 0
        self._trades_executed = 0
        self._trades_rejected = 0
        self._exits_by_advisor = 0  # NEW: Track RL exit recommendations
        self._protection_blocks = 0  # NEW: Track protection blocks
        
        # Timing
        self._last_trade_time: Optional[datetime] = None
        self._last_position_check: Optional[datetime] = None
        self._day_start: Optional[datetime] = None
        self._week_start: Optional[datetime] = None

        self._order_lock = threading.Lock()
        self._order_in_flight = False
        self._last_order_fingerprint: Optional[str] = None
        self._last_order_time: Optional[datetime] = None
        
        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_exit: Optional[Callable] = None  # NEW: Exit callback
        self.on_protection_event: Optional[Callable] = None  # NEW
        self.on_error: Optional[Callable] = None
        
        logger.info(f"LiveTradingBot v2 created for {self.config.symbol}")
    
    def initialize(self, starting_balance: Optional[float] = None):
        """Initialize bot with all components."""
        self.state = BotState.INITIALIZING
        
        # Get starting balance
        if starting_balance is not None:
            self._starting_balance = starting_balance
        elif self.executor:
            self._starting_balance = self.executor.get_account_balance()
        else:
            self._starting_balance = 10000.0
        
        self._current_balance = self._starting_balance
        
        # Initialize strategy
        if self.strategy is None:
            self.strategy = create_strategy(
                profile=self.config.profile,
                symbol=self.config.symbol,
                data_provider=self.data_provider,
                executor=self.executor,
                tcn_weights=self.config.tcn_weights,
                meta_model_path=self.config.meta_model_path
            )
        
        self.strategy.initialize()

        if self.config.persist_open_positions:
            self._recover_open_positions()
        
        # Initialize Phase 4: Exit Advisor
        if self.config.enable_exit_advisor and HAS_EXIT_ADVISOR:
            if self.config.exit_model_path and Path(self.config.exit_model_path).exists():
                try:
                    self.exit_advisor = ExitAdvisor.load(self.config.exit_model_path)
                    logger.info("Phase 4: Exit Advisor loaded")
                except Exception as e:
                    logger.warning(f"Could not load Exit Advisor: {e}")
            else:
                logger.info("Phase 4: Exit Advisor not configured (no model path)")
        
        # Initialize Phase 5: Capital Protection
        if self.config.enable_capital_protection and HAS_CAPITAL_PROTECTION:
            protection_config = ProtectionConfig(
                max_daily_loss_pct=self.config.max_daily_loss_percent,
                max_weekly_loss_pct=self.config.max_weekly_loss_percent,
                max_monthly_loss_pct=self.config.max_monthly_loss_percent,
                max_drawdown_pct=self.config.max_drawdown_percent,
                max_consecutive_losses=self.config.max_consecutive_losses,
                losing_streak_cooldown_minutes=self.config.cooldown_minutes
            )
            self.capital_protector = CapitalProtector(protection_config)
            self.capital_protector.initialize(self._starting_balance)
            logger.info("Phase 5: Capital Protection initialized")
        
        # Time tracking
        now = datetime.utcnow()
        self._day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self._week_start = now - timedelta(days=now.weekday())
        
        self.state = BotState.RUNNING
        logger.info(f"Bot initialized with balance: {self._starting_balance:.2f}")
    
    def run(self):
        """Run the trading bot (blocking)."""
        if self.state != BotState.RUNNING:
            raise RuntimeError("Bot must be initialized before running")
        
        logger.info("Starting trading loop...")
        
        while not self._stop_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Check for day/week rollover
                self._check_time_periods(current_time)

                # Execution connectivity fail-safe
                if self.config.halt_on_disconnect and self.executor:
                    ensure = getattr(self.executor, 'ensure_connected', None)
                    connected = True
                    if callable(ensure):
                        connected = bool(ensure())
                    else:
                        connected = bool(getattr(self.executor, 'connected', True))

                    if not connected:
                        self.state = BotState.DISCONNECTED
                        logger.critical("EXECUTION DISCONNECTED - Trading halted")
                        self._wait(60)
                        continue

                    if self.state == BotState.DISCONNECTED:
                        self.state = BotState.RUNNING
                
                # Check capital protection status
                if self.capital_protector:
                    protection_state = self.capital_protector.get_state()
                    if protection_state.level == ProtectionLevel.KILLED:
                        self.state = BotState.PROTECTION_HALT
                        logger.critical("KILL SWITCH ACTIVE - Trading halted")
                        if self.on_protection_event:
                            self.on_protection_event('kill_switch', protection_state)
                        self._wait(60)
                        continue
                
                # Skip if market closed
                if not self._is_market_open(current_time):
                    self._wait(60)
                    continue
                
                # Phase 4: Check open positions for exit
                if self.exit_advisor and self._open_positions:
                    self._check_positions_for_exit(current_time)
                
                # Evaluate market for new signals (if allowed)
                if self.state == BotState.RUNNING and self._can_open_new_trade():
                    self._evaluate_and_trade(current_time)
                
                # Wait for next check
                self._wait(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(e)
                self._wait(60)
        
        self.state = BotState.STOPPED
        logger.info("Trading loop stopped")
    
    def start(self):
        """Start bot in background thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Bot already running")
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        logger.info("Bot started in background")
    
    def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=10)
        
        self.state = BotState.STOPPED
        logger.info("Bot stopped")
    
    def _can_open_new_trade(self) -> bool:
        """Check if we can open a new trade."""
        # Check max open trades
        if len(self._open_positions) >= self.config.max_open_trades:
            return False
        
        # Check capital protection
        if self.capital_protector:
            check = self.capital_protector.check_trade(
                proposed_size=0.01,
                account_balance=self._current_balance
            )
            if not check['allowed']:
                self._protection_blocks += 1
                return False
        
        return True
    
    def _evaluate_and_trade(self, current_time: datetime):
        """Evaluate market and execute trades if appropriate."""
        # Get trading decision from strategy
        decision = self.strategy.evaluate(current_time)
        self._signals_generated += 1
        
        if decision is None:
            return
        
        # Callback for signal
        if self.on_signal:
            self.on_signal(decision)
        
        if not decision.should_trade:
            self._trades_rejected += 1
            logger.debug(f"Signal rejected: {decision.rejection_reasons}")
            return
        
        # Phase 5: Final capital protection check
        if self.capital_protector:
            final_check = self.capital_protector.check_trade(
                proposed_size=decision.position_size,
                account_balance=self._current_balance
            )
            
            if not final_check['allowed']:
                self._trades_rejected += 1
                self._protection_blocks += 1
                logger.info(f"Capital protection blocked: {final_check['reason']}")
                if self.on_protection_event:
                    self.on_protection_event('trade_blocked', final_check)
                return
            
            # Apply size adjustment
            if final_check['adjusted_size'] != decision.position_size:
                logger.info(
                    f"Size adjusted by protection: {decision.position_size:.4f} -> "
                    f"{final_check['adjusted_size']:.4f}"
                )
                decision.position_size = final_check['adjusted_size']
        
        # Create and execute order
        order = self.strategy.create_order(decision)
        if order is None:
            return
        
        # Dry run mode
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would execute: {order.direction} {order.volume} {order.symbol}")
            self._trades_executed += 1
            return
        
        # Execute trade
        fingerprint = (
            f"{order.symbol}|{order.direction}|{order.volume:.6f}|"
            f"{order.stop_loss:.8f}|{order.take_profit:.8f}"
        )

        with self._order_lock:
            if self._order_in_flight:
                return

            if (
                self._last_order_fingerprint == fingerprint
                and self._last_order_time is not None
                and (current_time - self._last_order_time).total_seconds() < self.config.min_order_interval_seconds
            ):
                return

            self._order_in_flight = True
            self._last_order_fingerprint = fingerprint
            self._last_order_time = current_time

        try:
            success = self.strategy.execute(order)
        finally:
            with self._order_lock:
                self._order_in_flight = False
        
        if success:
            self._trades_executed += 1
            self._trade_count += 1
            self._last_trade_time = current_time
            
            # Track open position
            ticket = order.ticket if hasattr(order, 'ticket') else str(current_time.timestamp())
            self._open_positions[ticket] = OpenPosition(
                ticket=ticket,
                symbol=order.symbol,
                direction=1 if order.direction == 'BUY' else -1,
                entry_price=order.price,
                entry_time=current_time,
                volume=order.volume,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit
            )

            if self.config.persist_open_positions:
                self._save_open_positions()
            
            # Log trade
            if self.config.log_trades:
                self._log_trade(order, decision)
            
            # Callback
            if self.on_trade:
                self.on_trade(order, decision.to_dict())
        else:
            logger.warning("Trade execution failed")
    
    def _check_positions_for_exit(self, current_time: datetime):
        """Check open positions for exit recommendations (Phase 4)."""
        if not self.exit_advisor:
            return
        
        # Rate limit position checks
        if self._last_position_check:
            time_since = (current_time - self._last_position_check).total_seconds()
            if time_since < self.config.position_check_seconds:
                return
        
        self._last_position_check = current_time
        
        for ticket, pos in list(self._open_positions.items()):
            try:
                # Get current market data
                market_data = self.data_provider.get_ohlcv(
                    pos.symbol,
                    timeframe='M5',
                    count=50
                )
                
                # Create Position object for advisor
                current_price = self.data_provider.get_price(pos.symbol)
                rl_position = Position(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    entry_time=0,
                    initial_size=pos.volume,
                    current_size=pos.volume,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    initial_sl=pos.stop_loss,
                    initial_tp=pos.take_profit
                )
                
                # Get exit recommendation
                recommendation = self.exit_advisor.get_recommendation(
                    rl_position,
                    market_data,
                    deterministic=True
                )
                
                # Process recommendation
                if recommendation['action'] == ExitAction.EXIT:
                    if recommendation['confidence'] >= self.config.exit_confidence_threshold:
                        logger.info(
                            f"Exit Advisor recommends EXIT for {ticket} "
                            f"(confidence: {recommendation['confidence']:.2%})"
                        )
                        self._execute_exit(ticket, 'rl_advisor')
                
                elif recommendation['action'] == ExitAction.TRAIL_STOP:
                    logger.debug(f"Exit Advisor recommends trailing stop for {ticket}")
                    # Could implement trailing stop adjustment here
                
                elif recommendation['action'] in [ExitAction.PARTIAL_25, ExitAction.PARTIAL_50, ExitAction.PARTIAL_75]:
                    if recommendation['confidence'] >= self.config.exit_confidence_threshold:
                        fraction = {
                            ExitAction.PARTIAL_25: 0.25,
                            ExitAction.PARTIAL_50: 0.50,
                            ExitAction.PARTIAL_75: 0.75
                        }.get(recommendation['action'], 0.5)
                        logger.info(
                            f"Exit Advisor recommends partial close ({fraction:.0%}) for {ticket}"
                        )
                        # Could implement partial close here
                
            except Exception as e:
                logger.error(f"Error checking position {ticket}: {e}")
    
    def _execute_exit(self, ticket: str, reason: str):
        """Execute position exit."""
        if ticket not in self._open_positions:
            return
        
        pos = self._open_positions[ticket]
        
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would close {ticket} ({reason})")
            return
        
        # Execute close
        try:
            result = self.executor.close_position(ticket)
            if result.get('success'):
                pnl = result.get('pnl', 0.0)
                self._on_position_closed(ticket, pnl, reason)
                self._exits_by_advisor += 1
                
                if self.on_exit:
                    self.on_exit(ticket, pnl, reason)
        except Exception as e:
            logger.error(f"Failed to close position {ticket}: {e}")
    
    def _on_position_closed(self, ticket: str, pnl: float, reason: str = 'normal'):
        """Handle position close event."""
        # Remove from tracking
        pos = self._open_positions.pop(ticket, None)

        if self.config.persist_open_positions:
            self._save_open_positions()
        
        # Update P&L
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        self._current_balance += pnl
        
        # Update win/loss tracking
        if pnl >= 0:
            self._win_count += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        
        # Update capital protector (Phase 5)
        if self.capital_protector:
            self.capital_protector.record_trade(
                pnl=pnl,
                is_win=pnl >= 0,
                trade_size=pos.volume if pos else 0.0
            )
            
            # Check if protection status changed
            state = self.capital_protector.get_state()
            if state.level in [ProtectionLevel.CRITICAL, ProtectionLevel.KILLED]:
                logger.warning(f"Capital protection activated: {state.level.value}")
                if self.on_protection_event:
                    self.on_protection_event('level_change', state)
        
        # Update strategy
        if self.strategy:
            self.strategy.on_trade_closed(ticket, pnl)
        
        logger.info(
            f"Position closed: {ticket} | PnL: {pnl:.2f} | Reason: {reason} | "
            f"Daily: {self._daily_pnl:.2f} | Balance: {self._current_balance:.2f}"
        )
    
    def on_position_closed(self, ticket: str, pnl: float):
        """Public method for external position close notifications."""
        self._on_position_closed(ticket, pnl, 'external')
    
    def _check_time_periods(self, current_time: datetime):
        """Check and reset time period counters."""
        # Check day rollover
        current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._day_start and current_day > self._day_start:
            self.reset_daily_stats()
            self._day_start = current_day
        
        # Check week rollover
        current_week = current_time - timedelta(days=current_time.weekday())
        current_week = current_week.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._week_start and current_week > self._week_start:
            self.reset_weekly_stats()
            self._week_start = current_week
    
    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if forex market is open."""
        weekday = current_time.weekday()
        hour = current_time.hour
        
        # Market closed Saturday and most of Sunday
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < self.config.market_open_hour:  # Sunday before open
            return False
        if weekday == 4 and hour >= self.config.market_close_hour:  # Friday after close
            return False
        
        return True
    
    def _wait(self, seconds: int):
        """Wait with stop check."""
        self._stop_event.wait(seconds)
    
    def _log_trade(self, order, decision):
        """Log trade to file."""
        try:
            log_path = Path(self.config.trades_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': order.symbol,
                'direction': order.direction,
                'volume': order.volume,
                'price': order.price,
                'stop_loss': order.stop_loss,
                'take_profit': order.take_profit,
                'decision': decision.to_dict() if hasattr(decision, 'to_dict') else str(decision)
            }
            
            # Append to file
            trades = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    trades = json.load(f)
            
            trades.append(entry)
            
            with open(log_path, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def _save_open_positions(self):
        try:
            path = Path(self.config.open_positions_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = []
            for ticket, pos in self._open_positions.items():
                data.append(
                    {
                        'ticket': ticket,
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat(),
                        'volume': pos.volume,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                    }
                )

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist open positions: {e}")

    def _recover_open_positions(self):
        try:
            path = Path(self.config.open_positions_file)
            if not path.exists():
                return

            with open(path, 'r') as f:
                data = json.load(f) or []

            recovered = {}
            for row in data:
                ticket = str(row.get('ticket'))
                recovered[ticket] = OpenPosition(
                    ticket=ticket,
                    symbol=row.get('symbol', self.config.symbol),
                    direction=int(row.get('direction', 0)),
                    entry_price=float(row.get('entry_price', 0.0)),
                    entry_time=datetime.fromisoformat(row.get('entry_time')),
                    volume=float(row.get('volume', 0.0)),
                    stop_loss=float(row.get('stop_loss', 0.0)),
                    take_profit=float(row.get('take_profit', 0.0)),
                )

            self._open_positions = recovered

            if self.executor and hasattr(self.executor, 'get_open_positions'):
                broker_positions = self.executor.get_open_positions(symbol=self.config.symbol)
                broker_tickets = {str(p.get('ticket')) for p in broker_positions}
                orphaned = [t for t in self._open_positions.keys() if t not in broker_tickets]
                for t in orphaned:
                    self._open_positions.pop(t, None)

            self._save_open_positions()

        except Exception as e:
            logger.error(f"Failed to recover open positions: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        self._daily_pnl = 0.0
        if self.strategy:
            self.strategy.reset_daily_stats()
        if self.capital_protector:
            self.capital_protector.reset_daily()
        logger.info("Daily stats reset")
    
    def reset_weekly_stats(self):
        """Reset weekly statistics."""
        self._weekly_pnl = 0.0
        if self.capital_protector:
            self.capital_protector.reset_weekly()
        logger.info("Weekly stats reset")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        status = {
            'state': self.state.value,
            'symbol': self.config.symbol,
            'profile': self.config.profile,
            'balance': self._current_balance,
            'starting_balance': self._starting_balance,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'monthly_pnl': self._monthly_pnl,
            'open_positions': len(self._open_positions),
            'trade_count': self._trade_count,
            'win_rate': self._win_count / self._trade_count if self._trade_count > 0 else 0,
            'consecutive_losses': self._consecutive_losses,
            'signals_generated': self._signals_generated,
            'trades_executed': self._trades_executed,
            'trades_rejected': self._trades_rejected,
            'exits_by_advisor': self._exits_by_advisor,
            'protection_blocks': self._protection_blocks
        }
        
        # Add Phase 5 protection status
        if self.capital_protector:
            prot_state = self.capital_protector.get_state()
            status['protection'] = {
                'level': prot_state.level.value,
                'action': prot_state.action.value,
                'size_multiplier': prot_state.size_multiplier,
                'trigger_reason': prot_state.trigger_reason
            }
        
        return status
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'direction': 'BUY' if pos.direction == 1 else 'SELL',
                'entry_price': pos.entry_price,
                'entry_time': pos.entry_time.isoformat(),
                'volume': pos.volume,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'current_pnl': pos.current_pnl
            }
            for pos in self._open_positions.values()
        ]
    
    def force_close_all(self, reason: str = "Manual close"):
        """Force close all open positions."""
        logger.warning(f"Force closing all positions: {reason}")
        
        for ticket in list(self._open_positions.keys()):
            self._execute_exit(ticket, reason)
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Manually activate kill switch."""
        if self.capital_protector:
            self.capital_protector.activate_kill_switch(reason)
            self.state = BotState.PROTECTION_HALT
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")


def create_bot(
    symbol: str,
    profile: str,
    data_provider,
    executor,
    starting_balance: Optional[float] = None,
    **kwargs
) -> LiveTradingBot:
    """Factory function to create configured bot."""
    config = BotConfig(
        symbol=symbol,
        profile=profile,
        **kwargs
    )
    
    bot = LiveTradingBot(
        config=config,
        data_provider=data_provider,
        executor=executor
    )
    
    bot.initialize(starting_balance)
    
    return bot
