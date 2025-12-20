"""
Portfolio-Level Performance Metrics
====================================

Comprehensive metrics calculation for backtesting results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Trade-level metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_trade_duration: timedelta = timedelta(0)
    avg_bars_in_trade: float = 0.0
    
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0


@dataclass
class RiskMetrics:
    """Risk-adjusted metrics."""
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: timedelta = timedelta(0)
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0


@dataclass
class ReturnMetrics:
    """Return metrics."""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    cagr: float = 0.0
    
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0
    
    monthly_return_mean: float = 0.0
    monthly_return_std: float = 0.0
    
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0


@dataclass
class ExecutionMetrics:
    """Execution quality metrics."""
    total_commission: float = 0.0
    total_slippage_pips: float = 0.0
    avg_slippage_pips: float = 0.0
    
    fill_rate: float = 1.0
    rejection_rate: float = 0.0
    requote_rate: float = 0.0
    
    avg_spread_pips: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""
    trade_metrics: TradeMetrics = field(default_factory=TradeMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    return_metrics: ReturnMetrics = field(default_factory=ReturnMetrics)
    execution_metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trade_metrics': self.trade_metrics.__dict__,
            'risk_metrics': self.risk_metrics.__dict__,
            'return_metrics': self.return_metrics.__dict__,
            'execution_metrics': self.execution_metrics.__dict__
        }


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics.
    
    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            trades=trade_history,
            equity_curve=equity_curve,
            initial_balance=10000
        )
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate(
        self,
        trades: List[Dict],
        equity_curve: Optional[pd.Series] = None,
        initial_balance: float = 10000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate all metrics.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Time series of equity values
            initial_balance: Starting balance
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            logger.warning("No trades to calculate metrics from")
            return metrics
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade metrics
        metrics.trade_metrics = self._calculate_trade_metrics(trades_df)
        
        # Calculate return metrics
        if equity_curve is not None:
            metrics.return_metrics = self._calculate_return_metrics(
                equity_curve, initial_balance, start_date, end_date
            )
            metrics.risk_metrics = self._calculate_risk_metrics(
                equity_curve, initial_balance, trades_df
            )
        else:
            # Build equity curve from trades
            equity_curve = self._build_equity_curve(trades_df, initial_balance)
            metrics.return_metrics = self._calculate_return_metrics(
                equity_curve, initial_balance, start_date, end_date
            )
            metrics.risk_metrics = self._calculate_risk_metrics(
                equity_curve, initial_balance, trades_df
            )
        
        # Calculate execution metrics
        metrics.execution_metrics = self._calculate_execution_metrics(trades_df)
        
        return metrics
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> TradeMetrics:
        """Calculate trade-level metrics."""
        metrics = TradeMetrics()
        
        if len(trades_df) == 0:
            return metrics
        
        # Basic counts
        metrics.total_trades = len(trades_df)
        
        pnls = trades_df['pnl'].values
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(pnls) if len(pnls) > 0 else 0
        
        # Win/loss statistics
        metrics.avg_win = np.mean(winners) if len(winners) > 0 else 0
        metrics.avg_loss = np.mean(losers) if len(losers) > 0 else 0
        metrics.largest_win = np.max(winners) if len(winners) > 0 else 0
        metrics.largest_loss = np.min(losers) if len(losers) > 0 else 0
        
        # Profit factor
        total_wins = np.sum(winners) if len(winners) > 0 else 0
        total_losses = abs(np.sum(losers)) if len(losers) > 0 else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        metrics.expectancy = np.mean(pnls) if len(pnls) > 0 else 0
        
        # Trade duration
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            durations = trades_df['exit_time'] - trades_df['entry_time']
            metrics.avg_trade_duration = durations.mean()
        
        # Long/short breakdown
        if 'direction' in trades_df.columns:
            long_trades = trades_df[trades_df['direction'] == 'BUY']
            short_trades = trades_df[trades_df['direction'] == 'SELL']
            
            metrics.long_trades = len(long_trades)
            metrics.short_trades = len(short_trades)
            
            if len(long_trades) > 0:
                long_winners = long_trades[long_trades['pnl'] > 0]
                metrics.long_win_rate = len(long_winners) / len(long_trades)
            
            if len(short_trades) > 0:
                short_winners = short_trades[short_trades['pnl'] > 0]
                metrics.short_win_rate = len(short_winners) / len(short_trades)
        
        return metrics
    
    def _calculate_return_metrics(
        self,
        equity_curve: pd.Series,
        initial_balance: float,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> ReturnMetrics:
        """Calculate return metrics."""
        metrics = ReturnMetrics()
        
        if len(equity_curve) == 0:
            return metrics
        
        final_balance = equity_curve.iloc[-1]
        
        # Total return
        metrics.total_return = final_balance - initial_balance
        metrics.total_return_pct = (final_balance / initial_balance - 1) * 100
        
        # CAGR
        if start_date and end_date:
            years = (end_date - start_date).days / 365.25
            if years > 0:
                metrics.cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 0:
            metrics.daily_return_mean = daily_returns.mean() * 100
            metrics.daily_return_std = daily_returns.std() * 100
            metrics.best_day = daily_returns.max() * 100
            metrics.worst_day = daily_returns.min() * 100
        
        # Monthly returns (if enough data)
        if len(equity_curve) > 30:
            try:
                monthly_equity = equity_curve.resample('M').last()
                monthly_returns = monthly_equity.pct_change().dropna()
                if len(monthly_returns) > 0:
                    metrics.monthly_return_mean = monthly_returns.mean() * 100
                    metrics.monthly_return_std = monthly_returns.std() * 100
                    metrics.best_month = monthly_returns.max() * 100
                    metrics.worst_month = monthly_returns.min() * 100
            except:
                pass
        
        return metrics
    
    def _calculate_risk_metrics(
        self,
        equity_curve: pd.Series,
        initial_balance: float,
        trades_df: pd.DataFrame
    ) -> RiskMetrics:
        """Calculate risk-adjusted metrics."""
        metrics = RiskMetrics()
        
        if len(equity_curve) == 0:
            return metrics
        
        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        metrics.max_drawdown = drawdown.min()
        metrics.max_drawdown_pct = drawdown_pct.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
            drawdown_lengths = drawdown_periods[in_drawdown].value_counts()
            if len(drawdown_lengths) > 0:
                max_dd_periods = drawdown_lengths.max()
                if isinstance(equity_curve.index, pd.DatetimeIndex):
                    avg_period = equity_curve.index.to_series().diff().mean()
                    metrics.max_drawdown_duration = avg_period * max_dd_periods
        
        # Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            metrics.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Sortino ratio (uses downside deviation)
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    excess_returns = returns - (self.risk_free_rate / 252)
                    metrics.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
        
        # Calmar ratio
        if metrics.max_drawdown_pct < 0:
            final_return_pct = (equity_curve.iloc[-1] / initial_balance - 1) * 100
            metrics.calmar_ratio = final_return_pct / abs(metrics.max_drawdown_pct)
        
        # Value at Risk (95%)
        if len(returns) > 0:
            metrics.var_95 = np.percentile(returns, 5) * initial_balance
            # Conditional VaR (expected shortfall)
            var_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                metrics.cvar_95 = tail_returns.mean() * initial_balance
        
        # Consecutive wins/losses
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            pnls = trades_df['pnl'].values
            
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            
            for pnl in pnls:
                if pnl > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    max_win_streak = max(max_win_streak, current_streak)
                elif pnl < 0:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))
            
            metrics.max_consecutive_wins = max_win_streak
            metrics.max_consecutive_losses = max_loss_streak
        
        # Recovery factor
        total_return = equity_curve.iloc[-1] - initial_balance
        if metrics.max_drawdown < 0:
            metrics.recovery_factor = total_return / abs(metrics.max_drawdown)
        
        # Ulcer Index (measure of downside volatility)
        drawdown_squared = (drawdown_pct ** 2).mean()
        metrics.ulcer_index = np.sqrt(drawdown_squared)
        
        return metrics
    
    def _calculate_execution_metrics(self, trades_df: pd.DataFrame) -> ExecutionMetrics:
        """Calculate execution quality metrics."""
        metrics = ExecutionMetrics()
        
        if len(trades_df) == 0:
            return metrics
        
        # Commission
        if 'commission' in trades_df.columns:
            metrics.total_commission = trades_df['commission'].sum()
        
        # Slippage
        if 'slippage_pips' in trades_df.columns:
            slippage = trades_df['slippage_pips'].dropna()
            if len(slippage) > 0:
                metrics.total_slippage_pips = slippage.sum()
                metrics.avg_slippage_pips = slippage.mean()
        
        # Spread
        if 'spread_pips' in trades_df.columns:
            spread = trades_df['spread_pips'].dropna()
            if len(spread) > 0:
                metrics.avg_spread_pips = spread.mean()
        
        return metrics
    
    def _build_equity_curve(
        self,
        trades_df: pd.DataFrame,
        initial_balance: float
    ) -> pd.Series:
        """Build equity curve from trades."""
        if len(trades_df) == 0 or 'exit_time' not in trades_df.columns:
            return pd.Series([initial_balance])
        
        trades_df = trades_df.sort_values('exit_time')
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        equity = initial_balance
        equity_data = [(trades_df['exit_time'].iloc[0], initial_balance)]
        
        for _, trade in trades_df.iterrows():
            equity += trade['pnl']
            if 'commission' in trade:
                equity -= trade['commission']
            equity_data.append((trade['exit_time'], equity))
        
        equity_curve = pd.Series(
            [e[1] for e in equity_data],
            index=[e[0] for e in equity_data]
        )
        
        return equity_curve
    
    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics for stability analysis.
        
        Args:
            equity_curve: Time series of equity
            window: Rolling window size
        
        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=equity_curve.index[window:])
        
        # Rolling Sharpe
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_metrics['sharpe'] = np.sqrt(252) * rolling_mean / rolling_std
        
        # Rolling max drawdown
        rolling_max = equity_curve.rolling(window).max()
        rolling_dd = (equity_curve - rolling_max) / rolling_max * 100
        rolling_metrics['max_dd_pct'] = rolling_dd.rolling(window).min()
        
        # Rolling win rate (approximate from returns)
        rolling_metrics['win_rate'] = (returns > 0).rolling(window).mean() * 100
        
        return rolling_metrics.dropna()
