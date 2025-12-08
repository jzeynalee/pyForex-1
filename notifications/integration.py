"""
Social Media Integration Module for pyForex Trading System.

Wires social media notifications into the existing trading infrastructure:
- Performance Monitor: Auto-post trade results and summaries
- Retraining Scheduler: Post model updates and alerts
- Trading Bot: Real-time trade notifications

Usage:
    from notifications import SocialMediaIntegration
    
    # Create integration
    integration = SocialMediaIntegration.from_env()
    
    # Connect to performance monitor
    integration.connect_performance_monitor(monitor)
    
    # Connect to retraining scheduler
    integration.connect_retraining_scheduler(scheduler)
    
    # Or use the quick setup
    integration = setup_social_notifications(
        performance_monitor=monitor,
        retraining_scheduler=scheduler,
    )
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import threading
import time
from dataclasses import dataclass

from .social_media import (
    SocialMediaNotifier,
    NotificationConfig,
    Platform,
    TradeData,
    PerformanceData,
    PostType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CHART GENERATOR
# =============================================================================

class ChartGenerator:
    """Generates charts for social media posts."""
    
    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_equity_curve(
        self, 
        equity_data: List[tuple],  # [(datetime, equity), ...]
        title: str = "Equity Curve",
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Generate equity curve chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not equity_data or len(equity_data) < 2:
                return None
            
            dates, values = zip(*equity_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot equity
            ax.plot(dates, values, 'b-', linewidth=2, label='Equity')
            ax.fill_between(dates, values, alpha=0.3)
            
            # Formatting
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Date formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.xticks(rotation=45)
            
            # Calculate return
            start_val = values[0]
            end_val = values[-1]
            total_return = (end_val - start_val) / start_val * 100
            
            ax.annotate(
                f'Return: {total_return:+.1f}%',
                xy=(0.02, 0.98),
                xycoords='axes fraction',
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            plt.tight_layout()
            
            # Save
            if filename is None:
                filename = f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except ImportError:
            logger.warning("matplotlib not installed, chart generation disabled")
            return None
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
    
    def generate_performance_summary(
        self,
        metrics: Dict[str, float],
        period: str = "Weekly",
        filename: Optional[str] = None
    ) -> Optional[str]:
        """Generate performance summary chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Left: Key metrics bar chart
            ax1 = axes[0]
            metric_names = ['Win Rate', 'Profit Factor', 'Sharpe']
            metric_values = [
                metrics.get('win_rate', 0) * 100,
                metrics.get('profit_factor', 1),
                metrics.get('sharpe_ratio', 0),
            ]
            colors = ['green' if v > 50 or (i > 0 and v > 1) else 'red' 
                     for i, v in enumerate(metric_values)]
            
            bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
            ax1.set_title(f'{period} Performance Metrics', fontweight='bold')
            ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Threshold')
            
            # Add value labels
            for bar, val in zip(bars, metric_values):
                ax1.annotate(f'{val:.1f}', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom')
            
            # Right: P&L info
            ax2 = axes[1]
            ax2.axis('off')
            
            total_pnl = metrics.get('total_pnl', 0)
            total_pnl_pct = metrics.get('total_pnl_pct', 0)
            trades = metrics.get('total_trades', 0)
            
            info_text = f"""
{period} Summary
━━━━━━━━━━━━━━━━

Total Trades: {trades}
Total P&L: ${total_pnl:,.2f}
Return: {total_pnl_pct:+.2f}%

Win Rate: {metrics.get('win_rate', 0):.1%}
Profit Factor: {metrics.get('profit_factor', 1):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
"""
            
            color = 'green' if total_pnl > 0 else 'red'
            ax2.text(0.5, 0.5, info_text, transform=ax2.transAxes,
                    fontsize=12, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            
            plt.tight_layout()
            
            if filename is None:
                filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except ImportError:
            return None
        except Exception as e:
            logger.error(f"Summary chart generation failed: {e}")
            return None


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class SocialMediaIntegration:
    """
    Integrates social media notifications with pyForex trading system.
    
    Handles:
    - Converting internal trade records to social media format
    - Scheduling periodic performance posts
    - Triggering alerts on significant events
    - Managing chart generation for posts
    """
    
    def __init__(
        self, 
        config: NotificationConfig,
        chart_dir: str = "./charts"
    ):
        self.config = config
        self.notifier = SocialMediaNotifier(config)
        self.chart_generator = ChartGenerator(chart_dir)
        
        # Connected components
        self.performance_monitor = None
        self.retraining_scheduler = None
        
        # Scheduling
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Tracking
        self.last_daily_post: Optional[datetime] = None
        self.last_weekly_post: Optional[datetime] = None
        self.posted_trade_ids: set = set()
        
        # Milestones tracking
        self.milestone_thresholds = {
            'trades': [10, 50, 100, 500, 1000],
            'pnl': [100, 500, 1000, 5000, 10000],
            'win_streak': [5, 10, 15, 20],
        }
        self.achieved_milestones: set = set()
        
        logger.info("SocialMediaIntegration initialized")
    
    @classmethod
    def from_env(cls, chart_dir: str = "./charts") -> 'SocialMediaIntegration':
        """Create integration with config from environment variables."""
        config = NotificationConfig.from_env()
        return cls(config, chart_dir)
    
    @classmethod
    def from_config_file(cls, filepath: str, chart_dir: str = "./charts") -> 'SocialMediaIntegration':
        """Create integration from config file."""
        config = NotificationConfig.from_file(filepath)
        return cls(config, chart_dir)
    
    # =========================================================================
    # CONNECT COMPONENTS
    # =========================================================================
    
    def connect_performance_monitor(self, monitor: 'PerformanceMonitor'):
        """
        Connect to PerformanceMonitor for automatic trade posting.
        
        This hooks into the monitor's trade recording to automatically
        post trade results to social media.
        """
        self.performance_monitor = monitor
        
        # Store original add_trade method
        original_add_trade = monitor.add_trade
        
        def wrapped_add_trade(trade_record):
            # Call original
            result = original_add_trade(trade_record)
            
            # Post to social media if trade is closed
            if trade_record.is_closed:
                self._on_trade_closed(trade_record)
            
            return result
        
        # Replace method
        monitor.add_trade = wrapped_add_trade
        
        logger.info("Connected to PerformanceMonitor")
    
    def connect_retraining_scheduler(self, scheduler: 'RetrainingScheduler'):
        """
        Connect to RetrainingScheduler for model update notifications.
        
        Posts alerts when:
        - Model retraining starts
        - Retraining completes (success/failure)
        - Model rollback occurs
        - Drift detected
        """
        self.retraining_scheduler = scheduler
        
        # Add callbacks
        scheduler.add_trigger_callback(self._on_retraining_triggered)
        scheduler.add_completion_callback(self._on_retraining_complete)
        scheduler.add_failure_callback(self._on_retraining_failed)
        
        logger.info("Connected to RetrainingScheduler")
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_trade_closed(self, trade_record: 'TradeRecord'):
        """Handle trade closed event."""
        # Avoid duplicate posts
        if trade_record.trade_id in self.posted_trade_ids:
            return
        
        # Convert to TradeData
        trade_data = self._convert_trade_record(trade_record)
        
        # Post to social media
        results = self.notifier.post_trade_result(trade_data)
        
        if any(results.values()):
            self.posted_trade_ids.add(trade_record.trade_id)
            logger.info(f"Posted trade {trade_record.trade_id} to social media")
        
        # Check for milestones
        self._check_milestones()
    
    def _on_retraining_triggered(self, event: 'RetrainingEvent'):
        """Handle retraining triggered event."""
        message = f"""Model retraining initiated

Trigger: {event.trigger_type.value}
Reason: {event.trigger_reason}
Profile: {event.profile_name}

The system will update you when training completes."""
        
        self.notifier.post_alert(
            "Model Retraining Started",
            message,
            platforms=[Platform.TELEGRAM]
        )
    
    def _on_retraining_complete(self, event: 'RetrainingEvent'):
        """Handle retraining completed event."""
        duration = event.training_duration_seconds / 60
        
        message = f"""Model retraining completed successfully!

New Model: {event.new_model_id}
Previous: {event.old_model_id or 'None'}
Duration: {duration:.1f} minutes

Validation Results:
{self._format_validation_results(event.validation_result)}"""
        
        self.notifier.post_alert(
            "Model Update Complete ✅",
            message,
            platforms=[Platform.TELEGRAM]
        )
        
        # Also post milestone
        self.notifier.post_milestone(
            "New Model Deployed",
            f"Model {event.new_model_id} is now live",
            platforms=[Platform.TELEGRAM, Platform.TWITTER]
        )
    
    def _on_retraining_failed(self, event: 'RetrainingEvent'):
        """Handle retraining failed event."""
        message = f"""Model retraining failed!

Profile: {event.profile_name}
Error: {event.error_message}

The system will continue using the previous model."""
        
        self.notifier.post_alert(
            "⚠️ Model Retraining Failed",
            message,
            platforms=[Platform.TELEGRAM]
        )
    
    def _format_validation_results(self, results: Optional[Dict]) -> str:
        """Format validation results for posting."""
        if not results:
            return "No validation data available"
        
        lines = []
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"• {key}: {value:.3f}")
            else:
                lines.append(f"• {key}: {value}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATA CONVERSION
    # =========================================================================
    
    def _convert_trade_record(self, record: 'TradeRecord') -> TradeData:
        """Convert internal TradeRecord to social media TradeData."""
        # Calculate duration
        if record.exit_time and record.entry_time:
            duration = int((record.exit_time - record.entry_time).total_seconds() / 60)
        else:
            duration = 0
        
        # Calculate P&L percentage
        if record.entry_price > 0:
            pnl_pct = (record.exit_price - record.entry_price) / record.entry_price * 100
            if record.direction == -1:  # Short
                pnl_pct = -pnl_pct
        else:
            pnl_pct = 0
        
        return TradeData(
            trade_id=record.trade_id,
            symbol=record.symbol,
            direction="LONG" if record.direction == 1 else "SHORT",
            entry_price=record.entry_price,
            exit_price=record.exit_price or record.entry_price,
            pnl=record.pnl,
            pnl_pct=pnl_pct,
            entry_time=record.entry_time,
            exit_time=record.exit_time or datetime.now(),
            duration_minutes=duration,
            signal_confidence=record.confidence,
        )
    
    def _create_performance_data(
        self, 
        period: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[PerformanceData]:
        """Create PerformanceData from monitor metrics."""
        if not self.performance_monitor:
            return None
        
        metrics = self.performance_monitor.get_current_metrics()
        summary = self.performance_monitor.get_performance_summary()
        
        # Get trade counts
        trades = self.performance_monitor.closed_trades
        period_trades = [
            t for t in trades 
            if t.exit_time and start_date <= t.exit_time <= end_date
        ]
        
        if not period_trades:
            return None
        
        winning = sum(1 for t in period_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in period_trades)
        
        return PerformanceData(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_trades=len(period_trades),
            winning_trades=winning,
            total_pnl=total_pnl,
            total_pnl_pct=summary.get('total_return_pct', 0),
            win_rate=metrics.get('win_rate', 0),
            profit_factor=metrics.get('profit_factor', 1),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            max_drawdown=metrics.get('max_drawdown'),
            avg_trade_pnl=total_pnl / len(period_trades) if period_trades else 0,
            best_trade_pnl=max(t.pnl for t in period_trades) if period_trades else 0,
            worst_trade_pnl=min(t.pnl for t in period_trades) if period_trades else 0,
        )
    
    # =========================================================================
    # SCHEDULED POSTS
    # =========================================================================
    
    def post_daily_summary(self) -> Dict[Platform, bool]:
        """Post daily performance summary."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        perf_data = self._create_performance_data("daily", start_date, end_date)
        
        if not perf_data or perf_data.total_trades == 0:
            logger.info("No trades for daily summary")
            return {}
        
        # Generate chart
        chart_path = None
        if self.performance_monitor and self.config.attach_charts:
            equity_data = self.performance_monitor.equity_curve
            if equity_data:
                chart_path = self.chart_generator.generate_equity_curve(
                    equity_data,
                    title="Daily Equity Curve"
                )
        
        results = self.notifier.post_performance_update(
            perf_data,
            chart_path=chart_path
        )
        
        self.last_daily_post = datetime.now()
        return results
    
    def post_weekly_summary(self) -> Dict[Platform, bool]:
        """Post weekly performance summary."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        perf_data = self._create_performance_data("weekly", start_date, end_date)
        
        if not perf_data or perf_data.total_trades == 0:
            logger.info("No trades for weekly summary")
            return {}
        
        # Generate chart
        chart_path = None
        if self.performance_monitor and self.config.attach_charts:
            equity_data = self.performance_monitor.equity_curve
            if equity_data:
                chart_path = self.chart_generator.generate_equity_curve(
                    equity_data,
                    title="Weekly Equity Curve"
                )
        
        results = self.notifier.post_performance_update(
            perf_data,
            chart_path=chart_path
        )
        
        self.last_weekly_post = datetime.now()
        return results
    
    def post_monthly_summary(self) -> Dict[Platform, bool]:
        """Post monthly performance summary."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        perf_data = self._create_performance_data("monthly", start_date, end_date)
        
        if not perf_data:
            return {}
        
        # Generate summary chart
        chart_path = None
        if self.performance_monitor and self.config.attach_charts:
            metrics = self.performance_monitor.get_current_metrics()
            metrics['total_trades'] = perf_data.total_trades
            metrics['total_pnl'] = perf_data.total_pnl
            metrics['total_pnl_pct'] = perf_data.total_pnl_pct
            
            chart_path = self.chart_generator.generate_performance_summary(
                metrics,
                period="Monthly"
            )
        
        return self.notifier.post_performance_update(
            perf_data,
            chart_path=chart_path
        )
    
    # =========================================================================
    # MILESTONES
    # =========================================================================
    
    def _check_milestones(self):
        """Check and post milestone achievements."""
        if not self.performance_monitor:
            return
        
        summary = self.performance_monitor.get_performance_summary()
        
        # Check trade count milestones
        total_trades = summary.get('total_trades', 0)
        for threshold in self.milestone_thresholds['trades']:
            milestone_key = f"trades_{threshold}"
            if total_trades >= threshold and milestone_key not in self.achieved_milestones:
                self.achieved_milestones.add(milestone_key)
                self.notifier.post_milestone(
                    f"{threshold} Trades Completed! 🎯",
                    f"Our trading system has now executed {total_trades} trades.",
                )
        
        # Check P&L milestones
        total_pnl = summary.get('total_pnl', 0)
        for threshold in self.milestone_thresholds['pnl']:
            milestone_key = f"pnl_{threshold}"
            if total_pnl >= threshold and milestone_key not in self.achieved_milestones:
                self.achieved_milestones.add(milestone_key)
                self.notifier.post_milestone(
                    f"${threshold:,} Profit Milestone! 💰",
                    f"Total profits have reached ${total_pnl:,.2f}",
                )
    
    # =========================================================================
    # BACKGROUND SCHEDULER
    # =========================================================================
    
    def start_scheduler(
        self, 
        daily_hour: int = 22,
        weekly_day: int = 6,  # Sunday
        weekly_hour: int = 20
    ):
        """
        Start background scheduler for periodic posts.
        
        Args:
            daily_hour: Hour (UTC) for daily summary (default: 22:00)
            weekly_day: Day for weekly summary (0=Mon, 6=Sun)
            weekly_hour: Hour for weekly summary
        """
        self._running = True
        self._daily_hour = daily_hour
        self._weekly_day = weekly_day
        self._weekly_hour = weekly_hour
        
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        
        logger.info(f"Social media scheduler started (daily@{daily_hour}:00, weekly@day{weekly_day} {weekly_hour}:00)")
    
    def stop_scheduler(self):
        """Stop background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Social media scheduler stopped")
    
    def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._running:
            try:
                now = datetime.now()
                
                # Check for daily post
                if (now.hour == self._daily_hour and 
                    (not self.last_daily_post or 
                     (now - self.last_daily_post).days >= 1)):
                    logger.info("Posting daily summary...")
                    self.post_daily_summary()
                
                # Check for weekly post
                if (now.weekday() == self._weekly_day and
                    now.hour == self._weekly_hour and
                    (not self.last_weekly_post or
                     (now - self.last_weekly_post).days >= 6)):
                    logger.info("Posting weekly summary...")
                    self.post_weekly_summary()
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            time.sleep(60)  # Check every minute
    
    # =========================================================================
    # MANUAL POSTING
    # =========================================================================
    
    def post_custom(
        self, 
        message: str, 
        platforms: Optional[List[Platform]] = None,
        image_path: Optional[str] = None
    ) -> Dict[Platform, bool]:
        """Post custom message to platforms."""
        results = {}
        target_platforms = platforms or list(self.notifier.platforms.keys())
        
        for platform in target_platforms:
            if image_path:
                success = self.notifier.platforms[platform].post_with_image(message, image_path)
            else:
                success = self.notifier.platforms[platform].post_text(message)
            results[platform] = success
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            'platforms': self.notifier.get_platform_status(),
            'connected_components': {
                'performance_monitor': self.performance_monitor is not None,
                'retraining_scheduler': self.retraining_scheduler is not None,
            },
            'scheduler_running': self._running,
            'last_daily_post': self.last_daily_post.isoformat() if self.last_daily_post else None,
            'last_weekly_post': self.last_weekly_post.isoformat() if self.last_weekly_post else None,
            'posted_trades_count': len(self.posted_trade_ids),
            'achieved_milestones': list(self.achieved_milestones),
        }


# =============================================================================
# QUICK SETUP FUNCTION
# =============================================================================

def setup_social_notifications(
    performance_monitor: Optional['PerformanceMonitor'] = None,
    retraining_scheduler: Optional['RetrainingScheduler'] = None,
    config: Optional[NotificationConfig] = None,
    start_scheduler: bool = True,
    **scheduler_kwargs
) -> SocialMediaIntegration:
    """
    Quick setup for social media notifications.
    
    Args:
        performance_monitor: PerformanceMonitor to connect
        retraining_scheduler: RetrainingScheduler to connect
        config: Notification config (or uses env vars)
        start_scheduler: Whether to start background scheduler
        **scheduler_kwargs: Arguments for scheduler (daily_hour, weekly_day, etc.)
    
    Returns:
        Configured SocialMediaIntegration instance
    
    Example:
        integration = setup_social_notifications(
            performance_monitor=monitor,
            retraining_scheduler=scheduler,
            start_scheduler=True,
            daily_hour=22,
        )
    """
    # Create integration
    if config:
        integration = SocialMediaIntegration(config)
    else:
        integration = SocialMediaIntegration.from_env()
    
    # Connect components
    if performance_monitor:
        integration.connect_performance_monitor(performance_monitor)
    
    if retraining_scheduler:
        integration.connect_retraining_scheduler(retraining_scheduler)
    
    # Start scheduler
    if start_scheduler:
        integration.start_scheduler(**scheduler_kwargs)
    
    return integration