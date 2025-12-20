"""
Comprehensive Backtesting Reporter
===================================

Generates detailed reports and validates against acceptance gates.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "backtest_reports"
    generate_plots: bool = True
    generate_html: bool = True
    generate_json: bool = True
    generate_csv: bool = True
    plot_dpi: int = 150
    plot_style: str = "seaborn-v0_8-darkgrid"


@dataclass
class AcceptanceGate:
    """Acceptance criteria for production readiness."""
    # Performance thresholds
    min_sharpe_ratio: float = 1.5
    max_drawdown_pct: float = 20.0
    min_profit_factor: float = 1.5
    min_win_rate: float = 0.45
    
    # Walk-forward validation
    max_walk_forward_decay_pct: float = 25.0
    
    # Risk violations
    max_risk_violations: int = 0
    
    # Execution quality
    max_execution_failures: int = 0
    
    # Trade requirements
    min_trades: int = 30
    
    def validate(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Validate metrics against acceptance criteria.
        
        Returns:
            (passed, list of failures)
        """
        failures = []
        
        # Extract metrics
        trade_metrics = metrics.get('trade_metrics', {})
        risk_metrics = metrics.get('risk_metrics', {})
        
        # Check Sharpe ratio
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        if sharpe < self.min_sharpe_ratio:
            failures.append(f"Sharpe ratio {sharpe:.2f} < {self.min_sharpe_ratio}")
        
        # Check max drawdown
        max_dd = abs(risk_metrics.get('max_drawdown_pct', 0))
        if max_dd > self.max_drawdown_pct:
            failures.append(f"Max drawdown {max_dd:.2f}% > {self.max_drawdown_pct}%")
        
        # Check profit factor
        pf = trade_metrics.get('profit_factor', 0)
        if pf < self.min_profit_factor:
            failures.append(f"Profit factor {pf:.2f} < {self.min_profit_factor}")
        
        # Check win rate
        wr = trade_metrics.get('win_rate', 0)
        if wr < self.min_win_rate:
            failures.append(f"Win rate {wr:.2%} < {self.min_win_rate:.2%}")
        
        # Check minimum trades
        total_trades = trade_metrics.get('total_trades', 0)
        if total_trades < self.min_trades:
            failures.append(f"Total trades {total_trades} < {self.min_trades}")
        
        passed = len(failures) == 0
        return passed, failures


class BacktestReporter:
    """
    Generate comprehensive backtest reports.
    
    Features:
    - Performance summary
    - Trade analysis
    - Risk metrics
    - Equity curve plots
    - Drawdown analysis
    - Monthly returns heatmap
    - Trade distribution
    - Acceptance gate validation
    - HTML report generation
    
    Usage:
        reporter = BacktestReporter(config)
        reporter.generate_report(results, metrics)
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        
        # Create output directory
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        try:
            plt.style.use(self.config.plot_style)
        except:
            logger.warning(f"Plot style '{self.config.plot_style}' not available")
    
    def generate_report(
        self,
        results: Dict,
        metrics: Dict,
        acceptance_gate: Optional[AcceptanceGate] = None
    ) -> Dict:
        """
        Generate comprehensive backtest report.
        
        Args:
            results: Backtest results dictionary
            metrics: Performance metrics dictionary
            acceptance_gate: Optional acceptance criteria
        
        Returns:
            Dictionary with report metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"backtest_report_{timestamp}"
        
        logger.info("=" * 80)
        logger.info("GENERATING BACKTEST REPORT")
        logger.info("=" * 80)
        
        # Validate against acceptance gate
        gate_result = None
        if acceptance_gate:
            passed, failures = acceptance_gate.validate(metrics)
            gate_result = {
                'passed': passed,
                'failures': failures
            }
            logger.info(f"\nAcceptance Gate: {'✅ PASSED' if passed else '❌ FAILED'}")
            if not passed:
                for failure in failures:
                    logger.info(f"  ❌ {failure}")
        
        # Generate plots
        plot_files = []
        if self.config.generate_plots:
            logger.info("\nGenerating plots...")
            plot_files = self._generate_plots(results, metrics, report_name)
        
        # Generate JSON report
        if self.config.generate_json:
            json_file = self._generate_json_report(
                results, metrics, gate_result, report_name
            )
            logger.info(f"JSON report: {json_file}")
        
        # Generate CSV exports
        if self.config.generate_csv:
            csv_files = self._generate_csv_exports(results, report_name)
            logger.info(f"CSV exports: {len(csv_files)} files")
        
        # Generate HTML report
        if self.config.generate_html:
            html_file = self._generate_html_report(
                results, metrics, gate_result, plot_files, report_name
            )
            logger.info(f"HTML report: {html_file}")
        
        logger.info("=" * 80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("=" * 80)
        
        return {
            'report_name': report_name,
            'timestamp': timestamp,
            'output_dir': str(self.output_path),
            'gate_result': gate_result
        }
    
    def _generate_plots(
        self,
        results: Dict,
        metrics: Dict,
        report_name: str
    ) -> List[str]:
        """Generate all plots."""
        plot_files = []
        
        # Build equity curve
        trades = results.get('trades', [])
        if trades:
            trades_df = pd.DataFrame(trades)
            initial_balance = results.get('config', {}).get('initial_balance', 10000)
            
            # Equity curve plot
            equity_file = self._plot_equity_curve(
                trades_df, initial_balance, report_name
            )
            if equity_file:
                plot_files.append(equity_file)
            
            # Drawdown plot
            dd_file = self._plot_drawdown(trades_df, initial_balance, report_name)
            if dd_file:
                plot_files.append(dd_file)
            
            # Trade distribution
            dist_file = self._plot_trade_distribution(trades_df, report_name)
            if dist_file:
                plot_files.append(dist_file)
            
            # Monthly returns heatmap
            heatmap_file = self._plot_monthly_returns(trades_df, initial_balance, report_name)
            if heatmap_file:
                plot_files.append(heatmap_file)
        
        return plot_files
    
    def _plot_equity_curve(
        self,
        trades_df: pd.DataFrame,
        initial_balance: float,
        report_name: str
    ) -> Optional[str]:
        """Plot equity curve."""
        try:
            if 'exit_time' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return None
            
            trades_df = trades_df.sort_values('exit_time')
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Build equity curve
            equity = [initial_balance]
            for pnl in trades_df['pnl']:
                equity.append(equity[-1] + pnl)
            
            times = [trades_df['exit_time'].iloc[0]] + trades_df['exit_time'].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(times, equity, linewidth=2, color='#2E86AB')
            ax.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(times, initial_balance, equity, 
                           where=[e >= initial_balance for e in equity],
                           alpha=0.3, color='green', label='Profit')
            ax.fill_between(times, initial_balance, equity,
                           where=[e < initial_balance for e in equity],
                           alpha=0.3, color='red', label='Loss')
            
            ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            filename = self.output_path / f"{report_name}_equity_curve.png"
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filename)
        
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None
    
    def _plot_drawdown(
        self,
        trades_df: pd.DataFrame,
        initial_balance: float,
        report_name: str
    ) -> Optional[str]:
        """Plot drawdown."""
        try:
            if 'exit_time' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return None
            
            trades_df = trades_df.sort_values('exit_time')
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Build equity curve
            equity = [initial_balance]
            for pnl in trades_df['pnl']:
                equity.append(equity[-1] + pnl)
            
            equity_series = pd.Series(equity)
            running_max = equity_series.expanding().max()
            drawdown = equity_series - running_max
            drawdown_pct = (drawdown / running_max) * 100
            
            times = [trades_df['exit_time'].iloc[0]] + trades_df['exit_time'].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(times, 0, drawdown_pct, color='red', alpha=0.3)
            ax.plot(times, drawdown_pct, linewidth=2, color='darkred')
            
            ax.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            filename = self.output_path / f"{report_name}_drawdown.png"
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filename)
        
        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
            return None
    
    def _plot_trade_distribution(
        self,
        trades_df: pd.DataFrame,
        report_name: str
    ) -> Optional[str]:
        """Plot trade P&L distribution."""
        try:
            if 'pnl' not in trades_df.columns:
                return None
            
            pnls = trades_df['pnl'].values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(pnls, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax1.axvline(x=np.mean(pnls), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
            ax1.set_title('Trade P&L Distribution', fontweight='bold')
            ax1.set_xlabel('P&L ($)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot([pnls[pnls > 0], pnls[pnls < 0]], 
                       labels=['Winners', 'Losers'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#2E86AB', alpha=0.7))
            ax2.set_title('Winners vs Losers', fontweight='bold')
            ax2.set_ylabel('P&L ($)')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = self.output_path / f"{report_name}_trade_distribution.png"
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filename)
        
        except Exception as e:
            logger.error(f"Error plotting trade distribution: {e}")
            return None
    
    def _plot_monthly_returns(
        self,
        trades_df: pd.DataFrame,
        initial_balance: float,
        report_name: str
    ) -> Optional[str]:
        """Plot monthly returns heatmap."""
        try:
            if 'exit_time' not in trades_df.columns or 'pnl' not in trades_df.columns:
                return None
            
            trades_df = trades_df.copy()
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df.set_index('exit_time', inplace=True)
            
            # Calculate monthly returns
            monthly_pnl = trades_df['pnl'].resample('M').sum()
            
            if len(monthly_pnl) < 2:
                return None
            
            # Create pivot table for heatmap
            monthly_pnl_df = monthly_pnl.to_frame()
            monthly_pnl_df['year'] = monthly_pnl_df.index.year
            monthly_pnl_df['month'] = monthly_pnl_df.index.month
            
            pivot = monthly_pnl_df.pivot_table(
                values='pnl',
                index='year',
                columns='month',
                aggfunc='sum'
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create heatmap
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            
            # Set ticks
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticklabels(pivot.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('P&L ($)', rotation=270, labelpad=20)
            
            # Add values
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if not np.isnan(pivot.values[i, j]):
                        text = ax.text(j, i, f'${pivot.values[i, j]:.0f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            plt.tight_layout()
            
            filename = self.output_path / f"{report_name}_monthly_returns.png"
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filename)
        
        except Exception as e:
            logger.error(f"Error plotting monthly returns: {e}")
            return None
    
    def _generate_json_report(
        self,
        results: Dict,
        metrics: Dict,
        gate_result: Optional[Dict],
        report_name: str
    ) -> str:
        """Generate JSON report."""
        report = {
            'report_name': report_name,
            'timestamp': datetime.now().isoformat(),
            'config': results.get('config', {}),
            'metrics': metrics,
            'summary': results.get('summary', {}),
            'acceptance_gate': gate_result
        }
        
        filename = self.output_path / f"{report_name}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filename)
    
    def _generate_csv_exports(
        self,
        results: Dict,
        report_name: str
    ) -> List[str]:
        """Generate CSV exports."""
        csv_files = []
        
        # Export trades
        trades = results.get('trades', [])
        if trades:
            trades_df = pd.DataFrame(trades)
            filename = self.output_path / f"{report_name}_trades.csv"
            trades_df.to_csv(filename, index=False)
            csv_files.append(str(filename))
        
        # Export decisions
        decisions = results.get('decisions', [])
        if decisions:
            decisions_df = pd.DataFrame(decisions)
            filename = self.output_path / f"{report_name}_decisions.csv"
            decisions_df.to_csv(filename, index=False)
            csv_files.append(str(filename))
        
        return csv_files
    
    def _generate_html_report(
        self,
        results: Dict,
        metrics: Dict,
        gate_result: Optional[Dict],
        plot_files: List[str],
        report_name: str
    ) -> str:
        """Generate HTML report."""
        html = self._build_html_report(results, metrics, gate_result, plot_files)
        
        filename = self.output_path / f"{report_name}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filename)
    
    def _build_html_report(
        self,
        results: Dict,
        metrics: Dict,
        gate_result: Optional[Dict],
        plot_files: List[str]
    ) -> str:
        """Build HTML report content."""
        trade_metrics = metrics.get('trade_metrics', {})
        risk_metrics = metrics.get('risk_metrics', {})
        return_metrics = metrics.get('return_metrics', {})
        exec_metrics = metrics.get('execution_metrics', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #2E86AB; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .gate-pass {{ background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin: 20px 0; }}
        .gate-fail {{ background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; margin: 20px 0; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2E86AB; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {self._html_acceptance_gate(gate_result) if gate_result else ''}
        
        <h2>📈 Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if return_metrics.get('total_return', 0) > 0 else 'negative'}">
                    ${return_metrics.get('total_return', 0):,.2f} ({return_metrics.get('total_return_pct', 0):.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{risk_metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{risk_metrics.get('max_drawdown_pct', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{trade_metrics.get('win_rate', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{trade_metrics.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{trade_metrics.get('total_trades', 0)}</div>
            </div>
        </div>
        
        <h2>📊 Charts</h2>
        {self._html_plots(plot_files)}
        
        <h2>💼 Trade Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Trades</td><td>{trade_metrics.get('total_trades', 0)}</td></tr>
            <tr><td>Winning Trades</td><td>{trade_metrics.get('winning_trades', 0)}</td></tr>
            <tr><td>Losing Trades</td><td>{trade_metrics.get('losing_trades', 0)}</td></tr>
            <tr><td>Win Rate</td><td>{trade_metrics.get('win_rate', 0):.2%}</td></tr>
            <tr><td>Average Win</td><td>${trade_metrics.get('avg_win', 0):.2f}</td></tr>
            <tr><td>Average Loss</td><td>${trade_metrics.get('avg_loss', 0):.2f}</td></tr>
            <tr><td>Largest Win</td><td>${trade_metrics.get('largest_win', 0):.2f}</td></tr>
            <tr><td>Largest Loss</td><td>${trade_metrics.get('largest_loss', 0):.2f}</td></tr>
            <tr><td>Profit Factor</td><td>{trade_metrics.get('profit_factor', 0):.2f}</td></tr>
            <tr><td>Expectancy</td><td>${trade_metrics.get('expectancy', 0):.2f}</td></tr>
        </table>
        
        <h2>⚠️ Risk Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Max Drawdown</td><td>${risk_metrics.get('max_drawdown', 0):.2f} ({risk_metrics.get('max_drawdown_pct', 0):.2f}%)</td></tr>
            <tr><td>Sharpe Ratio</td><td>{risk_metrics.get('sharpe_ratio', 0):.2f}</td></tr>
            <tr><td>Sortino Ratio</td><td>{risk_metrics.get('sortino_ratio', 0):.2f}</td></tr>
            <tr><td>Calmar Ratio</td><td>{risk_metrics.get('calmar_ratio', 0):.2f}</td></tr>
            <tr><td>VaR (95%)</td><td>${risk_metrics.get('var_95', 0):.2f}</td></tr>
            <tr><td>CVaR (95%)</td><td>${risk_metrics.get('cvar_95', 0):.2f}</td></tr>
            <tr><td>Max Consecutive Wins</td><td>{risk_metrics.get('max_consecutive_wins', 0)}</td></tr>
            <tr><td>Max Consecutive Losses</td><td>{risk_metrics.get('max_consecutive_losses', 0)}</td></tr>
        </table>
        
        <h2>⚙️ Execution Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Commission</td><td>${exec_metrics.get('total_commission', 0):.2f}</td></tr>
            <tr><td>Total Slippage</td><td>{exec_metrics.get('total_slippage_pips', 0):.1f} pips</td></tr>
            <tr><td>Average Slippage</td><td>{exec_metrics.get('avg_slippage_pips', 0):.2f} pips</td></tr>
        </table>
    </div>
</body>
</html>
"""
        return html
    
    def _html_acceptance_gate(self, gate_result: Dict) -> str:
        """Generate HTML for acceptance gate result."""
        if gate_result['passed']:
            return """
            <div class="gate-pass">
                <h3>✅ Acceptance Gate: PASSED</h3>
                <p>All acceptance criteria met. System is ready for production.</p>
            </div>
            """
        else:
            failures_html = ''.join([f'<li>{f}</li>' for f in gate_result['failures']])
            return f"""
            <div class="gate-fail">
                <h3>❌ Acceptance Gate: FAILED</h3>
                <p>The following criteria were not met:</p>
                <ul>{failures_html}</ul>
            </div>
            """
    
    def _html_plots(self, plot_files: List[str]) -> str:
        """Generate HTML for plots."""
        html = ""
        for plot_file in plot_files:
            plot_name = Path(plot_file).stem
            html += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{Path(plot_file).name}" alt="{plot_name}">
            </div>
            """
        return html
