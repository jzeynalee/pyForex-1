# alpha_factory/backtest_metrics.py
"""
Comprehensive Backtesting Metrics for Alpha Factory System

This module provides detailed performance evaluation metrics for the Alpha Factory
system, including trading performance, risk analysis, decision quality, and system efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
import random


logger = logging.getLogger(__name__)

class PlaceboTester:
    """
    Validates strategy edge by comparing against random entry placebo tests.
    Disproves 'luck' by shuffling entry signals while keeping market data intact.
    """
    
    @staticmethod
    def run_randomized_entry_test(market_data: pd.DataFrame, 
                                  actual_trades: List[Dict], 
                                  n_simulations: int = 500) -> Dict[str, Any]:
        """
        Runs N simulations with random entry times but actual strategy exit logic.
        """
        if not actual_trades or len(market_data) < 100:
            return {'status': 'insufficient_data'}
            
        actual_pnl = sum(t['pnl'] for t in actual_trades)
        actual_sharpe = PlaceboTester._calculate_simple_sharpe(actual_trades)
        
        simulated_sharpes = []
        simulated_pnls = []
        
        avg_duration = int(np.mean([t.get('duration_bars', 5) for t in actual_trades]))
        
        logger.info(f"Running {n_simulations} placebo simulations...")
        
        valid_indices = list(range(len(market_data) - avg_duration - 1))
        
        for _ in range(n_simulations):
            # Pick N random entry points (N = number of actual trades)
            random_entries = random.sample(valid_indices, len(actual_trades))
            sim_trades = []
            
            for entry_idx in random_entries:
                entry_price = market_data.iloc[entry_idx]['close']
                exit_idx = entry_idx + avg_duration
                exit_price = market_data.iloc[exit_idx]['close']
                
                # Random direction to test structural bias
                direction = random.choice([1, -1]) 
                pnl = (exit_price - entry_price) * direction
                sim_trades.append({'pnl': pnl})
                
            simulated_sharpes.append(PlaceboTester._calculate_simple_sharpe(sim_trades))
            simulated_pnls.append(sum(t['pnl'] for t in sim_trades))
            
        # Calculate p-value: Fraction of random sims that beat the actual strategy
        sharpe_beat_ratio = sum(s > actual_sharpe for s in simulated_sharpes) / n_simulations
        pnl_beat_ratio = sum(p > actual_pnl for p in simulated_pnls) / n_simulations
        
        return {
            'actual_sharpe': actual_sharpe,
            'placebo_mean_sharpe': np.mean(simulated_sharpes),
            'placebo_std_sharpe': np.std(simulated_sharpes),
            'p_value_sharpe': sharpe_beat_ratio, # < 0.05 implies significance
            'p_value_pnl': pnl_beat_ratio,
            'is_significant': sharpe_beat_ratio < 0.05
        }

    @staticmethod
    def _calculate_simple_sharpe(trades):
        pnls = [t['pnl'] for t in trades]
        if not pnls or np.std(pnls) == 0: return 0
        return np.mean(pnls) / np.std(pnls)


class BacktestMetrics:
    """
    Comprehensive metrics calculator for Alpha Factory backtesting results.
    
    Provides detailed analysis of:
    - Trading performance (returns, drawdowns, win rates)
    - Risk metrics (Sharpe, Sortino, Calmar ratios)
    - Decision quality analysis
    - System performance and efficiency
    - Market regime performance
    - Feature importance analysis
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics = {}
        self.analysis_timestamp = datetime.now()
    
    def calculate_trading_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive trading performance metrics.
        
        Args:
            trades: List of trade dictionaries with pnl, duration, etc.
            
        Returns:
            Dictionary with trading metrics
        """
        if not trades:
            return {'error': 'No trades to analyze'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = df['pnl'].sum()
        gross_profit = df[df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade statistics
        avg_trade_pnl = df['pnl'].mean()
        avg_win_pnl = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss_pnl = df[df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Best and worst trades
        largest_win = df['pnl'].max()
        largest_loss = df['pnl'].min()
        
        # Duration analysis
        avg_duration = df['duration_bars'].mean() if 'duration_bars' in df.columns else 0
        avg_win_duration = df[df['pnl'] > 0]['duration_bars'].mean() if winning_trades > 0 else 0
        avg_loss_duration = df[df['pnl'] <= 0]['duration_bars'].mean() if losing_trades > 0 else 0
        
        # Trade distribution analysis
        pnl_std = df['pnl'].std()
        pnl_skew = df['pnl'].skew()
        pnl_kurt = df['pnl'].kurtosis()
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_streaks(df['pnl'] > 0)
        consecutive_losses = self._calculate_consecutive_streaks(df['pnl'] <= 0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win_pnl': avg_win_pnl,
            'avg_loss_pnl': avg_loss_pnl,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration': avg_duration,
            'avg_win_duration': avg_win_duration,
            'avg_loss_duration': avg_loss_duration,
            'pnl_std': pnl_std,
            'pnl_skew': pnl_skew,
            'pnl_kurtosis': pnl_kurt,
            'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
            'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
            'avg_consecutive_wins': np.mean(consecutive_wins) if consecutive_wins else 0,
            'avg_consecutive_losses': np.mean(consecutive_losses) if consecutive_losses else 0
        }
    
    def calculate_risk_metrics(self, equity_curve: List[Tuple[datetime, float]], 
                            risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Dictionary with risk metrics
        """
        if len(equity_curve) < 2:
            return {'error': 'Insufficient equity curve data'}
        
        # Extract equity values
        equity_values = [eq for _, eq in equity_curve]
        timestamps = [ts for ts, _ in equity_curve]
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Remove zero returns to avoid division by zero
        returns = returns[returns != 0]
        
        if len(returns) == 0:
            return {'error': 'No non-zero returns found'}
        
        # Basic return statistics
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        
        # Annualized return (assuming daily data)
        days = (timestamps[-1] - timestamps[0]).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = total_return
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = np.max(drawdown)
        max_drawdown_duration = self._calculate_drawdown_duration(peak, equity_values)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR (5% level)
        var_5 = np.percentile(returns, 5)
        cvar_5 = np.mean(returns[returns <= var_5])
        
        # Win/Loss ratio
        win_returns = returns[returns > 0]
        loss_returns = returns[returns < 0]
        
        avg_win_return = np.mean(win_returns) if len(win_returns) > 0 else 0
        avg_loss_return = np.mean(loss_returns) if len(loss_returns) > 0 else 0
        win_loss_ratio = abs(avg_win_return / avg_loss_return) if avg_loss_return != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'win_loss_ratio': win_loss_ratio,
            'avg_win_return': avg_win_return,
            'avg_loss_return': avg_loss_return,
            'positive_days': len(win_returns),
            'negative_days': len(loss_returns),
            'up_capture_ratio': self._calculate_capture_ratio(returns, 'up'),
            'down_capture_ratio': self._calculate_capture_ratio(returns, 'down')
        }
    
    def calculate_decision_quality_metrics(self, decisions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze decision quality and confidence.
        
        Args:
            decisions: List of decision dictionaries
            
        Returns:
            Dictionary with decision quality metrics
        """
        if not decisions:
            return {'error': 'No decisions to analyze'}
        
        df = pd.DataFrame(decisions)
        
        # Decision distribution
        decision_counts = df['decision'].value_counts()
        total_decisions = len(decisions)
        
        # Confidence analysis
        confidences = df['confidence'].values
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Confidence by decision type
        confidence_by_decision = {}
        for decision_type in decision_counts.index:
            type_confidences = df[df['decision'] == decision_type]['confidence']
            confidence_by_decision[decision_type] = {
                'avg': type_confidences.mean(),
                'std': type_confidences.std(),
                'count': len(type_confidences)
            }
        
        # Regime distribution
        regime_counts = df['regime'].value_counts()
        
        # Risk score analysis
        risk_scores = df['risk_score'].values
        avg_risk_score = np.mean(risk_scores)
        
        # Expected return vs actual return (if available)
        if 'expected_return' in df.columns and 'actual_return' in df.columns:
            correlation = np.corrcoef(df['expected_return'], df['actual_return'])[0, 1]
            mse = np.mean((df['expected_return'] - df['actual_return']) ** 2)
        else:
            correlation = 0
            mse = 0
        
        # Decision accuracy (if actual outcomes available)
        accuracy_metrics = self._calculate_decision_accuracy(df)
        
        return {
            'total_decisions': total_decisions,
            'decision_distribution': decision_counts.to_dict(),
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'confidence_by_decision': confidence_by_decision,
            'regime_distribution': regime_counts.to_dict(),
            'avg_risk_score': avg_risk_score,
            'expected_vs_actual_correlation': correlation,
            'expected_return_mse': mse,
            **accuracy_metrics
        }
    
    def calculate_system_performance_metrics(self, processing_times: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze system performance and efficiency.
        
        Args:
            processing_times: Dictionary of processing times by component
            
        Returns:
            Dictionary with system performance metrics
        """
        if not processing_times:
            return {'error': 'No processing time data'}
        
        total_time = sum(processing_times.values())
        
        # Time distribution
        time_distribution = {
            component: {
                'time': time_taken,
                'percentage': (time_taken / total_time) * 100
            }
            for component, time_taken in processing_times.items()
        }
        
        # Performance bottlenecks
        sorted_times = sorted(processing_times.items(), key=lambda x: x[1], reverse=True)
        bottlenecks = [
            {'component': comp, 'time': time, 'percentage': (time / total_time) * 100}
            for comp, time in sorted_times[:3]
        ]
        
        # Efficiency metrics
        avg_time_per_bar = total_time / 1000 if 'bars_processed' in processing_times else total_time
        
        return {
            'total_processing_time': total_time,
            'time_distribution': time_distribution,
            'bottlenecks': bottlenecks,
            'avg_time_per_bar': avg_time_per_bar,
            'slowest_component': sorted_times[0][0],
            'fastest_component': sorted_times[-1][0]
        }
    
    def calculate_regime_performance(self, trades: List[Dict], decisions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze performance by market regime.
        
        Args:
            trades: List of trade dictionaries
            decisions: List of decision dictionaries
            
        Returns:
            Dictionary with regime-specific performance
        """
        if not trades or not decisions:
            return {'error': 'Insufficient data for regime analysis'}
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        decisions_df = pd.DataFrame(decisions)
        
        # Merge trades with decisions
        # Use outer join to handle missing matches
        merged = pd.merge(trades_df, decisions_df, left_on='entry_time', right_on='timestamp', how='outer')
        
        if len(merged) == 0:
            return {'error': 'No matching trades and decisions'}
        
        # Check if regime column exists
        if 'regime' not in merged.columns:
            return {'error': 'No regime information available'}
        
        regime_performance = {}
        
        for regime in merged['regime'].unique():
            if pd.isna(regime):
                continue  # Skip NaN values
                
            regime_data = merged[merged['regime'] == regime]
            
            regime_trades = len(regime_data)
            winning_trades = len(regime_data[regime_data['pnl'] > 0]) if 'pnl' in regime_data.columns else 0
            regime_win_rate = winning_trades / regime_trades if regime_trades > 0 else 0
            
            regime_pnl = regime_data['pnl'].sum() if 'pnl' in regime_data.columns else 0
            regime_avg_pnl = regime_data['pnl'].mean() if 'pnl' in regime_data.columns else 0
            
            regime_avg_confidence = regime_data['confidence'].mean() if 'confidence' in regime_data.columns else 0
            regime_avg_risk = regime_data['risk_score'].mean() if 'risk_score' in regime_data.columns else 0
            
            # Decision distribution in this regime
            if 'decision' in regime_data.columns:
                regime_decisions = regime_data['decision'].value_counts().to_dict()
            else:
                regime_decisions = {}
            
            regime_performance[regime] = {
                'total_trades': regime_trades,
                'winning_trades': winning_trades,
                'win_rate': regime_win_rate,
                'total_pnl': regime_pnl,
                'avg_pnl': regime_avg_pnl,
                'avg_confidence': regime_avg_confidence,
                'avg_risk_score': regime_avg_risk,
                'decision_distribution': regime_decisions,
                'profit_factor': self._calculate_profit_factor(regime_data['pnl']) if 'pnl' in regime_data.columns else 0
            }
        
        return regime_performance
    
    def calculate_feature_importance_metrics(self, causality_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze feature importance and consistency over time.
        
        Args:
            causality_results: List of causality analysis results
            
        Returns:
            Dictionary with feature importance metrics
        """
        if not causality_results:
            return {'error': 'No causality results to analyze'}
        
        # Aggregate feature rankings across all analyses
        feature_rankings = {}
        feature_scores = {}
        
        for result in causality_results:
            if 'causal_ranking' in result:
                for feature, data in result['causal_ranking'].items():
                    if feature not in feature_rankings:
                        feature_rankings[feature] = []
                        feature_scores[feature] = []
                    
                    feature_rankings[feature].append(data['rank'])
                    feature_scores[feature].append(data['combined_score'])
        
        # Calculate consistency metrics
        feature_consistency = {}
        for feature in feature_rankings:
            rankings = feature_rankings[feature]
            scores = feature_scores[feature]
            
            feature_consistency[feature] = {
                'avg_rank': np.mean(rankings),
                'std_rank': np.std(rankings),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'appearances': len(rankings),
                'rank_consistency': 1 - (np.std(rankings) / len(rankings)),  # Lower std = more consistent
                'score_consistency': 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
            }
        
        # Sort by average score
        sorted_features = sorted(feature_consistency.items(), 
                               key=lambda x: x[1]['avg_score'], reverse=True)
        
        # Top features analysis
        top_features = dict(sorted_features[:20])
        
        # Feature stability (how often features appear in top rankings)
        top_10_features = [f[0] for f in sorted_features[:10]]
        
        return {
            'total_analyses': len(causality_results),
            'total_features': len(feature_consistency),
            'top_features': top_features,
            'feature_consistency': feature_consistency,
            'most_consistent': sorted(feature_consistency.items(), 
                                  key=lambda x: x[1]['rank_consistency'], reverse=True)[:10],
            'top_10_stability': self._calculate_feature_stability(feature_rankings, top_10_features)
        }
    
    def generate_comprehensive_report(self, trades: List[Dict], decisions: List[Dict],
                                    equity_curve: List[Tuple[datetime, float]],
                                    processing_times: Dict[str, float],
                                    causality_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report.
        
        Args:
            trades: List of trade dictionaries
            decisions: List of decision dictionaries
            equity_curve: Equity curve data
            processing_times: Processing time data
            causality_results: Causality analysis results
            
        Returns:
            Comprehensive metrics report
        """
        report = {
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'summary': {
                'total_trades': len(trades),
                'total_decisions': len(decisions),
                'analysis_period': {
                    'start': equity_curve[0][0].isoformat() if equity_curve else None,
                    'end': equity_curve[-1][0].isoformat() if equity_curve else None,
                    'duration_days': (equity_curve[-1][0] - equity_curve[0][0]).days if equity_curve else 0
                }
            },
            'trading_performance': self.calculate_trading_metrics(trades),
            'risk_metrics': self.calculate_risk_metrics(equity_curve),
            'decision_quality': self.calculate_decision_quality_metrics(decisions),
            'system_performance': self.calculate_system_performance_metrics(processing_times),
            'regime_performance': self.calculate_regime_performance(trades, decisions),
            'feature_analysis': self.calculate_feature_importance_metrics(causality_results)
        }
        
        # Add overall performance score
        report['overall_score'] = self._calculate_overall_score(report)
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = None) -> str:
        """
        Save comprehensive report to file.
        
        Args:
            report: Report dictionary
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"backtest_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Backtest report saved to {output_path}")
        return output_path
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a comprehensive summary of the backtest results."""
        print("\n" + "=" * 80)
        print("ALPHA FACTORY COMPREHENSIVE BACKTEST REPORT")
        print("=" * 80)
        
        # Summary
        summary = report.get('summary', {})
        print(f"\n📊 SUMMARY")
        print(f"   Analysis Date: {report['analysis_timestamp']}")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Total Decisions: {summary.get('total_decisions', 0)}")
        print(f"   Analysis Period: {summary.get('analysis_period', {}).get('duration_days', 0)} days")
        
        # Trading Performance
        trading = report.get('trading_performance', {})
        if 'error' not in trading:
            print(f"\n💰 TRADING PERFORMANCE")
            print(f"   Total P&L: ${trading.get('total_pnl', 0):,.2f}")
            print(f"   Win Rate: {trading.get('win_rate', 0):.1%}")
            print(f"   Profit Factor: {trading.get('profit_factor', 0):.2f}")
            print(f"   Avg Trade: ${trading.get('avg_trade_pnl', 0):.2f}")
            print(f"   Largest Win: ${trading.get('largest_win', 0):.2f}")
            print(f"   Largest Loss: ${trading.get('largest_loss', 0):.2f}")
        
        # Risk Metrics
        risk = report.get('risk_metrics', {})
        if 'error' not in risk:
            print(f"\n⚠️  RISK METRICS")
            print(f"   Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            print(f"   Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}")
            print(f"   Calmar Ratio: {risk.get('calmar_ratio', 0):.2f}")
            print(f"   Max Drawdown: {risk.get('max_drawdown', 0):.1%}")
            print(f"   Max DD Duration: {risk.get('max_drawdown_duration', 0)} days")
            print(f"   Volatility: {risk.get('volatility', 0):.1%}")
        
        # Decision Quality
        decisions = report.get('decision_quality', {})
        if 'error' not in decisions:
            print(f"\n🎯 DECISION QUALITY")
            print(f"   Avg Confidence: {decisions.get('avg_confidence', 0):.3f}")
            print(f"   Confidence Std: {decisions.get('confidence_std', 0):.3f}")
            print(f"   Avg Risk Score: {decisions.get('avg_risk_score', 0):.3f}")
            
            print(f"\n   Decision Distribution:")
            for decision, count in decisions.get('decision_distribution', {}).items():
                print(f"     {decision}: {count}")
        
        # System Performance
        system = report.get('system_performance', {})
        if 'error' not in system:
            print(f"\n⚡ SYSTEM PERFORMANCE")
            print(f"   Total Time: {system.get('total_processing_time', 0):.2f}s")
            print(f"   Bottleneck: {system.get('slowest_component', 'N/A')}")
            
            print(f"\n   Time Distribution:")
            for component, data in system.get('time_distribution', {}).items():
                print(f"     {component}: {data['time']:.2f}s ({data['percentage']:.1f}%)")
        
        # Overall Score
        score = report.get('overall_score', 0)
        print(f"\n🏆 OVERALL SCORE: {score:.1f}/100")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)
    
    def _calculate_consecutive_streaks(self, condition: pd.Series) -> List[int]:
        """Calculate lengths of consecutive streaks."""
        streaks = []
        current_streak = 0
        
        for value in condition:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    def _calculate_drawdown_duration(self, peak: np.ndarray, equity: np.ndarray) -> int:
        """Calculate maximum drawdown duration in days."""
        drawdown = (peak - equity) / peak
        in_drawdown = drawdown > 0.01  # 1% threshold
        
        max_duration = 0
        current_duration = 0
        
        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_capture_ratio(self, returns: np.ndarray, direction: str) -> float:
        """Calculate upside/downside capture ratio."""
        if direction == 'up':
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
        else:
            positive_returns = returns[returns < 0]
            negative_returns = returns[returns > 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        return positive_returns.mean() / abs(negative_returns.mean()) if len(positive_returns) > 0 else 0
    
    def _calculate_decision_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate decision accuracy metrics."""
        accuracy_metrics = {}
        
        # Direction accuracy (if direction was correct)
        if 'decision_direction' in df.columns and 'price_movement' in df.columns:
            correct_directions = df[df['decision_direction'] == df['price_movement']]
            direction_accuracy = len(correct_directions) / len(df) if len(df) > 0 else 0
            accuracy_metrics['direction_accuracy'] = direction_accuracy
        
        # Confidence calibration (higher confidence should correlate with better outcomes)
        if 'confidence' in df.columns and 'pnl' in df.columns:
            # Correlate confidence with actual P&L
            confidence_pnl_corr = np.corrcoef(df['confidence'], df['pnl'])[0, 1]
            accuracy_metrics['confidence_pnl_correlation'] = confidence_pnl_corr
        
        return accuracy_metrics
    
    def _calculate_profit_factor(self, pnl_series: pd.Series) -> float:
        """Calculate profit factor for a P&L series."""
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series <= 0].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_feature_stability(self, rankings: Dict[str, List[int]], top_features: List[str]) -> Dict[str, float]:
        """Calculate stability of top features."""
        stability = {}
        
        for feature in top_features:
            if feature in rankings:
                feature_ranks = rankings[feature]
                # Calculate how often feature appears in top 10
                top_10_appearances = sum(1 for rank in feature_ranks if rank <= 10)
                stability[feature] = top_10_appearances / len(feature_ranks)
        
        return stability
    
    def _calculate_overall_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0
        
        # Trading performance (40%)
        trading = report.get('trading_performance', {})
        if 'error' not in trading:
            win_rate = trading.get('win_rate', 0)
            profit_factor = min(trading.get('profit_factor', 0), 3) / 3  # Cap at 3
            score += (win_rate * 0.6 + profit_factor * 0.4) * 40
        
        # Risk metrics (30%)
        risk = report.get('risk_metrics', {})
        if 'error' not in risk:
            sharpe = min(risk.get('sharpe_ratio', 0), 3) / 3  # Cap at 3
            drawdown = max(0, 1 - risk.get('max_drawdown', 0))  # Lower drawdown is better
            score += (sharpe * 0.7 + drawdown * 0.3) * 30
        
        # Decision quality (20%)
        decisions = report.get('decision_quality', {})
        if 'error' not in decisions:
            confidence = decisions.get('avg_confidence', 0)
            score += confidence * 20
        
        # System performance (10%)
        system = report.get('system_performance', {})
        if 'error' not in system:
            # Score based on efficiency (lower time is better)
            max_acceptable_time = 60  # 60 seconds
            actual_time = system.get('total_processing_time', max_acceptable_time)
            efficiency = max(0, 1 - (actual_time / max_acceptable_time))
            score += efficiency * 10
        
        return min(100, max(0, score))
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        # Trading performance recommendations
        trading = report.get('trading_performance', {})
        if 'error' not in trading:
            if trading.get('win_rate', 0) < 0.4:
                recommendations.append("Consider tightening entry criteria to improve win rate")
            
            if trading.get('profit_factor', 0) < 1.5:
                recommendations.append("Improve risk management to increase profit factor")
            
            if trading.get('avg_trade_pnl', 0) < 0:
                recommendations.append("Review strategy - average trade is negative")
        
        # Risk recommendations
        risk = report.get('risk_metrics', {})
        if 'error' not in risk:
            if risk.get('sharpe_ratio', 0) < 1:
                recommendations.append("Improve risk-adjusted returns (Sharpe < 1)")
            
            if risk.get('max_drawdown', 0) > 0.2:
                recommendations.append("Implement better drawdown control (>20%)")
        
        # Decision quality recommendations
        decisions = report.get('decision_quality', {})
        if 'error' not in decisions:
            if decisions.get('avg_confidence', 0) < 0.6:
                recommendations.append("Increase decision confidence threshold")
            
            if decisions.get('confidence_std', 0) > 0.3:
                recommendations.append("Reduce confidence variability")
        
        # System performance recommendations
        system = report.get('system_performance', {})
        if 'error' not in system:
            bottleneck = system.get('slowest_component', '')
            if 'causality_analysis' in bottleneck:
                recommendations.append("Optimize causal analysis (major bottleneck)")
            
            if system.get('total_processing_time', 0) > 30:
                recommendations.append("Improve system performance (>30s per analysis)")
        
        return recommendations


def analyze_backtest_results(trades: List[Dict], decisions: List[Dict],
                           equity_curve: List[Tuple[datetime, float]],
                           processing_times: Dict[str, float],
                           causality_results: List[Dict],
                           market_data: pd.DataFrame = None,
                           save_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze complete backtest results.
    
    Args:
        trades: List of trade dictionaries
        decisions: List of decision dictionaries
        equity_curve: Equity curve data
        processing_times: Processing time data
        causality_results: Causality analysis results
        save_report: Whether to save report to file
        
    Returns:
        Comprehensive metrics report
    """
    metrics = BacktestMetrics()
    report = metrics.generate_comprehensive_report(
        trades, decisions, equity_curve, processing_times, causality_results
    )

    # Run Placebo Test
    if market_data is not None and len(trades) > 10:
        placebo_results = PlaceboTester.run_randomized_entry_test(market_data, trades)
        report['placebo_validation'] = placebo_results
        
        # Print Placebo Summary
        print(f"\n🎲 PLACEBO VALIDATION")
        print(f"   Actual Sharpe: {placebo_results['actual_sharpe']:.2f}")
        print(f"   Placebo Mean Sharpe: {placebo_results['placebo_mean_sharpe']:.2f}")
        print(f"   P-Value (Luck Factor): {placebo_results['p_value_sharpe']:.4f}")
        if placebo_results['is_significant']:
            print(f"   ✅ Strategy Statistically Significant (p < 0.05)")
        else:
            print(f"   ❌ Strategy Indistinguishable from Noise")
    
    if save_report:
        metrics.save_report(report)
    
    metrics.print_summary(report)
    return report
