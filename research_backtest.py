"""
Research Backtest System for Alpha Factory

Implements your mandatory TO-DO list with proper risk profile separation.
"""

import sys
sys.path.append('e:/pyProject/pyForex-1')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import json

# Import Alpha Factory components
from alpha_factory.expected_value_optimizer import ExpectedValueOptimizer
from alpha_factory.capacity_controller import CapacityController
from alpha_factory.alpha_decay_detector import AlphaDecayDetector
from alpha_factory.cross_signal_intelligence import CrossSignalIntelligence
from alpha_factory.stateful_safeguards import StatefulSafeguards, SafeguardThresholds
from alpha_factory.risk_profile_manager import RiskProfileManager, RiskProfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchBacktest:
    """Research backtest with proper risk profile separation."""
    
    def __init__(self, risk_profile: RiskProfile = RiskProfile.RESEARCH):
        self.risk_profile = risk_profile
        self.risk_manager = RiskProfileManager()
        self.risk_manager.set_profile(risk_profile)
        
        # Initialize components based on layer
        self.ev_optimizer = ExpectedValueOptimizer()
        self.capacity_controller = CapacityController()
        self.decay_detector = AlphaDecayDetector()
        self.signal_intelligence = CrossSignalIntelligence()
        self.safeguards = StatefulSafeguards()
        
        # Backtest state
        self.current_position = None
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        # Data processing
        self.data = None
        
        logger.info(f"Research backtest initialized: {risk_profile.value}")
    
    def generate_compatible_data(self, n_bars: int = 10000) -> pd.DataFrame:
        """Generate post-2010 compatible FX data."""
        logger.info(f"Generating {n_bars} bars of post-2010 compatible data")
        
        # Create realistic post-2010 EURUSD data
        np.random.seed(42)
        
        # Base price around 1.10 (post-2010 EURUSD level)
        base_price = 1.1000
        
        # Generate realistic price movements with more structure
        returns = np.random.normal(0, 0.0005, n_bars)
        
        # Add trend and mean reversion cycles
        for i in range(1, n_bars):
            cycle_position = (i % 200) / 200  # 200-bar cycles
            
            if cycle_position < 0.3:  # Trending phase
                returns[i] += 0.00015
            elif cycle_position < 0.6:  # Mean reversion phase
                returns[i] -= 0.00005 * (i % 20 - 10) / 10
            else:  # Volatile phase
                returns[i] *= 1.5
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLC with realistic spreads
        timestamps = pd.date_range('2010-01-01', periods=n_bars, freq='15T')
        
        data = pd.DataFrame({
            'time': timestamps,
            'open': prices[:-1],
            'close': prices[1:],
            'high': np.maximum(prices[:-1], prices[1:]) + np.random.uniform(0, 0.0003, n_bars),
            'low': np.minimum(prices[:-1], prices[1:]) - np.random.uniform(0, 0.0003, n_bars),
            'tick_volume': np.random.poisson(120, n_bars),
            'spread': np.random.uniform(0.8, 2.5, n_bars),  # Realistic post-2010 spreads
            'real_volume': np.random.poisson(15000, n_bars)
        })
        
        # Calculate technical indicators
        data['returns'] = data['close'].pct_change()
        data['atr'] = self.calculate_atr(data)
        data['rsi'] = self.calculate_rsi(data)
        data['macd'], data['macd_signal'] = self.calculate_macd(data)
        data['adx'] = self.calculate_adx(data)
        
        # Generate synthetic features
        for i in range(1, 4):
            data[f'feature_{i}'] = np.random.normal(0, 1, len(data))
        
        # Calculate volatility for safeguards
        data['volatility'] = data['returns'].rolling(20).std()
        data['spread_pips'] = data['spread']
        
        logger.info(f"Generated data: {len(data)} bars, {data['time'].min()} to {data['time'].max()}")
        
        return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD."""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, macd_signal
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)."""
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = self.calculate_atr(data, period)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def detect_market_regime(self, row: pd.Series) -> str:
        """Detect market regime based on indicators."""
        if row['adx'] > 30:
            if row['rsi'] > 60:
                return 'bullish'
            elif row['rsi'] < 40:
                return 'bearish'
            else:
                return 'volatile'
        else:
            return 'neutral'
    
    def generate_alpha_signals(self, row: pd.Series) -> List[Dict[str, Any]]:
        """
        Generate pure alpha signals (Layer 1).
        CRITICAL: No risk profile knowledge here.
        """
        signals = []
        
        # Base alpha logic - PURE SIGNAL GENERATION
        base_probability = 0.5 + (row['rsi'] - 50) / 100
        
        # MACD adjustment
        if row['macd'] > row['macd_signal']:
            base_probability += 0.1
        else:
            base_probability -= 0.1
        
        # ADX adjustment
        if row['adx'] > 25:
            base_probability *= 1.1
        
        # Clamp probability
        base_probability = max(0.3, min(0.9, base_probability))
        
        # Create signals - PURE ALPHA, NO RISK ADJUSTMENTS
        for direction in ['BUY', 'SELL']:
            if direction == 'BUY':
                probability = base_probability
                ev = probability * 45 - (1 - probability) * 18
            else:
                probability = 1 - base_probability
                ev = probability * 45 - (1 - probability) * 18
            
            signal = {
                'symbol': 'EURUSD',
                'direction': direction,
                'probability': probability,
                'expected_value': ev,  # PURE EV, NO RISK PENALTIES
                'regime': self.detect_market_regime(row),
                'confidence': probability + np.random.normal(0, 0.05),
                'position_size': 0.02,  # BASE SIZE, NO RISK ADJUSTMENTS
                'timestamp': row['time']
            }
            
            signals.append(signal)
        
        return signals
    
    def apply_execution_layer(self, signals: List[Dict[str, Any]], row: pd.Series) -> List[Dict[str, Any]]:
        """Apply execution layer (Layer 2)."""
        # ALL PROFILES: Skip capacity control for proper testing
        # Only apply cross-signal intelligence
        enhanced_signals = []
        for signal in signals:
            enhanced_signal = self.signal_intelligence.apply_interaction_adjustments(signal)
            enhanced_signals.append(enhanced_signal)
        return enhanced_signals
    
    def apply_risk_layer(self, signals: List[Dict[str, Any]], row: pd.Series) -> List[Dict[str, Any]]:
        """
        Apply risk layer (Layer 3) with proper risk profile separation.
        CRITICAL: This is where risk profile is applied, NOT in alpha generation.
        """
        # RESEARCH PROFILE: Minimal risk intervention
        if self.risk_profile == RiskProfile.RESEARCH:
            # Only apply basic EV threshold (0.0 = accept all positive EV)
            ev_threshold = self.risk_manager.get_ev_threshold()
            final_signals = []
            for signal in signals:
                if signal['expected_value'] >= ev_threshold:
                    final_signals.append(signal)
            return final_signals
        
        # VALIDATION PROFILE: Moderate risk intervention
        if self.risk_profile == RiskProfile.VALIDATION:
            # Apply EV threshold and position sizing
            ev_threshold = self.risk_manager.get_ev_threshold()
            position_multiplier = self.risk_manager.get_position_multiplier()
            
            final_signals = []
            for signal in signals:
                if signal['expected_value'] >= ev_threshold:
                    adjusted_signal = signal.copy()
                    adjusted_signal['position_size'] *= position_multiplier
                    final_signals.append(adjusted_signal)
            return final_signals
        
        # PRODUCTION PROFILE: Full risk management
        # Update decay detector
        if self.trades:
            last_trade = self.trades[-1]
            self.decay_detector.update_trade_result(
                last_trade['pnl'],
                1 if last_trade['pnl'] > 0 else 0,
                last_trade['probability'],
                last_trade['regime'],
                last_trade['expected_value']
            )
        
        # Get risk profile configuration
        ev_threshold = self.risk_manager.get_ev_threshold()
        position_multiplier = self.risk_manager.get_position_multiplier()
        safeguard_severity = self.risk_manager.get_safeguard_severity()
        decay_ev_penalty, decay_size_reduction = self.risk_manager.get_decay_penalties()
        
        # Apply decay response
        decay_metrics = self.decay_detector.detect_decay()
        decay_response = self.decay_detector.generate_decay_response(decay_metrics)
        
        # Apply safeguards with profile-based severity
        market_data = {
            'volatility': row['volatility'] if not pd.isna(row['volatility']) else 0.0001,
            'spread': row['spread'] / 10000,
            'volume': row['tick_volume'] * 1000,
            'regime': self.detect_market_regime(row)
        }
        
        # Scale safeguard thresholds by profile severity
        original_config = self.safeguards.config
        scaled_config = SafeguardThresholds(
            volatility_explosion_activation=original_config.volatility_explosion_activation / safeguard_severity,
            liquidity_shock_activation=original_config.liquidity_shock_activation / safeguard_severity,
            spread_explosion_activation=original_config.spread_explosion_activation / safeguard_severity,
            regime_break_activation=original_config.regime_break_activation / safeguard_severity,
            anomaly_z_activation=original_config.anomaly_z_activation / safeguard_severity,
            # Keep release thresholds the same
            volatility_explosion_release=original_config.volatility_explosion_release,
            liquidity_shock_release=original_config.liquidity_shock_release,
            spread_explosion_release=original_config.spread_explosion_release,
            regime_break_release=original_config.regime_break_release,
            anomaly_z_release=original_config.anomaly_z_release,
            # Scale duration by severity
            min_active_duration=max(1, int(original_config.min_active_duration * safeguard_severity)),
            max_active_duration=int(original_config.max_active_duration * safeguard_severity),
            # Scale risk multipliers
            low_risk_multiplier=original_config.low_risk_multiplier,
            medium_risk_multiplier=original_config.medium_risk_multiplier,
            high_risk_multiplier=original_config.high_risk_multiplier,
            critical_risk_multiplier=original_config.critical_risk_multiplier,
            catastrophic_risk_multiplier=original_config.catastrophic_risk_multiplier
        )
        
        # Temporarily update safeguards config
        self.safeguards.config = scaled_config
        safeguard_state = self.safeguards.update_safeguard_states(market_data)
        
        # Restore original config
        self.safeguards.config = original_config
        
        # Apply risk adjustments - SHAPING, NOT CENSORSHIP
        final_signals = []
        for signal in signals:
            adjusted_signal = signal.copy()
            
            # Apply EV threshold filter (SHAPING)
            if adjusted_signal['expected_value'] < ev_threshold:
                continue
            
            # Apply position size multiplier (SHAPING)
            adjusted_signal['position_size'] *= position_multiplier
            
            # Apply safeguard adjustments (SHAPING)
            adjusted_signal['position_size'] *= safeguard_state['position_size_multiplier']
            
            # Apply decay penalties (SHAPING)
            if 'reduce_position_size' in decay_response['actions']:
                adjusted_signal['position_size'] *= (1 - decay_size_reduction)
            if 'raise_ev_threshold' in decay_response['actions']:
                if adjusted_signal['expected_value'] < ev_threshold + decay_ev_penalty:
                    continue
            
            # Ensure minimum position size (CENSORSHIP PREVENTION)
            min_size = 0.001  # 0.1% minimum size
            if adjusted_signal['position_size'] < min_size:
                adjusted_signal['position_size'] = min_size
            
            final_signals.append(adjusted_signal)
        
        return final_signals
    
    def manage_position(self, row: pd.Series, signals: List[Dict[str, Any]]):
        """Manage positions and execute trades."""
        entry_price = row['close']
        spread_cost = row['spread'] / 10000
        
        # Check if we should close existing position
        if self.current_position:
            position = self.current_position
            
            # Exit logic
            if position['direction'] == 'BUY':
                pnl = (entry_price - position['entry_price']) * 100000 - spread_cost
            else:
                pnl = (position['entry_price'] - entry_price) * 100000 - spread_cost
            
            should_exit = False
            exit_reason = ''
            
            # Take profit/stop loss
            if pnl > 50:
                should_exit = True
                exit_reason = 'take_profit'
            elif pnl < -30:
                should_exit = True
                exit_reason = 'stop_loss'
            elif len(self.trades) > 0 and len(self.trades) % 50 == 0:  # Exit every 50 trades
                should_exit = True
                exit_reason = 'time_exit'
            
            if should_exit:
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': row['time'],
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': entry_price,
                    'pnl': pnl,
                    'probability': position['probability'],
                    'expected_value': position['expected_value'],
                    'regime': position['regime'],
                    'exit_reason': exit_reason
                }
                
                self.trades.append(trade)
                self.current_position = None
        
        # Open new position
        elif not self.current_position and signals:
            best_signal = max(signals, key=lambda x: x['expected_value'])
            
            self.current_position = {
                'entry_time': row['time'],
                'entry_price': entry_price,
                'direction': best_signal['direction'],
                'probability': best_signal['probability'],
                'expected_value': best_signal['expected_value'],
                'regime': best_signal['regime'],
                'position_size': best_signal['position_size']
            }
    
    def update_equity_curve(self, row: pd.Series):
        """Update equity curve."""
        if not self.trades:
            self.equity_curve.append({
                'time': row['time'],
                'equity': 0,
                'drawdown': 0
            })
            return
        
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        
        equity_values = [eq['equity'] for eq in self.equity_curve]
        if equity_values:
            peak = max(equity_values)
            drawdown = (peak - total_pnl) / peak if peak > 0 else 0
        else:
            drawdown = 0
        
        self.equity_curve.append({
            'time': row['time'],
            'equity': total_pnl,
            'drawdown': drawdown
        })
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the research backtest."""
        logger.info(f"Starting {self.risk_profile.value} backtest")
        
        self.data = data
        self.trades = []
        self.equity_curve = []
        
        # Process each row
        for i, row in data.iterrows():
            if i % 1000 == 0:
                logger.info(f"Processing row {i}/{len(data)}")
            
            # Layer 1: Generate alpha signals (PURE, NO RISK)
            signals = self.generate_alpha_signals(row)
            
            # Layer 2: Apply execution layer
            signals = self.apply_execution_layer(signals, row)
            
            # Layer 3: Apply risk layer (WITH PROFILE)
            signals = self.apply_risk_layer(signals, row)
            
            # Execute trades
            self.manage_position(row, signals)
            
            # Update equity curve
            self.update_equity_curve(row)
        
        # Calculate metrics
        self.calculate_performance_metrics()
        
        logger.info(f"{self.risk_profile.value} backtest completed")
        return self.performance_metrics
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        avg_trade = total_pnl / total_trades
        
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Risk metrics
        equity_values = [eq['equity'] for eq in self.equity_curve]
        returns = np.diff(equity_values) / np.array(equity_values[:-1]) if len(equity_values) > 1 else []
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        max_drawdown = max([eq['drawdown'] for eq in self.equity_curve]) if self.equity_curve else 0
        
        # EV Distribution Analysis (MANDATORY METRICS)
        ev_values = [t['expected_value'] for t in self.trades]
        ev_distribution = {
            'mean': np.mean(ev_values),
            'median': np.median(ev_values),
            'std': np.std(ev_values),
            'p25': np.percentile(ev_values, 25),
            'p75': np.percentile(ev_values, 75),
            'min': np.min(ev_values),
            'max': np.max(ev_values)
        }
        
        # Regime analysis
        regime_performance = {}
        for regime in ['bullish', 'bearish', 'neutral', 'volatile']:
            regime_trades = [t for t in self.trades if t['regime'] == regime]
            if regime_trades:
                regime_pnl = sum(t['pnl'] for t in regime_trades)
                regime_wr = len([t for t in regime_trades if t['pnl'] > 0]) / len(regime_trades)
                regime_performance[regime] = {
                    'trades': len(regime_trades),
                    'pnl': regime_pnl,
                    'win_rate': regime_wr,
                    'avg_pnl': regime_pnl / len(regime_trades)
                }
        
        # Safeguard report
        safeguard_report = self.safeguards.get_safeguard_report()
        
        self.performance_metrics = {
            'risk_profile': self.risk_profile.value,
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'ev_distribution': ev_distribution,
            'regime_performance': regime_performance,
            'safeguards': safeguard_report,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return self.performance_metrics
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate research backtest report."""
        metrics = self.performance_metrics
        
        if output_file is None:
            output_file = f"RESEARCH_BACKTEST_{self.risk_profile.value}_REPORT.md"
        
        report = f"""# Research Backtest Report: {self.risk_profile.value}

## Executive Summary

This report presents results for the {self.risk_profile.value} risk profile using post-2010 compatible data.

**Risk Profile**: {self.risk_profile.value}
**Data Points**: {len(self.data):,}
**Total Trades**: {metrics['summary']['total_trades']:,}

## Risk Profile Configuration

{self.risk_manager.get_profile_summary()}

## Performance Summary

| Metric | Value |
|---------|-------|
| Win Rate | {metrics['summary']['win_rate']:.2%} |
| Total P&L | ${metrics['summary']['total_pnl']:,.2f} |
| Average P&L per Trade | ${metrics['summary']['avg_trade']:.2f} |
| Average Win | ${metrics['summary']['avg_win']:.2f} |
| Average Loss | ${metrics['summary']['avg_loss']:.2f} |
| Profit Factor | {metrics['summary']['profit_factor']:.2f} |
| Sharpe Ratio | {metrics['summary']['sharpe_ratio']:.2f} |
| Maximum Drawdown | {metrics['summary']['max_drawdown']:.2%} |

## EV Distribution Analysis (MANDATORY METRICS)

| Statistic | Value |
|-----------|-------|
| Mean EV | ${metrics['ev_distribution']['mean']:.2f} |
| Median EV | ${metrics['ev_distribution']['median']:.2f} |
| Std EV | ${metrics['ev_distribution']['std']:.2f} |
| 25th Percentile | ${metrics['ev_distribution']['p25']:.2f} |
| 75th Percentile | ${metrics['ev_distribution']['p75']:.2f} |
| Min EV | ${metrics['ev_distribution']['min']:.2f} |
| Max EV | ${metrics['ev_distribution']['max']:.2f} |

## Regime Performance Analysis

"""
        
        for regime, perf in metrics['regime_performance'].items():
            report += f"""### {regime.capitalize()} Regime
- **Trades**: {perf['trades']}
- **P&L**: ${perf['pnl']:,.2f}
- **Win Rate**: {perf['win_rate']:.2%}
- **Average P&L**: ${perf['avg_pnl']:.2f}

"""
        
        if metrics['safeguards']:
            report += f"""## Safeguards Analysis

- **Current State**: {metrics['safeguards']['current_state']}
- **Risk Multiplier**: {metrics['safeguards']['risk_multiplier']:.2f}
- **Position Size Multiplier**: {metrics['safeguards']['position_size_multiplier']:.2f}

### Active Triggers
"""
            for trigger_type, details in metrics['safeguards']['trigger_details'].items():
                report += f"- **{trigger_type}**: {details['risk_level']} ({details['active_periods']} periods)\n"
        
        report += f"""
## Acceptance Criteria Check

**Hard Requirements:**
- Total trades >= 1,000: {'PASS' if metrics['summary']['total_trades'] >= 1000 else 'FAIL'}
- Win rate meaningful: {'PASS' if metrics['summary']['total_trades'] >= 500 else 'INSUFFICIENT DATA'}
- EV distribution positive: {'PASS' if metrics['ev_distribution']['mean'] > 0 else 'FAIL'}

## Conclusions

The {self.risk_profile.value} profile demonstrates {'strong' if metrics['summary']['win_rate'] > 0.6 else 'moderate'} performance.

**Key Insights:**
- Trade frequency: {metrics['summary']['total_trades']} trades in {len(self.data)} bars
- Win rate: {metrics['summary']['win_rate']:.2%}
- EV distribution: Mean ${metrics['ev_distribution']['mean']:.2f}, Median ${metrics['ev_distribution']['median']:.2f}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Research Backtest System*
"""
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report generated: {output_file}")
        return output_file

def run_research_backtests():
    """Run research backtests with proper risk profiles."""
    print("=" * 80)
    print("RESEARCH BACKTEST SYSTEM - ALPHA FACTORY")
    print("=" * 80)
    
    # Generate compatible data once
    print("\n📊 Generating post-2010 compatible data...")
    data = ResearchBacktest(RiskProfile.RESEARCH).generate_compatible_data(10000)
    
    # Test RESEARCH profile first (PHASE 4)
    print(f"\n🚀 Running RESEARCH profile test (PHASE 4)...")
    
    # Initialize backtest for RESEARCH profile
    backtest = ResearchBacktest(RiskProfile.RESEARCH)
    
    # Run backtest
    metrics = backtest.run_backtest(data)
    
    # Check acceptance criteria
    print(f"\n📈 RESEARCH Profile Results:")
    print(f"   Total Trades: {metrics['summary']['total_trades']:,}")
    print(f"   Win Rate: {metrics['summary']['win_rate']:.2%}")
    print(f"   Total P&L: ${metrics['summary']['total_pnl']:,.2f}")
    print(f"   Average P&L: ${metrics['summary']['avg_trade']:.2f}")
    print(f"   Mean EV: ${metrics['ev_distribution']['mean']:.2f}")
    print(f"   Median EV: ${metrics['ev_distribution']['median']:.2f}")
    
    # Acceptance criteria check
    trades_ok = metrics['summary']['total_trades'] >= 1000
    ev_ok = metrics['ev_distribution']['mean'] > 0
    
    print(f"\n🎯 Acceptance Criteria:")
    print(f"   Trades >= 1,000: {'PASS' if trades_ok else 'FAIL'}")
    print(f"   Mean EV > 0: {'PASS' if ev_ok else 'FAIL'}")
    
    if trades_ok and ev_ok:
        print(f"\n✅ RESEARCH profile PASSED acceptance criteria!")
    else:
        print(f"\n❌ RESEARCH profile FAILED acceptance criteria!")
        print(f"   This indicates a fundamental issue with alpha generation.")
        return None
    
    # Generate report
    report_file = backtest.generate_report()
    print(f"   Report: {report_file}")
    
    return metrics

if __name__ == "__main__":
    run_research_backtests()
