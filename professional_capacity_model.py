"""
Professional Capacity Model for Alpha Factory
Institutional-Grade Implementation with Realistic FX Constraints

PART I: PROPER CAPACITY MODEL
- Participation constraints (HARD GATE)
- Nonlinear slippage model (CONVEX)
- Opportunity exhaustion (SOFT CAP)
"""

import sys
sys.path.append('e:/pyProject/pyForex-1')

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RejectionReason(Enum):
    LIQUIDITY = "liquidity"
    EV = "ev"
    CROWDING = "crowding"
    PARTICIPATION = "participation"

@dataclass
class MarketState:
    """Market state for capacity decisions."""
    regime: str  # 'normal', 'high_vol', 'low_liquidity'
    bar_volume_notional: float
    spread: float
    volatility: float
    session_id: str
    
    def get_max_participation(self) -> float:
        """Get maximum participation based on regime."""
        participation_limits = {
            'normal': 0.02,      # 2%
            'high_vol': 0.01,    # 1%
            'low_liquidity': 0.005  # 0.5%
        }
        return participation_limits.get(self.regime, 0.02)

@dataclass
class CapacityResult:
    """Result of capacity check."""
    accepted: bool
    rejection_reason: Optional[RejectionReason]
    adjusted_ev: float
    adjusted_size: float
    participation: float
    slippage: float
    order_notional: float

class ProfessionalCapacityModel:
    """Institutional-grade capacity model with realistic FX constraints."""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.session_trade_counts = {}  # symbol -> count
        self.session_symbol_trades = {}  # symbol -> list of trades
        
        # Nonlinear slippage parameters
        self.slippage_alpha = 1.5  # FX realistic 1.3-1.7
        self.slippage_k = 0.0002   # Conservative calibration
        
        # Opportunity exhaustion limits
        self.max_trades_per_session = self._get_session_limits()
        
        logger.info(f"Professional capacity model initialized for ${capital:,.0f}")
    
    def _get_session_limits(self) -> Dict[str, int]:
        """Get opportunity exhaustion limits based on capital."""
        if self.capital < 500000:
            return {'unlimited': True}
        elif self.capital <= 5000000:
            return {'max_per_symbol': 3}
        else:
            return {'max_per_symbol': 1}
    
    def detect_market_regime(self, row: pd.Series) -> str:
        """Detect market regime for participation limits."""
        volatility = row.get('volatility', 0.0001)
        volume = row.get('tick_volume', 1000)
        spread = row.get('spread', 1.0)
        
        # Regime detection logic
        if volatility > 0.0015:  # High volatility
            return 'high_vol'
        elif volume < 500 or spread > 3.0:  # Low liquidity
            return 'low_liquidity'
        else:
            return 'normal'
    
    def calculate_bar_volume_notional(self, row: pd.Series) -> float:
        """Calculate realistic bar volume notional for EURUSD M15."""
        # Base volume for EURUSD M15 (conservative estimate)
        base_volume_per_tick = 1000000  # $1M per tick average
        
        tick_volume = row.get('tick_volume', 1000)
        price = row.get('close', 1.0)
        
        # Calculate notional volume
        bar_volume = base_volume_per_tick * tick_volume * price
        
        # Adjust for volatility (higher vol = higher volume)
        vol_multiplier = 1.0 + (row.get('volatility', 0.0001) / 0.0001) * 0.5
        
        return bar_volume * vol_multiplier
    
    def calculate_order_notional(self, signal: Dict[str, Any]) -> float:
        """Calculate order notional based on signal and capital."""
        base_size = signal.get('position_size', 0.02)  # 2% base
        return self.capital * base_size
    
    def calculate_nonlinear_slippage(self, participation: float, spread: float) -> float:
        """Calculate convex slippage model."""
        # Convex slippage: spread + k * participation^alpha
        slippage = spread + self.slippage_k * (participation ** self.slippage_alpha)
        
        # Convert spread from pips to notional
        spread_notional = spread * 0.0001  # Convert pips to price
        
        return spread_notional + slippage
    
    def check_session_limits(self, symbol: str, session_id: str) -> bool:
        """Check opportunity exhaustion limits."""
        if 'unlimited' in self.max_trades_per_session:
            return True
        
        max_trades = self.max_trades_per_session.get('max_per_symbol', 3)
        current_trades = len(self.session_symbol_trades.get(f"{session_id}_{symbol}", []))
        
        return current_trades < max_trades
    
    def capacity_check(self, signal: Dict[str, Any], market_state: MarketState) -> CapacityResult:
        """
        Professional capacity check with all three constraints.
        
        Returns:
            CapacityResult with acceptance decision and details
        """
        # Calculate order notional
        order_notional = self.calculate_order_notional(signal)
        
        # Calculate participation
        participation = order_notional / market_state.bar_volume_notional
        
        # (A) Participation constraint (HARD GATE)
        max_participation = market_state.get_max_participation()
        
        if participation > max_participation:
            logger.debug(f"Liquidity rejection: {participation:.4f} > {max_participation:.4f}")
            return CapacityResult(
                accepted=False,
                rejection_reason=RejectionReason.LIQUIDITY,
                adjusted_ev=0,
                adjusted_size=0,
                participation=participation,
                slippage=0,
                order_notional=order_notional
            )
        
        # (B) Nonlinear slippage model (CONVEX)
        slippage = self.calculate_nonlinear_slippage(participation, market_state.spread)
        
        # Adjust EV for slippage
        adjusted_ev = signal['expected_value'] - slippage
        
        if adjusted_ev <= 0:
            logger.debug(f"EV rejection: {adjusted_ev:.4f} <= 0")
            return CapacityResult(
                accepted=False,
                rejection_reason=RejectionReason.EV,
                adjusted_ev=adjusted_ev,
                adjusted_size=0,
                participation=participation,
                slippage=slippage,
                order_notional=order_notional
            )
        
        # (C) Opportunity exhaustion (SOFT CAP)
        symbol = signal.get('symbol', 'EURUSD')
        if not self.check_session_limits(symbol, market_state.session_id):
            logger.debug(f"Crowding rejection: session limit reached for {symbol}")
            return CapacityResult(
                accepted=False,
                rejection_reason=RejectionReason.CROWDING,
                adjusted_ev=adjusted_ev,
                adjusted_size=0,
                participation=participation,
                slippage=slippage,
                order_notional=order_notional
            )
        
        # Accept trade
        adjusted_size = signal['position_size']  # Keep original size for now
        
        return CapacityResult(
            accepted=True,
            rejection_reason=None,
            adjusted_ev=adjusted_ev,
            adjusted_size=adjusted_size,
            participation=participation,
            slippage=slippage,
            order_notional=order_notional
        )
    
    def update_session_tracking(self, signal: Dict[str, Any], session_id: str):
        """Update session tracking for opportunity exhaustion."""
        symbol = signal.get('symbol', 'EURUSD')
        key = f"{session_id}_{symbol}"
        
        if key not in self.session_symbol_trades:
            self.session_symbol_trades[key] = []
        
        self.session_symbol_trades[key].append({
            'timestamp': signal.get('timestamp'),
            'notional': self.calculate_order_notional(signal)
        })
    
    def get_capacity_metrics(self) -> Dict[str, Any]:
        """Get capacity utilization metrics."""
        total_session_trades = sum(len(trades) for trades in self.session_symbol_trades.values())
        
        return {
            'capital': self.capital,
            'session_limits': self.max_trades_per_session,
            'total_session_trades': total_session_trades,
            'active_symbols': len(set(key.split('_')[1] for key in self.session_symbol_trades.keys())),
            'slippage_params': {
                'alpha': self.slippage_alpha,
                'k': self.slippage_k
            }
        }

class ProfessionalCapacityBacktest:
    """Backtest with professional capacity model."""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.capacity_model = ProfessionalCapacityModel(capital)
        self.trades = []
        self.rejections = {reason: 0 for reason in RejectionReason}
        self.capacity_metrics = []
        
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with professional capacity constraints."""
        logger.info(f"Running professional capacity backtest for ${self.capital:,.0f}")
        
        for idx, row in data.iterrows():
            # Generate base signal (simplified for capacity testing)
            signals = self._generate_signals(row)
            
            # Create market state
            market_state = MarketState(
                regime=self.capacity_model.detect_market_regime(row),
                bar_volume_notional=self.capacity_model.calculate_bar_volume_notional(row),
                spread=row.get('spread', 1.0),
                volatility=row.get('volatility', 0.0001),
                session_id=self._get_session_id(row)
            )
            
            # Process signals through capacity model
            for signal in signals:
                result = self.capacity_model.capacity_check(signal, market_state)
                
                if result.accepted:
                    # Execute trade
                    trade = self._execute_trade(signal, result, row)
                    self.trades.append(trade)
                    
                    # Update session tracking
                    self.capacity_model.update_session_tracking(signal, market_state.session_id)
                else:
                    # Record rejection
                    self.rejections[result.rejection_reason] += 1
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def _generate_signals(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate base signals for capacity testing."""
        # Simplified signal generation - focus on capacity constraints
        base_ev = 25.0 + np.random.normal(0, 5)  # Base EV around $25
        
        return [{
            'symbol': 'EURUSD',
            'direction': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'expected_value': max(0, base_ev),
            'position_size': 0.02,  # 2% base size
            'timestamp': row['time']
        }]
    
    def _execute_trade(self, signal: Dict[str, Any], result: CapacityResult, row: pd.Series) -> Dict[str, Any]:
        """Execute trade with capacity adjustments."""
        return {
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_price': row['close'],
            'expected_ev': signal['expected_value'],
            'adjusted_ev': result.adjusted_ev,
            'slippage': result.slippage,
            'participation': result.participation,
            'position_size': result.adjusted_size,
            'order_notional': result.order_notional,
            'timestamp': signal['timestamp']
        }
    
    def _get_session_id(self, row: pd.Series) -> str:
        """Get session identifier."""
        return f"session_{row['time'].date()}"
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive capacity metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_ev': 0,
                'avg_slippage': 0,
                'avg_participation': 0,
                'rejections': dict(self.rejections),
                'capacity_metrics': self.capacity_model.get_capacity_metrics()
            }
        
        # Calculate trade metrics
        total_trades = len(self.trades)
        avg_ev = np.mean([t['adjusted_ev'] for t in self.trades])
        avg_slippage = np.mean([t['slippage'] for t in self.trades])
        avg_participation = np.mean([t['participation'] for t in self.trades])
        
        # Calculate P&L (simplified)
        total_pnl = sum(t['adjusted_ev'] for t in self.trades)
        
        return {
            'total_trades': total_trades,
            'win_rate': 0.48,  # Simplified
            'total_pnl': total_pnl,
            'avg_ev': avg_ev,
            'avg_slippage': avg_slippage,
            'avg_participation': avg_participation,
            'rejections': dict(self.rejections),
            'capacity_metrics': self.capacity_model.get_capacity_metrics(),
            'trade_details': self.trades
        }

def run_professional_capacity_test():
    """Run professional capacity validation across multiple notional sizes."""
    print("=" * 80)
    print("PROFESSIONAL CAPACITY MODEL VALIDATION")
    print("Institutional-Grade Implementation")
    print("=" * 80)
    
    # Test across realistic notional sizes
    notional_sizes = [100000, 500000, 1000000, 2500000, 5000000, 10000000]
    results = {}
    
    for capital in notional_sizes:
        print(f"\n🚀 Testing Capital: ${capital:,}")
        
        # Generate test data
        np.random.seed(42)  # Consistent results
        data = _generate_test_data(10000)
        
        # Run professional capacity backtest
        backtest = ProfessionalCapacityBacktest(capital)
        metrics = backtest.run_backtest(data)
        
        results[capital] = metrics
        
        # Display key results
        print(f"   Trades: {metrics['total_trades']:,}")
        print(f"   Avg EV: ${metrics['avg_ev']:.2f}")
        print(f"   Avg Slippage: ${metrics['avg_slippage']:.4f}")
        print(f"   Avg Participation: {metrics['avg_participation']:.4f}")
        print(f"   Rejections: {sum(metrics['rejections'].values())}")
        
        # Show rejection breakdown
        for reason, count in metrics['rejections'].items():
            if count > 0:
                print(f"     {reason.value}: {count}")
    
    # Generate capacity analysis
    return _generate_capacity_analysis(results)

def _generate_test_data(n_bars: int) -> pd.DataFrame:
    """Generate realistic test data for capacity testing."""
    np.random.seed(42)
    
    # Time series
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='15min')
    
    # Price data (random walk)
    price_changes = np.random.normal(0, 0.0001, n_bars)
    prices = 1.1000 + np.cumsum(price_changes)
    
    # Volume data (realistic for EURUSD M15)
    base_volume = 1000
    volume_variation = np.random.lognormal(0, 0.5, n_bars)
    volumes = base_volume * volume_variation
    
    # Volatility data
    volatilities = np.abs(np.random.normal(0.0001, 0.00005, n_bars))
    
    # Spread data (realistic for EURUSD)
    spreads = np.random.choice([0.5, 1.0, 1.5, 2.0], n_bars, p=[0.3, 0.4, 0.2, 0.1])
    
    return pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 0.0005, n_bars),
        'low': prices - np.random.uniform(0, 0.0005, n_bars),
        'close': prices,
        'tick_volume': volumes,
        'spread': spreads,
        'volatility': volatilities
    })

def _generate_capacity_analysis(results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive capacity analysis."""
    print(f"\n📈 GENERATING PROFESSIONAL CAPACITY ANALYSIS...")
    
    # Calculate degradation metrics
    base_trades = results[100000]['total_trades']
    base_ev = results[100000]['avg_ev']
    
    analysis = {
        'trade_degradation': {},
        'ev_degradation': {},
        'participation_trend': {},
        'slippage_trend': {},
        'rejection_analysis': {},
        'capacity_breakpoint': None
    }
    
    for capital, metrics in results.items():
        # Trade degradation
        trade_deg = (base_trades - metrics['total_trades']) / base_trades
        analysis['trade_degradation'][capital] = trade_deg
        
        # EV degradation
        ev_deg = (base_ev - metrics['avg_ev']) / base_ev if base_ev > 0 else 0
        analysis['ev_degradation'][capital] = ev_deg
        
        # Participation trend
        analysis['participation_trend'][capital] = metrics['avg_participation']
        
        # Slippage trend
        analysis['slippage_trend'][capital] = metrics['avg_slippage']
        
        # Rejection analysis
        total_rejections = sum(metrics['rejections'].values())
        analysis['rejection_analysis'][capital] = {
            'total': total_rejections,
            'breakdown': metrics['rejections']
        }
    
    # Find capacity breakpoint (where trades drop significantly)
    for capital in sorted(results.keys()):
        if analysis['trade_degradation'][capital] > 0.5:  # 50% trade reduction
            analysis['capacity_breakpoint'] = capital
            break
    
    # Display results
    print(f"\n🎯 PROFESSIONAL CAPACITY ANALYSIS:")
    
    if analysis['capacity_breakpoint']:
        print(f"   Capacity Breakpoint: ${analysis['capacity_breakpoint']:,}")
        print(f"   ✅ Clear capacity limit detected")
    else:
        print(f"   Capacity Breakpoint: None")
        print(f"   ⚠️ No clear limit in tested range")
    
    print(f"\n📊 DEGRADATION PATTERNS:")
    for capital in sorted(results.keys()):
        trade_deg = analysis['trade_degradation'][capital] * 100
        ev_deg = analysis['ev_degradation'][capital] * 100
        participation = analysis['participation_trend'][capital] * 100
        slippage = analysis['slippage_trend'][capital] * 10000  # Convert to basis points
        
        print(f"   ${capital:,}:")
        print(f"     Trade Degradation: {trade_deg:.1f}%")
        print(f"     EV Degradation: {ev_deg:.1f}%")
        print(f"     Participation: {participation:.3f}%")
        print(f"     Slippage: {slippage:.1f} bps")
        
        # Show rejection breakdown
        rejections = analysis['rejection_analysis'][capital]
        if rejections['total'] > 0:
            print(f"     Rejections: {rejections['total']}")
            for reason, count in rejections['breakdown'].items():
                if count > 0:
                    print(f"       {reason.value}: {count}")
    
    return analysis

if __name__ == "__main__":
    run_professional_capacity_test()
