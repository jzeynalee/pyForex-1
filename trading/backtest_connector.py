
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any, NamedTuple
from dataclasses import dataclass, field

from trading.mt5_connector import OrderResult, AccountInfo

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    spread_pips: float = 1.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.0
    latency_ms: int = 0
    symbol: str = "EURUSD"

class BacktestConnector:
    """
    Simulates MT5Connector for backtesting.
    Holds historical data and simulates order execution and price updates.
    """
    def __init__(self, data: pd.DataFrame, config: BacktestConfig):
        self.data = data.sort_index()
        self.config = config
        
        # State
        self.current_idx = 0
        self.current_time = self.data.index[0]
        self.current_price_close = self.data['close'].iloc[0]
        self.current_price_high = self.data['high'].iloc[0]
        self.current_price_low = self.data['low'].iloc[0]
        
        # Account
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        
        # Positions and History
        self.positions: List[Dict] = []
        self.history: List[Dict] = []
        self.ticket_counter = 1000
        
        self.connected = True
        
        logger.info(f"BacktestConnector initialized with {len(data)} bars. Start: {self.current_time}")

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def ensure_connected(self) -> bool:
        return True

    def get_account_info(self) -> Optional[AccountInfo]:
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0.0, # Simplified
            free_margin=self.equity,
            profit=self.equity - self.config.initial_balance
        )

    def set_current_time(self, time: datetime):
        """
        Advance the connector to a specific time.
        Finds the row in data corresponding to or immediately preceding this time.
        """
        # Ideally, we iterate bar by bar.
        # This method might be used if we want to jump.
        # For sequential backtest, we'll use a `next_bar()` method.
        pass

    def next_bar(self) -> bool:
        """
        Advance to the next bar in history.
        Update prices, check SL/TP.
        Returns False if end of data.
        """
        if self.current_idx >= len(self.data) - 1:
            return False
        
        self.current_idx += 1
        row = self.data.iloc[self.current_idx]
        self.current_time = row.name if isinstance(row.name, datetime) else row['time'] # Handle index vs column
        
        # We use OHLC for SL/TP checking within the bar
        # Conservative simulation:
        # 1. Check if SL/TP hit by Low/High
        # 2. Update current price to Close for next decision
        
        self.current_price_close = row['close']
        self.current_price_high = row['high']
        self.current_price_low = row['low']
        
        self._check_sl_tp()
        self._update_equity()
        
        return True

    def get_data(self, n: int = 100, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Return the last n bars relative to current_idx.
        """
        start_idx = max(0, self.current_idx - n + 1)
        # Return slice including current_idx
        return self.data.iloc[start_idx : self.current_idx + 1].copy()
    
    def get_ohlcv(self, symbol: str = None, timeframe: str = None, count: int = 100) -> pd.DataFrame:
        """
        Return OHLCV data - alias for get_data to match MT5Connector interface.
        """
        return self.get_data(n=count, symbol=symbol, timeframe=timeframe)
    
    def get_account_balance(self) -> float:
        """Return current account balance."""
        return self.balance

    def get_current_price(self, symbol: Optional[str] = None) -> Optional[Dict[str, float]]:
        # Simulate bid/ask based on spread
        spread_val = self.config.spread_pips * 0.0001
        bid = self.current_price_close
        ask = bid + spread_val
        
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread_val,
            'time': self.current_time
        }

    def execute_order(
        self,
        signal: str = None,
        volume: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        symbol: Optional[str] = None,
        comment: str = "",
        **kwargs
    ) -> OrderResult:
        
        # Handle aliases from NeuralHybridStrategy
        if signal is None:
            signal = kwargs.get('direction')
        if sl == 0.0:
            sl = kwargs.get('stop_loss', 0.0)
        if tp == 0.0:
            tp = kwargs.get('take_profit', 0.0)
            
        if signal is None:
             return OrderResult(False, None, None, volume, "Missing signal/direction")

        self.ticket_counter += 1
        ticket = self.ticket_counter
        
        # Determine execution price (Bid for Sell, Ask for Buy)
        price_info = self.get_current_price()
        if not price_info:
            return OrderResult(False, None, None, volume, "No price")

        if signal.upper() == 'BUY':
            base_price = price_info['ask']
            # Slippage: buying at higher price
            slippage = np.random.normal(0, self.config.slippage_pips * 0.0001) if self.config.slippage_pips > 0 else 0
            exec_price = base_price + abs(slippage)
        else:
            base_price = price_info['bid']
            # Slippage: selling at lower price
            slippage = np.random.normal(0, self.config.slippage_pips * 0.0001) if self.config.slippage_pips > 0 else 0
            exec_price = base_price - abs(slippage)
            
        # Create position
        position = {
            'ticket': ticket,
            'symbol': symbol or self.config.symbol,
            'type': signal.upper(),
            'volume': volume,
            'price_open': exec_price,
            'price_current': exec_price,
            'sl': sl,
            'tp': tp,
            'profit': 0.0,
            'magic': 0,
            'time': self.current_time
        }
        
        self.positions.append(position)
        
        # Commission
        comm = self.config.commission_per_lot * volume
        self.balance -= comm
        
        logger.info(f"[BACKTEST] Opened {signal} {volume} @ {exec_price:.5f} (Slip: {abs(slippage)/0.0001:.1f}p) | SL: {sl}, TP: {tp}")
        
        return OrderResult(True, ticket, exec_price, volume, None)

    def entry(self, signal: str, volume: float, sl: float, tp: float) -> OrderResult:
        return self.execute_order(signal, volume, sl, tp)

    def get_open_positions(self, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        return self.positions

    def close_position(self, ticket: int) -> OrderResult:
        pos_idx = next((i for i, p in enumerate(self.positions) if p['ticket'] == ticket), None)
        if pos_idx is None:
            return OrderResult(False, None, None, 0, "Position not found")
        
        pos = self.positions.pop(pos_idx)
        
        # Close price
        price_info = self.get_current_price()
        if pos['type'] == 'BUY':
            close_price = price_info['bid']
            pnl_pips = (close_price - pos['price_open']) / 0.0001
        else:
            close_price = price_info['ask']
            pnl_pips = (pos['price_open'] - close_price) / 0.0001
            
        # Value per pip per lot approx $10 for standard pairs
        pip_value = 10.0 
        pnl = pnl_pips * pip_value * pos['volume']
        
        self.balance += pnl
        self._update_equity()
        
        # Record history
        history_record = {
            **pos,
            'price_close': close_price,
            'close_time': self.current_time,
            'final_pnl': pnl,
            'reason': 'SIGNAL'
        }
        self.history.append(history_record)
        
        logger.info(f"[BACKTEST] Closed {pos['type']} {pos['volume']} @ {close_price:.5f} | PnL: {pnl:.2f}")
        
        return OrderResult(True, ticket, close_price, pos['volume'], None)

    def _check_sl_tp(self):
        # Check if High/Low hit SL/TP for any position
        # For simplicity, we assume if both SL and TP are within the bar range, SL is hit first (conservative)
        # Or we could use open/close to guess.
        
        # Better approximation:
        # If Low <= SL (for Buy), hit.
        # If High >= TP (for Buy), hit.
        
        to_close = []
        
        for pos in self.positions:
            sl = pos['sl']
            tp = pos['tp']
            p_type = pos['type']
            
            close_price = None
            reason = None
            
            if p_type == 'BUY':
                # Check SL (Low)
                if sl > 0 and self.current_price_low <= sl:
                    close_price = sl
                    reason = 'SL'
                # Check TP (High) - only if SL not hit, or assume worst case
                elif tp > 0 and self.current_price_high >= tp:
                    close_price = tp
                    reason = 'TP'
            else: # SELL
                # Check SL (High)
                if sl > 0 and self.current_price_high >= sl:
                    close_price = sl
                    reason = 'SL'
                # Check TP (Low)
                elif tp > 0 and self.current_price_low <= tp:
                    close_price = tp
                    reason = 'TP'
            
            if close_price:
                to_close.append((pos, close_price, reason))

        for pos, price, reason in to_close:
             if pos in self.positions: # Double check
                self.positions.remove(pos)
                
                # Calculate PnL
                if pos['type'] == 'BUY':
                    pnl_pips = (price - pos['price_open']) / 0.0001
                else:
                    pnl_pips = (pos['price_open'] - price) / 0.0001
                
                pip_value = 10.0
                pnl = pnl_pips * pip_value * pos['volume']
                
                self.balance += pnl
                
                history_record = {
                    **pos,
                    'price_close': price,
                    'close_time': self.current_time,
                    'final_pnl': pnl,
                    'reason': reason
                }
                self.history.append(history_record)
                logger.info(f"[BACKTEST] {reason} hit for {pos['type']} {pos['volume']} @ {price:.5f} | PnL: {pnl:.2f}")

    def _update_equity(self):
        # Calculate unrealized PnL
        unrealized = 0.0
        pip_value = 10.0
        
        current_bid = self.current_price_close # Approximation
        current_ask = self.current_price_close + (self.config.spread_pips * 0.0001)
        
        for pos in self.positions:
            if pos['type'] == 'BUY':
                pnl_pips = (current_bid - pos['price_open']) / 0.0001
            else:
                pnl_pips = (pos['price_open'] - current_ask) / 0.0001
            
            unrealized += pnl_pips * pip_value * pos['volume']
            
        self.equity = self.balance + unrealized

