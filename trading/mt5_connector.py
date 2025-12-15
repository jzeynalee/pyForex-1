# trading/mt5_connector.py
"""
MetaTrader 5 connection and order execution.
"""
import pandas as pd
import logging
from typing import Optional, Dict, Any, NamedTuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed. Using mock connector.")


class OrderResult(NamedTuple):
    """Structured order result."""
    success: bool
    ticket: Optional[int]
    price: Optional[float]
    volume: float
    error: Optional[str]


@dataclass
class AccountInfo:
    """Account information."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float


class Timeframe(Enum):
    """MT5 timeframe mappings."""
    M1 = 1
    M5 = 5
    M15 = 15
    M30 = 30
    H1 = 60
    H4 = 240
    D1 = 1440


class MT5Connector:
    """
    MetaTrader 5 connection and trading interface.
    
    Provides:
    - Connection management
    - Market data retrieval
    - Order execution
    - Position management
    """
    
    # Timeframe mapping
    TF_MAP = {
        "M1": 1,   # mt5.TIMEFRAME_M1
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }
    
    def __init__(
        self,
        account: int = 0,
        password: str = "",
        server: str = "",
        path: str = "",
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        magic_number: int = 123456,
    ):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 library not available")    

        self.account = account
        self.password = password
        self.server = server
        self.path = path
        self.symbol = symbol
        self.timeframe = timeframe
        self.magic_number = magic_number
        
        self.connected = False
        self._timeframe_mt5 = self._get_mt5_timeframe(timeframe)
        
        self.connected = False
        self._timeframe_mt5 = self._get_mt5_timeframe(timeframe)
    
    def _get_mt5_timeframe(self, tf: str) -> int:
        """Convert string timeframe to MT5 constant."""
        tf_upper = tf.upper()
        if tf_upper not in self.TF_MAP:
            logger.warning(f"Unknown timeframe {tf}, defaulting to H1")
            return mt5.TIMEFRAME_H1
        
        # Map to MT5 constants
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(tf_upper, mt5.TIMEFRAME_H1)
    
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        # Initialize MT5
        init_args = {}
        if self.path:
            init_args['path'] = self.path
        
        if not mt5.initialize(**init_args):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        # Login
        if self.account and self.password:
            authorized = mt5.login(
                login=self.account,
                password=self.password,
                server=self.server,
            )
            
            if not authorized:
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
        
        self.connected = True
        logger.info(f"Connected to MT5 account {self.account}")
        return True
    
    def disconnect(self):
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if self.connected:
            # Verify connection is still alive
            term_info = mt5.terminal_info()
            if term_info is not None:
                return True
        
        return self.connect()
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get current account information."""
        if not self.ensure_connected():
            return None
        
        info = mt5.account_info()
        if info is None:
            return None
        
        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            profit=info.profit,
        )
    
    def get_data(
        self, 
        n: int = 100,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch latest candles as DataFrame.
        
        Args:
            n: Number of candles to fetch
            symbol: Override default symbol
            timeframe: Override default timeframe
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.ensure_connected():
            return pd.DataFrame()
        
        sym = symbol or self.symbol
        tf = self._get_mt5_timeframe(timeframe) if timeframe else self._timeframe_mt5
        
        rates = mt5.copy_rates_from_pos(sym, tf, 0, n)
        
        if rates is None:
            error = mt5.last_error()
            logger.warning(f"Failed to fetch data: {error}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename for consistency
        df.rename(columns={'volume': 'tick_volume'}, inplace=True, errors='ignore')
        
        return df
    
    def get_current_price(self, symbol: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get current bid/ask prices."""
        if not self.ensure_connected():
            return None
        
        sym = symbol or self.symbol
        tick = mt5.symbol_info_tick(sym)
        
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time),
        }
    
    def execute_order(
        self,
        signal: str,
        volume: float,
        sl: float,
        tp: float,
        symbol: Optional[str] = None,
        comment: str = "PyForex",
    ) -> OrderResult:
        """
        Execute a market order.
        
        Args:
            signal: 'BUY' or 'SELL'
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price
            symbol: Override default symbol
            comment: Order comment
        
        Returns:
            OrderResult with success status and details
        """
        if not self.ensure_connected():
            return OrderResult(False, None, None, volume, "Not connected")
        
        sym = symbol or self.symbol
        
        # Validate signal
        if signal.upper() not in ('BUY', 'SELL'):
            return OrderResult(False, None, None, volume, f"Invalid signal: {signal}")
        
        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if signal.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        
        # Get current price
        tick = mt5.symbol_info_tick(sym)
        if not tick:
            return OrderResult(False, None, None, volume, "Failed to get tick data")
        
        price = tick.ask if signal.upper() == 'BUY' else tick.bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            return OrderResult(False, None, None, volume, f"Order failed: {error}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                False, None, None, volume,
                f"Order rejected: {result.comment} (code: {result.retcode})"
            )
        
        logger.info(f"Order executed: {signal} {volume} {sym} @ {result.price}")
        
        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=result.volume,
            error=None,
        )
    
    # Alias for strategy compatibility
    def entry(
        self,
        signal: str,
        volume: float,
        sl: float,
        tp: float,
    ) -> OrderResult:
        """Alias for execute_order (strategy interface)."""
        return self.execute_order(signal, volume, sl, tp)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> list[Dict[str, Any]]:
        """Get all open positions."""
        if not self.ensure_connected():
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [
            {
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'price_open': p.price_open,
                'price_current': p.price_current,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.profit,
                'magic': p.magic,
                'time': datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]
    
    def close_position(self, ticket: int) -> OrderResult:
        """Close a specific position by ticket."""
        if not self.ensure_connected():
            return OrderResult(False, None, None, 0, "Not connected")
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(False, None, None, 0, f"Position {ticket} not found")
        
        pos = position[0]
        
        # Reverse direction to close
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "magic": self.magic_number,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            return OrderResult(False, None, None, pos.volume, f"Close failed: {error}")
        
        logger.info(f"Position {ticket} closed")
        return OrderResult(True, result.order, result.price, result.volume, None)
    
    def get_symbol_info(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get symbol trading specifications."""
        if not self.ensure_connected():
            return None
        
        sym = symbol or self.symbol
        info = mt5.symbol_info(sym)
        
        if info is None:
            return None
        
        return {
            'name': info.name,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_tick_value': info.trade_tick_value,
            'trade_tick_size': info.trade_tick_size,
            'contract_size': info.trade_contract_size,
        }


class MockMT5Connector:
    """
    Mock connector for testing without MT5 terminal.
    Simulates market data and order execution.
    """
    
    def __init__(self, symbol: str = "EURUSD", **kwargs):
        self.symbol = symbol
        self.connected = True
        self._positions: list[Dict] = []
        self._ticket_counter = 1000
        
        # Simulated balance
        self._balance = 10000.0
        self._equity = 10000.0
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def disconnect(self):
        self.connected = False
    
    def ensure_connected(self) -> bool:
        return self.connected
    
    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            balance=self._balance,
            equity=self._equity,
            margin=0,
            free_margin=self._equity,
            profit=self._equity - self._balance,
        )
    
    def get_data(self, n: int = 100, **kwargs) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        import numpy as np
        
        np.random.seed(42)
        base_price = 1.1000
        
        dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
        
        # Random walk
        returns = np.random.randn(n) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from prices
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
            'close': prices * (1 + np.random.randn(n) * 0.0002),
            'tick_volume': np.random.randint(100, 1000, n),
        })
        
        return df
    
    def execute_order(self, signal: str, volume: float, sl: float, tp: float, **kwargs) -> OrderResult:
        self._ticket_counter += 1
        ticket = self._ticket_counter
        price = 1.1000
        logger.info(f"[MOCK] Order: {signal} {volume} lots, SL={sl}, TP={tp}")
        self._positions.append(
            {
                'ticket': ticket,
                'symbol': self.symbol,
                'type': signal.upper(),
                'volume': float(volume),
                'price_open': price,
                'price_current': price,
                'sl': float(sl),
                'tp': float(tp),
                'profit': 0.0,
                'magic': kwargs.get('magic', 123456),
                'time': datetime.now(),
            }
        )
        return OrderResult(True, ticket, price, volume, None)
    
    def entry(self, signal: str, volume: float, sl: float, tp: float) -> OrderResult:
        return self.execute_order(signal, volume, sl, tp)
    
    def get_open_positions(self, **kwargs) -> list:
        symbol = kwargs.get('symbol')
        if symbol:
            return [p for p in self._positions if p.get('symbol') == symbol]
        return list(self._positions)

    def close_position(self, ticket: int) -> OrderResult:
        for i, p in enumerate(self._positions):
            if int(p.get('ticket')) == int(ticket):
                volume = float(p.get('volume', 0.0))
                price = float(p.get('price_current', p.get('price_open', 1.1000)))
                del self._positions[i]
                return OrderResult(True, int(ticket), price, volume, None)
        return OrderResult(False, None, None, 0.0, f"Position {ticket} not found")
    
    def get_current_price(self, **kwargs) -> Dict[str, float]:
        return {'bid': 1.0999, 'ask': 1.1001, 'spread': 0.0002}
    
    def get_symbol_info(self, **kwargs) -> Dict[str, Any]:
        return {
            'name': self.symbol,
            'point': 0.00001,
            'digits': 5,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01,
            'trade_tick_value': 1.0,
        }
