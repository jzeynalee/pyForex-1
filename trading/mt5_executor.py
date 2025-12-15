"""MT5 execution adapter for strategy/bot integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from trading.mt5_connector import MockMT5Connector, OrderResult


@dataclass
class MT5ExecutorConfig:
    symbol: str = "EURUSD"
    magic_number: int = 123456


class MT5Executor:
    def __init__(
        self,
        connector: Optional[object] = None,
        config: Optional[MT5ExecutorConfig] = None,
    ):
        self.config = config or MT5ExecutorConfig()
        if connector is None:
            connector = MockMT5Connector(symbol=self.config.symbol)
        self.connector = connector

    def ensure_connected(self) -> bool:
        ensure = getattr(self.connector, "ensure_connected", None)
        if callable(ensure):
            return bool(ensure())
        return bool(getattr(self.connector, "connected", True))

    def get_account_balance(self) -> float:
        info = getattr(self.connector, "get_account_info", None)
        if callable(info):
            account = info()
            if account is None:
                return 0.0
            return float(getattr(account, "balance", 0.0))
        return 0.0

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        direction: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "PyForex",
        magic_number: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.ensure_connected():
            return {"success": False, "ticket": None, "price": None, "volume": float(volume), "error": "Not connected"}

        signal = direction.upper()
        if signal not in {"BUY", "SELL"}:
            return {"success": False, "ticket": None, "price": None, "volume": float(volume), "error": f"Invalid direction: {direction}"}

        exec_fn = getattr(self.connector, "execute_order", None)
        if not callable(exec_fn):
            return {"success": False, "ticket": None, "price": None, "volume": float(volume), "error": "Connector has no execute_order"}

        result = exec_fn(
            signal=signal,
            volume=float(volume),
            sl=float(stop_loss),
            tp=float(take_profit),
            symbol=symbol,
            comment=comment,
        )

        if isinstance(result, OrderResult):
            return {
                "success": bool(result.success),
                "ticket": result.ticket,
                "price": result.price,
                "volume": float(result.volume),
                "error": result.error,
            }

        if isinstance(result, dict):
            return result

        return {"success": False, "ticket": None, "price": None, "volume": float(volume), "error": "Unknown order result type"}
