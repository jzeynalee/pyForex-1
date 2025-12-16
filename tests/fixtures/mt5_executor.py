import pytest

from trading.mt5_connector import MockMT5Connector
from trading.mt5_executor import MT5Executor, MT5ExecutorConfig


@pytest.fixture
def mt5_connector():
    return MockMT5Connector(symbol="EURUSD")


@pytest.fixture
def mt5_executor(mt5_connector):
    config = MT5ExecutorConfig(symbol=mt5_connector.symbol)
    return MT5Executor(connector=mt5_connector, config=config)
