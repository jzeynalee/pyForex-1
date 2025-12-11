#!/usr/bin/env python3
"""
Shared fixtures for utils module tests.
"""

import logging
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ============================================================================
# LOGGING CLEANUP (Windows compatibility)
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging handlers after each test."""
    yield
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)


# ============================================================================
# TEMPORARY DIRECTORIES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 200
    base_price = 1.1000
    prices = base_price + np.cumsum(np.random.randn(n) * 0.001)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0005),
        'low': prices - np.abs(np.random.randn(n) * 0.0005),
        'close': prices + np.random.randn(n) * 0.0003,
        'tick_volume': np.random.randint(100, 5000, n),
        'volume': np.random.randint(100, 5000, n),
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def small_ohlcv_df():
    """Create a small OHLCV DataFrame for quick tests."""
    np.random.seed(42)
    n = 30
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices + np.random.randn(n) * 0.2,
        'tick_volume': np.random.randint(100, 1000, n),
        'volume': np.random.randint(100, 1000, n),
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def bullish_trend_df():
    """Create DataFrame with clear bullish trend."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.linspace(0, 10, n) + np.random.randn(n) * 0.2
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.2),
        'close': prices + 0.05,  # Slight bullish bias
        'volume': np.random.randint(100, 1000, n),
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def bearish_trend_df():
    """Create DataFrame with clear bearish trend."""
    np.random.seed(42)
    n = 100
    prices = 110 - np.linspace(0, 10, n) + np.random.randn(n) * 0.2
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.2),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices - 0.05,  # Slight bearish bias
        'volume': np.random.randint(100, 1000, n),
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sideways_df():
    """Create DataFrame with sideways/ranging market."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.random.randn(n) * 0.5  # No trend
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.2),
        'low': prices - np.abs(np.random.randn(n) * 0.2),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 1000, n),
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


# ============================================================================
# CSV FILE FIXTURES
# ============================================================================

@pytest.fixture
def sample_csv_file(temp_dir, sample_ohlcv_df):
    """Create a sample CSV file."""
    csv_path = temp_dir / "test_data.csv"
    sample_ohlcv_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def minimal_csv_file(temp_dir):
    """Create a minimal CSV file with just OHLCV."""
    csv_path = temp_dir / "minimal.csv"
    df = pd.DataFrame({
        'open': [1.0, 1.1, 1.2],
        'high': [1.1, 1.2, 1.3],
        'low': [0.9, 1.0, 1.1],
        'close': [1.05, 1.15, 1.25],
        'tick_volume': [100, 200, 150],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_torch():
    """Create mock torch module."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.__version__ = "2.0.0"
    mock.device.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_checkpoint():
    """Create mock model checkpoint data."""
    return {
        'model_state': {'layer1.weight': np.random.randn(10, 10)},
        'feature_columns': ['rsi_14', 'macd', 'ema_20', 'adx'],
        'config': {
            'model': {'input_dim': 4, 'hidden_dim': 64},
            'training': {'num_classes': 3},
        },
        'metrics': {'best_val_acc': 0.75, 'test_accuracy': 0.72},
        'profile': 'INTRADAY',
        'created_at': '2024-01-01T00:00:00',
    }


# ============================================================================
# PATTERN TEST DATA
# ============================================================================

@pytest.fixture
def doji_candle_df():
    """Create DataFrame with a doji pattern."""
    return pd.DataFrame({
        'open': [100.0],
        'high': [100.5],
        'low': [99.5],
        'close': [100.02],  # Very small body
        'volume': [1000],
    })


@pytest.fixture
def hammer_candle_df():
    """Create DataFrame with hammer pattern."""
    return pd.DataFrame({
        'open': [100.0],
        'high': [100.2],
        'low': [98.5],  # Long lower shadow
        'close': [100.1],
        'volume': [1000],
    })


@pytest.fixture
def engulfing_pattern_df():
    """Create DataFrame with bullish engulfing pattern."""
    return pd.DataFrame({
        'open': [100.0, 99.0],
        'high': [100.2, 101.5],
        'low': [99.5, 98.8],
        'close': [99.6, 101.2],  # Second candle engulfs first
        'volume': [1000, 1500],
    })