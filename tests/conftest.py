# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data."""
    n = 200
    np.random.seed(42)
    base_price = 1.1000
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
    returns = np.random.randn(n) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        'close': prices * (1 + np.random.randn(n) * 0.001),
        'volume': np.random.randint(100, 1000, n),
    })

@pytest.fixture
def mtf_data(sample_ohlcv_data):
    """Multi-timeframe data dict."""
    return {
        'H4': sample_ohlcv_data.copy(),
        'H1': sample_ohlcv_data.copy(),
        'M15': sample_ohlcv_data.copy(),
    }

@pytest.fixture
def bullish_trend_data():
    """Data with clear uptrend."""
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
    # Steady upward drift
    prices = 1.1 + np.linspace(0, 0.05, n) + np.random.randn(n) * 0.001
    
    return pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices * 1.001,
        'volume': np.random.randint(100, 1000, n),
    })

@pytest.fixture
def bearish_trend_data():
    """Data with clear downtrend."""
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
    prices = 1.15 - np.linspace(0, 0.05, n) + np.random.randn(n) * 0.001
    
    return pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices * 0.999,
        'volume': np.random.randint(100, 1000, n),
    })

@pytest.fixture
def temp_csv_file(tmp_path, sample_ohlcv_data):
    """Create temporary CSV file with OHLCV data."""
    filepath = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def model_probabilities_buy():
    return np.array([0.75, 0.15, 0.10])

@pytest.fixture
def model_probabilities_sell():
    return np.array([0.10, 0.80, 0.10])

@pytest.fixture
def model_probabilities_uncertain():
    return np.array([0.40, 0.35, 0.25])

@pytest.fixture
def mock_trend_analysis_bullish():
    return {
        'trend_class': 2,
        'trend_name': 'Mature Bull Trend',
        'direction': 'BULLISH',
        'confidence': 0.75,
        'trend_strength': 70,
        'details': {}
    }

@pytest.fixture
def mock_trend_analysis_bearish():
    return {
        'trend_class': 4,
        'trend_name': 'Mature Bear Trend',
        'direction': 'BEARISH',
        'confidence': 0.75,
        'trend_strength': 70,
        'details': {}
    }

@pytest.fixture
def mock_trend_analysis_sideways():
    return {
        'trend_class': 0,
        'trend_name': 'Sideways/Compression',
        'direction': 'SIDEWAYS',
        'confidence': 0.60,
        'trend_strength': 25,
        'details': {}
    }