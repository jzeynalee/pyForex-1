
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Union, Any
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    passed: bool
    issues: List[str]
    stats: Dict[str, Any]

class DataLoader:
    """
    Handles loading and validation of historical data for backtesting.
    """
    
    REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close']
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.
        Expects columns: time, open, high, low, close, [volume]
        """
        path = Path(filepath)
        if not path.exists():
            # Try looking in data_dir
            path = self.data_dir / filepath
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
                
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Parse time
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'])
            df.drop(columns=['date'], inplace=True)
        elif 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'])
            df.drop(columns=['timestamp'], inplace=True)
        else:
            raise ValueError("No time/date/timestamp column found")
            
        # Set index
        df.set_index('time', inplace=True, drop=False)
        df.sort_index(inplace=True)
        
        # Ensure floats
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)
                
        # Fill volume if missing
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                df['volume'] = 100
                
        return df

    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data integrity according to backtesting requirements.
        - Timestamp monotonicity
        - OHLC relationships
        - Gaps
        """
        issues = []
        stats = {}
        
        # 1. Missing columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return ValidationResult(False, [f"Missing columns: {missing}"], stats)
            
        # 2. Timestamp monotonicity
        if not df.index.is_monotonic_increasing:
            issues.append("Timestamps are not strictly monotonic increasing")
            
        # 3. Duplicates
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            issues.append(f"Found {dupes} duplicate timestamps")
            
        # 4. OHLC Integrity
        # High >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"Found {invalid_hl} bars with High < Low")
            
        # High >= Open, High >= Close
        invalid_h_oc = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if invalid_h_oc > 0:
            issues.append(f"Found {invalid_h_oc} bars with High < Open/Close")
            
        # Low <= Open, Low <= Close
        invalid_l_oc = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if invalid_l_oc > 0:
            issues.append(f"Found {invalid_l_oc} bars with Low > Open/Close")
            
        # 5. Gap Detection
        # Check for gaps > 1 hour (assuming H1) or configurable
        # Ideally we infer frequency
        try:
            diffs = df.index.to_series().diff()
            # Most common diff
            mode_diff = diffs.mode().iloc[0]
            gaps = (diffs > mode_diff * 1.1).sum() # 10% tolerance
            if gaps > 0:
                issues.append(f"Found {gaps} time gaps larger than expected frequency {mode_diff}")
                
            stats['freq'] = str(mode_diff)
            stats['gaps'] = int(gaps)
        except Exception as e:
            issues.append(f"Failed to analyze time gaps: {e}")

        stats['bars'] = len(df)
        stats['start'] = str(df.index[0])
        stats['end'] = str(df.index[-1])
        
        passed = len(issues) == 0
        return ValidationResult(passed, issues, stats)
        
    def generate_synthetic_data(self, n: int = 1000, start_date: str = '2023-01-01', freq: str = 'H') -> pd.DataFrame:
        """Generate synthetic data for testing if no file provided."""
        dates = pd.date_range(start=start_date, periods=n, freq=freq)
        
        np.random.seed(42)
        base_price = 1.1000
        returns = np.random.randn(n) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
            'close': prices * (1 + np.random.randn(n) * 0.0002),
            'volume': np.random.randint(100, 1000, n)
        })
        
        # Fix OHLC
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        df.set_index('time', inplace=True, drop=False)
        return df
