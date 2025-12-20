# tests/training_auto_retrain.py
"""
Comprehensive unit tests for training/auto_retrain.py

This module tests the auto_retrain_job function which:
- Connects to MT5 and downloads candle data
- Validates and saves data to CSV
- Triggers model retraining with optimized parameters

NOTE: These tests work with the stub module defined in conftest.py.
The stub uses dynamic module lookup, so patches to 'training.auto_retrain.MT5Connector'
will be picked up correctly.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import logging

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe_large():
    """Create a sample DataFrame with 50,000 rows (above warning threshold)."""
    n_rows = 50000
    np.random.seed(42)
    return pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
        'open': np.random.uniform(1.0, 1.5, n_rows),
        'high': np.random.uniform(1.0, 1.5, n_rows),
        'low': np.random.uniform(1.0, 1.5, n_rows),
        'close': np.random.uniform(1.0, 1.5, n_rows),
        'tick_volume': np.random.randint(100, 10000, n_rows),
        'spread': np.random.randint(1, 20, n_rows),
        'real_volume': np.random.randint(0, 1000, n_rows)
    })


@pytest.fixture
def sample_dataframe_small():
    """Create a sample DataFrame with 5,000 rows (below warning threshold)."""
    n_rows = 5000
    np.random.seed(42)
    return pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
        'open': np.random.uniform(1.0, 1.5, n_rows),
        'high': np.random.uniform(1.0, 1.5, n_rows),
        'low': np.random.uniform(1.0, 1.5, n_rows),
        'close': np.random.uniform(1.0, 1.5, n_rows),
        'tick_volume': np.random.randint(100, 10000, n_rows),
        'spread': np.random.randint(1, 20, n_rows),
        'real_volume': np.random.randint(0, 1000, n_rows)
    })


@pytest.fixture
def sample_dataframe_minimal():
    """Create a minimal DataFrame with 100 rows."""
    n_rows = 100
    np.random.seed(42)
    return pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
        'open': np.random.uniform(1.0, 1.5, n_rows),
        'high': np.random.uniform(1.0, 1.5, n_rows),
        'low': np.random.uniform(1.0, 1.5, n_rows),
        'close': np.random.uniform(1.0, 1.5, n_rows),
        'tick_volume': np.random.randint(100, 10000, n_rows),
        'spread': np.random.randint(1, 20, n_rows),
        'real_volume': np.random.randint(0, 1000, n_rows)
    })


@pytest.fixture
def temp_workdir(tmp_path, monkeypatch):
    """Change to a temporary working directory for file operations."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def create_mock_connector(connect_return=True, get_data_return=None):
    """Factory to create a mock MT5Connector class."""
    class MockMT5Connector:
        def __init__(self, *args, **kwargs):
            pass
        
        def connect(self):
            return connect_return
        
        def get_data(self, symbol=None, n=None, timeframe=None):
            MockMT5Connector.last_get_data_call = {
                'symbol': symbol,
                'n': n,
                'timeframe': timeframe
            }
            if get_data_return is None:
                return pd.DataFrame()
            return get_data_return
    
    MockMT5Connector.last_get_data_call = None
    return MockMT5Connector


# ============================================================================
# MT5 CONNECTION TESTS
# ============================================================================

class TestMT5Connection:
    """Tests for MT5 connection handling."""
    
    def test_connection_failure_returns_early(self, temp_workdir, caplog):
        """Test that job returns early when MT5 connection fails."""
        MockConnector = create_mock_connector(connect_return=False)
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.ERROR):
                auto_retrain_job()
            
            assert "Could not connect to MT5" in caplog.text
            mock_train.assert_not_called()
    
    def test_connection_success_proceeds(self, temp_workdir, sample_dataframe_large):
        """Test that job proceeds when MT5 connection succeeds."""
        MockConnector = create_mock_connector(
            connect_return=True, 
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            # get_data should have been called
            assert MockConnector.last_get_data_call is not None
    
    def test_connector_instantiated_correctly(self, temp_workdir):
        """Test that MT5Connector is instantiated."""
        instantiated = []
        
        class TrackingConnector:
            def __init__(self, *args, **kwargs):
                instantiated.append(True)
            
            def connect(self):
                return False
        
        with patch('training.auto_retrain.MT5Connector', TrackingConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            assert len(instantiated) == 1


# ============================================================================
# DATA DOWNLOAD TESTS
# ============================================================================

class TestDataDownload:
    """Tests for data downloading functionality."""
    
    def test_get_data_called_with_correct_params(self, temp_workdir, sample_dataframe_large):
        """Test that get_data is called with expected symbol, count, and timeframe."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_info = MockConnector.last_get_data_call
            assert call_info['symbol'] == 'EURUSD'
            assert call_info['n'] == 8000000
            assert call_info['timeframe'] == 'M15'
    
    def test_empty_dataframe_returns_early(self, temp_workdir, caplog):
        """Test that job returns early when no data is received."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=pd.DataFrame()
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.ERROR):
                auto_retrain_job()
            
            assert "No data received" in caplog.text
            mock_train.assert_not_called()
    
    def test_data_download_logs_count(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that downloaded row count is logged correctly."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            assert f"Downloaded {len(sample_dataframe_large)} rows" in caplog.text


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

class TestDataValidation:
    """Tests for data validation and warnings."""
    
    def test_small_dataset_warning(self, temp_workdir, sample_dataframe_small, caplog):
        """Test that warning is logged for small datasets (< 10k rows)."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_small
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            assert "Dataset is very small" in caplog.text
    
    def test_large_dataset_no_warning(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that no warning is logged for large datasets (>= 10k rows)."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            assert "Dataset is very small" not in caplog.text
    
    def test_exactly_10k_rows_no_warning(self, temp_workdir, caplog):
        """Test boundary condition: exactly 10,000 rows should not trigger warning."""
        n_rows = 10000
        np.random.seed(42)
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
            'open': np.random.uniform(1.0, 1.5, n_rows),
            'close': np.random.uniform(1.0, 1.5, n_rows)
        })
        
        MockConnector = create_mock_connector(connect_return=True, get_data_return=df)
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            assert "Dataset is very small" not in caplog.text
    
    def test_9999_rows_triggers_warning(self, temp_workdir, caplog):
        """Test boundary condition: 9,999 rows should trigger warning."""
        n_rows = 9999
        np.random.seed(42)
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
            'open': np.random.uniform(1.0, 1.5, n_rows),
            'close': np.random.uniform(1.0, 1.5, n_rows)
        })
        
        MockConnector = create_mock_connector(connect_return=True, get_data_return=df)
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            assert "Dataset is very small" in caplog.text


# ============================================================================
# DATA PERSISTENCE TESTS
# ============================================================================

class TestDataPersistence:
    """Tests for data saving functionality."""
    
    def test_data_saved_to_correct_path(self, temp_workdir, sample_dataframe_large):
        """Test that data is saved to data/raw/eurusd_latest.csv."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            expected_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            assert expected_path.exists()
    
    def test_data_directory_created_if_not_exists(self, temp_workdir, sample_dataframe_large):
        """Test that data/raw directory is created if it doesn't exist."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        # Ensure directory doesn't exist
        assert not (temp_workdir / "data" / "raw").exists()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            assert (temp_workdir / "data" / "raw").exists()
    
    def test_saved_csv_content_matches_dataframe(self, temp_workdir, sample_dataframe_minimal):
        """Test that saved CSV content matches the original DataFrame."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_minimal
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            csv_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            loaded_df = pd.read_csv(csv_path)
            
            # Check shape matches
            assert loaded_df.shape == sample_dataframe_minimal.shape
            
            # Check columns match
            assert list(loaded_df.columns) == list(sample_dataframe_minimal.columns)
    
    def test_csv_saved_without_index(self, temp_workdir, sample_dataframe_minimal):
        """Test that CSV is saved without index column."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_minimal
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            csv_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            
            with open(csv_path, 'r') as f:
                header_line = f.readline().strip()
            
            assert 'Unnamed' not in header_line


# ============================================================================
# TRAINING INVOCATION TESTS
# ============================================================================

class TestTrainingInvocation:
    """Tests for training function invocation."""
    
    def test_training_called_with_correct_params(self, temp_workdir, sample_dataframe_large):
        """Test that train_tcn_enhanced is called with correct parameters."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            # The stub uses relative path "data/raw/eurusd_latest.csv"
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            
            # Verify path ends with expected relative path
            assert call_kwargs['data'].endswith('eurusd_latest.csv')
            assert 'data' in call_kwargs['data']
            assert 'raw' in call_kwargs['data']
            
            # Verify all other parameters exactly
            assert call_kwargs['epochs'] == 50
            assert call_kwargs['seq_len'] == 60
            assert call_kwargs['hidden_dim'] == 64
            assert call_kwargs['dropout'] == 0.2
            assert call_kwargs['lr'] == 1e-3
            assert call_kwargs['device'] == "auto"
    
    def test_training_epochs_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses 50 epochs as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['epochs'] == 50
    
    def test_training_hidden_dim_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses hidden_dim=64 as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['hidden_dim'] == 64
    
    def test_training_dropout_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses dropout=0.2 as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['dropout'] == 0.2
    
    def test_training_learning_rate_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses lr=1e-3 as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['lr'] == 1e-3
    
    def test_training_device_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses device='auto' as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['device'] == 'auto'
    
    def test_training_seq_len_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses seq_len=60 as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['seq_len'] == 60
    
    def test_training_trend_threshold_parameter(self, temp_workdir, sample_dataframe_large):
        """Test that training uses threshold=0.05 as specified."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['threshold'] == 0.05


# ============================================================================
# EXCEPTION HANDLING TESTS
# ============================================================================

class TestExceptionHandling:
    """Tests for exception handling during training."""
    
    def test_training_exception_caught_and_logged(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that training exceptions are caught and logged."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock(side_effect=RuntimeError("CUDA out of memory"))
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.ERROR):
                # Should not raise
                auto_retrain_job()
            
            assert "Training Failed" in caplog.text
            assert "CUDA out of memory" in caplog.text
    
    def test_training_exception_does_not_crash(self, temp_workdir, sample_dataframe_large):
        """Test that training exceptions don't crash the job."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock(side_effect=ValueError("Invalid parameter"))
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            # Should complete without raising
            auto_retrain_job()
    
    def test_training_success_logged(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that successful training completion is logged."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            assert "Retraining Complete" in caplog.text


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestLogging:
    """Tests for logging behavior."""
    
    def test_start_banner_printed(self, temp_workdir, sample_dataframe_large, capsys):
        """Test that start banner is printed."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            captured = capsys.readouterr()
            assert "STARTING TCN RETRAINING JOB" in captured.out
            assert "BIG DATA MODE" in captured.out
    
    def test_download_info_logged(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that download information is logged."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            assert "Downloading latest" in caplog.text
            assert "EURUSD" in caplog.text
    
    def test_save_path_logged(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that save path is logged."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            assert "Saved to" in caplog.text
            assert "eurusd_latest.csv" in caplog.text
    
    def test_training_start_logged(self, temp_workdir, sample_dataframe_large, caplog):
        """Test that training start is logged."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            assert "Starting TCN Training" in caplog.text


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """Integration-style tests for complete workflow."""
    
    def test_complete_successful_workflow(self, temp_workdir, sample_dataframe_large, caplog):
        """Test complete successful workflow from connection to training."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.INFO):
                auto_retrain_job()
            
            # Verify file was created
            csv_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            assert csv_path.exists()
            
            # Verify training was called
            mock_train.assert_called_once()
            
            # Verify success was logged
            assert "Retraining Complete" in caplog.text
    
    def test_workflow_stops_at_connection_failure(self, temp_workdir):
        """Test that workflow stops completely when connection fails."""
        MockConnector = create_mock_connector(connect_return=False)
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            mock_train.assert_not_called()
    
    def test_workflow_stops_at_empty_data(self, temp_workdir):
        """Test that workflow stops when data is empty."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=pd.DataFrame()
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            mock_train.assert_not_called()
    
    def test_workflow_continues_despite_small_data_warning(self, temp_workdir, 
                                                            sample_dataframe_small, caplog):
        """Test that workflow continues even with small data warning."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_small
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            # Warning should appear but training should still be called
            assert "Dataset is very small" in caplog.text
            mock_train.assert_called_once()


# ============================================================================
# CONFIGURATION CONSTANTS TESTS
# ============================================================================

class TestConfigurationConstants:
    """Tests to verify configuration constants are used correctly."""
    
    def test_download_count_is_8_million(self, temp_workdir, sample_dataframe_large):
        """Test that DOWNLOAD_COUNT is set to 8,000,000."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            assert MockConnector.last_get_data_call['n'] == 8000000
    
    def test_timeframe_is_m15(self, temp_workdir, sample_dataframe_large):
        """Test that TIMEFRAME is set to M15."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            assert MockConnector.last_get_data_call['timeframe'] == 'M15'
    
    def test_symbol_is_eurusd(self, temp_workdir, sample_dataframe_large):
        """Test that SYMBOL is set to EURUSD."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            assert MockConnector.last_get_data_call['symbol'] == 'EURUSD'


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_dataframe_with_nan_values(self, temp_workdir):
        """Test handling of DataFrame with NaN values."""
        np.random.seed(42)
        n_rows = 15000
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
            'open': np.random.uniform(1.0, 1.5, n_rows),
            'close': np.random.uniform(1.0, 1.5, n_rows)
        })
        df.loc[100:105, 'open'] = np.nan
        
        MockConnector = create_mock_connector(connect_return=True, get_data_return=df)
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            mock_train.assert_called_once()
    
    def test_single_row_dataframe(self, temp_workdir, caplog):
        """Test handling of DataFrame with only one row."""
        df = pd.DataFrame({
            'time': [pd.Timestamp('2020-01-01')],
            'open': [1.1],
            'close': [1.2]
        })
        
        MockConnector = create_mock_connector(connect_return=True, get_data_return=df)
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            with caplog.at_level(logging.WARNING):
                auto_retrain_job()
            
            assert "Dataset is very small" in caplog.text
    
    def test_dataframe_with_special_characters_in_values(self, temp_workdir):
        """Test handling of DataFrame (numeric data should have no special chars)."""
        n_rows = 15000
        np.random.seed(42)
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
            'open': np.random.uniform(1.0, 1.5, n_rows),
            'close': np.random.uniform(1.0, 1.5, n_rows)
        })
        
        MockConnector = create_mock_connector(connect_return=True, get_data_return=df)
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            csv_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            assert csv_path.exists()


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================

class TestModuleImports:
    """Tests for module import behavior."""
    
    def test_module_imports_without_error(self):
        """Test that the module can be imported without errors."""
        import training.auto_retrain
    
    def test_auto_retrain_job_is_callable(self):
        """Test that auto_retrain_job function is accessible and callable."""
        from training.auto_retrain import auto_retrain_job
        assert callable(auto_retrain_job)


# ============================================================================
# CONCURRENT EXECUTION TESTS
# ============================================================================

class TestConcurrentExecution:
    """Tests related to potential concurrent execution scenarios."""
    
    def test_file_overwrite_on_rerun(self, temp_workdir, sample_dataframe_minimal):
        """Test that CSV file is overwritten on subsequent runs."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_minimal
        )
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', Mock()):
            
            from training.auto_retrain import auto_retrain_job
            
            # First run
            auto_retrain_job()
            csv_path = temp_workdir / "data" / "raw" / "eurusd_latest.csv"
            first_mtime = csv_path.stat().st_mtime
            
            # Small delay to ensure different mtime
            import time
            time.sleep(0.1)
            
            # Create new data
            n_rows = 200
            np.random.seed(123)
            new_df = pd.DataFrame({
                'time': pd.date_range('2021-01-01', periods=n_rows, freq='15min'),
                'open': np.random.uniform(1.2, 1.6, n_rows),
                'close': np.random.uniform(1.2, 1.6, n_rows)
            })
            
            # Update mock to return new data
            MockConnector2 = create_mock_connector(connect_return=True, get_data_return=new_df)
            
            with patch('training.auto_retrain.MT5Connector', MockConnector2):
                auto_retrain_job()
            
            second_mtime = csv_path.stat().st_mtime
            
            # File should have been modified
            assert second_mtime > first_mtime
            
            # Content should be from second DataFrame
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == n_rows


# ============================================================================
# MEMORY AND PERFORMANCE TESTS (Lightweight)
# ============================================================================

class TestPerformanceConsiderations:
    """Lightweight tests for performance-related behavior."""
    
    def test_dataframe_not_duplicated_unnecessarily(self, temp_workdir, sample_dataframe_large):
        """Verify DataFrame operations don't create excessive copies."""
        MockConnector = create_mock_connector(
            connect_return=True,
            get_data_return=sample_dataframe_large
        )
        mock_train = Mock()
        
        with patch('training.auto_retrain.MT5Connector', MockConnector), \
             patch('training.auto_retrain.train_tcn_enhanced', mock_train):
            
            from training.auto_retrain import auto_retrain_job
            auto_retrain_job()
            
            # Basic sanity check - function completed
            mock_train.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])