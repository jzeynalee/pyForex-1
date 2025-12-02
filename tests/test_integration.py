# tests/test_integration.py
"""
Integration tests for pyForex system.
These tests verify that components work together correctly.
"""
import pytest
import numpy as np
import pandas as pd
import torch
import time
import os   
from datetime import datetime, timedelta

from utils.data_loader import DataLoader, DataConfig
from utils.candle_to_image import candle_image, normalize_for_model
from models.lstm import LSTMModel
from models.fusion import FusionNet
from models.yolo_detector import MockYOLODetector
from models.trend_classifier import TrendClassifier, generate_synthetic_training_data
from trading.signal_engine import generate_signal, SignalConfig
from trading.risk_manager import RiskManager
from trading.decision_engine import DecisionEngine
from trading.backtest import BacktestExecutor
from trend_detection.fusion_trend_detector import FusionFXTrendDetector

@pytest.mark.slow
class TestPerformanceSmoke:
    """Smoke tests to catch performance regressions."""
    
    @pytest.mark.slow
    def test_trend_detection_performance(self, mtf_data):
        """Ensure trend detection completes in reasonable time."""
        import time
        
        detector = FusionFXTrendDetector(ml_model=None)
        
        start = time.time()
        for _ in range(100):
            detector.detect_trend(mtf_data)
        elapsed = time.time() - start
        
        # Should complete 100 iterations in under 5 seconds
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 100 iterations"
    
    @pytest.mark.slow
    def test_backtest_large_dataset(self):
        """Test backtest can handle larger datasets."""
        # Generate 1000 candles
        n = 1000
        np.random.seed(42)
        prices = 1.1 * np.exp(np.cumsum(np.random.randn(n) * 0.001))
        
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n, freq='H'),
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(100, 1000, n),
        })
        
        executor = BacktestExecutor()
        
        for i, row in df.iterrows():
            executor.current_price = row['close']
            if i % 50 == 0 and len(executor.positions) == 0:
                executor.entry('BUY', volume=0.1, 
                              sl=row['close'] * 0.99, 
                              tp=row['close'] * 1.02)
            executor.update_price(row['close'])
        
        metrics = executor.get_performance_metrics()
        assert metrics['total_trades'] > 0


@pytest.mark.unit
class TestDataToModelPipeline:
    """Test data loading to model inference pipeline."""
    
    def test_csv_to_lstm_inference(self, temp_csv_file):
        """Test complete pipeline from CSV to LSTM inference."""
        # Load data
        loader = DataLoader(DataConfig(sequence_length=60))
        df = loader.load_csv(temp_csv_file)
        train, test, _ = loader.split_and_scale(df)
        
        # Create sequences
        X, y = loader.create_sequences(train)
        
        # Initialize model
        model = LSTMModel()
        model.eval()
        
        # Run inference
        x_tensor = torch.tensor(X[:4])
        with torch.no_grad():
            features = model(x_tensor, mode='features')
            logits = model(x_tensor, mode='classify')
        
        assert features.shape == (4, 64)
        assert logits.shape == (4, 3)
    
    def test_ohlcv_to_image_to_tensor(self, sample_ohlcv_data):
        """Test OHLCV data to normalized image tensor pipeline."""
        # Generate image
        img = candle_image(sample_ohlcv_data.tail(60), target_size=224)
        
        # Normalize for model
        normalized = normalize_for_model(img)
        
        # Convert to tensor
        tensor = torch.tensor(normalized).unsqueeze(0)  # Add batch dim
        
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

@pytest.mark.unit
class TestModelFusionPipeline:
    """Test multi-modal model fusion pipeline."""
    
    def test_full_fusion_pipeline(self, sample_ohlcv_data):
        """Test complete fusion model pipeline."""
        # Prepare inputs
        batch_size = 2
        
        # LSTM features
        lstm_model = LSTMModel()
        lstm_model.eval()
        x_seq = torch.randn(batch_size, 60, 5)
        
        # ViT features (mocked as random)
        vit_features = torch.randn(batch_size, 768)
        
        # YOLO features
        yolo_detector = MockYOLODetector()
        img = candle_image(sample_ohlcv_data)
        yolo_vec = yolo_detector.detect(img)
        yolo_features = torch.tensor(yolo_vec).float().unsqueeze(0).repeat(batch_size, 1)
        
        # Fusion
        fusion_model = FusionNet()
        fusion_model.eval()
        
        with torch.no_grad():
            lstm_feat = lstm_model(x_seq, mode='features')
            logits, gates = fusion_model.forward_with_gates(
                lstm_feat, vit_features, yolo_features
            )
        
        assert logits.shape == (batch_size, 3)
        assert gates.shape == (batch_size, 3)
        
        # Verify gate weights sum to 1
        assert torch.allclose(gates.sum(dim=1), torch.ones(batch_size), atol=1e-5)


@pytest.mark.unit
class TestSignalToTradePipeline:
    """Test signal generation to trade execution pipeline."""
    
    def test_model_output_to_signal(self):
        """Test converting model output to trading signal."""
        # Mock model output
        logits = torch.tensor([[2.0, -1.0, 0.5]])
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # Generate signal
        result = generate_signal(probs)
        
        assert result.signal is not None
        assert 0 <= result.confidence <= 1
    
    def test_signal_to_risk_params(self, sample_ohlcv_data):
        """Test generating risk-managed trade parameters from signal."""
        # Generate signal
        probs = np.array([0.75, 0.15, 0.10])  # Strong BUY
        signal_result = generate_signal(probs)
        
        # Calculate risk parameters
        risk_manager = RiskManager(account_balance=10000)
        params = risk_manager.get_params(sample_ohlcv_data, signal='BUY')
        
        assert params.volume > 0
        assert params.stop_loss > 0
        assert params.take_profit > 0
        assert params.atr > 0
    
    def test_full_trade_execution_flow(self, sample_ohlcv_data):
        """Test complete trade execution flow."""
        # 1. Model prediction (mocked)
        probs = np.array([0.75, 0.15, 0.10])
        signal_result = generate_signal(probs)
        
        # 2. Risk check
        risk_manager = RiskManager(account_balance=10000)
        allowed, reason = risk_manager.check_risk_limits(
            current_balance=10000,
            current_equity=10000,
            open_positions=0
        )
        
        assert allowed == True
        
        # 3. Get trade params
        params = risk_manager.get_params(sample_ohlcv_data, signal='BUY')
        
        # 4. Execute in backtest
        executor = BacktestExecutor()
        executor.current_price = sample_ohlcv_data['close'].iloc[-1]
        
        result = executor.entry(
            signal='BUY',
            volume=params.volume,
            sl=params.stop_loss,
            tp=params.take_profit
        )
        
        assert result['success'] == True
        assert len(executor.positions) == 1


@pytest.mark.unit
class TestTrendDetectionPipeline:
    """Test trend detection pipeline."""
    
    def test_full_trend_detection(self, mtf_data):
        """Test complete trend detection pipeline."""
        detector = FusionFXTrendDetector(ml_model=None)
        result = detector.detect_trend(mtf_data)
        
        # Validate output structure
        assert 'trend_class' in result
        assert 'direction' in result
        assert 'confidence' in result
        
        # Validate values
        assert result['trend_class'] in [0, 1, 2, 3, 4]
        assert result['direction'] in ['BULLISH', 'BEARISH', 'SIDEWAYS']
        assert 0 <= result['confidence'] <= 1
    
    def test_trend_with_ml_classifier(self, mtf_data):
        """Test trend detection with ML classifier."""
        # Train a simple classifier
        X, y = generate_synthetic_training_data(n_samples=500)
        ml_classifier = TrendClassifier()
        ml_classifier.fit(X, y, validate=False)
        
        # Use in detector
        detector = FusionFXTrendDetector(ml_model=ml_classifier)
        result = detector.detect_trend(mtf_data)
        
        # ML should have been used
        assert result['details']['ml_confidence'] != 0.5  # Default is 0.5 when no ML

@pytest.mark.unit
class TestDecisionEnginePipeline:
    """Test decision engine integration."""
    
    def test_pattern_and_trend_to_decision(self, mtf_data):
        """Test combining pattern recognition with trend analysis."""
        # Get trend analysis
        detector = FusionFXTrendDetector(ml_model=None)
        trend_result = detector.detect_trend(mtf_data)
        
        # Mock pattern probabilities
        pattern_probs = [0.75, 0.15, 0.10]  # Strong BUY
        
        # Make decision
        decision_engine = DecisionEngine(threshold=0.70)
        decision = decision_engine.decide(pattern_probs, trend_result)
        
        assert decision.signal in ['BUY', 'SELL', 'NO_TRADE']
        assert 0 <= decision.confidence <= 1
        assert decision.reason != ""


@pytest.mark.unit
class TestBacktestSimulation:
    """Test backtest simulation scenarios."""
    
    def test_simple_backtest_run(self, sample_ohlcv_data):
        """Test a simple backtest simulation."""
        executor = BacktestExecutor()
        
        # Simulate trading through price data
        prices = sample_ohlcv_data['close'].values
        
        # Enter a trade at the beginning
        executor.current_price = prices[0]
        executor.entry('BUY', volume=0.1, sl=prices[0] * 0.99, tp=prices[0] * 1.02)
        
        # Update through prices
        for i, price in enumerate(prices[1:]):
            executor.update_price(
                price, 
                time=datetime.now() - timedelta(hours=len(prices)-i)
            )
            
            # If position closed, enter new one
            if len(executor.positions) == 0 and i < len(prices) - 10:
                if i % 2 == 0:
                    executor.entry('BUY', volume=0.1, 
                                  sl=price * 0.99, tp=price * 1.02)
                else:
                    executor.entry('SELL', volume=0.1,
                                  sl=price * 1.01, tp=price * 0.98)
        
        # Close remaining positions
        executor.close_all_positions()
        
        # Get metrics
        metrics = executor.get_performance_metrics()
        
        assert metrics['total_trades'] > 0
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
    
    def test_risk_controlled_backtest(self, sample_ohlcv_data):
        """Test backtest with risk management."""
        executor = BacktestExecutor()
        risk_manager = RiskManager(account_balance=10000)
        
        prices = sample_ohlcv_data['close'].values[-50:]
        
        for i, price in enumerate(prices):
            executor.current_price = price
            
            # Check if trading allowed
            allowed, reason = risk_manager.check_risk_limits(
                current_balance=executor.balance,
                current_equity=executor.equity,
                open_positions=len(executor.positions)
            )
            
            if allowed and len(executor.positions) == 0 and i < len(prices) - 5:
                # Get risk-managed parameters
                df_slice = sample_ohlcv_data.tail(50 + i).head(50)
                params = risk_manager.get_params(df_slice, signal='BUY')
                
                executor.entry('BUY', volume=params.volume,
                              sl=params.stop_loss, tp=params.take_profit)
                risk_manager.record_trade()
            
            executor.update_price(price)
        
        # Verify risk limits were respected
        assert risk_manager.daily_trades <= risk_manager.config.max_daily_trades


@pytest.mark.integration
class TestEndToEndIntegration:
    """Full end-to-end system tests."""
    
    def test_complete_trading_cycle(self, sample_ohlcv_data, mtf_data):
        """Test complete trading cycle from data to execution."""
        # 1. Trend Detection
        trend_detector = FusionFXTrendDetector(ml_model=None)
        trend_result = trend_detector.detect_trend(mtf_data)
        
        # 2. Pattern Recognition (mocked)
        # In real scenario, this would come from LSTM/ViT/YOLO fusion
        pattern_probs = [0.72, 0.18, 0.10]
        
        # 3. Decision Engine
        decision_engine = DecisionEngine(threshold=0.65)
        decision = decision_engine.decide(pattern_probs, trend_result)
        
        # 4. If trade signal, proceed
        if decision.signal != 'NO_TRADE':
            # 5. Risk Management
            risk_manager = RiskManager(account_balance=10000)
            
            allowed, _ = risk_manager.check_risk_limits(
                current_balance=10000,
                current_equity=10000,
                open_positions=0
            )
            
            if allowed:
                # 6. Calculate trade parameters
                params = risk_manager.get_params(
                    sample_ohlcv_data, 
                    signal=decision.signal
                )
                
                # 7. Execute trade
                executor = BacktestExecutor()
                executor.current_price = sample_ohlcv_data['close'].iloc[-1]
                
                result = executor.entry(
                    signal=decision.signal,
                    volume=params.volume,
                    sl=params.stop_loss,
                    tp=params.take_profit
                )
                
                # Verify execution
                if decision.signal in ['BUY', 'SELL']:
                    assert len(executor.positions) == 1 or result['success'] == True


@pytest.mark.integration
class TestExecutionLatency:
    """Tests for critical path latency requirements."""
    
    def test_signal_generation_latency(self):
        """Signal generation should complete under 50ms."""
        probs = np.array([0.75, 0.15, 0.10])
        
        start = time.perf_counter()
        for _ in range(100):
            generate_signal(probs)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms per call
        
        assert elapsed < 50, f"Signal generation too slow: {elapsed:.2f}ms"
    
    def test_trend_detection_latency(self, mtf_data):
        """Trend detection should complete under 100ms."""
        detector = FusionFXTrendDetector(ml_model=None)
        
        # Warm up
        detector.detect_trend(mtf_data)
        
        start = time.perf_counter()
        for _ in range(10):
            detector.detect_trend(mtf_data)
        elapsed = (time.perf_counter() - start) / 10 * 1000
        
        assert elapsed < 100, f"Trend detection too slow: {elapsed:.2f}ms"
    
    def test_critical_path_latency(self, sample_ohlcv_data, mtf_data):
        """
        Full critical path: Data -> Trend -> Signal -> Risk -> Order
        Should complete under 200ms for live trading viability.
        """
        detector = FusionFXTrendDetector(ml_model=None)
        decision_engine = DecisionEngine(threshold=0.65)
        risk_manager = RiskManager(account_balance=10000)
        executor = BacktestExecutor()
        executor.current_price = sample_ohlcv_data['close'].iloc[-1]
        
        # Warm up
        detector.detect_trend(mtf_data)
        
        start = time.perf_counter()
        
        # 1. Trend detection
        trend_result = detector.detect_trend(mtf_data)
        
        # 2. Decision
        pattern_probs = [0.75, 0.15, 0.10]
        decision = decision_engine.decide(pattern_probs, trend_result)
        
        # 3. Risk check
        if decision.signal != 'NO_TRADE':
            allowed, _ = risk_manager.check_risk_limits(
                current_balance=10000,
                current_equity=10000,
                open_positions=0
            )
            
            if allowed:
                # 4. Calculate params
                params = risk_manager.get_params(sample_ohlcv_data, signal=decision.signal)
                
                # 5. Execute
                executor.entry(
                    signal=decision.signal,
                    volume=params.volume,
                    sl=params.stop_loss,
                    tp=params.take_profit
                )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 200, f"Critical path too slow: {elapsed:.2f}ms"


@pytest.mark.integration
@pytest.mark.skipif(
    not all([os.getenv('MT5_ACCOUNT'), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')]),
    reason="MT5 credentials not configured"
)
class TestMT5Integration:
    """Integration tests requiring actual MT5 connection."""
    
    @pytest.fixture
    def mt5_connector(self):
        """Create real MT5 connector from environment."""
        from trading.mt5_connector import MT5Connector
        
        connector = MT5Connector(
            account=int(os.getenv('MT5_ACCOUNT')),
            password=os.getenv('MT5_PASSWORD'),
            server=os.getenv('MT5_SERVER'),
            symbol='EURUSD',
            timeframe='H1',
        )
        yield connector
        connector.disconnect()
    
    def test_mt5_connection(self, mt5_connector):
        """Test actual MT5 connection."""
        assert mt5_connector.connect() == True
        assert mt5_connector.connected == True
    
    def test_mt5_account_info(self, mt5_connector):
        """Test fetching real account info."""
        mt5_connector.connect()
        info = mt5_connector.get_account_info()
        
        assert info is not None
        assert info.balance > 0
        assert info.equity > 0
    
    def test_mt5_fetch_data(self, mt5_connector):
        """Test fetching real market data."""
        mt5_connector.connect()
        df = mt5_connector.get_data(n=100)
        
        assert not df.empty
        assert len(df) == 100
        assert 'open' in df.columns
        assert 'close' in df.columns
    
    def test_mt5_symbol_info(self, mt5_connector):
        """Test fetching symbol specifications."""
        mt5_connector.connect()
        info = mt5_connector.get_symbol_info()
        
        assert info is not None
        assert 'point' in info
        assert 'volume_min' in info
        assert info['volume_min'] > 0