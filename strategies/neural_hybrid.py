# strategies/neural_hybrid.py 

from .base import Strategy
from trading.signal_engine import generate_signal
from inference.predictor import HybridPredictor # The refactored predictor we discussed

class NeuralHybridStrategy(Strategy):
    def __init__(self, data_provider, executor, risk_manager):
        super().__init__(data_provider, executor)
        self.predictor = HybridPredictor()
        self.risk_manager = risk_manager

    def on_bar(self, df):
        # 1. Prediction
        probs = self.predictor.predict(df)
        
        # 2. Signal Generation
        signal = generate_signal(probs)
        
        # 3. Execution (Works for both Live and Backtest)
        if signal != "NO_TRADE":
            # Calculate dynamic risk
            vol, sl, tp = self.risk_manager.get_params(df, signal)
            
            # The executor determines if this is a real MT5 order or a simulated backtest trade
            self.executor.entry(signal, vol, sl, tp)