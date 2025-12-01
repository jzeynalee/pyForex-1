# utils/data_loader.py
# utils/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False

    def load_csv(self, path):
        df = pd.read_csv(path)
        # Ensure correct column names and types
        df.columns = [c.lower() for c in df.columns]
        df = df[['open','high','low','close','tick_volume']]
        df.dropna(inplace=True)
        return df

    def split_and_scale(self, df, split_ratio=0.8):
        """
        Splits data FIRST, then fits scaler on training data only to avoid leakage.
        """
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Fit ONLY on training data
        self.scaler.fit(train_df)
        self.fitted = True

        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)

        return train_scaled, test_scaled

    def create_sequences(self, data, seq_len=60):
        """
        Creates sequences X (0 to t-1) and targets y (t).
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted! Call split_and_scale first.")

        X, y = [], []
        # We need seq_len history to predict index i
        for i in range(seq_len, len(data)):
            # Input: Sequence from i-seq_len to i-1
            X.append(data[i-seq_len:i])
            
            # Target: The Close price direction of the CURRENT candle (i) 
            # relative to previous candle (i-1)
            # Or use Close(i) > Open(i) for Green/Red prediction
            current_close = data[i][3] # Index 3 is Close
            prev_close = data[i-1][3]
            
            label = 1 if current_close > prev_close else 0
            y.append(label)
            
        return np.array(X), np.array(y)