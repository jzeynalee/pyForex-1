# utils/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def load_csv(self, path):
        df = pd.read_csv(path)
        df = df[['open','high','low','close','tick_volume']]
        df.dropna(inplace=True)
        return df

    def scale(self, df):
        scaled = self.scaler.fit_transform(df)
        return scaled

    def create_sequences(self, data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)-1):
            X.append(data[i-seq_len:i])
            y.append(self._label(data[i], data[i+1]))
        return np.array(X), np.array(y)

    def _label(self, cur, nxt):
        return 1 if nxt[3] > cur[3] else 0
