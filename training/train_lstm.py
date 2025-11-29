# training/train_lstm.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from models.lstm import LSTMModel
from utils.data_loader import DataLoader as MyLoader

device = "cuda"

loader = MyLoader()
df = loader.load_csv("data/raw/eurusd.csv")
scaled = loader.scale(df)
X, y = loader.create_sequences(scaled)

dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y))
dloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(25):
    for bx, by in dloader:
        bx, by = bx.to(device), by.to(device)

        pred = model(bx)
        loss = loss_fn(pred, by)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} | Loss: {loss.item()}")

torch.save(model.state_dict(), "models/lstm_best.pt")
