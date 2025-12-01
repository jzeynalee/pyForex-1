# training/train_lstm.py
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from models.lstm import LSTMModel
from utils.data_loader import DataLoader as MyLoader

def train_lstm_model(data_path="data/raw/eurusd_latest.csv", save_path="models/lstm_best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training LSTM on {device}...")

    # 1. Load & Process Data
    loader = MyLoader()
    df = loader.load_csv(data_path)
    
    # Split & Scale (Using the fixed method from previous steps)
    train_scaled, _ = loader.split_and_scale(df, split_ratio=1.0) # Use all data for retraining
    X, y = loader.create_sequences(train_scaled)

    if len(X) == 0:
        logging.warning("Not enough data to train LSTM")
        return

    # 2. PyTorch Setup
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y))
    dloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    for epoch in range(10): # Reduced epochs for auto-retrain speed
        total_loss = 0
        for bx, by in dloader:
            bx, by = bx.to(device), by.to(device)
            
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        logging.info(f"Epoch {epoch} | Loss: {total_loss / len(dloader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), save_path)
    logging.info(f"LSTM Model saved to {save_path}")

if __name__ == "__main__":
    train_lstm_model()