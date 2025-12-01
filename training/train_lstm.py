# training/train_lstm.py
"""
LSTM model training script with proper data handling.
"""
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm import LSTMModel, LSTMWithAttention
from utils.data_loader import DataLoader as MyDataLoader, DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def train_lstm_model(
    data_path: str = "data/raw/eurusd_latest.csv",
    save_dir: str = "models/weights",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seq_len: int = 60,
    use_attention: bool = False,
    device: Optional[str] = None,
) -> Tuple[nn.Module, dict]:
    """
    Train LSTM model with proper data handling.
    
    Args:
        data_path: Path to training data CSV
        save_dir: Directory to save model weights
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        seq_len: Input sequence length
        use_attention: Use attention-based LSTM
        device: Training device (auto-detect if None)
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on {device}")
    
    # 1. Load and prepare data
    config = DataConfig(
        sequence_length=seq_len,
        label_strategy='ternary',
        scaler_type='minmax',
    )
    loader = MyDataLoader(config)
    
    df = loader.load_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Split WITHOUT leakage
    train_scaled, test_scaled, val_scaled = loader.split_and_scale(
        df, 
        split_ratio=0.8,
        validation_ratio=0.1,
    )
    
    # Create sequences
    X_train, y_train = loader.create_sequences(train_scaled, seq_len)
    X_test, y_test = loader.create_sequences(test_scaled, seq_len)
    
    if len(X_train) == 0:
        raise ValueError("Insufficient data for training")
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 2. Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize model
    ModelClass = LSTMWithAttention if use_attention else LSTMModel
    model = ModelClass(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
    ).to(device)
    
    logger.info(f"Model: {ModelClass.__name__}")
    
    # 4. Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()
    
    # Class weights for imbalanced data
    class_counts = torch.bincount(torch.tensor(y_train))
    if len(class_counts) == 3:
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    # 5. Training loop
    best_loss = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, mode='classify')
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x, mode='classify')
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = correct / total
        
        # Update scheduler
        scheduler.step(test_loss)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            save_path = Path(save_dir) / "lstm_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model (loss: {best_loss:.4f})")
        
        # Log progress
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2%}"
            )
    
    # 6. Save scaler for inference
    scaler_path = Path(save_dir) / "scaler.joblib"
    loader.save_scaler(scaler_path)
    
    logger.info(f"Training complete. Best test loss: {best_loss:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--data', type=str, default="data/raw/eurusd_latest.csv")
    parser.add_argument('--save-dir', type=str, default="models/weights")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--attention', action='store_true')
    args = parser.parse_args()
    
    train_lstm_model(
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        use_attention=args.attention,
    )


if __name__ == "__main__":
    main()
