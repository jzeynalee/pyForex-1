# training/auto_retrain.py
import logging
from trading.mt5_connector import MT5Connector
from training.train_lstm import train_lstm_model
# from training.train_fusion import train_fusion_model # Uncomment when implemented

logging.basicConfig(level=logging.INFO)

def auto_retrain_job():
    logging.info("🔄 Starting Weekly Retraining Job...")
    
    # 1. Fetch Latest Data using the Class-based Connector
    connector = MT5Connector()
    if not connector.connect():
        logging.error("❌ Could not connect to MT5 for retraining.")
        return

    logging.info("⬇️ Downloading latest 5000 candles...")
    # Using the same .get_data method but requesting more history
    df = connector.get_data(n=5000)
    
    if df.empty:
        logging.error("❌ No data received.")
        return

    # Save to CSV for the training scripts to read
    csv_path = "data/raw/eurusd_latest.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"✅ Data saved to {csv_path}")
    
    # 2. Retrain LSTM
    try:
        train_lstm_model(data_path=csv_path)
    except Exception as e:
        logging.error(f"❌ LSTM Training Failed: {e}")

    # 3. Retrain Fusion (Placeholder)
    # try:
    #     train_fusion_model()
    # except Exception as e:
    #     logging.error(f"Fusion Training Failed: {e}")
    
    logging.info("✅ Retraining Complete. New models ready for Inference.")

if __name__ == "__main__":
    auto_retrain_job()