"""
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── images/
│   └── yolo_labels/
│
├── models/
│   ├── lstm.py
│   ├── vit.py
│   ├── fusion.py
│   ├── yolo_detector.py
│   └── load_models.py
│
├── training/
│   ├── train_lstm.py
│   ├── train_vit.py
│   ├── train_yolo.py
│   └── train_fusion.py
│
├── inference/
│   ├── run_prediction.py
│   └── build_features.py
│
├── trading/
│   ├── signal_engine.py
│   ├── risk_manager.py
│   ├── trade_manager.py
│   ├── mt5_connector.py
│   └── live_trading_bot.py
│
├── backtest/
│   ├── backtest_engine.py
│   └── feature_builder_bt.py
│
└── utils/
    ├── data_loader.py
    ├── candle_to_image.py
    ├── indicators.py
    └── config.py
"""