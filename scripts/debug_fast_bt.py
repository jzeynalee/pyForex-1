"""Quick diagnostic: why does fast backtest produce 0 trades?"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("debug_fast_bt")
logger.setLevel(logging.DEBUG)

df = pd.read_csv("E:/pyProject/data/raw/EURUSD_H1_latest.csv")
df.columns = [c.lower().strip() for c in df.columns]
df["time"] = pd.to_datetime(df["time"])
df = df.tail(500).reset_index(drop=True)
if "volume" not in df.columns:
    df["volume"] = df.get("tick_volume", 100)

# Avoid circular import in strategies/__init__.py by importing directly
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "unified_3tf_fast_backtest",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "strategies", "unified_3tf_fast_backtest.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["strategies.unified_3tf_fast_backtest"] = _mod
_spec.loader.exec_module(_mod)
create_fast_backtest_strategy = _mod.create_fast_backtest_strategy

cls = create_fast_backtest_strategy(df, profile="INTRADAY", use_fast_features=True)

from trading.backtest import BacktestExecutor, BacktestConfig

config = BacktestConfig(initial_balance=10000.0)
executor = BacktestExecutor(config=config)


class FakeDP:
    def __init__(self, data):
        self.data = data
        self.current_idx = 200

    def get_data(self, n):
        return self.data.iloc[max(0, self.current_idx - n) : self.current_idx]


dp = FakeDP(df)
strat = cls(data_provider=dp, executor=executor)

# Check store contents
store = cls._store
print("=== PrecomputedStore ===")
for tf in store._ohlcv:
    ohlcv = store._ohlcv[tf]
    feats = store._features[tf]
    fshape = feats.shape if feats is not None and not feats.empty else "EMPTY"
    print(f"  {tf}: ohlcv={len(ohlcv)} bars, features={fshape}")

# Check config
print(f"\n=== Config ===")
print(f"  htf={strat.config.htf}, mtf={strat.config.mtf}, ltf={strat.config.ltf}")
print(f"  sequence_length={strat.config.sequence_length}")
print(f"  min_htf_confidence={strat.config.min_htf_confidence}")
print(f"  min_mtf_confidence={strat.config.min_mtf_confidence}")
print(f"  min_ltf_confidence={strat.config.min_ltf_confidence}")
print(f"  min_stability={strat.config.min_stability}")
print(f"  relaxed_alignment={getattr(strat.config, 'relaxed_alignment', False)}")

# Run on_bar for several bars, track rejections
rejections = {}
signals = 0
for i in range(200, 450):
    w = df.iloc[max(0, i - 200) : i].reset_index(drop=True)
    executor.update_price(float(w["close"].iloc[-1]))
    sig = strat.on_bar(w)
    if sig:
        signals += 1
        print(f"  Bar {i}: SIGNAL={sig}")
        if signals >= 3:
            break
    else:
        stage = strat.last_rejection_stage
        reason = strat.last_rejection_reason
        rejections[stage] = rejections.get(stage, 0) + 1
        if i <= 205 or i % 50 == 0:
            print(f"  Bar {i}: REJECTED stage={stage} reason={reason[:120]}")

print(f"\n=== Summary ===")
print(f"Total signals: {signals}")
print(f"Rejection counts: {rejections}")

# Direct evaluate_fast test
print(f"\n=== Direct evaluate_fast ===")
engine = strat._get_engine()
if engine:
    end_time = df["time"].iloc[300]
    for tf in [strat.config.htf, strat.config.mtf, strat.config.ltf]:
        fw = store.get_feature_window(tf, end_time, 60)
        ow = store.get_ohlcv_window(tf, end_time, 200)
        if fw is not None and ow is not None:
            out = engine.evaluate_fast(
                df=ow, features=fw, timeframe=tf, current_equity=10000
            )
            print(
                f"  {tf}: dir={out.direction} conf={out.confidence:.3f} "
                f"p_bull={out.final_p_bull:.3f} p_bear={out.final_p_bear:.3f} "
                f"thresh={out.threshold_used:.3f} stab={out.stability_score:.3f}"
            )
        else:
            print(f"  {tf}: fw={'None' if fw is None else fw.shape} ow={'None' if ow is None else ow.shape}")
