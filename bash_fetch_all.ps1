@'
import os
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

account  = int(os.environ.get("MT5_ACCOUNT", "0"))
password = os.environ.get("MT5_PASSWORD", "5gPxJr@i")
server   = os.environ.get("MT5_SERVER", "Alpari-MT5-Demo")
path     = os.environ.get("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
symbol   = os.environ.get("SYMBOL", "EURUSD")
n_bars   = int(os.environ.get("N_BARS", "1000000"))
out_dir  = os.environ.get("OUT_DIR", r"data\raw\mt5")

tfs = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

os.makedirs(out_dir, exist_ok=True)

init_kwargs = {}
if path:
    init_kwargs["path"] = path

if not mt5.initialize(**init_kwargs):
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

try:
    if account and password:
        if not mt5.login(account, password=password, server=server):
            raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")

    for tf_name, tf_const in tfs.items():
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
        if rates is None:
            print(f"[{tf_name}] no data: {mt5.last_error()}")
            continue

        df = pd.DataFrame(rates)
        if df.empty:
            print(f"[{tf_name}] empty dataframe")
            continue

        df["time"] = pd.to_datetime(df["time"], unit="s")
        out_path = os.path.join(out_dir, f"{symbol}_{tf_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[{tf_name}] wrote {len(df)} rows -> {out_path}")

finally:
    mt5.shutdown()
'@ | python -