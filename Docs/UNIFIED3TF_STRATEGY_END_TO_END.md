# Unified3TF Strategy (End-to-End) — From Fetching Data to Order Closing

This document describes the **actual running pipeline** for the `unified3tf` strategy in this repo, covering:

- Fetching and storing MT5 OHLCV CSVs
- Loading/alignment of 3 timeframes for backtesting
- The backtest replay loop
- How the strategy evaluates HTF/MTF/LTF using `ProbabilisticAlphaFactory`
- All gating rules and key thresholds
- Risk sizing / SL/TP placement (both base and RiskManaged variant)
- Order placement interfaces (backtest + live)
- Position tracking and order closing (SL/TP, manual close)
- Exported outputs (CSV + logs)

All references are to **current code in this repository**.

---

## 0) Key files (map)

- **CLI entry / orchestration**: `main.py`
- **Strategy**:
  - Base: `strategies/unified_3tf_strategy.py` (`Unified3TFStrategy`, `Unified3TFConfig`)
  - Risk-managed wrapper: `strategies/unified_3tf_strategy_riskmanaged.py` (`RiskManagedUnified3TFStrategy`)
- **Decision engine**: `alpha_factory/probabilistic_alpha_factory.py`
  - Factory: `create_probabilistic_alpha_factory()`
  - Core: `ProbabilisticAlphaFactory.evaluate()`
- **MH-TCN integration (provider)**: `alpha_factory/mhtcn_integration.py` (`MHTCNFeatureProvider`)
- **Feature engineering**:
  - Full 220+ features: `alpha_factory/features_engineering.py` (`FeatureEngineerOptimized.generate_features()`)
  - Fast backtest minimal features: `Unified3TFStrategy._generate_fast_features()`
- **Backtest execution engine**: `trading/backtest.py` (`BacktestExecutor`)
- **Live MT5 connector**: `trading/mt5_connector.py` (`MT5Connector`)

---

## 1) Fetching historical data from MT5 (CSV generation)

### Entry point

- Function: `cmd_fetch_data(args, logger)` in `main.py`

### Inputs

- `--symbol` (default from settings, fallback `EURUSD`)
- `--timeframes` comma-separated (e.g. `M5,M15,H1`)
- `--bars` = bar count for the **base timeframe**

### Bar-scaling rule for alignment

To keep **time spans aligned** across timeframes, the code scales the number of bars by timeframe minutes:

- Helper: `_tf_to_minutes(tf)` in `cmd_fetch_data()`
- Base timeframe = first timeframe in `--timeframes`

Formula:

```
N_fetch(tf) = ceil(N_base * minutes(base_tf) / minutes(tf))
```

Example (SCALP): base=`M5`.

- M15 bars = `N_base * 5/15 = N_base/3`
- H1 bars  = `N_base * 5/60 = N_base/12`

### Output location

- Root output dir:

```
settings.ASSETS_DIR / 'data' / 'mt5' / symbol
```

Given your configuration, typically:

```
E:\pyProject\pyForex-assets\data\mt5\EURUSD
```

### Output file naming

`{symbol}_{tf}_{stamp}.csv` where `stamp = YYYYMMDD_HHMMSS`.

Reference: `cmd_fetch_data()` in `main.py`.

---

## 2) Backtest input loading + strict 3TF alignment

### Entry point

- Function: `cmd_backtest(args, logger)` in `main.py`

### Inputs

- `--data` (LTF CSV path)
- `--data-mtf` optional (MTF CSV path)
- `--data-htf` optional (HTF CSV path)
- `--profile` (`SCALP`, `INTRADAY`, `SWING`)
- `--strategy unified3tf`

### Strict 3TF alignment window

When `--data-mtf` or `--data-htf` is provided, `cmd_backtest` loads all CSVs and trims them to the **overlapping time window**.

Implementation details:

- `_prep_time_df()` normalizes columns, parses `time`, sorts, drops duplicates.
- Overlap window:

```
overlap_start = max(start_time of each TF)
overlap_end   = min(end_time of each TF)
```

- Each TF is trimmed by:

```
mask = (time >= overlap_start) & (time <= overlap_end)
```

Reference: the 3TF mode block around creation of `ThreeTFDataProvider` in `cmd_backtest()`.

### 3TF data provider for backtest

`cmd_backtest` constructs an internal provider:

- Class: `ThreeTFDataProvider` (defined inside `cmd_backtest`)
- Method used by strategy: `get_ohlcv(symbol, timeframe, count)`

Core behavior:

- `provider.current_time` is updated each iteration to the current LTF bar timestamp.
- For each timeframe, `get_ohlcv()` returns **all rows <= current_time**, tailing `count` rows.

This guarantees HTF/MTF windows are aligned with current LTF time.

---

## 3) Backtest replay loop (event simulation)

### Where the loop is

Inside `cmd_backtest()` in the explicit 3TF mode block.

### Iteration

- `window_size = 100` (rolling LTF window passed to the strategy)
- For each i from `window_size` to end:

1) Set provider cursor:

- `provider.current_idx = i`
- `provider.current_time = base_df.iloc[i]['time']`

2) Build LTF rolling window:

- `ltf_window = provider.get_ohlcv(..., timeframe=base_tf, count=window_size)`

3) Update executor price (for SL/TP hits):

- `executor.update_price(last_close, time=exec_ts)`

4) Run strategy on this bar:

- `sig = strategy.on_bar(ltf_window)`

5) If no signal, store rejection diagnostics:

- `strategy.last_rejection_stage`
- `strategy.last_rejection_reason`

6) Save per-bar signals to `signals_out`.

After loop:

- `executor.close_all_positions()`

### Outputs produced by backtest

- `executor.get_trade_history()`
- `executor.balance`
- `signals_out`
- rejection counts and examples

Reference: `cmd_backtest()` in `main.py`.

---

## 4) Strategy construction in backtest

In 3TF mode, `cmd_backtest` instantiates:

- `BacktestExecutor(config=BacktestConfig(initial_balance=args.balance))`
- `RiskManagedUnified3TFStrategy` (preferred) OR `Unified3TFStrategy`

### Fast backtest mode

CLI flag:

- `--fast-backtest`

Behavior:

- For `--strategy unified3tf`, `fast_backtest` is enabled by default in `main.py`.
- Passed into `Unified3TFConfig(fast_backtest=True)`.

Effects inside strategy:

- Cache HTF/MTF evaluation results by TF-candle timestamp.
- Use minimal feature set (`_generate_fast_features`).
- Skip swing extraction.
- Create `ProbabilisticAlphaFactory` with `ProbabilisticConfig(key_features_only=True)`.

References:

- `main.py`: strategy init passes `fast_backtest`.
- `Unified3TFStrategy._get_engine()` sets `ProbabilisticConfig(key_features_only=True)`.
- `Unified3TFStrategy._evaluate_timeframe()` caches and uses fast features.

---

## 5) Unified3TFStrategy: per-bar lifecycle

File: `strategies/unified_3tf_strategy.py`

### Main method

- `Unified3TFStrategy.on_bar(df: pd.DataFrame) -> Optional[str]`

High-level flow:

1) Initialize lazily (`initialize()`)
2) Determine `current_time` from last row `df['time']`
3) Daily reset: `_check_daily_reset(current_time)`
4) Sync open positions from executor: `_sync_open_positions()`
5) Enforce limits: `_check_limits()`
6) Fetch multi-timeframe windows: `_fetch_mtf_data(df)`
7) Reject if insufficient bars (`sequence_length`)
8) Evaluate HTF, then MTF, then LTF via `_evaluate_timeframe()`
9) Gate checks + alignment rules
10) Enforce max open trades / prevent stacking same direction
11) Execute trade: `_execute_trade_probabilistic()`
12) Update counters (`_daily_trades`, `_last_entry_time`) and return `'BUY'`/`'SELL'` or `None`

---

## 6) Timeframe evaluation (HTF/MTF/LTF)

Method:

- `Unified3TFStrategy._evaluate_timeframe(df, timeframe, equity, signal_id)`

### 6.1 Input normalization

- Columns are lowercased.
- If `volume` missing:
  - use `tick_volume` if present
  - else fill `volume=0.0`

### 6.2 Feature generation

Two paths:

- Normal mode: `FeatureEngineerOptimized.generate_features(d0, batch_processing=False)`
- Fast backtest mode:
  - `Unified3TFStrategy._generate_fast_features(d0)`

Fast feature set includes:

- `rsi`
- `macd`, `macd_histogram`
- `adx`
- `bb_position`
- `atr_ratio`
- `momentum`
- `trend_strength`
- `volatility_ratio`

### 6.3 Swing points extraction

- Normal mode:
  - If features contain `swing_high`/`swing_low`, convert them into `SwingPoint` list
  - Else fallback to `MarketData(d0).extract_swings(lookback=5, strength_threshold=0.3)`
- Fast backtest mode:
  - **Skip swing extraction** and pass `swing_points=None`

### 6.4 Decision engine call

- Engine is created by `Unified3TFStrategy._get_engine()` which calls:

`alpha_factory.probabilistic_alpha_factory.create_probabilistic_alpha_factory()`

Then:

`ProbabilisticAlphaFactory.evaluate(df, features, timeframe, swing_points, causality_results=None, current_equity, signal_id)`

Output type:

- `DecisionOutput` (from `alpha_factory/probabilistic_alpha_factory.py`)

Key fields used by strategy:

- `direction` in `{LONG, SHORT, HOLD}`
- `confidence` (decayed)
- `stability_score`
- `final_p_bull`, `final_p_bear` (used for directional score)
- `size_multiplier`

---

## 7) ProbabilisticAlphaFactory: decision stages and formulas

File: `alpha_factory/probabilistic_alpha_factory.py`

Main method:

- `ProbabilisticAlphaFactory.evaluate(...) -> DecisionOutput`

Stages:

### Stage 1 — Regime prior

- `RegimePriorCalculator.calculate(df, swing_points)`
- If `swing_points` absent, it falls back to price-change based slope.

### Stage 2 — Feature probability calibration

- `_calibrate_features(features, regime_context, causality_results)`

If `config.key_features_only=True`, only features in `config.key_features` are calibrated.

Default key features (see `ProbabilisticConfig.__post_init__`):

- `rsi`, `macd`, `macd_histogram`, `adx`, `bb_position`, `atr_ratio`, `momentum`, `trend_strength`, `volatility_ratio`

### Stage 3 — Stability

- `_calculate_stability(df)`

Formula:

- `returns = pct_change(close)`
- `vol = std(returns)`
- `normalized_vol = clip(vol / 0.01, 0, 2)`
- `stability = 1 - normalized_vol/2`
- clamp to `[0.2, 1.0]`

### Stage 4 — MH-TCN probabilities (optional)

- If `mhtcn_provider` exists, `_get_mhtcn_probs(df, timeframe)` uses `provider.predict()`

It maps direction probabilities to:

- `bear = direction_probs[0]`
- `neutral = direction_probs[1]`
- `bull = direction_probs[2]`

### Stage 5 — Evidence aggregation

- `EvidenceAggregator.aggregate(regime_prior, feature_probs, stability_score, mhtcn_probs)`

Returns:

- `(p_bull, p_bear, p_neutral, contributions)`

### Stage 6 — Dynamic threshold

- `threshold = DynamicThresholdCalculator.calculate_threshold(current_vol, recent_drawdown)`

Where:

- `current_vol = std(pct_change(close).tail(20))`
- drawdown updated via `threshold_calculator.update_drawdown(current_equity)`

### Stage 7 — Alpha decay

- `bars_since = alpha_decay.get_bars_since_signal(signal_id)`
- `decayed_confidence = alpha_decay.apply_decay(signal_id, confidence, bars_since)`

Decision:

- LONG if `p_bull > threshold` and `p_bull > p_bear`
- SHORT if `p_bear > threshold` and `p_bear > p_bull`
- else HOLD

Size multiplier (excess over threshold):

```
size_mult = (p - threshold) / (1 - threshold)
size_mult is clipped to [0,1]
```

Returned `DecisionOutput.confidence` is **decayed_confidence**.

---

## 8) Strategy gating & alignment rules

File: `strategies/unified_3tf_strategy.py`

### Gate check

- `_passes_gate(out, min_conf, min_stability)`

Conditions:

- `out.confidence >= min_conf`
- `out.stability_score >= min_stability`

### Directional score check

- `_directional_score_ok(out)`

Directional score formula:

```
score = abs(out.final_p_bull - out.final_p_bear)
pass if score >= config.min_directional_score
```

### SCALP relaxed alignment

`Unified3TFConfig.__post_init__` modifies SCALP defaults:

- `relaxed_alignment = True`
- `max_open_trades = 1`
- `max_daily_trades = 2`
- thresholds raised vs previous run:
  - `min_htf_confidence = 0.56`
  - `min_mtf_confidence = 0.62`
  - `min_ltf_confidence = 0.66`
  - `min_stability = 0.45`
  - `min_directional_score = 0.18`

Relaxed alignment logic:

- Require `MTF.direction == LTF.direction`
- HTF is optional, BUT if HTF passes its gate and is non-HOLD:
  - veto if `HTF.direction != LTF.direction`

Non-relaxed (strict) alignment:

- Require `MTF.direction == HTF.direction`
- Require `LTF.direction == HTF.direction`

---

## 9) Limits, daily reset, cooldown

### Daily reset

- `_check_daily_reset(current_time)`

Resets:

- `_daily_trades = 0`
- `_daily_pnl = 0.0`

when date changes.

### Limits

- `_check_limits()`

Checks:

- max daily trades: `_daily_trades >= config.max_daily_trades`
- max open trades: `len(_open_positions) >= config.max_open_trades`
- daily loss limit (uses executor balance and `_daily_pnl`)

Note: `_daily_pnl` is currently not updated in `Unified3TFStrategy` on closes; it is reset daily and otherwise stays 0.0 unless extended.

### Cooldown (RiskManaged variant)

File: `strategies/unified_3tf_strategy_riskmanaged.py`

- `RiskManagedUnified3TFStrategy.on_bar()` checks `_cooldown_ok(now)` before calling `super().on_bar()`.
- `_cooldown_ok` uses the max of:
  - `_last_entry_time` (set on entry)
  - `_last_exit_time` (inferred from `executor.trade_history[-1].exit_time`)

Rejection:

- `last_rejection_stage = "COOLDOWN"`

---

## 10) Position tracking inside strategy

Method:

- `_sync_open_positions()`

Sources:

- If executor has `get_open_positions()`:
  - uses it (with or without `symbol=` depending on signature)
- Else if executor has `.positions` list

Stored as:

- `self._open_positions: Dict[str, dict]` keyed by ticket.

Anti-stacking rule:

- `_has_open_direction(direction)` prevents multiple BUYs or SELLs stacked.

---

## 11) Order placement (entry)

### Strategy call site

- In `Unified3TFStrategy.on_bar()`:

`self._execute_trade_probabilistic(direction, ltf_out, data_ltf)`

### Base Unified3TFStrategy execution

File: `strategies/unified_3tf_strategy.py`

- `_execute_trade_probabilistic(direction, decision_out, df)`

Steps:

1) `entry_price = last close`
2) Compute SL/TP: `_atr_sltp(df, entry_price, direction)`
3) Compute lots: `_calculate_position_size(entry_price, sl, size_multiplier)`
4) Place order:

- Prefer `executor.entry(signal, volume, sl, tp)`
- Else fallback `executor.open_position(...)`

### RiskManagedUnified3TFStrategy execution

File: `strategies/unified_3tf_strategy_riskmanaged.py`

Overrides `_execute_trade_probabilistic()`:

- Attempts to use:
  - `risk_management.phase2_risk_calc.sl_tp_calculator.calculate_sl_tp_from_predictions`
  - `risk_management.phase2_risk_calc.position_sizing.PositionSizingCalculator.calculate`

It optionally calls the MH-TCN provider directly:

- `engine.mhtcn_provider.predict(df.copy(), LTF_timeframe)`

If predictions are unavailable, it falls back to base ATR SL/TP.

Then:

- Enforces `min_sl_pips`
- clamps `volume`:
  - minimum 0.01
  - maximum `config.max_lot`
  - rounds to 2 decimals

Finally:

- `executor.entry(...)`

---

## 12) SL/TP formulas used

### 12.1 Base ATR SL/TP (Unified3TFStrategy)

Method:

- `_atr_sltp(df, entry_price, direction)`

True Range:

```
TR = max(
  high - low,
  abs(high - prev_close),
  abs(low - prev_close)
)
```

ATR:

```
ATR = mean(TR, period=14)
```

SL distance:

```
sl_mult = 1.5
sl_dist = ATR * sl_mult
```

TP distance:

```
rr = config.min_risk_reward (default 2.0)
TP_dist = SL_dist * rr
```

Direction:

- BUY:
  - `SL = entry - sl_dist`
  - `TP = entry + tp_dist`
- SELL:
  - `SL = entry + sl_dist`
  - `TP = entry - tp_dist`

Fallback:

- `_fallback_sltp()` uses fixed pips:
  - `SL=30 pips`, `TP=60 pips`, pip=0.0001.

### 12.2 RiskManaged SL/TP

Uses:

- `calculate_sl_tp_from_predictions(entry_price, direction, predictions, atr, config=SLTPConfig(min_risk_reward=...))`

If SL is too tight (`sl_pips < min_sl_pips`), it expands SL and recomputes TP with the configured risk-reward.

---

## 13) Position sizing formulas used

### 13.1 Base position sizing (Unified3TFStrategy)

Method:

- `_calculate_position_size(entry_price, stop_loss, size_multiplier)`

Risk amount:

```
risk_amount = balance * (base_risk_percent / 100) * size_multiplier
```

SL pips:

```
pip_value = 0.0001 (or 0.01 for JPY pairs)
sl_pips = abs(entry - stop_loss) / pip_value

effective_sl_pips = max(sl_pips, min_sl_pips)
```

Lots (assumes $10/pip/lot):

```
pip_value_per_lot = 10.0
lots = risk_amount / (effective_sl_pips * pip_value_per_lot)
```

Clamp:

- `lots ∈ [0.01, max_lot]`
- round to 2 decimals

### 13.2 RiskManaged sizing

Uses:

- `PositionSizingCalculator.calculate(account_balance, entry_price, stop_loss, pair, direction_confidence, volatility)`

Then multiplies by:

- `DecisionOutput.size_multiplier`

Finally clamps/rounds lots similarly.

---

## 14) Live order placement (MT5)

File: `trading/mt5_connector.py`

Primary method:

- `MT5Connector.execute_order(signal, volume, sl, tp, symbol=None, comment='PyForex')`

Key points:

- Uses `mt5.symbol_info_tick(sym)`.
- BUY uses `tick.ask`, SELL uses `tick.bid`.
- Builds MT5 request with:
  - `action=TRADE_ACTION_DEAL`
  - `type=ORDER_TYPE_BUY/SELL`
  - includes `sl`, `tp`, `magic`, `comment`
- Sends via `mt5.order_send(request)`.

Strategy interface uses alias:

- `MT5Connector.entry(...)` calls `execute_order(...)`.

Open positions:

- `MT5Connector.get_open_positions(symbol=None)` uses `mt5.positions_get()`.

Manual close:

- `MT5Connector.close_position(ticket)` sends reverse market order against position.

---

## 15) Backtest execution and order closing

File: `trading/backtest.py`

### Entry execution (simulated)

Method:

- `BacktestExecutor.entry(signal, volume, sl, tp)`

Spread model:

- BUY entry price = `current_price + spread_pips*0.0001`
- SELL entry price = `current_price - spread_pips*0.0001`

Commission model:

- On entry:

```
commission = commission_per_lot * volume
balance -= commission
```

### Price updates and SL/TP hit detection

Method:

- `BacktestExecutor.update_price(price, time=None)`

For each open position:

- BUY:
  - if `price <= sl` => close at `sl` (CLOSED_SL)
  - elif `price >= tp` => close at `tp` (CLOSED_TP)
- SELL:
  - if `price >= sl` => close at `sl`
  - elif `price <= tp` => close at `tp`

### Closing P&L formulas

Method:

- `BacktestExecutor._close_position(pos, reason, exit_price)`

Pips:

- BUY:

```
pips = (exit_price - entry_price) / 0.0001
```

- SELL:

```
pips = (entry_price - exit_price) / 0.0001
```

Gross P&L:

```
pnl = pips * pip_value * volume
```

Where `pip_value` defaults to `$10 per pip per lot`.

Net P&L recorded into trade history (to reflect commission already deducted at entry):

```
commission = commission_per_lot * volume
pnl_net = pnl - commission
trade.pnl = pnl_net
```

Balance update:

- balance adds **gross** pnl here, because commission was already removed at entry:

```
balance += pnl
```

### Forced close at end of backtest

- `BacktestExecutor.close_all_positions()` closes remaining positions at current price.

---

## 16) Backtest exports (CSV)

File: `main.py` (`cmd_backtest`)

If you pass `--export-csv`:

- Export directory:

- If `--export-dir` provided: that path
- Else default:

```
settings.ASSETS_DIR / 'backtests'
```

Files:

- Trades:

`backtest_{strategy}_{symbol}_{profile}_trades_{stamp}.csv`

- Summary:

`backtest_{strategy}_{symbol}_{profile}_summary_{stamp}.csv`

Where `stamp = YYYYMMDD_HHMMSS`.

Summary row fields include:

- `strategy, symbol, profile, period_start, period_end, candles, initial_balance, final_balance, pnl, pnl_pct, total_trades`

---

## 17) Known critical behaviors that affect results

These are not opinions; they are properties of the current code:

- **Backtest frequency**: `on_bar()` is called for every base TF bar, and will attempt to trade if gates pass.
- **No partial closes**: only SL/TP closes are modeled in `BacktestExecutor.update_price()`.
- **No trailing stop**: none in `Unified3TFStrategy` base.
- **Daily loss tracking**: `_daily_pnl` is not updated on closes in `Unified3TFStrategy`.
- **Spread + commission** are applied in backtest engine (see Section 15).

---

## 18) How to trace a single trade in logs

To trace a trade end-to-end:

1) In backtest log, find:

- `Trade signal: BUY/SELL EURUSD (conf=...)` from `Unified3TFStrategy.on_bar()`

2) Then executor log lines:

- `[BACKTEST] Opened BUY/SELL ... (SL..., TP...)`
- later
- `[BACKTEST] Closed BUY/SELL ... (CLOSED_SL/CLOSED_TP) P&L: ...`

---

## 19) Recommended “review checklist”

- Verify `Unified3TFConfig` parameters for your profile (SCALP)
- Verify directional gating uses `final_p_bull/final_p_bear`
- Verify risk sizing assumptions (`$10/pip/lot`) match your intended backtest model
- Verify commission (`$7/lot`) and spread (`1.0 pips`) are realistic for your broker
- Verify `window_size` and `sequence_length` are sufficient for stable evaluation

---

End of document.
