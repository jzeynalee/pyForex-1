# pyForex-1 Comprehensive Backtesting Plan

**Version:** 2.0  
**Date:** December 2024  
**Status:** Implementation Ready

---

## Executive Summary

This plan defines a **zero-divergence backtesting framework** that runs the exact production pipeline using historical data. The backtest reuses all existing modules—`NeuralHybridStrategy`, `EnhancedDecisionEngine`, `RiskManager` (Phases 1-5), `HybridPredictor`, and `MTFTrendDetector`—with minimal new orchestration code.

---

## 1. Core Backtesting Principles (Non-Negotiable)

| # | Principle | Implementation |
|---|-----------|----------------|
| 1 | Zero code divergence between live and backtest | Connector injection via `TradingBot(connector=BacktestConnector)` |
| 2 | Event-driven replay, not candle loops | `BacktestEngine` processes `BAR_CLOSE` events |
| 3 | ML weights frozen unless explicitly testing retraining | `model.eval()` + `requires_grad=False` |
| 4 | Execution realism > optimistic fills | Slippage, spread, commission, latency simulation |
| 5 | Risk system tested as first-class ML component | Full Phase 1-5 pipeline with weight validation |

---

## 2. Backtesting Execution Modes

| Mode | Purpose | Implementation |
|------|---------|----------------|
| `HISTORICAL_REPLAY` | Deterministic evaluation | Default mode, sequential bar processing |
| `WALK_FORWARD` | Generalization testing | Rolling train/validate/trade windows |
| `PAPER_REPLAY` | Live-like timing | Real-time delays between bars |
| `STRESS_SIMULATION` | Adversarial markets | Injected volatility spikes, gaps, spread explosions |

Each mode reuses identical pipeline stages via the same `BacktestOrchestrator`.

---

## 3. Phase 1: Data Ingestion Layer

### 3.1 Data Sources

| Source | Module | Status |
|--------|--------|--------|
| CSV files | `trading/data_loader.py` | ✅ Exists |
| MT5 history export | `trading/data_loader.py` | ✅ Exists |
| Parquet files | `utils/features_engineering.py` | ✅ Exists |
| Synthetic data | `trading/data_loader.py` | ✅ Exists |

### 3.2 Data Validation Tests

| Test | Description | Module |
|------|-------------|--------|
| Timestamp monotonicity | No backward time jumps | `backtesting/data_validator.py` |
| TF alignment | Lower TF strictly inside higher TF | `backtesting/data_validator.py` |
| Gap detection | Weekend/holiday gaps identified | `backtesting/data_validator.py` |
| Session tagging | Asia/London/NY session markers | `trading/data_loader.py` |
| Spread reconstruction | Per-bar spread realism | `BacktestConnector` |
| OHLC integrity | High >= Low, Open/Close within range | `backtesting/data_validator.py` |

### 3.3 Failure Injection Points

```python
# In BacktestConnector
failure_injection = {
    'missing_bars': 0.001,      # 0.1% random bar drops
    'delayed_ticks': 50,        # ms latency
    'duplicate_ticks': 0.0005,  # 0.05% duplicates
    'spread_spikes': 0.01,      # 1% chance of 3x spread
}
```

### 3.4 Multi-Timeframe Data Loading

```python
# Required for MTFTrendDetector
mtf_data = {
    'M5': df_m5,    # For SCALP profile
    'M15': df_m15,  # For SCALP/SWING
    'H1': df_h1,    # Primary for INTRADAY
    'H4': df_h4,    # For SWING
    'D1': df_d1,    # Higher TF context
}
```

---

## 4. Phase 2: Feature Engineering Layer

### 4.1 Feature Categories (220+ Features)

| Category | Examples | Module |
|----------|----------|--------|
| Price | OHLC, returns, log-returns | `utils/features_engineering.py` |
| Moving Averages | SMA, EMA, DEMA, TEMA, HMA, LSMA, McGinley | `utils/features_engineering.py` |
| Volatility | ATR, Bollinger, Keltner, realized vol | `utils/features_engineering.py` |
| Momentum | RSI, MACD, Stochastic, CCI, Williams %R | `utils/features_engineering.py` |
| Trend | ADX, Aroon, Ichimoku, PSAR | `utils/features_engineering.py` |
| Structure | Swings, Higher Lows, Lower Highs, ABC patterns | `utils/features_engineering.py` |
| Volume | OBV, MFI, VWAP, Volume Profile | `utils/features_engineering.py` |
| Temporal | Time-of-day, session, day-of-week | `utils/features_engineering.py` |
| Visual | Chart images (for YOLO/ViT) | Chart rendering |

### 4.2 Top Features Selection

```python
# training/feature_selector.py
class DynamicFeatureSelector:
    def select(self, df, target_col, exclude_cols) -> list:
        """
        Selects top N features based on Random Forest importance.
        - Uses recent data for relevance
        - Returns ranked feature list
        """
```

**Current Top Features (from training):**
- Trend indicators: `adx`, `aroon_osc`, `trend_medium`
- Momentum: `rsi`, `macd_hist`, `stoch_k`
- Volatility: `atr_14`, `bb_width`, `keltner_width`
- Structure: `swing_high`, `swing_low`, `higher_lows`

### 4.3 Feature Validation Rules

| Rule | Description | Check |
|------|-------------|-------|
| Causal computation | No future data leakage | Rolling windows only use past data |
| History availability | Rolling window ≤ available bars | `warmup_bars >= max_lookback` |
| TF alignment | All features aligned to decision timestamp | Timestamp validation |
| NaN handling | No NaN in model inputs | `fillna(0)` or forward-fill |

### 4.4 Feature Drift Tracking

```python
# ml/drift_detector.py
class DriftDetector:
    def detect(self, reference_data, current_data):
        """
        - Distribution shift (KL divergence)
        - Feature importance decay
        - Feature saturation detection
        """
```

---

## 5. Phase 3: Analysis Layer (ML Models)

### 5.1 Model Inventory & Weight Validation

#### TCN Models (Per Profile × Per Timeframe)

| Profile | Timeframes | Weight Files | Status |
|---------|------------|--------------|--------|
| SCALP | M5, M15, H1 | `scalp_m5_best.pt`, `scalp_m15_best.pt`, `scalp_h1_best.pt` | ✅ Exist |
| INTRADAY | M15, H1, H4 | `intraday_m15_best.pt`, `intraday_h1_best.pt`, `intraday_h4_best.pt` | ✅ Exist |
| SWING | H1, H4, D1 | `swing_h1_best.pt`, `swing_h4_best.pt`, `swing_d1_best.pt` | ✅ Exist |

#### Other Models

| Model | Weight File | Status |
|-------|-------------|--------|
| TCN (generic) | `tcn_best.pt` | ✅ Exists |
| TCN Enhanced | `tcn_enhanced_best.pt` | ✅ Exists |
| Fusion | `fusion_best.pt` | ✅ Exists |
| ViT | `vit_best.pt` | ⚠️ Check needed |
| YOLO | `yolo_patterns.pt` | ⚠️ Check needed |
| Meta-labeling | `meta_model.joblib` | ⚠️ Check needed |
| Exit Advisor | `exit_model.pt` | ⚠️ Check needed |

### 5.2 Model Execution Integrity Logging

Each model prediction logs:
```python
logger.info(
    f"[{model_name}] Hash:{input_hash} | Latency:{latency_ms:.2f}ms | "
    f"Out:{signal}({confidence:.2f}) | Vol:{volatility:.5f}"
)
```

### 5.3 Model-Specific Backtests

#### TCN
- Sequence length sensitivity (30, 60, 120 bars)
- Regime persistence accuracy
- Overreaction to noise (false signal rate)

#### YOLO
- Pattern false positive rate
- Pattern lifespan validity (how long patterns remain actionable)
- Pattern overlap conflict resolution

#### ViT
- Structural misclassification rate
- Confidence decay over time
- Image normalization robustness

### 5.4 Model Ablation Matrix

| Variant | Components | Purpose |
|---------|------------|---------|
| Full Stack | TCN + ViT + YOLO + Fusion | Production baseline |
| −TCN | ViT + YOLO | Visual-only performance |
| −YOLO | TCN + ViT | Without pattern detection |
| −ViT | TCN + YOLO | Without chart classification |
| TCN Only | TCN | Sequence-only baseline |

---

## 6. Phase 4: Decision-Making Layer

### 6.1 Decision Engine: `EnhancedDecisionEngine`

**Location:** `trading/decision_engine.py`

#### Decision Pipeline (6 Steps)

```
Step 1: Capital Protection Pre-Check (Phase 5)
    ↓
Step 2: Direction Extraction from Predictions
    ↓
Step 3: Market Regime Detection
    ↓
Step 4: SL/TP Calculation (Phase 2)
    ↓
Step 5: Position Sizing (Phase 2)
    ↓
Step 6: Hard Rules Validation (Phase 2)
    ↓
Step 7: Meta-labeling Filter (Phase 3)
    ↓
Step 8: MTF Alignment Check
    ↓
Step 9: Capital Protection Final Check (Phase 5)
    ↓
TRADE APPROVED or REJECTED
```

### 6.2 Decision Inputs

| Input | Source | Required |
|-------|--------|----------|
| `direction_probs` | TCN/HybridPredictor | ✅ Yes |
| `volatility` | TCN risk head | ✅ Yes |
| `quantiles` | TCN risk head | ✅ Yes |
| `entry_price` | Current market price | ✅ Yes |
| `account_balance` | Connector | ✅ Yes |
| `market_data` | DataFrame with OHLCV | ✅ Yes |
| `mtf_data` | Multi-timeframe dict | Optional |
| `current_spread` | Connector | Optional |

### 6.3 Decision Logic Validation Tests

| Test | Description | Threshold |
|------|-------------|-----------|
| Confidence gating | Min direction confidence | ≥ 0.55 |
| Risk-reward check | Min R:R ratio | ≥ 1.5 |
| MTF alignment | Higher TF agreement | ≥ 0.6 |
| Meta-score filter | Meta-labeling approval | ≥ 0.5 |
| Regime restriction | No trading in VOLATILE regime | Configurable |
| Cooldown enforcement | Min time between trades | Profile-dependent |

### 6.4 Decision Metrics to Track

| Metric | Description |
|--------|-------------|
| Signal acceptance ratio | Approved / Total signals |
| Average confidence | Mean confidence of approved trades |
| Decision entropy | Diversity of decisions |
| Reversal frequency | Signal flips within N bars |
| Rejection breakdown | Count per rejection reason |

---

## 7. Phase 5: ML-Based Risk Management

### 7.1 Risk Management Architecture (5 Phases)

```
┌─────────────────────────────────────────────────────────────┐
│                    RiskManager (v2)                         │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Predictive Foundation                             │
│  ├── MultiHeadTCN (direction, volatility, quantiles)        │
│  └── Weights: models/weights/{profile}_{tf}_best.pt         │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Risk Calculations                                 │
│  ├── SLTPCalculator (quantile-based SL/TP)                  │
│  ├── PositionSizingCalculator (volatility-adjusted)         │
│  └── TradeGatekeeper (hard rules)                           │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Trade Filtering                                   │
│  ├── TripleBarrierLabeler                                   │
│  ├── MetaLabelingModel                                      │
│  └── TradeFilter                                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: Adaptive Exit (RL)                                │
│  ├── ExitTradingEnv                                         │
│  ├── PPOAgent                                               │
│  └── ExitAdvisor                                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: Capital Protection                                │
│  ├── CapitalProtector                                       │
│  ├── ProtectionManager                                      │
│  └── TradingGuard                                           │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Risk Model Inputs

| Input | Source | Used By |
|-------|--------|---------|
| Volatility | Feature layer / TCN | SL/TP, Position sizing |
| Structure | Swing points | SL placement |
| Confidence | Decision layer | Position sizing |
| Account state | Equity, DD | Capital protection |
| Regime | TCN / RegimeDetector | All phases |
| Quantiles | TCN risk head | SL/TP calculation |

### 7.3 Risk Outputs

| Output | Description | Phase |
|--------|-------------|-------|
| `stop_loss` | Price level | Phase 2 |
| `take_profit` | Price level | Phase 2 |
| `position_size` | Lots | Phase 2 |
| `risk_amount` | $ at risk | Phase 2 |
| `meta_score` | Trade quality | Phase 3 |
| `exit_action` | HOLD/CLOSE/PARTIAL | Phase 4 |
| `protection_level` | NORMAL/CAUTION/CRITICAL | Phase 5 |

### 7.4 Risk Backtests

#### SL/TP Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MAE (Max Adverse Excursion) | Worst drawdown during trade | < SL distance |
| MFE (Max Favorable Excursion) | Best unrealized profit | > TP distance |
| Stop-hunt sensitivity | SL hit then price reverses | < 10% of losses |
| Exit efficiency | Actual exit vs optimal exit | > 70% |

#### Drawdown Control Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| DD depth vs rule | Actual DD vs configured max | DD < max_dd |
| Recovery speed | Bars to recover from DD | < 100 bars |
| Capital preservation | Balance after max DD | > 80% of peak |

#### Trailing Logic Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Premature exit rate | Exits before TP | < 30% |
| Profit giveback ratio | Lost profit after peak | < 40% |

### 7.5 Risk Model Stress Tests

| Scenario | Implementation |
|----------|----------------|
| Volatility spikes | 3x ATR injection |
| Spread explosions | 5x normal spread |
| Consecutive losses | Force 5+ losing trades |
| Partial fills | 50% fill rate |
| Gap opens | Weekend gap simulation |

---

## 8. Phase 6: MTF (Multi-Timeframe) Logic

### 8.1 MTF Profiles

| Profile | Primary TF | Higher TFs | Lower TFs |
|---------|------------|------------|-----------|
| SCALP | M5 | M15, H1 | - |
| INTRADAY | H1 | H4, D1 | M15 |
| SWING | H4 | D1, W1 | H1 |

### 8.2 MTF Trend Detection

**Module:** `trend_detection/mtf_trend_detector.py`

```python
class MTFTrendDetector:
    """
    Pipeline:
    1. Fetch MTF data
    2. Structural analysis per TF
    3. MTF confluence analysis
    4. Regime classification
    5. ML feature generation
    6. Signal generation
    """
```

### 8.3 MTF Alignment Calculation

```python
def calculate_mtf_alignment(direction: str, mtf_data: Dict) -> float:
    """
    Returns alignment score 0-1.
    - 1.0 = All TFs agree with direction
    - 0.5 = Mixed signals
    - 0.0 = All TFs disagree
    """
```

### 8.4 MTF Validation in Backtest

| Check | Description |
|-------|-------------|
| TF data availability | All required TFs have data |
| Timestamp alignment | Lower TF bars align with higher TF |
| Trend consistency | No contradictory signals within same bar |

---

## 9. Phase 7: Execution Simulation

### 9.1 Execution Engine: `BacktestConnector`

**Location:** `trading/backtest_connector.py`

#### Simulated Factors

| Factor | Implementation |
|--------|----------------|
| Market latency | Configurable delay (ms) |
| Slippage | Normal distribution around expected price |
| Spread | Base spread + random variation |
| Commission | Per-lot cost |
| Requotes | Probability-based rejection |
| Partial fills | Volume-based fill rate |

### 9.2 Order Lifecycle Validation

| Stage | Validation |
|-------|------------|
| Order creation | Correct sizing, valid SL/TP |
| Submission | Broker rules (min lot, stop distance) |
| Fill | Realistic price with slippage |
| Modification | Trailing SL updates |
| Closure | Correct PnL calculation |

### 9.3 Broker Constraints

```python
broker_constraints = {
    'min_lot_size': 0.01,
    'max_lot_size': 100.0,
    'min_stop_distance_pips': 5,
    'margin_requirement': 0.01,  # 1% margin
    'fifo_rules': True,
}
```

---

## 10. Phase 8: Metrics & Reporting

### 10.1 Portfolio-Level Metrics

| Category | Metrics |
|----------|---------|
| Return | Net P&L, CAGR, Total Return % |
| Risk | Max DD, Max DD Duration, CVaR, VaR |
| Efficiency | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| Stability | Rolling Sharpe, Rolling Win Rate |
| Execution | Slippage cost, Commission cost |
| Trading | Win Rate, Profit Factor, Avg Win/Loss |

### 10.2 ML-Specific Metrics

| Metric | Description |
|--------|-------------|
| Model accuracy | Direction prediction accuracy |
| Confidence calibration | Predicted vs actual win rate |
| Feature importance drift | Changes over time |
| Regime detection accuracy | Correct regime classification |

### 10.3 Decision Audit Trail

```python
decision_log = {
    'timestamp': datetime,
    'signal': 'BUY/SELL/HOLD',
    'confidence': float,
    'approved': bool,
    'rejection_reasons': list,
    'phase_scores': {
        'direction_confidence': float,
        'meta_score': float,
        'mtf_alignment': float,
        'protection_level': str,
    },
    'risk_params': {
        'sl': float,
        'tp': float,
        'position_size': float,
    }
}
```

---

## 11. Implementation Architecture

### 11.1 New Module: `backtesting/orchestrator.py`

```python
class BacktestOrchestrator:
    """
    Unified backtest runner that integrates all layers.
    
    Components:
    - DataLoader (MTF support)
    - BacktestConnector (MT5 interface)
    - NeuralHybridStrategy (production strategy)
    - EnhancedDecisionEngine (full pipeline)
    - RiskManager (Phases 1-5)
    - MetricsCollector
    - ReportGenerator
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.validate_weights()  # Check all model weights exist
        
    def validate_weights(self):
        """Verify all required model weights exist."""
        
    def run(self) -> BacktestResults:
        """Execute backtest with full pipeline."""
        
    def generate_report(self) -> str:
        """Generate comprehensive HTML/PDF report."""
```

### 11.2 File Structure

```
backtesting/
├── __init__.py           # ✅ Exists
├── engine.py             # ✅ Exists (event-driven)
├── execution_simulator.py # ✅ Exists
├── data_validator.py     # ✅ Exists
├── metrics.py            # ✅ Exists
├── reporter.py           # ✅ Exists
├── orchestrator.py       # 🆕 NEW - Main entry point
└── weight_validator.py   # 🆕 NEW - Model weight checks

trading/
├── backtest_connector.py # ✅ Exists (enhance MTF)
├── backtest_runner.py    # ✅ Exists (legacy, keep for compat)
├── data_loader.py        # ✅ Exists (enhance MTF)
└── bot.py                # ✅ Exists (step() method)
```

---

## 12. Walk-Forward & Regime Validation

### 12.1 Walk-Forward Setup

```python
walk_forward_config = {
    'train_window': 252 * 2,    # 2 years training
    'validate_window': 63,      # 3 months validation
    'trade_window': 21,         # 1 month trading
    'step_size': 21,            # Roll monthly
    'min_trades_per_window': 20,
}
```

### 12.2 Regime Segmentation

| Regime | Detection | Validation Target |
|--------|-----------|-------------------|
| Trend | ADX > 25, clear direction | Profitable |
| Range | ADX < 20, BB squeeze | Avoid or reduce size |
| High-volatility | ATR > 2x average | Reduced position size |
| News-driven | Scheduled events | No trading |

**Requirement:** Strategy must be profitable in at least 2 regimes.

---

## 13. Stress & Adversarial Scenarios

| Scenario | Implementation | Expected Behavior |
|----------|----------------|-------------------|
| Flash crash | -5% in 5 bars | SL triggered, limited loss |
| Weekend gap | 2% gap on open | Position closed at gap price |
| Liquidity drought | 10x spread | No new trades |
| API outage | No price updates | Graceful degradation |
| Model hallucination | Random outputs | Confidence filter blocks |

---

## 14. Automation & CI

### 14.1 Nightly Backtest Pipeline

```yaml
# .github/workflows/nightly_backtest.yml
schedule:
  - cron: '0 2 * * *'  # 2 AM daily

jobs:
  backtest:
    - Run full backtest on last 6 months
    - Compare to baseline metrics
    - Alert if degradation > 10%
    - Archive artifacts
```

### 14.2 Baseline Regression Detection

| Metric | Baseline | Alert Threshold |
|--------|----------|-----------------|
| Sharpe | 1.8 | < 1.5 |
| Max DD | 12% | > 15% |
| Win Rate | 58% | < 50% |
| Profit Factor | 1.6 | < 1.3 |

---

## 15. Final Acceptance Gate

| Requirement | Threshold | Status |
|-------------|-----------|--------|
| Max Drawdown | < 20% | ⬜ |
| Sharpe Ratio | > 1.5 | ⬜ |
| Walk-forward decay | < 25% | ⬜ |
| Risk violations | 0 | ⬜ |
| Execution failures | 0 | ⬜ |
| Model weight validation | All exist | ⬜ |
| Feature validation | No lookahead | ⬜ |

---

## 16. Production Readiness Checklist

- [ ] Identical pipeline (live = backtest)
- [ ] No lookahead bias
- [ ] Risk ML validated (all phases)
- [ ] Execution realistic (slippage, spread, commission)
- [ ] Failure isolation (errors don't crash)
- [ ] Audit-ready logs
- [ ] All model weights validated
- [ ] MTF data properly aligned
- [ ] Feature engineering causally correct
- [ ] Decision audit trail complete

---

## 17. Implementation Roadmap

| Step | Description | Priority | Effort |
|------|-------------|----------|--------|
| 1 | Create `backtesting/orchestrator.py` | HIGH | Medium |
| 2 | Create `backtesting/weight_validator.py` | HIGH | Low |
| 3 | Enhance `trading/data_loader.py` for MTF | HIGH | Low |
| 4 | Add decision audit logging | MEDIUM | Low |
| 5 | Implement walk-forward mode | MEDIUM | Medium |
| 6 | Create comprehensive report generator | MEDIUM | Medium |
| 7 | Add stress test scenarios | LOW | Medium |
| 8 | CI/CD integration | LOW | Low |

---

## Appendix A: Model Weight Paths

```python
MODEL_WEIGHTS = {
    # TCN by profile and timeframe
    'tcn': {
        'SCALP': {
            'M5': 'models/weights/scalp_m5_best.pt',
            'M15': 'models/weights/scalp_m15_best.pt',
            'H1': 'models/weights/scalp_h1_best.pt',
        },
        'INTRADAY': {
            'M15': 'models/weights/intraday_m15_best.pt',
            'H1': 'models/weights/intraday_h1_best.pt',
            'H4': 'models/weights/intraday_h4_best.pt',
        },
        'SWING': {
            'H1': 'models/weights/swing_h1_best.pt',
            'H4': 'models/weights/swing_h4_best.pt',
            'D1': 'models/weights/swing_d1_best.pt',
        },
    },
    # Generic models
    'tcn_generic': 'models/weights/tcn_best.pt',
    'tcn_enhanced': 'models/weights/tcn_enhanced_best.pt',
    'fusion': 'models/weights/fusion_best.pt',
    'vit': 'models/weights/vit_best.pt',
    'yolo': 'models/weights/yolo_patterns.pt',
    'meta_model': 'models/weights/meta_model.joblib',
    'exit_model': 'models/weights/exit_model.pt',
}
```

---

## Appendix B: Configuration Template

```python
from backtesting import BacktestConfig, BacktestMode

config = BacktestConfig(
    # Mode
    mode=BacktestMode.HISTORICAL_REPLAY,
    
    # Data
    data_path='data/EURUSD_H1.csv',
    symbol='EURUSD',
    primary_timeframe='H1',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    warmup_bars=200,
    
    # Execution
    initial_balance=10000.0,
    commission_per_lot=7.0,
    base_spread_pips=1.0,
    slippage_enabled=True,
    latency_ms=50,
    
    # Strategy
    profile='INTRADAY',
    strategy_class=NeuralHybridStrategy,
    
    # Risk
    max_positions=1,
    max_daily_trades=10,
    
    # ML
    freeze_model_weights=True,
    validate_weights=True,
    
    # Validation
    validate_data=True,
    validate_features=True,
    check_lookahead=True,
    
    # Output
    save_artifacts=True,
    artifacts_dir='backtest_artifacts',
    generate_report=True,
)
```

---

*Document generated for pyForex-1 backtesting framework implementation.*
