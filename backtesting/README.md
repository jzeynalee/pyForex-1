# Comprehensive Backtesting System

## Overview

This is a production-grade backtesting framework designed for the pyForex trading system. It implements **pipeline-faithful, ML-aware, execution-realistic** backtesting that gives confidence for live trading.

## Core Principles

1. **Zero code divergence** between live and backtest paths
2. **Event-driven replay**, not candle loops
3. **ML weights frozen** unless explicitly testing retraining
4. **Execution realism > optimistic fills**
5. **Risk system tested as a first-class ML component**

## Architecture

### Components

```
backtesting/
├── engine.py                    # Event-driven backtesting engine
├── data_validator.py            # Data integrity validation
├── execution_simulator.py       # Realistic execution simulation
├── metrics.py                   # Performance metrics calculation
├── reporter.py                  # Report generation
├── feature_validator.py         # Feature engineering validation
├── model_tracker.py             # ML model execution tracking
├── walk_forward.py              # Walk-forward validation
└── stress_tester.py             # Stress testing scenarios
```

## Features

### 1. Event-Driven Engine

The core engine processes market events in chronological order, maintaining proper causality:

```python
from backtesting import BacktestEngine, BacktestConfig, BacktestMode

config = BacktestConfig(
    mode=BacktestMode.HISTORICAL_REPLAY,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_balance=10000
)

engine = BacktestEngine(config)
engine.set_strategy(my_strategy)
engine.set_data_provider(data_provider)
engine.set_execution_simulator(execution_sim)

results = engine.run()
```

### 2. Data Validation

Comprehensive data validation prevents common backtesting errors:

- **Timestamp monotonicity** - No backward time jumps
- **OHLC relationships** - Valid price data
- **Gap detection** - Weekend/holiday gaps
- **Lookahead leak detection** - Prevents future data usage
- **Completeness checks** - Missing data detection

```python
from backtesting import DataValidator

validator = DataValidator(
    check_monotonicity=True,
    check_gaps=True,
    check_lookahead=True,
    check_prices=True
)

result = validator.validate(df)
if not result.is_valid:
    print(result.summary())
```

### 3. Realistic Execution Simulation

Simulates real-world execution conditions:

- **Slippage models**: Fixed, Normal, Volatility-based, Realistic
- **Latency simulation**: Time-of-day dependent
- **Requotes and rejections**: Probabilistic failures
- **Partial fills**: For large orders
- **Spread widening**: During news/low liquidity
- **Market impact**: Size-dependent slippage
- **Broker constraints**: Min/max lot sizes, stop distances

```python
from backtesting import RealisticExecutionSimulator, ExecutionConfig, SlippageModel

config = ExecutionConfig(
    slippage_model=SlippageModel.REALISTIC,
    slippage_mean_pips=0.3,
    latency_model=LatencyModel.REALISTIC,
    requote_probability=0.02,
    enable_market_impact=True,
    enable_spread_widening=True
)

simulator = RealisticExecutionSimulator(config)
```

### 4. Comprehensive Metrics

Calculates 50+ performance metrics:

**Trade Metrics**
- Total trades, win rate, profit factor
- Average win/loss, largest win/loss
- Long/short breakdown
- Trade duration statistics

**Risk Metrics**
- Maximum drawdown ($ and %)
- Sharpe, Sortino, Calmar ratios
- Value at Risk (VaR), Conditional VaR
- Consecutive wins/losses
- Recovery factor, Ulcer index

**Return Metrics**
- Total return, CAGR
- Daily/monthly return statistics
- Best/worst day and month

**Execution Metrics**
- Total commission and slippage
- Fill rate, rejection rate
- Average spread and latency

```python
from backtesting import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate(
    trades=trade_history,
    equity_curve=equity_curve,
    initial_balance=10000
)

print(f"Sharpe Ratio: {metrics.risk_metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.risk_metrics.max_drawdown_pct:.2f}%")
```

### 5. Acceptance Gates

Validates strategy readiness for production:

```python
from backtesting import AcceptanceGate

gate = AcceptanceGate(
    min_sharpe_ratio=1.5,
    max_drawdown_pct=20.0,
    min_profit_factor=1.5,
    min_win_rate=0.45,
    max_walk_forward_decay_pct=25.0,
    max_risk_violations=0,
    max_execution_failures=0,
    min_trades=30
)

passed, failures = gate.validate(metrics)
if passed:
    print("✅ Ready for production!")
else:
    print("❌ Failed acceptance criteria:")
    for failure in failures:
        print(f"  - {failure}")
```

### 6. Comprehensive Reporting

Generates detailed HTML, JSON, and CSV reports with:

- Performance summary dashboard
- Equity curve and drawdown charts
- Trade distribution analysis
- Monthly returns heatmap
- Risk metrics breakdown
- Execution quality statistics

```python
from backtesting import BacktestReporter, ReportConfig

config = ReportConfig(
    output_dir="backtest_reports",
    generate_plots=True,
    generate_html=True,
    generate_json=True,
    generate_csv=True
)

reporter = BacktestReporter(config)
reporter.generate_report(results, metrics, acceptance_gate)
```

## Execution Modes

### 1. Historical Replay (Deterministic)

Standard backtesting mode for strategy evaluation:

```python
config = BacktestConfig(mode=BacktestMode.HISTORICAL_REPLAY)
```

### 2. Walk-Forward (Generalization Testing)

Tests strategy robustness across different time periods:

```python
config = BacktestConfig(mode=BacktestMode.WALK_FORWARD)
```

### 3. Paper Replay (Live-like Timing)

Simulates real-time execution with actual timing:

```python
config = BacktestConfig(mode=BacktestMode.PAPER_REPLAY)
```

### 4. Stress Simulation (Adversarial Markets)

Tests strategy under extreme conditions:

```python
config = BacktestConfig(mode=BacktestMode.STRESS_SIMULATION)
```

## Integration with pyForex

### With Decision Engine

```python
from trading.decision_engine import EnhancedDecisionEngine
from backtesting import BacktestEngine

decision_engine = EnhancedDecisionEngine(config)
decision_engine.initialize(10000)

engine = BacktestEngine(backtest_config)
engine.set_decision_engine(decision_engine)
engine.set_risk_manager(risk_manager)
```

### With ML Models

```python
# Models are automatically frozen during backtest
config = BacktestConfig(
    freeze_model_weights=True,
    enable_model_tracking=True
)

# Model predictions are logged for analysis
engine.set_model_tracker(model_tracker)
```

### With Risk Management

```python
from risk_management import RiskManager

risk_manager = RiskManager.create_for_profile('INTRADAY')
engine.set_risk_manager(risk_manager)

# Risk decisions are validated and tracked
```

## Best Practices

### 1. Data Preparation

- Always validate data before backtesting
- Check for lookahead bias in features
- Ensure sufficient warmup period
- Handle missing data appropriately

### 2. Realistic Execution

- Use realistic slippage models
- Enable latency simulation
- Account for spread widening
- Consider market impact for large orders

### 3. Risk Management

- Test risk system as first-class component
- Validate SL/TP placement
- Check position sizing logic
- Monitor drawdown control

### 4. Validation

- Use walk-forward validation
- Test across different market regimes
- Run stress tests
- Validate against acceptance gates

### 5. Reporting

- Generate comprehensive reports
- Analyze trade distribution
- Review execution quality
- Document all assumptions

## Example Usage

See `examples/comprehensive_backtest_example.py` for a complete example:

```bash
python examples/comprehensive_backtest_example.py
```

This will:
1. Generate sample data
2. Validate data integrity
3. Run comprehensive backtest
4. Calculate all metrics
5. Generate HTML report with charts
6. Validate against acceptance gates

## Output Files

After running a backtest, you'll find:

```
backtest_reports/
├── backtest_report_20231220_153045.html    # Interactive HTML report
├── backtest_report_20231220_153045.json    # Full results JSON
├── backtest_report_20231220_153045_trades.csv
├── backtest_report_20231220_153045_decisions.csv
├── backtest_report_20231220_153045_equity_curve.png
├── backtest_report_20231220_153045_drawdown.png
├── backtest_report_20231220_153045_trade_distribution.png
└── backtest_report_20231220_153045_monthly_returns.png
```

## Performance Considerations

- **Large datasets**: Use batch processing (automatically enabled)
- **Multiple runs**: Parallel processing available
- **Memory usage**: Artifacts saved incrementally
- **Speed**: Event-driven architecture is efficient

## Acceptance Criteria (Production Readiness)

| Requirement | Threshold |
|------------|-----------|
| Max Drawdown | < 20% |
| Sharpe Ratio | > 1.5 |
| Profit Factor | > 1.5 |
| Win Rate | > 45% |
| Walk-forward Decay | < 25% |
| Risk Violations | 0 |
| Execution Failures | 0 |
| Minimum Trades | 30 |

## Troubleshooting

### Data Validation Errors

If you get data validation errors:
1. Check timestamp ordering
2. Verify OHLC relationships
3. Look for missing data
4. Check for lookahead bias in features

### Execution Issues

If execution seems unrealistic:
1. Adjust slippage model parameters
2. Check spread settings
3. Verify latency configuration
4. Review broker constraints

### Poor Performance

If backtest performance is poor:
1. Check strategy logic
2. Verify risk parameters
3. Analyze trade distribution
4. Review execution costs

## Advanced Features

### Custom Validators

Create custom validators for specific checks:

```python
class CustomValidator:
    def validate(self, df):
        # Your validation logic
        return ValidationResult(...)

engine.set_validators(custom_validator=my_validator)
```

### Custom Metrics

Add custom performance metrics:

```python
class CustomMetrics:
    def calculate(self, trades):
        # Your metrics calculation
        return custom_metrics

calculator.add_custom_metrics(CustomMetrics())
```

### Event Hooks

Hook into backtest events:

```python
def on_trade_close(trade):
    # Custom logic on trade close
    pass

engine.register_hook('trade_close', on_trade_close)
```

## Contributing

When adding new features:
1. Maintain zero code divergence principle
2. Add comprehensive tests
3. Update documentation
4. Validate against acceptance gates

## License

Part of the pyForex trading system.

## Support

For issues or questions:
1. Check this documentation
2. Review example code
3. Check validation errors
4. Review generated reports
