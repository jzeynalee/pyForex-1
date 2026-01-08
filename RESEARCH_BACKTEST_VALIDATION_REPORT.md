# Research Backtest Report: VALIDATION

## Executive Summary

This report presents results for the VALIDATION risk profile using post-2010 compatible data.

**Risk Profile**: VALIDATION
**Data Points**: 10,000
**Total Trades**: 3,873

## Risk Profile Configuration

VALIDATION PROFILE SUMMARY:
- EV Threshold: $2.00
- Position Multiplier: 0.7x
- Safeguard Severity: 0.5x
- Max Concurrent Trades: 10
- Drawdown Limits: 20.0% / 35.0%
- Catastrophic Block: Enabled
- Emergency Stop: Disabled

## Performance Summary

| Metric | Value |
|---------|-------|
| Win Rate | 48.23% |
| Total P&L | $19,056.98 |
| Average P&L per Trade | $4.92 |
| Average Win | $104.87 |
| Average Loss | $-88.20 |
| Profit Factor | 1.19 |
| Sharpe Ratio | 0.00 |
| Maximum Drawdown | 252.39% |

## EV Distribution Analysis (MANDATORY METRICS)

| Statistic | Value |
|-----------|-------|
| Mean EV | $26.52 |
| Median EV | $26.10 |
| Std EV | $7.68 |
| 25th Percentile | $20.52 |
| 75th Percentile | $32.94 |
| Min EV | $13.53 |
| Max EV | $38.70 |

## Regime Performance Analysis

### Bullish Regime
- **Trades**: 1117
- **P&L**: $17,566.63
- **Win Rate**: 53.36%
- **Average P&L**: $15.73

### Bearish Regime
- **Trades**: 507
- **P&L**: $516.55
- **Win Rate**: 45.36%
- **Average P&L**: $1.02

### Neutral Regime
- **Trades**: 1488
- **P&L**: $2,403.45
- **Win Rate**: 47.24%
- **Average P&L**: $1.62

### Volatile Regime
- **Trades**: 761
- **P&L**: $-1,429.66
- **Win Rate**: 44.55%
- **Average P&L**: $-1.88

## Safeguards Analysis

- **Current State**: normal
- **Risk Multiplier**: 1.00
- **Position Size Multiplier**: 1.00

### Active Triggers

## Acceptance Criteria Check

**Hard Requirements:**
- Total trades >= 1,000: PASS
- Win rate meaningful: PASS
- EV distribution positive: PASS

## Conclusions

The VALIDATION profile demonstrates moderate performance.

**Key Insights:**
- Trade frequency: 3873 trades in 10000 bars
- Win rate: 48.23%
- EV distribution: Mean $26.52, Median $26.10

---
*Report generated on 2026-01-06 08:53:11*
*Research Backtest System*
