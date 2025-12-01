#!/usr/bin/env python3
# main.py
"""
PyForex Trading Bot - Main Entry Point

Usage:
    python main.py              # Run live trading
    python main.py --mock       # Run with mock connector (testing)
    python main.py --backtest   # Run backtest
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)


def run_live(args):
    """Run live trading bot."""
    from trading.bot import TradingBot, BotConfig
    
    config = BotConfig(
        use_mock=args.mock,
        log_level=args.log_level,
    )
    
    print(f"🚀 Starting PyForex Bot {'(MOCK MODE)' if args.mock else '(LIVE)'}")
    
    bot = TradingBot(config=config)
    bot.run()


def run_backtest(args):
    """Run backtest."""
    import pandas as pd
    from trading.bot import BacktestBot
    from strategies.neural_hybrid import NeuralHybridStrategy
    
    print("📊 Running Backtest...")
    
    # Load historical data
    data_path = args.data or "data/raw/eurusd_latest.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Generate data with: python training/auto_retrain.py")
        return
    
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    
    if 'time' not in df.columns:
        df['time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='H')
    else:
        df['time'] = pd.to_datetime(df['time'])
    
    print(f"Loaded {len(df)} bars from {data_path}")
    
    # Run backtest
    bot = BacktestBot(
        data=df,
        strategy_class=NeuralHybridStrategy,
        initial_balance=args.balance,
    )
    
    results = bot.run()
    
    # Print results
    metrics = bot.executor.get_performance_metrics()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Win Rate:         {metrics['win_rate']:.1%}")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Total P&L:        ${metrics['total_pnl']:.2f}")
    print(f"Final Balance:    ${metrics['final_balance']:.2f}")
    print(f"Return:           {metrics['return_pct']:.2f}%")
    print("="*50)
    
    # Optionally save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'trades': results['trades'],
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyForex Neural Hybrid Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock connector instead of real MT5',
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest instead of live trading',
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to historical data CSV for backtest',
    )
    
    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Initial balance for backtest (default: 10000)',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for backtest results JSON',
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='trading_session.log',
        help='Log file path (default: trading_session.log)',
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        if args.backtest:
            run_backtest(args)
        else:
            run_live(args)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        
    except Exception as e:
        logging.critical(f"🔥 Fatal error: {e}", exc_info=True)
        sys.exit(1)


# Need this import for type hints
from typing import Optional

if __name__ == "__main__":
    main()
