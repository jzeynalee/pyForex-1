#!/usr/bin/env python3
"""
Example: Running pyForex for Prop Firm Challenge

This script demonstrates how to set up and run pyForex
for a prop firm evaluation challenge.

Prerequisites:
1. Trained TCN model (tcn_best.pt)
2. Trained meta-labeling model (meta_model.joblib) [optional]
3. Data provider and executor configured
4. Notification clients set up [optional]
"""

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_dry_run():
    """
    Example: Dry run with FTMO settings.
    
    This is safe to run - no real trades executed.
    """
    from trading import create_prop_firm_bot
    
    # Mock data provider and executor for testing
    class MockDataProvider:
        def get_ohlcv(self, symbol, timeframe='H1', count=100):
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(end=datetime.utcnow(), periods=count, freq='H')
            close = 1.1000 + np.cumsum(np.random.randn(count) * 0.0005)
            
            return pd.DataFrame({
                'open': close - np.random.rand(count) * 0.0003,
                'high': close + np.random.rand(count) * 0.0005,
                'low': close - np.random.rand(count) * 0.0005,
                'close': close,
                'volume': np.random.randint(1000, 10000, count)
            }, index=dates)
        
        def get_price(self, symbol):
            return 1.1000
        
        def get_spread(self, symbol):
            return 1.0
    
    class MockExecutor:
        def get_account_balance(self):
            return 100000
        
        def execute_order(self, **kwargs):
            logger.info(f"[MOCK] Would execute: {kwargs}")
            return {'success': True, 'ticket': 'MOCK-001', 'price': 1.1000}
        
        def close_position(self, ticket):
            return {'success': True, 'pnl': 50.0}
    
    # Create bot
    bot = create_prop_firm_bot(
        firm='FTMO',
        account_size=100000,
        phase='evaluation',
        symbol='EURUSD',
        data_provider=MockDataProvider(),
        executor=MockExecutor(),
        conservative=True,
        dry_run=True  # Paper trading!
    )
    
    # Get status
    status = bot.get_challenge_status()
    print("\n" + "="*50)
    print("PROP FIRM BOT STATUS")
    print("="*50)
    print(status['summary'])
    print("="*50 + "\n")
    
    # Note: In a real scenario, you would call bot.run()
    # bot.run()  # Blocking
    
    logger.info("Dry run example completed")


def example_with_notifications():
    """
    Example: Full setup with signal publishing.
    
    Shows how to integrate with notification systems.
    """
    from trading import create_prop_firm_bot
    from signals import SignalPublisher, PublisherConfig
    
    # Your notification clients (implement these based on your setup)
    # telegram_client = YourTelegramBot(token='...')
    # twitter_client = YourTwitterAPI(api_key='...')
    
    # For this example, we'll use mock clients
    class MockNotifier:
        def __init__(self, name):
            self.name = name
        
        def send_message(self, message):
            print(f"\n[{self.name}] {message}\n")
    
    telegram = MockNotifier("TELEGRAM")
    twitter = MockNotifier("TWITTER")
    
    # Create publisher
    publisher = SignalPublisher(PublisherConfig(
        telegram_enabled=True,
        twitter_enabled=True,
        linkedin_enabled=False,
        publish_entries=True,
        publish_exits=True,
        publish_daily_summary=True,
        min_confidence_to_publish=0.6,
    ))
    
    publisher.set_notifiers(telegram=telegram, twitter=twitter)
    
    # Test signal publishing
    signal = publisher.publish_entry(
        symbol='EURUSD',
        direction='BUY',
        entry_price=1.1000,
        stop_loss=1.0960,
        take_profit=1.1080,
        confidence=0.75,
        risk_percent=0.5,
        position_size=0.1
    )
    
    if signal:
        print(f"Signal published: {signal.signal_id}")
    
    # Simulate exit
    publisher.publish_exit(
        symbol='EURUSD',
        entry_price=1.1000,
        exit_price=1.1065,
        pnl=650.0,
        pnl_pips=65,
        direction='BUY'
    )
    
    # Daily summary
    publisher.publish_daily_summary()


def example_prop_firm_comparison():
    """
    Compare different prop firm configurations.
    """
    from config import get_prop_firm_config
    
    firms = ['FTMO', 'MyForexFunds', 'The5ers', 'Funded Next']
    account_size = 100000
    
    print("\n" + "="*80)
    print("PROP FIRM COMPARISON (Evaluation Phase, $100K Account)")
    print("="*80)
    print(f"{'Firm':<15} {'Daily Loss':<12} {'Max DD':<10} {'Target':<10} {'Profit Split':<12}")
    print("-"*80)
    
    for firm in firms:
        try:
            config = get_prop_firm_config(firm, account_size, 'evaluation')
            print(
                f"{firm:<15} "
                f"{config.rules.max_daily_loss_pct:.1f}%{'':<7} "
                f"{config.rules.max_total_drawdown_pct:.1f}%{'':<5} "
                f"{config.rules.profit_target_pct:.1f}%{'':<5} "
                f"{config.rules.profit_split_pct:.0f}%"
            )
        except Exception as e:
            print(f"{firm:<15} Error: {e}")
    
    print("="*80)
    print("\nWith Conservative Mode (Safety Buffers):")
    print("-"*80)
    print(f"{'Firm':<15} {'Eff. Daily':<12} {'Eff. DD':<10} {'Risk/Trade':<12}")
    print("-"*80)
    
    for firm in firms:
        try:
            config = get_prop_firm_config(firm, account_size, 'evaluation', conservative=True)
            print(
                f"{firm:<15} "
                f"{config.effective_daily_loss_pct:.1f}%{'':<7} "
                f"{config.effective_max_drawdown_pct:.1f}%{'':<5} "
                f"{config.max_risk_per_trade_pct:.1f}%"
            )
        except Exception as e:
            print(f"{firm:<15} Error: {e}")
    
    print("="*80 + "\n")


def example_capital_protection_integration():
    """
    Show how capital protection works with prop firm limits.
    """
    from config import get_prop_firm_config, PropFirmMonitor
    from risk_management import CapitalProtector, ProtectionConfig
    
    # Get FTMO config
    prop_config = get_prop_firm_config('FTMO', 100000, 'evaluation', conservative=True)
    
    # Create capital protector with prop firm limits
    protection_config = ProtectionConfig(**prop_config.to_protection_config())
    protector = CapitalProtector(protection_config)
    protector.initialize(100000)
    
    # Create prop firm monitor
    monitor = PropFirmMonitor(prop_config)
    monitor.initialize(100000)
    
    print("\n" + "="*60)
    print("CAPITAL PROTECTION INTEGRATION")
    print("="*60)
    
    # Simulate some trades
    trades = [
        (-150, False),  # Loss
        (200, True),    # Win
        (-180, False),  # Loss
        (250, True),    # Win
        (-200, False),  # Loss
        (-220, False),  # Loss - approaching limits
    ]
    
    balance = 100000
    daily_pnl = 0
    
    for pnl, is_win in trades:
        balance += pnl
        daily_pnl += pnl
        
        # Update both systems
        protector.record_trade(pnl, is_win)
        monitor.update(daily_pnl, balance - 100000, balance)
        
        # Check status
        prot_state = protector.get_state()
        can_trade, reason = monitor.can_trade()
        
        result = "✅ WIN" if is_win else "❌ LOSS"
        print(f"{result} ${pnl:+.0f} | Balance: ${balance:,.0f} | "
              f"Protection: {prot_state.level.value} | Can Trade: {can_trade}")
        
        if not can_trade:
            print(f"   ⚠️  {reason}")
            break
    
    print("\n" + monitor.status.get_status_message())
    print("="*60 + "\n")


def main():
    """Run all examples."""
    print("\n" + "🚀 pyForex Prop Firm Configuration Examples 🚀".center(60))
    
    # Run examples
    example_prop_firm_comparison()
    example_capital_protection_integration()
    example_with_notifications()
    example_dry_run()


if __name__ == "__main__":
    main()
