# alpha_factory/run_3tf.py

import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict

from .alpha_factory import AlphaFactory
from .three_tf_system import ThreeTFOrchestrator, FeatureAdapter, FeatureSnapshot
from .trading_profiles import get_profile, TradingProfile, TimeFrame

def load_data_for_profile(profile: TradingProfile) -> Dict[str, pd.DataFrame]:
    """
    Mock data loader. In production, this would query your DB/Parquet files
    for the specific timeframes defined in the profile.
    """
    print(f"Loading data for {profile.type.value}: {profile.ltf.value}, {profile.mtf.value}, {profile.htf.value}")
    
    # Mock data generation
    def make_df(freq):
        dates = pd.date_range(end=datetime.now(), periods=500, freq=freq.replace('m', 'min').replace('h', 'H').replace('d', 'D'))
        data = {
            'open': np.random.randn(500).cumsum() + 100,
            'high': np.random.randn(500).cumsum() + 101,
            'low': np.random.randn(500).cumsum() + 99,
            'close': np.random.randn(500).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 500)
        }
        return pd.DataFrame(data, index=dates)

    return {
        'htf': make_df(profile.htf.value),
        'mtf': make_df(profile.mtf.value),
        'ltf': make_df(profile.ltf.value)
    }

def run_profile_pipeline(profile_name: str, symbol: str = "EURUSD"):
    """
    Runs the 3TF logic for a specific trading profile.
    """
    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        print(e)
        return

    # 1. Load Data per Profile
    data_map = load_data_for_profile(profile)
    
    # 2. Initialize Feature Pools (Alpha Factories)
    # In a real system, these might be loaded from saved states
    print(f"--- Initializing Feature Pools for {symbol} ({profile_name}) ---")
    factory_htf = AlphaFactory()
    factory_mtf = AlphaFactory()
    factory_ltf = AlphaFactory()
    
    # 3. Process Data (Generate Raw Intelligence)
    print("Processing HTF...")
    strat_htf = factory_htf.process_data(data_map['htf'])
    
    print("Processing MTF...")
    strat_mtf = factory_mtf.process_data(data_map['mtf'])
    
    print("Processing LTF...")
    strat_ltf = factory_ltf.process_data(data_map['ltf'])
    
    # 4. Create Snapshots
    snapshot_htf = FeatureAdapter.create_snapshot(
        timestamp=datetime.now(),
        timeframe=profile.htf.value,
        decision_signal=strat_htf['decision'],
        causality_results=factory_htf.causality_results,
        market_regime=strat_htf['decision']['regime']
    )
    
    snapshot_mtf = FeatureAdapter.create_snapshot(
        timestamp=datetime.now(),
        timeframe=profile.mtf.value,
        decision_signal=strat_mtf['decision'],
        causality_results=factory_mtf.causality_results,
        market_regime=strat_mtf['decision']['regime']
    )
    
    snapshot_ltf = FeatureAdapter.create_snapshot(
        timestamp=datetime.now(),
        timeframe=profile.ltf.value,
        decision_signal=strat_ltf['decision'],
        causality_results=factory_ltf.causality_results,
        market_regime=strat_ltf['decision']['regime']
    )
    
    # 5. Run Orchestrator
    orchestrator = ThreeTFOrchestrator(symbol, profile)
    instruction = orchestrator.process_3tf(snapshot_htf, snapshot_mtf, snapshot_ltf)
    
    if instruction:
        print(f"\n✅ {profile_name} TRADE EXECUTED")
        print(f"Direction: {instruction.direction}")
        print(f"Confidence: {instruction.confidence:.2f}")
        print(f"Logic Path: {instruction.logic_path}")
    else:
        print(f"\n❌ {profile_name} NO TRADE (Constraints Active)")

if __name__ == "__main__":
    # Demonstrate all 3 profiles
    print("=========================================")
    run_profile_pipeline("SCALPING")
    print("\n=========================================")
    run_profile_pipeline("INTRADAY")
    print("\n=========================================")
    run_profile_pipeline("SWING")