import pytest

from risk_management.phase2_risk_calc.sl_tp_calculator import SLTPCalculator, SLTPConfig
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig


@pytest.fixture
def sltp_calculator():
    return SLTPCalculator(SLTPConfig(min_risk_reward=1.5))


@pytest.fixture
def decision_engine():
    config = DecisionEngineConfig(
        profile="INTRADAY",
        min_direction_confidence=0.55,
        min_risk_reward=1.5,
        enable_capital_protection=True,
        max_drawdown_pct=10.0,
        max_daily_loss_pct=3.0,
        max_weekly_loss_pct=6.0,
    )
    engine = EnhancedDecisionEngine(config=config, meta_model=None)
    engine.initialize(starting_balance=10000.0)
    return engine
