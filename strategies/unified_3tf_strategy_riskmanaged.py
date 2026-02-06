import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from strategies.unified_3tf_strategy import Unified3TFStrategy, Unified3TFConfig
from utils.config import settings


class RiskManagedUnified3TFStrategy(Unified3TFStrategy):
    def __init__(
        self,
        config: Optional[Unified3TFConfig] = None,
        data_provider=None,
        executor=None,
        risk_percent: Optional[float] = None,
        cooldown_minutes: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(config=config, data_provider=data_provider, executor=executor, **kwargs)
        
        p = self.config.profile.upper()
        
        if p == 'SCALP':
            cd_def = settings.SCALP_COOLDOWN
            risk_def = settings.SCALP_BASE_RISK
        elif p == 'SWING':
            cd_def = settings.SWING_COOLDOWN
            risk_def = settings.SWING_BASE_RISK
        else:
            cd_def = settings.INTRADAY_COOLDOWN
            risk_def = settings.INTRADAY_BASE_RISK

        self._cooldown_minutes = cooldown_minutes if cooldown_minutes is not None else cd_def
        self._risk_percent_override = risk_percent
        self._last_exit_time: Optional[datetime] = None

        if self._risk_percent_override is not None:
            self.config.base_risk_percent = float(self._risk_percent_override)
        else:
            self.config.base_risk_percent = risk_def

    def _infer_last_exit_time(self):
        try:
            th = getattr(self.executor, "trade_history", None)
            if not th:
                return
            last = th[-1]
            exit_time = None
            if isinstance(last, dict):
                exit_time = last.get("exit_time")
            else:
                exit_time = getattr(last, "exit_time", None)
            if exit_time is None:
                return
            try:
                et = pd.to_datetime(exit_time).to_pydatetime()
            except Exception:
                et = exit_time
            if self._last_exit_time is None:
                self._last_exit_time = et
                return
            if isinstance(et, datetime) and isinstance(self._last_exit_time, datetime) and et > self._last_exit_time:
                self._last_exit_time = et
        except Exception:
            return

    def _cooldown_ok(self, now: Optional[datetime]) -> bool:
        try:
            cd = float(self._cooldown_minutes or 0.0)
        except Exception:
            cd = 0.0
        if cd <= 0 or now is None:
            return True

        last = None
        try:
            last = getattr(self, "_last_entry_time", None)
        except Exception:
            last = None
        if self._last_exit_time is not None:
            if last is None:
                last = self._last_exit_time
            elif isinstance(last, datetime) and isinstance(self._last_exit_time, datetime) and self._last_exit_time > last:
                last = self._last_exit_time

        if last is None:
            return True

        try:
            mins = (now - last).total_seconds() / 60.0
        except Exception:
            return True

        if mins < cd:
            self.last_rejection_stage = "COOLDOWN"
            self.last_rejection_reason = f"cooldown {mins:.1f}m<{cd:.1f}m"
            return False
        return True

    def on_bar(self, df: pd.DataFrame):
        self._infer_last_exit_time()

        now = None
        if df is not None and not df.empty and "time" in df.columns:
            try:
                now = pd.to_datetime(df["time"].iloc[-1]).to_pydatetime()
            except Exception:
                now = None

        if not self._cooldown_ok(now):
            return None

        return super().on_bar(df)

    @staticmethod
    def _pip_value_for_pair(pair: str) -> float:
        return 0.01 if "JPY" in str(pair).upper() else 0.0001

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        try:
            d0 = df.copy()
            d0.columns = [str(c).lower().strip() for c in d0.columns]
            high = d0["high"]
            low = d0["low"]
            close = d0["close"]
            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr = float(tr.rolling(int(period), min_periods=1).mean().iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                return None
            return atr
        except Exception:
            return None

    def _execute_trade_probabilistic(self, direction: str, decision_out, df: pd.DataFrame):
        if self.executor is None:
            return

        entry_price = float(df["close"].iloc[-1])

        balance = 10000.0
        try:
            if hasattr(self.executor, "balance"):
                balance = float(self.executor.balance)
            elif hasattr(self.executor, "get_account_balance"):
                balance = float(self.executor.get_account_balance())
        except Exception:
            balance = 10000.0

        sl, tp = None, None
        volume = None

        try:
            from risk_management.phase2_risk_calc.sl_tp_calculator import (
                SLTPConfig,
                calculate_sl_tp_from_predictions,
                MarketRegime
            )
            from risk_management.phase2_risk_calc.position_sizing import (
                PositionSizingCalculator,
                PositionSizingConfig,
            )

            # Use the pre-fetched prediction if available from the decision engine
            pred = getattr(decision_out, 'mhtcn_prediction', None)
            
            # Map Decision engine regime to Risk Management regime
            regime_map = {
                'bull': MarketRegime.TRENDING_WEAK,
                'bear': MarketRegime.TRENDING_WEAK,
                'neutral': MarketRegime.RANGING,
                'volatile': MarketRegime.VOLATILE
            }
            
            engine_regime = 'neutral'
            try:
                if decision_out and hasattr(decision_out, 'regime_probs'):
                    engine_regime = decision_out.regime_probs.dominant_regime
            except Exception:
                pass
            
            target_regime = regime_map.get(engine_regime, MarketRegime.RANGING)
            
            # Refine regime if trending
            if engine_regime in ['bull', 'bear']:
                try:
                    ds = float(getattr(decision_out, 'final_p_bull', 0.5) - getattr(decision_out, 'final_p_bear', 0.5))
                    if abs(ds) > 0.4:
                        target_regime = MarketRegime.TRENDING_STRONG
                except Exception:
                    pass

            predictions = None
            if pred is not None:
                predictions = {
                    "quantiles": getattr(pred, "quantiles", None),
                    "volatility": getattr(pred, "volatility", None),
                    "direction_probs": getattr(pred, "direction_probs", None),
                }

            atr = self._atr(df)

            if predictions is not None:
                # Map strategy constants to SLTPConfig
                sltp_cfg = SLTPConfig(
                    min_risk_reward=float(getattr(self.config, "min_risk_reward", 1.5) or 1.5),
                    min_sl_atr_multiple=float(getattr(self.config, "atr_sl_mult", 2.0) or 2.0),
                    target_risk_reward=float(getattr(self.config, "min_risk_reward", 1.5) or 1.5) + 0.5
                )
                
                sltp = calculate_sl_tp_from_predictions(
                    entry_price=float(entry_price),
                    direction=str(direction).upper(),
                    predictions=predictions,
                    regime=target_regime.value if hasattr(target_regime, 'value') else str(target_regime),
                    atr=atr,
                    config=sltp_cfg,
                )
                sl = float(sltp.stop_loss)
                tp = float(sltp.take_profit)
            else:
                sl, tp = self._atr_sltp(df, entry_price, direction)

            try:
                min_sl_pips = float(getattr(self.config, "min_sl_pips", 8.0) or 8.0)
            except Exception:
                min_sl_pips = 8.0

            pip_value = self._pip_value_for_pair(getattr(self.config, "symbol", "EURUSD"))
            sl_pips = abs(entry_price - float(sl)) / pip_value

            if sl_pips < min_sl_pips:
                if str(direction).upper() == "BUY":
                    sl = float(entry_price) - (min_sl_pips * pip_value)
                    tp = float(entry_price) + (abs(entry_price - float(sl)) * float(getattr(self.config, "min_risk_reward", 2.0) or 2.0))
                else:
                    sl = float(entry_price) + (min_sl_pips * pip_value)
                    tp = float(entry_price) - (abs(float(sl) - entry_price) * float(getattr(self.config, "min_risk_reward", 2.0) or 2.0))

            try:
                conf = float(getattr(decision_out, "confidence", None))
            except Exception:
                conf = None

            vol = None
            try:
                vol = predictions.get("volatility") if predictions else None
                if hasattr(vol, "item"):
                    vol = vol.item()
                if vol is not None:
                    vol = float(vol)
            except Exception:
                vol = None

            ps_cfg = PositionSizingConfig(
                base_risk_percent=float(getattr(self.config, "base_risk_percent", 0.25) or 0.25),
                max_risk_percent=float(getattr(self.config, "max_daily_loss_pct", 1.5) or 1.5),
                lot_size_precision=2
            )
            ps_calc = PositionSizingCalculator(ps_cfg)
            ps = ps_calc.calculate(
                account_balance=float(balance),
                entry_price=float(entry_price),
                stop_loss=float(sl),
                pair=str(getattr(self.config, "symbol", "EURUSD") or "EURUSD").upper(),
                direction_confidence=conf,
                volatility=vol,
            )

            size_mult = float(getattr(decision_out, "size_multiplier", 1.0) or 1.0)
            volume = float(getattr(ps, "position_size", 0.0) or 0.0) * float(size_mult)
        except Exception:
            sl, tp = self._atr_sltp(df, entry_price, direction)
            size_mult = float(getattr(decision_out, "size_multiplier", 1.0) or 1.0)
            volume = self._calculate_position_size(entry_price, sl, size_mult)

        try:
            max_lot = float(getattr(self.config, "max_lot", 1.0) or 1.0)
        except Exception:
            max_lot = 1.0

        try:
            volume = float(volume or 0.0)
        except Exception:
            volume = 0.01

        volume = max(0.01, min(volume, max_lot))
        volume = round(volume, 2)

        if hasattr(self.executor, "entry"):
            self.executor.entry(signal=direction, volume=volume, sl=float(sl), tp=float(tp))
        elif hasattr(self.executor, "open_position"):
            self.executor.open_position(direction=direction, volume=volume, stop_loss=float(sl), take_profit=float(tp))
