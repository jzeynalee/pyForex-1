# trading/risk_manager.py
"""
Risk Manager: Dynamic SL/TP Calculation using ML Predictions.

Uses outputs from TCNRiskModel to calculate:
- Phase 1: Volatility-based SL/TP
- Phase 2: Quantile-based SL/TP (asymmetric risk)

The Risk Manager translates model predictions into actionable
stop-loss and take-profit levels.

Usage:
    from trading.risk_manager import RiskManager
    
    risk_mgr = RiskManager.from_checkpoint("models/weights/tcn_risk_best.pt")
    
    levels = risk_mgr.calculate_levels(
        features=X,
        entry_price=1.1050,
        direction='BUY',
    )
    
    print(f"SL: {levels['stop_loss']}, TP: {levels['take_profit']}")
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Literal, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RiskMethod(Enum):
    """Risk calculation method."""
    VOLATILITY = "volatility"       # Phase 1: ATR-like multiplier
    QUANTILE = "quantile"           # Phase 2: Distribution-based
    HYBRID = "hybrid"               # Blend of both
    FIXED_RR = "fixed_rr"           # Fixed risk-reward with vol-based SL


@dataclass
class RiskConfig:
    """Configuration for risk calculations."""
    # Method
    method: RiskMethod = RiskMethod.HYBRID
    
    # Volatility-based parameters (Phase 1)
    sl_volatility_multiplier: float = 1.5   # SL = entry ± vol * multiplier
    tp_volatility_multiplier: float = 2.5   # TP = entry ± vol * multiplier
    
    # Quantile-based parameters (Phase 2)
    sl_quantile: float = 0.10     # Use Q10 for SL (10% worst case)
    tp_quantile: float = 0.75     # Use Q75 for TP (75th percentile)
    
    # Risk-reward constraints
    min_risk_reward: float = 1.5  # Minimum acceptable RR ratio
    max_risk_reward: float = 5.0  # Maximum RR (avoid unrealistic TPs)
    
    # Position sizing
    risk_per_trade: float = 0.01  # 1% risk per trade
    
    # Safety limits
    min_sl_pips: float = 5.0      # Minimum SL in pips
    max_sl_pips: float = 100.0    # Maximum SL in pips
    min_tp_pips: float = 10.0     # Minimum TP in pips
    max_tp_pips: float = 500.0    # Maximum TP in pips
    
    # Instrument info
    pip_value: float = 0.0001     # For EURUSD


@dataclass
class RiskLevels:
    """Calculated risk levels."""
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    sl_pips: float
    tp_pips: float
    predicted_volatility: float
    confidence: float
    method_used: str
    
    # Optional quantile info
    quantile_sl: Optional[float] = None
    quantile_tp: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward_ratio,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'volatility': self.predicted_volatility,
            'confidence': self.confidence,
            'method': self.method_used,
        }


# =============================================================================
# Risk Manager
# =============================================================================

class RiskManager:
    """
    Dynamic SL/TP calculator using ML predictions.
    
    Supports multiple calculation methods:
    - Volatility-based: Simple multiplier on predicted volatility
    - Quantile-based: Uses predicted price distribution
    - Hybrid: Combines both for robust estimates
    
    Example:
        risk_mgr = RiskManager.from_checkpoint("models/weights/tcn_risk_best.pt")
        
        # Calculate levels for a BUY trade
        levels = risk_mgr.calculate_levels(
            features=X_sequence,
            entry_price=1.1050,
            direction='BUY',
        )
        
        print(f"SL: {levels.stop_loss:.5f}")
        print(f"TP: {levels.take_profit:.5f}")
        print(f"RR: {levels.risk_reward_ratio:.2f}")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_columns: List[str],
        config: Optional[RiskConfig] = None,
        device: str = 'auto',
    ):
        """
        Initialize Risk Manager.
        
        Args:
            model: Trained TCNRiskModel
            feature_columns: Features used by the model
            config: Risk calculation configuration
            device: Device for inference
        """
        self.model = model
        self.feature_columns = feature_columns
        self.config = config or RiskConfig()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cache quantile indices
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantiles'):
            self.quantiles = self.model.config.quantiles
        else:
            self.quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[RiskConfig] = None,
        device: str = 'auto',
    ) -> 'RiskManager':
        """
        Create RiskManager from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Risk configuration (optional)
            device: Device for inference
        """
        from models.tcn_risk import load_risk_model_checkpoint
        
        model, feature_columns, checkpoint = load_risk_model_checkpoint(
            checkpoint_path, device=device
        )
        
        return cls(
            model=model,
            feature_columns=feature_columns,
            config=config,
            device=device,
        )
    
    def calculate_levels(
        self,
        features: Union[np.ndarray, torch.Tensor],
        entry_price: float,
        direction: Literal['BUY', 'SELL'],
        method: Optional[RiskMethod] = None,
    ) -> RiskLevels:
        """
        Calculate SL/TP levels for a trade.
        
        Args:
            features: Input features (seq_len, n_features) or (1, seq_len, n_features)
            entry_price: Trade entry price
            direction: Trade direction ('BUY' or 'SELL')
            method: Override default calculation method
        
        Returns:
            RiskLevels with calculated stop-loss and take-profit
        """
        method = method or self.config.method
        
        # Get model predictions
        predictions = self._get_predictions(features)
        
        # Calculate based on method
        if method == RiskMethod.VOLATILITY:
            sl, tp = self._calculate_volatility_based(
                entry_price, direction, predictions['volatility']
            )
            method_name = "volatility"
            
        elif method == RiskMethod.QUANTILE:
            sl, tp = self._calculate_quantile_based(
                entry_price, direction, predictions['quantiles']
            )
            method_name = "quantile"
            
        elif method == RiskMethod.HYBRID:
            sl, tp = self._calculate_hybrid(
                entry_price, direction,
                predictions['volatility'], predictions['quantiles']
            )
            method_name = "hybrid"
            
        elif method == RiskMethod.FIXED_RR:
            sl, tp = self._calculate_fixed_rr(
                entry_price, direction, predictions['volatility']
            )
            method_name = "fixed_rr"
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply safety limits
        sl, tp = self._apply_limits(entry_price, direction, sl, tp)
        
        # Calculate metrics
        sl_pips = abs(entry_price - sl) / self.config.pip_value
        tp_pips = abs(tp - entry_price) / self.config.pip_value
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        
        # Direction confidence
        direction_probs = predictions['direction']
        if direction == 'BUY':
            confidence = direction_probs[2]  # Bull probability
        else:
            confidence = direction_probs[0]  # Bear probability
        
        return RiskLevels(
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=rr_ratio,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            predicted_volatility=predictions['volatility'],
            confidence=confidence,
            method_used=method_name,
            quantile_sl=predictions.get('quantile_sl'),
            quantile_tp=predictions.get('quantile_tp'),
        )
    
    def _get_predictions(self, features: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Get model predictions."""
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(features)
        
        # Extract values
        direction_probs = F.softmax(outputs['direction'], dim=1).cpu().numpy()[0]
        volatility = outputs['volatility'].cpu().numpy()[0, 0]
        quantiles = outputs['quantiles'].cpu().numpy()[0]
        
        # Map quantiles to named values
        quantile_dict = {q: quantiles[i] for i, q in enumerate(self.quantiles)}
        
        return {
            'direction': direction_probs,
            'volatility': volatility,
            'quantiles': quantiles,
            'quantile_dict': quantile_dict,
            'quantile_sl': quantile_dict.get(self.config.sl_quantile),
            'quantile_tp': quantile_dict.get(self.config.tp_quantile),
        }
    
    def _calculate_volatility_based(
        self,
        entry_price: float,
        direction: str,
        volatility: float,
    ) -> Tuple[float, float]:
        """Phase 1: Simple volatility-based calculation."""
        sl_distance = volatility * entry_price * self.config.sl_volatility_multiplier
        tp_distance = volatility * entry_price * self.config.tp_volatility_multiplier
        
        if direction == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return sl, tp
    
    def _calculate_quantile_based(
        self,
        entry_price: float,
        direction: str,
        quantiles: np.ndarray,
    ) -> Tuple[float, float]:
        """Phase 2: Quantile-based calculation."""
        # Find closest quantile indices
        sl_idx = self._find_quantile_idx(self.config.sl_quantile)
        tp_idx = self._find_quantile_idx(self.config.tp_quantile)
        
        if direction == 'BUY':
            # SL based on downside quantile (e.g., Q10)
            sl_return = quantiles[sl_idx]
            # TP based on upside quantile (e.g., Q75)
            tp_return = quantiles[tp_idx]
            
            sl = entry_price * (1 + sl_return)
            tp = entry_price * (1 + tp_return)
        else:
            # For SELL, invert the logic
            sl_return = quantiles[tp_idx]  # Upside risk
            tp_return = quantiles[sl_idx]  # Downside target
            
            sl = entry_price * (1 + sl_return)
            tp = entry_price * (1 + tp_return)
        
        return sl, tp
    
    def _calculate_hybrid(
        self,
        entry_price: float,
        direction: str,
        volatility: float,
        quantiles: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Hybrid calculation: Conservative blend of volatility and quantile methods.
        
        Takes the more conservative of the two approaches for SL,
        and more realistic of the two for TP.
        """
        # Get both estimates
        vol_sl, vol_tp = self._calculate_volatility_based(entry_price, direction, volatility)
        quant_sl, quant_tp = self._calculate_quantile_based(entry_price, direction, quantiles)
        
        if direction == 'BUY':
            # SL: Take the tighter (higher) SL
            sl = max(vol_sl, quant_sl)
            # TP: Take average for realistic target
            tp = (vol_tp + quant_tp) / 2
        else:
            # SL: Take the tighter (lower) SL
            sl = min(vol_sl, quant_sl)
            # TP: Take average
            tp = (vol_tp + quant_tp) / 2
        
        return sl, tp
    
    def _calculate_fixed_rr(
        self,
        entry_price: float,
        direction: str,
        volatility: float,
    ) -> Tuple[float, float]:
        """Fixed risk-reward with volatility-based SL."""
        # SL based on volatility
        sl_distance = volatility * entry_price * self.config.sl_volatility_multiplier
        
        # TP based on minimum RR ratio
        tp_distance = sl_distance * self.config.min_risk_reward
        
        if direction == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return sl, tp
    
    def _find_quantile_idx(self, target_quantile: float) -> int:
        """Find index of closest quantile."""
        return min(
            range(len(self.quantiles)),
            key=lambda i: abs(self.quantiles[i] - target_quantile)
        )
    
    def _apply_limits(
        self,
        entry_price: float,
        direction: str,
        sl: float,
        tp: float,
    ) -> Tuple[float, float]:
        """Apply safety limits to SL/TP."""
        pip = self.config.pip_value
        
        # Calculate current distances in pips
        sl_pips = abs(entry_price - sl) / pip
        tp_pips = abs(tp - entry_price) / pip
        
        # Apply SL limits
        if sl_pips < self.config.min_sl_pips:
            sl_pips = self.config.min_sl_pips
        elif sl_pips > self.config.max_sl_pips:
            sl_pips = self.config.max_sl_pips
        
        # Apply TP limits
        if tp_pips < self.config.min_tp_pips:
            tp_pips = self.config.min_tp_pips
        elif tp_pips > self.config.max_tp_pips:
            tp_pips = self.config.max_tp_pips
        
        # Ensure minimum RR ratio
        current_rr = tp_pips / sl_pips if sl_pips > 0 else 0
        if current_rr < self.config.min_risk_reward:
            tp_pips = sl_pips * self.config.min_risk_reward
        elif current_rr > self.config.max_risk_reward:
            tp_pips = sl_pips * self.config.max_risk_reward
        
        # Reconstruct prices
        if direction == 'BUY':
            sl = entry_price - sl_pips * pip
            tp = entry_price + tp_pips * pip
        else:
            sl = entry_price + sl_pips * pip
            tp = entry_price - tp_pips * pip
        
        return sl, tp
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Account balance in account currency
            entry_price: Trade entry price
            stop_loss: Stop loss price
            risk_percent: Override default risk per trade
        
        Returns:
            Dict with lots, units, risk_amount
        """
        risk_pct = risk_percent or self.config.risk_per_trade
        
        # Risk amount in account currency
        risk_amount = account_balance * risk_pct
        
        # SL distance in pips
        sl_pips = abs(entry_price - stop_loss) / self.config.pip_value
        
        # Value per pip (simplified, assumes USD account for USD pairs)
        # For accurate calculation, need exchange rates
        pip_value_per_lot = 10.0  # $10 per pip for standard lot
        
        # Position size in lots
        lots = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Convert to units (1 lot = 100,000 units)
        units = lots * 100_000
        
        return {
            'lots': round(lots, 2),
            'units': int(units),
            'risk_amount': risk_amount,
            'sl_pips': sl_pips,
            'pip_value': pip_value_per_lot * lots,
        }
    
    def analyze_trade(
        self,
        features: Union[np.ndarray, torch.Tensor],
        entry_price: float,
        direction: Literal['BUY', 'SELL'],
        account_balance: float = 10000.0,
    ) -> Dict:
        """
        Complete trade analysis with SL/TP and position sizing.
        
        Returns comprehensive trade plan.
        """
        # Get risk levels
        levels = self.calculate_levels(features, entry_price, direction)
        
        # Get position size
        position = self.calculate_position_size(
            account_balance, entry_price, levels.stop_loss
        )
        
        # Combine into trade plan
        return {
            'entry': entry_price,
            'direction': direction,
            'stop_loss': levels.stop_loss,
            'take_profit': levels.take_profit,
            'risk_reward': levels.risk_reward_ratio,
            'sl_pips': levels.sl_pips,
            'tp_pips': levels.tp_pips,
            'position_lots': position['lots'],
            'position_units': position['units'],
            'risk_amount': position['risk_amount'],
            'predicted_volatility': levels.predicted_volatility,
            'direction_confidence': levels.confidence,
            'method': levels.method_used,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_risk_manager(
    checkpoint_path: str,
    method: str = 'hybrid',
    **config_kwargs
) -> RiskManager:
    """
    Quick function to create a RiskManager.
    
    Args:
        checkpoint_path: Path to model checkpoint
        method: Risk calculation method ('volatility', 'quantile', 'hybrid', 'fixed_rr')
        **config_kwargs: Override RiskConfig parameters
    """
    # Create config with overrides
    config = RiskConfig(
        method=RiskMethod(method),
        **config_kwargs
    )
    
    return RiskManager.from_checkpoint(checkpoint_path, config=config)


def calculate_sl_tp(
    checkpoint_path: str,
    features: np.ndarray,
    entry_price: float,
    direction: str,
) -> Dict[str, float]:
    """
    One-liner to calculate SL/TP levels.
    
    Usage:
        levels = calculate_sl_tp(
            "models/weights/tcn_risk_best.pt",
            features=X,
            entry_price=1.1050,
            direction='BUY'
        )
    """
    risk_mgr = RiskManager.from_checkpoint(checkpoint_path)
    result = risk_mgr.calculate_levels(features, entry_price, direction)
    return result.to_dict()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Risk Manager Test")
    print("=" * 60)
    
    # Create mock model for testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.tcn_risk import TCNRiskModel, RiskModelConfig
    
    config = RiskModelConfig(input_dim=25, hidden_dim=64)
    model = TCNRiskModel(config)
    
    # Create risk manager
    risk_mgr = RiskManager(
        model=model,
        feature_columns=[f'feature_{i}' for i in range(25)],
        config=RiskConfig(),
    )
    
    # Test features
    features = torch.randn(1, 30, 25)
    entry_price = 1.1050
    
    print("\nTesting BUY trade:")
    levels = risk_mgr.calculate_levels(features, entry_price, 'BUY')
    print(f"   Entry: {entry_price}")
    print(f"   SL: {levels.stop_loss:.5f} ({levels.sl_pips:.1f} pips)")
    print(f"   TP: {levels.take_profit:.5f} ({levels.tp_pips:.1f} pips)")
    print(f"   RR: {levels.risk_reward_ratio:.2f}")
    print(f"   Vol: {levels.predicted_volatility:.6f}")
    print(f"   Method: {levels.method_used}")
    
    print("\nTesting SELL trade:")
    levels = risk_mgr.calculate_levels(features, entry_price, 'SELL')
    print(f"   Entry: {entry_price}")
    print(f"   SL: {levels.stop_loss:.5f} ({levels.sl_pips:.1f} pips)")
    print(f"   TP: {levels.take_profit:.5f} ({levels.tp_pips:.1f} pips)")
    print(f"   RR: {levels.risk_reward_ratio:.2f}")
    
    print("\nTesting position sizing:")
    position = risk_mgr.calculate_position_size(
        account_balance=10000,
        entry_price=1.1050,
        stop_loss=1.1030,
    )
    print(f"   Account: $10,000")
    print(f"   Risk: 1%")
    print(f"   Lots: {position['lots']}")
    print(f"   Units: {position['units']}")
    print(f"   Risk Amount: ${position['risk_amount']:.2f}")
    
    print("\nTesting full trade analysis:")
    analysis = risk_mgr.analyze_trade(features, entry_price, 'BUY', 10000)
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.5f}")
        else:
            print(f"   {k}: {v}")
    
    print("\n✅ All tests passed!")