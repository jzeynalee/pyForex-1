"""
Phase 3: Triple Barrier Labeling

Implements the triple barrier method from López de Prado's "Advances in Financial ML".

The triple barrier labels each trade based on which barrier is hit first:
1. Upper barrier (Take Profit) → WIN (+1)
2. Lower barrier (Stop Loss) → LOSS (-1)
3. Time barrier (Max holding period) → TIMEOUT (0)

This creates supervised learning targets that incorporate:
- Directional correctness
- Risk management (SL/TP)
- Time decay

Key advantage: Labels represent actual tradeable outcomes, not just price direction.

In addition to sparse entry-point labeling, this module also supports
generating dense, side-specific outcome labels for supervised learning:
- y_long[t]  = 1 if TP is hit before SL within the time barrier when entering long at t
- y_short[t] = 1 if TP is hit before SL within the time barrier when entering short at t

These labels align with training probability heads (p_long/p_short).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BarrierOutcome(Enum):
    """Possible outcomes of triple barrier."""
    WIN = 1        # Hit take profit
    TIMEOUT = 0    # Hit time barrier
    LOSS = -1      # Hit stop loss


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    # Barrier calculation method
    use_dynamic_barriers: bool = True   # Use model-predicted SL/TP vs fixed
    
    # Fixed barrier multipliers (if not using dynamic)
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 3.0
    
    # Time barrier
    max_holding_periods: Dict[str, int] = None  # By profile
    
    # Labeling options
    include_return: bool = True          # Include actual return in label
    min_return_threshold: float = 0.0    # Minimum return to count as WIN
    
    # Vertical barrier (time)
    vertical_barrier_periods: int = 20   # Default if not specified
    
    def __post_init__(self):
        if self.max_holding_periods is None:
            self.max_holding_periods = {
                'SCALP': 12,      # 12 candles
                'INTRADAY': 24,   # 24 candles
                'SWING': 48       # 48 candles
            }


@dataclass
class BarrierLabel:
    """Result of triple barrier labeling for a single sample."""
    outcome: BarrierOutcome
    return_pct: float           # Actual return achieved
    holding_periods: int        # How long position was held
    barrier_hit: str            # 'tp', 'sl', or 'time'
    entry_price: float
    exit_price: float
    entry_idx: int
    exit_idx: int


class TripleBarrierLabeler:
    """
    Generates triple barrier labels for training data.
    
    Process:
    1. For each entry signal, define three barriers:
       - Upper: Take profit level
       - Lower: Stop loss level
       - Vertical: Maximum holding time
    2. Walk forward through price data
    3. Record which barrier is hit first
    4. Assign label based on outcome
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        self.config = config or TripleBarrierConfig()
    
    def generate_labels(
        self,
        prices: pd.DataFrame,
        entry_signals: np.ndarray,
        directions: np.ndarray,
        sl_levels: Optional[np.ndarray] = None,
        tp_levels: Optional[np.ndarray] = None,
        atr: Optional[np.ndarray] = None,
        profile: str = 'INTRADAY'
    ) -> Tuple[np.ndarray, List[BarrierLabel]]:
        """
        Generate triple barrier labels for all entry signals.
        
        Args:
            prices: DataFrame with 'high', 'low', 'close' columns
            entry_signals: Boolean array of entry points
            directions: Array of 1 (BUY) or -1 (SELL) for each entry
            sl_levels: Pre-calculated stop loss prices (optional)
            tp_levels: Pre-calculated take profit prices (optional)
            atr: ATR values for dynamic barrier calculation
            profile: Trading profile for time barrier
        
        Returns:
            (labels_array, detailed_labels_list)
        """
        n_samples = len(prices)
        labels = np.zeros(n_samples, dtype=np.int32)
        detailed_labels = []
        
        # Get time barrier
        max_periods = self.config.max_holding_periods.get(
            profile.upper(),
            self.config.vertical_barrier_periods
        )
        
        # Find entry indices
        entry_indices = np.where(entry_signals)[0]
        
        for idx in entry_indices:
            if idx >= n_samples - 1:
                continue
            
            entry_price = prices['close'].iloc[idx]
            direction = directions[idx]
            
            # Calculate barriers
            if self.config.use_dynamic_barriers and sl_levels is not None and tp_levels is not None:
                sl = sl_levels[idx]
                tp = tp_levels[idx]
            elif atr is not None:
                # Use ATR-based barriers
                atr_val = atr[idx]
                if direction == 1:  # BUY
                    sl = entry_price - (atr_val * self.config.sl_atr_multiplier)
                    tp = entry_price + (atr_val * self.config.tp_atr_multiplier)
                else:  # SELL
                    sl = entry_price + (atr_val * self.config.sl_atr_multiplier)
                    tp = entry_price - (atr_val * self.config.tp_atr_multiplier)
            else:
                logger.warning(f"No barrier method available for index {idx}")
                continue
            
            # Walk forward to find barrier hit
            label_result = self._find_barrier_hit(
                prices=prices,
                entry_idx=idx,
                entry_price=entry_price,
                direction=direction,
                sl=sl,
                tp=tp,
                max_periods=max_periods
            )
            
            labels[idx] = label_result.outcome.value
            detailed_labels.append(label_result)
        
        return labels, detailed_labels

    def generate_outcome_labels(
        self,
        prices: pd.DataFrame,
        atr: np.ndarray,
        profile: str = 'INTRADAY'
    ) -> np.ndarray:
        """Generate dense binary outcome labels for long/short entries.

        This is intended for training probability heads:
        - p_long  ≈ P(TP hits before SL | enter long now)
        - p_short ≈ P(TP hits before SL | enter short now)

        Barrier construction is ATR-based (no model-derived SL/TP) to avoid
        label leakage.

        Args:
            prices: DataFrame with 'high', 'low', 'close' columns
            atr: ATR values aligned to prices (same length)
            profile: Trading profile for time barrier

        Returns:
            Array of shape (n_samples, 2) with columns [y_long, y_short]
            where each label is 1 for TP-first, 0 otherwise.
        """
        n_samples = len(prices)
        labels = np.zeros((n_samples, 2), dtype=np.float32)

        max_periods = self.config.max_holding_periods.get(
            profile.upper(),
            self.config.vertical_barrier_periods
        )

        for idx in range(n_samples):
            if idx >= n_samples - 1:
                continue

            entry_price = prices['close'].iloc[idx]
            atr_val = atr[idx] if atr is not None else None
            if atr_val is None or atr_val <= 0:
                continue

            # Long barriers
            sl_long = entry_price - (atr_val * self.config.sl_atr_multiplier)
            tp_long = entry_price + (atr_val * self.config.tp_atr_multiplier)
            res_long = self._find_barrier_hit(
                prices=prices,
                entry_idx=idx,
                entry_price=entry_price,
                direction=1,
                sl=sl_long,
                tp=tp_long,
                max_periods=max_periods
            )
            labels[idx, 0] = 1.0 if res_long.outcome == BarrierOutcome.WIN else 0.0

            # Short barriers
            sl_short = entry_price + (atr_val * self.config.sl_atr_multiplier)
            tp_short = entry_price - (atr_val * self.config.tp_atr_multiplier)
            res_short = self._find_barrier_hit(
                prices=prices,
                entry_idx=idx,
                entry_price=entry_price,
                direction=-1,
                sl=sl_short,
                tp=tp_short,
                max_periods=max_periods
            )
            labels[idx, 1] = 1.0 if res_short.outcome == BarrierOutcome.WIN else 0.0

        return labels
    
    def _find_barrier_hit(
        self,
        prices: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        direction: int,
        sl: float,
        tp: float,
        max_periods: int
    ) -> BarrierLabel:
        """
        Walk forward through prices to find which barrier is hit first.
        """
        n_samples = len(prices)
        end_idx = min(entry_idx + max_periods, n_samples - 1)
        
        # Walk forward candle by candle
        for i in range(entry_idx + 1, end_idx + 1):
            high = prices['high'].iloc[i]
            low = prices['low'].iloc[i]
            close = prices['close'].iloc[i]
            
            if direction == 1:  # BUY position
                # Check if TP hit (high reached TP)
                if high >= tp:
                    return BarrierLabel(
                        outcome=BarrierOutcome.WIN,
                        return_pct=(tp - entry_price) / entry_price * 100,
                        holding_periods=i - entry_idx,
                        barrier_hit='tp',
                        entry_price=entry_price,
                        exit_price=tp,
                        entry_idx=entry_idx,
                        exit_idx=i
                    )
                
                # Check if SL hit (low reached SL)
                if low <= sl:
                    return BarrierLabel(
                        outcome=BarrierOutcome.LOSS,
                        return_pct=(sl - entry_price) / entry_price * 100,
                        holding_periods=i - entry_idx,
                        barrier_hit='sl',
                        entry_price=entry_price,
                        exit_price=sl,
                        entry_idx=entry_idx,
                        exit_idx=i
                    )
                    
            else:  # SELL position
                # Check if TP hit (low reached TP)
                if low <= tp:
                    return BarrierLabel(
                        outcome=BarrierOutcome.WIN,
                        return_pct=(entry_price - tp) / entry_price * 100,
                        holding_periods=i - entry_idx,
                        barrier_hit='tp',
                        entry_price=entry_price,
                        exit_price=tp,
                        entry_idx=entry_idx,
                        exit_idx=i
                    )
                
                # Check if SL hit (high reached SL)
                if high >= sl:
                    return BarrierLabel(
                        outcome=BarrierOutcome.LOSS,
                        return_pct=(entry_price - sl) / entry_price * 100,
                        holding_periods=i - entry_idx,
                        barrier_hit='sl',
                        entry_price=entry_price,
                        exit_price=sl,
                        entry_idx=entry_idx,
                        exit_idx=i
                    )
        
        # Time barrier hit
        exit_price = prices['close'].iloc[end_idx]
        
        if direction == 1:
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100
        
        # Determine outcome based on return
        if return_pct > self.config.min_return_threshold:
            outcome = BarrierOutcome.WIN
        elif return_pct < -self.config.min_return_threshold:
            outcome = BarrierOutcome.LOSS
        else:
            outcome = BarrierOutcome.TIMEOUT
        
        return BarrierLabel(
            outcome=outcome,
            return_pct=return_pct,
            holding_periods=end_idx - entry_idx,
            barrier_hit='time',
            entry_price=entry_price,
            exit_price=exit_price,
            entry_idx=entry_idx,
            exit_idx=end_idx
        )
    
    def label_with_model_predictions(
        self,
        prices: pd.DataFrame,
        model_predictions: Dict[str, np.ndarray],
        confidence_threshold: float = 0.5,
        profile: str = 'INTRADAY'
    ) -> Tuple[np.ndarray, List[BarrierLabel]]:
        """
        Generate sparse entry-point labels using model-predicted barriers.

        This method is useful for strategy simulation where SL/TP are derived
        from a model. For training probability heads, prefer
        `generate_outcome_labels()` to avoid label leakage.
        
        Args:
            prices: OHLC price data
            model_predictions: Dict with 'direction_probs', 'quantiles', 'volatility'
            confidence_threshold: Minimum confidence to generate entry signal
            profile: Trading profile
        
        Returns:
            (labels, detailed_labels)
        """
        direction_probs = model_predictions['direction_probs']
        quantiles = model_predictions['quantiles']
        
        # Generate entry signals from direction predictions
        pred_directions = np.argmax(direction_probs, axis=1)  # 0=Bear, 1=Side, 2=Bull
        confidence = np.max(direction_probs, axis=1)
        
        # Entry signals: high confidence non-sideways predictions
        entry_signals = (pred_directions != 1) & (confidence >= confidence_threshold)
        
        # Convert direction to +1/-1
        directions = np.where(pred_directions == 2, 1, -1)
        
        # Calculate SL/TP from quantiles
        n_samples = len(prices)
        sl_levels = np.zeros(n_samples)
        tp_levels = np.zeros(n_samples)
        
        for i in range(n_samples):
            entry_price = prices['close'].iloc[i]
            q = quantiles[i]  # [Q5, Q25, Q50, Q75, Q95]
            
            if directions[i] == 1:  # BUY
                sl_levels[i] = entry_price + q[0]  # Q5 (negative move)
                tp_levels[i] = entry_price + q[3]  # Q75 (positive move)
            else:  # SELL
                sl_levels[i] = entry_price + q[4]  # Q95 (positive = bad for sell)
                tp_levels[i] = entry_price + q[1]  # Q25 (negative = good for sell)
        
        return self.generate_labels(
            prices=prices,
            entry_signals=entry_signals,
            directions=directions,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            profile=profile
        )


class TripleBarrierDataset:
    """
    Creates a dataset with triple barrier labels for training.
    
    Handles the complexities of:
    - Forward-looking label generation
    - Proper train/test splitting (respecting time series)
    - Class balancing
    """
    
    def __init__(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        labeler: TripleBarrierLabeler,
        sequence_length: int = 60
    ):
        self.features = features
        self.prices = prices
        self.labeler = labeler
        self.sequence_length = sequence_length
    
    def create_labeled_dataset(
        self,
        entry_signals: np.ndarray,
        directions: np.ndarray,
        sl_levels: np.ndarray,
        tp_levels: np.ndarray,
        profile: str = 'INTRADAY'
    ) -> Dict[str, np.ndarray]:
        """
        Create feature/label pairs for training.
        
        Returns:
            Dict with 'X', 'y', 'returns', 'holding_periods'
        """
        # Generate labels
        labels, detailed = self.labeler.generate_labels(
            prices=self.prices,
            entry_signals=entry_signals,
            directions=directions,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            profile=profile
        )
        
        # Create sequences aligned with labels
        X = []
        y = []
        returns = []
        holding_periods = []
        
        for detail in detailed:
            idx = detail.entry_idx
            
            # Skip if not enough history for sequence
            if idx < self.sequence_length:
                continue
            
            # Get feature sequence ending at entry
            seq = self.features[idx - self.sequence_length:idx]
            X.append(seq)
            y.append(detail.outcome.value)
            returns.append(detail.return_pct)
            holding_periods.append(detail.holding_periods)
        
        return {
            'X': np.array(X),
            'y': np.array(y),
            'returns': np.array(returns),
            'holding_periods': np.array(holding_periods)
        }
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced labels.
        
        Returns weights inversely proportional to class frequency.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        
        return weights


def create_triple_barrier_labels_from_model(
    prices: pd.DataFrame,
    features: np.ndarray,
    model,  # MultiHeadTCN
    sequence_length: int = 60,
    confidence_threshold: float = 0.5,
    profile: str = 'INTRADAY',
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Convenience function to create triple barrier labels using model predictions.
    
    Args:
        prices: OHLC price DataFrame
        features: Feature array
        model: Trained MultiHeadTCN model
        sequence_length: Input sequence length
        confidence_threshold: Minimum confidence for entry signals
        profile: Trading profile
        device: PyTorch device
    
    Returns:
        Dict with labeled training data
    """
    import torch
    
    model.eval()
    
    # Generate predictions for all samples
    n_samples = len(prices)
    all_predictions = {
        'direction_probs': [],
        'quantiles': [],
        'volatility': []
    }
    
    with torch.no_grad():
        for i in range(sequence_length, n_samples):
            seq = features[i - sequence_length:i]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            output = model(seq_tensor, mode='all')
            
            all_predictions['direction_probs'].append(output['direction'].cpu().numpy()[0])
            all_predictions['quantiles'].append(output['quantiles'].cpu().numpy()[0])
            all_predictions['volatility'].append(output['volatility'].cpu().numpy()[0])
    
    # Pad beginning with zeros
    pad_length = sequence_length
    for key in all_predictions:
        arr = np.array(all_predictions[key])
        pad_shape = (pad_length,) + arr.shape[1:]
        all_predictions[key] = np.concatenate([np.zeros(pad_shape), arr], axis=0)
    
    # Create labeler and generate labels
    labeler = TripleBarrierLabeler()
    labels, detailed = labeler.label_with_model_predictions(
        prices=prices,
        model_predictions=all_predictions,
        confidence_threshold=confidence_threshold,
        profile=profile
    )
    
    # Create dataset
    dataset_creator = TripleBarrierDataset(
        features=features,
        prices=prices,
        labeler=labeler,
        sequence_length=sequence_length
    )
    
    # Extract entry signals and directions from predictions
    pred_dirs = np.argmax(all_predictions['direction_probs'], axis=1)
    confidence = np.max(all_predictions['direction_probs'], axis=1)
    entry_signals = (pred_dirs != 1) & (confidence >= confidence_threshold)
    directions = np.where(pred_dirs == 2, 1, -1)
    
    # Get SL/TP from quantiles
    sl_levels = np.zeros(n_samples)
    tp_levels = np.zeros(n_samples)
    
    for i in range(n_samples):
        entry_price = prices['close'].iloc[i]
        q = all_predictions['quantiles'][i]
        
        if directions[i] == 1:
            sl_levels[i] = entry_price + q[0]
            tp_levels[i] = entry_price + q[3]
        else:
            sl_levels[i] = entry_price + q[4]
            tp_levels[i] = entry_price + q[1]
    
    return dataset_creator.create_labeled_dataset(
        entry_signals=entry_signals,
        directions=directions,
        sl_levels=sl_levels,
        tp_levels=tp_levels,
        profile=profile
    )
