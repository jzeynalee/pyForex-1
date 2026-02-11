# alpha_factory/mhtcn_integration.py
"""
MH-TCN + Alpha Factory 3TF Integration Module

This module bridges the Multi-Head TCN predictions with the Alpha Factory's
3TF (Three-Timeframe) decision system, creating a unified pipeline for
trading decisions.

Architecture:
    MH-TCN (per TF) → FeatureSnapshot → ThreeTFOrchestrator → TradeInstruction

Key Features:
- Uses MH-TCN's direction probabilities for directional_score
- Uses MH-TCN's volatility head for stability estimation
- Uses MH-TCN's quantiles for asymmetric SL/TP
- Uses MH-TCN's outcome head (p_long/p_short) for final confidence
"""

import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .three_tf_system import (
    FeatureSnapshot, HTFDecision, MTFDecision, LTFSignal,
    ThreeTFOrchestrator, ThreeTFLogic, TradeInstruction
)
from .trading_profiles import TradingProfile, ProfileType, TimeFrame, get_profile

logger = logging.getLogger(__name__)


@dataclass
class MHTCNPrediction:
    """Container for MH-TCN prediction outputs."""
    direction_probs: np.ndarray  # [P(Bear), P(Sideways), P(Bull)]
    volatility: float
    quantiles: np.ndarray  # [Q5, Q25, Q50, Q75, Q95]
    p_long: Optional[float] = None
    p_short: Optional[float] = None
    features: Optional[np.ndarray] = None
    
    @property
    def directional_score(self) -> float:
        """Convert direction probs to [-1, +1] score."""
        # Bull - Bear gives directional bias
        return float(self.direction_probs[2] - self.direction_probs[0])
    
    @property
    def confidence(self) -> float:
        """Tradeability confidence.

        Prefer outcome-head probabilities (TP-before-SL) when available.
        This avoids a systematic ~0.33-0.45 ceiling from the 3-class direction head.
        """
        try:
            if self.p_long is not None and self.p_short is not None:
                return float(max(self.p_long, self.p_short))
        except Exception:
            pass
        return float(np.max(self.direction_probs))
    
    @property
    def predicted_direction(self) -> str:
        """Get predicted direction as string."""
        idx = int(np.argmax(self.direction_probs))
        return ['BEAR', 'SIDEWAYS', 'BULL'][idx]
    
    @property
    def stability(self) -> float:
        """Convert volatility to stability score (inverse relationship)."""
        # Higher volatility = lower stability
        # Normalize volatility to [0, 1] range and invert
        # Typical forex volatility is 0.0001 to 0.01
        normalized_vol = np.clip(self.volatility / 0.01, 0, 1)
        return float(1.0 - normalized_vol)


class MHTCNFeatureProvider:
    """
    Bridges MH-TCN predictions to 3TF FeatureSnapshots.
    
    This class:
    1. Loads MH-TCN models for each timeframe
    2. Generates predictions from OHLCV data
    3. Converts predictions to FeatureSnapshots for 3TF logic
    """
    
    def __init__(
        self,
        profile: TradingProfile,
        weights_dir: Optional[str] = None,
        device: str = 'auto'
    ):
        self.profile = profile
        self.weights_dir = Path(weights_dir) if weights_dir else self._get_default_weights_dir()
        self.device = self._resolve_device(device)
        
        # Model cache (lazy loaded)
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}

        # Optional heavy feature engineer (lazy init)
        self._feature_engineer = None
        
        # Feature configuration
        self.sequence_length = 60
        
        logger.info(f"MHTCNFeatureProvider initialized for {profile.type.value}")
    
    def _get_default_weights_dir(self) -> Path:
        """Get default weights directory from settings."""
        try:
            from utils.config import settings
            return Path(getattr(settings, 'WEIGHTS_DIR', 'models/weights'))
        except ImportError:
            return Path('models/weights')
    
    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
    
    def _get_model(self, timeframe: str) -> Any:
        """Get or load MH-TCN model for a timeframe."""
        if timeframe not in self._models:
            self._load_model(timeframe)
        return self._models.get(timeframe)

    def _get_expected_input_dim(self, timeframe: str) -> int:
        """Return the model's expected input feature dimension for a timeframe."""
        try:
            m = self._get_model(timeframe)
            if m is not None and hasattr(m, 'config') and hasattr(m.config, 'input_channels'):
                return int(m.config.input_channels)
        except Exception:
            pass
        return 64

    def _align_feature_dim(self, features: np.ndarray, expected_dim: int) -> np.ndarray:
        """Pad/truncate feature matrix to expected feature dimension."""
        if features is None:
            return features
        try:
            expected_dim = int(expected_dim)
        except Exception:
            expected_dim = 64

        if expected_dim <= 0:
            return features

        cur_dim = int(features.shape[1]) if features.ndim == 2 else 0
        if cur_dim == expected_dim:
            return features

        if cur_dim > expected_dim:
            return features[:, :expected_dim]

        pad = expected_dim - cur_dim
        if pad <= 0:
            return features
        return np.pad(features, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)
    
    def _load_model(self, timeframe: str):
        """Load MH-TCN model for a specific timeframe."""
        try:
            from risk_management.phase1_predictive.tcn_backbone import (
                MultiHeadTCN, TCNConfig, TradingProfile as TCNProfile
            )
            
            # Map profile to TCN profile
            profile_map = {
                ProfileType.SCALPING: TCNProfile.SCALP,
                ProfileType.INTRADAY: TCNProfile.INTRADAY,
                ProfileType.SWING: TCNProfile.SWING,
            }
            tcn_profile = profile_map.get(self.profile.type, TCNProfile.INTRADAY)
            
            # Create config
            config = TCNConfig(
                input_channels=64,  # Will be updated when loading weights
                hidden_channels=128,
                profile=tcn_profile
            )
            
            # Create model
            model = MultiHeadTCN(config).to(self.device)
            
            # Try to load weights
            weight_file = self._find_weight_file(timeframe)
            if weight_file and weight_file.exists():
                checkpoint = torch.load(weight_file, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                state_dict = checkpoint.get('model_state_dict', 
                             checkpoint.get('state_dict', checkpoint))
                
                # Get config from checkpoint
                if 'config' in checkpoint:
                    ckpt_config = checkpoint['config']
                    if isinstance(ckpt_config, dict):
                        input_dim = ckpt_config.get('input_channels', 64)
                        hidden_dim = ckpt_config.get('hidden_channels', 128)
                        
                        # Recreate model with correct dimensions
                        config = TCNConfig(
                            input_channels=input_dim,
                            hidden_channels=hidden_dim,
                            profile=tcn_profile
                        )
                        model = MultiHeadTCN(config).to(self.device)
                
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded MH-TCN weights for {timeframe} from {weight_file}")
                except Exception as e:
                    logger.warning(f"Could not load weights for {timeframe}: {e}")
                
                # Load scaler if available
                if 'scaler_params' in checkpoint:
                    self._scalers[timeframe] = checkpoint['scaler_params']
            else:
                logger.warning(f"No weight file found for {timeframe}, using random initialization")
            
            model.eval()
            self._models[timeframe] = model
            
        except ImportError as e:
            logger.error(f"Could not import MH-TCN: {e}")
            self._models[timeframe] = None
    
    def _find_weight_file(self, timeframe: str) -> Optional[Path]:
        """Find weight file for a timeframe."""
        profile_name = self.profile.type.value.upper()
        aliases = [profile_name]
        if profile_name == 'SCALPING':
            aliases.append('SCALP')
        elif profile_name == 'SCALP':
            aliases.append('SCALPING')

        tf = str(timeframe or '').upper().strip()
        
        # Try different naming conventions
        candidates = []
        for name in aliases:
            candidates.extend([
                self.weights_dir / f"multihead_tcn_{name}_{tf}.pth" if tf else None,
                self.weights_dir / f"multihead_tcn_{name}_{tf}_pa_v1.pth" if tf else None,
                self.weights_dir / f"mhtcn_{name}_{tf}.pth" if tf else None,
                self.weights_dir / f"multihead_tcn_{name}.pth",
                self.weights_dir / f"multihead_tcn_{name}_pa_v1.pth",
                self.weights_dir / f"mhtcn_{name}.pth",
            ])
        candidates.extend([
            self.weights_dir / f"multihead_tcn_INTRADAY.pth",  # Fallback
            self.weights_dir / f"multihead_tcn_INTRADAY_pa_v1.pth",  # Fallback
        ])
        
        for candidate in [c for c in candidates if c is not None]:
            if candidate.exists():
                return candidate
        
        return None
    
    def predict(self, df: pd.DataFrame, timeframe: str) -> Optional[MHTCNPrediction]:
        """
        Generate MH-TCN prediction from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string (e.g., 'M5', 'M15', 'H1')
        
        Returns:
            MHTCNPrediction or None if prediction fails
        """
        model = self._get_model(timeframe)
        if model is None:
            return None
        
        try:
            # Prepare features
            features = self._prepare_features(df, timeframe)
            if features is None:
                return None

            expected_dim = self._get_expected_input_dim(timeframe)
            features = self._align_feature_dim(features, expected_dim)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                outputs = model(x, mode='all')
            
            # Extract outputs
            direction_probs = torch.softmax(outputs['direction'], dim=-1).cpu().numpy()[0]
            volatility = float(outputs['volatility'].cpu().numpy().item())
            quantiles = outputs['quantiles'].cpu().numpy()[0]
            
            # Validate quantiles: enforce monotonicity and realistic range
            if quantiles is not None and len(quantiles) >= 2:
                max_abs = float(np.max(np.abs(quantiles)))
                is_monotonic = all(quantiles[i] <= quantiles[i + 1] for i in range(len(quantiles) - 1))
                if not is_monotonic:
                    # Force monotonicity via sorting
                    quantiles = np.sort(quantiles)
                if max_abs > 0.1:
                    # Unrealistic quantiles (>1000 pips), clamp to safe range
                    quantiles = np.clip(quantiles, -0.05, 0.05)
            
            p_long = None
            p_short = None
            if 'p_long' in outputs:
                p_long = float(outputs['p_long'].cpu().numpy().item())
            if 'p_short' in outputs:
                p_short = float(outputs['p_short'].cpu().numpy().item())
            
            features_out = None
            if 'features' in outputs:
                features_out = outputs['features'].cpu().numpy()[0]
            
            return MHTCNPrediction(
                direction_probs=direction_probs,
                volatility=volatility,
                quantiles=quantiles,
                p_long=p_long,
                p_short=p_short,
                features=features_out
            )
            
        except Exception as e:
            logger.error(
                f"MH-TCN prediction failed for {timeframe}: {e} "
                f"(seq_len={getattr(df, '__len__', lambda: 'n/a')()}, expected_in={self._get_expected_input_dim(timeframe)})"
            )
            return None
    
    def predict_fast(self, df: pd.DataFrame, timeframe: str) -> Optional[MHTCNPrediction]:
        """Fast MH-TCN prediction — skips FeatureEngineerOptimized (~100x faster).

        Uses minimal OHLCV-derived features instead of the full 220+ indicator
        pipeline. Suitable for backtesting where per-bar speed matters.
        """
        model = self._get_model(timeframe)
        if model is None:
            return None

        try:
            features = self._prepare_features_fast(df, timeframe)
            if features is None:
                return None

            expected_dim = self._get_expected_input_dim(timeframe)
            features = self._align_feature_dim(features, expected_dim)

            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            model.eval()
            with torch.no_grad():
                outputs = model(x, mode='all')

            direction_probs = torch.softmax(outputs['direction'], dim=-1).cpu().numpy()[0]
            volatility = float(outputs['volatility'].cpu().numpy().item())
            quantiles = outputs['quantiles'].cpu().numpy()[0]

            if quantiles is not None and len(quantiles) >= 2:
                if not all(quantiles[i] <= quantiles[i + 1] for i in range(len(quantiles) - 1)):
                    quantiles = np.sort(quantiles)
                if float(np.max(np.abs(quantiles))) > 0.1:
                    quantiles = np.clip(quantiles, -0.05, 0.05)

            p_long = float(outputs['p_long'].cpu().numpy().item()) if 'p_long' in outputs else None
            p_short = float(outputs['p_short'].cpu().numpy().item()) if 'p_short' in outputs else None
            features_out = outputs['features'].cpu().numpy()[0] if 'features' in outputs else None

            return MHTCNPrediction(
                direction_probs=direction_probs,
                volatility=volatility,
                quantiles=quantiles,
                p_long=p_long,
                p_short=p_short,
                features=features_out,
            )
        except Exception as e:
            logger.error(f"MH-TCN fast prediction failed for {timeframe}: {e}")
            return None

    def _prepare_features_fast(self, df: pd.DataFrame, timeframe: str) -> Optional[np.ndarray]:
        """Minimal OHLCV feature preparation — no FeatureEngineerOptimized."""
        if df is None or len(df) < self.sequence_length:
            return None

        try:
            d = df.copy()
            d.columns = [c.lower() for c in d.columns]
            if 'volume' not in d.columns:
                d['volume'] = d.get('tick_volume', 1000)

            d = d.tail(self.sequence_length).copy()
            close = d['close'].values.astype(np.float64)
            high = d['high'].values.astype(np.float64)
            low = d['low'].values.astype(np.float64)
            opn = d['open'].values.astype(np.float64)
            vol = d['volume'].values.astype(np.float64)

            base = close[0] if close[0] != 0 else 1.0
            returns = np.diff(close, prepend=close[0]) / (np.abs(close) + 1e-8)
            vol_norm = vol / (np.mean(vol) + 1e-8)

            feature_array = np.stack([
                (close - base) / base,
                (opn - base) / base,
                (high - base) / base,
                (low - base) / base,
                returns,
                vol_norm,
            ], axis=1).astype(np.float32)

            scaler = self._scalers.get(timeframe)
            if scaler is not None:
                feature_array = self._apply_scaling(feature_array, scaler)

            return np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            logger.error(f"Fast feature preparation failed: {e}")
            return None

    def _prepare_features(self, df: pd.DataFrame, timeframe: str) -> Optional[np.ndarray]:
        """Prepare feature array from OHLCV DataFrame."""
        if df is None or len(df) < self.sequence_length:
            return None
        
        try:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            if 'volume' not in df.columns:
                if 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume']
                else:
                    df['volume'] = 1000

            expected_dim = self._get_expected_input_dim(timeframe)

            # Prefer the same feature engineering approach used during training.
            featured_df = None
            try:
                from alpha_factory.features_engineering import FeatureEngineerOptimized
                if self._feature_engineer is None:
                    self._feature_engineer = FeatureEngineerOptimized()
                featured_df = self._feature_engineer.generate_features(df)
            except Exception:
                featured_df = None

            if featured_df is not None and not featured_df.empty:
                # Preserve column order (to match training) and keep only numeric columns.
                feature_cols = [
                    c for c in featured_df.columns
                    if c not in ['time', 'date', 'datetime']
                    and featured_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                ]
                if feature_cols:
                    if expected_dim and len(feature_cols) > int(expected_dim):
                        feature_cols = feature_cols[:int(expected_dim)]
                    featured_df = featured_df[feature_cols].copy()
                    featured_df = featured_df.fillna(0.0)
                    featured_df = featured_df.tail(self.sequence_length)
                    feature_array = featured_df.values.astype(np.float32)
                else:
                    featured_df = None

            if featured_df is None:
                # Fallback: minimal feature set (will be padded/truncated in predict()).
                df = df.tail(self.sequence_length).copy()
                close = df['close'].values.astype(float)
                base_price = close[0] if close[0] != 0 else 1.0
                returns = np.diff(close, prepend=close[0]) / (close + 1e-8)
                vol = df['volume'].values.astype(float)
                vol_norm = vol / (np.mean(vol) + 1e-8)

                feature_array = np.stack([
                    (df['close'].values - base_price) / base_price,
                    (df['open'].values - base_price) / base_price,
                    (df['high'].values - base_price) / base_price,
                    (df['low'].values - base_price) / base_price,
                    returns,
                    vol_norm,
                ], axis=1).astype(np.float32)
            
            # Apply scaling if available
            scaler = self._scalers.get(timeframe)
            if scaler is not None:
                feature_array = self._apply_scaling(feature_array, scaler)
            
            # Handle NaN/Inf
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _apply_scaling(self, features: np.ndarray, scaler: Dict) -> np.ndarray:
        """Apply saved scaling parameters."""
        if 'mean' in scaler and 'std' in scaler:
            mean = np.array(scaler['mean'])
            std = np.array(scaler['std'])
            # Handle dimension mismatch
            if features.shape[1] == len(mean):
                features = (features - mean) / (std + 1e-8)
        elif 'center' in scaler and 'scale' in scaler:
            center = np.array(scaler['center'])
            scale = np.array(scaler['scale'])
            if features.shape[1] == len(center):
                features = (features - center) / (scale + 1e-8)
        return features
    
    def create_snapshot(
        self,
        df: pd.DataFrame,
        timeframe: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[FeatureSnapshot]:
        """
        Create a FeatureSnapshot from OHLCV data using MH-TCN.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string
            timestamp: Optional timestamp (uses last bar time if not provided)
        
        Returns:
            FeatureSnapshot or None if prediction fails
        """
        prediction = self.predict(df, timeframe)
        if prediction is None:
            return None
        
        # Get timestamp
        if timestamp is None:
            if 'time' in df.columns:
                timestamp = pd.to_datetime(df['time'].iloc[-1]).to_pydatetime()
            else:
                timestamp = datetime.utcnow()
        
        # Create regime flags from prediction
        regime_flags = {
            'volatility': prediction.volatility,
            'quantiles': prediction.quantiles.tolist() if prediction.quantiles is not None else [],
            'p_long': prediction.p_long,
            'p_short': prediction.p_short,
        }
        
        # Generate version hash
        import hashlib
        content = f"{timeframe}_{prediction.directional_score:.4f}_{prediction.volatility:.6f}"
        version = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return FeatureSnapshot(
            timestamp=timestamp,
            timeframe=timeframe,
            directional_score=prediction.directional_score,
            confidence=prediction.confidence,
            stability=prediction.stability,
            regime_flags=regime_flags,
            version=version
        )


class UnifiedThreeTFEngine:
    """
    Unified 3TF Trading Engine using MH-TCN predictions.
    
    This is the main entry point for the integrated system.
    It combines:
    - MHTCNFeatureProvider for predictions
    - ThreeTFOrchestrator for decision logic
    - Risk management integration
    """
    
    # Higher thresholds for better trade quality
    MIN_HTF_CONFIDENCE = 0.60  # Was 0.50
    MIN_MTF_CONFIDENCE = 0.65  # Was 0.60
    MIN_LTF_CONFIDENCE = 0.70  # Was 0.65
    MIN_STABILITY = 0.50       # Was 0.40
    MIN_DIRECTIONAL_SCORE = 0.30  # Was 0.25
    
    def __init__(
        self,
        symbol: str,
        profile_type: str = 'INTRADAY',
        weights_dir: Optional[str] = None
    ):
        self.symbol = symbol
        self.profile = get_profile(profile_type)

        self.last_rejection_stage = ""
        self.last_rejection_reason = ""
        
        # Initialize components
        self.feature_provider = MHTCNFeatureProvider(
            profile=self.profile,
            weights_dir=weights_dir
        )
        self.orchestrator = ThreeTFOrchestrator(
            symbol=symbol,
            profile=self.profile
        )
        
        # Override default thresholds with higher values
        self._apply_higher_thresholds()
        
        logger.info(f"UnifiedThreeTFEngine initialized for {symbol} ({profile_type})")
    
    def _apply_higher_thresholds(self):
        """Apply higher thresholds to the 3TF logic."""
        # We'll create a custom logic class with higher thresholds
        pass  # Thresholds are applied in evaluate()
    
    def evaluate(
        self,
        data_htf: pd.DataFrame,
        data_mtf: pd.DataFrame,
        data_ltf: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> Optional[TradeInstruction]:
        """
        Evaluate trading opportunity using 3TF cascade.
        
        Args:
            data_htf: OHLCV data for HTF (e.g., H1 for SCALP)
            data_mtf: OHLCV data for MTF (e.g., M15 for SCALP)
            data_ltf: OHLCV data for LTF (e.g., M5 for SCALP)
            current_time: Current timestamp
        
        Returns:
            TradeInstruction if trade approved, None otherwise
        """
        current_time = current_time or datetime.utcnow()
        
        # Get timeframe strings
        htf_str = self._tf_to_string(self.profile.htf)
        mtf_str = self._tf_to_string(self.profile.mtf)
        ltf_str = self._tf_to_string(self.profile.ltf)
        
        # Create snapshots
        snapshot_htf = self.feature_provider.create_snapshot(data_htf, htf_str, current_time)
        snapshot_mtf = self.feature_provider.create_snapshot(data_mtf, mtf_str, current_time)
        snapshot_ltf = self.feature_provider.create_snapshot(data_ltf, ltf_str, current_time)
        
        if any(s is None for s in [snapshot_htf, snapshot_mtf, snapshot_ltf]):
            logger.warning("Failed to create one or more snapshots")
            self.last_rejection_stage = "SNAPSHOT"
            self.last_rejection_reason = "Failed to create one or more snapshots"
            return None
        
        # Apply higher thresholds before processing
        # Check HTF with higher threshold
        if snapshot_htf.confidence < self.MIN_HTF_CONFIDENCE:
            logger.debug(f"HTF confidence {snapshot_htf.confidence:.2f} < {self.MIN_HTF_CONFIDENCE}")
            self.last_rejection_stage = "HTF_PRECHECK"
            self.last_rejection_reason = f"confidence {snapshot_htf.confidence:.2f} < {self.MIN_HTF_CONFIDENCE:.2f}"
            return None
        
        if snapshot_htf.stability < self.MIN_STABILITY:
            logger.debug(f"HTF stability {snapshot_htf.stability:.2f} < {self.MIN_STABILITY}")
            self.last_rejection_stage = "HTF_PRECHECK"
            self.last_rejection_reason = f"stability {snapshot_htf.stability:.2f} < {self.MIN_STABILITY:.2f}"
            return None
        
        if abs(snapshot_htf.directional_score) < self.MIN_DIRECTIONAL_SCORE:
            logger.debug(f"HTF directional score {abs(snapshot_htf.directional_score):.2f} < {self.MIN_DIRECTIONAL_SCORE}")
            self.last_rejection_stage = "HTF_PRECHECK"
            self.last_rejection_reason = (
                f"abs(directional_score) {abs(snapshot_htf.directional_score):.2f} < {self.MIN_DIRECTIONAL_SCORE:.2f}"
            )
            return None
        
        # Ensure version consistency (use HTF version for all)
        # Create new snapshots with matching versions
        version = snapshot_htf.version
        
        snapshot_htf = FeatureSnapshot(
            timestamp=snapshot_htf.timestamp,
            timeframe=snapshot_htf.timeframe,
            directional_score=snapshot_htf.directional_score,
            confidence=snapshot_htf.confidence,
            stability=snapshot_htf.stability,
            regime_flags=snapshot_htf.regime_flags,
            version=version
        )
        
        snapshot_mtf = FeatureSnapshot(
            timestamp=snapshot_mtf.timestamp,
            timeframe=snapshot_mtf.timeframe,
            directional_score=snapshot_mtf.directional_score,
            confidence=snapshot_mtf.confidence,
            stability=snapshot_mtf.stability,
            regime_flags=snapshot_mtf.regime_flags,
            version=version
        )
        
        snapshot_ltf = FeatureSnapshot(
            timestamp=snapshot_ltf.timestamp,
            timeframe=snapshot_ltf.timeframe,
            directional_score=snapshot_ltf.directional_score,
            confidence=snapshot_ltf.confidence,
            stability=snapshot_ltf.stability,
            regime_flags=snapshot_ltf.regime_flags,
            version=version
        )

        # Derive rejection reasons from strict 3TF logic (without relying on logs).
        try:
            htf = self.orchestrator.logic.htf_decide(snapshot_htf)
            if not htf.allow:
                self.last_rejection_stage = "HTF"
                self.last_rejection_reason = str(htf.reason)
                return None

            mtf = self.orchestrator.logic.mtf_validate(snapshot_mtf, htf)
            if not mtf.pass_structure:
                self.last_rejection_stage = "MTF"
                self.last_rejection_reason = str(mtf.reason)
                return None

            ltf = self.orchestrator.logic.ltf_trigger(snapshot_ltf, htf, mtf)
            if not ltf.trigger:
                self.last_rejection_stage = "LTF"
                self.last_rejection_reason = str(ltf.reason)
                return None
        except Exception:
            # Fall back to orchestrator errors below.
            pass
        
        # Process through 3TF orchestrator
        try:
            instruction = self.orchestrator.process_3tf(
                snapshot_htf=snapshot_htf,
                snapshot_mtf=snapshot_mtf,
                snapshot_ltf=snapshot_ltf
            )

            if instruction is None:
                # If we reached here, precheck + derived logic said trade should pass,
                # so treat this as an invariant/assertion rejection.
                if not self.last_rejection_stage:
                    self.last_rejection_stage = "ORCHESTRATOR"
                    self.last_rejection_reason = "process_3tf returned None"
            else:
                self.last_rejection_stage = ""
                self.last_rejection_reason = ""
            
            if instruction is not None:
                logger.debug(
                    f"Trade instruction generated: {instruction.direction} {self.symbol} "
                    f"(conf={instruction.confidence:.2f}, size_mult={instruction.size_multiplier:.2f})"
                )
            
            return instruction
            
        except AssertionError as e:
            logger.warning(f"3TF assertion failed: {e}")
            self.last_rejection_stage = "ASSERT"
            self.last_rejection_reason = str(e)
            return None
        except Exception as e:
            logger.error(f"3TF processing error: {e}")
            self.last_rejection_stage = "ERROR"
            self.last_rejection_reason = str(e)
            return None
    
    def _tf_to_string(self, tf: TimeFrame) -> str:
        """Convert TimeFrame enum to string format used by MH-TCN."""
        tf_map = {
            TimeFrame.M5: 'M5',
            TimeFrame.M15: 'M15',
            TimeFrame.H1: 'H1',
            TimeFrame.H4: 'H4',
            TimeFrame.D1: 'D1',
        }
        return tf_map.get(tf, 'H1')
    
    def get_sltp_from_quantiles(
        self,
        prediction: MHTCNPrediction,
        entry_price: float,
        direction: str
    ) -> Tuple[float, float]:
        """
        Calculate SL/TP from MH-TCN quantile predictions.
        
        Uses asymmetric quantiles for better risk management:
        - For LONG: SL at Q5, TP at Q75/Q95
        - For SHORT: SL at Q95, TP at Q25/Q5
        """
        if prediction.quantiles is None or len(prediction.quantiles) < 5:
            # Fallback to ATR-based
            return self._fallback_sltp(entry_price, direction)
        
        q5, q25, q50, q75, q95 = prediction.quantiles
        
        if direction == 'LONG':
            # SL below entry, TP above
            sl_distance = abs(q5) if q5 < 0 else abs(q25)
            tp_distance = abs(q75) if q75 > 0 else abs(q95)
            
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            # SL above entry, TP below
            sl_distance = abs(q95) if q95 > 0 else abs(q75)
            tp_distance = abs(q25) if q25 < 0 else abs(q5)
            
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def _fallback_sltp(self, entry_price: float, direction: str) -> Tuple[float, float]:
        """Fallback SL/TP calculation using fixed pips."""
        sl_pips = 30  # 30 pips SL
        tp_pips = 60  # 60 pips TP (2:1 R:R)
        pip_value = 0.0001
        
        if direction == 'LONG':
            return entry_price - sl_pips * pip_value, entry_price + tp_pips * pip_value
        else:
            return entry_price + sl_pips * pip_value, entry_price - tp_pips * pip_value
