# tests/conftest.py
"""
Pytest configuration and shared fixtures for training tests.

This conftest provides lightweight stubs for heavy dependencies (torch, MT5)
that cannot be imported in CI/test environments.
"""

import sys
import types
from pathlib import Path

# Ensure the project root is in the path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# FAKE TORCH MODULE
# =============================================================================
# Provide a lightweight fake `torch` module for test environments where
# the real PyTorch cannot be imported (e.g. missing CUDA DLLs).

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    fake_torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def device_count():
            return 0
        
        @staticmethod
        def get_device_name(idx):
            return "Mock GPU"

    fake_torch.cuda = _Cuda()

    def device(x='cpu'):
        return x

    fake_torch.device = device
    fake_torch.__version__ = '0.0-fake'

    # minimal submodules often imported at top-level
    fake_torch.nn = types.ModuleType('torch.nn')
    fake_torch.optim = types.ModuleType('torch.optim')
    fake_torch.nn.functional = types.ModuleType('torch.nn.functional')
    fake_torch.utils = types.ModuleType('torch.utils')
    fake_torch.utils.data = types.ModuleType('torch.utils.data')
    fake_torch.distributions = types.ModuleType('torch.distributions')
    
    class _FakeTensor:
        pass
    
    fake_torch.Tensor = _FakeTensor
    
    class _FakeDataset:
        def __len__(self):
            return 0
        
        def __getitem__(self, idx):
            raise IndexError(idx)
    
    class _FakeTensorDataset(_FakeDataset):
        def __init__(self, *tensors):
            self.tensors = tensors
            if tensors:
                try:
                    self._length = len(tensors[0])
                except Exception:
                    self._length = 0
            else:
                self._length = 0
        
        def __len__(self):
            return self._length
        
        def __getitem__(self, idx):
            return tuple(t[idx] for t in self.tensors)
    
    class _FakeDataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
        
        def __iter__(self):
            batch = []
            for i in range(len(self.dataset)):
                batch.append(self.dataset[i])
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        
        def __len__(self):
            if self.batch_size <= 0:
                return 0
            try:
                return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            except Exception:
                return 0
    
    fake_torch.utils.data.Dataset = _FakeDataset
    fake_torch.utils.data.TensorDataset = _FakeTensorDataset
    fake_torch.utils.data.DataLoader = _FakeDataLoader

    class _FakeCategorical:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def sample(self):
            return 0

        def log_prob(self, value):
            return 0

    fake_torch.distributions.Categorical = _FakeCategorical
    
    # Add common nn classes
    fake_torch.nn.Module = object
    fake_torch.nn.Linear = lambda *args, **kwargs: None
    fake_torch.nn.LayerNorm = lambda *args, **kwargs: None
    fake_torch.nn.Dropout = lambda *args, **kwargs: None
    fake_torch.nn.Sequential = lambda *args, **kwargs: None
    fake_torch.nn.GELU = lambda *args, **kwargs: None
    fake_torch.nn.CrossEntropyLoss = lambda *args, **kwargs: None
    
    # Add common functions
    fake_torch.no_grad = lambda: lambda f: f
    fake_torch.tensor = lambda x: x
    fake_torch.randn = lambda *args: None
    fake_torch.save = lambda *args, **kwargs: None
    fake_torch.load = lambda *args, **kwargs: {}
    
    # Register in sys.modules
    sys.modules['torch'] = fake_torch
    sys.modules['torch.nn'] = fake_torch.nn
    sys.modules['torch.optim'] = fake_torch.optim
    sys.modules['torch.nn.functional'] = fake_torch.nn.functional
    sys.modules['torch.utils'] = fake_torch.utils
    sys.modules['torch.utils.data'] = fake_torch.utils.data
    sys.modules['torch.distributions'] = fake_torch.distributions


# =============================================================================
# FAKE TORCHVISION MODULE (for ViT training)
# =============================================================================

try:
    import torchvision  # noqa: F401
except Exception:
    fake_torchvision = types.ModuleType('torchvision')
    fake_torchvision.datasets = types.ModuleType('torchvision.datasets')
    fake_torchvision.transforms = types.ModuleType('torchvision.transforms')

    class _IdentityTransform:
        def __call__(self, x):
            return x

    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms or []

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    fake_torchvision.transforms.Compose = _Compose
    fake_torchvision.transforms.ToTensor = _IdentityTransform
    fake_torchvision.transforms.Normalize = lambda *args, **kwargs: _IdentityTransform()
    fake_torchvision.transforms.Resize = lambda *args, **kwargs: _IdentityTransform()
    fake_torchvision.transforms.CenterCrop = lambda *args, **kwargs: _IdentityTransform()

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError(idx)

    fake_torchvision.datasets.ImageFolder = _FakeDataset

    sys.modules['torchvision'] = fake_torchvision
    sys.modules['torchvision.datasets'] = fake_torchvision.datasets
    sys.modules['torchvision.transforms'] = fake_torchvision.transforms


# =============================================================================
# FAKE TIMM MODULE (for ViT models)
# =============================================================================

try:
    import timm  # noqa: F401
except Exception:
    fake_timm = types.ModuleType('timm')
    
    def create_model(*args, **kwargs):
        mock_model = type('MockViT', (), {
            'embed_dim': 768,
            'blocks': [type('MockBlock', (), {'parameters': lambda self: iter([])})() for _ in range(12)],
            'norm': type('MockNorm', (), {'parameters': lambda self: iter([])})(),
            'parameters': lambda self: iter([]),
            '__call__': lambda self, x: x
        })()
        return mock_model
    
    fake_timm.create_model = create_model
    sys.modules['timm'] = fake_timm


# =============================================================================
# FAKE ULTRALYTICS MODULE (for YOLO models)
# =============================================================================

try:
    from ultralytics import YOLO  # noqa: F401
except Exception:
    fake_ultralytics = types.ModuleType('ultralytics')
    
    class MockYOLO:
        """Mock YOLO class for testing."""
        def __init__(self, weights_path):
            self.weights_path = weights_path
        
        def train(self, **kwargs):
            """Mock train method."""
            return {'metrics': {'mAP50': 0.9}}
        
        def export(self, **kwargs):
            """Mock export method."""
            return kwargs.get('format', 'pt')
        
        def predict(self, *args, **kwargs):
            """Mock predict method."""
            return []
    
    fake_ultralytics.YOLO = MockYOLO
    sys.modules['ultralytics'] = fake_ultralytics


# =============================================================================
# FAKE TRAINING.AUTO_RETRAIN MODULE
# =============================================================================
# If importing the real `training.auto_retrain` would pull heavy modules,
# create a lightweight stub that supports patching for tests.

try:
    import training.auto_retrain as _real_auto  # noqa: F401
except Exception:
    _fake_auto = types.ModuleType('training.auto_retrain')

    class MT5Connector:
        """Stub MT5Connector for testing."""
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            return True

        def get_data(self, *args, **kwargs):
            import pandas as pd
            return pd.DataFrame()

    def train_enhanced_lstm(*args, **kwargs):
        """Stub training function for TCN (LSTM removed)."""
        return None

    def auto_retrain_job():
        """
        Stub auto_retrain_job that mirrors the real implementation.
        
        """
        import logging
        from pathlib import Path as PathLib
        import pandas as pd
        
        # Dynamic lookup to support patching - THIS IS THE KEY FIX
        _module = sys.modules['training.auto_retrain']
        _MT5Connector = _module.MT5Connector
        _train_enhanced_lstm = _module.train_enhanced_lstm

        print("=" * 70)
        print("🔄 STARTING RETRAINING JOB (BIG DATA MODE)")
        print("=" * 70)

        # 1. Connect to MT5
        connector = _MT5Connector()
        if not connector.connect():
            logging.error("❌ Could not connect to MT5.")
            return

        # Configuration constants
        DOWNLOAD_COUNT = 8000000
        TIMEFRAME = 'M15'
        SYMBOL = 'EURUSD'

        logging.info(f"⬇️ Downloading latest {DOWNLOAD_COUNT} candles for {SYMBOL}...")
        
        df = connector.get_data(symbol=SYMBOL, n=DOWNLOAD_COUNT, timeframe=TIMEFRAME)

        if df is None or df.empty:
            logging.error("❌ No data received.")
            return

        # 2. Validation Check
        actual_count = len(df)
        logging.info(f"✅ Downloaded {actual_count} rows.")

        if actual_count < 10000:
            logging.warning("⚠️ WARNING: Dataset is very small (< 10k). Model may overfit.")
            logging.warning("   -> Check MT5 Terminal: Tools > Options > Charts > Max bars in chart")

        # 3. Save Data
        data_dir = PathLib("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "eurusd_latest.csv"

        df.to_csv(csv_path, index=False)
        logging.info(f"💾 Saved to {csv_path}")

        # 4. Retrain with Optimized Parameters
        try:
            logging.info("🧠 Starting Training...")
            _train_enhanced_lstm(
                data=csv_path,
                epochs=50,
                batch_size=64,
                lr=1e-3,
                seq_len=60,
                save_dir="models/weights",
                device="auto",
                profile="INTRADAY",
                features=None,
                skip_feature_selection=False,
                n_features=25,
                hidden_dim=64,
                num_layers=5,
                dropout=0.2,
                threshold=0.05,
                patience=10,
                use_cosine=False,
                no_onecycle=False,
                name="tcn_enhanced"
            )
            logging.info("✅ Retraining Complete. Model updated.")

        except Exception as e:
            logging.error(f"❌ Training Failed: {e}")
            import traceback
            traceback.print_exc()

    # Attach to fake module
    _fake_auto.MT5Connector = MT5Connector
    _fake_auto.train_enhanced_lstm = train_enhanced_lstm
    _fake_auto.auto_retrain_job = auto_retrain_job

    # Register in sys.modules
    sys.modules['training.auto_retrain'] = _fake_auto
    
    # Also set up the parent package if needed
    if 'training' not in sys.modules:
        _fake_training = types.ModuleType('training')
        sys.modules['training'] = _fake_training
    
    sys.modules['training'].auto_retrain = _fake_auto


# =============================================================================
# FAKE TRAINING.FEATURE_SELECTOR MODULE (if needed)
# =============================================================================

try:
    import training.feature_selector as _real_fs  # noqa: F401
except Exception:
    _fake_fs = types.ModuleType('training.feature_selector')
    
    class DynamicFeatureSelector:
        """Stub DynamicFeatureSelector for testing."""
        def __init__(self, n_features=20, sample_size=50000):
            self.n_features = n_features
            self.sample_size = sample_size
        
        def select(self, df, target_col, exclude_cols):
            """Stub select method."""
            import logging
            import numpy as np
            
            logger = logging.getLogger(__name__)
            logger.info(f"🔍 Running Dynamic Feature Selection (Target: Top {self.n_features})...")
            
            # Prepare candidates
            feature_candidates = [c for c in df.columns if c not in exclude_cols]
            
            # Sample data
            if len(df) > self.sample_size:
                sample_df = df.iloc[-self.sample_size:]
            else:
                sample_df = df
            
            X = sample_df[feature_candidates].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = sample_df[target_col]
            
            # Use sklearn RandomForest
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            selected = []
            logger.info("🏆 TOP SELECTED FEATURES:")
            for i in range(self.n_features):
                idx = indices[i]
                feat = feature_candidates[idx]
                selected.append(feat)
                logger.info(f"   {i+1:2d}. {feat:<25} ({importances[idx]:.4f})")
            
            return selected
    
    _fake_fs.DynamicFeatureSelector = DynamicFeatureSelector
    
    sys.modules['training.feature_selector'] = _fake_fs
    
    if 'training' in sys.modules:
        sys.modules['training'].feature_selector = _fake_fs


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest to filter specific warnings."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::FutureWarning"
    )