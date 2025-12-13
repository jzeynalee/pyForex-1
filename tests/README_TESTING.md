# PyForex Testing Guide

## Quick Start

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_inference_predictor.py -v
```

## Test Files Overview

### New Test Files (Post-LSTM Removal)

1. **test_inference_predictor.py** - TCN-based prediction system
   - Tests `RiskAwareTCNPredictor` and `HybridPredictor`
   - Validates risk management outputs (volatility, quantiles)
   - Covers device selection and model loading

2. **test_strategies_style.py** - Style-specific trading strategies
   - Tests `ScalpingStrategy`, `IntradayStrategy`, `SwingStrategy`
   - Validates signal generation and filtering
   - Covers risk/reward calculations

3. **test_utils_checkpoint_loader.py** - Model checkpoint loading
   - Tests loading TCN checkpoints
   - Validates backward compatibility
   - Covers feature extraction from checkpoints

### Updated Test Files

4. **test_models.py** - ML model architectures (EXPANDED)
   - Added TCN variant tests
   - Tests `CausalConv1d`, `TCNBlock`, `TCNBackbone`
   - Validates profile presets (SCALP, INTRADAY, SWING)

5. **test_main.py** - CLI interface (UPDATED)
   - Updated for TCN instead of LSTM
   - Tests all CLI commands
   - Validates argument parsing

### Existing Test Files

6. **test_training_auto_retrain.py** - Auto-retraining system
7. **test_training_feature_selector.py** - Feature selection
8. **test_training_finetune_vit.py** - ViT fine-tuning
9. **test_training_train_tcn_enhanced.py** - TCN training
10. **test_training_train_yolo.py** - YOLO training
11. **test_trading.py** - Trading logic
12. **test_trend_detection.py** - Trend detection
13. **test_integration.py** - Integration tests

## Running Tests by Category

### Unit Tests Only
```bash
pytest tests/ -v -m unit
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

### Training Tests
```bash
pytest tests/test_training_*.py -v
```

### Model Tests
```bash
pytest tests/test_models.py -v
pytest tests/test_inference_predictor.py -v
```

### Strategy Tests
```bash
pytest tests/test_strategies_style.py -v
pytest tests/test_trading.py -v
```

## Running Tests with Coverage

### Full Coverage Report
```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

### Module-Specific Coverage
```bash
# Inference module
pytest tests/test_inference_predictor.py --cov=inference --cov-report=term

# Strategies module
pytest tests/test_strategies_style.py --cov=strategies --cov-report=term

# Models module
pytest tests/test_models.py --cov=models --cov-report=term
```

### View HTML Coverage Report
After running with `--cov-report=html`, open:
```
htmlcov/index.html
```

## Continuous Integration

### Fast Test Run (CI-friendly)
```bash
pytest tests/ -v --tb=short -x
```

### Parallel Execution
```bash
pytest tests/ -v -n auto
```

## Test Configuration

### pytest.ini
Located in project root, controls:
- Test discovery patterns
- Warning filters
- Coverage settings
- Markers

### conftest.py
Located in `tests/`, provides:
- Fake torch/timm/ultralytics modules for CI
- Shared fixtures
- Test configuration

## Troubleshooting

### Import Errors
If you get import errors:
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v
```

### Torch/CUDA Issues
Tests use mocked torch for CI compatibility. If you need real torch:
```python
# In conftest.py, ensure torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
```

### File Lock Issues (Windows)
The test suite includes automatic cleanup for logging handlers to prevent file lock issues on Windows.

## Writing New Tests

### Test File Structure
```python
import pytest
from module_to_test import function_to_test

@pytest.mark.unit
class TestMyFeature:
    """Test suite for MyFeature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test()
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Fixtures
```python
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({'close': [1.1, 1.2, 1.3]})

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert len(sample_data) == 3
```

### Using Mocks
```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mock objects."""
    with patch('module.function') as mock_func:
        mock_func.return_value = 42
        result = my_function()
        assert result == 42
```

## Test Markers

Available markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (skipped in quick runs)

Run specific markers:
```bash
pytest tests/ -v -m unit
pytest tests/ -v -m "not slow"
```

## Best Practices

1. **Test Naming**: Use descriptive names starting with `test_`
2. **Docstrings**: Add clear docstrings to all tests
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Isolation**: Each test should be independent
5. **Mocking**: Mock external dependencies (MT5, torch, etc.)
6. **Fixtures**: Use fixtures for common setup
7. **Coverage**: Aim for >80% coverage on critical paths

## Common Test Patterns

### Testing Exceptions
```python
def test_raises_error():
    with pytest.raises(ValueError, match="invalid input"):
        function_that_raises(bad_input)
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply(input, expected):
    assert multiply_by_two(input) == expected
```

### Testing with Temporary Files
```python
def test_file_operation(tmp_path):
    """Test using pytest's tmp_path fixture."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test data")
    assert file_path.exists()
```

## Performance Tips

1. **Use pytest-xdist** for parallel execution
2. **Mock expensive operations** (torch.load, etc.)
3. **Use fixtures with scope** to share setup
4. **Skip slow tests** in quick runs
5. **Cache test data** when possible

## Getting Help

- View test results: `pytest tests/ -v`
- Show print statements: `pytest tests/ -v -s`
- Stop at first failure: `pytest tests/ -v -x`
- Show local variables on failure: `pytest tests/ -v -l`
- Run last failed tests: `pytest tests/ --lf`

---

**Updated**: 2025-12-13
**For**: pyForex v2.0 (Post-LSTM Removal)
