# tests/test_training_train_yolo.py
"""
Comprehensive unit tests for training/train_yolo.py

This module tests the YOLOv8 training script which includes:
- CUDA/GPU availability checking and reporting
- YOLO model initialization from pretrained weights
- Training configuration with GPU acceleration
- Model export after training

IMPORTANT: Since train_yolo.py executes training at module import time,
ALL tests use source code analysis only - the module is NEVER imported.

Test Coverage Summary:
======================

| Test Class                  | Focus Area                                    | Tests |
|-----------------------------|-----------------------------------------------|-------|
| TestCUDADetection           | CUDA availability, GPU info, version printing |   6   |
| TestYOLOModelInitialization | Model creation from pretrained weights        |   4   |
| TestTrainingConfiguration   | Training parameters and configuration         |  12   |
| TestModelExport             | Model export functionality                    |   3   |
| TestDeviceConfiguration     | Device selection (GPU/CPU/multi-GPU)          |   4   |
| TestDataLoaderSettings      | Batch size, workers, caching                  |   5   |
| TestMixedPrecision          | AMP (automatic mixed precision) settings      |   3   |
| TestEarlyStopping           | Patience and early stopping configuration     |   3   |
| TestOutputAndLogging        | Print statements and verbose output           |   3   |
| TestModuleStructure         | Module structure and constants                |   5   |
| TestEndToEndWorkflow        | Complete training workflow                    |   4   |
| TestConfigurationValues     | Verify hardcoded configuration values         |   8   |
| TestCodeQuality             | Code quality and documentation                |   4   |

Total: 64 tests
"""

import sys
import os
from pathlib import Path
import re
import ast

import pytest


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_train_yolo_source():
    """Get the source code of train_yolo.py for analysis (NO IMPORT)."""
    # Find the module file path without importing
    project_root = Path(__file__).parent.parent
    source_path = project_root / "training" / "train_yolo.py"
    return source_path.read_text(encoding='utf-8')


def extract_train_call_kwargs(source):
    """Extract keyword arguments from model.train() call in source."""
    # Find the start of model.train(
    start_match = re.search(r'model\.train\s*\(', source)
    if not start_match:
        return None
    
    start_idx = start_match.end()
    
    # Count parentheses to find matching close
    paren_count = 1
    end_idx = start_idx
    
    while paren_count > 0 and end_idx < len(source):
        if source[end_idx] == '(':
            paren_count += 1
        elif source[end_idx] == ')':
            paren_count -= 1
        end_idx += 1
    
    # Return everything between the parentheses (excluding the closing paren)
    return source[start_idx:end_idx - 1]


def extract_value_from_source(source, param_name):
    """Extract a parameter value from the train() call in source code."""
    train_call = extract_train_call_kwargs(source)
    if not train_call:
        return None
    
    # Pattern to find param_name=value
    pattern = rf'{param_name}\s*=\s*([^,\n\)]+)'
    match = re.search(pattern, train_call)
    if match:
        value_str = match.group(1).strip().rstrip(',')
        # Try to evaluate the value
        try:
            return eval(value_str)
        except:
            return value_str
    return None


def extract_export_format(source):
    """Extract the format parameter from model.export() call."""
    pattern = r'model\.export\s*\([^)]*format\s*=\s*["\'](\w+)["\']'
    match = re.search(pattern, source)
    if match:
        return match.group(1)
    return None


def extract_yolo_weights(source):
    """Extract the weights file from YOLO() instantiation."""
    pattern = r'YOLO\s*\(\s*["\']([^"\']+)["\']\s*\)'
    match = re.search(pattern, source)
    if match:
        return match.group(1)
    return None


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope='module')
def train_yolo_source():
    """Get the source code of train_yolo.py (cached per module)."""
    return get_train_yolo_source()


# ============================================================================
# CUDA DETECTION TESTS
# ============================================================================

class TestCUDADetection:
    """Tests for CUDA/GPU detection and reporting."""
    
    def test_cuda_is_available_check_in_source(self, train_yolo_source):
        """Test that torch.cuda.is_available() is called in source."""
        assert 'torch.cuda.is_available()' in train_yolo_source
    
    def test_gpu_device_name_retrieved(self, train_yolo_source):
        """Test that GPU device name is retrieved in source."""
        assert 'torch.cuda.get_device_name' in train_yolo_source
    
    def test_cuda_version_accessed(self, train_yolo_source):
        """Test that CUDA version is accessed in source."""
        assert 'torch.version.cuda' in train_yolo_source
    
    def test_pytorch_version_accessed(self, train_yolo_source):
        """Test that PyTorch version is accessed in source."""
        assert 'torch.__version__' in train_yolo_source
    
    def test_warning_for_no_cuda(self, train_yolo_source):
        """Test warning message exists for no CUDA."""
        assert 'WARNING' in train_yolo_source
        assert 'CUDA not available' in train_yolo_source or 'CPU' in train_yolo_source
    
    def test_gpu_check_banner_exists(self, train_yolo_source):
        """Test GPU CHECK banner exists in source."""
        assert 'GPU CHECK' in train_yolo_source


# ============================================================================
# YOLO MODEL INITIALIZATION TESTS
# ============================================================================

class TestYOLOModelInitialization:
    """Tests for YOLO model initialization."""
    
    def test_yolo_imported_from_ultralytics(self, train_yolo_source):
        """Test YOLO is imported from ultralytics."""
        assert 'from ultralytics import YOLO' in train_yolo_source
    
    def test_yolov8n_weights_used(self, train_yolo_source):
        """Test YOLOv8n pretrained weights are used."""
        weights = extract_yolo_weights(train_yolo_source)
        assert weights == 'yolov8n.pt'
    
    def test_model_variable_created(self, train_yolo_source):
        """Test model variable is created."""
        assert 'model = YOLO' in train_yolo_source
    
    def test_pretrained_weights_extension(self, train_yolo_source):
        """Test pretrained weights have .pt extension."""
        weights = extract_yolo_weights(train_yolo_source)
        assert weights is not None
        assert weights.endswith('.pt')


# ============================================================================
# TRAINING CONFIGURATION TESTS
# ============================================================================

class TestTrainingConfiguration:
    """Tests for training configuration parameters."""
    
    def test_data_yaml_path(self, train_yolo_source):
        """Test correct data yaml path is used."""
        value = extract_value_from_source(train_yolo_source, 'data')
        assert value == "data/yolo.yaml"
    
    def test_epochs_value(self, train_yolo_source):
        """Test epochs is set to 80."""
        value = extract_value_from_source(train_yolo_source, 'epochs')
        assert value == 80
    
    def test_imgsz_value(self, train_yolo_source):
        """Test image size is set to 256."""
        value = extract_value_from_source(train_yolo_source, 'imgsz')
        assert value == 256
    
    def test_device_value(self, train_yolo_source):
        """Test device is set to 0 (GPU)."""
        value = extract_value_from_source(train_yolo_source, 'device')
        assert value == 0
    
    def test_batch_size_value(self, train_yolo_source):
        """Test batch size is set to 16."""
        value = extract_value_from_source(train_yolo_source, 'batch')
        assert value == 16
    
    def test_workers_value(self, train_yolo_source):
        """Test workers is set to 4."""
        value = extract_value_from_source(train_yolo_source, 'workers')
        assert value == 4
    
    def test_amp_enabled(self, train_yolo_source):
        """Test AMP is enabled."""
        value = extract_value_from_source(train_yolo_source, 'amp')
        assert value == True
    
    def test_cache_enabled(self, train_yolo_source):
        """Test cache is enabled."""
        value = extract_value_from_source(train_yolo_source, 'cache')
        assert value == True
    
    def test_patience_value(self, train_yolo_source):
        """Test patience is set to 20."""
        value = extract_value_from_source(train_yolo_source, 'patience')
        assert value == 20
    
    def test_verbose_enabled(self, train_yolo_source):
        """Test verbose is enabled."""
        value = extract_value_from_source(train_yolo_source, 'verbose')
        assert value == True
    
    def test_train_method_called(self, train_yolo_source):
        """Test model.train() is called."""
        assert 'model.train(' in train_yolo_source
    
    def test_all_train_params_present(self, train_yolo_source):
        """Test all required training parameters are present."""
        required_params = ['data', 'epochs', 'imgsz', 'device', 'batch', 
                          'workers', 'amp', 'cache', 'patience', 'verbose']
        
        train_call = extract_train_call_kwargs(train_yolo_source)
        assert train_call is not None
        
        for param in required_params:
            assert f'{param}=' in train_call, f"Missing parameter: {param}"


# ============================================================================
# MODEL EXPORT TESTS
# ============================================================================

class TestModelExport:
    """Tests for model export functionality."""
    
    def test_export_method_called(self, train_yolo_source):
        """Test model.export() is called."""
        assert 'model.export(' in train_yolo_source
    
    def test_export_format_is_pt(self, train_yolo_source):
        """Test export format is 'pt'."""
        format_value = extract_export_format(train_yolo_source)
        assert format_value == 'pt'
    
    def test_completion_message_exists(self, train_yolo_source):
        """Test completion message exists."""
        assert 'Training complete' in train_yolo_source or 'complete' in train_yolo_source.lower()


# ============================================================================
# DEVICE CONFIGURATION TESTS
# ============================================================================

class TestDeviceConfiguration:
    """Tests for device (GPU/CPU) configuration."""
    
    def test_device_is_gpu_zero(self, train_yolo_source):
        """Test device=0 for single GPU."""
        value = extract_value_from_source(train_yolo_source, 'device')
        assert value == 0
    
    def test_cpu_alternative_documented(self, train_yolo_source):
        """Test CPU alternative is documented in comments."""
        assert "'cpu'" in train_yolo_source or '"cpu"' in train_yolo_source
    
    def test_multi_gpu_documented(self, train_yolo_source):
        """Test multi-GPU option is documented."""
        assert '[0,1]' in train_yolo_source or 'multi-GPU' in train_yolo_source
    
    def test_device_comment_exists(self, train_yolo_source):
        """Test device has explanatory comment."""
        assert 'Force GPU' in train_yolo_source or 'GPU' in train_yolo_source


# ============================================================================
# DATA LOADER SETTINGS TESTS
# ============================================================================

class TestDataLoaderSettings:
    """Tests for data loading configuration."""
    
    def test_batch_size_positive(self, train_yolo_source):
        """Test batch size is positive."""
        value = extract_value_from_source(train_yolo_source, 'batch')
        assert value > 0
    
    def test_workers_positive(self, train_yolo_source):
        """Test workers is positive."""
        value = extract_value_from_source(train_yolo_source, 'workers')
        assert value > 0
    
    def test_cache_is_boolean(self, train_yolo_source):
        """Test cache is boolean."""
        value = extract_value_from_source(train_yolo_source, 'cache')
        assert isinstance(value, bool)
    
    def test_imgsz_multiple_of_32(self, train_yolo_source):
        """Test image size is multiple of 32 (YOLO requirement)."""
        value = extract_value_from_source(train_yolo_source, 'imgsz')
        assert value % 32 == 0
    
    def test_data_yaml_extension(self, train_yolo_source):
        """Test data path has .yaml extension."""
        value = extract_value_from_source(train_yolo_source, 'data')
        assert value.endswith('.yaml')


# ============================================================================
# MIXED PRECISION TESTS
# ============================================================================

class TestMixedPrecision:
    """Tests for AMP (Automatic Mixed Precision) settings."""
    
    def test_amp_enabled(self, train_yolo_source):
        """Test AMP is enabled."""
        value = extract_value_from_source(train_yolo_source, 'amp')
        assert value == True
    
    def test_amp_is_boolean(self, train_yolo_source):
        """Test AMP is boolean type."""
        value = extract_value_from_source(train_yolo_source, 'amp')
        assert isinstance(value, bool)
    
    def test_amp_benefits_documented(self, train_yolo_source):
        """Test AMP benefits are documented."""
        # Check for comments about AMP benefits
        assert 'Mixed precision' in train_yolo_source or 'FP16' in train_yolo_source or 'faster' in train_yolo_source


# ============================================================================
# EARLY STOPPING TESTS
# ============================================================================

class TestEarlyStopping:
    """Tests for early stopping configuration."""
    
    def test_patience_positive(self, train_yolo_source):
        """Test patience is positive."""
        value = extract_value_from_source(train_yolo_source, 'patience')
        assert value > 0
    
    def test_patience_reasonable(self, train_yolo_source):
        """Test patience is reasonable (5-50)."""
        value = extract_value_from_source(train_yolo_source, 'patience')
        assert 5 <= value <= 50
    
    def test_patience_documented(self, train_yolo_source):
        """Test patience is documented."""
        assert 'Early stopping' in train_yolo_source or 'patience' in train_yolo_source.lower()


# ============================================================================
# OUTPUT AND LOGGING TESTS
# ============================================================================

class TestOutputAndLogging:
    """Tests for print statements and verbose output."""
    
    def test_verbose_enabled(self, train_yolo_source):
        """Test verbose is enabled."""
        value = extract_value_from_source(train_yolo_source, 'verbose')
        assert value == True
    
    def test_separator_lines_used(self, train_yolo_source):
        """Test separator lines are used for formatting."""
        # Check for the expression pattern (e.g., "=" * 50)
        assert '"=" * 50' in train_yolo_source or "'=' * 50" in train_yolo_source or \
               '=' * 50 in train_yolo_source or '=' * 40 in train_yolo_source
    
    def test_emoji_in_output(self, train_yolo_source):
        """Test emojis are used in output."""
        assert '✅' in train_yolo_source or '⚠️' in train_yolo_source


# ============================================================================
# MODULE STRUCTURE TESTS
# ============================================================================

class TestModuleStructure:
    """Tests for module structure and organization."""
    
    def test_docstring_exists(self, train_yolo_source):
        """Test module has docstring."""
        assert '"""' in train_yolo_source[:200]  # Docstring should be near top
    
    def test_imports_at_top(self, train_yolo_source):
        """Test imports are at top of file."""
        lines = train_yolo_source.split('\n')
        import_found = False
        for line in lines[:20]:  # Check first 20 lines
            if line.startswith('from ') or line.startswith('import '):
                import_found = True
                break
        assert import_found
    
    def test_torch_imported(self, train_yolo_source):
        """Test torch is imported."""
        assert 'import torch' in train_yolo_source
    
    def test_ultralytics_imported(self, train_yolo_source):
        """Test ultralytics is imported."""
        assert 'ultralytics' in train_yolo_source
    
    def test_yolo_class_imported(self, train_yolo_source):
        """Test YOLO class is imported."""
        assert 'YOLO' in train_yolo_source


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """Tests for complete training workflow structure."""
    
    def test_workflow_order_in_source(self, train_yolo_source):
        """Test workflow follows correct order: import -> init -> train -> export."""
        # Find positions of key operations
        import_pos = train_yolo_source.find('from ultralytics import YOLO')
        init_pos = train_yolo_source.find('model = YOLO')
        train_pos = train_yolo_source.find('model.train(')
        export_pos = train_yolo_source.find('model.export(')
        
        assert import_pos < init_pos < train_pos < export_pos
    
    def test_model_used_for_train_and_export(self, train_yolo_source):
        """Test same model variable used for train and export."""
        assert 'model.train(' in train_yolo_source
        assert 'model.export(' in train_yolo_source
    
    def test_print_after_export(self, train_yolo_source):
        """Test completion message after export."""
        export_pos = train_yolo_source.find('model.export(')
        completion_pos = train_yolo_source.find('Training complete')
        
        if completion_pos == -1:
            completion_pos = train_yolo_source.lower().find('complete')
        
        assert completion_pos > export_pos
    
    def test_cuda_check_before_training(self, train_yolo_source):
        """Test CUDA check happens before training."""
        cuda_check_pos = train_yolo_source.find('torch.cuda.is_available()')
        train_pos = train_yolo_source.find('model.train(')
        
        assert cuda_check_pos < train_pos


# ============================================================================
# CONFIGURATION VALUES TESTS
# ============================================================================

class TestConfigurationValues:
    """Tests to verify specific configuration values."""
    
    def test_epochs_is_80(self, train_yolo_source):
        """Test epochs equals 80."""
        value = extract_value_from_source(train_yolo_source, 'epochs')
        assert value == 80
    
    def test_imgsz_is_256(self, train_yolo_source):
        """Test imgsz equals 256."""
        value = extract_value_from_source(train_yolo_source, 'imgsz')
        assert value == 256
    
    def test_device_is_0(self, train_yolo_source):
        """Test device equals 0."""
        value = extract_value_from_source(train_yolo_source, 'device')
        assert value == 0
    
    def test_batch_is_16(self, train_yolo_source):
        """Test batch equals 16."""
        value = extract_value_from_source(train_yolo_source, 'batch')
        assert value == 16
    
    def test_workers_is_4(self, train_yolo_source):
        """Test workers equals 4."""
        value = extract_value_from_source(train_yolo_source, 'workers')
        assert value == 4
    
    def test_patience_is_20(self, train_yolo_source):
        """Test patience equals 20."""
        value = extract_value_from_source(train_yolo_source, 'patience')
        assert value == 20
    
    def test_amp_is_true(self, train_yolo_source):
        """Test amp equals True."""
        value = extract_value_from_source(train_yolo_source, 'amp')
        assert value == True
    
    def test_cache_is_true(self, train_yolo_source):
        """Test cache equals True."""
        value = extract_value_from_source(train_yolo_source, 'cache')
        assert value == True


# ============================================================================
# CODE QUALITY TESTS
# ============================================================================

class TestCodeQuality:
    """Tests for code quality and documentation."""
    
    def test_no_hardcoded_paths_outside_data(self, train_yolo_source):
        """Test no hardcoded absolute paths."""
        # Should not have hardcoded absolute paths like /home/user/...
        assert '/home/' not in train_yolo_source
        assert 'C:\\' not in train_yolo_source
    
    def test_comments_present(self, train_yolo_source):
        """Test that code has comments."""
        comment_count = train_yolo_source.count('#')
        assert comment_count >= 5  # At least some comments
    
    def test_training_params_have_comments(self, train_yolo_source):
        """Test that training parameters have inline comments."""
        # Check for comments in the train() call area
        train_call = extract_train_call_kwargs(train_yolo_source)
        assert train_call is not None
        assert '#' in train_call  # At least some inline comments
    
    def test_install_instructions_present(self, train_yolo_source):
        """Test that CUDA install instructions are present."""
        assert 'pip install' in train_yolo_source or 'Install' in train_yolo_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])