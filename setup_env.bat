@echo off
:: Create fresh environment
python -m venv venv
call venv\Scripts\activate

:: 1. Install PyTorch 2.9 (CUDA 12.4 compatible for 2025) first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: 2. Install Torch Geometric dependencies (pinned to Torch 2.9)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.9.0+cu124.html

:: 3. Install the rest
pip install -r requirements.txt
echo Environment Setup Complete.
pause