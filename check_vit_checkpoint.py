#!/usr/bin/env python3
"""
Check ViT checkpoint structure to identify the issue.
"""

import torch
from pathlib import Path

def check_vit_checkpoint():
    """Check the structure of ViT checkpoints."""
    
    checkpoints = [
        'models/weights/vit_SCALP.pth',
        'models/weights/vit_INTRADAY.pth', 
        'models/weights/vit_SWING.pth'
    ]
    
    for checkpoint_path in checkpoints:
        if Path(checkpoint_path).exists():
            print(f"\n{'='*60}")
            print(f"Checking: {checkpoint_path}")
            print(f"{'='*60}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"Keys in checkpoint: {list(checkpoint.keys())}")
                
                for key, value in checkpoint.items():
                    if isinstance(value, dict):
                        print(f"{key}: {type(value)} with keys: {list(value.keys())[:5]}...")
                    elif hasattr(value, 'shape'):
                        print(f"{key}: {type(value)} - shape: {value.shape}")
                    else:
                        print(f"{key}: {type(value)} - value: {value}")
                        
                # Check if it's a state_dict or full model
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"\nState dict keys (first 10): {list(state_dict.keys())[:10]}")
                    
                    # Look for typical ViT keys
                    vit_keys = [k for k in state_dict.keys() if 'vit' in k or 'head' in k or 'patch' in k]
                    print(f"ViT-related keys: {vit_keys[:5]}...")
                    
            except Exception as e:
                print(f"Error loading {checkpoint_path}: {e}")
        else:
            print(f"\nCheckpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    check_vit_checkpoint()
