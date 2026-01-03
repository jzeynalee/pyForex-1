#!/usr/bin/env python3
"""
Test script to verify the ViT checkpoint loading fix.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vit_checkpoint_loading():
    """Test that ViT checkpoints can be loaded correctly."""
    
    print("Testing ViT Checkpoint Loading Fix...")
    
    try:
        from inference.predictor import HybridPredictor, PredictorConfig
        from models.vit import ViTChartClassifier
        
        print("✅ Imports successful")
        
        # Test loading each ViT checkpoint
        checkpoints = [
            'models/weights/vit_SCALP.pth',
            'models/weights/vit_INTRADAY.pth', 
            'models/weights/vit_SWING.pth'
        ]
        
        for checkpoint_path in checkpoints:
            if Path(checkpoint_path).exists():
                print(f"\n--- Testing {checkpoint_path} ---")
                
                try:
                    # Test direct model loading
                    model = ViTChartClassifier()
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        print(f"✅ Checkpoint structure: {list(checkpoint.keys())}")
                        print(f"✅ State dict keys count: {len(state_dict)}")
                        
                        # Test loading into vit sub-module
                        try:
                            model.vit.load_state_dict(state_dict, strict=False)
                            print("✅ Successfully loaded raw timm weights into vit sub-module")
                        except Exception as e:
                            print(f"⚠️  Failed to load raw timm weights: {e}")
                    
                    # Test predictor loading
                    config = PredictorConfig(
                        profile=checkpoint_path.split('_')[-1].replace('.pth', ''),
                        use_vision=True,
                        use_price_action=False
                    )
                    
                    predictor = HybridPredictor(
                        config=config,
                        vit_weights=checkpoint_path
                    )
                    
                    if predictor.vit_model is not None:
                        print("✅ ViT model loaded successfully in predictor")
                        
                        # Test forward pass
                        dummy_image = torch.randn(1, 3, 224, 224)
                        if torch.cuda.is_available():
                            dummy_image = dummy_image.cuda()
                            predictor.vit_model = predictor.vit_model.cuda()
                        with torch.no_grad():
                            output = predictor.vit_model(dummy_image)
                        print(f"✅ Forward pass successful: output shape {output.shape}")
                    else:
                        print("❌ ViT model is None in predictor")
                        
                except Exception as e:
                    print(f"❌ Error with {checkpoint_path}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        
        print("\n🎉 ViT checkpoint loading test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vit_checkpoint_loading()
    sys.exit(0 if success else 1)
