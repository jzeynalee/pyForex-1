# 1. Activate Environment
& d:\myBot\.venv312\scripts\activate.ps1; 

# 2. Train Profile-Specific YOLO Models (Patterns)
python training/train_yolo_profiles.py --profile ALL;

# 3. Train Core Models (MultiHead-TCN, ViT, Meta-Labeling, Exit-RL)
# Note: We skip the generic YOLO in favor of the profile-specific ones above
python scripts/train_all_models.py --models tcn vit meta exit --profiles all --skip-yolo --epochs 50;

# 4. Train Decision Fusion Layers (Combining TCN + ViT + YOLO)
# We explicitly link the checkpoints generated in step 3 to the fusion trainer
python training/train_decision_fusion.py --profile SCALP --epochs 30 --tcn_checkpoint models/weights/multihead_tcn_SCALP.pth --vit_checkpoint models/weights/vit_SCALP.pth --yolo_checkpoint models/yolo/yolo_scalp.pt;
python training/train_decision_fusion.py --profile INTRADAY --epochs 30 --tcn_checkpoint models/weights/multihead_tcn_INTRADAY.pth --vit_checkpoint models/weights/vit_INTRADAY.pth --yolo_checkpoint models/yolo/yolo_intraday.pt;
python training/train_decision_fusion.py --profile SWING --epochs 30 --tcn_checkpoint models/weights/multihead_tcn_SWING.pth --vit_checkpoint models/weights/vit_SWING.pth --yolo_checkpoint models/yolo/yolo_swing.pt