# 1. Activate Environment
& d:\myBot\.venv312\scripts\activate.ps1; 

# 2. Train Core Models (MultiHead-TCN, ViT, Meta-Labeling, Exit-RL)
# Note: We skip the generic YOLO in favor of the profile-specific ones above
python scripts/train_all_models.py --models tcn vit meta exit --profiles all --skip-yolo --epochs 50;

# 3. Train Profile-Specific YOLO Models (Patterns)
#python training/train_yolo_profiles.py --profile ALL;

# 4. Train Decision Fusion Layers (Combining TCN + ViT + YOLO)
# We execute training for every timeframe in every profile (9 sessions total) to support MTF decision making.

# SCALP (M5, M15, H1)
python training/train_decision_fusion.py --profile SCALP --timeframe M5 --epochs 30 --tcn_checkpoint models/weights/scalp_m5_best.pt --vit_checkpoint models/weights/vit_SCALP.pth --yolo_checkpoint models/yolo/yolo_scalp.pt;
python training/train_decision_fusion.py --profile SCALP --timeframe M15 --epochs 30 --tcn_checkpoint models/weights/scalp_m15_best.pt --vit_checkpoint models/weights/vit_SCALP.pth --yolo_checkpoint models/yolo/yolo_scalp.pt;
python training/train_decision_fusion.py --profile SCALP --timeframe H1 --epochs 30 --tcn_checkpoint models/weights/scalp_h1_best.pt --vit_checkpoint models/weights/vit_SCALP.pth --yolo_checkpoint models/yolo/yolo_scalp.pt;

# INTRADAY (M15, H1, H4)
python training/train_decision_fusion.py --profile INTRADAY --timeframe M15 --epochs 30 --tcn_checkpoint models/weights/intraday_m15_best.pt --vit_checkpoint models/weights/vit_INTRADAY.pth --yolo_checkpoint models/yolo/yolo_intraday.pt;
python training/train_decision_fusion.py --profile INTRADAY --timeframe H1 --epochs 30 --tcn_checkpoint models/weights/intraday_h1_best.pt --vit_checkpoint models/weights/vit_INTRADAY.pth --yolo_checkpoint models/yolo/yolo_intraday.pt;
python training/train_decision_fusion.py --profile INTRADAY --timeframe H4 --epochs 30 --tcn_checkpoint models/weights/intraday_h4_best.pt --vit_checkpoint models/weights/vit_INTRADAY.pth --yolo_checkpoint models/yolo/yolo_intraday.pt;

# SWING (H1, H4, D1)
python training/train_decision_fusion.py --profile SWING --timeframe H1 --epochs 30 --tcn_checkpoint models/weights/swing_h1_best.pt --vit_checkpoint models/weights/vit_SWING.pth --yolo_checkpoint models/yolo/yolo_swing.pt;
python training/train_decision_fusion.py --profile SWING --timeframe H4 --epochs 30 --tcn_checkpoint models/weights/swing_h4_best.pt --vit_checkpoint models/weights/vit_SWING.pth --yolo_checkpoint models/yolo/yolo_swing.pt;
python training/train_decision_fusion.py --profile SWING --timeframe D1 --epochs 30 --tcn_checkpoint models/weights/swing_d1_best.pt --vit_checkpoint models/weights/vit_SWING.pth --yolo_checkpoint models/yolo/yolo_swing.pt;