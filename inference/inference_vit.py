# inference/inference_vit.py
"""
Inference script for ViT fine-tuned using cached features.
Loads ViTExtractor + your trained classifier head.
"""

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import argparse
import json

from models.vit_extractor import ViTExtractor


# -------------------------------------------
# Args
# -------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="ViT Inference Script")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--class_map", type=str, required=True,
                        help="JSON: index → class name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_head.pth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# -------------------------------------------
# Load model
# -------------------------------------------
def load_model(num_classes, checkpoint, device):
    vit = ViTExtractor().to(device)
    vit.eval()

    classifier = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, num_classes)
    ).to(device)

    classifier.load_state_dict(torch.load(checkpoint, map_location=device))
    classifier.eval()

    return vit, classifier


# -------------------------------------------
# Preprocess single image
# -------------------------------------------
def preprocess(path):
    img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return transform(img).unsqueeze(0)   # [1,3,224,224]


# -------------------------------------------
# Predict
# -------------------------------------------
def predict(vit, classifier, img_tensor, class_names, device):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        features = vit(img_tensor)                 # [1, 768]
        logits = classifier(features)              # [1, num_classes]
        probs = torch.softmax(logits, dim=1)

    top_prob, top_idx = probs[0].max(0)
    return class_names[top_idx.item()], float(top_prob.item())


# -------------------------------------------
# Main
# -------------------------------------------
def main():
    args = get_args()

    # Load class map e.g. {"0":"cat", "1":"dog"}
    with open(args.class_map, "r") as f:
        class_map = json.load(f)

    num_classes = len(class_map)

    # Load backbone + classifier
    vit, classifier = load_model(num_classes, args.checkpoint, args.device)

    # Preprocess input
    img_tensor = preprocess(args.image)

    # Predict
    label, prob = predict(vit, classifier, img_tensor, class_map, args.device)

    print(f"\nPrediction → {label} ({prob:.4f})\n")


if __name__ == "__main__":
    main()
