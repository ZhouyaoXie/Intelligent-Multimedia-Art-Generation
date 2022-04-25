import os
import clip
import torch
import sys

# usage:   --  python3 download_weights.py "ViT-B/32" clip_weights.pt

model_name, output_path = sys.argv[1], sys.argv[2]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load(model_name, device)

torch.save(model.state_dict(), output_path)