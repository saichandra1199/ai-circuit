"""
Inspect a .pth checkpoint: print backbone (if saved), layer key patterns, and param count.

    python -m utils.inspect_checkpoint path/to/best_model.pth
"""
import sys
from pathlib import Path

import torch


def inspect(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        backbone = ckpt.get("backbone", "unknown (not saved)")
        state_dict = ckpt["state_dict"]
        print(f"Backbone (saved):  {backbone}")
    else:
        state_dict = ckpt
        print("Backbone (saved):  not present — raw state_dict")

    keys = list(state_dict.keys())
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total params:      {total_params:,}")
    print(f"Total layers:      {len(keys)}")
    print(f"\nFirst 10 keys:")
    for k in keys[:10]:
        print(f"  {k}")

    # heuristic backbone detection from key patterns
    guess = None
    first = keys[0] if keys else ""
    if "features." in first:
        guess = "efficientnet_b* or mobilenet"
    elif "stages." in first:
        guess = "convnext_*"
    elif "layers." in first and "blocks." in first:
        guess = "swin_*"
    elif "blocks." in first and "patch_embed" in " ".join(keys[:20]):
        guess = "vit_*"
    elif "layer1." in first or "layer2." in first:
        guess = "resnet*"

    if guess:
        print(f"\nArchitecture guess from key patterns: {guess}")
    else:
        print("\nArchitecture guess: unknown — check key names above")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.inspect_checkpoint <path_to.pth>")
        sys.exit(1)
    inspect(sys.argv[1])
