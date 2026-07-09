"""
Creates data/sample/ with a fixed number of images per class per split.
Copies files (no symlinks) for maximum compatibility.
"""
import shutil
import random
from pathlib import Path

SRC = Path("../raw_data/")
DST = Path("data/sample")
SEED = 42
COUNTS = {"train": 500, "val": 63, "test": 63}

random.seed(SEED)

for split, n in COUNTS.items():
    for cls_dir in sorted((SRC / split).iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))
        selected = random.sample(imgs, min(n, len(imgs)))
        out_dir = DST / split / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in selected:
            shutil.copy2(img, out_dir / img.name)
        print(f"{split}/{cls_dir.name}: {len(selected)} images")

print(f"\nSample dataset written to {DST}")
