"""
Creates data/sample/ with a fixed number of images per class per split.
Copies files (no symlinks) for maximum compatibility.
Also writes class_mapping.json and class_weights.json inside DST/.
"""
import json
import shutil
import random
from collections import Counter
from pathlib import Path

SRC = Path("../raw_data/")
DST = Path("data/sample")
SEED = 42
COUNTS = {"train": 500, "val": 63, "test": 63}

random.seed(SEED)

train_counts: Counter = Counter()

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
        if split == "train":
            train_counts[cls_dir.name] = len(selected)

# class_mapping: index → class name (sorted)
class_names = sorted(train_counts.keys())
class_mapping = {str(i): cls for i, cls in enumerate(class_names)}

# class_weights: inverse frequency
total_train = sum(train_counts.values())
n_classes = len(class_names)
class_weights = {cls: round(total_train / (n_classes * train_counts[cls]), 4) for cls in class_names}

with open(DST / "class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)
with open(DST / "class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)

print(f"\nClass mapping  → {DST}/class_mapping.json")
print(f"Class weights  → {DST}/class_weights.json")
print(f"Sample dataset written to {DST}")
