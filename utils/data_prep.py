"""
Prepare H&M 5-class dataset with quality checks, dedup, balanced cap, and split.
Output: data/<output_dir>/{train,val,test}/<class>/<img>.jpg
Also writes: class_mapping.json, class_weights.json, dataset_stats.json

Run from project root:
    python -m utils.data_prep
    python -m utils.data_prep --raw ../HM_Data/raw_data --out data/prepared --max-per-class 500
"""

# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIG — edit here or pass as CLI args (see --help)
# ══════════════════════════════════════════════════════════════════════════════

RAW_DATA_DIR  = "../HM_Data/raw_data"   # path to raw dataset (needs articles.csv + images/)
OUTPUT_DIR    = "data/prepared"         # where to write train/val/test splits
MAX_PER_CLASS = 5000                    # cap images per class for balance; None = use all
SEED          = 42

# Classes to include — must match values in the CSV label column
CLASSES = [
    "Garment Upper body",
    "Garment Lower body",
    "Garment Full body",
    "Accessories",
    "Shoes",
]

# CSV column names — change only if using a different dataset
LABEL_COL = "product_group_name"
ID_COL    = "article_id"

# ══════════════════════════════════════════════════════════════════════════════

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _img_path(images_dir: Path, article_id: str) -> Path:
    padded = article_id.zfill(10)
    return images_dir / padded[:3] / f"{padded}.jpg"


def _dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash for near-duplicate detection."""
    gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr  = np.array(gray)
    diff = arr[:, 1:] > arr[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def _is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _compute_mean_std(paths: list, sample_n: int = 2000) -> tuple:
    """Per-channel mean/std on random sample; images resized to 224x224 first."""
    rng     = np.random.default_rng(SEED)
    sampled = rng.choice(paths, min(sample_n, len(paths)), replace=False)
    means, stds = [], []
    for p in tqdm(sampled, desc="mean/std", leave=False):
        try:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB").resize((224, 224))) / 255.0
            means.append(arr.mean(axis=(0, 1)))
            stds.append(arr.std(axis=(0, 1)))
        except Exception:
            continue
    return np.array(means).mean(axis=0).tolist(), np.array(stds).mean(axis=0).tolist()


def prepare(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir:   Path = OUTPUT_DIR,
    max_per_class: int = MAX_PER_CLASS,
    seed: int = SEED,
) -> dict:
    images_dir = Path(raw_data_dir) / "images"
    out = Path(output_dir)

    # ── 1. load CSV, filter to 5 classes ────────────────────────────────────
    df = pd.read_csv(Path(raw_data_dir) / "articles.csv", dtype=str)
    df = df[df[LABEL_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"Rows in 5 classes: {len(df)}")
    print(df[LABEL_COL].value_counts().to_string())

    # ── 2. resolve paths, keep only those on disk ───────────────────────────
    df["_path"] = df[ID_COL].apply(lambda aid: _img_path(images_dir, aid))
    df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"\nImages on disk: {len(df)}")

    # ── 3. validate — PIL can actually open them ─────────────────────────────
    print("Validating images...")
    valid = [_is_valid(p) for p in tqdm(df["_path"], leave=False)]
    n_corrupt = valid.count(False)
    df = df[valid].reset_index(drop=True)
    print(f"Corrupt dropped: {n_corrupt}  →  {len(df)} remain")

    # ── 4. perceptual-hash dedup ─────────────────────────────────────────────
    print("Deduplicating...")
    seen: dict = {}
    keep = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        try:
            with Image.open(row["_path"]) as im:
                h = _dhash(im)
        except Exception:
            keep.append(False)
            continue
        if h not in seen:
            seen[h] = True
            keep.append(True)
        else:
            keep.append(False)
    n_dupes = keep.count(False)
    df = df[keep].reset_index(drop=True)
    print(f"Duplicates removed: {n_dupes}  →  {len(df)} remain")

    # ── 5. cap per class for balance ─────────────────────────────────────────
    df = (df.groupby(LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
            .reset_index(drop=True))
    print(f"\nAfter cap ({max_per_class}/class):")
    print(df[LABEL_COL].value_counts().to_string())

    # ── 6. stratified 80/10/10 split ─────────────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df[LABEL_COL], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[LABEL_COL], random_state=seed
    )
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    # ── 7. copy images into split/class dirs ─────────────────────────────────
    print("\nCopying images...")
    for split_name, split_df in splits:
        for cls in CLASSES:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name, leave=False):
            dst = out / split_name / row[LABEL_COL] / row["_path"].name
            if not dst.exists():
                shutil.copy2(row["_path"], dst)

    # ── 8. class mapping + inverse-frequency weights ──────────────────────────
    class_names   = sorted(CLASSES)
    class_mapping = {str(i): cls for i, cls in enumerate(class_names)}
    train_counts  = Counter(train_df[LABEL_COL])
    total_train   = sum(train_counts.values())
    n             = len(class_names)
    class_weights = {
        cls: round(total_train / (n * train_counts[cls]), 4)
        for cls in class_names
    }

    with open(out / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    with open(out / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    # ── 9. dataset-specific mean/std ─────────────────────────────────────────
    print("Computing dataset mean/std...")
    train_paths = list(train_df["_path"])
    mean, std   = _compute_mean_std(train_paths)

    dataset_stats = {
        "mean": mean,
        "std":  std,
        "note": "Computed on 224x224 RGB images normalised to [0,1].",
        # Images are portrait (~1166x1750). At training time:
        # resize height→256, pad width→256, then random 224 crop (train) / center crop (val/test).
        "portrait_crop_recommendation": {
            "train": "Resize(256, interpolation=bilinear) on height, pad width, RandomCrop(224)",
            "val":   "Resize(256) on height, pad width, CenterCrop(224)",
        },
    }
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # ── 10. summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Saved → {out}")
    for split_name, split_df in splits:
        counts = Counter(split_df[LABEL_COL])
        row_s  = "  ".join(f"{cls.split()[-1]}={counts.get(cls, 0)}" for cls in class_names)
        print(f"  {split_name:<5}: {len(split_df):>5}  |  {row_s}")
    print(f"Weights: { {c.split()[-1]: v for c, v in class_weights.items()} }")
    print(f"Mean: {[round(x, 4) for x in mean]}")
    print(f"Std:  {[round(x, 4) for x in std]}")
    print(f"{'─'*60}")

    return {
        "data_dir":      str(out),
        "class_mapping": str(out / "class_mapping.json"),
        "class_weights": str(out / "class_weights.json"),
        "dataset_stats": str(out / "dataset_stats.json"),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",           default=RAW_DATA_DIR,  help="Path to raw dataset dir")
    ap.add_argument("--out",           default=OUTPUT_DIR,    help="Output dir for splits")
    ap.add_argument("--max-per-class", default=MAX_PER_CLASS, type=int, help="Max images per class (default: %(default)s)")
    ap.add_argument("--seed",          default=SEED,          type=int)
    a = ap.parse_args()
    prepare(raw_data_dir=a.raw, output_dir=a.out, max_per_class=a.max_per_class, seed=a.seed)
