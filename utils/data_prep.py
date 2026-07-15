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

RAW_DATA_DIR  = "../HM_Data/raw_data"
OUTPUT_DIR    = "data/local_sample"  # output dir for train/val/test splits
MAX_PER_CLASS = 50
SEED          = 42

MIN_IMG_SIZE          = 128   # drop images smaller than this in either dimension
DEDUP_HAMMING_THRESH  = 4     # hamming distance ≤ this = near-duplicate (0 = exact only)
RESIZE_PAD_SIZE       = 256   # resize+pad images to this square before writing to disk

CLASSES = [
    "Garment_Upper_body",
    "Garment_Lower_body",
    "Garment_Full_body",
    "Accessories",
    "Shoes",
]

LABEL_COL = "product_group_name"
ID_COL    = "article_id"

# ══════════════════════════════════════════════════════════════════════════════

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _img_path(images_dir: Path, article_id: str) -> Path:
    padded = article_id.zfill(10)
    return images_dir / padded[:3] / f"{padded}.jpg"


def _dhash(img: Image.Image, size: int = 8) -> int:
    gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr  = np.array(gray)
    diff = arr[:, 1:] > arr[:, :-1]
    return int(np.packbits(diff.flatten()).tobytes().hex(), 16)


def _hamming_min(h: int, seen: list) -> int:
    """Min hamming distance from h to any hash in seen. Vectorized via numpy."""
    if not seen:
        return 999
    arr  = np.array(seen, dtype=np.uint64) ^ np.uint64(h)
    bits = np.unpackbits(arr.view(np.uint8)).reshape(len(seen), -1).sum(axis=1)
    return int(bits.min())


def _is_valid(path: Path, min_size: int = MIN_IMG_SIZE) -> bool:
    """PIL verify + minimum resolution check."""
    try:
        with Image.open(path) as im:
            im.verify()
        # re-open after verify (verify closes the file)
        with Image.open(path) as im:
            w, h = im.size
            if w < min_size or h < min_size:
                return False
        return True
    except Exception:
        return False


def _resize_pad(img: Image.Image, size: int = RESIZE_PAD_SIZE) -> Image.Image:
    """Resize preserving aspect ratio, then pad to square with white."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    padded = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    padded.paste(img, offset)
    return padded


def _compute_mean_std(paths: list, sample_n: int = 2000) -> tuple:
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


def _audit_labels(df: pd.DataFrame) -> None:
    """Print top subcategories per class so user can spot mislabeled clusters."""
    sub_col = next((c for c in ["product_type_name", "garment_group_name", "index_group_name"]
                    if c in df.columns), None)
    if sub_col is None:
        print("[audit] No subcategory column found — skipping label audit.")
        return
    print(f"\n[audit] Top subcategories per class (column: {sub_col})")
    print("  Review these — noisy subcategories = mislabeled data")
    for cls in CLASSES:
        counts = df[df[LABEL_COL] == cls][sub_col].value_counts().head(5)
        print(f"\n  {cls}:")
        for subcat, n in counts.items():
            print(f"    {subcat:<40} {n:>5}")


def prepare(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir:   Path = OUTPUT_DIR,
    max_per_class: int = MAX_PER_CLASS,
    seed: int = SEED,
) -> dict:
    images_dir = Path(raw_data_dir) / "images"
    out = Path(output_dir)

    # ── 1. load CSV, filter to target classes ────────────────────────────────
    df = pd.read_csv(Path(raw_data_dir) / "articles.csv", dtype=str)
    df[LABEL_COL] = df[LABEL_COL].str.replace(" ", "_")
    df = df[df[LABEL_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"Rows in {len(CLASSES)} classes: {len(df)}")
    print(df[LABEL_COL].value_counts().to_string())

    # ── 2. label audit — spot mislabeled subcategory clusters ────────────────
    _audit_labels(df)

    # ── 3. resolve paths, keep only those on disk ────────────────────────────
    df["_path"] = df[ID_COL].apply(lambda aid: _img_path(images_dir, aid))
    df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"\nImages on disk: {len(df)}")

    # ── 4. validate — PIL can open + minimum resolution (MIN_IMG_SIZE px) ────
    print(f"Validating images (min size {MIN_IMG_SIZE}px)...")
    valid = [_is_valid(p) for p in tqdm(df["_path"], leave=False)]
    n_dropped = valid.count(False)
    df = df[valid].reset_index(drop=True)
    print(f"Corrupt/tiny dropped: {n_dropped}  →  {len(df)} remain")

    # ── 5. near-duplicate removal via hamming distance on dhash ──────────────
    # hamming ≤ DEDUP_HAMMING_THRESH treated as duplicate (catches watermark variants)
    print(f"Deduplicating (hamming threshold={DEDUP_HAMMING_THRESH})...")
    seen_hashes: list = []
    keep = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        try:
            with Image.open(row["_path"]) as im:
                h = _dhash(im)
        except Exception:
            keep.append(False)
            continue
        if _hamming_min(h, seen_hashes) > DEDUP_HAMMING_THRESH:
            seen_hashes.append(h)
            keep.append(True)
        else:
            keep.append(False)
    n_dupes = keep.count(False)
    df = df[keep].reset_index(drop=True)
    print(f"Near-duplicates removed: {n_dupes}  →  {len(df)} remain")

    # ── 6. cap per class for balance ─────────────────────────────────────────
    df = (df.groupby(LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
            .reset_index(drop=True))
    print(f"\nAfter cap ({max_per_class}/class):")
    print(df[LABEL_COL].value_counts().to_string())

    # ── 7. product-level stratified 80/10/10 split ───────────────────────────
    # Derive product_id from article_id (first 7 chars in H&M).
    # Split by product so the same product never appears in both train and test.
    df["_product_id"] = df[ID_COL].str[:7]
    train_rows, val_rows, test_rows = [], [], []
    for cls in CLASSES:
        cls_df = df[df[LABEL_COL] == cls]
        products = cls_df["_product_id"].unique()
        if len(products) < 3:
            # too few products to split — fall back to row-level split
            tr, tmp = train_test_split(cls_df, test_size=0.2, random_state=seed)
            v, te = train_test_split(tmp, test_size=0.5, random_state=seed)
        else:
            tr_prod, tmp_prod = train_test_split(products, test_size=0.2, random_state=seed)
            v_prod, te_prod   = train_test_split(tmp_prod, test_size=0.5, random_state=seed)
            tr = cls_df[cls_df["_product_id"].isin(tr_prod)]
            v  = cls_df[cls_df["_product_id"].isin(v_prod)]
            te = cls_df[cls_df["_product_id"].isin(te_prod)]
        train_rows.append(tr)
        val_rows.append(v)
        test_rows.append(te)

    train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    # sanity: no product_id overlap between splits
    tr_p  = set(train_df["_product_id"])
    val_p = set(val_df["_product_id"])
    te_p  = set(test_df["_product_id"])
    leaks = len(tr_p & val_p) + len(tr_p & te_p) + len(val_p & te_p)
    print(f"\nProduct-level split — train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Product leakage across splits: {leaks} (should be 0)")

    # ── 8. resize+pad and copy into split/class dirs ─────────────────────────
    print(f"\nResizing to {RESIZE_PAD_SIZE}×{RESIZE_PAD_SIZE} (aspect-ratio preserving pad) and copying...")
    for split_name, split_df in splits:
        for cls in CLASSES:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name, leave=False):
            dst = out / split_name / row[LABEL_COL] / row["_path"].name
            if not dst.exists():
                try:
                    with Image.open(row["_path"]) as im:
                        _resize_pad(im, RESIZE_PAD_SIZE).save(dst, "JPEG", quality=95)
                except Exception:
                    shutil.copy2(row["_path"], dst)  # fallback: copy as-is

    # ── 9. class mapping + inverse-frequency weights ──────────────────────────
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

    # ── 10. dataset mean/std ──────────────────────────────────────────────────
    print("Computing dataset mean/std...")
    train_paths = list(train_df["_path"])
    mean, std   = _compute_mean_std(train_paths)

    dataset_stats = {
        "mean": mean,
        "std":  std,
        "resize_pad_size": RESIZE_PAD_SIZE,
        "note": f"Images stored as {RESIZE_PAD_SIZE}×{RESIZE_PAD_SIZE} JPEG (aspect-ratio preserving, white pad). Mean/std on 224x224 RGB [0,1].",
    }
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # ── 11. summary ───────────────────────────────────────────────────────────
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
    ap.add_argument("--raw",           default=RAW_DATA_DIR)
    ap.add_argument("--out",           default=OUTPUT_DIR)
    ap.add_argument("--max-per-class", default=MAX_PER_CLASS, type=int)
    ap.add_argument("--seed",          default=SEED, type=int)
    a = ap.parse_args()
    prepare(raw_data_dir=a.raw, output_dir=a.out, max_per_class=a.max_per_class, seed=a.seed)
