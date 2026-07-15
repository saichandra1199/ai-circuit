"""
Configurable data prep pipeline for the Full Agent workflow.

All steps are toggled by PIPELINE_CONFIG — the LLM agent in data_prep_agent.py
analyzes data stats and writes a config dict; this file runs whatever it decides.
The config is also saved to output_dir/pipeline_config.json for transparency.

Manual use:
    from utils.data_prep_pipeline import run_pipeline, DEFAULT_PIPELINE_CONFIG
    run_pipeline("../HM_Data/raw_data", "data/out", ["Shoes", "Accessories"], DEFAULT_PIPELINE_CONFIG)
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# reuse helpers from the manual prep script — no duplication
from utils.data_prep import (
    _img_path,
    _is_valid,
    _dhash,
    _hamming_min,
    _resize_pad,
    _compute_mean_std,
)


# ── default pipeline config ────────────────────────────────────────────────────
# LLM agent may override any value; this is the fallback when LLM parse fails.

DEFAULT_PIPELINE_CONFIG = {
    "validate_images":     {"enabled": True,  "min_size": 128},
    "dedup":               {"enabled": True,  "hamming_thresh": 4},
    "resize_pad":          {"enabled": True,  "size": 256},
    "product_level_split": {"enabled": True},
    "compute_mean_std":    {"enabled": False, "sample_n": 2000},
    "max_train_per_class": 50,
    "eval_cap_ratio":      0.125,   # val/test cap = max_train * eval_cap_ratio
}


# ── pipeline steps ─────────────────────────────────────────────────────────────

def _step_validate(df: pd.DataFrame, id_col: str, images_dir: Path, min_size: int) -> pd.DataFrame:
    print(f"  [validate] Checking images (min {min_size}px)...")
    valid = [_is_valid(_img_path(images_dir, aid), min_size) for aid in df[id_col]]
    n_dropped = valid.count(False)
    df = df[valid].reset_index(drop=True)
    print(f"  [validate] Dropped {n_dropped} corrupt/tiny → {len(df)} remain")
    return df


def _step_dedup(df: pd.DataFrame, id_col: str, images_dir: Path, hamming_thresh: int) -> pd.DataFrame:
    print(f"  [dedup] hamming_thresh={hamming_thresh}...")
    seen_hashes: list = []
    keep = []
    for aid in df[id_col]:
        path = _img_path(images_dir, aid)
        try:
            from PIL import Image
            with Image.open(path) as im:
                h = _dhash(im)
        except Exception:
            keep.append(False)
            continue
        if _hamming_min(h, seen_hashes) > hamming_thresh:
            seen_hashes.append(h)
            keep.append(True)
        else:
            keep.append(False)
    n_dupes = keep.count(False)
    df = df[keep].reset_index(drop=True)
    print(f"  [dedup] Removed {n_dupes} near-duplicates → {len(df)} remain")
    return df


def _split_product_level(cls_df: pd.DataFrame, id_col: str, label_col: str, seed: int):
    """Split by product ID (first 7 chars of article_id) to prevent leakage."""
    cls_df = cls_df.copy()
    cls_df["_product_id"] = cls_df[id_col].str[:7]
    products = cls_df["_product_id"].unique()
    if len(products) < 3:
        # fall back to row-level
        tr, tmp = train_test_split(cls_df, test_size=0.2, random_state=seed)
        v, te = train_test_split(tmp, test_size=0.5, random_state=seed)
        return tr.drop(columns=["_product_id"]), v.drop(columns=["_product_id"]), te.drop(columns=["_product_id"])
    tr_prod, tmp_prod = train_test_split(products, test_size=0.2, random_state=seed)
    v_prod, te_prod = train_test_split(tmp_prod, test_size=0.5, random_state=seed)
    tr = cls_df[cls_df["_product_id"].isin(tr_prod)].drop(columns=["_product_id"])
    v  = cls_df[cls_df["_product_id"].isin(v_prod)].drop(columns=["_product_id"])
    te = cls_df[cls_df["_product_id"].isin(te_prod)].drop(columns=["_product_id"])
    return tr, v, te


def _copy_or_resize(src: Path, dst: Path, resize_pad: bool, size: int) -> None:
    from PIL import Image, ImageOps
    try:
        with Image.open(src) as im:
            if resize_pad:
                _resize_pad(im, size).save(dst, "JPEG", quality=95)
            else:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, "JPEG", quality=95)
        return
    except Exception:
        shutil.copy2(src, dst)


# ── main entry point ───────────────────────────────────────────────────────────

def run_pipeline(
    raw_data_dir: str,
    output_dir: str,
    classes: list[str],
    pipeline_config: dict,
    seed: int = 42,
    label_col: str = "product_group_name",
    id_col: str = "article_id",
) -> dict:
    """
    Run the configurable data prep pipeline.

    Args:
        raw_data_dir: path to raw dataset (needs articles.csv + images/)
        output_dir:   where to write train/val/test/<class>/ tree
        classes:      list of class name strings (already normalized, underscores)
        pipeline_config: dict — see DEFAULT_PIPELINE_CONFIG for schema
        seed, label_col, id_col: standard params

    Returns:
        dict with keys: data_dir, class_mapping, class_weights
    """
    cfg = {**DEFAULT_PIPELINE_CONFIG, **pipeline_config}
    raw = Path(raw_data_dir)
    images_dir = raw / "images"
    out = Path(output_dir)

    # merge nested step configs with defaults
    for step in ("validate_images", "dedup", "resize_pad", "product_level_split", "compute_mean_std"):
        default_step = DEFAULT_PIPELINE_CONFIG.get(step, {})
        incoming = pipeline_config.get(step, {})
        if isinstance(default_step, dict) and isinstance(incoming, dict):
            cfg[step] = {**default_step, **incoming}

    print(f"\nPipeline config:")
    print(f"  validate_images:     {cfg['validate_images']}")
    print(f"  dedup:               {cfg['dedup']}")
    print(f"  resize_pad:          {cfg['resize_pad']}")
    print(f"  product_level_split: {cfg['product_level_split']}")
    print(f"  compute_mean_std:    {cfg['compute_mean_std']}")
    print(f"  max_train_per_class: {cfg['max_train_per_class']}")
    print(f"  eval_cap_ratio:      {cfg['eval_cap_ratio']}")

    # 1. load CSV, normalize labels, filter to classes
    df = pd.read_csv(raw / "articles.csv", dtype=str)
    df[label_col] = df[label_col].str.replace(" ", "_")
    df = df[df[label_col].isin(classes)].reset_index(drop=True)

    # 2. keep only rows with image on disk
    before = len(df)
    df = df[df[id_col].apply(lambda aid: _img_path(images_dir, aid).exists())].reset_index(drop=True)
    print(f"\n{len(df)} items with images ({before - len(df)} missing).")

    # 3. cap per class FIRST — dedup/validate on small set, not full 90k
    max_per = cfg.get("max_train_per_class")
    pre_cap = max_per * 4 if max_per else None  # keep 4x buffer so dedup has room to remove
    if pre_cap:
        df = df.groupby(label_col, group_keys=False).head(pre_cap).reset_index(drop=True)

    # 4. validate images
    if cfg["validate_images"]["enabled"]:
        df = _step_validate(df, id_col, images_dir, cfg["validate_images"]["min_size"])

    # 5. dedup (now runs on pre-capped set, not full dataset)
    if cfg["dedup"]["enabled"]:
        df = _step_dedup(df, id_col, images_dir, cfg["dedup"]["hamming_thresh"])

    # 6. final cap per class
    if max_per:
        df = df.groupby(label_col, group_keys=False).head(max_per).reset_index(drop=True)

    # 7. split
    do_product_split = cfg["product_level_split"]["enabled"]
    eval_cap = max(1, int(max_per * cfg["eval_cap_ratio"])) if max_per else None

    train_rows, val_rows, test_rows = [], [], []
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        if len(cls_df) == 0:
            continue
        if do_product_split:
            tr, v, te = _split_product_level(cls_df, id_col, label_col, seed)
        else:
            tr, tmp = train_test_split(cls_df, test_size=0.2, stratify=cls_df[label_col], random_state=seed)
            v, te = train_test_split(tmp, test_size=0.5, stratify=tmp[label_col], random_state=seed)
        train_rows.append(tr)
        val_rows.append(v)
        test_rows.append(te)

    train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    # apply eval cap
    if eval_cap:
        val_df  = val_df.groupby(label_col, group_keys=False).head(eval_cap).reset_index(drop=True)
        test_df = test_df.groupby(label_col, group_keys=False).head(eval_cap).reset_index(drop=True)

    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    # 7. copy / resize images
    do_resize = cfg["resize_pad"]["enabled"]
    resize_size = cfg["resize_pad"].get("size", 256)
    print("Copying images...")
    for split_name, split_df in splits:
        for cls in classes:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for _, row in split_df.iterrows():
            src = _img_path(images_dir, row[id_col])
            dst = out / split_name / row[label_col] / f"{row[id_col]}.jpg"
            if not dst.exists():
                _copy_or_resize(src, dst, do_resize, resize_size)

    # 8. class mapping + weights
    class_names = sorted(train_df[label_col].unique())
    class_mapping = {str(i): cls for i, cls in enumerate(class_names)}
    train_counts = Counter(train_df[label_col])
    total_train = sum(train_counts.values())
    n = len(class_names)
    class_weights = {cls: (total_train / (n * train_counts[cls])) for cls in class_names}

    with open(out / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    with open(out / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    # 9. mean/std
    if cfg["compute_mean_std"]["enabled"]:
        print("Computing mean/std...")
        train_paths = [_img_path(images_dir, aid) for aid in train_df[id_col]]
        mean, std = _compute_mean_std(train_paths, cfg["compute_mean_std"].get("sample_n", 2000))
        with open(out / "dataset_stats.json", "w") as f:
            json.dump({"mean": mean, "std": std, "resize_size": resize_size}, f, indent=2)

    # 10. save pipeline config used (for transparency)
    with open(out / "pipeline_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # 11. summary
    class_names_actual = sorted(train_df[label_col].unique())
    print(f"\n{'─'*60}")
    print(f"Data prepared → {out}")
    for split_name, split_df in splits:
        counts = Counter(split_df[label_col])
        dist_line = "  ".join(f"{cls}={counts.get(cls, 0)}" for cls in class_names_actual)
        print(f"  {split_name:<5}: {len(split_df):>6} imgs  |  {dist_line}")
    print(f"Class weights: { {c: round(w, 3) for c, w in class_weights.items()} }")
    print(f"{'─'*60}\n")

    return {
        "data_dir": str(out),
        "class_mapping": str(out / "class_mapping.json"),
        "class_weights": str(out / "class_weights.json"),
    }
