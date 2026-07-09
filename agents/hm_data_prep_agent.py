import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.llm_api import chat as _dial_chat

_CLASS_DECISION_PROMPT = """\
You are an ML engineer selecting classes for an image classification task.

Dataset: H&M fashion articles ({total} total articles)
Label column: product_group_name
Distribution (class: image count):
{distribution}

Select 3–6 classes that:
1. Have at least 500 images each (more = better)
2. Are visually distinct from each other
3. Cover meaningful fashion categories worth classifying

Return ONLY valid JSON — no other text:
{{"classes": ["Class A", "Class B", ...], "rationale": "one sentence why these classes"}}
"""


def _img_path(images_dir: Path, article_id: str) -> Path:
    return images_dir / article_id[:3] / f"{article_id}.jpg"


def prepare_data(
    raw_data_dir: str,
    output_dir: str = "data/auto",
    max_train_per_class: int | None = None,
    seed: int = 42,
) -> dict:
    """
    LLM-driven data prep: decide classes → split → copy images → write metadata.
    Returns dict of paths for training_config.yaml.
    """
    raw = Path(raw_data_dir)
    images_dir = raw / "images"
    out = Path(output_dir)

    # 1. load CSV, count per class
    df = pd.read_csv(raw / "articles.csv", dtype=str)
    dist = df["product_group_name"].value_counts()
    dist_str = "\n".join(f"  {k}: {v}" for k, v in dist.items())
    print(f"Product group distribution:\n{dist_str}\n")

    # 2. LLM decides classes
    resp = _dial_chat(_CLASS_DECISION_PROMPT.format(
        total=len(df), distribution=dist_str
    ))
    match = re.search(r"\{.*\}", resp, re.DOTALL)
    decision = json.loads(match.group(0))
    classes: list[str] = decision["classes"]
    print(f"LLM selected: {classes}")
    print(f"Rationale: {decision['rationale']}\n")

    # 3. filter to selected classes + image exists
    df = df[df["product_group_name"].isin(classes)].copy()
    df = df[df["article_id"].apply(lambda aid: _img_path(images_dir, aid).exists())]
    print(f"{len(df)} articles with images found for selected classes.")

    # 4. stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["product_group_name"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["product_group_name"], random_state=seed
    )

    # cap training set if requested
    if max_train_per_class:
        train_df = (
            train_df.groupby("product_group_name", group_keys=False)
            .apply(lambda g: g.head(max_train_per_class))
            .reset_index(drop=True)
        )

    # 5. copy images into split/class dirs
    print("Copying images...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for cls in classes:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for _, row in split_df.iterrows():
            src = _img_path(images_dir, row["article_id"])
            dst = out / split_name / row["product_group_name"] / f"{row['article_id']}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)

    # 6. class mapping + inverse-frequency weights
    class_names = sorted(classes)
    class_mapping = {str(i): cls for i, cls in enumerate(class_names)}
    train_counts = Counter(train_df["product_group_name"])
    total_train = sum(train_counts.values())
    n = len(class_names)
    class_weights = {cls: (total_train / (n * train_counts[cls])) for cls in class_names}

    mapping_path = "data/class_mapping_auto.json"
    weights_path = "data/class_weights_auto.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)

    # summary
    print(f"\n{'─'*60}")
    print(f"Data prepared → {out}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = Counter(split_df["product_group_name"])
        dist_line = "  ".join(f"{cls}={counts.get(cls, 0)}" for cls in class_names)
        print(f"  {split_name:<5}: {len(split_df):>6} imgs  |  {dist_line}")
    print(f"Class weights: { {c: round(w, 3) for c, w in class_weights.items()} }")
    print(f"{'─'*60}\n")

    return {
        "data_dir": str(out),
        "class_mapping": mapping_path,
        "class_weights": weights_path,
    }
