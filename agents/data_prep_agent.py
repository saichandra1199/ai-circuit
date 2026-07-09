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

Dataset: image classification dataset ({total} total items)
Label column: {label_col}
Distribution (class: image count):
{distribution}

Select 3–6 classes that:
1. Have at least 500 images each (more = better)
2. Are visually distinct from each other
3. Cover meaningful categories worth classifying

Return ONLY valid JSON — no other text:
{{"classes": ["Class A", "Class B", ...], "rationale": "one sentence why these classes"}}
"""

_ANALYSIS_PROMPT = """\
You are an ML engineer writing notes about a dataset you just prepared for image classification.

Raw dataset stats:
- Total items in CSV: {total_raw}
- Total unique classes: {total_classes}
- Items with missing images: {missing_count} ({missing_pct:.1f}%)
- Items available for selected classes (after image filter): {items_available}
- Items actually used after capping (max_train_per_class={max_train_per_class}): {items_used}

All available classes (name: count):
{all_classes}

Selected classes: {selected_classes}
Selection rationale: {rationale}

Final split counts per class (train | val | test):
{split_table}

Class weights (inverse frequency for loss weighting):
{weights_table}

Imbalance ratio (largest / smallest class in train): {imbalance_ratio:.2f}x

Write Markdown notes with EXACTLY these 3 sections (no preamble, no other text):

## Dataset Overview
Size, class composition, coverage of selected classes vs full dataset.

## Data Quality
Flag concerns: missing images rate, class imbalance severity, any classes too small, potential label noise risk.

## Preparation Decisions
Why these classes were selected, what to watch during training based on the distribution.
"""


def _cap(df, label_col, n):
    return (df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.head(n))
              .reset_index(drop=True))


def _img_path(images_dir: Path, article_id: str) -> Path:
    padded = article_id.zfill(10)  # CSV omits leading zero; filenames are always 10 digits
    return images_dir / padded[:3] / f"{padded}.jpg"


def prepare_data(
    raw_data_dir: str,
    output_dir: str = "data/auto",
    max_train_per_class: int | None = None,
    seed: int = 42,
    label_col: str = "product_group_name",
    id_col: str = "article_id",
    llm_model: str = "gpt-4o-mini",
    instructions: str | None = None,
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
    total_raw = len(df)
    dist = df[label_col].value_counts()
    dist_str = "\n".join(f"  {k}: {v}" for k, v in dist.items())
    print(f"Label distribution:\n{dist_str}\n")

    # 2. LLM decides classes
    prompt = _CLASS_DECISION_PROMPT.format(
        total=len(df), label_col=label_col, distribution=dist_str
    )
    if instructions:
        prompt += f"\nAdditional human instructions: {instructions}\n"
    resp = _dial_chat(prompt, model=llm_model)
    match = re.search(r"\{.*\}", resp, re.DOTALL)
    decision = json.loads(match.group(0))
    classes: list[str] = decision["classes"]
    print(f"LLM selected: {classes}")
    print(f"Rationale: {decision['rationale']}\n")

    # 3. filter to selected classes + image exists
    df = df[df[label_col].isin(classes)].copy()
    before_filter = len(df)
    df = df[df[id_col].apply(lambda aid: _img_path(images_dir, aid).exists())]
    missing_count = before_filter - len(df)
    print(f"{len(df)} items with images found ({missing_count} missing images).")

    # 4. stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=seed
    )

    # cap splits preserving 80/10/10 ratio
    if max_train_per_class:
        eval_cap = max(1, max_train_per_class // 8)
        train_df = _cap(train_df, label_col, max_train_per_class)
        val_df   = _cap(val_df,   label_col, eval_cap)
        test_df  = _cap(test_df,  label_col, eval_cap)

    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    # 5. copy images into split/class dirs
    print("Copying images...")
    for split_name, split_df in splits:
        for cls in classes:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for _, row in split_df.iterrows():
            src = _img_path(images_dir, row[id_col])
            dst = out / split_name / row[label_col] / f"{row[id_col]}.jpg"
            if not dst.exists():
                shutil.copy2(src, dst)

    # 6. class mapping + inverse-frequency weights
    class_names = sorted(classes)
    class_mapping = {str(i): cls for i, cls in enumerate(class_names)}
    train_counts = Counter(train_df[label_col])
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
    split_counts = {}
    for split_name, split_df in splits:
        counts = Counter(split_df[label_col])
        split_counts[split_name] = counts
        dist_line = "  ".join(f"{cls}={counts.get(cls, 0)}" for cls in class_names)
        print(f"  {split_name:<5}: {len(split_df):>6} imgs  |  {dist_line}")
    print(f"Class weights: { {c: round(w, 3) for c, w in class_weights.items()} }")
    print(f"{'─'*60}\n")

    # LLM analysis notes
    print("Generating data prep notes...")
    split_table = "\n".join(
        f"  {cls:<30} {split_counts['train'].get(cls, 0):>5} | {split_counts['val'].get(cls, 0):>4} | {split_counts['test'].get(cls, 0):>4}"
        for cls in class_names
    )
    weights_table = "\n".join(f"  {cls}: {w:.3f}" for cls, w in class_weights.items())
    train_counts_vals = [train_counts[c] for c in class_names]
    imbalance_ratio = max(train_counts_vals) / max(1, min(train_counts_vals))

    total_used = sum(sum(c.values()) for c in split_counts.values())
    notes = _dial_chat(_ANALYSIS_PROMPT.format(
        total_raw=total_raw,
        total_classes=len(dist),
        missing_count=missing_count,
        missing_pct=100 * missing_count / max(1, before_filter),
        items_available=len(df),
        items_used=total_used,
        max_train_per_class=max_train_per_class or "unlimited",
        all_classes="\n".join(f"  {k}: {v}" for k, v in dist.items()),
        selected_classes=", ".join(class_names),
        rationale=decision["rationale"],
        split_table=split_table,
        weights_table=weights_table,
        imbalance_ratio=imbalance_ratio,
    ), model=llm_model)

    header = f"# Data Prep Notes\n\n**Human instructions:** {instructions or 'none'}\n\n"
    notes_path = out / "data_prep_notes.md"
    notes_path.write_text(header + notes)
    print(f"Data prep notes → {notes_path}\n")

    return {
        "data_dir": str(out),
        "class_mapping": mapping_path,
        "class_weights": weights_path,
    }
