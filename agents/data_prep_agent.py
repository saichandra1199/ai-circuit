import json
import re
import shutil
from pathlib import Path

import pandas as pd

from agents.prompts import CLASS_DECISION_PROMPT, DATASET_ANALYSIS_PROMPT, PIPELINE_CONFIG_PROMPT
from utils.llm_api import chat as _dial_chat
from utils.data_prep import _img_path
from utils.data_prep_pipeline import run_pipeline, DEFAULT_PIPELINE_CONFIG


def _analyze_data_stats(
    raw_data_dir: str,
    classes: list[str],
    label_col: str,
    id_col: str,
) -> dict:
    """Fast stats pass — no image loading, just file existence checks."""
    raw = Path(raw_data_dir)
    images_dir = raw / "images"

    df = pd.read_csv(raw / "articles.csv", dtype=str)
    df[label_col] = df[label_col].str.replace(" ", "_")
    df = df[df[label_col].isin(classes)].reset_index(drop=True)

    total = len(df)
    class_counts = df[label_col].value_counts().to_dict()

    has_image = df[id_col].apply(lambda aid: _img_path(images_dir, aid).exists())
    missing_count = (~has_image).sum()
    missing_pct = 100 * missing_count / max(1, total)

    counts_vals = list(class_counts.values())
    imbalance_ratio = max(counts_vals) / max(1, min(counts_vals)) if counts_vals else 1.0

    return {
        "total_items": int(has_image.sum()),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "missing_count": int(missing_count),
        "missing_pct": float(missing_pct),
        "imbalance_ratio": float(imbalance_ratio),
    }


def _decide_pipeline_config(
    stats: dict,
    max_train_per_class: int | None,
    llm_model: str,
    instructions: str | None = None,
) -> dict:
    """Ask LLM to choose pipeline steps based on data stats. Falls back to DEFAULT on parse error."""
    dist_table = "\n".join(
        f"  {cls}: {count}" for cls, count in stats["class_counts"].items()
    )
    selected_classes = ", ".join(stats["class_counts"].keys())

    prompt = PIPELINE_CONFIG_PROMPT.format(
        total_items=stats["total_items"],
        selected_classes=selected_classes,
        dist_table=dist_table,
        missing_count=stats["missing_count"],
        missing_pct=stats["missing_pct"],
        imbalance_ratio=stats["imbalance_ratio"],
        max_train_per_class=max_train_per_class or "null",
    )
    if instructions:
        prompt += f"\nAdditional instructions: {instructions}\n"

    raw = _dial_chat(prompt, model=llm_model)

    if "```" in raw:
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        raw = match.group(1).strip() if match else raw

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        cfg = json.loads(match.group(0))
        print(f"Pipeline config decided by LLM:\n{json.dumps(cfg, indent=2)}\n")
        return cfg
    except Exception:
        print("[WARN] Could not parse LLM pipeline config — using DEFAULT_PIPELINE_CONFIG")
        return DEFAULT_PIPELINE_CONFIG


def prepare_data(
    raw_data_dir: str,
    output_dir: str = "data/auto",
    max_train_per_class: int | None = None,
    seed: int = 42,
    label_col: str = "product_group_name",
    id_col: str = "article_id",
    llm_model: str = "gpt-4o-mini",
    instructions: str | None = None,
    force_classes: list | None = None,
) -> dict:
    """
    LLM-driven data prep:
      1. Decide classes (LLM or force_classes)
      2. Analyze data stats
      3. LLM decides which pipeline steps to run
      4. run_pipeline() executes the steps
      5. Generate data prep notes
    Returns dict of paths for training_config.yaml.
    """
    raw = Path(raw_data_dir)
    out = Path(output_dir)

    # wipe output dir so stale class dirs never confuse ImageFolder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. load CSV for class selection
    df = pd.read_csv(raw / "articles.csv", dtype=str)
    df[label_col] = df[label_col].str.replace(" ", "_")
    total_raw = len(df)
    dist = df[label_col].value_counts()
    dist_str = "\n".join(f"  {k}: {v}" for k, v in dist.items())
    print(f"Label distribution:\n{dist_str}\n")

    # 2. class selection — human override or LLM
    if force_classes:
        classes: list[str] = force_classes
        rationale = "Human-specified classes (LLM selection skipped)."
        print(f"Using forced classes: {classes}\n")
    else:
        prompt = CLASS_DECISION_PROMPT.format(
            total=len(df), label_col=label_col, distribution=dist_str
        )
        if instructions:
            prompt += f"\nAdditional human instructions: {instructions}\n"
        resp = _dial_chat(prompt, model=llm_model)
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        decision = json.loads(match.group(0))
        classes = decision["classes"]
        rationale = decision["rationale"]
        print(f"LLM selected: {classes}")
        print(f"Rationale: {rationale}\n")

    # 3. analyze data stats (fast — file existence only, no image loading)
    print("Analyzing data stats...")
    stats = _analyze_data_stats(raw_data_dir, classes, label_col, id_col)
    print(f"  total_items={stats['total_items']}  missing={stats['missing_count']} ({stats['missing_pct']:.1f}%)  imbalance={stats['imbalance_ratio']:.2f}x\n")

    # 4. LLM decides pipeline config
    print("Deciding pipeline config...")
    pipeline_cfg = _decide_pipeline_config(stats, max_train_per_class, llm_model, instructions)

    # ensure max_train_per_class from caller takes precedence if set
    if max_train_per_class is not None:
        pipeline_cfg["max_train_per_class"] = max_train_per_class

    # 5. run pipeline
    data_paths = run_pipeline(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        classes=classes,
        pipeline_config=pipeline_cfg,
        seed=seed,
        label_col=label_col,
        id_col=id_col,
    )

    # 6. reload split counts for notes (pipeline already wrote class_mapping.json)
    with open(data_paths["class_mapping"]) as f:
        class_mapping = json.load(f)
    class_names = list(class_mapping.values())

    split_counts: dict = {}
    for split in ("train", "val", "test"):
        split_dir = out / split
        if split_dir.exists():
            split_counts[split] = {
                cls: len(list((split_dir / cls).glob("*")))
                for cls in class_names
                if (split_dir / cls).exists()
            }

    with open(data_paths["class_weights"]) as f:
        class_weights_data = json.load(f)

    split_table = "\n".join(
        f"  {cls:<30} {split_counts.get('train', {}).get(cls, 0):>5} | "
        f"{split_counts.get('val', {}).get(cls, 0):>4} | "
        f"{split_counts.get('test', {}).get(cls, 0):>4}"
        for cls in class_names
    )
    weights_table = "\n".join(f"  {cls}: {w:.3f}" for cls, w in class_weights_data.items())
    train_counts_vals = [split_counts.get("train", {}).get(c, 0) for c in class_names]
    items_used = sum(sum(s.values()) for s in split_counts.values())
    imbalance_ratio = (max(train_counts_vals) / max(1, min(train_counts_vals))
                       if train_counts_vals else 1.0)

    # 7. LLM data prep notes
    print("Generating data prep notes...")
    notes = _dial_chat(DATASET_ANALYSIS_PROMPT.format(
        total_raw=total_raw,
        total_classes=len(dist),
        missing_count=stats["missing_count"],
        missing_pct=stats["missing_pct"],
        items_available=stats["total_items"],
        items_used=items_used,
        max_train_per_class=pipeline_cfg.get("max_train_per_class") or "unlimited",
        all_classes="\n".join(f"  {k}: {v}" for k, v in dist.items()),
        selected_classes=", ".join(class_names),
        rationale=rationale,
        split_table=split_table,
        weights_table=weights_table,
        imbalance_ratio=imbalance_ratio,
    ), model=llm_model)

    header = (
        f"# Data Prep Notes\n\n"
        f"**Human instructions:** {instructions or 'none'}\n\n"
        f"**Pipeline config:** see `pipeline_config.json` in this directory\n\n"
    )
    notes_path = out / "data_prep_notes.md"
    notes_path.write_text(header + notes)
    print(f"Data prep notes → {notes_path}\n")

    return data_paths
