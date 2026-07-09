"""
Workflow 2: Fully autonomous agent.
LLM decides classes + data splits, then trains.
"""
import argparse
import copy

import yaml

from agents.hm_data_prep_agent import prepare_data
from agents.hm_training_agent import run as run_training


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-data-dir", default="../HM_Data/raw_data",
                    help="Path to directory containing articles.csv and images/")
    ap.add_argument("--max-train-per-class", type=int, default=None,
                    help="Cap training images per class (None = use all)")
    ap.add_argument("--base-config", default="training_config.yaml",
                    help="Base training config to inherit non-data settings from")
    ap.add_argument("--max-iterations", type=int, default=5)
    ap.add_argument("--target-f1", type=float, default=0.75)
    args = ap.parse_args()

    # step 1: LLM-driven data preparation
    print("=" * 60)
    print("WORKFLOW 2 — Autonomous data prep + training")
    print("=" * 60)

    data_paths = prepare_data(
        raw_data_dir=args.raw_data_dir,
        output_dir="data/auto",
        max_train_per_class=args.max_train_per_class,
    )

    # step 2: patch base config with agent-chosen data paths
    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["data_dir"] = data_paths["data_dir"]
    cfg["paths"]["class_mapping"] = data_paths["class_mapping"]
    cfg["paths"]["class_weights"] = data_paths["class_weights"]
    cfg.get("paths", {}).pop("output_dir", None)

    auto_config_path = "data/auto_training_config.yaml"
    with open(auto_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Auto config written → {auto_config_path}\n")

    # step 3: run training agent loop
    run_training(
        config_path=auto_config_path,
        max_iterations=args.max_iterations,
        target_f1=args.target_f1,
    )
