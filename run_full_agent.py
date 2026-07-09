"""
Workflow 2: Fully autonomous agent.
LLM decides classes + data splits, then trains.
"""
import argparse
import yaml

from agents.data_prep_agent import prepare_data
from agents.training_agent import run as run_training


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    agent = cfg.get("agent", {})
    max_iterations = agent.get("max_iterations", 5)
    target_f1 = agent.get("target_f1", 0.75)
    raw_data_dir = agent.get("raw_data_dir", "../raw_data")
    max_train_per_class = agent.get("max_train_per_class", None)

    print("=" * 60)
    print("WORKFLOW 2 — Autonomous data prep + training")
    print("=" * 60)

    # step 1: LLM-driven data preparation
    data_paths = prepare_data(
        raw_data_dir=raw_data_dir,
        output_dir="data/auto",
        max_train_per_class=max_train_per_class,
    )

    # step 2: patch base config with agent-chosen data paths + workflow note
    cfg["paths"]["data_dir"] = data_paths["data_dir"]
    cfg["paths"]["class_mapping"] = data_paths["class_mapping"]
    cfg["paths"]["class_weights"] = data_paths["class_weights"]
    cfg.get("paths", {}).pop("output_dir", None)
    cfg.setdefault("experiment", {})["notes"] = "Workflow 2 — autonomous data prep + training."

    auto_config_path = "data/auto_training_config.yaml"
    with open(auto_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Auto config written → {auto_config_path}\n")

    # step 3: run training agent loop
    run_training(
        config_path=auto_config_path,
        max_iterations=max_iterations,
        target_f1=target_f1,
    )
