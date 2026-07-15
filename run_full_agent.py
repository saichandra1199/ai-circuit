"""
Full Agent workflow: fully autonomous.
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
    llm_model = agent.get("llm_model", "gpt-4o-mini")
    exp_name = agent.get("experiment_name") or ""
    dp_cfg = cfg.get("data_prep", {})

    print("=" * 60)
    print("FULL AGENT — Autonomous data prep + training")
    print("=" * 60)

    # step 1: initial data preparation
    data_dir_name = exp_name.replace(" ", "_") if exp_name else "auto"
    data_output_dir = f"data/{data_dir_name}"
    data_paths = prepare_data(
        raw_data_dir=dp_cfg.get("raw_data_dir", "../raw_data"),
        output_dir=data_output_dir,
        max_train_per_class=dp_cfg.get("max_train_per_class"),
        llm_model=llm_model,
        instructions=dp_cfg.get("instructions"),
        force_classes=dp_cfg.get("force_classes"),
    )

    # step 2: patch base config with data paths (training_config.yaml is never written back)
    cfg["paths"]["data_dir"] = data_paths["data_dir"]
    cfg["paths"]["class_mapping"] = data_paths["class_mapping"]
    cfg["paths"]["class_weights"] = data_paths["class_weights"]
    cfg.get("paths", {}).pop("output_dir", None)
    cfg.setdefault("experiment", {})["notes"] = "Full Agent — autonomous data prep + training."

    # step 3: run training agent loop
    # dp_cfg passed so agent can modify data prep in-memory; source file is never touched
    run_training(
        config_path=cfg,
        max_iterations=max_iterations,
        target_f1=target_f1,
        workflow="full_agent",
        data_prep_config=dp_cfg,
        data_prep_output_dir=data_output_dir,
    )
