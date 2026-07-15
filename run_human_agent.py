import argparse
import yaml
from agents.training_agent import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    agent = cfg.get("agent", {})
    run(
        config_path=args.config,
        max_iterations=agent.get("max_iterations", 5),
        target_f1=agent.get("target_f1", 0.75),
    )
