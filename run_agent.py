import argparse
from agents.hm_training_agent import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    ap.add_argument("--max-iterations", type=int, default=5)
    ap.add_argument("--target-f1", type=float, default=0.75)
    args = ap.parse_args()

    run(config_path=args.config, max_iterations=args.max_iterations, target_f1=args.target_f1)
