#!/usr/bin/env python
from pathlib import Path
import argparse

from src.config import ExperimentConfig
from src.train import train_one
from src.evaluate import evaluate_one, summarize_evaluation
from src.plotting import plot_training, plot_evaluation
from src.reporting import print_results


def main():
    parser = argparse.ArgumentParser(description="Reproduce DDQN-2 DQN vs DDQN experiment.")
    parser.add_argument("--quick", action="store_true", help="Smoke run: 5,000 steps and 5 eval episodes.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = ExperimentConfig()
    if args.quick:
        cfg = ExperimentConfig(total_steps=5_000, learning_starts=500, epsilon_decay_steps=3_000, eval_episodes=5)
    cfg.save(root / "logs" / "experiment_config.json")

    for algo in ("dqn", "ddqn"):
        for seed in cfg.train_seeds:
            print(f"Training {algo.upper()} seed={seed}")
            train_one(cfg, algo, seed, root)
    for algo in ("dqn", "ddqn"):
        for seed in cfg.train_seeds:
            print(f"Evaluating {algo.upper()} seed={seed}")
            evaluate_one(cfg, algo, seed, root)
    summarize_evaluation(root)
    plot_training(root); plot_evaluation(root); print_results(root)


if __name__ == "__main__":
    main()
