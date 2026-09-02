
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import torch

from .agent import DQNAgent
from .env import make_env
from .utils import set_global_seed


def evaluate_one(config, algorithm: str, train_seed: int, root: str | Path):
    root = Path(root)
    set_global_seed(train_seed)
    env = make_env(config)
    obs, _ = env.reset(seed=config.eval_seed_start)
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = DQNAgent(obs_dim, env.action_space.n, config, double_dqn=(algorithm == "ddqn"))
    model_path = root / "models" / algorithm / f"seed_{train_seed}.pt"
    agent.load(model_path)
    agent.online.eval()

    out_dir = root / "logs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algorithm}_seed_{train_seed}.csv"
    fields = ["algorithm", "train_seed", "episode", "episode_seed", "return", "steps", "success", "collision"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for ep in range(config.eval_episodes):
            ep_seed = config.eval_seed_start + ep
            obs, _ = env.reset(seed=ep_seed)
            total_reward, steps = 0.0, 0
            success = collision = False
            while True:
                action = agent.act(obs, epsilon=0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward; steps += 1
                success = success or bool(info.get("success", False))
                collision = collision or bool(info.get("collision", False))
                if terminated or truncated:
                    break
            writer.writerow({
                "algorithm": algorithm, "train_seed": train_seed, "episode": ep + 1,
                "episode_seed": ep_seed, "return": total_reward, "steps": steps,
                "success": int(success), "collision": int(collision),
            })
    env.close()
    return out_path


def summarize_evaluation(root: str | Path):
    root = Path(root)
    files = sorted((root / "logs" / "evaluation").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No evaluation CSV files found. Run evaluation first.")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    seed_metrics = df.groupby(["algorithm", "train_seed"]).apply(
        lambda g: pd.Series({
            "task_completion_rate": g["success"].mean(),
            "collision_rate": g["collision"].mean(),
            "average_steps": g["steps"].mean(),
            "average_steps_success": g.loc[g["success"] == 1, "steps"].mean(),
            "average_return": g["return"].mean(),
        }), include_groups=False
    ).reset_index()
    summary = seed_metrics.groupby("algorithm").agg(
        task_completion_rate_mean=("task_completion_rate", "mean"),
        task_completion_rate_sd=("task_completion_rate", "std"),
        collision_rate_mean=("collision_rate", "mean"),
        collision_rate_sd=("collision_rate", "std"),
        average_steps_mean=("average_steps", "mean"),
        average_steps_sd=("average_steps", "std"),
        average_return_mean=("average_return", "mean"),
        average_return_sd=("average_return", "std"),
    ).reset_index()
    seed_metrics.to_csv(root / "logs" / "evaluation_seed_metrics.csv", index=False)
    summary.to_csv(root / "logs" / "evaluation_summary.csv", index=False)
    return seed_metrics, summary
