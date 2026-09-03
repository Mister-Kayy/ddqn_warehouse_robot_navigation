from __future__ import annotations
from pathlib import Path
import csv
import time
import numpy as np
import torch

from .agent import DQNAgent
from .env import make_env
from .replay_buffer import ReplayBuffer
from .utils import set_global_seed, linear_epsilon


def train_one(config, algorithm: str, seed: int, root: str | Path):
    algorithm = algorithm.lower()
    if algorithm not in {"dqn", "ddqn"}:
        raise ValueError("algorithm must be 'dqn' or 'ddqn'")
    set_global_seed(seed)
    root = Path(root)
    log_dir = root / "logs" / algorithm
    model_dir = root / "models" / algorithm
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(obs_dim, n_actions, config, double_dqn=(algorithm == "ddqn"), device=device)
    replay = ReplayBuffer(config.replay_capacity, obs_dim, device)

    csv_path = log_dir / f"seed_{seed}.csv"
    fields = [
        "algorithm", "seed", "episode", "global_step", "episode_return",
        "episode_steps", "success", "collision", "epsilon", "mean_loss", "mean_q"
    ]
    episode = 0
    ep_return = 0.0
    ep_steps = 0
    losses, qmeans = [], []
    start_time = time.time()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, config.total_steps + 1):
            epsilon = linear_epsilon(step, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            ep_steps += 1

            if replay.size >= max(config.learning_starts, config.batch_size) and step % config.train_frequency == 0:
                loss, qmean = agent.update(replay.sample(config.batch_size))
                losses.append(loss)
                qmeans.append(qmean)

            if step % config.target_update_frequency == 0:
                agent.sync_target()

            if done:
                episode += 1
                writer.writerow({
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": episode,
                    "global_step": step,
                    "episode_return": ep_return,
                    "episode_steps": ep_steps,
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "epsilon": epsilon,
                    "mean_loss": np.mean(losses) if losses else "",
                    "mean_q": np.mean(qmeans) if qmeans else "",
                })
                f.flush()
                obs, _ = env.reset()
                ep_return, ep_steps = 0.0, 0
                losses.clear(); qmeans.clear()

    model_path = model_dir / f"seed_{seed}.pt"
    agent.save(model_path, metadata={
        "algorithm": algorithm,
        "seed": seed,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "elapsed_seconds": time.time() - start_time,
        "config": config.to_dict(),
    })
    env.close()
    return csv_path, model_path
