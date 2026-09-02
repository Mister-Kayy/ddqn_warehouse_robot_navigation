#!/usr/bin/env python
from pathlib import Path
import argparse
import imageio.v2 as imageio
import numpy as np

from src.config import ExperimentConfig
from src.env import make_env
from src.agent import DQNAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["dqn", "ddqn"], default="ddqn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-seed", type=int, default=10000)
    p.add_argument("--out", default="figures/ddqn_rollout.mp4")
    a = p.parse_args()
    root = Path(__file__).resolve().parent; cfg = ExperimentConfig()
    # Keep a raw render env and duplicate the observation wrappers for action selection.
    env = make_env(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=a.episode_seed)
    agent = DQNAgent(int(np.prod(env.observation_space.shape)), env.action_space.n, cfg, double_dqn=(a.algorithm=="ddqn"))
    agent.load(root / "models" / a.algorithm / f"seed_{a.seed}.pt")
    frames = []
    while True:
        frame = env.render()
        if frame is not None: frames.append(frame)
        action = agent.act(obs, epsilon=0.0)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    env.close()
    out = root / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=6)
    print(out)

if __name__ == "__main__": main()
