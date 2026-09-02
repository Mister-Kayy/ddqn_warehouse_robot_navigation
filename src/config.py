
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen = True)
class ExperimentConfig:
    env_id: str = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    frame_stack: int = 4
    total_steps: int = 100_000
    max_episode_steps: int = 256
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    hidden_dim: int = 256
    gradient_clip_norm: float = 10.0
    success_reward: float = 10.0
    collision_penalty: float = -10.0
    step_penalty: float = -0.01
    train_seeds: tuple[int, ...] = (42, 123, 2026)
    eval_episodes: int = 30
    eval_seed_start: int = 10_000

    def to_dict(self):
        d = asdict(self)
        d["train_seeds"] = list(self.train_seeds)
        return d

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(self.to_dict(), indent = 2), encoding = "utf-8")
