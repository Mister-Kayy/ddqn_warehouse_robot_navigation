
from __future__ import annotations
from collections import deque
import numpy as np
import gymnasium as gym
import minigrid
from gymnasium import spaces

class WarehouseRewardWrapper(gym.Wrapper):
    """Replace MiniGrid's built-in reward with the project's declared reward.

    DynamicObstacles uses a negative reward on collision, positive reward on goal,
    and zero on ordinary steps/timeouts. We preserve the environment dynamics and
    termination logic but map those events to fixed, interpretable weights.
    """

    def __init__(self, env, success_reward=10.0, collision_penalty=-10.0, step_penalty=-0.01):
        super().__init__(env)
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.step_penalty = float(step_penalty)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        success = bool(terminated and base_reward > 0)
        collision = bool(terminated and base_reward < 0)
        reward = self.step_penalty
        if success:
            reward += self.success_reward
        if collision:
            reward += self.collision_penalty
        info = dict(info)
        info.update(
            success=success,
            collision=collision,
            base_minigrid_reward=float(base_reward),
        )
        return obs, float(reward), terminated, truncated, info


class SymbolicFrameStack(gym.Wrapper):

    """Stack normalized symbolic MiniGrid images and append direction one-hot.

    MiniGrid images encode each visible tile by (object, colour, state). Four
    recent frames help expose short-term obstacle motion in this partially
    observable task.
    """

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = int(n_frames)
        self.frames = deque(maxlen=self.n_frames)
        image_space = env.observation_space["image"]
        self.h, self.w, self.c = image_space.shape
        obs_dim = self.h * self.w * self.c * self.n_frames + 4
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    @staticmethod
    def _normalize_image(image):
        x = image.astype(np.float32).copy()
        # Conservative MiniGrid symbolic scales. Clipping protects against future additions.
        x[..., 0] = np.clip(x[..., 0] / 12.0, 0.0, 1.0)  # object index
        x[..., 1] = np.clip(x[..., 1] / 6.0, 0.0, 1.0)   # colour index
        x[..., 2] = np.clip(x[..., 2] / 2.0, 0.0, 1.0)   # state
        return x

    def _encode(self, obs):
        stacked = np.concatenate(list(self.frames), axis=-1).reshape(-1)
        direction = np.zeros(4, dtype=np.float32)
        direction[int(obs["direction"])] = 1.0
        return np.concatenate([stacked, direction]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._normalize_image(obs["image"])
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(frame.copy())
        return self._encode(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._normalize_image(obs["image"]))
        return self._encode(obs), reward, terminated, truncated, info


def make_env(config, render_mode=None):
    env = gym.make(
        config.env_id,
        render_mode=render_mode,
        max_steps=config.max_episode_steps,
    )
    env = WarehouseRewardWrapper(
        env,
        success_reward = config.success_reward,
        collision_penalty = config.collision_penalty,
        step_penalty = config.step_penalty,
    )
    env = SymbolicFrameStack(env, n_frames = config.frame_stack)
    return env
