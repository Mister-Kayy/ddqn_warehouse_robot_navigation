
from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_states[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )

