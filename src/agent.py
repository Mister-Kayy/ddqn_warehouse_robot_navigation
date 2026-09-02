
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn

from .network import QNetwork


class DQNAgent:

    """One implementation supporting Standard DQN and Double DQN targets."""

    def __init__(self, obs_dim, n_actions, config, double_dqn = False, device = None):
        self.n_actions = int(n_actions)
        self.gamma = config.gamma
        self.double_dqn = bool(double_dqn)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(obs_dim, n_actions, config.hidden_dim).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr = config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.grad_clip = config.gradient_clip_norm

    @torch.no_grad()
    def act(self, state, epsilon = 0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim = 1).item())

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                # DDQN: online network selects the action; target network evaluates it.
                next_actions = self.online(next_states).argmax(dim = 1, keepdim = True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                # DQN: target network both selects and evaluates via max.
                next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none = True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item()), float(q.detach().mean().item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, metadata=None):
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "double_dqn": self.double_dqn,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        return ckpt
