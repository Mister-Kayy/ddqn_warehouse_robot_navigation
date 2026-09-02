import numpy as np
import torch
from types import SimpleNamespace
from src.network import QNetwork
from src.agent import DQNAgent


def _cfg():
    return SimpleNamespace(gamma=0.99, hidden_dim=32, learning_rate=1e-3, gradient_clip_norm=10.0)


def test_network_shape():
    net = QNetwork(20, 3, 32)
    assert net(torch.zeros(4, 20)).shape == (4, 3)


def test_agent_action_range():
    agent = DQNAgent(20, 3, _cfg(), double_dqn=True, device=torch.device("cpu"))
    a = agent.act(np.zeros(20, dtype=np.float32), epsilon=0.0)
    assert 0 <= a < 3
