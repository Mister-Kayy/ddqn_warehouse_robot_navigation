from src.config import ExperimentConfig
from src.env import make_env

cfg = ExperimentConfig()
env = make_env(cfg)
obs, info = env.reset(seed=42)
print("observation shape:", obs.shape)
print("action space:", env.action_space)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
print("Smoke test passed")
env.close()
