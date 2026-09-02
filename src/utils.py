
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferable for an examination protocol; may reduce speed.
    torch.use_deterministic_algorithms(True, warn_only=True)


def linear_epsilon(step, start, end, decay_steps):
    frac = min(max(step / float(decay_steps), 0.0), 1.0)
    return start + frac * (end - start)
