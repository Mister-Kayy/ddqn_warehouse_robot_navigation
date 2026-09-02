
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _interpolate_seed(df, xgrid, metric):
    d = df.sort_values("global_step")
    if len(d) < 2:
        return np.full_like(xgrid, np.nan, dtype=float)
    y = d[metric].astype(float).rolling(20, min_periods=1).mean().to_numpy()
    x = d["global_step"].to_numpy()
    return np.interp(xgrid, x, y, left=np.nan, right=y[-1])


def plot_training(root: str | Path):
    root = Path(root); fig_dir = root / "figures"; fig_dir.mkdir(exist_ok=True)
    for metric, ylabel, fname in [
        ("episode_return", "Episode return (20-episode rolling mean)", "training_return.png"),
        ("success", "Success indicator (20-episode rolling mean)", "training_success.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for algo in ["dqn", "ddqn"]:
            files = sorted((root / "logs" / algo).glob("seed_*.csv"))
            if not files:
                continue
            dfs = [pd.read_csv(p) for p in files]
            max_common = min(d["global_step"].max() for d in dfs)
            xgrid = np.linspace(1, max_common, 300)
            ys = np.vstack([_interpolate_seed(d, xgrid, metric) for d in dfs])
            mean = np.nanmean(ys, axis=0); sd = np.nanstd(ys, axis=0, ddof=1 if len(dfs)>1 else 0)
            plt.plot(xgrid, mean, label=algo.upper())
            plt.fill_between(xgrid, mean - sd, mean + sd, alpha=0.2)
        plt.xlabel("Environment steps"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=200); plt.close()


def plot_evaluation(root: str | Path):
    root = Path(root); fig_dir = root / "figures"; fig_dir.mkdir(exist_ok=True)
    summary = pd.read_csv(root / "logs" / "evaluation_summary.csv").set_index("algorithm")
    metrics = [
        ("task_completion_rate", "Task completion rate", "eval_completion.png"),
        ("collision_rate", "Collision rate", "eval_collision.png"),
        ("average_return", "Average cumulative reward", "eval_return.png"),
        ("average_steps", "Average steps per episode", "eval_steps.png"),
    ]
    for key, ylabel, fname in metrics:
        algos = [a for a in ["dqn", "ddqn"] if a in summary.index]
        means = [summary.loc[a, f"{key}_mean"] for a in algos]
        sds = [summary.loc[a, f"{key}_sd"] for a in algos]
        plt.figure(figsize=(5, 4))
        plt.bar([a.upper() for a in algos], means, yerr=sds, capsize=5)
        plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(fig_dir / fname, dpi=200); plt.close()
