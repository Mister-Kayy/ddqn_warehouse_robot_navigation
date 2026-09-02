from __future__ import annotations
from pathlib import Path
import pandas as pd


def print_results(root):
    root = Path(root)
    summary_path = root / "logs" / "evaluation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run evaluation first.")
    summary = pd.read_csv(summary_path)
    print("\nHEADLINE EVALUATION SUMMARY (mean ± SD across training seeds)\n")
    for _, r in summary.iterrows():
        print(f"{r.algorithm.upper()}: completion {r.task_completion_rate_mean:.3f} ± {r.task_completion_rate_sd:.3f}; "
              f"collision {r.collision_rate_mean:.3f} ± {r.collision_rate_sd:.3f}; "
              f"steps {r.average_steps_mean:.1f} ± {r.average_steps_sd:.1f}; "
              f"return {r.average_return_mean:.3f} ± {r.average_return_sd:.3f}")
