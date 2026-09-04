# Warehouse Robot Navigation with Double DQN

## Overview

This project explores the use of deep reinforcement learning for autonomous robot navigation in a dynamic warehouse environment. 
The robot learns how to navigate an 8×8 grid, reach a designated targe, and avoid obstacles that move throughout the environment.
We implemented and compared two value-based reinforcement learning approaches: 
Deep Q-Network (DQN) and Double Deep Q-Network (DDQN). 
DQN provides the baseline, while DDQN is investigated as an alternative designed to reduce overestimation in action-value estimates.
To ensure a fair comparison, both algorithms are trained using the same environment, observation representation, neural network architecture, reward function, replay buffer, exploration strategy, training budget, random seeds and evaluation episodes. 
The project evaluates the agents based on their ability to complete the navigation task, avoid collisions, minimize unnecessary movement and maintain consistent performance across different random seeds.

## Environment and MDP

| Property | Value |
|---|---|
| Environment | `MiniGrid-Dynamic-Obstacles-8x8-v0` |
| API | Gymnasium 1.3.0 with MiniGrid 3.1.0 |
| Observation | 592-dimensional float vector in [0, 1] |
| Actions | `Discrete(3)` |
| Episode limit | 256 steps |
| Discount factor | γ = 0.99 |

### Action space

| Index | Action |
|---|---|
| 0 | Turn left |
| 1 | Turn right |
| 2 | Move forward |

`DynamicObstacles` restricts the action space to three actions natively; MiniGrid's remaining four (pickup, drop, toggle, done) have no effect in this environment and are not exposed.

### State representation

The agent receives a 7×7×3 symbolic observation representing its egocentric view of the environment. 
Each cell contains information about the object, colour and state.
To capture the movement of dynamic obstacles, the `SymbolicFrameStack` wrapper:
- Normalizes the observation values to [0, 1].
- Stacks the four most recent frames: `7 × 7 × 3 × 4 = 588`.
- Adds a 4-way one-hot encoding of the robot's direction.
This gives a total observation size of 592 features.

Frame stacking provides temporal information that is not available from a single observation, allowing the agent to infer obstacle movement.
Although the environment is largely observable spatially, a single frame does not reveal obstacle motion. 
Therefore, frame stacking is used to provide additional temporal information without claiming that the resulting representation is strictly Markov.

### Reward function

The `WarehouseRewardWrapper` replaces the default MiniGrid reward with a custom reward scheme:

| Event | Reward |
|---|---:|
| Reach target | +10 |
| Hit obstacle | −10 |
| Normal step | −0.01 |

The small step penalty encourages the agent to reach the target efficiently while the larger rewards and penalties encourage successful navigation and obstacle avoidance.

### Termination and Truncation

The two episode-ending conditions are handled separately:

| Condition | Meaning |
|---|---|
| Terminated | Robot reaches the target or hits an obstacle |
| Truncated | 256-step episode limit is reached |

Only `terminated` is stored as the terminal flag in the replay buffer, ensuring that time-limit truncation can still use bootstrapped value estimates.
The `done` flag (`terminated or truncated`) is used to end and reset the episode.

## Algorithms

Both DQN and DDQN use an online network and a periodically updated target network. The main difference is how the next-state value is estimated.

**DQN**

DQN uses the target network to select and evaluate the highest-value next action:
```text
y = r + γ · max_a Q_target(s′, a)
```

**Double DQN** 
DDQN separates action selection from action evaluation. 
The online network selects the best action, while the target network evaluates it
```text
a* = argmax_a Q_online(s′, a)
y  = r + γ · Q_target(s′, a*)
```

### Network

Both algorithms use the same multilayer perceptron (MLP):

```text
592 → 256 → ReLU → 256 → ReLU → 3
```

### Experience replay
A replay buffer stores the agent's past experiences:
```text
(state, action, reward, next_state, terminated)
```
---

## Hyperparameters

The same hyperparameters were used for DQN and DDQN across all three training seeds. 
The configuration was fixed before the final experiment, with no hyperparameter tuning or search performed.

| Parameter | Value |
|---|---|
| Total steps per run | 100,000 |
| Max episode steps | 256 |
| Discount factor γ | 0.99 |
| Learning rate | 5 × 10⁻⁴ (Adam) |
| Batch size | 64 |
| Replay capacity | 50,000 |
| Learning starts | 2,000 steps |
| Train frequency | every 4 steps |
| Target sync frequency | every 1,000 steps |
| ε start → end | 1.0 → 0.05 |
| ε decay | linear over 60,000 steps |
| Hidden dimension | 256 |
| Gradient clip norm | 10.0 |
| Loss | Smooth L1 (Huber) |
| Success / collision / step reward | +10.0 / −10.0 / −0.01 |
| Training seeds | 42, 123, 2026 |
| Evaluation episodes | 30 per trained seed |
| Evaluation seed range | 10000–10029 |

The complete configuration is saved to `logs/experiment_config.json` at the start of each run.
---

## Installation

### Requirements

- Python 3.10 or later
- Git

### Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Mister-Kayy/ddqn_warehouse_robot_navigation.git
cd ddqn_warehouse_robot_navigation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`DDQN2_Warehouse_Robot_Navigation_Runner.ipynb` in the repository root runs the complete pipeline in Colab, cell by cell, with commentary at each stage. This is how the submitted results were produced.

### Verify Installation

Before training, run the smoke test to confirm that the environment and observation/action spaces are configured correctly:
```bash
python smoke_test.py
```

Expected output:

```text
observation shape: (592,)
action space: Discrete(3)
Smoke test passed
```

---

## Reproducing the headline result

A single entry point runs everything — six training runs, twelve evaluations, aggregation, figures and the summary table:

```bash
python run_experiment.py
```

Around 30 minutes. A quick mode exists purely to confirm the pipeline executes:

```bash
python run_experiment.py --quick
```

Quick mode trains for 5,000 steps and evaluates on 5 episodes. 
**Its output must not be used in the analysis** — it exists to catch broken code before committing to a full run, and it overwrites `logs/experiment_config.json` with the reduced settings.

To check which configuration produced the logs currently in the repository:

```bash
grep total_steps logs/experiment_config.json      # 100000 for the submitted run
```

### Regenerating figures without retraining

```python
from src.plotting import plot_training, plot_evaluation
from src.reporting import print_results

plot_training("."); plot_evaluation("."); print_results(".")
```

### Recording a demonstration rollout

```bash
python record_rollout.py --algorithm ddqn --seed 42 --episode-seed 10000 \
    --out figures/rollout_ddqn_seed42_success.mp4
```

---

## Outputs

| Path | Contents |
|---|---|
| `logs/{dqn,ddqn}/seed_*.csv` | per-episode training records: return, steps, success, collision, ε, loss, mean Q |
| `logs/evaluation/*.csv` | per-episode evaluation records |
| `logs/evaluation_seed_metrics.csv` | metrics per training seed |
| `logs/evaluation_summary.csv` | mean and SD across seeds |
| `logs/experiment_config.json` | the exact configuration used |
| `models/{dqn,ddqn}/seed_*.pt` | trained weights, optimiser state and metadata |
| `figures/*.png` | training curves and evaluation charts |
| `figures/*.mp4` | greedy rollouts of trained agents |

Every figure in the report is generated from the committed CSV files. No number is transcribed by hand.

### Figure and video map

| File | Produced by | Reads |
|---|---|---|
| `training_return.png` | `plot_training()` | `logs/{dqn,ddqn}/seed_*.csv` |
| `training_success.png` | `plot_training()` | as above |
| `eval_completion.png` | `plot_evaluation()` | `logs/evaluation_seed_metrics.csv` |
| `eval_collision.png` | `plot_evaluation()` | as above |
| `eval_steps.png` | `plot_evaluation()` | as above |
| `eval_return.png` | `plot_evaluation()` | as above |
| `rollout_ddqn_seed42_success.mp4` | `record_rollout.py` | `models/ddqn/seed_42.pt` |
| `rollout_dqn_seed123_success.mp4` | `record_rollout.py` | `models/dqn/seed_123.pt` |
| `rollout_dqn_seed42_failure.mp4` | `record_rollout.py` | `models/dqn/seed_42.pt` |

---

## Experimental protocol

**Training.** 
Three seeds per algorithm — 42, 123, 2026 — giving six runs of 100,000 environment steps. 
A single frozen `ExperimentConfig` is passed to every run, so no per-seed variation is possible.

**Evaluation.** 
Each trained model is evaluated on 30 episodes using seeds 10000–10029. 
Both algorithms face the same thirty episodes, so obstacle behaviour is identical across the comparison. Exploration is disabled (`epsilon = 0.0`); actions are selected greedily.

**Aggregation.** 
Two stages: within a seed across its 30 episodes, then across the three seeds to give mean and standard deviation. 
Both algorithms pass through the same code path in `summarize_evaluation()`.

---

## Results

Mean ± SD across the three training seeds:

| Algorithm | Completion | Collision | Steps | Return |
|---|---:|---:|---:|---:|
| **DDQN** | 0.911 ± 0.154 | 0.000 ± 0.000 | 60.0 ± 72.6 | 8.511 ± 2.265 |
| **DQN** | 0.667 ± 0.577 | 0.000 ± 0.000 | 103.2 ± 132.5 | 5.635 ± 7.097 |

### Per seed

| Algorithm | Seed | Completion | Avg steps | Avg steps (successes) | Return |
|---|---:|---:|---:|---:|---:|
| DDQN | 42 | 1.000 | 18.8 | 18.8 | 9.812 |
| DDQN | 123 | 0.733 | 143.8 | 103.0 | 5.896 |
| DDQN | 2026 | 1.000 | 17.4 | 17.4 | 9.826 |
| DQN | 42 | 0.000 | 256.0 | — | −2.560 |
| DQN | 123 | 1.000 | 20.3 | 20.3 | 9.797 |
| DQN | 2026 | 1.000 | 33.4 | 33.4 | 9.666 |

### What these numbers do and do not support

**We do not claim that DDQN outperforms DQN.** 
The 24-point difference in mean completion rate is smaller than DQN's standard deviation across seeds (0.577). 
With three seeds, this difference does not exceed run-to-run variation, and no claim of superiority is supportable from it.

**What the per-seed table does show is a difference in worst case.** 
Every DDQN seed reached at least 73% completion. DQN produced two near-perfect seeds and one complete failure. 
Seed 42 recorded 0% completion, exactly 256 steps on every evaluation episode, and zero collisions which the signature of an agent that never moves forward, turning in place until the time limit expires. 
Its return of −2.560 is precisely 256 × −0.01, confirming that no terminal reward of either sign was ever received.

Under our reward weighting this behaviour is locally rational: spinning for a full episode costs 2.56, while a single collision costs 10. 
An agent that has not yet discovered the goal is, by its own value estimates, better off doing nothing. The three rollout videos show all three regimes directly — DDQN succeeding, DQN succeeding, and DQN seed 42 spinning.

Both algorithms are therefore capable of solving this environment, and both are sensitive to initialisation. 
This is consistent with Double DQN's reduced overestimation bias making the do-nothing local optimum less attractive, but **three seeds cannot establish that mechanism**, and we do not present it as demonstrated.

**Zero collisions everywhere.** 
Both algorithms learned collision avoidance completely — no evaluation episode of any seed ended in a collision. 
Given the −10 penalty against a −0.01 step cost, this is the easiest part of the task to learn and the earliest behaviour to appear.

---

## Time-limit handling

An earlier version of `train.py` stored the combined flag as the bootstrap target:

```python
done = terminated or truncated
replay.add(obs, action, reward, next_obs, done)     # incorrect
```

This treats hitting the step limit as though the episode genuinely ended, zeroing the value of a state whose future still exists. The corrected version stores `terminated` alone:

```python
done = terminated or truncated                       # controls logging and reset
replay.add(obs, action, reward, next_obs, terminated)  # controls bootstrapping
```

The bug mattered asymmetrically here. A policy that times out truncates *every* episode, so every one of its transitions taught the agent that the timed-out state carries no further cost — systematically flattering the stand-still policy that DQN seed 42 adopted. The correction follows Pardo et al. (2018).

All results in this README come from the corrected implementation. The fix is recorded in the commit history.

---

## Repository structure

```text
ddqn_warehouse_robot_navigation/
│
├── src/
│   ├── config.py
│   ├── env.py
│   ├── network.py
│   ├── replay_buffer.py
│   ├── agent.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plotting.py
│   ├── reporting.py
│   └── utils.py
│
├── tests/
│   └── Unit tests
│
├── logs/
│   └── Experiment logs and configuration files
│
├── models/
│   └── Trained model weights
│
├── figures/
│   └── Generated figures and rollout videos
│
├── run_experiment.py
├── smoke_test.py
├── record_rollout.py
├── DDQN2_Warehouse_Robot_Navigation_Runner.ipynb
├── requirements.txt
└── README.md
```

## Limitations

**Number of seeds:** 
Only three training seeds were used. 
This provides limited evidence for comparing algorithms, particularly given the variability observed across runs. 

**Training budget:** 
Each run was limited to 100,000 training steps. 
This was a practical constraint and should not be interpreted as evidence that the agents had fully converged.

**Training and evaluation seeding:** 
Training and evaluation use separate seed ranges, but only the initial training reset is explicitly seeded. 
Therefore, identical environment layouts between training and evaluation cannot be completely ruled out.

**Q-value analysis:** 
Mean Q-values are recorded during training but are not currently included in the analysis. 
Comparing predicted Q-values with realised returns could provide additional evidence about overestimation and would be possible using the existing logs without retraining.

**Reward sensitivity:** 
The experiment uses a single reward configuration. 
Different step penalties or collision penalties could affect agent behaviour, so additional reward ablation experiments would help distinguish the effect of reward design from the effect of the algorithm.

**Environment scope:** 
The results are specific to `MiniGrid-Dynamic-Obstacles-8x8-v0` and its 7×7 observation window. 
They should not be assumed to generalise directly to larger environments, heavily occluded environments, or continuous-control tasks.
.
