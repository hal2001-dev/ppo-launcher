"""Evaluate saved policies: 100 episodes, deterministic (argmax), raw env reward.

Usage:
    python src/evaluate.py models/<run>.pt               # src/ppo.py checkpoint
    python src/evaluate.py models/<run>.zip              # SB3 checkpoint
    python src/evaluate.py models/*.pt --episodes 100    # several at once, prints a summary table
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def eval_sb3(path, episodes, seed):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(gym.make("LunarLander-v3"))
    env.reset(seed=seed)
    mean, std = evaluate_policy(PPO.load(path, device="cpu"), env, n_eval_episodes=episodes, deterministic=True)
    print(f"eval {Path(path).name}: {episodes} episodes mean={mean:.2f} std={std:.2f}")
    return mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    a = p.parse_args()

    rows = []
    for m in a.models:
        if m.endswith(".zip"):
            mean, std = eval_sb3(m, a.episodes, a.seed)
        else:
            from ppo import evaluate
            mean, std = evaluate(m, a.episodes, seed=a.seed)
        rows.append((Path(m).name, mean, std))

    if len(rows) > 1:
        print("\n| model | mean | std | >=200 |\n|---|---|---|---|")
        for name, mean, std in rows:
            print(f"| {name} | {mean:.2f} | {std:.2f} | {'✅' if mean >= 200 else '❌'} |")
        print(f"\noverall mean = {np.mean([r[1] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
