"""Stage 1: Stable-Baselines3 PPO baseline on LunarLander-v3.

Usage:
    python src/baseline_sb3.py                  # seed 0, 1M steps, then 100-episode deterministic eval
    python src/baseline_sb3.py --seeds 0 1 2
    python src/baseline_sb3.py --preset cleanrl # same hyperparameters as src/ppo.py (for curve comparison)

Hyperparameters ("zoo" preset) follow rl-baselines3-zoo's LunarLander-v2/v3 PPO entry, which reliably
reaches 200+ within 1M steps. Episodic returns are additionally written as `charts/episodic_return`
so the curve overlays with CleanRL / src/ppo.py in TensorBoard.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

ROOT = Path(__file__).resolve().parent.parent

PRESETS = {
    "zoo": dict(n_envs=16, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98,
                ent_coef=0.01, learning_rate=3e-4, clip_range=0.2),
    "cleanrl": dict(n_envs=4, n_steps=128, batch_size=128, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                    ent_coef=0.01, learning_rate=2.5e-4, clip_range=0.2),
}


class EpisodicReturnCallback(BaseCallback):
    """Log every finished episode as charts/episodic_return (CleanRL metric name)."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("charts/episodic_return", info["episode"]["r"])
                self.logger.record_mean("charts/episodic_length", info["episode"]["l"])
        return True


def linear_schedule(initial):
    return lambda progress_remaining: progress_remaining * initial


def run(seed, total_timesteps, preset, device):
    hp = dict(PRESETS[preset])
    n_envs = hp.pop("n_envs")
    lr = hp.pop("learning_rate")
    env = make_vec_env("LunarLander-v3", n_envs=n_envs, seed=seed)
    run_name = f"LunarLander-v3__sb3_{preset}__{seed}__{int(time.time())}"
    model = PPO("MlpPolicy", env, learning_rate=linear_schedule(lr) if preset == "cleanrl" else lr,
                seed=seed, device=device, verbose=1, tensorboard_log=str(ROOT / "runs"), **hp)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=EpisodicReturnCallback())

    path = ROOT / "models" / f"{run_name}.zip"
    path.parent.mkdir(exist_ok=True)
    model.save(path)

    eval_env = Monitor(gym.make("LunarLander-v3"))
    eval_env.reset(seed=10_000 + seed)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(f"[sb3 {preset} seed={seed}] eval 100 episodes: mean={mean:.2f} std={std:.2f} "
          f"{'PASS' if mean >= 200 else 'FAIL'} (target >= 200) -> {path.name}")
    return mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--preset", choices=sorted(PRESETS), default="zoo")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    means = [run(s, a.total_timesteps, a.preset, a.device) for s in a.seeds]
    print(f"seeds={a.seeds} mean of means={np.mean(means):.2f}")


if __name__ == "__main__":
    main()
