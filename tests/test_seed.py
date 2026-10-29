"""FR-8: same seed -> identical training (parameters bit-for-bit equal)."""

import torch

from ppo import DEFAULTS, train


def _run(seed, tmp_path):
    # 4 updates of 4x128 — small but exercises rollout, GAE, minibatch shuffling and Adam
    cfg = dict(DEFAULTS, seed=seed, num_steps=128, num_minibatches=4, total_timesteps=2048,
               save_model=False, run_dir=str(tmp_path))
    agent, _ = train(cfg)
    return torch.cat([p.detach().flatten() for p in agent.parameters()])


def test_same_seed_same_params(tmp_path):
    assert torch.equal(_run(0, tmp_path), _run(0, tmp_path))


def test_different_seed_differs(tmp_path):
    assert not torch.equal(_run(0, tmp_path), _run(1, tmp_path))
