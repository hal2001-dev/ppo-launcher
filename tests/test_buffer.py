"""FR-1/FR-2/FR-3: vector env shape, network split, rollout buffer shapes."""

import pytest
import torch

from ppo import DEFAULTS, Agent, RolloutBuffer, count_params, make_vec_env

T, N, OBS = 128, 4, 8


def test_buffer_shapes():
    buf = RolloutBuffer(T, N, OBS, torch.device("cpu"))
    assert buf.obs.shape == (T, N, OBS)
    for name in ("actions", "logprobs", "rewards", "dones", "values"):
        assert getattr(buf, name).shape == (T, N), name
    assert buf.actions.dtype == torch.long


def test_buffer_fill_and_flatten():
    buf = RolloutBuffer(T, N, OBS, torch.device("cpu"))
    for t in range(T):
        buf.add(torch.full((N, OBS), float(t)), torch.zeros(N, dtype=torch.long), torch.zeros(N),
                torch.ones(N), torch.zeros(N), torch.zeros(N))
    assert buf.full()
    with pytest.raises(AssertionError):
        buf.add(torch.zeros(N, OBS), torch.zeros(N, dtype=torch.long), *[torch.zeros(N)] * 4)

    b = buf.flatten(torch.zeros(T, N), torch.zeros(T, N))
    assert b["obs"].shape == (T * N, OBS)
    for k in ("actions", "logprobs", "values", "advantages", "returns"):
        assert b[k].shape == (T * N,), k
    # row-major flatten: first N rows come from t=0, the next N from t=1
    assert b["obs"][:N].eq(0).all() and b["obs"][N:2 * N].eq(1).all()

    buf.reset()
    assert buf.ptr == 0 and not buf.full()


def test_vec_env_obs_shape():
    envs = make_vec_env(dict(DEFAULTS, num_envs=N))
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (N, OBS)
    envs.close()


def test_agent_actor_critic_separate():
    agent = Agent(OBS, 4)
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    # 8*64+64 + 64*64+64 + 64*k+k
    assert count_params(agent.actor) == 8 * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4
    assert count_params(agent.critic) == 8 * 64 + 64 + 64 * 64 + 64 + 64 + 1
    a, logp, ent, v = agent.get_action_and_value(torch.zeros(N, OBS))
    assert a.shape == logp.shape == ent.shape == v.shape == (N,)
