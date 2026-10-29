"""FR-4: GAE numeric checks on hand-computable cases."""

import pytest
import torch

from ppo import compute_gae, compute_mc_returns

G, LAM = 0.9, 0.5


def col(*xs):
    """(T,) values -> (T, 1) tensor (single env)."""
    return torch.tensor(xs, dtype=torch.float32).unsqueeze(1)


def test_no_done_hand_computed():
    rewards, values = col(1.0, 1.0), col(0.5, 0.5)
    dones, next_value, next_done = col(0.0, 0.0), torch.tensor([2.0]), torch.tensor([0.0])
    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, G, LAM)
    d1 = 1.0 + G * 2.0 - 0.5          # 2.3
    d0 = 1.0 + G * 0.5 - 0.5          # 0.95
    a1, a0 = d1, d0 + G * LAM * d1    # 2.3, 1.985
    assert adv.squeeze().tolist() == pytest.approx([a0, a1])
    assert ret.squeeze().tolist() == pytest.approx([a0 + 0.5, a1 + 0.5])


def test_done_cuts_bootstrap_and_trace():
    # dones[1] = 1 -> transition 0 ended an episode: no V(s_1) bootstrap, no trace from t=1 into t=0.
    rewards, values = col(1.0, 1.0), col(0.5, 0.5)
    dones, next_value, next_done = col(0.0, 1.0), torch.tensor([2.0]), torch.tensor([0.0])
    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, G, LAM)
    assert adv.squeeze().tolist() == pytest.approx([1.0 - 0.5, 1.0 + G * 2.0 - 0.5])


def test_next_done_cuts_last_bootstrap():
    rewards, values = col(1.0), col(0.5)
    adv, _ = compute_gae(rewards, values, col(0.0), torch.tensor([100.0]), torch.tensor([1.0]), G, LAM)
    assert adv.item() == pytest.approx(0.5)


def test_lambda_zero_is_td_error():
    torch.manual_seed(0)
    T, N = 6, 3
    r, v, nv = torch.randn(T, N), torch.randn(T, N), torch.randn(N)
    d, nd = torch.zeros(T, N), torch.zeros(N)
    adv, _ = compute_gae(r, v, d, nv, nd, G, 0.0)
    next_v = torch.cat([v[1:], nv.unsqueeze(0)])
    assert torch.allclose(adv, r + G * next_v - v)


def test_lambda_one_matches_mc_returns():
    torch.manual_seed(1)
    T, N = 8, 4
    r, v, nv = torch.randn(T, N), torch.randn(T, N), torch.randn(N)
    d = (torch.rand(T, N) < 0.2).float()
    nd = (torch.rand(N) < 0.5).float()
    adv_gae, ret_gae = compute_gae(r, v, d, nv, nd, G, 1.0)
    adv_mc, ret_mc = compute_mc_returns(r, v, d, nv, nd, G)
    assert torch.allclose(adv_gae, adv_mc, atol=1e-5)
    assert torch.allclose(ret_gae, ret_mc, atol=1e-5)


def test_multi_env_independent():
    # Each env column must be computed independently of the others.
    torch.manual_seed(2)
    T, N = 5, 3
    r, v, nv = torch.randn(T, N), torch.randn(T, N), torch.randn(N)
    d, nd = (torch.rand(T, N) < 0.3).float(), torch.zeros(N)
    adv, _ = compute_gae(r, v, d, nv, nd, 0.99, 0.95)
    for i in range(N):
        a_i, _ = compute_gae(r[:, i:i + 1], v[:, i:i + 1], d[:, i:i + 1], nv[i:i + 1], nd[i:i + 1], 0.99, 0.95)
        assert torch.allclose(adv[:, i:i + 1], a_i)
