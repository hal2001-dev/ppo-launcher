# References — 참고 자료

**상태**: active
**마지막 업데이트**: 2026-10-28

## 논문
- Schulman et al., 2017 — *Proximal Policy Optimization Algorithms* — https://arxiv.org/abs/1707.06347
- Schulman et al., 2015 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — https://arxiv.org/abs/1506.02438
- Engstrom et al., 2020 — *Implementation Matters in Deep Policy Gradients* — https://arxiv.org/abs/2005.12729
- Andrychowicz et al., 2020 — *What Matters In On-Policy Reinforcement Learning?* — https://arxiv.org/abs/2006.05990

## 구현 / 블로그
- CleanRL `ppo.py` — https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
  (스냅샷: `raw/research/cleanrl_ppo_fe8d8a0.py.txt`, commit fe8d8a0)
- CleanRL PPO 문서 — https://docs.cleanrl.dev/rl-algorithms/ppo/
- Huang et al., *The 37 Implementation Details of PPO* (ICLR Blog Track 2022) — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- John Schulman, *Approximating KL Divergence* — http://joschu.net/blog/kl-approx.html
- Stable-Baselines3 PPO — https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- rl-baselines3-zoo `hyperparams/ppo.yml` (LunarLander 항목) — https://github.com/DLR-RM/rl-baselines3-zoo

## 환경
- Gymnasium LunarLander — https://gymnasium.farama.org/environments/box2d/lunar_lander/
  (v3: 200점 이상이면 "solved", 착륙 +100~140, 추락 −100, 엔진 분사마다 −0.3/−0.03, 1000 스텝 제한)
- Gymnasium 1.x 벡터 환경 autoreset 모드 — https://farama.org/Vector-Autoreset-Mode

## 위키 패턴 / 다른 프로젝트
- karpathy LLM Wiki pattern — https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- 형제 프로젝트 위키·공용 문서 → [related-projects.md](related-projects.md)
