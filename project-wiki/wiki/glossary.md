# Glossary — 용어 사전

**상태**: active
**마지막 업데이트**: 2026-10-28

| 용어 | 뜻 | 코드/지표 |
|---|---|---|
| **PPO** | Proximal Policy Optimization. 중요도 비율을 [1-ε, 1+ε] 로 클리핑해 한 번의 업데이트에서 정책이 너무 멀리 가지 않게 하는 on-policy 정책 경사법 | `ppo_update` |
| **롤아웃 (rollout)** | 현재 정책으로 N 개 환경 × T 스텝 수집한 전이 묶음. 여기선 4×1024 = 4096 (기본값, EXP-007 이전엔 4×128) | `RolloutBuffer` |
| **iteration** | 롤아웃 1회 + 업데이트 1회. `total_timesteps // (num_envs*num_steps)` 회 | `train` |
| **어드밴티지 A(s,a)** | 그 행동이 평균(V(s))보다 얼마나 나았는가. 정책 경사의 가중치 | `advantages` |
| **TD 오차 δ** | r + γV(s') − V(s). 1-스텝 어드밴티지 추정 | `compute_gae` 내부 `delta` |
| **GAE** | Generalized Advantage Estimation. Σ(γλ)^k δ_{t+k}. λ=0 → TD(편향↑ 분산↓), λ=1 → MC(편향↓ 분산↑) | `compute_gae`, `gae_lambda` |
| **리턴 / 가치 타깃** | A + V(롤아웃 당시). critic 회귀 목표 = TD(λ) 리턴 | `returns` |
| **부트스트랩** | 롤아웃 끝(또는 미종료 에피소드)의 나머지 미래 보상을 V(s_T) 로 대체 | `next_value` |
| **done / nextnonterminal** | done 이면 다음 상태 가치와 GAE trace 를 끊는다. 이 구현은 terminated∨truncated 를 모두 done 처리 | `dones`, `next_done` |
| **중요도 비율 r** | π_new(a|s)/π_old(a|s) = exp(logπ_new − logπ_old) | `ratio` |
| **클리핑 surrogate** | min(r·A, clip(r,1±ε)·A) 를 최대화 | `pg_loss`, `clip_coef` |
| **가치 손실 클리핑** | V_new 를 V_old ± ε 로 잘라 두 손실 중 큰 쪽 사용 | `clip_vloss` |
| **엔트로피 보너스** | 정책 분포 엔트로피를 손실에서 빼 탐색 유지 | `ent_coef`, `losses/entropy` |
| **approx_kl** | KL(π_old‖π_new) 추정 (k3 = (r−1) − log r). 0.02 초과 지속 시 LR/에폭 점검 | `losses/approx_kl` |
| **clipfrac** | |r−1| > ε 인 샘플 비율 | `losses/clipfrac` |
| **explained variance (EV)** | 1 − Var(R−V)/Var(R). 1 이면 V 가 리턴을 완벽 예측, 목표 ≥ 0.8 | `losses/explained_variance` |
| **어드밴티지 정규화** | 미니배치 내에서 A 를 평균 0, 표준편차 1 로 | `norm_adv` |
| **보상 정규화** | 할인 리턴의 이동 표준편차로 보상을 나눔 (gym `NormalizeReward`) + ±10 clip | `norm_reward` |
| **LR annealing** | 학습률을 선형으로 0 까지 감소 | `anneal_lr`, `charts/learning_rate` |
| **글로벌 grad clipping** | 전체 파라미터 그래디언트 L2 노름을 0.5 이하로 | `max_grad_norm` |
| **SPS** | Steps Per Second (환경 스텝/초) | `charts/SPS` |
| **SAME_STEP autoreset** | Gymnasium 1.x 벡터 환경에서 종료 스텝이 곧바로 다음 에피소드 첫 관측을 반환하는 모드. 기본 NEXT_STEP 은 "행동이 무시되는 가짜 스텝"을 하나 끼움 | `make_vec_env` |
| **호버링 (hovering)** | LunarLander 국소 최적: 착륙하지 않고 떠 있다가 1000 스텝 truncation. 보상 −30 ~ −80 대, `episodic_length` ≈ 1000 | [troubleshooting](troubleshooting/common.md) |
| **mean100** | 최근 100 에피소드 평균 episodic_return (학습 중 확률적 정책) | 콘솔 로그 |
| **eval 100ep** | 저장된 정책을 argmax(결정적)로 100 에피소드 평가한 평균. **성공 기준은 이것 ≥ 200** | `evaluate` |
