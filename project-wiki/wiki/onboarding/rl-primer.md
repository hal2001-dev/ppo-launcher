# 강화학습 입문 — src/ppo.py 로 읽는 PPO

**상태**: active
**마지막 업데이트**: 2026-10-29
**관련 페이지**: [../architecture/code-map.md](../architecture/code-map.md), [../glossary.md](../glossary.md), [../experiments/ablation.md](../experiments/ablation.md), [../experiments/results.md](../experiments/results.md)

## 요약
강화학습의 기본 개념(상태·행동·보상·정책·가치)을 이 프로젝트의 실제 코드 `src/ppo.py` 에 대응시켜 설명한다.
한 iteration = **경험 수집 → 어드밴티지 계산 → PPO 업데이트**. 줄 번호는 2026-10-29 기준(커밋 `1dfb668`).
용어 정의는 [glossary.md](../glossary.md), 함수 흐름도는 [code-map.md](../architecture/code-map.md) 참고.

---

## 1. 강화학습이란 — 에이전트와 환경의 반복

```
상태(obs) → [에이전트] → 행동(action) → [환경] → 보상(reward), 다음 상태 → ...
```

에이전트는 시행착오로 **누적 보상을 최대화하는 행동 규칙(정책, policy)** 을 배운다.
지도학습처럼 정답 레이블이 있는 게 아니라 보상 신호만으로 스스로 찾아낸다.

| 개념 | LunarLander-v3 | 코드 위치 |
|---|---|---|
| 상태 | 8차원 (위치, 속도, 각도, 각속도, 양 다리 접지 여부) | `make_env` (`src/ppo.py:107`) |
| 행동 | 4개 이산 (무동작 / 왼쪽 / 메인 / 오른쪽 엔진) | `Categorical` 분포 (`src/ppo.py:157`) |
| 보상 | 착지점 근접 +, 연료 사용 −, 착륙 +100, 추락 −100 | 환경 내부 |
| 목표 | 결정적 정책 100 에피소드 평균 ≥ 200 | `evaluate` (`src/ppo.py:432`) |

---

## 2. 에이전트 = Actor + Critic (`src/ppo.py:146`)

```python
self.critic = mlp(obs_dim, 1, out_std=1.0)          # V(s): 이 상태가 얼마나 좋은가
self.actor  = mlp(obs_dim, n_actions, out_std=0.01) # π(a|s): 각 행동을 할 확률
```

- **Actor(정책)**: 상태 → 4개 행동의 확률. 학습 중엔 확률대로 **샘플링**(`dist.sample()`)해 탐험하고,
  평가 땐 **argmax**(`act_greedy`)로 가장 좋은 행동만 고른다.
- **Critic(가치함수)**: "이 상태에서 앞으로 받을 보상 합이 대략 얼마인가"를 예측. 직접 행동하지 않고 Actor 를 평가하는 기준점.
- 출력층 `std=0.01` → 초기 정책이 거의 균등 확률 → 처음엔 고르게 탐험.
- 64-64 Tanh MLP 로 매우 작아서 GPU 보다 CPU 1 스레드가 가장 빠르다 (15,160 vs MPS 2,450 SPS, ADR-004).

---

## 3. 학습 루프: 수집 → 평가 → 개선 (`train`, `src/ppo.py:350`)

1 iteration 에 4 env × 1024 스텝 = **4,096 개 경험**. 2M 스텝이면 약 488 회 반복.

### ① 경험 수집 (rollout) — `src/ppo.py:355`

```python
action, logprob, _, value = agent.get_action_and_value(next_obs)
obs_np, reward, term, trunc, infos = envs.step(action)
buf.add(next_obs, action, logprob, reward, next_done, value)
```

현재 정책으로 직접 플레이하며 (상태, 행동, 행동 확률, 보상, 종료 여부, Critic 예측) 을 버퍼에 쌓는다.
이때 저장한 `logprob`(수집 당시 정책의 확률)이 나중에 PPO 비율 계산의 재료가 된다.

- `SyncVectorEnv` (`src/ppo.py:123`): 환경 4개를 동시에 돌려 데이터 수집량 4배. `SAME_STEP` autoreset 으로 가짜 스텝 없음 (ADR-006).
- `NormalizeReward` (`src/ppo.py:113`): 보상 스케일을 맞춰 학습 안정화 (+ ±10 clip).

### ② 어드밴티지 — "그 행동이 평소보다 얼마나 좋았나" (`src/ppo.py:220`)

강화학습의 핵심 난제는 **credit assignment**(어느 행동이 좋은 결과의 원인이었나)다.
착륙 +100 은 마지막에 들어오지만, 원인은 수백 스텝 전의 엔진 조작일 수 있다.

```python
delta = r_t + γ·V(s_{t+1}) − V(s_t)      # TD 오차: 예상보다 얼마나 좋았나
A_t   = delta + γ·λ·A_{t+1}              # GAE: 뒤에서부터 누적
```

- **γ (gamma) = 0.999**: 미래 보상을 얼마나 중시할지. 0.99 는 실질 시야 ~100 스텝이라 착륙까지 내다보지 못해
  **호버링** 국소 최적에 빠졌다 ([EXP-002](../experiments/EXP-002-cleanrl-contrast.md), [EXP-003](../experiments/EXP-003-tuning-round1.md)). 0.999 는 ~1000 스텝.
- **λ (lambda) = 0.98**: 1 에 가까울수록 실제 보상을 신뢰(분산↑ 편향↓), 0 에 가까울수록 Critic 예측을 신뢰(분산↓ 편향↑).
- **A > 0** → 평소보다 좋은 행동, **A < 0** → 나쁜 행동.
- `(1 - done)` 항: 에피소드 경계에서 다음 에피소드 가치를 끌어오지 않도록 끊는다.
- 반환값 `returns = advantages + values` 가 Critic 의 학습 목표가 된다.

### ③ 정책 개선 — PPO 업데이트 (`src/ppo.py:257`)

4,096 개를 섞어 미니배치 128 × 32 개로 나누고 **4 에폭** 반복 학습.

**(a) 정책 손실 — Clipped Surrogate (PPO 의 핵심)**

```python
ratio = exp(newlogprob − oldlogprob)          # 새 정책 확률 / 수집 당시 정책 확률
pg_loss1 = -adv * ratio
pg_loss2 = -adv * clamp(ratio, 0.8, 1.2)
pg_loss  = max(pg_loss1, pg_loss2).mean()
```

- 기본 아이디어(policy gradient): A > 0 인 행동은 확률을 높이고, A < 0 은 낮춘다.
- 문제: 같은 데이터로 여러 번 학습하면 정책이 수집 당시와 너무 달라져 데이터가 무의미해지고 학습이 붕괴할 수 있다.
- 해결: 확률 비율을 **±20% 로 잘라** 한 번에 너무 멀리 가지 못하게 한다 → "**Proximal** Policy Optimization".

**(b) 가치 손실**: `0.5 * (V(s) − returns)²` — Critic 이 실제 리턴을 더 잘 맞히도록.
가치 손실 클리핑(`clip_vloss`)은 이 환경의 리턴 스케일에서 Critic 학습을 늦춰서 끔 ([EXP-006](../experiments/EXP-006-sb3-gap.md)).

**(c) 엔트로피 보너스**: `− 0.01 · entropy` — 정책이 너무 빨리 한 행동으로 굳지 않게 탐험 유지.

**(d) 합쳐서 한 번에 최적화**

```python
loss = pg_loss − ent_coef·entropy + vf_coef·v_loss
clip_grad_norm_(..., 0.5)   # 그래디언트 폭주 방지
```

학습률은 2.5e-4 → 0 으로 선형 감소 (`anneal_lr`, `src/ppo.py:351`): 초반엔 크게, 후반엔 미세 조정.

---

## 4. 학습 상태를 보는 지표 (TensorBoard, 포트 3005)

| 지표 | 의미 | 바람직한 모습 |
|---|---|---|
| `charts/episodic_return` | 에피소드 점수 | 200 이상으로 상승 |
| `charts/episodic_length` | 에피소드 길이 | ≈1000 고착이면 호버링 |
| `losses/approx_kl` | 업데이트 1회에 정책이 변한 정도 | 작게 유지 (~0.01) |
| `losses/clipfrac` | 클리핑에 걸린 샘플 비율 | 0.1~0.2 |
| `losses/entropy` | 정책의 무작위성 | 서서히 감소 |
| `losses/explained_variance` | Critic 예측 정확도 | 1 에 가까울수록 (≥ 0.8) |

실행 명령은 [../deployment/runbook.md](../deployment/runbook.md).

---

## 5. 실험으로 확인한 각 부품의 역할

기본 설정 eval 평균 **268.1** (3 시드) 에서 하나씩 제거 ([ablation.md](../experiments/ablation.md), [EXP-008](../experiments/EXP-008-ablation.md)).

| 제거 | 결과 | 교훈 |
|---|---|---|
| GAE | 253.2 (−14.9), 학습 곡선 AUC 1/4 | 최종 성능보다 **학습 안정성(분산 감소)** 에 기여 |
| 보상 정규화 | 255.3 (−12.8) | 이 설정에선 이득, 작은 배치·γ 0.99 에선 손해 → **조건 의존** |
| 클리핑 | 271.1 (+3.0), KL 8배 | 평균은 비슷하지만 정규화가 없을 땐 2/3 시드 실패 → **최악 시드 보험** |

튜닝에서 얻은 가장 큰 교훈: 알고리즘 구현보다 **롤아웃 길이(128 → 1024)와 γ(0.99 → 0.999)** 가 결정적이었다
([EXP-004](../experiments/EXP-004-tuning-round2.md), [EXP-005](../experiments/EXP-005-tuning-round3.md)).
LunarLander 처럼 보상이 늦게 오는 문제에선 "얼마나 멀리 내다보느냐"가 핵심이다.

---

## 더 읽을 거리
- `src/ppo_annotated.py` — CleanRL 원본에 줄 단위 주석
- `notes/02_cleanrl_reading.md` — CleanRL 코드 읽기 노트
- [../references.md](../references.md) — PPO 논문, 37 implementation details 블로그 등

## 출처
- `src/ppo.py` (커밋 `1dfb668`)
- [../experiments/results.md](../experiments/results.md), [../experiments/ablation.md](../experiments/ablation.md)
