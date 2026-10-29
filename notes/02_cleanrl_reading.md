# 02. CleanRL ppo.py 읽기 메모

- 원본: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py (commit `fe8d8a0`, 312줄)
- 주석본: `src/ppo_annotated.py` — 원본 로직 그대로, `# [수정]` 3곳만 Gymnasium 1.x 대응
- 원본 스냅샷: `project-wiki/raw/research/cleanrl_ppo_fe8d8a0.py.txt`

## 전체 흐름 (한 iteration = 512 전이)

```
for iteration:
    lr ← lr0 · (1 − (it−1)/N)                      # 선형 감소
    for step in 128:                                # (1) 롤아웃
        obs[step], dones[step] ← next_obs, next_done
        a, logπ_old, V ← agent(next_obs)            # no_grad
        next_obs, r, term, trunc ← envs.step(a)
    A ← GAE(rewards, values, dones, V(next_obs))    # (2) 뒤에서 앞으로
    R ← A + V
    for epoch in 4: shuffle; for mb in 4 (128개):   # (3) 16 gradient step
        ratio ← exp(logπ_new − logπ_old)
        L = max(−A·r, −A·clip(r)) + 0.5·c1·max((V−R)², (Vclip−R)²) − c2·H
        clip_grad_norm(0.5); adam.step()
```

## 함수별 핵심

### make_env
- 환경을 만드는 **thunk 리스트**를 벡터 환경에 넘긴다. RecordEpisodeStatistics 가 `info["episode"]` 에 리턴/길이를 넣는다.

### layer_init / Agent
- 직교 초기화, 은닉층 std √2, **정책 출력 std 0.01**(초기 균등 분포 → 탐색 최대), 가치 출력 std 1.
- actor·critic 몸통 공유 없음. `get_action_and_value` 는 롤아웃(샘플링)과 업데이트(주어진 행동 재평가) 둘 다에 쓰인다.

### 롤아웃 버퍼
- 전부 `(num_steps, num_envs)` 고정 텐서. **`dones[t]` 는 "obs[t] 가 새 에피소드 첫 관측"** 이라는 의미로 한 칸 밀려 저장된다.
  그래서 GAE 에서 `dones[t+1]` 을 본다. 처음 읽을 때 가장 헷갈린 부분.

### GAE
- `delta = r_t + γ V_{t+1} (1−d_{t+1}) − V_t`, `A_t = delta + γλ (1−d_{t+1}) A_{t+1}`.
- 마지막 스텝은 `next_value`, `next_done` 으로 부트스트랩.
- `returns = advantages + values` — 가치 타깃은 TD(λ) 리턴.
- truncation 도 done 으로 처리 → 시간 초과 시 부트스트랩이 끊긴다 (엄밀히는 틀림. 영향은 EXP-006 에서 측정).

### 업데이트
- `ratio` 는 첫 에폭 첫 미니배치에서 정확히 1 이어야 한다 → 디버깅 체크포인트.
- `approx_kl = mean((r−1) − log r)` (Schulman k3 추정, 항상 ≥ 0).
- 어드밴티지 정규화는 **미니배치 단위**.
- 정책 손실: 논문 min(·) 최대화를 부호 뒤집어 max(·) 최소화.
- 가치 손실도 ε 로 클리핑 (`clip_vloss=True`).
- 기록되는 loss/KL 은 **마지막 미니배치 값**(평균 아님), clipfrac 만 전체 평균.
- `explained_variance` 는 업데이트 *전* V 로 계산 → "이번 롤아웃을 얼마나 잘 예측했었나".

## Gymnasium 1.x 에서 그대로 안 돌던 부분
1. 벡터 환경 기본 autoreset 이 `NEXT_STEP` → 종료 직후 스텝이 "행동이 무시되는 가짜 전이"가 됨. `SAME_STEP` 으로 지정해 원본 의미 보존.
2. `infos["final_info"]` 가 리스트-of-dict 에서 **dict-of-arrays + `_episode` 마스크**로 바뀜 → 원본 로깅 코드가 조용히 아무것도 안 찍음(에러도 안 남!).
3. 기본 `env_id` 가 CartPole-v1 → LunarLander-v3 로 변경.

## 읽으면서 생긴 질문 → 답
| 질문 | 답 / 확인 방법 |
|---|---|
| actor/critic 분리인데 왜 손실을 합쳐서 한 optimizer 로? | 파라미터가 겹치지 않으니 합쳐도 각자 독립 그래디언트. 코드 단순화 목적 |
| 가치 손실 클리핑 ε=0.2 는 보상 스케일 큰 LunarLander 에서 과한가? | SB3 는 기본 off. EXP-006 에서 on/off 비교 |
| 기본 하이퍼파라미터로 LunarLander 가 풀리나? | **안 풀림.** 원본도 +71 정점 후 −43 으로 붕괴(호버링) → EXP-002 |
| truncation 을 done 처리하는 영향? | γ=0.999 + 긴 에피소드에서 커짐 → EXP-006 |
