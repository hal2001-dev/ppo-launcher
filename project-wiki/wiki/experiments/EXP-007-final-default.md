# EXP-007: 최종 기본값 확정 + 3 시드 본 실험 + 원본 비교

**날짜**: 2026-10-28
**상태**: done — **Stage 3 / 3b 통과**
**설정**: `configs/default.json` (= `python src/ppo.py`)
**시드**: 0, 1, 2
**run 디렉터리**: `runs/LunarLander-v3__ppo__{0,1,2}__1793160254`, 모델 `models/LunarLander-v3__ppo__*__1793160254.pt`

## 경과
1. **후보 X** (4×1024, γ0.999, λ0.98, LR 감소, 가치 클리핑 off, 보상 정규화 off, 2M) 3 시드 → 261.3 / 268.5 / 236.0 (평균 255.3, 3/3)
2. X 를 기본값으로 바꾸고 `python src/ppo.py --seed {0,1,2}` 재실행 → **261.34 / 268.48 / 235.98 소수점까지 동일** (FR-8 전체 규모 확인)
3. 같은 배치에서 돌린 제거 실험 중 "보상 정규화 **켜기**"가 268.1 로 더 좋음 → [EXP-008](EXP-008-ablation.md)
4. `norm_reward=True` 를 기본값에 포함(ADR-005 갱신) 후 기본값·제거 실험 12개 재실행 → 아래가 최종

## 최종 기본값
| 키 | PRD 초기값 | 최종 | 근거 |
|---|---|---|---|
| num_envs | 4 | 4 | FR-1 유지 |
| num_steps | 128 | **1024** | EXP-002/004 (호버링, 롤아웃 길이) |
| num_minibatches | 4 | **32** | 미니배치 크기 128 유지 |
| gamma | 0.99 | **0.999** | EXP-002/003 (유효 지평) |
| gae_lambda | 0.95 | **0.98** | zoo 설정 |
| clip_vloss | (CleanRL true) | **false** | EXP-006 |
| norm_reward | (없음) | **true** | EXP-008 |
| total_timesteps | 1M (SB3 기준) | **2M** | EXP-005 (4 env 로 1M 은 시드 편차 큼) |
| 나머지 (ε 0.2, ent 0.01, K 4, lr 2.5e-4 선형 감소, grad clip 0.5, 어드밴티지 정규화) | | 동일 | FR-5~7 |

## 결과 — 재구현 (`python src/ppo.py --seed s`)
| seed | eval 100ep (학습 종료 시, 평가 시드 10000+s) | `evaluate.py` (평가 시드 10000) | 학습 mean100 200 도달 | 최종 mean100 | 최종 trunc% |
|---|---|---|---|---|---|
| 0 | **259.07** ± 28.63 | 259.07 | 570k | 275 | 0 |
| 1 | **274.29** ± 21.56 | 274.53 | 668k | 265 | 2 |
| 2 | **270.86** ± 21.74 | 271.20 | 1,185k | 274 | 3 |
| **평균** | **268.07** | 268.27 | **808k** | 271 | |

- approx_kl 평균 0.003~0.004 (최대 0.023), clipfrac ~0.016, explained_variance(후반) 0.94~0.96 — 전부 PRD 목표 범위.
- 소요: 2M 스텝 ≈ 2분 15초 (단독), 12개 병렬 시 ≈ 3분.
- **보상 정규화를 켜자 학습 중 truncation 이 80% → 0~3% 로 사라짐** (착지 후 엔진 깔짝임 해소) → 학습 mean100 이 eval 과 비슷해짐.

## 원본(CleanRL)과 비교 — PRD "수렴 스텝 차 ≤ 20%"
`src/ppo_annotated.py` 에 같은 하이퍼파라미터를 넣어 실행:
```sh
python src/ppo_annotated.py --exp-name ppo_annotated_tuned --seed $s --total-timesteps 2000000 \
  --num-steps 1024 --num-minibatches 32 --gamma 0.999 --gae-lambda 0.98 --no-clip-vloss
```
(CleanRL 원본에는 보상 정규화가 없으므로 비교 대상은 재구현의 `ablation_no_reward_norm`)

| seed | 지표 | CleanRL 원본 | 재구현 | 차이 |
|---|---|---|---|---|
| 0 | mean100 ≥ 0 / 100 / 150 / 200 | 117k / 388k / 413k / 477k | 117k / 388k / 413k / 477k | **0%** |
| 1 | 〃 | 131k / 366k / 390k / 576k | 131k / 366k / 390k / 576k | **0%** |
| 2 | 〃 | 124k / 437k / 미도달 / 미도달 | 124k / 437k / 미도달 / 미도달 | **0%** |

CleanRL 기본값(`configs/cleanrl_match.json`) 비교도 3 시드 모두 동일:

| seed | 지표 | CleanRL 원본 | 재구현 |
|---|---|---|---|
| 0 | mean100 ≥ 0 / 정점 / 최종 / trunc% | 421k / 91 / −51 / 72 | 421k / 91 / −51 / 72 |
| 1 | 〃 | 358k / 40 / −38 / 46 | 358k / 40 / −38 / 46 |
| 2 | 〃 | 153k / 68 / −21 / 10 | 153k / 68 / −21 / 10 |

**곡선이 비트 단위로 일치한다.** 이유: 두 구현이 같은 순서로 같은 난수를 소비하기 때문
(torch 시드 → 초기화·Categorical 샘플, numpy 시드 → 미니배치 셔플, env reset seed). 재구현은 연산을 함수로
나눴을 뿐 순서와 수식이 같다. → PRD 기준(20% 이내) 충족, **재구현의 정확성에 대한 가장 강한 증거**.

> 학습 포인트: 일치가 깨졌다면 "어느 iteration 의 어느 텐서부터 달라지는가"를 이분 탐색하면 된다(PRD 리스크 대응 "CleanRL 과 변수별 값 비교").

## 다음 행동
제거 실험 3종 정리 → [EXP-008](EXP-008-ablation.md), 요약 → [results.md](results.md)
