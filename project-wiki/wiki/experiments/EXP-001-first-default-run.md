# EXP-001: 첫 학습 — PRD 기본 하이퍼파라미터 + 보상 정규화

**날짜**: 2026-10-28
**상태**: done (실패 → [ISSUE-001](../issues/resolved/ISSUE-001.md) 로 이어짐)
**설정**: 당시 DEFAULTS (= PRD FR 기본값 + `norm_reward=true`)
**시드**: 0, 1, 2 (병렬)
**run 디렉터리**: `runs/LunarLander-v3__ppo__{0,1,2}__1793158912`

## 목적
`src/ppo.py` 를 처음 작성한 직후, PRD 기본값 그대로 200점이 나오는지 확인.

## 설정 (핵심만)
| 키 | 값 | 출처 |
|---|---|---|
| num_envs × num_steps | 4 × 128 = 512 | FR-1, FR-3 |
| gamma / gae_lambda | 0.99 / 0.95 | FR-4 |
| clip_coef / ent_coef | 0.2 / 0.01 | FR-5 |
| update_epochs / num_minibatches | 4 / 4 | FR-6 |
| learning_rate (anneal) | 2.5e-4 → 0 | FR-7 |
| norm_reward | **true** | ablation 대상이라 "켠 상태가 기본"이라고 해석 |
| total_timesteps | 1,000,000 | |

```sh
for s in 0 1 2; do python src/ppo.py --seed $s & done; wait
```

## 사전 점검 (학습 전)
- 스모크 테스트 20k 스텝: 에러 없음, **SPS ≈ 14,600** (1M ≈ 70초 예상)
- `[FR-2] params actor=4996 critic=4801`, `[FR-1] obs shape=(4, 8)` 로그 확인
- 단위 테스트 10개 통과 (GAE 손계산 포함)

## 결과
| seed | 최종 mean100 (학습) | eval 100ep (argmax) | 판정 |
|---|---|---|---|
| 0 | −65.41 | **−74.69** ± 28.26 | FAIL |
| 1 | −74.54 | **−84.31** ± 41.15 | FAIL |
| 2 | −87.14 | **−101.06** ± 46.68 | FAIL |

seed 0 학습 곡선 (mean100):

| step | 512 | 105k | 212k | 318k | 426k | 534k | 640k | 746k | 853k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|
| mean100 | −170 | **+8.5** | +8.0 | −60 | −62 | −65 | −88 | −55 | −76 | −65 |

## 관찰
- 무작위(−170) → **+8 까지는 오른다** → 학습 신호 자체는 있다. 코드가 완전히 깨진 건 아님.
- 30만 스텝 이후 **오히려 하락**해서 −60 대에 고착. PRD 리스크 표의 "50만 스텝 후에도 보상 0 이하"에 정확히 해당.
- 후반 `approx_kl` = 0.0000 은 LR 이 0 에 가까워서이지 버그 신호는 아님.

## 해석 / 다음 행동
"돌아가는데 안 배우는" 조용한 버그인지, 설정 문제인지 먼저 갈라야 한다.
PRD 대응 순서대로 → ① GAE 테스트(이미 통과) ② **CleanRL 원본과 같은 조건에서 비교** → [EXP-002](EXP-002-cleanrl-contrast.md)
