# EXP-003: 튜닝 1차 — γ 0.999 / 보상 정규화 / 학습 길이

**날짜**: 2026-10-28
**상태**: done
**설정**: PRD 기본(4×128, λ0.95) 위에 CLI override
**시드**: 0, 1 (8개 실행 병렬)
**run 디렉터리**: `runs/_tuning/*__{g999,g999_rn,g999_2m,g99_3m}__*`

## 가설
- A: γ=0.999 만 올리면 착륙 보너스가 보여 호버링을 벗어난다.
- B: γ=0.999 에서는 보상 정규화가 도움이 될 수도.
- C: γ=0.999 + 2M 스텝이면 충분.
- E: γ 는 그대로(0.99) 두고 3M 스텝으로 오래 돌리면 결국 탈출한다.

## 명령
```sh
P() { python src/ppo.py --run-dir runs/_tuning "$@"; }
P --exp-name g999    --gamma 0.999 --norm-reward false                          --seed $s   # A
P --exp-name g999_rn --gamma 0.999 --norm-reward true                           --seed $s   # B
P --exp-name g999_2m --gamma 0.999 --norm-reward false --total-timesteps 2000000 --seed $s  # C
P --exp-name g99_3m  --gamma 0.99  --norm-reward false --total-timesteps 3000000 --seed $s  # E
```
> 첫 시도는 `P="python …"; $P` 형태로 실행했다가 **zsh 가 변수를 단어 분리하지 않아 8개 전부 즉시 실패**.
> 함수 정의로 바꿔 재실행 → [troubleshooting](../troubleshooting/common.md) 에 등재.

## 결과
| 변형 | γ | norm_reward | steps | 최종 mean100 (s0 / s1) | eval 100ep (s0 / s1) |
|---|---|---|---|---|---|
| A | 0.999 | off | 1M | +81 / +44 | −5.2 / −37.3 |
| B | 0.999 | on | 1M | +15 / −22 | −29.3 / −68.9 |
| C | 0.999 | off | 2M | −24 / −100 | −23.1 / −95.2 |
| E | 0.99 | off | 3M | +102 / +88 | **+110.4 / +96.7** |

## 해석
- γ 만 올려서는 부족(A). 분산이 커져(std ≈ 100~126) 4×128=512 짜리 작은 배치로는 불안정.
  C 처럼 오래 돌리면 오히려 더 망가지는 시드도 있음 → **배치가 작아서 신호/잡음비가 낮다**는 쪽이 유력.
- 보상 정규화는 이 조합에서도 일관되게 손해(B < A).
- E(γ0.99, 3M)가 가장 좋지만 eval ~100 으로 목표 200 과 거리가 멀다.
- 공통 병목 후보: **롤아웃 길이 128 이 에피소드 길이(수백 스텝)에 비해 너무 짧다** → GAE 가 대부분 부트스트랩 V 에 의존.

## 다음 행동
SB3 zoo 의 LunarLander 설정(16 env × 1024 스텝, γ0.999, λ0.98, 미니배치 64)을 기준으로
롤아웃·배치 크기를 키운다 → [EXP-004](EXP-004-tuning-round2.md)
