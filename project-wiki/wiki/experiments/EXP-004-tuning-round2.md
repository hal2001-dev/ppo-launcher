# EXP-004: 튜닝 2차 — 롤아웃 길이 / 환경 수 / 미니배치

**날짜**: 2026-10-28
**상태**: done
**설정**: 공통 `--gamma 0.999 --gae-lambda 0.98 --norm-reward false`, 1M 스텝
**시드**: 0, 1
**run 디렉터리**: `runs/_tuning/*__{F_zoo,G_16x256,I_4x1024,J_4x512}__*`

## 가설
롤아웃을 길게(≥256), 배치를 크게 하면 γ0.999 의 분산을 감당할 수 있다.
FR-1 기본값 `num_envs=4` 를 지킬 수 있는지도 같이 본다 (I, J).

## 명령 / 변형
| 변형 | num_envs × num_steps | 배치 | 미니배치 수 (크기) | lr |
|---|---|---|---|---|
| F (zoo 재현) | 16 × 1024 | 16,384 | 256 (64) | 3e-4 |
| G | 16 × 256 | 4,096 | 16 (256) | 2.5e-4 |
| I | **4** × 1024 | 4,096 | 32 (128) | 2.5e-4 |
| J | **4** × 512 | 2,048 | 16 (128) | 3e-4 |

```sh
P() { python src/ppo.py --run-dir runs/_tuning --gamma 0.999 --gae-lambda 0.98 --norm-reward false "$@"; }
P --exp-name F_zoo    --num-envs 16 --num-steps 1024 --num-minibatches 256 --learning-rate 3e-4 --seed $s
P --exp-name G_16x256 --num-envs 16 --num-steps 256  --num-minibatches 16                       --seed $s
P --exp-name I_4x1024 --num-envs 4  --num-steps 1024 --num-minibatches 32                       --seed $s
P --exp-name J_4x512  --num-envs 4  --num-steps 512  --num-minibatches 16 --learning-rate 3e-4  --seed $s
```

## 결과
| 변형 | 최종 mean100 (s0 / s1) | eval 100ep (s0 / s1) | eval 평균 |
|---|---|---|---|
| F | +110 / +109 | **193.3 / 170.8** | 182.1 |
| G | +99 / +106 | −19.7 / 147.6 | 64.0 |
| I | +116 / +126 | 170.6 / 123.3 | 147.0 |
| J | +92 / +116 | 138.2 / 142.7 | 140.5 |
| (참고) EXP-003 E | +102 / +88 | 110.4 / 96.7 | 103.6 |

SPS: G 는 16 env 라 23k SPS 로 가장 빠름, 나머지 ~14k.

## 해석
- **롤아웃 길이가 결정적.** 128 → 512/1024 로 늘린 모든 변형이 1차 최고치(E, 3M 스텝)를 1M 만에 넘김.
- F(zoo)가 최고지만 1M 에서 아직 200 미만. 학습 중 mean100(~110)과 eval(~180) 차이가 큼 → 확률적 정책의 탐색 노이즈.
- G 의 seed 0 붕괴(−19.7): 미니배치 256 + 16 epoch-step 은 시드 민감.
- 1M 마지막에 LR 이 0 → 아직 오르는 중에 학습이 멈춘 것으로 보임 → **스텝 수를 늘리면 200 도달 가능성**.

## 다음 행동
F(zoo)와 I(4 env 유지)를 2M/3M 으로 늘려 **3 시드** 로 확인 → [EXP-005](EXP-005-tuning-round3.md)
