# EXP-005: 튜닝 3차 — 학습 길이 2M/3M, 3 시드 확인

**날짜**: 2026-10-28
**상태**: done — **첫 PASS**
**설정**: 공통 `--gamma 0.999 --gae-lambda 0.98 --norm-reward false`
**시드**: 0, 1, 2 (9개 병렬)
**run 디렉터리**: `runs/_tuning/*__{K_4x1024_2m,L_4x1024_3m,M_zoo_2m}__*`

## 가설
EXP-004 에서 LR 이 0 이 되는 1M 시점에 아직 성능이 오르고 있었다 → 스텝 수만 늘리면 200 도달.

## 명령
```sh
P() { python src/ppo.py --run-dir runs/_tuning --gamma 0.999 --gae-lambda 0.98 --norm-reward false "$@"; }
I="--num-envs 4 --num-steps 1024 --num-minibatches 32"
F="--num-envs 16 --num-steps 1024 --num-minibatches 256 --learning-rate 3e-4"
P --exp-name K_4x1024_2m ${=I} --total-timesteps 2000000 --seed $s
P --exp-name L_4x1024_3m ${=I} --total-timesteps 3000000 --seed $s
P --exp-name M_zoo_2m    ${=F} --total-timesteps 2000000 --seed $s
```

## 결과 (`scripts/summarize_runs.py` + 종료 시 eval)
| 변형 | seed | eval 100ep | 판정 | 학습 mean100 | 학습 중 truncation% | mean100 200 도달 스텝 |
|---|---|---|---|---|---|---|
| K (4×1024, 2M) | 0 | 208.6 ± 71.5 | PASS | 132.1 | 79 | 미도달 |
| | 1 | 231.7 ± 59.7 | PASS | 135.1 | 76 | 미도달 |
| | 2 | 238.9 ± 60.4 | PASS | 130.8 | 81 | 미도달 |
| | **평균** | **226.4** | **3/3** | | | |
| L (4×1024, 3M) | 0 | 223.3 ± 90.2 | PASS | 201.6 | 8 | 2,229,476 |
| | 1 | 234.6 ± 58.1 | PASS | 128.8 | 80 | 미도달 |
| | 2 | 229.4 ± 64.8 | PASS | 122.2 | 77 | 미도달 |
| | **평균** | **229.1** | **3/3** | | | |
| M (zoo 16×1024, 2M) | 0 | 213.8 ± 37.1 | PASS | 121.4 | 98 | 미도달 |
| | 1 | 174.8 ± 62.1 | FAIL | 120.8 | 93 | 미도달 |
| | 2 | 210.7 ± 47.5 | PASS | 106.8 | 91 | 미도달 |
| | **평균** | **199.8** | 2/3 | | | |

## 해석
- **K 가 최선의 비용 대비 결과**: 2M 스텝(약 2.5분)에 3/3 PASS, FR-1 기본값 num_envs=4 유지. L(3M)은 평균이 +2.7 높을 뿐.
- **학습 중 mean100 과 eval 의 큰 격차** (130 vs 226): truncation 이 80% 가까이 됨.
  argmax 정책으로 30 에피소드를 따로 돌려 보니(EXP-004 F seed 0 모델) 평균 길이 469, truncation 6.7%, 끝날 때 두 다리 접지 87%.
  → 확률적 정책은 **착지한 뒤에도 엔진을 조금씩 계속 분사**해 lander 가 "잠들지(not awake)" 못하고 1000 스텝까지 간다.
  호버링(EXP-002)과 다르다: 리턴은 양수(+100 대)이고 착지는 성공.
- 이 상태에서는 truncation 을 terminal 로 취급하는 것(CleanRL 방식)이 가치 타깃을 왜곡한다:
  γ=0.999 면 1000 스텝 시점의 남은 가치가 크다. → [EXP-006](EXP-006-sb3-gap.md) 에서 truncation 부트스트랩으로 검증.

## 다음 행동
- K 를 기본값 후보로 채택 (ADR-005).
- SB3 zoo 가 같은 설정으로 1M 에서 253점인데 재구현 M/F 는 그보다 낮음 → 차이 분리 실험(EXP-006).
