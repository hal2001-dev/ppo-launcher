# EXP-006: SB3 와의 성능 차이 분해 — truncation 부트스트랩 / LR 감소 / 가치 클리핑

**날짜**: 2026-10-28
**상태**: done
**설정**: zoo 하이퍼파라미터(16×1024, γ0.999, λ0.98, 미니배치 64, lr 3e-4) + 스위치 조합, 1M 스텝
**시드**: 0, 1
**run 디렉터리**: `runs/_tuning/*__{N_zoo_tb,O_zoo_const,Q_zoo_sb3eq,R_zoo_constlr,S_zoo_novclip,V_prd_novclip,T_4x1024_novclip,W_4x1024_novclip_lr3}__*`

## 배경
- SB3 PPO (zoo 설정, seed 0, 1M): **eval 253.1 ± 21.6** (`notes/01_sb3_baseline.md`)
- 재구현 같은 하이퍼파라미터 (EXP-004 F): 193.3 / 170.8
- 같은 하이퍼파라미터인데 왜 60~80점 차이가 나나? SB3 와 재구현(=CleanRL)의 **구현 디테일 차이**를 목록화:

| 디테일 | SB3 PPO | 재구현 / CleanRL |
|---|---|---|
| 시간 제한(truncation) | `γ·V(final_obs)` 를 보상에 더해 부트스트랩 | done 으로 처리(부트스트랩 끊음) |
| 학습률 | zoo 설정은 **고정** 3e-4 | **선형 감소** → 0 (FR-7) |
| 가치 손실 클리핑 | 기본 **없음** (`clip_range_vf=None`) | **있음** (ε=0.2) |
| 네트워크 / 초기화 / 어드밴티지 정규화 / grad clip | 64-64 Tanh 분리, 직교 초기화, 미니배치 정규화, 0.5 | 동일 |

## 구현 변경
`bootstrap_truncation` 스위치 추가 (`src/ppo.py` 롤아웃 루프): truncated & not terminated 인 서브 환경에 대해
`infos["final_obs"]` 로 V 를 계산해 `reward += γ·V(s_final)`. SAME_STEP autoreset 이라 final_obs 가 info 로 온다.

## 명령
```sh
Z="--gamma 0.999 --gae-lambda 0.98 --num-envs 16 --num-steps 1024 --num-minibatches 256 --learning-rate 3e-4"
P() { python src/ppo.py --run-dir runs/_tuning --norm-reward false "$@"; }
P --exp-name N_zoo_tb      ${=Z} --bootstrap-truncation true
P --exp-name O_zoo_const   ${=Z} --anneal-lr false --clip-vloss false
P --exp-name Q_zoo_sb3eq   ${=Z} --bootstrap-truncation true --anneal-lr false --clip-vloss false
P --exp-name R_zoo_constlr ${=Z} --anneal-lr false
P --exp-name S_zoo_novclip ${=Z} --clip-vloss false
```

## 결과 (eval 100ep, seed 0 / 1)
| 변형 | trunc 부트스트랩 | LR | 가치 클리핑 | eval | 평균 |
|---|---|---|---|---|---|
| F (EXP-004) | off | 감소 | on | 193.3 / 170.8 | 182.1 |
| N | **on** | 감소 | on | 178.3 / 205.7 | 192.0 |
| R | off | **고정** | on | 229.4 / 228.9 | 229.2 |
| S | off | 감소 | **off** | 212.2 / 215.8 | 214.0 |
| O | off | 고정 | off | **255.8 / 253.4** (std ≈ 21) | **254.6** |
| Q (SB3 등가) | on | 고정 | off | 252.6 / 253.6 | 253.1 |
| SB3 zoo | (on) | 고정 | off | 253.1 (seed 0) | |

2×2 요약 (truncation 부트스트랩 off):

| | LR 감소 | LR 고정 |
|---|---|---|
| 가치 클리핑 on | 182 | 229 |
| 가치 클리핑 off | 214 | **255** |

부가 확인 (1M):

| 변형 | 설정 | eval | 해석 |
|---|---|---|---|
| V | PRD 기본(4×128, γ0.99) + 가치 클리핑 off | −64.7 / 13.5 | 가치 클리핑만으로는 PRD 기본값을 못 살림 → 근본 원인은 지평/롤아웃 (EXP-002) |
| T | 4×1024 + 가치 클리핑 off, lr 2.5e-4 | 202.1 / 66.8 / 100.2 | 4 env 로 1M 은 시드 편차 큼 |
| W | 4×1024 + 가치 클리핑 off, lr 3e-4 | 214.0 / 185.9 / 109.4 | 동일 |

## 해석
- **재구현이 SB3 와 같은 디테일을 쓰면(Q) SB3 와 같은 점수(253)** → 재구현 PPO 코어는 정확하다는 강한 증거.
- 차이의 주원인은 **LR 선형 감소**(+47), 그다음 **가치 손실 클리핑**(+32). truncation 부트스트랩은 이 설정에서 거의 영향 없음(±).
  - LR 감소: 1M 예산에서 후반 절반의 업데이트가 사실상 멈춘다.
  - 가치 클리핑 ε=0.2: LunarLander 리턴 스케일(수백)에서 V 가 한 업데이트에 0.2 밖에 못 움직임 → critic 이 느려짐.
    (CleanRL 은 원래 Atari 처럼 보상이 클리핑된 환경 기준으로 정한 값)
- 그러나 **FR-7 은 LR 선형 감소를 요구** → 끌 수 없다. 대신 가치 클리핑(FR 에 없음)은 끌 수 있다.

## 다음 행동
4 env 유지 + LR 감소 유지 + 가치 클리핑 off + 2M 을 3 시드로 검증 → [EXP-007](EXP-007-final-default.md)
