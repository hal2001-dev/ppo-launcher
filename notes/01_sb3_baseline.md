# 01. SB3 기준선

**날짜**: 2026-10-28 · **통과 기준**: SB3 PPO 1M 스텝 → 100 에피소드(결정적) 평균 ≥ 200 → **PASS (3/3)**

## 명령
```sh
python src/baseline_sb3.py --seeds 0     # 각 시드 별도 프로세스로 실행
python src/baseline_sb3.py --seeds 1
python src/baseline_sb3.py --seeds 2
```

## 하이퍼파라미터 (`--preset zoo`, rl-baselines3-zoo LunarLander 항목)
| n_envs | n_steps | batch_size | n_epochs | gamma | gae_lambda | ent_coef | lr | clip_range | 가치 클리핑 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 1024 | 64 | 4 | 0.999 | 0.98 | 0.01 | 3e-4 (고정) | 0.2 | 없음 |

## 결과
| seed | eval 100ep (argmax) | 학습 ep_rew_mean @250k / 500k / 750k / 1M | 최종 explained_variance | 소요 |
|---|---|---|---|---|
| 0 | **253.12 ± 21.60** | −22.6 / 73.1 / 142.2 / 151.5 | 0.96 | 463 s |
| 1 | **261.57 ± 22.03** | 6.2 / 126.6 / 143.9 / 259.2 | 0.73 | 477 s |
| 2 | **251.55 ± 24.34** | −8.1 / 107.4 / 127.7 / 148.8 | 0.98 | ~470 s |
| **평균** | **255.41** | | | |

- 학습 중 ep_rew_mean(확률적 정책)은 1M 시점에도 150 안팎인데 eval(argmax)은 250+ → 결정적 정책이 훨씬 깔끔하게 착지한다.
- SB3 는 ~2,100 SPS (16 env, SB3 오버헤드) — 재구현(4 env, 1 스레드)의 약 1/7.
- 첫 로그는 1~2초 내 출력 (PRD 검증 기준 "1분 내 첫 로그" 충족).

## 이 기준선이 이후 단계에 준 것
- "같은 하이퍼파라미터인데 재구현이 60~80점 낮다" → 구현 디테일 차이 분해 실험(위키 EXP-006)의 출발점.
  결론: SB3 와 같은 디테일(LR 고정, 가치 클리핑 없음)을 쓰면 재구현도 253점 → 재구현 PPO 코어가 정확하다는 근거.

## TensorBoard
`runs/LunarLander-v3__sb3_zoo__<seed>__*` — `rollout/ep_rew_mean`, `charts/episodic_return`(CleanRL 이름 호환), `train/*`.
