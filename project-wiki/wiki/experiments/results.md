# 실험 결과 요약 (최신)

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [protocol.md](protocol.md), [../overview.md](../overview.md)

## PRD 성공 기준 판정
| 목표 | 기준 | 결과 | 판정 | 상세 |
|---|---|---|---|---|
| SB3 기준선 | 1M, eval ≥ 200 | 253.1 / 261.6 / 251.6 (평균 **255.4**) | ✅ | `notes/01_sb3_baseline.md` |
| CleanRL 이해 | 핵심 함수 주석 | `src/ppo_annotated.py` (378줄), `notes/02_cleanrl_reading.md` | ✅ | |
| 재구현 | 3 시드 ≥ 200 | 259.1 / 274.3 / 270.9 (평균 **268.1**), 학습 mean100 도 3/3 200 도달 | ✅ | [EXP-007](EXP-007-final-default.md) |
| 원본과 비교 | 같은 하이퍼파라미터에서 수렴 스텝 차 ≤ 20% | **0%** (3 시드 곡선 비트 단위 일치, 두 설정 모두) | ✅ | [EXP-007](EXP-007-final-default.md) |
| 제거 실험 | 3종 영향 기록 | GAE −14.9 / 정규화 −12.8 / 클리핑 +3.0 (KL 8배) | ✅ | [ablation.md](ablation.md) |
| 학습 외 | 한 파일 < 500줄, 한 줄 재현 | `src/ppo.py` 458줄, `python src/ppo.py` | ✅ | |
| FR-8 | 같은 시드 → 같은 곡선 | 독립 실행·병렬 부하에서도 eval 소수점까지 동일 + `tests/test_seed.py` | ✅ | |

## 실험 이력 한눈에
| EXP | 내용 | 핵심 수치 | 결론 |
|---|---|---|---|
| [001](EXP-001-first-default-run.md) | PRD 기본값 + 정규화 | eval −75 / −84 / −101 | 실패 |
| [002](EXP-002-cleanrl-contrast.md) | CleanRL 원본 대조, 길이 분석 | 원본도 붕괴, trunc 95% | 구현 아닌 설정 문제, 호버링 |
| [003](EXP-003-tuning-round1.md) | γ / 정규화 / 길이 | 최고 110 (γ0.99, 3M) | γ 만으론 부족 |
| [004](EXP-004-tuning-round2.md) | 롤아웃·배치 확대 | 최고 193 (zoo 1M) | 롤아웃 길이가 핵심 |
| [005](EXP-005-tuning-round3.md) | 2M/3M, 3 시드 | 4×1024 2M **226 (3/3)** | 첫 PASS |
| [006](EXP-006-sb3-gap.md) | SB3 차이 분해 | LR 감소 −47, 가치 클리핑 −32 | SB3 등가 시 253 = SB3 |
| [007](EXP-007-final-default.md) | 최종 기본값 | **268.1 (3/3)**, CleanRL 과 비트 일치 | 기본값 확정 |
| [008](EXP-008-ablation.md) | 제거 실험 | 위 표 | 조건 의존성 확인 |

## 성능 메모 (ADR-004)
| 설정 | SPS (단독, 65k 스텝) |
|---|---|
| CPU, 1 스레드 (기본) | **15,160** |
| CPU, 8 스레드 | 11,939 |
| MPS | 2,450 (CPU 대비 6.2배 느림, 결과도 CPU 와 비결정적으로 다름) |
| SB3 zoo (16 env) | ~2,100 |

## 아티팩트 위치
- 최종 모델: `models/LunarLander-v3__ppo__{0,1,2}__1793160254.pt`, 제거 실험 `models/*ppo_no_*__1793160254.pt`, SB3 `models/*sb3_zoo*.zip`
- 튜닝 47개: `models/_tuning/`, `runs/_tuning/`
- 폐기된 기본값 시절: `models/_superseded/`, `runs/_superseded/`
- 표 재생성: `python scripts/summarize_runs.py runs/` / `python src/evaluate.py models/*.pt`
