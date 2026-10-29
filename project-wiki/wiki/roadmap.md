# Roadmap — 단계별 마일스톤

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [overview.md](overview.md), [experiments/results.md](experiments/results.md)

각 단계는 통과 기준을 만족해야 다음으로 넘어간다. PRD 의 주차 표 본문이 비어 있어 기간 대신 **통과 기준과 실제 완료일**을 기록한다.

| 단계 | 내용 | 통과 기준 | 산출물 | 상태 |
|---|---|---|---|---|
| Stage 0 | 환경 셋업 | `gym.make("LunarLander-v3")` OK, SB3 1분 내 첫 로그 | pyproject.toml, uv.lock | **done** 2026-10-28 |
| Stage 1 | SB3 기준선 | SB3 PPO 1M → eval 100ep ≥ 200 | `src/baseline_sb3.py`, `notes/01_sb3_baseline.md` | **done** 2026-10-28 |
| Stage 2 | CleanRL 읽기 | 핵심 함수마다 주석 | `src/ppo_annotated.py`, `notes/02_cleanrl_reading.md` | **done** 2026-10-28 (주석 1차 — 직접 읽으며 `[Q]` 보강 권장) |
| Stage 3 | 재구현 | 3 시드 eval ≥ 200, FR-1~8 검증, <500줄, 한 줄 재현 | `src/ppo.py`, `tests/` | **done** 2026-10-28 |
| Stage 3b | 원본 비교 | 동일 하이퍼파라미터에서 수렴 스텝 차 ≤ 20% | [experiments/results.md](experiments/results.md) | **done** 2026-10-28 |
| Stage 4 | 제거 실험 3종 | GAE / 보상 정규화 / 클리핑 영향 기록 | `configs/ablation_*.json`, `notes/04_ablation.md` | **done** 2026-10-28 |
| Stage 5 | 확장 (후속 프로젝트) | 범위 밖 — 완료 후 착수 | [related-projects.md](related-projects.md) | todo |

## 학습자 TODO (스스로 해볼 것)
이 프로젝트의 목적은 코드 수준 이해이므로, 자동 구성된 산출물 위에서 다음을 직접 해 보길 권장:
- [ ] `src/ppo_annotated.py` 를 처음부터 읽으며 자기 말로 `[Q]` 주석 추가
- [ ] `src/ppo.py` 를 보지 않고 빈 파일에서 다시 작성 → `tests/` 통과 여부로 확인 (PRD "참고 없이 재구현")
- [ ] `ratio == 1` (첫 미니배치) 를 assert 로 넣어 보기
- [ ] `bootstrap_truncation=true` 를 기본값 위에서 3 시드로 돌려 효과 확인 (EXP-006 은 zoo 설정에서만 확인)
- [ ] `--device mps` SPS 를 직접 비교해 ADR-004 확인
- [ ] 엔트로피 계수 0 / 0.05 로 바꿔 착지 후 "엔진 깔짝임"(truncation 80%) 이 줄어드는지 관찰
