# Overview — PPO LunarLander

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [roadmap.md](roadmap.md), [experiments/results.md](experiments/results.md), [log.md](log.md)

## 요약
LunarLander-v3(이산 4 액션)에서 PPO 를 단일 파일로 직접 구현해 100 에피소드 평균 200점 이상을 달성하는 학습 프로젝트.
목적은 라이브러리 사용이 아니라 **롤아웃·GAE·클리핑 손실·엔트로피 보너스의 코드 수준 이해**.
Mac Studio CPU 위주, MPS 는 옵션. 원문: `raw/requirements/2026-10-28-prd.md`

## 현재 상태 (2026-10-28)
**PRD 의 모든 성공 기준 충족.** → [experiments/results.md](experiments/results.md)

| 항목 | 값 |
|---|---|
| 재구현 eval (3 시드) | 259.1 / 274.3 / 270.9 — 평균 **268.1** |
| SB3 기준선 eval (3 시드) | 253.1 / 261.6 / 251.6 — 평균 255.4 |
| CleanRL 원본과의 수렴 스텝 차 | **0%** (곡선 비트 단위 일치) |
| `src/ppo.py` | 458줄, `python src/ppo.py` 한 줄 재현 (2M 스텝 ≈ 2분 15초) |
| 테스트 | 12 passed (GAE 6, 버퍼/네트워크 4, 시드 재현 2) |
| 포트 | TensorBoard 3005 (상위 PORTS.md 등록) |

## 진행 경과 한 줄 요약
PRD 기본값으로 첫 학습 실패(−87) → CleanRL 원본도 똑같이 실패하는 것 확인(설정 문제) →
episodic_length 로 **호버링** 진단 → γ·롤아웃 길이 탐색으로 첫 PASS(226) →
SB3 와의 차이 분해(LR 감소·가치 클리핑) → 최종 기본값(268) → 제거 실험.
자세한 과정: [issues/resolved/ISSUE-001.md](issues/resolved/ISSUE-001.md), 실험 페이지 EXP-001~008, [log.md](log.md).

## 이번에 배운 것 (핵심 6가지)
1. **같이 망가지면 설정 문제다.** 원본 구현을 같은 조건으로 돌리는 것이 가장 빠른 버그 판별법 (EXP-002).
2. **episodic_return 만 보면 원인을 모른다.** 길이·truncation 비율이 호버링/엔진 깔짝임을 드러냈다.
3. **1/(1−γ) 와 롤아웃 길이를 에피소드 길이와 비교하라.** CleanRL 기본값은 CartPole 용이다.
4. **"같은 하이퍼파라미터"라도 구현 디테일이 60~80점을 가른다** (LR 감소, 가치 클리핑, truncation 처리) (EXP-006).
5. **ablation 결론은 조건 의존적이다.** 보상 정규화는 한 설정에선 손해, 다른 설정에선 이득 (EXP-008).
6. **클리핑은 평균이 아니라 최악 시드를 지킨다.** 제거하면 빠르지만 KL 8배, 조건 따라 붕괴 (EXP-008).

## PRD 대비 달라진 점 (의도적)
- 기본 하이퍼파라미터 변경 (ADR-005): num_steps 1024, γ 0.999, λ 0.98, 가치 클리핑 off, 보상 정규화 on, 2M 스텝.
  FR 에 명시된 메커니즘(4 env, ε 0.2, ent 0.01, K 4, LR 선형 감소, grad clip 0.5)은 그대로.
- TensorBoard 태그는 CleanRL 원본 이름(`losses/approx_kl` 등) (ADR-002).
- 설치는 `uv pip install` 대신 `pyproject.toml` + `uv sync`.

## 할 일
- [ ] (학습자) [roadmap.md](roadmap.md) 의 "학습자 TODO" — 특히 `ppo.py` 를 보지 않고 다시 작성해 테스트 통과시키기
- [ ] git 저장소 초기화 + 첫 커밋 (PRD "ppo_annotated.py 커밋" 기준) — 아직 git 아님
- [ ] (선택) 클리핑 제거 실험 시드 확대, 정규화-호버링 가설 검증 (EXP-008 다음 행동)
- [ ] 후속 프로젝트(트레이딩 환경) 착수 시 [related-projects.md](related-projects.md) 갱신
