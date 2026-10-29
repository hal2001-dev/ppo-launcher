# Architecture Decision Records

**상태**: active
**마지막 업데이트**: 2026-10-28

---

## ADR-001: 단일 파일 + 순수 함수 분리
**날짜**: 2026-10-28
**상태**: accepted
### 배경
PRD: `src/ppo.py` 는 500줄 이내, 단독 실행. 동시에 FR-3/FR-4 는 단위 테스트를 요구.
### 선택지
1. CleanRL 처럼 `__main__` 한 블록 — 테스트 불가
2. 패키지로 모듈 분할 — "단일 파일" 위배
3. 한 파일 안에서 `RolloutBuffer`, `compute_gae`, `ppo_update`, `train`, `evaluate` 로 분리
### 결정
3. 테스트는 `pyproject.toml` 의 `pythonpath = ["src"]` 로 `from ppo import …`.
### 결과
테스트 12개, 파일 약 460줄. 후속 프로젝트(트레이딩 환경)에서 함수 단위로 재사용 가능.

---

## ADR-002: TensorBoard 태그는 CleanRL 원본 이름을 따른다
**날짜**: 2026-10-28
**상태**: accepted
### 배경
PRD 표는 `charts/approx_kl`, `charts/explained_variance` 로 적었지만, 같은 PRD 가 "지표 이름은 CleanRL 과 동일하게"를 요구. CleanRL 원본은 `losses/approx_kl`, `losses/clipfrac`, `losses/explained_variance`.
### 결정
CleanRL 원본 이름을 따른다 (곡선 겹쳐 보기가 목적이므로). SB3 기준선에도 `charts/episodic_return` 을 콜백으로 추가 기록.
### 결과
`ppo_annotated.py` 와 `ppo.py` 곡선이 TensorBoard 에서 같은 패널에 겹친다.

---

## ADR-003: TensorBoard 포트 3005, 0.0.0.0 바인딩
**날짜**: 2026-10-28
**상태**: accepted
### 배경
상위 `PORTS.md` 규칙: Web 대역 3000부터 다음 빈 번호. 3000~3004 점유 → 3005. TensorBoard 기본 6006 은 대역 밖.
springboot-4.1.1 은 공개된 고정 비밀번호 때문에 127.0.0.1 바인딩.
### 결정
`scripts/tensorboard.sh` 로 3005 고정, `0.0.0.0` (LAN 의 macstudio 에서 확인 가능). TensorBoard 는 읽기 전용이고 자격증명이 없다.
### 결과
PORTS.md 점유 표에 행 추가, "다음 프로젝트" 예시를 web 3006 으로 갱신. → [deployment/ports.md](../deployment/ports.md)

---

## ADR-004: 기본 디바이스 CPU, torch 1 스레드
**날짜**: 2026-10-28
**상태**: accepted
### 배경
64-64 MLP, 배치 128 미니배치 → GPU 연산 이득보다 전송 오버헤드가 큼. 환경 스텝이 병목.
### 결정
`device=cpu`, `torch_threads=1`. `--device mps` 는 옵션.
### 결과
단독 실행 ≈ 14~15k SPS. 1 스레드라 여러 시드를 코어마다 병렬로 돌릴 수 있고(16코어에서 15개 동시 실행), 결과가 결정적.
MPS 측정치는 [experiments/results.md](../experiments/results.md) 참고.

---

## ADR-005: 기본 하이퍼파라미터를 PRD 초기값에서 변경
**날짜**: 2026-10-28
**상태**: accepted
### 배경
PRD FR 의 괄호 기본값(4×128, γ0.99, λ0.95) = CleanRL 기본값(CartPole 기준).
이 값으로는 재구현과 CleanRL 원본 **모두** 호버링 국소 최적에 빠져 eval −75 ~ −101 ([ISSUE-001](../issues/resolved/ISSUE-001.md), EXP-001~002).
PRD 최종 목표(직접 구현 PPO 가 200점 이상, `python ppo.py` 한 줄 재현)를 만족하려면 기본값을 바꿔야 한다.
### 선택지 (EXP-003 ~ EXP-007, 총 40여 회 실행)
| 후보 | eval 3시드 평균 | FR 준수 | 비고 |
|---|---|---|---|
| PRD 기본 | −87 | ✅ | 호버링 |
| zoo 16×1024, LR 고정, 가치 클리핑 off, 1M | 254.6 (2시드) | ❌ FR-7(LR 감소), FR-1 기본 4 | SB3 와 동일 |
| zoo 16×1024, LR 감소, 가치 클리핑 off, 1M | 207.8 | △ FR-1 기본 4 아님 | 3시드 중 1 FAIL |
| 4×1024, LR 감소, 가치 클리핑 on, 2M | 226.4 | ✅ | 3/3 PASS |
| 4×1024, LR 감소, 가치 클리핑 off, 2M | 255.3 | ✅ | 3/3 PASS |
| **위 + 보상 정규화 on** | **268.1** | ✅ | **3/3 PASS, 학습 중 truncation 소멸 (EXP-007/008)** |
### 결정
`num_steps 128→1024`, `num_minibatches 4→32` (미니배치 128 유지), `gamma 0.99→0.999`, `gae_lambda 0.95→0.98`,
`clip_vloss True→False`, `norm_reward True`(보상 정규화 포함), `total_timesteps 1M→2M`.
> 경과: 처음엔 EXP-001/003 결과로 `norm_reward=False` 로 정했으나, 새 설정 위 ablation(EXP-008)에서 +12.8 로 뒤집혀 True 로 재변경.
**유지**: num_envs 4, ε 0.2, ent 0.01, K 4, 어드밴티지 정규화, LR 선형 감소, grad clip 0.5 — FR 에 명시된 메커니즘은 모두 그대로.
### 결과
- `python src/ppo.py` 한 줄로 200 이상 재현 (EXP-007).
- PRD 의 "원본과 비교"는 두 갈래로 기록: (a) CleanRL 기본값 그대로 (`configs/cleanrl_match.json`, 둘 다 실패하는 곡선 비교) (b) 최종 하이퍼파라미터를 CleanRL 원본에도 넣어 수렴 스텝 비교.
- FR-4 의 "λ=0.95, γ=0.99" 는 GAE 단위 테스트에서 임의 값으로 검증하므로 기능 요구는 충족, 기본값만 다름.

---

## ADR-006: Gymnasium 1.x 벡터 환경은 SAME_STEP autoreset
**날짜**: 2026-10-28
**상태**: accepted
### 배경
Gymnasium 1.x 기본 `NEXT_STEP`: 종료 다음 `step()` 은 행동을 무시하고 reset 관측을 보상 0 으로 돌려준다 → 버퍼에 가짜 전이가 섞임.
또 `final_info` 형식이 dict-of-arrays 로 바뀌어 CleanRL 원본의 에피소드 로깅이 **에러 없이 조용히** 동작하지 않는다.
### 결정
`ppo.py`, `ppo_annotated.py` 모두 `autoreset_mode=SAME_STEP` 명시, `infos["final_info"]["_episode"]` 마스크로 파싱.
### 결과
CleanRL(gym 0.29) 원래 의미 보존. truncation 부트스트랩도 `infos["final_obs"]` 로 구현 가능(EXP-006).
