# 기능 요구사항 추적표 (FR-1 ~ FR-8)

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [../architecture/code-map.md](../architecture/code-map.md), [../testing/strategy.md](../testing/strategy.md)

## 요약
PRD 의 FR 8개 각각이 `src/ppo.py` 어디에 구현돼 있고 무엇으로 검증하는지.

| ID | 요구사항 | 구현 (`src/ppo.py`) | 검증 | 상태 |
|---|---|---|---|---|
| FR-1 | SyncVectorEnv N=4 병렬 롤아웃 | `make_vec_env` (SAME_STEP autoreset) | `tests/test_buffer.py::test_vec_env_obs_shape` (4, 8) + 학습 시작 로그 `[FR-1] obs shape=(4, 8)` | done |
| FR-2 | Actor/Critic 분리, 64-64 Tanh | `mlp`, `Agent` | `test_agent_actor_critic_separate` (파라미터 집합 disjoint, 개수) + 로그 `[FR-2] params actor=4996 critic=4801` + TB text `model/params` | done |
| FR-3 | 128 스텝 고정 버퍼 (obs/action/logprob/reward/done/value) | `RolloutBuffer` | `test_buffer_shapes`, `test_buffer_fill_and_flatten` | done |
| FR-4 | GAE λ=0.95 γ=0.99 | `compute_gae` (+ ablation `compute_mc_returns`) | `tests/test_gae.py` 6 케이스(손계산, done 절단, λ=0 ⇒ TD, λ=1 ⇒ MC, env 독립) | done |
| FR-5 | 클리핑 정책 손실 ε=0.2 + 가치 손실 + 엔트로피 0.01 | `ppo_update` | TB `losses/policy_loss`, `losses/value_loss`, `losses/entropy` | done |
| FR-6 | K=4 에폭 미니배치 + 어드밴티지 정규화 | `ppo_update` (`norm_adv`) | TB `losses/approx_kl`, `losses/clipfrac` | done |
| FR-7 | LR 선형 감소 + 글로벌 grad clip 0.5 | `train` (anneal), `ppo_update` (`clip_grad_norm_`) | TB `charts/learning_rate` | done |
| FR-8 | `--seed --total-timesteps --device`, 시드 전부 고정 | `parse_args`, `set_seed`, `envs.reset(seed)`, `envs.action_space.seed` | `tests/test_seed.py` (같은 시드 → 파라미터 bit 동일, 다른 시드 → 다름) | done |
| 추가 | `.pt` 저장 + `--eval` 100ep 평균 | `train` 말미 저장, `evaluate`, `load_agent` | `python src/ppo.py --eval models/<run>.pt` | done |

> 지표 이름 메모: PRD 표는 `charts/approx_kl`, `charts/explained_variance` 로 적었지만 같은 PRD 가
> "CleanRL 과 동일한 이름"을 요구하므로 CleanRL 원본 태그 `losses/approx_kl`, `losses/clipfrac`,
> `losses/explained_variance` 를 따른다 (ADR-002).

## 학습 외 기준
- 한 파일 500줄 이내: `wc -l src/ppo.py` → 458줄 (2026-10-28)
- `python src/ppo.py` 한 줄 재현: DEFAULTS = `configs/default.json`

## 출처
- raw/requirements/2026-10-28-prd.md
