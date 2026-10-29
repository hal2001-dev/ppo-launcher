# 설정 파일 스펙 (configs/*.json)

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [../architecture/decisions.md](../architecture/decisions.md), [../experiments/ablation.md](../experiments/ablation.md)

## 로드 규칙
`src/ppo.py` 의 `DEFAULTS` → `--config <json>` → CLI 인자 순으로 덮어쓴다.
- JSON 에서 `_` 로 시작하는 키(`_description` 등)는 주석으로 무시
- 모르는 키는 `ValueError` (오타 방지)
- `configs/default.json` 은 **DEFAULTS 의 사본**. 둘을 항상 같이 바꾼다 (lint 항목)
- bool CLI: `--norm-reward false` / `true` / `0` / `1`

## 키
| 키 | 기본 | 설명 | FR |
|---|---|---|---|
| exp_name | ppo | run 이름 `{env_id}__{exp_name}__{seed}__{unix}` | |
| env_id | LunarLander-v3 | | |
| seed | 0 | python/numpy/torch/env/action_space | FR-8 |
| total_timesteps | 2000000 | 전체 환경 스텝 | FR-8 |
| device | cpu | cpu / mps / cuda (불가 시 cpu 폴백) | FR-8 |
| torch_threads | 1 | `torch.set_num_threads` | |
| learning_rate | 0.00025 | Adam(eps=1e-5) 초기 LR | |
| anneal_lr | true | 선형 감소 | FR-7 |
| num_envs | 4 | SyncVectorEnv 개수 | FR-1 |
| num_steps | 1024 | env 당 롤아웃 길이 | FR-3 |
| gamma | 0.999 | 할인율 | FR-4 |
| gae_lambda | 0.98 | GAE λ | FR-4 |
| use_gae | true | false → 부트스트랩 MC 리턴 (ablation) | FR-4 |
| norm_reward | true | NormalizeReward + ±10 clip (ablation) | |
| bootstrap_truncation | false | 시간 제한 truncation 시 γV(final_obs) 를 보상에 더함 | |
| num_minibatches | 32 | 배치를 나누는 개수 | FR-6 |
| update_epochs | 4 | K | FR-6 |
| norm_adv | true | 미니배치 어드밴티지 정규화 | FR-6 |
| clip_policy | true | false → 비클리핑 surrogate (ablation) | FR-5 |
| clip_coef | 0.2 | ε | FR-5 |
| clip_vloss | false | 가치 손실 클리핑 | FR-5 |
| ent_coef | 0.01 | | FR-5 |
| vf_coef | 0.5 | | FR-5 |
| max_grad_norm | 0.5 | 글로벌 grad clip | FR-7 |
| target_kl | null | KL 조기 종료 (off) | |
| eval_episodes | 100 | 종료 후 / `--eval` 평가 에피소드 | |
| save_model | true | `models/<run>.pt` | |
| run_dir | runs | TensorBoard 루트 (프로젝트 루트 기준, 절대경로 가능) | |

## 파일 목록
| 파일 | 내용 |
|---|---|
| `default.json` | 최종 기본값 (`python src/ppo.py` 와 동일) |
| `cleanrl_match.json` | CleanRL ppo.py 와 같은 하이퍼파라미터 (원본 비교용) |
| `ablation_no_gae.json` | `use_gae=false` |
| `ablation_no_reward_norm.json` | `norm_reward=false` |
| `ablation_no_clip.json` | `clip_policy=false`, `clip_vloss=false` |
