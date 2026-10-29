# Changelog

**상태**: active
**마지막 업데이트**: 2026-10-28

## [0.1.0] — 2026-10-28
### Added
- 프로젝트 골격: `pyproject.toml`(uv, Python 3.11), `.gitignore`, `README.md`
- `src/baseline_sb3.py` — SB3 PPO 기준선 (`--preset zoo|cleanrl`), `charts/episodic_return` 콜백
- `src/ppo_annotated.py` — CleanRL ppo.py(fe8d8a0) 사본 + 한국어 주석, Gymnasium 1.x 수정 3곳
- `src/ppo.py` — 단일 파일 재구현 (FR-1~8, `--eval`, `.pt` 저장, 종료 시 자동 평가)
  - ablation 스위치: `use_gae`, `norm_reward`, `clip_policy`, `clip_vloss`
  - `bootstrap_truncation` (SB3 방식 truncation 부트스트랩, 기본 off)
  - `run_dir` (튜닝 로그 분리)
- `src/evaluate.py` — `.pt`/`.zip` 일괄 평가 + 마크다운 표
- `tests/test_gae.py`(6), `tests/test_buffer.py`(4), `tests/test_seed.py`(2)
- `configs/default.json`, `cleanrl_match.json`, `ablation_{no_gae,no_reward_norm,no_clip}.json`
- `scripts/tensorboard.sh` (포트 3005), `scripts/summarize_runs.py` (수렴 스텝·trunc% 표)
- `notes/01_sb3_baseline.md`, `02_cleanrl_reading.md`, `04_ablation.md`
- `project-wiki/` (CLAUDE.md 스키마, raw/, wiki/)
- 상위 `PORTS.md` 에 ppo-lunarlander 행 (Web 3005)

### Changed (같은 날, 실험 결과 반영)
- 기본값 `norm_reward` True → False (EXP-001~003) → 다시 **True** (EXP-008, 새 설정 위에서 +12.8)
- 기본값 `num_steps 128→1024`, `num_minibatches 4→32`, `gamma 0.99→0.999`, `gae_lambda 0.95→0.98`,
  `clip_vloss True→False`, `total_timesteps 1M→2M` (ADR-005, ISSUE-001)
- `ablation_no_reward_norm.json` ↔ `ablation_reward_norm.json` 왕복 (기본값 변경에 따라). 최종은 `ablation_no_reward_norm.json`
- 이전 기본값 시절 run/모델은 `runs/_superseded/`, `models/_superseded/` 로 이동, 튜닝은 `_tuning/`
- `tests/test_seed.py` 가 새 기본값(배치 4096)에서 업데이트 0회가 되는 문제 → 4×128 로 override
