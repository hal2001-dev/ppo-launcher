# ppo-lunarlander

LunarLander-v3(이산 4 액션)에서 **PPO 를 직접 구현**해 100 에피소드 평균 200점 이상을 달성하는 학습 프로젝트.
SB3 기준선 → CleanRL 코드 읽기 → 단일 파일 재구현 → 원본 비교 → 구성 요소 제거 실험 순서로 진행한다.

- 요구사항(PRD): `project-wiki/raw/requirements/2026-10-28-prd.md`
- 진행 상황 / 실험 기록 / 시행착오: **`project-wiki/wiki/`** (wiki-viewer: http://localhost:3003/?project=ppo-lunarlander&path=wiki/index.md)

## 빠른 시작

```sh
brew install swig uv
uv sync && source .venv/bin/activate
uv run pytest -q                       # GAE·버퍼·시드 재현 테스트
python src/ppo.py                      # configs/default.json 으로 학습 → models/*.pt 저장 → 100ep 평가 출력
python src/ppo.py --eval models/<run>.pt
scripts/tensorboard.sh                 # http://localhost:3005 (포트는 ../PORTS.md 에 등록)
```

## 구성

```
src/
  baseline_sb3.py     Stage 1  SB3 PPO 기준선 (--preset zoo|cleanrl)
  ppo_annotated.py    Stage 2  CleanRL ppo.py 사본 + 한국어 주석 (Gymnasium 1.x 최소 수정)
  ppo.py              Stage 3  재구현 — 최종 산출물, 단일 파일 (<500줄), 프로젝트 내부 import 없음
  evaluate.py                  .pt / .zip 일괄 평가 (argmax, 100ep)
tests/                         test_gae.py, test_buffer.py, test_seed.py
configs/                       default.json, cleanrl_match.json, ablation_*.json
scripts/                       tensorboard.sh (3005), summarize_runs.py (수렴 스텝 표)
notes/                         01_sb3_baseline.md, 02_cleanrl_reading.md, 04_ablation.md
project-wiki/                  위키 (CLAUDE.md 스키마, raw/ 원본, wiki/ 정리본)
runs/ models/                  TensorBoard 로그 / 체크포인트 (.gitignore)
```

## CLI (src/ppo.py)

`DEFAULTS` → `--config <json>` → 개별 인자 순으로 덮어쓴다. 모든 키가 `--kebab-case` 인자로 노출된다.

```sh
python src/ppo.py --seed 1 --total-timesteps 2000000 --device mps
python src/ppo.py --config configs/ablation_no_gae.json --seed 2
python src/ppo.py --gamma 0.99 --num-steps 128 --clip-vloss true     # 개별 override
```

결과 요약은 `project-wiki/wiki/experiments/results.md` 참고.
