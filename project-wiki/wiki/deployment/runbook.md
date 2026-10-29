# Runbook — 학습 / 평가 / 비교 / TensorBoard

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [ports.md](ports.md), [../experiments/protocol.md](../experiments/protocol.md), [../config/files.md](../config/files.md)

모든 명령은 레포 루트에서, venv 활성화(`source .venv/bin/activate`) 기준.

## Stage 1 — SB3 기준선
```sh
python src/baseline_sb3.py --seeds 0 1 2            # zoo 하이퍼파라미터, 1M, 끝나면 100ep 평가
python src/baseline_sb3.py --preset cleanrl          # ppo.py 와 같은 하이퍼파라미터 (비교용)
```
→ `models/LunarLander-v3__sb3_<preset>__<seed>__<ts>.zip`, `runs/…`

## Stage 2 — CleanRL 원본(주석본) 실행
```sh
python src/ppo_annotated.py --seed 0                 # tyro CLI, --help 로 전체 인자
```

## Stage 3 — 재구현
```sh
python src/ppo.py                                    # = configs/default.json, seed 0
for s in 0 1 2; do python src/ppo.py --seed $s & done; wait   # 3 시드 병렬 (각 1 스레드)
python src/ppo.py --device mps                       # MPS 실험 (기본 cpu)
python src/ppo.py --config configs/cleanrl_match.json --seed 0  # CleanRL 과 동일 설정
```
→ 종료 시 `models/<run>.pt` 저장 후 자동으로 100ep argmax 평가 출력 + TB `eval/mean_return`.

## 평가
```sh
python src/ppo.py --eval models/<run>.pt
python src/evaluate.py models/*.pt models/*.zip      # 일괄 + 마크다운 표
```

## Stage 4 — 제거 실험
```sh
for c in no_gae no_reward_norm no_clip; do
  for s in 0 1 2; do python src/ppo.py --config configs/ablation_$c.json --seed $s & done; wait
done
```

## TensorBoard (포트 3005, PORTS.md 등록)
```sh
scripts/tensorboard.sh                               # http://localhost:3005, LAN http://macstudio:3005
LOGDIR=runs/_tuning scripts/tensorboard.sh           # 튜닝 실행만 보기
```
CleanRL 과 태그 이름이 같으므로 왼쪽 run 필터에 `cleanrl_match|ppo_annotated` 를 넣으면 곡선이 겹쳐 보인다.

## 정리
`runs/`, `models/` 는 `.gitignore`. 지울 땐 결과를 `experiments/results.md` 에 옮긴 뒤.
