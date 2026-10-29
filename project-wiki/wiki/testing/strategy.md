# 테스트 전략

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [../features/requirements.md](../features/requirements.md), [../experiments/protocol.md](../experiments/protocol.md)

## 요약
RL 버그는 대부분 "돌아가는데 안 배우는" 조용한 버그 → **수치로 손계산 가능한 부분은 단위 테스트**,
학습 품질은 **실험 프로토콜(3 시드 + 결정적 평가)** 로 나눠 검증한다.

| 계층 | 대상 | 파일 | 실행 시간 |
|---|---|---|---|
| L1 수치 | GAE 손계산, done 절단, λ=0/1 극한, env 독립성 | `tests/test_gae.py` | <0.1s |
| L1 구조 | 버퍼 shape/flatten 순서, vec env (4,8), actor/critic 분리·파라미터 수 | `tests/test_buffer.py` | <0.5s |
| L2 재현성 | 같은 시드 2회 → 파라미터 bit 동일, 다른 시드 → 다름 (2048 스텝 미니 학습) | `tests/test_seed.py` | ~1s |
| L3 학습 | 3 시드 × 1M 스텝, eval 100ep ≥ 200 | `python src/ppo.py --seed {0,1,2}` | 시드당 ~1.5분 |
| L4 비교 | CleanRL 원본 vs 재구현 수렴 스텝 차 ≤ 20% | `ppo_annotated.py` vs `ppo.py --config configs/cleanrl_match.json` | |

```sh
uv run pytest -q
```

## 안 배울 때 점검 순서 (PRD 리스크 대응)
1. `uv run pytest` — GAE/버퍼 테스트부터
2. 첫 업데이트 첫 미니배치 `ratio == 1` 인지 (logprob 저장/재계산 일치)
3. `pg_loss` 부호: `-adv * ratio` (최소화)
4. 어드밴티지 정규화가 미니배치 단위인지
5. CleanRL(`ppo_annotated.py`)과 같은 설정으로 곡선 겹쳐 보기 → 같이 망가지면 구현이 아니라 **설정 문제**
6. `charts/episodic_length` 가 1000 근처면 호버링 → [troubleshooting](../troubleshooting/common.md)
