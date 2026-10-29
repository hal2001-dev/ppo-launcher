# 실험 프로토콜

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [results.md](results.md), [../deployment/runbook.md](../deployment/runbook.md)

## 규칙
1. **시드 3개 (0, 1, 2)**. 튜닝 탐색 단계에서만 2개 허용, 결론은 3개로 재확인.
2. **성공 판정 = eval 100ep 평균 ≥ 200**: 학습 종료 정책을 argmax(결정적)로, 평가 시드 `10000 + seed` 부터 100 에피소드.
   학습 중 mean100(확률적 정책)은 진행도 지표일 뿐 판정에 쓰지 않는다.
3. **수렴 스텝** = TensorBoard `charts/episodic_return` 의 최근 100 에피소드 이동평균이 처음 200 을 넘는 global_step.
   한 번도 넘지 않으면 "미도달"로 기록.
4. 튜닝 실행은 `--run-dir runs/_tuning` 으로 분리해 본 실험 곡선과 섞이지 않게 한다.
5. 병렬 실행 허용(각 프로세스 `torch_threads=1`). 결과는 결정적이며 병렬 여부와 무관.

## TensorBoard 태그 (CleanRL 과 동일)
| 태그 | 주기 | 의미 / 목표 |
|---|---|---|
| `charts/episodic_return` | 에피소드 종료 | 1차 지표 (원시 보상) |
| `charts/episodic_length` | 에피소드 종료 | 1000 근처 = 호버링 경고 |
| `charts/learning_rate` | 업데이트 | 선형 감소 확인 (FR-7) |
| `charts/SPS` | 1초 | 성능 회귀 감지 |
| `losses/policy_loss`, `losses/value_loss`, `losses/entropy` | 업데이트 | 손실 분해 (FR-5) |
| `losses/approx_kl`, `losses/old_approx_kl`, `losses/clipfrac` | 업데이트 | 정책 변화 폭. approx_kl > 0.02 지속 시 LR 점검 (FR-6) |
| `losses/explained_variance` | 업데이트 | 가치 함수 품질, ≥ 0.8 목표 |
| `eval/mean_return`, `eval/std_return` | 학습 종료 | 100ep argmax 평가 |

> PRD 표의 `charts/approx_kl`·`charts/explained_variance` 는 CleanRL 원본에서 `losses/` 아래에 있으므로 원본을 따름 (ADR-002).

## 이벤트 파일에서 수치 뽑기
```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("runs/<run>", size_guidance={"scalars": 0}); ea.Reload()
[(e.step, e.value) for e in ea.Scalars("charts/episodic_length")]
```
