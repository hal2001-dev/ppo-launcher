# EXP-002: CleanRL 원본과 대조 + 호버링 진단

**날짜**: 2026-10-28
**상태**: done
**설정**: `configs/cleanrl_match.json` (EXP-001 에서 `norm_reward=false` 만 변경) vs `src/ppo_annotated.py`
**시드**: 재구현 0, 1 / CleanRL 0
**run 디렉터리**: `runs/*ppo_cleanrl_match__{0,1}__1793159073`, `runs/*ppo_annotated__0__*`

## 가설
H1. 재구현에 버그가 있다 → CleanRL 원본은 잘 배우고 재구현만 망가질 것.
H2. 설정(하이퍼파라미터) 문제다 → **둘 다 똑같이** 망가질 것.

## 방법
```sh
python src/ppo_annotated.py --seed 0                                 # CleanRL 원본 (Gymnasium 1.x 최소 수정)
python src/ppo.py --config configs/cleanrl_match.json --seed 0       # 같은 하이퍼파라미터
python src/ppo.py --config configs/cleanrl_match.json --seed 1
```
CleanRL 원본은 보상 정규화가 없으므로 비교 설정도 `norm_reward=false`.

## 결과
| 구현 | seed | 최고 mean100 (시점) | 최종 mean100 | eval 100ep |
|---|---|---|---|---|
| CleanRL 원본 | 0 | **+71** (~528k) | −43 | (평가 기능 없음) |
| 재구현 | 0 | **+82** (~507k) | −51 | −66.65 ± 30.34 |
| 재구현 | 1 | +40 (~460k) | −38 | −75.12 ± 40.53 |

→ **두 구현이 같은 모양**(50만 근처 정점 → 붕괴)을 보인다. H1 기각, H2 채택.
보너스: 보상 정규화를 끄자 정점이 +8 → +80 으로 올라감 (EXP-001 대비).

## 진단: 왜 붕괴하나? — episodic_length 분석
TensorBoard 이벤트 파일을 직접 읽어 (`EventAccumulator`) 100 에피소드 이동 통계를 냄. 재구현 seed 0:

| step | 평균 길이 | 평균 리턴 | truncation(=1000 스텝) 비율 |
|---|---|---|---|
| 23k | 91 | −159 | 0% |
| 124k | 286 | −51 | 7% |
| 323k | **977** | −64 | **95%** |
| 512k | 694 | **+80** | 22% |
| 634k | 465 | +43 | 10% |
| 797k | 767 | −6 | 22% |
| 1M | 887 | −52 | **71%** |

`explained_variance` 0.0 → 0.9 (가치 함수는 잘 학습됨), `entropy` 1.39 → 0.85 (정책이 결정적으로 수렴하는 중).

**결론: 호버링(hovering) 국소 최적.** 에피소드 대부분이 1000 스텝 시간 제한에 걸림 = 착륙 안 하고 떠 있음.
- γ=0.99 → 유효 지평 ≈ 1/(1−γ) = 100 스텝. 착륙 보너스(+100)는 수백 스텝 뒤라 할인되면 작다.
- 반면 추락(−100)은 즉시·확실한 손해 → "떠서 버티기"가 지역적으로 더 안전한 정책.
- 한 번 착륙을 배웠다가도(512k 시점 +80) 다시 호버링으로 돌아감 → LR 이 줄어드는 후반에 탈출 못 함.

## 배운 점
- **곡선이 같이 망가지면 구현이 아니라 설정 문제** — 원본을 같은 조건에서 돌려보는 게 가장 빠른 버그 판별법.
- `episodic_return` 만 보면 원인을 모른다. `episodic_length` 와 truncation 비율이 진단의 핵심.
- PRD 의 기본값(CleanRL 기본값)은 CartPole 용으로 정해진 값이라 LunarLander 에는 지평이 짧다.

## 다음 행동
γ 를 늘리고(0.999), 필요하면 롤아웃 길이·배치를 키운다 → [EXP-003](EXP-003-tuning-round1.md)
