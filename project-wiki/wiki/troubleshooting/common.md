# 자주 발생하는 문제

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [../testing/strategy.md](../testing/strategy.md), [../issues/resolved/ISSUE-001.md](../issues/resolved/ISSUE-001.md)

## 설치

| 증상 | 원인 | 해결 |
|---|---|---|
| `pip install gymnasium[box2d]` 중 swig 에러 | swig 없음 | `brew install swig` 후 `uv sync` |
| `uv: command not found` | uv 미설치 | `brew install uv` |
| `ModuleNotFoundError: tyro` | `ppo_annotated.py` 전용 의존성 | `uv sync` (pyproject 에 포함) |
| `DeprecationWarning: builtin type swigvarlink…` | Box2D SWIG 바인딩 경고 | 무해. 무시 |

## 학습

| 증상 | 원인 | 해결 |
|---|---|---|
| 초반 +0 ~ +80 까지 오르다 −40 ~ −90 으로 **다시 떨어짐**, `charts/episodic_length` ≈ 1000, truncation 비율 70%+ | **호버링 국소 최적.** γ=0.99(유효 지평 ≈100 스텝)에선 착륙 보너스가 멀고, 추락(−100) 회피가 더 이득 → 떠 있기 학습 | γ=0.999 + 큰 롤아웃(스텝 1024) + λ=0.98. [ISSUE-001](../issues/resolved/ISSUE-001.md), ADR-005 |
| 보상 정규화 켰더니 오히려 낮음 | 4×128·γ0.99 설정에선 정규화가 호버링 탈출을 돕지 못함. 최종 설정(4×1024·γ0.999)에서는 반대로 이득 | 기본 `norm_reward=true`. 조건 의존 → [ablation](../experiments/ablation.md) |
| 학습 mean100 은 ~130 인데 eval 은 230+, truncation 80% | 확률적 정책이 **착지 후 엔진을 계속 분사**해 1000 스텝까지 감 (호버링과 달리 리턴은 양수) | 보상 정규화 on 으로 해소(trunc 0~3%). 판정은 eval 기준 |
| 학습 후반 `approx_kl` ≈ 0.0000 | LR 선형 감소로 마지막 iteration LR ≈ 0 | 정상. 수렴 전에 LR 이 0 이 되면 `total_timesteps` 를 늘린다 |
| mean100(학습) < eval 100ep | 학습 중엔 확률적 정책, 평가는 argmax | 정상. 성공 기준은 eval |
| 같은 시드인데 곡선이 다름 | 스레드 수/디바이스 차이, action_space 미시드 | `torch_threads=1`, cpu, `tests/test_seed.py` 로 확인 |
| `--device mps` 가 더 느림 (2,450 vs CPU 15,160 SPS) | 64-64 MLP, 스텝마다 작은 텐서 CPU↔GPU 전송 → 오버헤드 > 연산 | 기본 cpu (ADR-004) |
| `evaluate.py` 점수가 학습 종료 시 eval 과 조금 다름 (274.53 vs 274.29) | 평가 시드가 다름: 학습 종료 시 `10000+seed`, `evaluate.py` 기본 `10000` | 정상. 비교는 같은 평가 시드끼리 |

## 도구

| 증상 | 해결 |
|---|---|
| zsh 에서 `P="python …"; $P` 실행 시 "no such file or directory" | zsh 는 변수를 단어 분리하지 않음. 함수 `P() { …; }` 또는 `${=VAR}` 사용 |
| TensorBoard 가 6006 으로 뜸 | `tensorboard` 직접 실행 대신 `scripts/tensorboard.sh` (3005) |
