# 기술 스택

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [decisions.md](decisions.md), [../onboarding/setup.md](../onboarding/setup.md)

## 설치 확인된 버전 (2026-10-28, `uv sync`)

| 구성 요소 | 선택 | 설치 버전 | 선택 이유 |
|---|---|---|---|
| 언어 | Python | 3.11.18 (uv 관리) | PRD 지정. Box2D 휠 호환 |
| 패키지 관리 | uv | 0.13.6 (Homebrew) | `pyproject.toml` + `uv.lock` 으로 재현 |
| 딥러닝 | PyTorch | 2.14.1 | cpu 기본, `--device mps` 옵션 |
| 환경 | Gymnasium[box2d] | 1.4.0 | LunarLander-v3. `AutoresetMode.SAME_STEP` 필요(≥1.1) |
| Box2D 빌드 | swig | 4.5.1 (Homebrew) | box2d 빌드 선행 |
| 기준선 | Stable-Baselines3 | 2.9.2 | Stage 1 전용 |
| 참고 구현 CLI | tyro | (pyproject `>=0.8`) | `ppo_annotated.py`(CleanRL 원본)용. `ppo.py` 는 argparse 만 사용 |
| 로깅 | TensorBoard | 2.21.0 | CleanRL/SB3 와 동일 포맷 |
| 테스트 | pytest | dev group | `uv run pytest` |

## 하드웨어 메모
- Mac Studio (arm64, macOS 26.7). MPS 사용 가능 확인(`torch.backends.mps.is_available() == True`).
- CPU 1 스레드 기준 `src/ppo.py` ≈ **13~15k SPS** (4 env, 단독 실행) → 1M 스텝 ≈ 70~80초.
  64-64 MLP 에서는 환경 스텝이 병목이라 MPS 이점 없음 (PRD 예상과 일치, ADR-004).
