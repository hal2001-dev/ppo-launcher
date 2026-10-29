# 개발 환경 셋업

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [../architecture/stack.md](../architecture/stack.md), [../deployment/runbook.md](../deployment/runbook.md), [../troubleshooting/common.md](../troubleshooting/common.md)

## 요약
Homebrew 로 swig·uv 설치 → `uv sync` 한 번이면 끝. PRD 의 `uv pip install …` 대신 `pyproject.toml` 로 고정.

## 절차
```sh
brew install swig uv
cd ~/workspace/projects/personal/ppo-lunarlander
uv sync                         # .venv 생성 (Python 3.11 자동 다운로드) + 의존성 + dev(pytest)
source .venv/bin/activate       # 선택. 안 하면 `uv run …` 또는 .venv/bin/python
```

## 검증 (PRD 기준)
```sh
python -c 'import gymnasium as gym; print(gym.make("LunarLander-v3"))'   # 에러 없이 생성
uv run pytest -q                                                       # 12 passed
python src/baseline_sb3.py --total-timesteps 20000                      # SB3 가 1분 내 첫 로그 출력
```
2026-10-28 확인: 세 항목 모두 통과 (SB3 첫 로그 수 초 내).

## 출처
- raw/requirements/2026-10-28-prd.md (기술 스택과 환경 요구사항)
