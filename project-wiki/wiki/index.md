# Wiki Index — PPO LunarLander

**상태**: active
**마지막 업데이트**: 2026-10-29

전체 위키 페이지 카탈로그. 새 페이지 생성 시 반드시 여기에 등록.
처음 읽는다면: [overview.md](overview.md) → [onboarding/rl-primer.md](onboarding/rl-primer.md) → [issues/resolved/ISSUE-001.md](issues/resolved/ISSUE-001.md) → 실험 EXP-001 부터 순서대로.

## 핵심 (Core)
- [overview.md](overview.md) — 프로젝트 요약 + 현재 상태 + 배운 것 + 할 일
- [roadmap.md](roadmap.md) — Stage 0~5 통과 기준/완료일 + 학습자 TODO
- [log.md](log.md) — 작업 이력 (append-only, 시간순 상세)
- [changelog.md](changelog.md) — 코드/설정 변경 이력
- [glossary.md](glossary.md) — 용어 사전 (GAE, ratio, clipfrac, EV, 호버링 …)
- [references.md](references.md) — 논문/블로그/코드
- [related-projects.md](related-projects.md) — ★ 다른 프로젝트 참조 (PORTS.md, wiki-viewer, 위키 원형, 후속 프로젝트)

## 아키텍처 (architecture/)
- [decisions.md](architecture/decisions.md) — ADR-001~006 (단일 파일, 태그 이름, 포트, CPU, 기본 하이퍼파라미터, autoreset)
- [stack.md](architecture/stack.md) — 기술 스택 + 설치 버전
- [code-map.md](architecture/code-map.md) — src/ppo.py 구조 ↔ PPO 알고리즘 (mermaid)

## 기능 (features/)
- [requirements.md](features/requirements.md) — FR-1~8 ↔ 구현 함수 ↔ 검증 추적표

## 실험 (experiments/)
- [results.md](experiments/results.md) — ★ PRD 성공 기준 판정 + 전체 결과 요약
- [protocol.md](experiments/protocol.md) — 시드/평가/수렴 스텝 정의, TensorBoard 태그
- [ablation.md](experiments/ablation.md) — 제거 실험 요약
- [EXP-001](experiments/EXP-001-first-default-run.md) — 첫 학습: PRD 기본값 → 실패
- [EXP-002](experiments/EXP-002-cleanrl-contrast.md) — CleanRL 대조 + 호버링 진단
- [EXP-003](experiments/EXP-003-tuning-round1.md) — 튜닝 1차: γ / 정규화 / 길이
- [EXP-004](experiments/EXP-004-tuning-round2.md) — 튜닝 2차: 롤아웃·배치 크기
- [EXP-005](experiments/EXP-005-tuning-round3.md) — 튜닝 3차: 2M/3M, 첫 PASS
- [EXP-006](experiments/EXP-006-sb3-gap.md) — SB3 와의 차이 분해
- [EXP-007](experiments/EXP-007-final-default.md) — 최종 기본값 + CleanRL 비트 일치
- [EXP-008](experiments/EXP-008-ablation.md) — 제거 실험 3종 상세

## 이슈 (issues/)
- open: 없음
- [ISSUE-001](issues/resolved/ISSUE-001.md) (resolved) — PRD 기본값으로 수렴하지 않음(호버링), 조치 이력 10단계

## 테스트 / 설정 / 운영
- [testing/strategy.md](testing/strategy.md) — L1~L4 테스트 계층, 안 배울 때 점검 순서
- [config/files.md](config/files.md) — configs/*.json 키 스펙 + 기본값
- [deployment/ports.md](deployment/ports.md) — TensorBoard 3005 (PORTS.md)
- [deployment/runbook.md](deployment/runbook.md) — 학습/평가/제거 실험/TensorBoard 명령
- [onboarding/setup.md](onboarding/setup.md) — 설치 + 검증
- [onboarding/rl-primer.md](onboarding/rl-primer.md) — ★ 강화학습 입문: src/ppo.py 로 읽는 PPO (Actor-Critic, GAE, 클리핑, 지표, 제거 실험 교훈)
- [troubleshooting/common.md](troubleshooting/common.md) — 설치·학습·도구 문제

## 레포 산출물 (위키 밖)
- `notes/01_sb3_baseline.md`, `notes/02_cleanrl_reading.md`, `notes/04_ablation.md` — PRD 지정 제출용 노트
- `src/ppo_annotated.py` — CleanRL 주석본
