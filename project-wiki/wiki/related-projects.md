# Related Projects — 다른 프로젝트 참조

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [deployment/ports.md](deployment/ports.md), [references.md](references.md), [roadmap.md](roadmap.md)

## 요약
`~/workspace/projects/personal/` 아래 형제 프로젝트 중 이 프로젝트가 **의존하거나 형식을 따르거나
후속으로 이어지는** 것의 목록. 다른 프로젝트를 가리키는 링크는 이 페이지에서만 관리한다.

> 링크 규칙: wiki-viewer 는 프로젝트 밖 상대경로를 깨진 링크로 처리하므로,
> **뷰어 딥링크**(wiki-viewer 실행 중일 때 클릭) + **파일 경로**(백틱)를 함께 적는다.

## 공용 문서 (상위 디렉터리)

| 문서 | 경로 | 이 프로젝트와의 관계 |
|---|---|---|
| 포트 레지스트리 | `~/workspace/projects/personal/PORTS.md` | **단일 진실 공급원.** TensorBoard 를 Web 대역 3005 로 등록 (2026-10-28) → [deployment/ports.md](deployment/ports.md) |
| RAG Playbook | `~/workspace/projects/personal/RAG-PLAYBOOK.md` | 해당 없음(RAG 아님). "공용 표준은 상위에 두고 각 프로젝트가 참조"하는 패턴만 차용 |

## 인프라 / 도구 프로젝트

| 프로젝트 | 포트 | 관계 | 링크 |
|---|---|---|---|
| **wiki-viewer** | 3003 | 이 위키를 읽는 뷰어. `project-wiki/` 를 자동 발견하므로 별도 등록 불필요 | [ppo-lunarlander 위키 열기](http://localhost:3003/?project=ppo-lunarlander&path=wiki/index.md) · `wiki-viewer/README.md` |
| monitoring | 8003 | 위키 스키마(CLAUDE.md 구조, 진행 레이블, lint 체크리스트)의 직접 원형 | [monitoring CLAUDE.md](http://localhost:3003/?project=monitoring&path=CLAUDE.md) · `monitoring/project-wiki/CLAUDE.md` |
| notebooklm | 3002 / 3100 | 위키 패턴 원조(plans/, rag-standard.md 처럼 "상위 공용 문서 참조" 페이지 방식) | [notebooklm index](http://localhost:3003/?project=notebooklm&path=wiki/index.md) · `notebooklm/project-wiki/wiki/index.md` |
| bulk-upload | 8002 | 포트 배정 직후 PORTS.md 행 추가 + log 기록 절차의 선례 | [bulk-upload log](http://localhost:3003/?project=bulk-upload&path=wiki/log.md) · `bulk-upload/project-wiki/wiki/log.md` |
| knowledge-rag / regulations-rag | 3000·8000 / 3001·8001 | 같은 위키 3-layer 구조. 직접 의존 없음 | `knowledge-rag/project-wiki/`, `regulations-rag/project-wiki/` |
| springboot-4.1.1 | 3004 / 8004·8005 | 직접 의존 없음. **127.0.0.1 바인딩 정책 참고**(이 프로젝트는 비밀이 없어 0.0.0.0 유지 → ADR-003) | `springboot-4.1.1/docs/13-network.md` |

## 후속 프로젝트 (PRD "확장 방향") — 아직 없음

| 예정 | 재사용 대상 | 생성 시 할 일 |
|---|---|---|
| 커스텀 트레이딩 환경 + PPO | `src/ppo.py` (Agent/RolloutBuffer/compute_gae/ppo_update 그대로) | 새 디렉터리 + `project-wiki/`, PORTS.md 다음 빈 번호(web 3006 …), 이 표에 행 추가 |
| 연속 행동 PPO (LunarLanderContinuous-v3) | `Agent` 의 Categorical → Normal 헤드만 교체 | 이 레포 브랜치 또는 별도 프로젝트 |
| MLX 포팅 | `tests/test_gae.py` 수치 케이스를 동일 결과 검증에 재사용 | 별도 프로젝트 권장 |

## 출처
- raw/requirements/2026-10-28-prd.md (확장 방향, 포트·위키 지시)
- `~/workspace/projects/personal/PORTS.md`, `wiki-viewer/README.md`
