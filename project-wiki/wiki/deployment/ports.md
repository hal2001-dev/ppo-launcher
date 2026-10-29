# 포트 구성

**상태**: active
**마지막 업데이트**: 2026-10-28
**관련 페이지**: [runbook.md](runbook.md), [../related-projects.md](../related-projects.md), [../architecture/decisions.md](../architecture/decisions.md)

## 요약
이 프로젝트는 상주 서버가 없다. 유일하게 띄우는 웹 UI 인 **TensorBoard 를 3005** 에 둔다.
배정 근거는 상위 `~/workspace/projects/personal/PORTS.md` 의 대역 규칙(Web 3000부터 +1, 다음 빈 번호).

## 배정표

| 서비스 | 포트 | 바인딩 | 실행 | 비고 |
|---|---|---|---|---|
| TensorBoard | **3005** | `0.0.0.0` | `scripts/tensorboard.sh` | Web 대역. 기본값 6006 은 대역 밖이라 사용 안 함 |
| API / Postgres / Qdrant | - | - | - | 미사용 |

- PORTS.md 반영: 점유 현황 표에 `ppo-lunarlander | 3005 | - | - | - |` 행 + 메모 추가(2026-10-28),
  "다음 프로젝트" 예시를 web 3006 으로 갱신.
- 배정 시점 점유 확인: `lsof -iTCP -sTCP:LISTEN -P -n` → 3004(springboot vite)만 listen, 3005 비어 있음.
- 접속: 로컬 `http://localhost:3005`, LAN `http://macstudio:3005`.
- 임시로 다른 포트가 필요하면 `PORT=… scripts/tensorboard.sh` (PORTS.md 에 없는 상시 포트는 만들지 않는다).

## 바인딩 결정 (ADR-003 요약)
TensorBoard 는 학습 곡선과 하이퍼파라미터 텍스트만 노출하고 자격증명이 없다 → wiki-viewer(3003)와 같이
`0.0.0.0`. 로그에 민감 정보가 생기면 `HOST=127.0.0.1` 로 내린다.

## 출처
- `~/workspace/projects/personal/PORTS.md`
- raw/requirements/2026-10-28-prd.md ("서버의 포트구성은 상위의 PORTS.md 파일을 참조")
