# PPO LunarLander — Project Wiki Schema & Operations

이 파일은 LLM 이 본 프로젝트 위키를 어떻게 구축·유지할지 정의합니다.
**모든 세션 시작 시 이 파일을 먼저 읽고, `wiki/index.md` 와 `wiki/log.md` 최근 항목을 확인하세요.**

> Inspired by [karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
> — 3-layer 구조: **raw/** (immutable) / **wiki/** (LLM 관리) / **CLAUDE.md** (schema).
> 형식은 형제 프로젝트(monitoring, notebooklm, bulk-upload …)의 `project-wiki/` 와 동일하게 맞춘다
> → `wiki/related-projects.md`.

프로젝트: **PPO LunarLander** — LunarLander-v3(이산 4 액션)에서 PPO 를 직접 구현해
최근 100 에피소드 평균 200점 이상을 달성하는 개인 학습 프로젝트.
SB3 기준선 → CleanRL 코드 읽기 → 단일 파일 재구현 → 원본 비교 → 구성 요소 제거 실험.

---

## 디렉토리 구조

```
project-wiki/
├── CLAUDE.md                       ← 이 파일 (스키마)
├── raw/                            ← 원본 (절대 수정 금지)
│   ├── requirements/               ← PRD 원문, 후속 요구 변경
│   ├── research/                   ← 참고 구현 원본(CleanRL ppo.py 스냅샷), 논문 메모
│   ├── meetings/  issues/  reviews/
└── wiki/
    ├── index.md                    ← 전체 페이지 카탈로그 (항상 최신 유지)
    ├── log.md                      ← 작업 이력 (append-only, 최신이 위)
    ├── overview.md                 ← 프로젝트 요약 + 현재 상태 + 할 일
    ├── roadmap.md                  ← 단계(Stage 1~5) 마일스톤 + 통과 기준
    ├── changelog.md                ← 코드/설정 변경 이력
    ├── glossary.md                 ← 용어 사전 (GAE, ratio, clipfrac, EV …)
    ├── references.md               ← 논문/블로그/코드 링크
    ├── related-projects.md         ← ★ 다른 프로젝트 참조(포트, 위키 패턴, 후속 프로젝트)
    ├── architecture/
    │   ├── decisions.md            ← ADR
    │   ├── stack.md                ← 기술 스택 + 선택 이유
    │   └── code-map.md             ← src/ppo.py 구조 ↔ PPO 알고리즘 단계
    ├── features/
    │   └── requirements.md         ← FR-1~8 ↔ 구현 위치 ↔ 검증 방법 추적표
    ├── experiments/                ← 실험 = 1 페이지 (설정, 시드, 결과, 결론)
    │   ├── protocol.md             ← 시드/평가/지표 규칙
    │   ├── results.md              ← ★ 전체 결과 요약표 (최신)
    │   └── ablation.md             ← 제거 실험 3종
    ├── testing/strategy.md
    ├── config/files.md             ← configs/*.json 키 스펙
    ├── deployment/
    │   ├── ports.md                ← PORTS.md 배정(TensorBoard 3005)
    │   └── runbook.md              ← 학습/평가/TensorBoard 실행 절차
    ├── onboarding/
    │   ├── setup.md
    │   └── rl-primer.md            ← 강화학습 입문 (src/ppo.py 기준 개념 설명)
    ├── troubleshooting/common.md
    ├── meetings/  reviews/
    └── issues/{open,resolved}/
```

---

## 페이지 공통 형식

```markdown
# 페이지 제목

**상태**: [active|draft|deprecated]
**마지막 업데이트**: YYYY-MM-DD
**관련 페이지**: `다른 페이지 링크`

## 요약
(2~3줄 핵심 설명)

## 본문
...

## 출처
- raw/requirements/2026-10-28-prd.md
```

## 실험 페이지 형식 (experiments/*.md)

```markdown
# EXP-NNN: 제목
**날짜**: YYYY-MM-DD
**상태**: [planned|running|done|abandoned]
**설정**: configs/xxx.json (+ CLI override)
**시드**: 0, 1, 2
**run 디렉터리**: runs/<run_name>
## 가설
## 결과 (표: seed | 최종 mean100 | eval 100ep | 200 도달 스텝)
## 해석
## 다음 행동
```

## 이슈 / ADR 형식
형제 프로젝트와 동일 (`ISSUE-NNN`: 증상/원인 분석/해결 방법/재발 방지,
`ADR-NNN`: 배경/선택지/결정/결과).

---

## Operations

### 1. Ingest (새 정보 추가)
1. 원본을 `raw/` 적절한 폴더에 저장
2. 관련 wiki 페이지 업데이트 또는 신규 생성
3. `wiki/index.md` 갱신
4. `wiki/log.md` 에 항목 추가:
   ```
   ## [YYYY-MM-DD] ingest | 제목
   - 영향받은 페이지: xxx.md, yyy.md
   - 주요 변경: 한 줄 요약
   ```
5. `wiki/overview.md` 진행 상황 업데이트

### 2. Query
`wiki/index.md` 먼저 → 관련 페이지 → 답변 + 출처. 새 인사이트면 페이지로 저장 제안.

### 3. 실험 기록 (이 프로젝트의 핵심 루틴)
1. 설정 파일을 `configs/` 에 추가(또는 CLI override 를 그대로 기록)
2. 시드 0/1/2 실행, 결과는 `src/evaluate.py` 표 출력 그대로 붙여 넣기
3. `experiments/results.md` 요약표 갱신 + 개별 실험 페이지
4. 결론이 설계를 바꾸면 `architecture/decisions.md` ADR 추가, `config/files.md` 동기화
5. 레포 루트 `notes/*.md`(PRD 지정 산출물)도 함께 갱신 — 위키는 상세/이력, notes 는 제출용 요약

### 4. Issue 등록 / 해결
- 등록: `raw/issues/` → `wiki/issues/open/ISSUE-NNN.md` → 반복 가능하면 `troubleshooting/common.md`
- 해결: `open/` → `resolved/` 이동 + 재발 방지

### 5. Lint — `"wiki lint 해줘"`
- [ ] overview.md 진행 상황 / roadmap 체크가 최신인가
- [ ] experiments/results.md 수치가 notes/*.md 와 일치하는가
- [ ] configs/*.json 키가 config/files.md 와 src/ppo.py DEFAULTS 와 일치하는가
- [ ] features/requirements.md 의 구현 위치(함수명)가 실제 코드와 일치하는가
- [ ] src/ppo.py 가 500줄 이내인가 (`wc -l src/ppo.py`)
- [ ] 끊어진 링크 / 고아 페이지는 없는가 (wiki-viewer 에서 빨간 링크 확인)
- [ ] 상위 PORTS.md 의 ppo-lunarlander 행과 deployment/ports.md 가 일치하는가
- [ ] glossary.md 에 없는 새 용어가 본문에 등장하지 않는가

---

## 도메인 특화 규칙

### 하이퍼파라미터 기본값 변경 시
- `src/ppo.py` DEFAULTS 와 `configs/default.json` 을 **동시에** 바꾼다 (둘이 다르면 lint 실패)
- 근거 실험을 `experiments/` 에 남기고 ADR 추가, `changelog.md` 기록

### 지표(TensorBoard 태그) 추가·변경 시
- CleanRL 태그 이름(`charts/*`, `losses/*`)과 동일하게 유지 → `experiments/protocol.md` 표 갱신

### 다른 프로젝트 참조 규칙
- wiki-viewer 는 프로젝트 밖 상대경로(`../../other/…`)를 깨진 링크로 처리한다.
  다른 프로젝트 페이지는 **뷰어 딥링크** `http://localhost:3003/?project=<p>&path=<path>` +
  **파일 경로**(백틱) 를 함께 적는다. 목록은 `wiki/related-projects.md` 한 곳에서 관리.
- 포트는 상위 `~/workspace/projects/personal/PORTS.md` 가 단일 진실 공급원. 변경 시 그 파일 먼저.

---

## 진행 상태 레이블

`todo` · `in-progress` · `done` · `blocked` · `experimental` · `deprecated` · `needs-review`

## 세션 시작 루틴
1. 이 파일 읽기 → 2. `wiki/log.md` 최근 5개 → 3. `wiki/issues/open/` →
4. `wiki/overview.md` 현재 상태 → 5. 사용자에게 브리핑 후 작업 시작
