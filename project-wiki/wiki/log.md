# Work Log (append-only)

최신 항목을 위에 추가. 학습 프로젝트이므로 **무엇을 왜 했고 무엇이 실패했는지**까지 남긴다.

## [2026-10-28] docs | 위키 정리 + PRD 판정
- 영향받은 페이지: overview.md, index.md, experiments/results.md, roadmap.md, changelog.md, decisions.md(ADR-005 갱신), ISSUE-001(10단계)
- 주요 변경: PRD 성공 기준 전 항목 ✅ 판정표 작성. 모델/run 을 최종 · `_tuning/`(47개) · `_superseded/` 로 정리.
- 남은 일: git 초기화, 학습자 TODO (roadmap).

## [2026-10-28] measure | CPU vs MPS SPS
- CPU 1 스레드 15,160 / CPU 8 스레드 11,939 / MPS 2,450 SPS (65k 스텝 단독). MPS 6.2배 느림 → ADR-004 확인.
- MPS 는 CPU 와 결과도 다름(mean100 −60 vs −90 @65k) → 재현성 위해 CPU 기본 유지.

## [2026-10-28] experiment | EXP-008 제거 실험 + 보상 정규화 기본값 재변경
- 1차 배치(정규화 off 기준): 정규화 **켜기**가 268.1 로 기준(255.3)보다 좋음. GAE 제거 1 FAIL, 클리핑 제거 2 FAIL(최저 4.8).
- 판단: 정규화를 기본값에 포함 → `norm_reward=True`, `ablation_reward_norm.json` → `ablation_no_reward_norm.json`.
  이전 run 은 `runs/_superseded/` 로 이동 후 기본·제거 실험 12개 재실행.
- 최종: 기본 268.1 / −GAE 253.2 / −정규화 255.3 / −클리핑 271.1(KL 8배, 3배 빠른 수렴).
- 교훈: ablation 결론은 기준 설정에 따라 뒤집힌다 (EXP-001/003 에서 정규화는 손해였음).

## [2026-10-28] experiment | EXP-007 최종 기본값 + CleanRL 비교
- `DEFAULTS`/`configs/default.json` 변경 (ADR-005). `tests/test_seed.py` 가 새 배치(4096)에서 업데이트 0회가 되는 문제 발견 → 4×128 override 로 수정.
- `python src/ppo.py --seed {0,1,2}` = 튜닝 X 실행과 eval 소수점까지 동일 (FR-8 전체 규모 확인).
- `ppo_annotated.py`(CleanRL) 를 같은 하이퍼파라미터로 실행 → 재구현과 **곡선 비트 단위 일치** (3 시드, 2 설정). 수렴 스텝 차 0%.

## [2026-10-28] experiment | EXP-006 SB3 와의 차이 분해
- 계기: SB3 zoo 253 vs 재구현 같은 하이퍼파라미터 182.
- `bootstrap_truncation` 스위치 구현 (SAME_STEP 의 `infos["final_obs"]` 사용).
- 2×2 분해: LR 감소 −47, 가치 클리핑 −32, truncation 부트스트랩 ≈ 0. SB3 등가 조합 = 253 → 코어 정확.
- FR-7(LR 감소)은 유지해야 하므로 가치 클리핑을 끄는 방향 선택. 4×1024 + 2M 후보 X → 255.3 (3/3).

## [2026-10-28] baseline | Stage 1 SB3 기준선 통과
- seed 0/1/2: 253.12 / 261.57 / 251.55 (평균 255.4), 각 ~470초. `notes/01_sb3_baseline.md` 작성.

## [2026-10-28] experiment | EXP-003~005 튜닝 3라운드
- 1차(γ/정규화/길이): 최고 E(γ0.99, 3M) 110. γ0.999 단독은 분산↑. **실수**: zsh 에서 `$P` 단어 분리 안 됨 → 8개 즉시 실패, 함수로 재실행.
- 2차(롤아웃/배치): zoo F 182, 4×1024 I 147 — 롤아웃 길이가 핵심.
- 3차(2M/3M, 3 시드): 4×1024 2M **K 226 (3/3) 첫 PASS**.
- 관찰: 학습 중 truncation ~80% 인데 eval 은 높음 → argmax 정책 30ep 분석: 길이 469, 두 다리 접지 87% → **착지 후 엔진 깔짝임** (호버링과 구분).
- `scripts/summarize_runs.py` 추가 (수렴 스텝 + trunc% 표).

## [2026-10-28] debug | EXP-001~002 첫 학습 실패 → 호버링 진단 (ISSUE-001)
- EXP-001: PRD 기본값 + 정규화, 1M, 3 시드 → −74.7 / −84.3 / −101.1. +8 정점 후 하락.
- PRD 리스크 대응대로: GAE 테스트(통과) → CleanRL 원본 동일 조건 실행 → 원본도 +71 정점 후 −43 으로 붕괴 → **설정 문제** 판정.
- `EventAccumulator` 로 episodic_length 분석 → 32만 스텝에 truncation 95% → 호버링 확정. 원인: γ0.99 의 100 스텝 지평.

## [2026-10-28] setup | 프로젝트 골격 + 위키 구성
- `brew install swig uv`, `uv sync` (Python 3.11.18, torch 2.14.1, gymnasium 1.4.0, SB3 2.9.2).
- 검증: `gym.make("LunarLander-v3")` OK, 스모크 20k 스텝 14.6k SPS.
- Gymnasium 1.x 확인: 기본 NEXT_STEP autoreset, `final_info` dict-of-arrays → SAME_STEP 채택 (ADR-006).
- CleanRL ppo.py 다운로드(commit fe8d8a0) → `raw/research/` 보관, `src/ppo_annotated.py` 주석본 작성, tyro 의존성 추가.
- 포트: PORTS.md 확인 → Web 3000~3004 점유, 3005 비어 있음(lsof) → TensorBoard 3005 배정, PORTS.md 행·메모 추가, "다음 프로젝트" 예시 3006 으로 갱신.
- 위키: monitoring/notebooklm 형식을 따라 CLAUDE.md·raw/·wiki/ 구성. wiki-viewer 가 프로젝트 밖 상대경로를 broken 처리하는 것 확인 → 타 프로젝트는 뷰어 딥링크 + 파일 경로로 참조 (`related-projects.md`).
- PRD 원문 → `raw/requirements/2026-10-28-prd.md`.
