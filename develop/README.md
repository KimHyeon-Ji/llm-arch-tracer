# develop/

검증/개선 작업 공간.— 새 모델을 검증하고,
실패를 고치고, 규칙을 늘려가는 기록이 여기 계속 쌓이며, 이후에도 계속 쓴다.

역할 분리: **프로파일·작업 출력은 여기(`develop/`), 검증 통과한 완성 산출물은 최상위 `../models/`**.
`develop/out/`에서 검증(`01-main.md` §9)하고, 통과한 출력 폴더만 `../models/`로 승격한다(`promote.py`).
평소에는 지우지 않는다 — 새 모델을 또 검증할 때 필요한 기록·프로파일이 전부 여기 있다.

- `models/` — 프로파일(`.yaml`). 초안 작성과 통과분 보관 모두 여기 — 재생성/재추적 소스.
- `out/` — `run.py` 실행 결과가 나오는 작업 공간(검증 전/반복 중). 통과분은 `../models/`로 승격.
- `promote.py` — 승격 게이트(`report.md` FAIL 0이면 `out/<model>` → `../models/`).
- `regen_tables.py` / `regen_summaries.py` — 재추적 없이 `../models/` 결과물을 현재 포맷으로 갱신.
- `canary/` — 통과한 모델들의 회귀 테스트 스위트(`suite.yaml`). 새 모델/규칙 추가 후 재확인용.
- `escalations/` — Tier 3 요청 패킷과 사람 답변 기록(`src/escalate.py`가 씀). `../rules/`에 반영 후에도 근거로 남긴다.
- `logs/` — Phase별 실행 로그.
