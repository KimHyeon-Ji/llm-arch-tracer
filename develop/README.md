# develop/

검증·개선 작업 공간. **프로파일과 작업 중 출력은 여기, 검증 통과한 완성 산출물은 최상위
`../models/`.** `out/` 에서 검증하고, 통과한 폴더만 `promote.py` 로 승격한다.

| 항목 | 무엇 |
|---|---|
| `models/` | 모델 프로파일(`.yaml`) — 초안과 통과분 모두. 재생성·재추적의 소스 |
| `out/` | `run.py` 실행 결과가 나오는 곳(검증 전). 통과분은 `../models/` 로 승격 |
| `promote.py` | 승격 게이트 — `report.md` FAIL 0 일 때만 옮긴다 |
| `verify_all.py` | **단일 게이트.** `rules/` 나 `src/` 를 고쳤으면 반드시 돌린다 |
| `verify_selftest.py` | 게이트 자체 검증 — 각 검사에 결함을 주입해 살아있는지 확인 |
| `verify/` | 게이트 베이스라인 + ③ 라벨 검토 원장(`review_ledger.yaml`) |
| `regen_summaries.py` | 재추적 없이 전 모델 산출물 갱신(심볼라이저·규칙 변경 반영) |
| `regen_tables.py` | 표만 다시 렌더(사이드카 없이 저장된 심볼 shape에서) |
| `make_review_packet.py` | 모델별 리뷰 패킷(`full/review.md`) 생성 |
| `backfill_module_classes.py` | 모듈 경로 ↔ 소스 클래스 대응 채우기 — 소속 검사의 조인 키. 가중치 없는 meta 빌드 한 번(모델당 1초 미만), 재추적 없음 |
| `canary/` | 회귀 테스트 — 이미 통과한 모델을 다시 돌려 안 깨졌는지 확인 |
| `sources/` | 각 모델의 실제 modeling/configuration 소스 캐시. transformers 본체에 없으면 **모델 저장소의 remote code** 를 받아 `<model_id>__<파일명>.py` 로 둔다 (gitignore, 자동 재다운로드) |
| `validation/` | Phase 0 블라인드 검증 기록 (온보딩 전 가설과 실측 대조) |
| `escalations/` | Tier 3 사람 검증 기록·리서치 소스 |
| `03-labeling-roadmap.md` | 라벨링을 "추론"에서 "확인"으로 옮긴 검토 기록 + 남은 과제 |

`develop/` 은 지우지 않는다 — 새 모델을 검증할 때 필요한 과거 기록·테스트·프로파일이 전부 여기 있다.
