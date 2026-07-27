# llm-arch-tracer

Hugging Face LLM을 **가중치 없이 meta/fake device로 forward만 실행**해, 레이어 단위의 연산(op)·텐서 shape·의존관계(dependency)를 ATen 레벨에서 추출하는 파이프라인이다. 결과로 op별 표(주요 op / 전체 트레이스), 구조 요약(`structure.yaml`), 모델 요약 카드(`model_summary.md`), 검증 리포트를 낸다. shape은 특정 실행에 묶이지 않도록 심볼(`B, T, d_model, …`)로 렌더한다.

이 저장소에는 그 **전체 방법이 정리되어 있다**:
- **결과물 뽑는 법** — `src/`의 참조 구현(`run.py` 진입점)으로 config·modeling forward를 실제 실행해 표·요약을 생성. 무엇을/어떻게는 `01-main.md`(워크플로우 스펙)와 `USAGE.md`(사용법)에.
- **검증하는 법** — 추출이 맞는지 판정하는 C1~C17 체크리스트(`src/validate.py`, 스펙은 `01-main.md §9`), 회귀 테스트(`develop/canary/`), 그리고 **`develop/verify_all.py`** — rules/src를 고친 뒤 반드시 돌리는 단일 게이트(파일 무결성 + 모델별 지표 + 외부 수치 대조 + 퇴행 검사).
- **신규/미지 모델 대응** — 처음 보는 모듈을 만났을 때의 절차(Tier 0~3)는 `02-new-module-handling.md`.
- **누적 자산** — 검증 통과한 완성 산출물(`models/`, 모델별 출력 폴더), 프로파일·규칙(`develop/models/`, `rules/`), 검증 작업 공간(`develop/`).

## 구조

```
01-main.md                     # 메인 워크플로우 + 출력 검증 + 공통 심볼 + 구조/모델 요약 스펙
02-new-module-handling.md      # 신규/미지 모듈 대응 절차 (Tier 0~3)
USAGE.md                        # 입력/실행/출력 사용법 템플릿
src/                            # 참조 구현 (01-main.md의 실제 코드)
models/                         # 검증 통과한 완성 산출물 (모델별 출력 폴더 — 여기 있는 건 다 믿을 수 있음)
rules/                          # 정규화 규칙표 · 오류 패턴표 · 공통 심볼 · 구조 라이브러리(structures/) (계속 누적)
develop/                        # 검증/개선 작업 공간 (아래 설명)
```

## models/ ↔ develop/ 관리 흐름

**프로파일 = 레시피, 출력 = 완성품**으로 관리한다. `models/`엔 검증 통과한 완성 산출물만, 그걸 만드는 과정·재료는 전부 `develop/`에 있다.

1. `develop/models/<name>.yaml` — 프로파일(레시피·재생성 소스)을 작성/보관.
2. `python src/run.py --profile develop/models/<name>.yaml --out develop/out/` — 출력이 `develop/out/`에서 검증·반복.
3. `report.md` FAIL 0 (+ C13 PASS)이면 그 출력 폴더를 `models/`로 승격(`develop/promote.py`가 게이트 검사 후 이동).

## develop/ 폴더

- `develop/models/` — 프로파일(초안 + 통과분 보관, 재생성 소스).
- `develop/out/` — `run.py` 실행 결과가 나오는 **작업 공간**(검증 전/반복 중). 통과분은 `models/`로 승격.
- `develop/canary/` — 회귀 테스트 스위트(이미 통과한 모델들을 다시 돌려 깨지지 않았는지 확인).
- `develop/escalations/` — Tier 3 사람 검증 기록·리서치 소스(`02-new-module-handling.md`).
- `develop/04-verification-plan.md` — 진행 계획.
- `promote.py`(승격 게이트) / `regen_*.py`(재추적 없이 `models/` 결과물을 현재 포맷으로 갱신) — 관리 헬퍼.

`develop/`은 지우지 않는다 — 나중에 새 모델을 또 검증할 때 필요한 과거 기록·테스트·근거·프로파일이 전부 여기 있다.

## 사용법

입력(프로파일 작성) / 실행 / 출력 읽는 법은 `USAGE.md` 참고.

```bash
python src/run.py --profile develop/models/<id>.yaml --out develop/out/
# 검증(report.md) 통과 시:  python develop/promote.py <id>   # -> models/ 로 승격
```

## src/ 파일 설명

| 파일 | 역할 |
|---|---|
| `provenance.py` | revision을 commit hash로 고정, config 스냅샷, 아키텍처 지원 여부 확인(01-main.md Step 1) |
| `loader.py` | meta / fake device로 가중치 없이 모델 구조만 로드(Step 2) |
| `introspect.py` | config를 훑어 안전한 seq_len 하한, 레이어 스케줄, 추가 진입점을 자동 도출(Step 3) |
| `inputs.py` | arange 토큰ID + forward 시그니처 필터링으로 prefill/decode 입력 구성(Step 4) |
| `scope.py` | `ScopeLabeler` — module hook 기반 layer/block 라벨링 전용, op 캡처 아님 |
| `tracer.py` | `OpGraphTracer` — `TorchDispatchMode`로 실제 op/shape/dependency 캡처(핵심 로직) |
| `adapt.py` | Tier 0(환경/접근 오류 배제) + Tier 1(알려진 오류 패턴 → 조치 → 재시도) 적응 루프 |
| `normalize.py` | raw aten → 사람이 읽는 `op_type` 매핑, 매핑 실패는 `unmapped`로 표시 |
| `escalate.py` | Tier 3 — 사람 검증 요청 패킷 생성·기록 |
| `build_table.py` | 트레이싱 결과를 전체 표(`full/<phase>.csv`·`.trace.raw.jsonl`)와 주요 operator 표(`<phase>.csv`·`.jsonl`)로 출력(depends_on 컬럼 포함, 별도 graph.json 없음) |
| `major_ops.py` | 전체 표 → 주요 operator 표 파생: latency 무관 op 제거·norm 롤업·반복 레이어 접기(block_type/repeat/layers)·의존관계 그래프 축약(01-main.md §6.1) |
| `validate.py` | C1~C16 체크리스트 구현 |
| `summarize.py` | `structure.yaml`(구조 요약)과 `model_summary.md`(모델 요약 + 참고 소스) 생성 |
| `run.py` | 위 전부를 엮는 진입점. `python src/run.py --profile ... --out ...` |

각 파일의 역할과 대응하는 스펙 단계는 `01-main.md` §5(실행 단계)에 더 자세히 설명되어 있다.

## 구조 라이브러리

`rules/structures/`에 attention, MoE/FFN, 정규화, 위치 인코딩, 잔차 연결, 보조 모듈
등 카테고리별로 구조적 패턴(MHA, GQA, MLA, CSA, HCA, mHC, MTP 등)을 패턴 단위로
문서화해 쌓아간다 — 자세한 형식과 갱신 시점은 `rules/structures/README.md`, 개념은
`01-main.md` §12 참고. **지금은 각 카테고리의 가장 기본적인 패턴만 채워뒀고**,
복잡한 패턴은 `develop/`에서 실제로 만날 때 그 시점에 채운다 — 미리 답을 적어두지
않는다.
