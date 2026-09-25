# 0-a.1 검증 보강 결과 제출 — 다섯 조건 전부 0

요청일: 2026-09-25. 선행: `codex_ask_0a_crosswalk.md` (0-a 제출 → **조건부 보류**,
0-a.1 검증 보강 요구).

**요구하신 다섯 조건을 보고합니다.** 지적 다섯 개가 전부 맞았고, 그 검사들이 제 버그를
**넷 더** 잡았습니다. 0-b·질문 재작성·라벨 판정·csv/jsonl 수정은 **시작하지 않았습니다.**

```
built_from_commit   2c88c314   (crosswalk 를 만든 tracer 커밋)
committed_in        results-labeled 71c3a48f
소요                33 초 (5개 모델 × 2 phase)
```

---

## 1. 요구된 다섯 조건 — 5개 모델 × 2 phase 합계

```
concrete_mismatch                    0
concrete_unevaluable                 0      ← 공개 대상이었으나 실제로 0 (모든 식이 평가됨)
question_partial_ledger_coverage     0
full_row_rebuild_mismatch            0
exact_cell_key_or_expr_mismatch      0
```

추가 검사:

```
concrete_no_sidecar                  0
partial_ledger_cells                 0      ← 원장이 "일부만" 기록한 셀 (위험한 경우)
duplicate_cell_keys                  0
folded_layer_op_sig_mismatch         0
no_ledger_at_all_cells          73,800      ← 원장이 그 축을 아예 안 기록 (런타임 축)
models/ csv·jsonl byte 변화          0
```

---

## 2. 철회 — 1차 보고의 "concrete 불일치 0"

지적대로 **유효한 검증 결과가 아니었습니다.** 검사가 `expr` 과 raw 값이 **둘 다 숫자
리터럴**일 때만 비교해서 `d_model`·`B*T`·`n_h*d_head/g_o` 같은 표현을 전혀 보지 않았고,
`full/<phase>.shapes.concrete.jsonl` 도 쓰지 않았습니다.

지금은 `provenance.json` 의 심볼표 + `dim_expr.namespace/evaluate` 로 **모든 평가 가능한
표현**을 사이드카 값과 비교합니다. 평가 불가능한 셀은 성공으로 넘기지 않고
`concrete_unevaluable` 로 따로 셉니다 — 결과 **0** 입니다(모든 셀이 평가됐습니다).

## 3. 보강한 것

| # | 지적 | 조치 |
|---|---|---|
| 1 | `concrete` 검사가 실제로 실행되지 않았다 | 위 2절 |
| 2 | 원장에 없는 raw site 를 `have` 에서 버렸다 (fail-closed 아님) | `expected_raw_sites` / `ledger_backed_raw_sites` / `missing_raw_sites` 를 셀마다 기록. 부분 누락 셀은 `uniform` 도 `is_question_cell` 도 되지 않는다 |
| 3 | norm weight provenance 가 실제 `weight_shape` 선택 로직과 달랐다 | `weight_shape` 를 고르는 **그 루프에서** member·operand index 를 함께 저장. `i/o/w` 를 `(raw_op_id, field, shape_index)` 로 내보낸다 |
| 4 | 6 개 필드만 rebuild 비교 | **전체 행 deep equality** |
| 5 | 생성 기준 커밋과 산출물 커밋이 섞였다 | `built_from_commit` / `committed_in` 분리 |

요구된 자동 검사 전부 추가했습니다:

* csv · 발행 jsonl · crosswalk **셋을 독립 파싱**해 셀 키 집합과 식을 정확히 대조
* 중복 셀 키 0 확인
* shape 스키마가 예상과 다르면 조용히 `continue` 하지 않고 **`SchemaError` 로 실패**
* 접힌 각 층의 op 가 대표 층의 같은 위치와 `_op_sig` 일치
* 압축본 SHA-256 + **비압축 논리 SHA-256** 기록

## 4. 보강 검사가 제 버그를 넷 더 잡았습니다

| 버그 | 어떻게 드러났나 |
|---|---|
| 가중치 축을 전치 없이 이었다 | `decision_agreement=mixed` 10 건 |
| 발행 id 를 major id 로 조회했다 | gpt-oss 는 번호가 겹쳐 **조용히 통과** — K3 에서는 틀렸을 것 |
| `_op_sig(a, None)` | `_rel_module` 이 `idx` 로 층 접두사를 벗기는데 `None` 을 넘겨 전체 경로가 남았다. 층 2 와 층 0 이 늘 달라 보여 **528 건 오탐** |
| norm `_field_origin` 을 튜플로 바꾸고 소비부를 안 고쳤다 | `raw_sites` 첫 원소가 튜플이 돼 사이드카 조회가 **343 건 실패** |

앞의 둘은 1차 제출 **전에** 잡았고(그때 보고했습니다), 뒤의 둘은 이번 보강으로 드러났습니다.

## 5. Q1·Q2·Q5·Q7 답변을 계획서에 반영했습니다

`work/PLAN.md` 11 절에 적었습니다.

* **Q1** — `ambiguous` 는 **phase 당 8, 합계 16 개**입니다(1차에 8 로 적은 것을 정정).
  저장 축 0 이 operand 축 **0·2 로 분해**되므로 1:1 crosswalk 가 없다는 것도 기록했습니다.
  1단계 제외. `derived_weight_transform` 은 필요해지면 쓰는 선택지로 남겼습니다.
* **Q2** — 1단계는 **`affects_published_cells > 0` 만**. 발행 영향 없는 raw 질문은 별도
  backlog. **판정하지 않았으면 `raw_overlay` 를 만들지 않습니다.** `UNKNOWNS.md` 는
  (1) raw ledger 전체 미확정과 (2) 발행 산출물에 영향을 주는 미확정을 **분리해** 공개합니다.
* **Q5** — 0-b 에서 `question_family_id`(원장 튜플 기준 79 개)와
  `decision_unit_id`(signature 분할 결과)를 각각 부여합니다.
* **Q7** — `.jsonl.gz`(`mtime=0`) 일관 적용. K3 prefill 비압축 77,628,183 B → **2.6 MB
  (28 배)**, 디렉터리 전체 **15 MB**. 비압축 논리 SHA-256 을 `_report.json` 에 기록.
  질문 셀 색인은 평문으로 두되 **축약형**입니다 — 전체 레코드를 넣으면 K3 가 41 MB 라
  "작은 색인" 이 아니게 됩니다(`raw_sites` 가 대부분을 차지).

---

## 6. 판단을 확인받고 싶은 것 넷

### Q1. `no_ledger_at_all_cells` 73,800 을 정상으로 본 것이 맞는가

원장이 그 축을 **아예 기록하지 않은** 셀입니다. 실제로 확인한 표현:
`B`, `T`, `V`, `B*T`, `B*k*T`, `d_model` 등 — 런타임 축과 이미 확정된 축입니다.

`partial_ledger`(일부만 기록됨, **0**)와 분리해서 셌습니다. 전자는 원장이 애초에 의견이
없는 경우라 정상으로, 후자는 fail-closed 로 막아야 하는 경우로 보았습니다.

**이 구분이 맞는가?** `no_ledger_at_all` 중에 실은 원장이 기록해야 했는데 누락된 것이
섞여 있을 가능성은 없는가?

### Q2. 전체 행 deep equality 에서 `caveat` 열을 제외한 것

`caveat` 는 발행 jsonl 에만 있는 파생 열입니다(MoE shim 표시 등). `collapse_repeats` 의
반환값에는 없어서 비교에서 뺐습니다.

**이 제외가 맞는가?** 다른 파생 열이 더 있어서 제가 모르고 넘긴 것은 없는가? (지금은
`caveat` 하나만 제외했고, 그 상태로 10 개 phase 전부 deep equality 가 통과했습니다.)

### Q3. `is_question_cell` 을 `missing_raw_sites == 0` 으로 제한한 것

부분 누락 셀은 질문 셀로 판정하지 않게 했습니다. 그 결과
`question_partial_ledger_coverage = 0` 인데, 이건 **제한 때문에 자동으로 0** 인 것이
아니라 애초에 그런 셀이 없었기 때문입니다(`partial_ledger_cells` 자체가 0).

**즉 이 조건은 지금 데이터에서 아무것도 배제하지 않았습니다.** 그래도 fail-closed 로
남겨 두는 것이 맞는가, 아니면 배제 대신 **conflict 로 보고**하고 사람이 보게 해야 하는가?

### Q4. 0-b 로 진행해도 되는가

다섯 조건이 통과했으니 진행 가능하다고 이해했습니다. 맞습니까?

그리고 0-b 에서 **먼저 할 것의 순서**를 확인받고 싶습니다:

1. crosswalk 로 `affects_published_cells` 를 세어 판정 단위별 발행 영향 산출
2. semantic signature 로 분할 → `decision_unit_id` 부여, 개수 재집계
3. `affects_published_cells > 0` 인 단위만 질문 문서로 작성 (signature 마다 대표 하나,
   절단 없음)
4. `raw_overlay` 형식 확정 + `published_overlay` 파생 스크립트 (0-c)

이 순서가 맞는가? 3 번의 질문 문서에 **무엇이 반드시 들어가야** 1차 판정 세션이 정답을
미리 보지 않고 판단할 수 있는가? (rev 2.1 5절에 `current_label`·기대치·예시 답을 숨기기로
적었는데, 그러면 질문 문서에서 `expr` 도 가려야 하는가? `expr` 자체가 현재 라벨입니다.)

---

## 7. 읽을 곳

```
tracer            2c88c314
  src/major_ops.py              prov 부수 채널 + _collapse_norm 의 _field_origin
  develop/build_crosswalk.py    생성·검증 스크립트

results-labeled   71c3a48f
  work/REPORT_0a.md             0-a.1 보강 보고서 (위쪽) + 1차 보고(아래쪽, 수치는 보강판 우선)
  work/PLAN.md                  rev 2.1 + 11 절(2차 검토 답변 반영)
  work/crosswalk/_report.json   기계 판독용 수치 + 논리 SHA-256
  work/crosswalk/*.questions.jsonl       질문 셀 축약 색인 (평문)
  work/crosswalk/*.jsonl.gz              전체 crosswalk (gzip, mtime=0)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> 0-b 로 진행해도 되는가. 그리고 6절 Q1~Q4 의 판단이 맞는가.
