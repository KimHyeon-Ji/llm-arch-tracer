# 0-c 순서 1 재제출 — 레코드 종류 셋, 상태별 결박, 부분 산출 금지

요청일: 2026-09-26. 선행: `codex_ask_0c_step1.md` (→ **미승인. 차단 결함 다섯**).

```
tracer            51b90c15
results-labeled   749a1b3c   (파일럿 3 건 그대로, revision 7)
```

---

## 0. 먼저 제 오보를 정정합니다

**지난 제출의 `69/69` 는 틀렸습니다. 실제로는 `68/69` 였습니다.**

원인은 지적하신 그대로입니다 — `ALLOW_DIRTY_BUILD` 는 dirty 를 **만드는** 변수가 아니라
dirty 트리에서 실행을 **허용하는** 변수입니다. 제가 시험에
`dirty_build is True` 를 단정해 뒀고, 당시 제 트리가 미커밋이어서 통과했습니다. 커밋 뒤
깨끗한 checkout 에서 돌리니 그 한 항목이 실패했습니다.

직접 확인했습니다.

```
git status --porcelain | wc -l   ->  0
develop/test_overlay_candidates.py
  FAIL **시험 환경에서는 dirty_build 가 참이고, 계약이 그것을 거부한다**
  68/69
```

지시대로 기준선은 `dirty_build=false` 로 만들고, dirty 거부는 metadata 주입으로 따로
확인합니다.

## 1. `no_name` 후보가 실제로 스키마를 통과하지 못했다

재현했습니다.

```
state_of("no_name", None)  ->  ("pending_no_name_adjudication",
                                "no_name_exists_candidate")
validate(그 레코드)        ->  ["모르는 verdict 'no_name_exists_candidate'"]
```

지적하신 원인도 맞습니다 — 시험이 `state_of()` 반환값만 보고 `validate()`·`build()` 에
넣지 않아 놓쳤습니다. **이제 시험이 `no_name` 후보를 실제로 통과시킵니다**, 그리고
`build()` 로 만든 후보에 `pending_no_name_adjudication` 이 실제로 들어 있는지도 봅니다.

권장하신 대로 후보와 최종을 **분리**했습니다.

```
answer_candidate   candidate_kind: confirm | rename | no_name_exists | undetermined
                   final_verdict 필드가 **아예 없다**
unit_adjudication  비승인: final_verdict·resulting_expr·grade_after 전부 null
                   승인:   final_verdict ∈ {confirm, rename, no_name_exists}
raw_overlay        승인 상태만. approval_id 필수
```

## 2. 상태·verdict·grade 를 규칙표 하나로 못 박았습니다

지적하신 모순 조합 둘을 재현하고 막았습니다.

```
고치기 전
  state=approved_confirm / verdict=rename / grade_after=unresolved  -> 통과 (모순)
  state=approved_no_name_exists / grade_after=unresolved            -> 통과 (모순)
  GRADES_AFTER 에 confirmed_no_name 이 **없었다**
```

지정해 주신 표 그대로입니다.

```python
APPROVED_RULES = {
  "approved_confirm":         {"final_verdict": "confirm",
                               "grade_after": "confirmed",
                               "resulting_expr": "current_or_null"},
  "approved_rename":          {"final_verdict": "rename",
                               "grade_after": "confirmed",
                               "resulting_expr": "required"},
  "approved_no_name_exists":  {"final_verdict": "no_name_exists",
                               "grade_after": "confirmed_no_name",
                               "resulting_expr": "null"},
}
```

`approved_confirm` 의 `resulting_expr` 는 지정표에 없었는데, 적용기가 추측하지 않게
**null 이거나 현재 라벨과 같아야** 한다고 좁혔습니다(더 엄격한 쪽).

비승인 상태는 세 필드가 모두 null 이어야 하고, `approval_id`·`adjudicated_by`·
`adjudicated_at` 도 승인 상태에서만 요구합니다.

`pending_first_pass` 도 고쳤습니다 — **답이 없는 상태이므로 answer 필드를 요구하지 않고,
있으면 거부**합니다.

## 3. 부분 산출을 막았습니다

지적하신 다섯 경로(단위 없음 / crosswalk 없음 / 발행 셀 없음 / raw 사이트 없음 / 스키마
위반) 뒤에도 정상 후보 일부를 쓰고 `exit 1` 만 냈습니다.

```
지금
  문제가 하나라도 있으면  candidates.jsonl 을 **쓰지 않는다**
                        옛 후보 파일도 **지운다** (남아서 쓰이지 않게)
                        진단은 _rejected.jsonl 로만
  입력 계약·사전 대조 실패  -> exit 2
  생성 중 오류             -> exit 3
```

확인: units 를 신뢰 해시와 다르게 만들면 `exit 2` 이고 **기존 후보 파일의 줄 수가 그대로**
입니다(덧쓰지 않음).

## 4. 입력 provenance 를 사전 대조로 바꿨습니다

```
units       읽기 **전에** 배정 manifest 의 input_sha256 과 대조    (10 파일)
crosswalk   읽기 **전에** 같은 방식으로 대조                        (10 파일)
둘 다 후보 산출물의 input_sha256 에도 포함
crosswalk 중복 셀 키  ->  **즉시 오류** (dict 로 덮어쓰지 않는다)
```

배정 manifest 에 units 10 + crosswalk 10 이 이미 들어 있어 그대로 신뢰 기준으로 썼습니다.

### `accepted.jsonl` 도 결박했습니다

지적이 정확했습니다 — 세 플래그와 revision 만 맞으면 `proposal`·`comparison` 을 바꿔도
통과했습니다. 수집기가 이제 기록합니다.

```
_ingest.json   accepted_sha256   accepted_rows   rejected_sha256
계약에서 **다시 계산해** 대조
```

확인: `accepted.jsonl` 의 `proposed_expr` 를 "완전히다른이름" 으로 바꾸면
**"accepted.jsonl 이 바뀌었다"** 로 걸립니다.

`built_from_commit` 이 40 자리 hex 인지, `generator_sha256` 이 있는지도 봅니다.

## 5. Q1·Q2 반영

**Q1** — 둘 다 `cannot_determine` 이면 최종 상태는 `pending_cannot_determine` 입니다.
지표는 지정하신 대로 분리합니다.

```
response_agreement    두 응답이 같다
semantic_resolution   축 이름이 해결됐는가
```

`response_agreement` 상태는 `semantic_resolution` 을 **필수**로 요구하도록 스키마에
넣었습니다(그 필드가 없으면 거부). `no_name` 두 개도 자동 승인하지 않고
`pending_no_name_adjudication` 으로 둡니다.

**Q2** — 중복 조정은 **판정 단위 단위**입니다. 후보 통계도 `단위` / `중복 답이 있는 단위`
로 세고, 사이트 수는 영향 범위 지표로만 둡니다.

## 6. 자기검사 — 69 → 109 항목

```
1) 상태·규칙표 일치                          8
2) 1 차 답 -> 상태 (approved 경로 없음)      22
3) answer_candidate (**no_name 실경로 포함**) 16
4) unit_adjudication 상태별 결박             19  <- 모순 조합 넷 포함
5) raw_overlay 승인 참조                      6
6) 입력 계약 실패 주입                       17  <- accepted 내용 변조 포함
7) 신뢰 해시 사전 대조                        3
8) 실제 후보 생성 (no_name 포함)             11
9) 부분 산출 금지                             7
                                           109/109
```

전체:

```
test_overlay_candidates    109/109
test_expr_compare           92/92     test_ingest_answers    80/80
test_packet_export          58/58     test_bundle_guards     63/63
test_provenance_fixture     22/22     test_segment_verdict    6/6
test_provenance_untouched    3/3      test_lowering_proof     6/6
test_tracer_alias            4/4      test_corrections_cover  5/5
                           448 항목
```

## 7. 읽을 곳

```
tracer 51b90c15
  develop/overlay_schema.py            종류 셋 / APPROVED_RULES / 계약 / 사전 대조
  develop/build_overlay_candidates.py  answer_candidate 만 만든다 / 부분 산출 금지
  develop/ingest_answers.py            accepted_sha256·accepted_rows
  develop/test_overlay_candidates.py   109 항목

실행해 볼 것
  .venv\Scripts\python.exe develop\overlay_schema.py     -> 상태 표
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> 다섯이 막혔는가. 특히
>
> * 레코드 종류 셋이 각 역할에 맞게 분리됐는가 (`answer_candidate` 에 결과 필드가
>   **아예 없는** 것이 맞는 방향인가)
> * `APPROVED_RULES` 가 상태별 의미를 충분히 못 박는가
> * 문제가 있으면 공개 파일을 쓰지 않고 옛 파일까지 지우는 것이 과하지 않은가
>
> 하나 확인받고 싶습니다. `approved_confirm` 의 `resulting_expr` 를 **null 이거나 현재
> 라벨과 같아야** 한다고 좁혔습니다(지정표에는 없던 조건). 적용기가 "confirm 은 라벨을
> 바꾸지 않는다" 를 추측하지 않게 하려는 것인데, 이대로 두어도 됩니까? 아니면
> `resulting_expr` 를 아예 금지(null 강제)하는 편이 낫습니까?
>
> 승인되면 순서 2(1·2 차 답 수집과 Q5 중복 조정)로 갑니다.
