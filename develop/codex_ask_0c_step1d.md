# 0-c 순서 1 — 후보 결박 · comparison 도메인 · id 목록 자료형

요청일: 2026-09-26. 선행: `codex_ask_0c_step1c.md`
(→ **이전 넷 해결. 추가 세 결함으로 보류**).

셋 다 재현하고 고쳤습니다. 순서 2 로 넘어가지 않았습니다.

```
tracer            2142b799
results-labeled   749a1b3c   (파일럿 3 건 그대로, revision 7)
```

---

## 1. `candidates.jsonl` 을 metadata 에 결박했습니다

지적이 정확합니다 — `accepted.jsonl` 에서 고친 것과 **같은 구멍**이 후보 쪽에 남아
있었습니다.

```
_candidates.json 에 추가
  candidates_sha256     후보 파일의 digest
  candidates_rows       행 수
metadata 도 임시 파일 뒤 os.replace  (두 파일 교체 사이 불일치는 digest 로 fail-closed)
```

순서 2 가 쓸 계약 함수를 만들었습니다 — `check_candidates_contract(overlay_dir)`.

```
candidates_written is True / invalidated 아님 / errors 0
dirty_build false / built_from_commit 40 자리 hex
candidates_sha256 를 **다시 계산해** 대조
candidates_rows 를 **다시 세어** 대조
input_sha256 를 다시 계산해 대조
**모든 행**에 kind == answer_candidate 와 validate() 통과를 요구
```

실측(후보 2,801 행):

```
후보 계약 통과 (문제 0)                         O
**후보를 한 글자 바꾸면 거부**                  O   ("candidates.jsonl 이 바뀌었다")
  -- "proposal": "named" -> "no_name" 한 곳
행 수를 거짓으로 적으면 거부                    O
**스키마를 어긴 행을 끼워 넣으면 거부**         O   (state=approved_confirm 한 행)
무효화 뒤에는 계약이 거부                       O
후보 metadata 의 dirty_build 도 거부            O
```

## 2. `state_of()` 의 도메인 강제

지적하신 세 조합이 모두 통과했습니다.

```
named            + "garbage"     ->  ('pending_cannot_determine', 'undetermined')
no_name          + "different"   ->  ('pending_no_name_adjudication', …)
cannot_determine + "same"        ->  ('pending_cannot_determine', …)
```

마지막 포괄 반환이 임의 문자열을 보류로 흡수하고 있었습니다. 지정해 주신 도메인을 **먼저**
강제합니다.

```
named                       comparison ∈ {same, alias, different, cannot_determine, None}
no_name, cannot_determine   comparison is None
그 밖                       SchemaError
```

```
named + "garbage"            ->  "모르는 comparison 'garbage'"
no_name + "different"        ->  "no_name 에는 comparison 이 없어야 한다"
cannot_determine + "same"    ->  같은 메시지
named + 3 (정수)             ->  "모르는 comparison 3"
```

`named + None` 은 여전히 `pending_cannot_determine` 입니다(비교가 결론을 내지 못한 경우).

## 3. `source_answer_ids` 자료형

```
"A-1" (문자열 하나)   ->  전에는 통과.  "A-1" 이 목록으로 쓰이면 글자 단위로 돈다
[]                    ->  전에는 통과
[""]                  ->  전에는 통과
["A", "A"]            ->  전에는 통과
```

지정하신 대로 **비어 있지 않은 `list[str]`, 원소도 비지 않고 중복 없음** 으로 강제하고,
`unit_adjudication` 과 `raw_overlay` **둘 다**에 걸었습니다. `answers` 는 목록인지
검사합니다.

`answers` 와 `source_answer_ids` 의 ID 집합 일치는 **순서 2 에서** 하겠습니다 — 지금은
`answers` 를 만드는 코드가 없습니다.

## 4. 자기검사 — 131 → 159 항목

```
1) 상태·규칙표                                   8
2) 1 차 답 -> 상태 (**도메인 거부 5** 포함)      29
3) answer_candidate (no_name 실경로 + 의미 모순)  21
4) unit_adjudication (**id 자료형 7** 포함)       36
5) raw_overlay                                    8
6) 입력 계약 실패 주입                           17
7) 신뢰 해시 사전 대조                            3
8) 실제 후보 생성                                11
9) 모든 실패에서 무효화                          14
10) **후보 결박**                                12
                                               159/159
```

전체:

```
test_overlay_candidates    159/159
test_expr_compare           92/92     test_ingest_answers    80/80
test_packet_export          58/58     test_bundle_guards     63/63
test_provenance_fixture     22/22     test_segment_verdict    6/6
test_provenance_untouched    3/3      test_lowering_proof     6/6
test_tracer_alias            4/4      test_corrections_cover  5/5
                           498 항목
```

## 5. 읽을 곳

```
tracer 2142b799
  develop/overlay_schema.py            PROPOSALS·COMPARISONS / _id_list /
                                       check_candidates_contract
  develop/build_overlay_candidates.py  candidates_sha256·rows / metadata 원자적 교체
  develop/test_overlay_candidates.py   159 항목 (10 절이 후보 결박)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 6. 원하는 판정

> 셋이 막혔는가. 순서 2 가 후보를 안전하게 소비할 수 있는 상태인가.
>
> 순서 2 에서 만들 것을 미리 적어 두겠습니다 — 틀렸으면 지금 잡아 주십시오.
>
> ```
> build_unit_adjudication.py
>   1  check_candidates_contract() 로 후보를 받는다 (문제가 있으면 만들지 않는다)
>   2  decision_unit_id 로 묶는다. 같은 단위의 답이 둘이면 중복 배정이다
>   3  response_agreement  = 두 답의 (proposal, proposed_expr 정규형) 이 같은가
>      semantic_resolution = 그 답이 축 이름을 정했는가
>                            (proposal == named 이고 비교가 same|alias|different)
>   4  상태:  답 1 개        -> 그 답의 state 를 그대로
>             둘이 같다      -> response_agreement
>             둘이 다르다    -> conflict_duplicate
>             답 없음        -> pending_first_pass
>   5  **approved_* 를 만들지 않는다.** 사람 판정은 다음 단계다
>   6  불일치율을 family·grade·model 별로 집계 (분모는 unit, raw site 는 영향 범위)
> ```
>
> `response_agreement` 를 비교할 때 `proposed_expr` 를 **`expr_compare` 로** 보는 것이
> 맞습니까? 문자열이 달라도 `same`/`alias` 면 같은 답으로 셀 수 있는데, 그러면 "두 응답이
> 같다" 가 아니라 "두 응답이 의미상 같다" 가 됩니다. 문자열 정규형만 보는 편이
> 나을까요?
