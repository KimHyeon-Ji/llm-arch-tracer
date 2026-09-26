# 0-c 순서 1 — 의미 결박 · 상태별 필드 표 · 실패 시 무효화 · 사전 대조 완전성

요청일: 2026-09-26. 선행: `codex_ask_0c_step1b.md` (→ **미승인. 차단 결함 넷**).

넷 다 재현하고 고쳤습니다. 순서 2 로 넘어가지 않았습니다.

```
tracer            e70928be
results-labeled   749a1b3c   (파일럿 3 건 그대로, revision 7)
```

---

## 1. `answer_candidate` 의 의미 결박

지적하신 두 조합이 모두 통과했습니다.

```
state=proposed_rename / candidate_kind=no_name_exists /
  proposal=named / comparison=same              ->  통과 (모순)
site=[m, prefill, …] / model=other / phase=decode ->  통과
```

이제 `validate()` 가 **`state_of(proposal, comparison)` 을 다시 계산해** 대조합니다.

```
state 가 proposal·comparison 과 맞지 않는다 ('proposed_rename', 기대 'proposed_confirm')
candidate_kind 가 맞지 않는다 ('no_name_exists', 기대 'confirm')
site 의 model·phase 가 레코드와 다르다 (['m','prefill'] != ['other','decode'])
```

`site[:2] == [model, phase]` 는 `answer_candidate` 와 `raw_overlay` **둘 다**에
강제합니다. `answer_candidate` 에는 승인 필드를 아예 두지 않았으므로 `approval_id` 를
넣으면 "모르는 필드" 로 걸립니다.

## 2. 상태별 필드 표

지적하신 네 가지가 전부 통과했습니다. 표 하나로 바꿨습니다.

```python
STATE_FIELDS = {                      # (필수, 금지)
  "pending_first_pass":  ((), source_answer_ids·answers·agreement 2·승인 3·결과 3),
  "response_agreement":  (source_answer_ids·response_agreement·semantic_resolution,
                          승인 3·결과 3),
  "conflict_duplicate":  (source_answer_ids·response_agreement, 승인 3·결과 3),
  approved_*:            (source_answer_ids·승인 3, ()),
}
_DEFAULT_UNIT_FIELDS =   (source_answer_ids, agreement 2·승인 3·결과 3)
```

추가로:

```
agreement 두 필드는 **엄격한 bool**        semantic_resolution="no"  -> 거부
response_agreement 상태 -> response_agreement is True 를 요구
conflict_duplicate      -> response_agreement is False 를 요구
answers 는 목록이어야 한다
```

실측:

```
pending_first_pass + answers=[…]   ->  "pending_first_pass 에서는 `answers` 를 쓸 수 없다"
response_agreement=False           ->  "상태인데 response_agreement 가 참이 아니다"
semantic_resolution="no"           ->  "`semantic_resolution` 는 bool 이어야 한다"
pending_human + approval_id        ->  "pending_human 에서는 `approval_id` 를 쓸 수 없다"
```

## 3. "옛 후보까지 지운다" 가 구현과 반대였습니다

지적이 정확했습니다. 입력 계약·사전 해시 실패에서 `cands is None` 으로 곧바로 돌아가
**옛 `candidates.jsonl` 이 그대로 남았고**, 제 시험은 오히려 "후보 파일이 그대로다" 를
성공 조건으로 두고 있었습니다.

```
지금
  성공        임시 파일에 다 쓴 뒤 os.replace 로 **원자적 공개**
  모든 실패   invalidate() 가 공개 파일을 **지운다**
              _candidates.json 에 invalidated: true, reason 을 남긴다
              입력 계약·사전 대조 실패 -> exit 2
              생성 중 오류             -> exit 3 (+ _rejected.jsonl 진단)
```

시험도 `not exists` 를 확인하도록 바꿨습니다.

```
사전 대조 실패     exit 2  /  **공개 후보가 사라진다**
units 파일 사라짐  exit 2  /  공개 후보가 사라진다
입력 계약 위반     exit 2  /  공개 후보가 사라진다
되돌리면           exit 0  /  다시 만들어진다
```

## 4. 사전 대조 완전성 — "10+10 전부" 를 사실로 만들었습니다

지적하신 대로 전에는 디렉터리를 열거해서 **파일이 사라진 것 자체를 검사하지 못했고**,
crosswalk 는 accepted 행에서 쓰인 것만 봤습니다.

```python
expected_from_manifest(trusted, "units")      # manifest 키에서 결정적으로 뽑는다
expected_from_manifest(trusted, "crosswalk")
check_set(expected, actual, verified, label)   # 기대 == 실제 == 검증
```

* **units**: manifest 의 `*.units.jsonl` 10 개를 기대 집합으로 두고, 디렉터리의 실제
  집합과 검증 집합을 대조합니다. 하나를 감추면 `exit 2` 입니다.
* **crosswalk**: 이 실행이 쓸 집합을 accepted 행에서 결정적으로 뽑고, 그것이 manifest
  기대 집합의 **부분집합**인지 먼저 봅니다(밖의 파일이 필요하면 즉시 거부). 그 뒤
  **쓸 것 == 쓴 것 == 검증한 것** 을 확인합니다.

지적하신 두 갈래 중 후자를 택했습니다 — crosswalk 는 10 개 전부가 아니라 **그 실행에
필요한 subset** 을 assignment 에서 결정적으로 파생하고 그 집합의 완전성을 검사합니다.
"10+10 전부 사전 대조" 주장은 units 에만 해당하고, crosswalk 는 subset 입니다.
`_candidates.json` 에 `units 검증` / `crosswalk 검증` 수를 적습니다.

## 5. `approved_confirm.resulting_expr` — null 강제로 좁혔습니다

지시대로 바꿨습니다.

```
approved_confirm         resulting_expr = null
approved_rename          resulting_expr = 새 식 필수
approved_no_name_exists  resulting_expr = null
```

현재 라벨을 적어도 거부합니다("null 이어야 한다"). 같은 결정을 두 방식으로 직렬화하지
않는다는 이유를 docstring 에 적었습니다.

## 6. 자기검사 — 109 → 131 항목

```
1) 상태·규칙표                                 8
2) 1 차 답 -> 상태 (approved 경로 없음)        22
3) answer_candidate (no_name 실경로 + 의미 모순 5)  21
4) unit_adjudication 상태별 결박               28  <- agreement·승인 금지 포함
5) raw_overlay (site 결박 포함)                 7
6) 입력 계약 실패 주입                          17
7) 신뢰 해시 사전 대조                           3
8) 실제 후보 생성                               11
9) **모든 실패에서 무효화**                     14
                                             131/131
```

전체:

```
test_overlay_candidates    131/131
test_expr_compare           92/92     test_ingest_answers    80/80
test_packet_export          58/58     test_bundle_guards     63/63
test_provenance_fixture     22/22     test_segment_verdict    6/6
test_provenance_untouched    3/3      test_lowering_proof     6/6
test_tracer_alias            4/4      test_corrections_cover  5/5
                           470 항목
```

## 7. 읽을 곳

```
tracer e70928be
  develop/overlay_schema.py            STATE_FIELDS / 의미 재계산 / site 결박
  develop/build_overlay_candidates.py  expected_from_manifest / check_set / invalidate
  develop/test_overlay_candidates.py   131 항목 (9 절이 무효화)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> 넷이 막혔는가. 특히
>
> * `answer_candidate` 가 `state_of()` 를 다시 계산해 대조하는 것이 충분한가
>   (`proposal`·`comparison` 이 진리값이고 `state`·`candidate_kind` 는 그것의 함수라는
>   전제입니다)
> * `STATE_FIELDS` 표가 상태별 계약을 빠짐없이 덮는가
> * crosswalk 를 "10 개 전부" 가 아니라 **그 실행에 필요한 subset 의 완전성**으로 검사한
>   판단이 맞는가
>
> 승인되면 순서 2(1·2 차 답 수집과 Q5 중복 조정)로 갑니다. 그때 만들 것은
> `unit_adjudication` 레코드를 **판정 단위 단위로** 만드는 도구이고, 지적하신 대로
> 분모를 unit/answer 로 두고 raw site 수는 영향 범위 지표로만 쓰겠습니다.
