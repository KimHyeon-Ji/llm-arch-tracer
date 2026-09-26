# 0-c 순서 1: raw overlay 후보 + 판정 상태 스키마

요청일: 2026-09-26. 선행: `codex_ask_0c_fixed3.md` (→ **승인. 순서 1 로 진행**).

완료 조건으로 지정하신 입력 계약을 **실제 코드와 실패 주입 검사로** 넣었습니다.
overlay 를 적용하지 않았고, actionable verdict 를 하나도 만들지 않았습니다.

```
tracer            329608c4
results-labeled   749a1b3c   (파일럿 3 건 그대로, revision 7)
```

---

## 1. 키는 원장 사이트 튜플

```
site = (model, phase, raw_op_id, field, shape_index, axis)
```

발행 셀 하나가 raw 사이트 여러 개에서 옵니다(`_collapse_norm` 이 합성한 norm 행은 입력이
`first`, 출력이 `last`, 가중치가 `*.weight` 를 가진 멤버에서 왔습니다). 발행 셀을 키로 쓰면
그 구조를 되돌릴 수 없으므로 crosswalk 의 `raw_sites` 로 펼칩니다. 후보에는 `origin`
(`raw_slot` / `synthesized_norm` / `canonical_weight` / `ambiguous`)과 `published_cell` 도
함께 적어, 순서 4 에서 발행 쪽으로 내려갈 때 쓸 수 있게 했습니다.

## 2. 상태 11 개 — actionable 은 셋뿐이고 전부 사람 승인을 거친다

```
상태                            actionable  뜻
pending_first_pass                -       답이 아직 없다
proposed_confirm                  -       1차가 현재 라벨과 같은 이름 (same/alias)
proposed_rename                   -       1차가 다른 이름 (different)
pending_no_name_adjudication      -       1차가 no_name -- 라벨·grade 를 바꾸지 않는다
pending_cannot_determine          -       판정 보류 또는 비교 불가
conflict_duplicate                -       중복 배정 둘이 어긋난다 -> 2차
pending_human                     -       사람 판정 대기
approved_confirm                  O       사람이 승인 -- 현재 라벨 확정
approved_rename                   O       사람이 승인 -- 이름 교체
approved_no_name_exists           O       사람이 승인 -- 정수 축으로 복원
withdrawn                         -       철회
```

지시하신 대로 **`no_name` 과 `cannot_determine` 는 actionable verdict 가 아니라 대기
상태**입니다. 1 차만으로 라벨도 grade 도 바꾸지 않고, 자동 `demote` 도 없습니다.

### 스키마가 그것을 강제합니다

```
actionable 이 아니면        resulting_expr 와 grade_after 가 **반드시 None**
actionable 과 state 불일치   거부
approved_rename             resulting_expr 없으면 거부
approved_no_name_exists     resulting_expr 있으면 거부 (정수 축으로 돌아가므로)
actionable 인데 grade_after 없음  거부
```

`state_of(proposal, comparison)` 은 1 차 답에서 **`proposed_*` 와 `pending_*` 만**
만듭니다. `approved_*` 를 만들 경로가 없습니다.

```
named / same        -> proposed_confirm
named / alias       -> proposed_confirm
named / different   -> proposed_rename
named / cannot_determine 또는 None -> pending_cannot_determine
no_name             -> pending_no_name_adjudication
cannot_determine    -> pending_cannot_determine
그 밖              -> SchemaError
```

## 3. 입력 계약 — 코드로 막고 주입으로 증명했습니다

`overlay_schema.check_input_contract()` 가 후보를 만들기 전에 봅니다.

```
_ingest.json     errors == 0
                 rejected == 0
                 strict == true
                 dirty_build == false
                 input_sha256 를 **다시 계산해** 대조 (기록만 읽지 않는다)
accepted.jsonl   모든 행이 eligible_for_adjudication == true
                 packet_quarantined == false
                 assignment_revision 이 현재 배정과 일치
```

하나라도 어긋나면 `build()` 가 `None` 을 돌려주고 산출물을 쓰지 않습니다.

### 실패 주입 — 아홉 가지를 하나씩 어겨 봤습니다

```
_ingest.json errors=1                  -> 거부   O
             rejected=1                -> 거부   O
             strict=false              -> 거부   O
             dirty_build=true          -> 거부   O
             input_sha256={}           -> 거부   O
기록된 입력 파일을 실제로 한 줄 바꿈    -> 거부   O  ("입력이 바뀌었다")
             되돌리면 통과             -> 통과   O
accepted eligible_for_adjudication=false -> 거부  O
         packet_quarantined=true        -> 거부  O
         assignment_revision 불일치     -> 거부  O
_ingest.json 삭제                      -> 거부   O
```

`input_sha256` 재계산이 실제로 발화하는지 보려고 **기록된 입력 파일을 진짜로 바꿔** 봤고,
"입력이 바뀌었다" 로 걸렸습니다.

### 시험 환경에 대한 주의

이 자기검사는 `ALLOW_DIRTY_BUILD` 로 돌기 때문에 수집 산출물의 `dirty_build` 가 **참**
입니다. 그래서 시험은 먼저 *계약이 그것을 거부하는지* 확인하고, 그 다음에 그 한 필드만
깨끗한 값으로 바꿔 기준선을 만듭니다. 계약을 무르게 한 것이 아닙니다 — 그 항목의 주입
검사는 따로 있습니다.

## 4. 실제로 후보를 만들어 확인했습니다

파일럿 shard 하나(12 단위)로 합성 답을 만들어 돌렸습니다.

```
후보가 만들어졌다                          O
만드는 중 문제 0                           O
**actionable 이 하나도 없다**              O
**resulting_expr 가 전부 None**            O
**grade_after 가 전부 None**               O
스키마를 전부 통과한다                     O
site 가 6-튜플 / raw op_id 가 정수          O
발행 셀·origin 이 기록된다                 O
상태가 1 차 것뿐  {pending_cannot_determine, proposed_confirm}
입력 목록에 crosswalk 이 들어간다          O
```

같은 사이트에 후보가 둘 이상인 경우(중복 배정)는 세기만 하고 조정하지 않습니다 —
그것이 순서 2 입니다.

## 5. 자기검사

```
test_overlay_candidates    69/69     <- 새로 만듦
test_expr_compare          92/92
test_ingest_answers        80/80
test_packet_export         58/58
test_bundle_guards         63/63
test_provenance_fixture    22/22
test_segment_verdict        6/6      test_provenance_untouched  3/3
test_lowering_proof         6/6      test_tracer_alias          4/4
test_corrections_cover      5/5
                          408 항목
```

## 6. 읽을 곳

```
tracer 329608c4
  develop/overlay_schema.py            상태·verdict·필드·검사·입력 계약
  develop/build_overlay_candidates.py  crosswalk 로 raw 사이트까지 펼친다
  develop/test_overlay_candidates.py   69 항목 (4 절이 주입 검사)

실행해 볼 것
  .venv\Scripts\python.exe develop\overlay_schema.py
      -> 상태 표. actionable 3 개
```

아직 실제 답이 없으므로 `work/overlay/` 는 만들지 않았습니다(자기검사는 임시
디렉터리에서 돕니다).

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 7. 원하는 판정

> 순서 1 이 완료 조건을 채웠는가. 특히
>
> * `actionable` 이 사람 승인 상태에만 붙고, 그 밖에서는 `resulting_expr`·`grade_after`
>   를 쓸 수 없게 막은 것이 충분한가
> * 입력 계약 아홉 가지를 실패 주입으로 확인한 것이 요구하신 "실제 코드와 실패 주입
>   검사" 인가
>
> 그리고 순서 2 를 시작하기 전에 두 가지 확인을 부탁합니다.
>
> **Q1.** `conflict_duplicate` 판정 기준. 중복 배정 둘이 **같은 이름**을 냈으면
> `agreement` 인데, 둘 다 `cannot_determine` 을 냈으면 "일치" 입니까, 아니면 여전히
> `pending_cannot_determine` 입니까? 저는 후자(일치로 세지 않는다)가 맞다고 봅니다 —
> 둘이 모른다고 한 것이 아는 것이 되지는 않으니까요.
>
> **Q2.** 한 단위의 발행 셀이 여러 개일 때, 후보가 raw 사이트마다 하나씩 생깁니다
> (지금 파일럿에서 12 단위 → 후보 수가 더 많습니다). 순서 2 의 중복 조정은 **단위
> 단위로** 해야 합니까, **사이트 단위로** 해야 합니까? 1 차 답은 단위 하나에 하나뿐이라
> 단위 단위가 자연스러운데, 사이트 단위로 보면 같은 답이 여러 번 세어집니다.
