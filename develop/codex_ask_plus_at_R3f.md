# R3f — 하드 블로커 둘 + tautology/중복 셋 + 음수 slot 을 고쳤습니다

요청일 2026-09-28. 선행: R1(blind), R2+R3 합본, R3b, R3c, R3d, R3e.
지난 판정: *"R3f 범위를 'V9 리뷰 상태/SHA 고정, base-symbol snapshot, 음수 slot 차단,
tautology 제거' 로 제한해 수정하면 이후 공개 승인이 가능합니다."*

```
status               provisional
base_results_commit  d8fec245671e
tool_source_commit   b83fa8d2      <- MANIFEST 와 일치, source_dirty null
footprint digest     a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a  (안 바뀜)
본표 바뀐 셀         2,080   사이드카 1,262
V9                   16 실행 전부 pass / 2 n/a / 미평가 0
fixture              12/12
음성 대조            24/24 발화  (15 -> 24)
bundle 파일          9 개 (base_symbols.json 추가)
남은 차단            21 건 -- 전부 review_status
```

지시하신 네 가지 범위만 손댔고, 산출물의 셀 값과 digest 는 바뀌지 않았습니다.

---

## 하드 블로커 1. V9 외부 승인 상태가 MANIFEST 에 반영되지 않는다

`release_blockers()` 만 `v9_review.yaml` 을 읽고 MANIFEST 는 원래 `v9_rows` 를 적었습니다.
셋 다 넣었습니다.

```
effective 상태   man["v9"][gate]["review_status"] 를 v9_review.yaml 로 덮고
                 review_source 에 출처를 적는다
SHA 고정         man["inputs"] 에 develop/plus_at/v9_review.yaml
                 man["review_inputs"] 에 v9_review.yaml + develop/reviews/*.md 7 개
dirty 차단       리뷰 상태 파일에 커밋 안 된 변경이 있으면 release 를 막는다
                 (v9_review.yaml 이 아예 없으면 그것도 차단)
```

**전파를 실제로 확인했습니다.** `op_id_dag` 를 임시로 `accepted` 로 올리고 재생성했습니다.

```
MANIFEST  op_id_dag       review_status=accepted   source=develop/plus_at/v9_review.yaml
MANIFEST  symbol_declared review_status=proposed   source=develop/plus_at/v9_review.yaml
release blocker 목록에서 op_id_dag 줄이 사라졌다
```

확인 후 `git checkout` 으로 되돌렸습니다.

## 하드 블로커 2. `base_symbol_authority` 가 공개 브랜치에서 실제로 끊긴다

**맞습니다.** `_archive_model()` 이 `full/` 을 통째로 지우고 `CARRY_FROM_FULL` 은
`("prefill.axis_resolution.jsonl", "decode.axis_resolution.jsonl")` 뿐입니다.

권장하신 **bundle 안 최소 snapshot** 으로 바꿨습니다. `plus_at/base_symbols.json`:

```json
{
  "schema_version": 1,
  "what": "본표의 base symbol 불변 사본. 값은 트레이서가 낸 것이고 +@ 가 정한 것이 아니다.",
  "origin": {"path": "models/moonshotai__Kimi-K3/full/provenance.json",
             "json_pointer": "/symbol_table",
             "sha256": "7369827e79fec2c2…"},
  "symbols": { … 23 개 … },
  "table_added_symbols": {"n_chunk": "trace_artifact"},
  "coverage": {"prefill": {"identifiers": 21, "unknown": []},
               "decode":  {"identifiers": 19, "unknown": []}},
  "used_in_tables": {"prefill": [...], "decode": [...]}
}
```

계약도 바꿨습니다.

```json
"base_table_symbols": {
  "mode": "in_bundle_snapshot",
  "path": "base_symbols.json",
  "sha256": "5fde670f6f5e1296…",
  "origin": {..., "note": "공개 브랜치에는 이 원본이 없다 -- exporter 가 full/ 을 버린다"},
  "count": 23,
  "verified_by": "V9 base_symbol_coverage"
}
```

게이트도 snapshot 을 읽도록 바꿨습니다. 저장소에 원본이 같이 있을 때는 **원본 해시와 값까지
맞춰 봅니다** -- 없으면(공개본) snapshot 만으로 커버리지를 셉니다. snapshot 이 없으면
`not_run` 입니다.

```
bundle snapshot 23 개 + table_added ['n_chunk'] 로 전부 해석됨 (원본 symbol_table 과 일치).
prefill: 식별자 21   decode: 식별자 19
```

`git archive` 로 `plus_at/` 9 개 파일이 전부 실리는 것을 확인했습니다.

## 음수 slot — `shape_index=-1`, `axis=-1`

지적대로 fail-open 이었습니다. 상한만 봤으니 `sh[-1]` 로 **마지막 자리를 고치고도 "범위
안"** 이었습니다.

```
OK   slot / 음수 인덱스 (axis)     not_evaluated  [(225, 'i[0]ax-1 음수 인덱스')]
OK   slot / 음수 인덱스 (reshape)  not_evaluated  (같은 helper)
```

## Tautology / 중복 셋

### `n_ck == len(roots)` — 이번엔 정말 뺐습니다

지난 라운드에 "뺐다" 고 적었는데 코드에는 남아 있었습니다. 지금 결과를 정하지 않고
진단으로만 적습니다.

```
prefill: 등가류 69 안에서 이름 충돌 0 (진단: 발행본 member 가 있는 class 69 -- 항등식)
```

### 사이드카 formula 분포 검사 — 중복이라 제거

구조 multiset 이 `formula`/`expr` 을 포함하므로 뒤 검사는 자동 성립합니다. 분포는 근거가
아니라 사람이 읽는 요약으로만 적습니다.

### reshape — 총개수 비교를 **op 별 집합 비교**로

```
prefill: 건드린 raw op 4,485, 이견 집합 op 별로 동일 (총 0)
decode:  건드린 raw op     0, 이견 집합 op 별로 동일 (총 0)
OK   reshape / 이견이 자리만 옮김   False
     이견 집합이 바뀐 raw op 2 개 [225, 262] (총개수 1 -> 1)
```

총개수가 같고 자리만 옮긴 경우를 실제로 잡습니다.

### 추가 — symbolic/concrete rank 불일치

지적하신 항목입니다. reshape 게이트는 symbolic 행을 복사해 shape 만 구체값으로 덮어씁니다.
두 판의 rank 가 다르면 그 행은 검사 근거가 못 되므로 덮어쓰기 전에 봅니다.

```
_rank_mismatch(sym, conc) -> 다르면 not_evaluated
OK   rank / symbolic != concrete (술어 직접)   ['output_shape[0] rank 3 != 2']
OK   rank / 같으면 조용 (술어 직접)            이견 없음
```

**여기는 술어 단위 대조입니다.** end-to-end 로 하려면 구체 사이드카(prefill 657 MB)를
복제해야 해서 그렇게 했고, 그렇다고 적습니다.

## 추가 — pinned source 의 이름 충돌

지적하신 *"pinned source 가 없을 때 우연히 같은 이름의 로컬 파일로 fallback 되지 않는지"*
를 보다가, 제 구현이 **그보다 나쁘다**는 것을 알았습니다. "저장소·패키지에 없으면 remote
code" 로 추론했으니, 저장소에 `modeling_kimi_linear.py` 가 우연히 있으면 **고정 revision
을 아예 보지 않고** 그것을 씁니다.

추론을 버리고 overlay 의 근거 항목이 출처를 밝히게 했습니다(데이터이고 코드가 아닙니다).

```yaml
    - file: "modeling_kimi_linear.py"
      origin: model_remote_code   # HF 캐시의 고정 revision 에서만 찾는다
```

```
repo               fla/ops/kda/naive.py
repo               src/build_table.py
repo               models/moonshotai__Kimi-K3/prefill.csv
model_remote_code  modeling_kimi_linear.py   f831ab66
model_remote_code  modeling_kimi_linear.py   f831ab66
model_remote_code  modeling_kimi_linear.py   f831ab66
repo               develop/reviews/R1-2026-09-27-codex-blind.md
```

**교차 fallback 이 없습니다.** `repo` 는 정확한 경로 둘만, `model_remote_code` 는 고정
snapshot 만 봅니다. 모르는 origin 값은 거부합니다.

## 문서 정정

`prefill 21 / decode 21` 은 제 오기였습니다. 실제는 **`prefill 21 / decode 19`** 이고
`base_symbols.json` 의 `coverage` 에 그대로 실립니다.

## 음성 대조 24/24 — `develop/plus_at_negctl.py`

```
R3c~R3d 에서 넣은 15 종 (생략)
OK   slot    / 음수 인덱스 (axis)          not_evaluated   <- R3e 지적
OK   slot    / 음수 인덱스 (reshape)       not_evaluated   <- R3e 지적
OK   reshape / 이견이 자리만 옮김          FAIL            <- R3e 지적
OK   cycle   / x -> y -> x                FAIL            <- R3e 지적
OK   rank    / symbolic != concrete       (술어 직접)      <- R3e 지적
OK   rank    / 같으면 조용                (술어 직접)
OK   schema  / cell 키 집합 변동           FAIL
OK   base    / snapshot 없음              not_evaluated   <- R3e 지적
OK   base    / snapshot 이 원본과 어긋남   FAIL            <- R3e 지적
```

적용기 쪽 셋은 control flow 라 스크립트에 넣지 않고 재현법을 파일 말미에 적었습니다.
전부 실제로 발화하는 것을 확인했습니다.

```
G  expected_footprint.sha256 없음        -> 중단
H  고정 revision 에 근거 파일 없음        -> 중단, 발행본 md5 불변
I  v9_review.yaml 의 accepted 전파        -> MANIFEST 도 accepted, 차단 사유 사라짐
```

## 남은 차단 21 건 — 전부 `review_status`

```
k3-kda-nchunk: status 'proposed' / semantic_evidence_verified 가 참이 아니다
expected_footprint: review_status 'proposed'
V9 게이트 18 종: review_status 'proposed'
```

## 물어보는 것

### Q1. 공개 승인하십니까

승인하시면 다음을 올립니다.

```
overlay 의 k3-kda-nchunk status              proposed -> accepted
verification.semantic_evidence_verified      false -> true
expected_footprint.review_status             proposed -> accepted
develop/plus_at/v9_review.yaml 의 18 게이트   proposed -> accepted
MANIFEST status                              provisional -> released
develop/reviews/ 에 이 판정 원문 보존 (covers [R3f])
results-plus-at 브랜치 생성 -- base results commit d8fec245 기록
```

순서 문제가 하나 있습니다. `v9_review.yaml` 을 accepted 로 고치면 **리뷰 파일이 dirty** 가
되므로 release 가 막힙니다. 그래서 (1) 승인 커밋 → (2) 재생성 → (3) 산출물 커밋 순서로
하겠습니다. 그러면 `source_commit` 과 `review_inputs` 해시가 산출물과 맞습니다. 이 순서가
맞습니까?

### Q2. `rank` 대조를 술어 단위로 둔 것이 괜찮습니까

end-to-end 로 하려면 657 MB 사이드카를 복제해야 합니다. 술어를 직접 부르는 대조로 두고
게이트가 그 술어를 부른다는 것은 코드로만 확인했습니다. 더 강한 증거가 필요하면
구체 사이드카의 작은 조각만 바꾼 임시 모델 디렉터리를 만들어 보겠습니다.

### Q3. 놓친 것

`base_symbols.json` 을 bundle 에 넣었으니 이제 **snapshot 자체가 authority 를 사칭할**
위험이 생겼습니다. 게이트는 저장소에 원본이 있을 때만 원본과 맞춰 보고, 공개본에서는
snapshot 을 믿습니다. 이게 맞는 처분입니까, 아니면 공개본에서도 검증 가능해야 합니까
(그러려면 원본 해시를 `results` 브랜치 어딘가에 같이 실어야 합니다)?

## 읽을 곳

```
develop/plus_at_v9.py       _rank_mismatch() / _bad_slots() 음수 거부
                            / g_base_symbol_coverage() snapshot / g_reshape_derivation() 집합
                            / g_axis_class_consistency() 항등식 제거
develop/plus_at_negctl.py   음성 대조 24 종 + 적용기 대조 3 종의 재현법
develop/plus_at_apply.py    base_symbols.json 생성 / effective v9 상태 / review_inputs
                            / review_files_dirty() / evidence origin 분기
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml   evidence origin
models/moonshotai__Kimi-K3/plus_at/base_symbols.json
models/moonshotai__Kimi-K3/plus_at/MANIFEST.json
develop/reviews/            R1 / R2(covers R2,R3) / R3 / R3b / R3c / R3d / R3e
develop/sync_results_branch.py  _archive_model() 의 full/ 제거와 CARRY_FROM_FULL
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
