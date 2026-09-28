# R3e — 차단 4 건 + Q4 fail-open 4 건을 고쳤습니다. 공개 승인을 청합니다

요청일 2026-09-28. 선행: R1(blind), R2+R3 합본, R3b, R3c, R3d.
지난 판정: *"reshape slot 검증, self-cycle, sidecar expression integrity, 실제 base-symbol
authority 를 닫고, SHA/source fallback 까지 fail-closed 로 바꾼 짧은 R3e 가 필요합니다."*

```
status               provisional
base_results_commit  d8fec245671e
tool_source_commit   43371702      <- MANIFEST 와 일치. source_dirty null
footprint digest     a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a  (안 바뀜)
본표 바뀐 셀         2,080   (계열 A)
사이드카 셀          1,262   (계열 C)
V9                   16 실행 전부 pass / 2 n/a / 미평가 0   (게이트 18 종, 둘은 새로 넣음)
fixture              12/12
음성 대조            15/15 발화  (6 -> 15)
남은 차단            21 건 -- 전부 review_status
```

**먼저 적습니다: 지난 라운드에 제가 "6/6 발화" 를 근거로 냈는데 검토자가 probe 를 더 짜서
fail-open 넷을 더 찾았습니다.** 제 음성 대조가 제가 고친 자리만 겨냥했기 때문입니다.
이번에는 지적받은 자리마다 대조를 붙여 15 종으로 늘렸고, 아래 각 항목에 **발화 증거**를
같이 적습니다.

---

## 차단 1. reshape 의 raw slot 범위가 fail-open

지적이 정확했고 **재현하신 결과도 재현됩니다.** axis 에만 범위 검사를 넣고 reshape 에는
넣지 않아서, 축이 범위 밖이면 그 변경을 적용하지 않은 채 "이견 0 → 0" 을 냈습니다.

말씀대로 **helper 를 빼서 두 게이트가 공유**합니다.

```python
def _bad_slots(cw, changed, phase, raw_by, conc_by):
    """crosswalk 이 가리킨 raw 자리가 원장에 실제로 있는가. 두 게이트가 공유한다."""
```

```
OK   reshape / 범위 밖 축   not_evaluated
     prefill: crosswalk 이 가리킨 raw slot 1 개를 원장에서 찾을 수 없다 [(225, 'i[0]ax99 범위 밖')]
OK   axis    / 범위 밖 축   not_evaluated   (같은 helper, 같은 메시지)
```

## 차단 2. `expression_no_cycle` 이 직접 자기 참조를 놓친다

`and t != name` 을 지웠습니다. 기존 DFS 의 stack 검사가 바로 잡습니다.

```
OK   cycle / x: {expr: x}   False   -- 순환 참조 ['x']
```

## 차단 3. 사이드카 symbol/formula 무결성이 V9 에서 검사되지 않는다

새 게이트 `sidecar_expression_integrity` 를 넣었습니다. 지정하신 여섯 가지를 전부 봅니다.

```
formula 가 overlay registry 에 존재
record.expr == formulas[record.formula]      <- registry 가 두 군데 있던 문제를 닫는다
식별자가 사이드카 namespace 로 전부 해석   (R_res, L_layers, l + ceil)
join key 유일
식 평가값 == value
선언했는데 레코드가 없으면 not_run (N/A 아니다)
```

**registry 두 군데** 지적이 핵심이었습니다. overlay 의 `residual_sidecar.formulas` 와
`plus_at_resid.TOKEN` 이 갈라져도 아무 검사가 없었습니다. 이제 레코드의 `expr` 을
**overlay registry 와** 맞추므로, 상수가 갈라지면 여기서 FAIL 합니다.

```
OK   sidecar / 미등록 formula          False  formula 'mystery' 가 registry 에 없다
OK   sidecar / registry 불일치         False  expr 'ceil(l/R_res)+99' != registry 'ceil((l+1)/R_res)+1'
OK   sidecar / 식값 != value           False  식값 2 != value 3
OK   sidecar / 선언했는데 레코드 없음  not_evaluated
```

현재 결과: `레코드 1262 개: formula 전부 registry 일치, 식별자 ['L_layers','R_res','l'] +
['ceil'] 로 전부 해석, 식값 == value, join key 유일`.

## 차단 4. `base_table_symbols` 가 dangling pointer

**맞습니다. 제가 설명한 `full/symbol_table.json` 은 없습니다.** 실제 authority 는
`full/provenance.json` 의 `symbol_table` 이고(`v6_substitute()` 가 원래 그것을 읽습니다),
MANIFEST input 에 이미 pin 돼 있었습니다. 제가 계약 문구에 틀린 경로를 적었습니다.

말씀하신 형태로 바꿨습니다.

```json
"base_table_symbols": {
  "mode": "external_reference",
  "path": "models/moonshotai__Kimi-K3/full/provenance.json",
  "json_pointer": "/symbol_table",
  "count": 23,
  "sha256": "7369827e79fec2c20f7e12046ecbd01b0209526c4f1fe9d9bce12ed04357aab3",
  "verified_by": "V9 base_symbol_coverage"
}
```

그리고 **주장으로 두지 않고 세는 게이트**를 넣었습니다 -- `base_symbol_coverage` 가 본표의
모든 식별자가 `authority ∪ table_added_symbols` 에 있는지 봅니다.

```
authority 23 + table_added ['n_chunk'] 로 전부 해석됨.  prefill: 식별자 21  decode: 식별자 21
```

즉 **본표 식별자 21 개 중 20 개가 authority, 1 개(`n_chunk`)가 +@ 가 넣은 것**입니다.
registry 가 실제로 완전하므로 `out_of_scope_trusted_base` 로 범위를 줄이지 않고
external_reference 로 적었습니다. 단독 배포 시 snapshot 을 bundle 에 넣어야 한다는 말씀은
`results-plus-at` 를 만들 때 반영하겠습니다 -- Q2 로 확인받겠습니다.

```
OK   base / 미등록 식별자   False   선언 밖 식별자 ['d_unregistered']
```

## Q4-1. `zero_axis_only_initial_residual` 이 `found ⊆ allow` 만 본다

`set(found) == allow` 로 양방향입니다.

```
현재  `0` 축 2 자리 == 선언 2 자리 (양방향 일치)
OK    zero / 선언했는데 없는 자리   False
      선언 밖 `0` 축 [] / 선언했는데 없는 자리 [('prefill', 999999, 'input_shape', 0, 0)]
```

## Q4-2. `sidecar_phase_consistency` 가 formula 재배치를 놓친다

`op_id` 만 제외한 구조 multiset 을 맞춥니다. 독립 검사에서 보신 대로 **실제로 정확히
일치합니다**(각 631, multiset 동일).

```
_K = (field, shape_index, axis, value, stage, formula, expr, layer_idx, layers,
      module_path, op_type)
OK   sidecar / formula 재배치   False   구조 multiset 이 다르다 -- decode 만 [...]
```

## Q4-3. `expected_footprint.sha256` 이 없으면 경고만 한다

이제 중단합니다. `review_status: accepted` 는 이미 R3d 에서 차단에 넣었습니다.

```
**overlay 에 expected_footprint.sha256 이 없다 -- 고정되지 않았다**
  overlay 의 expected_footprint.sha256 에 이 값을 적어야 한다: a13ed393e0…
```

## Q4-4. source pin 이 fallback 할 수 있다

`glob` 을 **전부 지웠습니다.** 후보는 정확한 경로 둘(`PROJ/p`, site-packages/`p`)뿐이고,
둘 다 없으면 그 파일은 **remote code** 이므로 고정 revision 경로를 **단독 후보**로 씁니다.
없으면 중단합니다. 레코드에 `pinned_revision` 도 적습니다.

```
modeling_kimi_linear.py  ->  .../snapshots/f831ab66…/modeling_kimi_linear.py
                             pinned_revision f831ab66814297da540d832a5235f8e904f29d06
fla/ops/kda/naive.py     ->  .venv/Lib/site-packages/fla/ops/kda/naive.py   (정확 경로)
```

발화 확인(수동. 적용기 control flow 라 스크립트에 넣지 않고 `plus_at_negctl.py` 말미에
재현법을 적었습니다):

```
provenance 의 revision_resolved 를 가짜로 바꾸고 --publish
-> "근거 파일을 못 찾았다: modeling_kimi_linear.py ... publish 하지 않는다" 로 중단
-> 발행본 MANIFEST 의 md5 가 그대로다 (교체는 근거 해석 뒤에 일어난다)
```

## Q3. 항등식 — 권고대로 (b)

`n_ck == len(roots)` 를 **게이트 근거에서 뺐습니다.** detail 문구도 항등식을 세지 않습니다.
axis 게이트의 근거는 crosswalk 커버리지 / raw slot 존재와 범위 / ports coverage / 등가류
내부 label 충돌입니다.

## 메타데이터 — 생성-후-커밋 순서

`tool.source_commit` 을 HEAD 대신 **도구 파일을 마지막으로 건드린 커밋**으로 바꿨습니다.
산출물을 커밋해도 움직이지 않습니다. 그리고 도구에 커밋 안 된 변경이 있으면
`source_dirty` 에 적고 **release 를 막습니다** -- 그 경우 `source_commit` 은 이 산출물을
낸 코드가 아니기 때문입니다.

```
tool_source_commit  43371702   (MANIFEST 와 이 문서가 같다)
source_dirty        null
head_at_generation  43371702
```

## V9 전체 (게이트 18 / 16 실행 pass / 2 n/a / 미평가 0)

```
pass  rerun  axis_class_consistency          crosswalk 커버리지·slot 존재·라벨 충돌 0
pass  rerun  base_symbol_coverage            authority 23 + n_chunk 로 전부 해석      <- 새로
pass  rerun  sidecar_expression_integrity    레코드 1,262 무결성                      <- 새로
pass  rerun  sidecar_phase_consistency       구조 multiset 동일 (op_id 만 제외)
pass  rerun  reshape_derivation              raw op 4,485, 이견 0 -> 0
pass  rerun  port_coverage                   raw 631,705 + 33,199 집합 일치
pass  rerun  zero_axis_only_initial_residual 2 자리 == 선언 2 자리 (양방향)
pass  rerun  expression_no_cycle             self-edge 포함
pass  rerun  schema_shape_rank_token_type / symbol_declared / substitution_nonneg_integer
pass  rerun  batch_seq_head_axis_consistency / head_scope_exclusive
pass  rerun  layers_repeat_consistency / row_metadata_preserved / op_id_dag
n/a          moe_quotient_remainder_consistency   (MoE 철회)
n/a          prefill_decode_structure             (본표 판정이 prefill 전용)
```

## 음성 대조 15/15 — `develop/plus_at_negctl.py`

```
OK   reshape / 없는 raw op            not_evaluated
OK   reshape / 범위 밖 축             not_evaluated     <- R3d 지적
OK   reshape / crosswalk 커버리지     not_evaluated
OK   axis    / 없는 raw op            not_evaluated
OK   axis    / 범위 밖 축             not_evaluated
OK   axis    / crosswalk 커버리지     not_evaluated
OK   port    / raw != ports 집합      FAIL
OK   cycle   / x: {expr: x}           FAIL              <- R3d 지적
OK   sidecar / 미등록 formula         FAIL              <- R3d 지적
OK   sidecar / registry 불일치        FAIL              <- R3d 지적
OK   sidecar / 식값 != value          FAIL              <- R3d 지적
OK   sidecar / 레코드 없음            not_evaluated     <- R3d 지적
OK   sidecar / formula 재배치         FAIL              <- R3d 지적
OK   zero    / 선언했는데 없는 자리   FAIL              <- R3d 지적
OK   base    / 미등록 식별자          FAIL              <- R3d 지적
```

## 남은 차단 21 건 — 전부 `review_status`

```
k3-kda-nchunk: status 'proposed' / semantic_evidence_verified 가 참이 아니다
expected_footprint: review_status 'proposed'
V9 게이트 18 종: review_status 'proposed'   (16 -> 18, 새 게이트 둘)
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
develop/reviews/ 에 이 판정 원문 보존 (covers [R3e])
results-plus-at 브랜치 생성 -- base results commit d8fec245 기록
```

### Q2. `results-plus-at` 를 단독 배포물로 볼 것인지

말씀하신 대로 단독 소비 가능해야 하면 `provenance.json` 의 `symbol_table` snapshot 을
bundle 에 넣어야 하고, 저장소와 함께 소비하는 derived view 라면 path + SHA 외부 참조로
충분합니다. **저는 후자로 보고 있습니다** -- `results-plus-at` 는 `results` 브랜치 위의
derived view 이고 `models/<m>/full/` 이 같은 트리에 있습니다. 이 판단이 맞습니까? 단독
배포 쪽이면 브랜치 생성 때 `plus_at/base_symbols.json` 으로 불변 사본을 넣겠습니다.

### Q3. 제 음성 대조가 아직 좁습니까

지난 라운드에 제 6/6 을 근거로 냈는데 검토자가 네 군데를 더 찾았습니다. 지금 15 종이
**어느 축을 아직 안 건드리는지** 짚어 주시면 좋겠습니다. 제가 보는 빈 자리는 이렇습니다.

```
V1~V8 (적용기 자체 검증) 에는 음성 대조가 없다 -- fixture 12 종이 대신하고 있다
crosswalk 을 망가뜨리는 대조는 전부 prefill 쪽이다
`layers` 접힘·repeat 쪽 게이트에는 대조가 없다
```

### Q4. 놓친 것

`n_ck == len(roots)` 처럼 **구조적으로 발화 불가능한 검사**가 다른 게이트에 또 있는지,
그리고 `not_evaluated` 로 빠질 수 있는 경로 중 실제로는 조용히 통과하는 것이 남았는지
봐 주십시오.

## 읽을 곳

```
develop/plus_at_v9.py       _bad_slots() / g_sidecar_expression_integrity()
                            / g_base_symbol_coverage() / g_zero_axis_*() / g_sidecar_phase_*()
develop/plus_at_negctl.py   음성 대조 15 종 + 적용기 대조 2 종의 재현법
develop/plus_at_apply.py    base_table_symbols / tool_files_commit() / tool_files_dirty()
                            / expected sha 필수 / source 고정 경로 단독
models/moonshotai__Kimi-K3/plus_at/MANIFEST.json   v9 18 종, source_pin, bundle_contract
models/moonshotai__Kimi-K3/plus_at/DIFF.md
develop/plus_at/v9_review.yaml                    게이트 18 종
develop/reviews/            R1 / R2(covers R2,R3) / R3 / R3b / R3c / R3d
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
