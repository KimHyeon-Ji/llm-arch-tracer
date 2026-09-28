# R3d — 차단 4 건 + port_coverage 를 고쳤습니다. 공개 승인을 청합니다

요청일 2026-09-28. 선행: R1(blind), R2+R3 합본, R3b, R3c.
지난 판정: *"이 항목만 고친 짧은 R3d 면 충분하며 R1/R2 재수행은 필요 없습니다."*

```
status               provisional
base_results_commit  d8fec245671e
tool_source_commit   9d97fe75
footprint digest     a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a  (안 바뀜)
본표 바뀐 셀         2,080   (계열 A)
사이드카 셀          1,262   (계열 C, expressions.yaml)
V9                   14 실행 전부 pass / 2 n/a / 미평가 0
fixture              12/12
음성 대조            6/6 발화   <- 새로 넣음 (develop/plus_at_negctl.py)
입력 pin             15 개
남은 차단            19 건 -- 전부 review_status (expected_footprint 가 늘었다)
```

---

## 1. bundle symbol 계약이 `expressions.yaml` 과 충돌한다 → namespace 로 분리

`architecture_symbols: []` 하나로 적어 두면 소비자가 사이드카의 `R_res`, `L_layers`,
`l`, `ceil` 을 미선언으로 보고 거부합니다. 지적이 정확했습니다.

```json
"symbol_namespaces": {
  "table_added_symbols": {"n_chunk": "trace_artifact"},
  "sidecar_architecture_symbols": ["L_layers", "R_res"],
  "sidecar_row_variables": ["l"],
  "sidecar_allowed_functions": ["ceil"],
  "base_table_symbols": "inherited_from_authority"
}
```

`reject_unknown_symbol` 을 namespace 별 규칙으로 다시 적었습니다.

> 본표의 토큰은 트레이서 심볼표 + `table_added_symbols` 로 해석돼야 하고, 사이드카의 식은
> `sidecar_architecture_symbols` + `sidecar_row_variables` + `sidecar_allowed_functions`
> 로 해석돼야 한다. 모든 심볼이 `symbols.yaml` 에 있어야 한다고 보면 정상적인 사이드카
> 식도 거부된다.

`base_table_symbols` 를 열거하지 않고 `inherited_from_authority` 로 적은 이유는, 그 목록의
authority 가 `models/<m>/full/symbol_table.json` 이고 +@ 가 사본을 두면 두 판이 갈라지기
때문입니다. 이 선택이 맞는지 Q2 로 묻습니다.

## 2. `expected_footprint.review_status` 가 release blocker 에 없다 → 넣었습니다

`develop/plus_at_apply.py` 의 `release_blockers()` 가 이제 봅니다. 차단이 18 → **19** 건.

```
expected_footprint: review_status 'proposed' (accepted 아님)
```

## 3. crosswalk 이 없는 raw op/slot 을 가리키면 게이트가 건너뛰고 PASS 한다 → 닫았습니다

### 3-a `g_reshape_derivation`

`raw.get(roid)` 가 없으면 `continue` 하던 것을 지웠습니다. 없으면 **그 자리를 못 본 것**이고
못 본 것을 PASS 로 세지 않습니다.

```
crosswalk 이 가리킨 raw op N 개가 원장/구체 사이드카에 없다  ->  not_evaluated  ->  not_run
```

### 3-b `g_axis_class_consistency`

`uf.find()` 는 모르는 키를 singleton 으로 넣어 버립니다. 그래서 그 class 는 member 가
없고 `labels` 가 비어 조용히 건너뛰어졌습니다. 이제 `uf.find` 를 부르기 **전에** raw op
존재와 축 범위를 확인합니다.

```
crosswalk 이 가리킨 raw slot N 개가 원장에 없다  ->  not_evaluated
  예: (1000000000, 'op 없음') / (225, 'i[0]ax99 범위 밖')
```

### 3-c 두 게이트 공통 — crosswalk **커버리지**

이게 더 중요하다고 판단해 넣었습니다. 바뀐 셀의 키만 맞춰 보면 *"그 2,080 셀만 실린 낡은
crosswalk"* 도 통과합니다. 그래서 집합을 양방향으로 맞춥니다.

```
crosswalk 키 집합 == 발행본 셀 집합
prefill  crosswalk 126,822  발행본 126,822  차이 0 / 0
decode   crosswalk  15,510  발행본  15,510  차이 0 / 0
```

### 3-d `n_ck == len(roots)` 는 넣었지만 **항등식입니다**

요구대로 넣었습니다. 그런데 발화시키려고 음성 대조를 두 번 짰고 두 번 다 실패했습니다.
이유는 `rev` 를 **같은 crosswalk** 에서 만들기 때문입니다 -- 자리를 주입하는 순간 그 자리는
발행본 키로 되돌아오고, 그 키는 항상 `pub` 에 있습니다. 즉 이 등식은 현재 구조에서 깨질 수
없습니다.

그래서 코드 주석에 그렇게 적었습니다. **이 줄은 보험이고, "class 를 조용히 건너뛰지
않았다" 의 실제 증거는 3-b 의 raw slot 존재 검사와 3-c 의 커버리지 검사입니다.** 통과
개수로 세지 않는 편이 맞다고 봅니다. Q3 로 묻습니다.

## 4. source evidence 가 provenance 의 revision 에 고정되지 않았다 → 고정했습니다

**지적이 옳았고, 실제로 틀린 snapshot 을 집고 있었습니다.**

```
provenance.revision_resolved  f831ab66814297da540d832a5235f8e904f29d06
glob 이 집던 것              9f62e4e9fffbd0a83ddd60e1c209d828994b3569   <- 틀렸다
```

`models--*/snapshots/*` glob 을 지우고 `models--<slug>/snapshots/<revision_resolved>/`
를 후보 0 번에 둡니다. 고정 snapshot 에 없으면 다른 snapshot 으로 넘어가지 않고
`SystemExit` 합니다.

다만 정직하게 적습니다 -- **sha256 은 안 바뀌었습니다.** 세 snapshot 의
`modeling_kimi_linear.py` 가 같은 blob 이라(`9e3564c70ac21854…`, 세 개 전부 동일) 근거
해시는 원래 맞았고 `resolved` 경로만 틀렸습니다. 위험은 실재했고 이번엔 실현되지
않았습니다.

## 5. `port_coverage` 를 이름대로 검사하게 했습니다

지적: *"published cell-key 불변과 raw ports coverage 는 다른 증거다. 이름대로
`port_coverage` 를 승인하려면 raw/raw 집합 일치, 중복, schema 를 직접 검사해야 한다."*

파일 해시 + cell 키 불변만 보던 것을 바꿨습니다. 증거를 둘로 나눠 적습니다.

```
raw 증거     raw op-id 집합 == ports op-id 집합 / ports 중복 0 / schema 단일
             prefill  631,705/631,705  schema v2  sha baa0d80a…
             decode    33,199/33,199   schema v2  sha …
발행본 증거  canonical cell 키 불변 (derived view 가 발행 구조를 안 바꿨다)
```

`decision` 을 `inherited_unchanged` → **`rerun`** 으로 올렸습니다. 이제 실제로 돕니다.

## 6. 음성 대조 — `develop/plus_at_negctl.py` (새로 넣음)

*"fail-closed 로 고쳤다"* 는 말만으로는 지난 라운드와 같습니다. 그래서 실제 모델 데이터
위에서 crosswalk 을 일부러 망가뜨려 각 경로가 발화하는지 봅니다.

```
OK   reshape / 없는 raw op          not_evaluated
OK   axis    / 없는 raw op          not_evaluated
OK   axis    / 범위 밖 축           not_evaluated
OK   reshape / crosswalk 커버리지   not_evaluated
OK   axis    / crosswalk 커버리지   not_evaluated
OK   port    / raw!=ports 집합      FAIL
6/6 발화
```

첫 판은 **1/4 만 발화**했습니다. 원인은 게이트가 아니라 제 probe 였습니다 -- bogus site
를 `changed` 에 없는 셀에 넣어서 읽히지 않았습니다. 그것을 고친 뒤 4/4, 커버리지 대조를
더해 6/6 입니다. 그리고 3-d 의 항등식은 여기서도 발화하지 않았습니다.

이 스크립트를 MANIFEST `tool.negctl` 로 해시 pin 했습니다.

## V9 전체 (14 실행 / 2 n/a / 미평가 0)

```
pass  rerun  axis_class_consistency       prefill: 건드린 class 69 == 검사한 class 69, 충돌 0
pass  rerun  reshape_derivation           prefill raw op 4,485, 이견 0 -> 0 / decode 0
pass  rerun  port_coverage                raw 631,705+33,199 집합 일치 / 발행본 cell 키 불변
pass  rerun  sidecar_phase_consistency    phase 별 631, 식 분포 동일
pass  rerun  schema_shape_rank_token_type / symbol_declared / expression_no_cycle
pass  rerun  substitution_nonneg_integer / zero_axis_only_initial_residual
pass  rerun  batch_seq_head_axis_consistency / head_scope_exclusive
pass  rerun  layers_repeat_consistency / row_metadata_preserved / op_id_dag
n/a          moe_quotient_remainder_consistency   (MoE 철회)
n/a          prefill_decode_structure             (본표 판정이 prefill 전용)
```

## 남은 차단 19 건 — 전부 `review_status`

```
k3-kda-nchunk: status 'proposed'
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
expected_footprint: review_status 'proposed'      <- 이번에 추가
V9 게이트 16 종: review_status 'proposed'
```

## 물어보는 것

### Q1. 공개 승인하십니까

승인하시면 다음을 올립니다.

```
overlay 의 k3-kda-nchunk status              proposed -> accepted
verification.semantic_evidence_verified      false -> true
expected_footprint.review_status             proposed -> accepted
develop/plus_at/v9_review.yaml 의 16 게이트   proposed -> accepted
MANIFEST status                              provisional -> released
develop/reviews/ 에 이 판정 원문 보존 (covers [R3d])
results-plus-at 브랜치 생성 -- base results commit d8fec245 기록
```

### Q2. `base_table_symbols: inherited_from_authority` 가 맞습니까

본표의 `B`, `T`, `d_model`, `n_h_kda`, `d_chunk` … 는 트레이서가 낸 것이고 authority 는
`models/<m>/full/symbol_table.json` 입니다. +@ 가 사본을 두면 두 판이 갈라집니다. 그래서
열거하지 않고 포인터만 뒀습니다. 소비자에게 전체 목록을 주려면 그 파일도 bundle 에 넣어야
하는데, 그러면 +@ 가 authority 파일을 복제하게 됩니다. 어느 쪽이 맞습니까?

### Q3. 항등식 게이트(3-d)를 어떻게 적어야 합니까

지금은 `pass` 로 나오지만 발화 불가능합니다. 셋 중 어느 쪽이 맞습니까?

```
(a) 지금처럼 두고 주석·문서에 "항등식" 이라고만 적는다
(b) 게이트에서 빼고 커버리지 검사만 남긴다
(c) 결과를 `vacuous` 같은 별도 값으로 적어 pass 개수에 안 들어가게 한다
```

### Q4. 놓친 것

특히 이번에 고친 세 경로 말고 **다른 fail-open** 이 남아 있는지 봐 주십시오. 그리고
`n_ck == len(roots)` 처럼 **구조적으로 발화 불가능한 검사**가 다른 게이트에도 있는지
(0 == 0 을 통과로 세는 자리) 짚어 주시면 좋겠습니다.

## 읽을 곳

```
develop/plus_at_v9.py       g_reshape_derivation() / g_axis_class_consistency()
                            / g_port_coverage_inherited() -- 커버리지 검사와 주석
develop/plus_at_negctl.py   음성 대조 6 종
develop/plus_at_apply.py    release_blockers() / bundle_contract symbol_namespaces
                            / source evidence 의 revision 고정
models/moonshotai__Kimi-K3/plus_at/MANIFEST.json   source_pin, symbol_namespaces, v9
models/moonshotai__Kimi-K3/plus_at/DIFF.md
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml
develop/plus_at/expected-moonshotai__Kimi-K3.jsonl   2,080 줄
develop/reviews/            R1 / R2(covers R2,R3) / R3 / R3b / R3c
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
