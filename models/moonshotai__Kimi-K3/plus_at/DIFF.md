# DIFF — `moonshotai__Kimi-K3` derived view (+@)

**이 문서는 사람이 읽는 요약이다.** 전수 목록은 `actual_footprint.jsonl` (2080 줄)이고, 그 digest 가 `expected_footprint.jsonl` 과 같아야 한다.

```
status                provisional
base_results_commit   d8fec245671e7b77955a471af210e31dc4ca13fe
tool_source_commit    43371702c07fa988143a00b1843b8e7f14081c2f
expected footprint    a13ed393e02581378837e5f9be916461…
바뀐 셀               2080
```

원본 `{prefill,decode}.{csv,jsonl}` 은 **건드리지 않았다.** 이 디렉터리는 원본 + `develop/plus_at/overlay-*.yaml` 로 언제든 재생성된다. 손으로 고치지 말 것.

## 무엇을 무엇으로 바꿨나

| sub_id | before | after | 셀 | 자리 |
|---|---|---|---:|---|
| `k3-kda-nchunk` | `5` | `n_chunk` | 2080 | self_attn |

## phase / field 별

```
k3-kda-nchunk     prefill  input_shape     1040
k3-kda-nchunk     prefill  output_shape    1040
합계                                         2080
```

## 심볼 (symbols.yaml)

`kind` 가 **아키텍처**와 **트레이스 제어**를 가른다. 트레이스 제어 심볼은 모델 속성이 아니라 이 트레이스가 만든 값이다.

| 심볼 | kind | 식 | 이 트레이스의 값 |
|---|---|---|---|
| `n_chunk` | trace_artifact | `T / d_chunk` | 5 |

## 바꾸지 않고 남긴 맨 정수 (숨기지 않는다)

```
decode   0:1  2:80  3:80  4:80  5:80  6:80  7:80  8:79  9:72  12:2392     3024 자리
prefill  0:1  2:80  3:80  4:80  5:80  6:80  7:80  8:79  9:72  3840:2392     3024 자리
```

| 값 | 자리 | 왜 리터럴인가 |
|---|---:|---|
| `12` | 2392 | MoE 라우팅 토큰 (decode). 같은 이유 |
| `3840` | 2392 | MoE 라우팅 토큰 (prefill). shim 이 전문가 4 개로 균등분할한 **대체값**이고 실제 per-expert 축은 라우팅이 정하므로 결정 불가 -- overlay 의 moe_aggregate 참조 |
| `2` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `3` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `4` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `5` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `6` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `7` | 160 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `8` | 158 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `9` | 144 | residual 누적 폭. **사이드카에 식이 있다** (expressions.yaml -- stage 와 ceil(l/R_res) 계열 식). 본표는 리터럴을 유지한다 |
| `0` | 2 | 초기 빈 residual 버퍼. 리터럴이 맞다 |

배치 크기만 스윕하면 이 값들은 변하지 않는다. `d_chunk` 나 층 배치를 스윕하려면 이 자리는 아직 맞지 않는다.

## V9 게이트

| 결과 | 처분 | 게이트 | 내용 |
|---|---|---|---|
| pass | rerun | `axis_class_consistency` | prefill: 건드린 class 69 == 검사한 class 69, 이름 충돌 0 |
| pass | rerun | `base_symbol_coverage` | authority 23 + table_added ['n_chunk'] 로 전부 해석됨.  prefill: 식별자 21  decode: 식별자 19 |
| pass | rerun | `batch_seq_head_axis_consistency` | B·T·head 축 자리 불변 |
| pass | rerun | `expression_no_cycle` | 심볼 1개 위상 정렬 OK |
| pass | rerun | `head_scope_exclusive` | head 심볼 배타성 OK |
| pass | rerun | `layers_repeat_consistency` | 전 행 일치 |
| n/a | not_applicable | `moe_quotient_remainder_consistency` | overlay 가 MoE 축을 건드리지 않는다 (R1 판정으로 철회) -- 해당 없음 |
| pass | rerun | `op_id_dag` | 유일·존재·비순환 OK |
| pass | rerun | `port_coverage` | raw 포트 커버리지 [prefill: 631705/631705 schema v2 sha baa0d80aa1ee; decode: 33199/33199 schema |
| n/a | not_applicable | `prefill_decode_structure` | 활성 판정이 전부 단일 phase 다 -- 해당 없음 |
| pass | rerun | `reshape_derivation` | decode: 건드린 raw op 0, 이견 0 -> 0  prefill: 건드린 raw op 4485, 이견 0 -> 0 |
| pass | rerun | `row_metadata_preserved` | 메타데이터 불변 |
| pass | rerun | `schema_shape_rank_token_type` | 스키마 보존 |
| pass | rerun | `sidecar_expression_integrity` | 레코드 1262 개: formula 전부 registry 일치, 식별자 ['L_layers', 'R_res', 'l'] + ['ceil'] 로 전부 해석, 식값  |
| pass | rerun | `sidecar_phase_consistency` | phase 별 631 레코드, 식 분포 동일 {'c_post': 288, 'c_pre': 276, 'b_after': 28, 'b_in': 26, 'b_final |
| pass | rerun | `substitution_nonneg_integer` | 전부 정수·비음수·복원 일치 (식 토큰은 행 단위) |
| pass | rerun | `symbol_declared` | 선언 1개 + 허용 함수 ['ceil'] 로 전부 해석됨 |
| pass | rerun | `zero_axis_only_initial_residual` | `0` 축 2 자리 == 선언 2 자리 (양방향 일치) |

## 소비자 계약

| 항목 | 내용 |
|---|---|
| `canonical_cell_rule` | op_id 는 **발행본 표의 번호**다(0.. 로 재번호된 것). 원시 원장 op_id 와 다른 번호 공간이므로 원시와 잇는 데는 crosswalk 이 필요하다. |
| `caveat_stays` | caveat 열은 MoE 행에 그대로 남아 있다. 총 expert projection FLOPs 는 보존되나 전문가별 분포·active expert 수·weight traffic·cache·latency 는 보존되지 않는다. |
| `inseparable` | 표(csv/jsonl)는 symbols.yaml 과 **분리 불가**하다. 표만 떼어 배포하면 trace_artifact 심볼이 아키텍처 심볼로 오독된다. |
| `one_bundle` | actual_footprint.jsonl, decode.csv, decode.jsonl, expected_footprint.jsonl, expressions.yaml, prefill.csv, prefill.jsonl, symbols.yaml |
| `phase_consistency` | prefill 과 decode 의 사이드카는 op_id 만 다르고 나머지 필드의 multiset 이 같아야 한다 -- V9 의 sidecar_phase_consistency 가 검사한다 |
| `reject_unknown_symbol` | **namespace 별로** 적용한다. 본표의 토큰은 트레이서 심볼표 + table_added_symbols 로 해석돼야 하고, 사이드카의 식은 sidecar_architecture_symbols + sidecar_row_variables + sidecar_allowed_functions 로 해석돼야 한다. 모든 심볼이 symbols.yaml 에 있어야 한다고 보면 정상적인 사이드카 식도 거부된다. |
| `sidecar` | expressions.yaml 도 같은 bundle 이다. 본표에 리터럴로 남은 residual 누적 폭(2..9)의 stage 와 식이 거기 있다. 사이드카가 없는 소비자는 숫자 표만 쓸 수 있고 residual recurrence 의미는 복원할 수 없다. |
| `sidecar_expression_integrity` | 레코드의 formula 는 이 계약의 registry(expressions.yaml 의 formulas)에 있어야 하고, expr 는 registry 의 식과 같아야 하고, 식값은 value 와 같아야 한다 -- V9 의 sidecar_expression_integrity 가 검사한다 |
| `sidecar_join_key` | phase, op_id, field, shape_index, axis |
| `symbol_namespaces` | {'base_table_symbols': {'count': 23, 'json_pointer': '/symbol_table', 'mode': 'external_reference', 'path': 'models/moonshotai__Kimi-K3/full/provenance.json', 'sha256': '7369827e79fec2c20f7e12046ecbd01b0209526c4f1fe9d9bce12ed04357aab3', 'verified_by': 'V9 base_symbol_coverage'}, 'sidecar_allowed_functions': ['ceil'], 'sidecar_architecture_symbols': ['L_layers', 'R_res'], 'sidecar_row_variables': ['l'], 'table_added_symbols': {'n_chunk': 'trace_artifact'}} |

## 왜 provisional 인가

```
k3-kda-nchunk: status 'proposed' (accepted 아님)
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
expected_footprint: review_status 'proposed' (accepted 아님)
V9 schema_shape_rank_token_type: review_status 'proposed' (accepted 아님)
V9 symbol_declared: review_status 'proposed' (accepted 아님)
V9 expression_no_cycle: review_status 'proposed' (accepted 아님)
V9 substitution_nonneg_integer: review_status 'proposed' (accepted 아님)
V9 zero_axis_only_initial_residual: review_status 'proposed' (accepted 아님)
V9 batch_seq_head_axis_consistency: review_status 'proposed' (accepted 아님)
V9 head_scope_exclusive: review_status 'proposed' (accepted 아님)
V9 moe_quotient_remainder_consistency: review_status 'proposed' (accepted 아님)
V9 prefill_decode_structure: review_status 'proposed' (accepted 아님)
V9 layers_repeat_consistency: review_status 'proposed' (accepted 아님)
V9 row_metadata_preserved: review_status 'proposed' (accepted 아님)
V9 op_id_dag: review_status 'proposed' (accepted 아님)
V9 reshape_derivation: review_status 'proposed' (accepted 아님)
V9 port_coverage: review_status 'proposed' (accepted 아님)
V9 axis_class_consistency: review_status 'proposed' (accepted 아님)
V9 sidecar_phase_consistency: review_status 'proposed' (accepted 아님)
V9 sidecar_expression_integrity: review_status 'proposed' (accepted 아님)
V9 base_symbol_coverage: review_status 'proposed' (accepted 아님)
```
