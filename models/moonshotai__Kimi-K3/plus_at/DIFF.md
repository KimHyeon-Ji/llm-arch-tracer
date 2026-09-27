# DIFF — `moonshotai__Kimi-K3` derived view (+@)

**이 문서는 사람이 읽는 요약이다.** 전수 목록은 `actual_footprint.jsonl` (3342 줄)이고, 그 digest 가 `expected_footprint.jsonl` 과 같아야 한다.

```
status                provisional
base_results_commit   d8fec245671e7b77955a471af210e31dc4ca13fe
tool_source_commit    ccc3660f70107bc39661b2f12d305ab0fdf64a40
expected footprint    997726a9476fa6d21446358072d000c4…
바뀐 셀               3342
```

원본 `{prefill,decode}.{csv,jsonl}` 은 **건드리지 않았다.** 이 디렉터리는 원본 + `develop/plus_at/overlay-*.yaml` 로 언제든 재생성된다. 손으로 고치지 말 것.

## 무엇을 무엇으로 바꿨나

| sub_id | before | after | 셀 | 자리 |
|---|---|---|---:|---|
| `k3-kda-nchunk` | `5` | `n_chunk` | 2080 | self_attn |
| `k3-residual-accum` | `2` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `2` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `2` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `2` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `3` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `3` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `3` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `3` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `4` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `4` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `4` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `4` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `5` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `5` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `5` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `5` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `6` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `6` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `6` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `6` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `7` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `7` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `7` | `ceil(l/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `7` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `8` | `ceil((l+1)/R_res)` | 8 | 10 개 (module, op) |
| `k3-residual-accum` | `8` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `8` | `ceil(L_layers/R_res)` | 2 | 10 개 (module, op) |
| `k3-residual-accum` | `8` | `ceil(l/R_res)` | 4 | 10 개 (module, op) |
| `k3-residual-accum` | `8` | `ceil(l/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `9` | `ceil((l+1)/R_res)+1` | 72 | 10 개 (module, op) |
| `k3-residual-accum` | `9` | `ceil(L_layers/R_res)+1` | 24 | 10 개 (module, op) |
| `k3-residual-accum` | `9` | `ceil(l/R_res)+1` | 48 | 10 개 (module, op) |

## phase / field 별

```
k3-kda-nchunk     prefill  input_shape     1040
k3-kda-nchunk     prefill  output_shape    1040
k3-residual-accum decode   input_shape      384
k3-residual-accum decode   output_shape     247
k3-residual-accum prefill  input_shape      384
k3-residual-accum prefill  output_shape     247
합계                                         3342
```

## 심볼 (symbols.yaml)

`kind` 가 **아키텍처**와 **트레이스 제어**를 가른다. 트레이스 제어 심볼은 모델 속성이 아니라 이 트레이스가 만든 값이다.

| 심볼 | kind | 식 | 이 트레이스의 값 |
|---|---|---|---|
| `n_chunk` | architecture | `T / d_chunk` | 5 |
| `N_route` | architecture | `prefill B*T*k / decode B*k` | prefill 15360 / decode 48 |
| `l` | row_field | `prefill None / decode None` | prefill None / decode None |
| `R_res` | architecture | `config.text_config.attn_res_block_size` | 12 |
| `L_layers` | architecture | `config.text_config.num_hidden_layers` | 93 |
| `C_trace` | trace_control | `trace.expert_cap` | 4 |
| `n_trace_regular` | trace_control | `floor(N_route / C_trace)` | prefill 3840 / decode 12 |
| `n_trace_last` | trace_control | `N_route - (C_trace - 1) * floor(N_route / C_trace)` | prefill 3840 / decode 12 |

## 바꾸지 않고 남긴 맨 정수 (숨기지 않는다)

```
decode   0:1  12:2392     2393 자리
prefill  0:1  3840:2392     2393 자리
```

전부 attention-residual 누적 경로다(계열 C). 배치 크기만 스윕하면 이 값은 변하지 않는다. `d_chunk` 나 층 배치를 스윕하려면 이 자리는 아직 맞지 않는다.

## V9 게이트

| 결과 | 처분 | 게이트 | 내용 |
|---|---|---|---|
| not_run | not_evaluated | `axis_class_consistency` | 이 작업이 축 의미를 바꾸므로 직접 관련된다. 등가류는 원시 원장 + 포트 사이드카 + crosswalk 위에 서는데 그 재실행을 아직 배선하지 않았다. 'not |
| pass | rerun | `batch_seq_head_axis_consistency` | B·T·head 축 자리 불변 |
| pass | rerun | `expression_no_cycle` | 심볼 8개 위상 정렬 OK |
| pass | rerun | `head_scope_exclusive` | head 심볼 배타성 OK |
| pass | rerun | `layers_repeat_consistency` | 전 행 일치 |
| n/a | not_applicable | `moe_quotient_remainder_consistency` | overlay 가 MoE 축을 건드리지 않는다 (R1 판정으로 철회) -- 해당 없음 |
| pass | rerun | `op_id_dag` | 유일·존재·비순환 OK |
| pass | inherited_unchanged | `port_coverage` | derived view 는 포트를 건드리지 않는다 (사이드카 불변 + cell 키 불변): prefill.ports.jsonl sha baa0d80aa1eed79 |
| pass | rerun | `prefill_decode_structure` | 공통 자리 22 건 전부 같은 심볼 |
| pass | rerun | `reshape_derivation` | 원본 이견 0 -> 파생 0 |
| pass | rerun | `row_metadata_preserved` | 메타데이터 불변 |
| pass | rerun | `schema_shape_rank_token_type` | 스키마 보존 |
| pass | rerun | `substitution_nonneg_integer` | 전부 정수·비음수·복원 일치 (식 토큰은 행 단위) |
| pass | rerun | `symbol_declared` | 선언 8개 + 허용 함수 ['ceil'] 로 전부 해석됨 |
| pass | rerun | `zero_axis_only_initial_residual` | `0` 축 2 자리 전부 선언된 자리 |

## 소비자 계약

이 표는 symbols.yaml 과 **분리 불가**하다. csv/jsonl 만 떼어 배포하면 trace_control 심볼(C_trace, n_trace_regular, n_trace_last)이 아키텍처 심볼로 오독된다. 소비자는 symbols.yaml 에 없는 심볼을 만나면 거부해야 한다. caveat 열은 MoE 행에 그대로 남아 있다 -- 총 expert projection FLOPs 는 보존되나 전문가별 분포·active expert 수·weight traffic·cache·latency 는 보존되지 않는다.

## 왜 provisional 인가

```
k3-kda-nchunk: status 'proposed' (accepted 아님)
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
k3-residual-accum: status 'proposed' (accepted 아님)
k3-residual-accum: semantic_evidence_verified 가 참이 아니다
검토 기록 없음: develop/reviews/R2-*
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
V9 axis_class_consistency: 평가되지 않았다 (not_evaluated)
```
