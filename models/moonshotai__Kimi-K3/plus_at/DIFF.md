# DIFF — `moonshotai__Kimi-K3` derived view (+@)

**이 문서는 사람이 읽는 요약이다.** 전수 목록은 `actual_footprint.jsonl` (2080 줄)이고, 그 digest 가 `expected_footprint.jsonl` 과 같아야 한다.

```
status                provisional
base_results_commit   d8fec245671e7b77955a471af210e31dc4ca13fe
tool_source_commit    70050971ac2c9aff662401442b253659b60df8b8
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

전부 attention-residual 누적 경로다(계열 C). 배치 크기만 스윕하면 이 값은 변하지 않는다. `d_chunk` 나 층 배치를 스윕하려면 이 자리는 아직 맞지 않는다.

## V9 게이트

| 결과 | 처분 | 게이트 | 내용 |
|---|---|---|---|
| pass | rerun | `axis_class_consistency` | prefill: 건드린 class 69, 발행본 member 가 있는 class 69, 이름 충돌 0 |
| pass | rerun | `batch_seq_head_axis_consistency` | B·T·head 축 자리 불변 |
| pass | rerun | `expression_no_cycle` | 심볼 1개 위상 정렬 OK |
| pass | rerun | `head_scope_exclusive` | head 심볼 배타성 OK |
| pass | rerun | `layers_repeat_consistency` | 전 행 일치 |
| n/a | not_applicable | `moe_quotient_remainder_consistency` | overlay 가 MoE 축을 건드리지 않는다 (R1 판정으로 철회) -- 해당 없음 |
| pass | rerun | `op_id_dag` | 유일·존재·비순환 OK |
| pass | inherited_unchanged | `port_coverage` | derived view 는 포트를 건드리지 않는다 (사이드카 불변 + cell 키 불변): prefill.ports.jsonl sha baa0d80aa1eed79 |
| n/a | not_applicable | `prefill_decode_structure` | 활성 판정이 전부 단일 phase 다 -- 해당 없음 |
| pass | rerun | `reshape_derivation` | 원본 이견 0 -> 파생 0 |
| pass | rerun | `row_metadata_preserved` | 메타데이터 불변 |
| pass | rerun | `schema_shape_rank_token_type` | 스키마 보존 |
| pass | rerun | `substitution_nonneg_integer` | 전부 정수·비음수·복원 일치 (식 토큰은 행 단위) |
| pass | rerun | `symbol_declared` | 선언 1개 + 허용 함수 ['ceil'] 로 전부 해석됨 |
| pass | rerun | `zero_axis_only_initial_residual` | `0` 축 2 자리 전부 선언된 자리 |

## 소비자 계약

이 표는 symbols.yaml 과 **분리 불가**하다. csv/jsonl 만 떼어 배포하면 trace_control 심볼(C_trace, n_trace_regular, n_trace_last)이 아키텍처 심볼로 오독된다. 소비자는 symbols.yaml 에 없는 심볼을 만나면 거부해야 한다. caveat 열은 MoE 행에 그대로 남아 있다 -- 총 expert projection FLOPs 는 보존되나 전문가별 분포·active expert 수·weight traffic·cache·latency 는 보존되지 않는다.

## 왜 provisional 인가

```
k3-kda-nchunk: status 'proposed' (accepted 아님)
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
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
```
