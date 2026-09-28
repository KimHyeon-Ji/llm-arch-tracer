# R3 재승인 (짧은 것) — 조건 8 개를 고쳤습니다

요청일 2026-09-28. 선행: R1(blind, 반영 완료), R3 1 차(차단 5 종, 반영 완료),
R2/R3 합본(조건 8 종 — 이 문서가 그 처리 결과입니다).

> 지난 판정: "위 조건을 고친 뒤에는 R2 를 처음부터 다시 할 필요 없이, 변경된 classifier
> fixture 와 footprint 를 재실행한 짧은 R3 재승인이면 충분합니다."

```
status               provisional
base_results_commit  d8fec245671e
tool_source_commit   70050971ac2c   (지금은 9907d51e)
footprint digest     a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a
본표 바뀐 셀         2,080   (계열 A 만)
사이드카 셀          1,262   (계열 C -- 본표는 리터럴 유지)
```

---

## 조건별 처리

### 1. A 의 kind -> `trace_artifact`

지적을 받아들였습니다. `d_chunk=64` 는 Kimi-K3 config 값이 아니라 KDA 구현의 kernel
default(`src/summarize.py:155`)이고 shim 은 Triton 대신 torch reference 를 돌립니다
(`src/kda_shim.py:550`).

```yaml
n_chunk:
  kind: trace_artifact
  origin: kda_torch_reference
  expr: "T / d_chunk"
  value: 5
  validity_domain: ["T % d_chunk == 0"]
```

치환과 식은 유지했습니다 -- 이 트레이스를 정확히 기술하므로.

### 2. C 의 stage 판별 -> **파라미터 lineage** + cardinality 강제

지적하신 기준을 그대로 채택했습니다. 그룹의 `elementwise_mul` 이 소비하는 norm 가중치가
stage 를 직접 말합니다.

```
self_attention_res_norm -> mix_pre     실측 prefill/decode 각 23
mlp_res_norm            -> mix_post                        각 24
output_attn_res_norm    -> mix_final                       각  1
그 밖의 residual concat -> boundary_append
```

지적하신 수(23 / 24 / 1)와 같고 옛 op-순서 판정과도 일치합니다. pre 가 24 가 아니라 23 인
것은 층 0 에 pre-mix 가 없기 때문입니다.

**cardinality 를 강제합니다. 추측하지 않고 실패합니다.**

```
층 0        pre 0, post 1
그 밖       pre 1, post 1
final       정확히 1
norm 가중치가 없거나 둘 이상이면 ValueError
```

옛 op-순서 판별의 깨질 방식들(중간 view/cast, append 의 fan-out, mix 셋 이상, op 순서
의존)은 모두 사라졌습니다 -- 더 이상 순서를 보지 않습니다.

`strict` 플래그로 나눴습니다: 실제 모델은 켜고, fixture 의 단일 사례(전체가 아닌 입력)는
끕니다. **cardinality 가 실제로 발화하는지는 fixture 의 전용 사례가 시험합니다**
(`case_residual_cardinality_fires` -- strict 를 켜고 부분 입력을 주면 실패해야 한다).

### 3. `l` 을 사이드카로

권고 1 번을 택했습니다. **본표의 residual 축은 리터럴 2..9 로 그대로 둡니다.** 행 필드를
참조하는 식을 축 토큰에 넣지 않습니다.

`plus_at/expressions.yaml` (1,262 레코드):

```yaml
records:
  - {phase, op_id, field, shape_index, axis, value, stage, formula, expr,
     layer_idx, layers, module_path, op_type}
symbols: {R_res(12), L_layers(93), l 의 뜻}
formulas: {b_in, b_after, c_pre, c_post, b_final, c_final}
limitations: {sweep_of_R_res, batch_sweep}
```

말씀대로 R sweep 을 지원하지 않으므로 잃은 sweep 기능은 없습니다.

### 4. 등가류 이름 일관성 -> **실제로 돌렸습니다. 다만 제 첫 구현이 틀렸습니다**

이게 이번 라운드에서 제가 찾은 가장 큰 오류입니다.

```
발행본 op_id   0 .. 15,197   (재번호. 접힌 행의 대표)
원시 원장      0 .. 631,704
-> **다른 번호 공간이다.**
```

첫 구현은 발행본 셀의 op_id 를 원시 원장 기반 등가류에 그대로 넣었습니다. 그래서 관계없는
자리를 비교해 **244 건의 거짓 충돌**을 냈습니다 -- 발행본 op92 는 `exp`(KDA) 인데 원시
op92 는 conv1d 의 `slice` 입니다.

crosswalk 가 그 대응을 줍니다(`work/crosswalk/<model>.<phase>.jsonl.gz`,
발행본 셀 -> `raw_sites`). 그걸 경유해 다시 돌렸습니다:

```
prefill  건드린 class 69,  발행본 member 가 있는 class 69,  **이름 충돌 0**
```

그리고 **crosswalk 가 낡으면 거짓 결과 대신 미평가로 빠집니다** -- 바꾼 셀마다 crosswalk 의
`expr` 이 발행본의 원래 라벨(`before`)과 같은지 먼저 확인하고, 하나라도 다르면 `n/a` 로
보고합니다.

### 5. 근거 SHA-256 -> 못 찾으면 실패

null 을 적고 넘어가지 않습니다. 탐색 경로를 HF 캐시와 site-packages 재귀까지 넓혔고,
사이드카 근거도 같은 루프를 탑니다.

```
k3-kda-nchunk        fla/ops/kda/naive.py:108-109                  60a32285d4b6
k3-kda-nchunk        src/build_table.py:1250                       83ca11e915b0
k3-kda-nchunk        models/…/prefill.csv:3168 cells               a77a5da56e15
k3-residual-sidecar  modeling_kimi_linear.py:1188-1192             9e3564c70ac2
k3-residual-sidecar  modeling_kimi_linear.py:995-998               9e3564c70ac2
k3-residual-sidecar  modeling_kimi_linear.py:1075-1087             9e3564c70ac2
k3-residual-sidecar  develop/reviews/R1-…-blind.md:축 3            f33d22b59186
```

`resolved: null` / `sha256: null` 은 이제 없습니다.

### 6. DIFF 의 잔여 리터럴 설명

하드코딩 문구를 지우고 값별 표로 바꿨습니다.

```
| 값     | 자리 | 왜 리터럴인가 |
| 3840   | 2392 | MoE 라우팅 토큰 (shim 의 균등분할 대체값 -- moe_aggregate 를 보라) |
| 12     | 2392 | MoE 라우팅 토큰, decode |
| 2..9   |  ... | residual 누적 -- 사이드카에 식이 있다 |
| 0      |    2 | 초기 빈 residual 버퍼 (리터럴이 맞다) |
```

### 7. 철회된 MoE 심볼 제거

`C_trace` / `n_trace_regular` / `n_trace_last` 를 활성 `symbols.yaml` 에서 뺐습니다
(`withdrawn_symbols` 주석으로 이동). 지금 활성 심볼은 `n_chunk` 하나입니다 --
표에서 참조되는 것과 정확히 일치합니다. `moe_quotient_remainder_consistency` 를 N/A 로 둔
것과의 어긋남도 사라졌습니다.

### 8. V9 와 footprint 재생성

```
pass  rerun                axis_class_consistency       건드린 class 69, 이름 충돌 0
pass  rerun                schema_shape_rank_token_type
pass  rerun                symbol_declared              선언 1 개 + 허용 함수 ceil
pass  rerun                expression_no_cycle
pass  rerun                substitution_nonneg_integer
pass  rerun                zero_axis_only_initial_residual
pass  rerun                batch_seq_head_axis_consistency
pass  rerun                head_scope_exclusive
pass  rerun                layers_repeat_consistency
pass  rerun                row_metadata_preserved
pass  rerun                op_id_dag
pass  rerun                reshape_derivation           원본 이견 0 -> 파생 0
pass  inherited_unchanged  port_coverage                사이드카 해시 + cell 키 불변
n/a   not_applicable       moe_quotient_remainder_consistency   (MoE 철회로 대상 없음)
n/a   not_applicable       prefill_decode_structure             (활성 판정이 단일 phase)
```

**미평가(not_run) 0 건입니다.** 지난 라운드에는 `axis_class_consistency` 하나가
미평가였습니다.

`fixture 11/11`. V1~V8 전부 통과. 멱등·결정론 확인.

## 남은 차단 18 건 — **전부 검토 상태입니다**

```
k3-kda-nchunk: status 'proposed'
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
검토 기록 없음: develop/reviews/R2-*
V9 게이트 15 종: review_status 'proposed'
```

기계적 조건은 닫혔습니다. 남은 것은 승인 기록뿐입니다.

## 물어보는 것

### Q1. publish 를 승인하십니까

승인하시면 다음을 올립니다.

```
overlay 의 k3-kda-nchunk status         proposed -> accepted
verification.semantic_evidence_verified false    -> true
develop/plus_at/v9_review.yaml 의 게이트 review_status  proposed -> accepted
MANIFEST status                         provisional -> released
develop/reviews/ 에 이 판정 원문 보존
results-plus-at 브랜치 생성 (base results commit d8fec245 기록)
```

### Q2. R2 기록을 어떻게 처리합니까

release gate 가 `develop/reviews/R2-*` 를 요구합니다. R2/R3 를 합본으로 보냈고 그 답을
`R3b` 로 받게 되는데, 이 경우

```
(a) 합본 답을 R2 와 R3 두 기록으로 나눠 저장한다
(b) release gate 의 필수 라운드를 R1/R3 로 바꾸고 합본 사실을 기록한다
(c) 그 밖
```

어느 쪽이 맞습니까? 제 생각은 (a) 입니다 -- 합본이었음을 명시하고 같은 원문을 두 파일로
두면 gate 를 약화시키지 않습니다.

### Q3. `n/a` 두 건을 그대로 두어도 됩니까

```
moe_quotient_remainder_consistency  MoE 판정을 철회해 검사 대상이 0 건
prefill_decode_structure            활성 판정(A)이 prefill 전용이라 공통 자리가 0 건
```

둘 다 "overlay 가 그 대상을 선언하지 않았다" 라서 `n/a` 로 적었고, **"선언했는데 0 건" 은
여전히 FAIL** 로 남겨 뒀습니다. 이 구분이 타당합니까?

### Q4. 계열 C 를 사이드카로 옮긴 뒤 본표가 덜 유용해졌습니까

사용자 목적이 배치 스윕이고 residual 값은 B 에 불변이므로 손실이 없다고 봤습니다. 다만
본표만 보는 소비자는 `2..9` 가 무엇인지 모릅니다 -- `expressions.yaml` 을 함께 읽어야
합니다. MANIFEST 의 bundle 계약에 그 문장을 넣어야 합니까?

### Q5. 놓친 것

특히 **제 4 번 오류(op_id 번호 공간)와 같은 종류**가 다른 검사에도 있는지 봐 주십시오.
발행본과 원시 원장을 잇는 다른 자리에서 같은 착오를 했을 수 있습니다.

## 읽을 곳

```
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml      n_chunk kind, residual_sidecar,
                                                      withdrawn_symbols, withdrawn, moe_aggregate
develop/plus_at/expected-moonshotai__Kimi-K3.jsonl    사전 승인 입력 2,080 줄
develop/plus_at_resid.py                              파라미터 lineage + cardinality
develop/plus_at_v9.py                                 _crosswalk(), g_axis_class_consistency()
develop/plus_at_apply.py                              근거 SHA 게이트, 사이드카 생성
develop/test_plus_at.py                               11/11
develop/fixtures/plus_at/cases.yaml                   norm 가중치를 든 사례
models/moonshotai__Kimi-K3/plus_at/                   MANIFEST·DIFF·expressions·footprint
develop/reviews/                                      R1, R3 1 차 원문
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
