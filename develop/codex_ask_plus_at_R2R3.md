# R2 + R3 합본 — 반증 검토와 publish 승인을 한 번에 청합니다

요청일 2026-09-28. 선행: R1(blind 독립 도출, 2026-09-27) 완료 및 반영,
R3 1 차(차단 사항 5 종) 완료 및 반영.

**먼저 알려야 할 것: R2 와 R3 를 합쳤습니다.** 계획(v3)은 셋을 따로 돌리기로 했는데,
사용자가 속도를 요구하는 상황이라 반증(R2)과 전수·publish(R3)를 한 요청으로 묶었습니다.
**라운드를 분리해야 한다고 보시면 그렇게 말씀해 주십시오** -- 다시 나눠 보내겠습니다.
R1(blind)은 이미 별도 세션에서 끝났고 그 판정을 반영했습니다.

```
base_results_commit  d8fec245671e
tool_source_commit   ccc3660f7010  (지금은 e2a4ae41)
footprint digest     997726a9476fa6d21446358072d000c4...
바뀐 셀              3,342
status               provisional (차단 21 건 -- 이 검토가 그중 다수를 해소합니다)
```

---

## 1. R1 이 바꿔 놓은 것 (반영 완료)

```
KDA `5` -> n_chunk      R1 과 **완전 일치**. 선택자와 footprint 2,080 자리를 독립적으로
                        같게 도출했다. R1 이 든 근거 중 하나가 내가 찾은 것과 같다 --
                        같은 발행본에서 그 축을 곱 안에서는 이미 `B*n_h_kda*n_chunk` 로
                        쓴다(3,168 자리). 표가 자기모순이었다.
MoE `3840`/`12`         **철회했다.** R1 이 "변경 0 건" 을 권고했다.
residual `2..9`         **추가했다.** R1 이 architecture 로 판정하고 식을 줬다.
```

R1 원문: `develop/reviews/R1-2026-09-27-codex-blind.md`

최종 변경량이 R1 의 권고량과 정확히 같습니다: **2,080 + 1,262 = 3,342 shape 자리.**

## 2. 지금 상태 — 원본 8,128 -> 적용본 4,786

```
바꾼 것   3,342
남긴 것   MoE 3840/12  4,784   (R1 판정. overlay 의 withdrawn 절에 원문 보존)
          초기 빈 버퍼 `0`  2   (리터럴이 맞다)
```

다른 네 모델(Llama-4-Maverick, gpt-oss-20b/120b, DeepSeek-V4-Pro)은 맨정수 0 이라 +@ 가
필요 없습니다.

footprint 분포:

```
k3-kda-nchunk       prefill  input_shape   1040    output_shape  1040
k3-residual-accum   prefill  input_shape    384    output_shape   247
                    decode   input_shape    384    output_shape   247
                                                          합계  3,342
```

토큰별:

```
2080  n_chunk
 552  ceil(l/R_res)+1          c_pre
 576  ceil((l+1)/R_res)+1      c_post
  52  ceil(l/R_res)            b_in
  56  ceil((l+1)/R_res)        b_after
  24  ceil(L_layers/R_res)+1   c_final
   2  ceil(L_layers/R_res)     b_final
```

## 3. 판정 A — KDA 청크 축 (2,080 자리, prefill)

```
선택자  module `\.self_attn$` / op_type `exp` /
        shape 전체 일치 ["B","n_h_kda","5","d_chunk","d_head_kda"] /
        field input_shape·output_shape / shape_index 0 / axis 2
식      n_chunk = T / d_chunk = 320/64 = 5
유효    T % d_chunk == 0,  prefill 의 chunk_kda reference 경로
근거    fla/ops/kda/naive.py:106,108-112,114-116  (NT = T//BT, 출력이 [B,H,NT,BT,...])
        src/build_table.py:1250  (트레이서가 T == d_chunk*n_chunk 를 이미 등재)
        발행본 자신  (곱 안에서는 이미 n_chunk. 단독일 때만 5 였다 -- 자기모순)
반증    T=384, d_chunk=64 로 재트레이스하면 구체값이 6 이어야 한다
```

## 4. 판정 C — residual 누적 (1,262 자리)

`R_res = attn_res_block_size = 12`, `L_layers = num_hidden_layers = 93`,
`l = 그 행의 layer_idx` (0-based):

```
b_in(l)    = ceil(l/R_res)           진입 시 저장된 residual stream 수
b_after(l) = ceil((l+1)/R_res)       경계(l % R == 0)에서 추가한 뒤
c_pre(l)   = ceil(l/R_res) + 1       pre-attention 혼합 후보 수   (l > 0)
c_post(l)  = ceil((l+1)/R_res) + 1   post-attention 혼합 후보 수
b_final    = ceil(L_layers/R_res)
c_final    = ceil(L_layers/R_res) + 1
```

근거: `modeling_kimi_linear.py:1188-1192`(저장소가 너비 0 으로 시작), `:995-998`(경계에서
stream 추가), `:1075-1087`(`_apply_attn_res` 가 저장소 + 현재 prefix 를 concat 하고 그 축으로
score/softmax/BMM). **R1 과 앞선 R3 세션이 별개로 같은 식에 도달했습니다.**

### 핵심 -- stage 를 값으로 고르면 안 됩니다

```
1,262 셀 중 식이 **유일하게** 결정되는 것은 216 셀뿐이다.
950 셀이 두 식에, 96 셀이 세 식에 맞는다.  (실측)
```

그래서 `develop/plus_at_resid.py` 가 **op lineage** 로 판별합니다.

```
boundary_append  concat 의 출력이 혼합 체인 op 로 안 간다 (버퍼를 키운다)
mix_pre/post     concat 출력이 혼합 체인(elementwise_mul/sum/softmax/batched_matmul)으로
                 간다. 층 안에서 **뒤에서부터** 배정 -- 마지막이 post, 그 앞이 pre
mix_final        module == model
```

결과: **1,262 셀 전부 stage 유일, 접힌 행(`layers` 범위)의 모든 층에서 식이 그 값을 냄,
불일치 0, stage 없는 셀 0.**

층 0 에서 제 첫 구현이 틀렸습니다. 앞에서부터 세면 층 0(버퍼가 비어 pre-mix 가 없다 --
R1 의 `c_pre` 는 `l>0`)에서 12 건이 어긋납니다. 뒤에서부터 배정해 0 이 됐습니다.

`l` 은 심볼이 아니라 `kind: row_field` 로 선언한 **그 행의 layer_idx** 입니다.

## 5. 검증 결과 (전수)

```
V1 격리        비대상 canonical cell 전부 original == derived      통과
               (파일 바이트가 아니라 cell 비교. CRLF 로 되쓰면 원본과 바이트 동일함은
                다섯 모델에서 따로 확인)
V2 집합 동일성 actual == expected(사전 승인 입력), digest 일치      통과
V3 sanity      한 shape 에 같은 맨정수 둘 이상 -> 전체 실패         위반 0
V4 형태 보존   cell 키 집합 동일 (행·피연산자 수·축 개수 불변)      통과
V5 canonical   csv 와 jsonl 을 같은 파서로 배열화해 비교            통과
V6 점 대입     n_chunk=5 -> 5.  식 토큰은 **행 단위** -- 접힌 행의
               모든 층에서 원본 값 복원                             통과
V7 잔여 보고   남긴 맨정수 4,786 전수 보고                          보고됨
V8 의미 맥락   바뀐 셀의 (module, op_type) 이 선언 집합 안          통과
멱등·결정론    재적용 시 출력 해시 동일                             확인
fixture        사람이 쓴 사례 + 산술, 10/10                          통과
```

V9 (`develop/plus_at_v9.py` 가 실제로 돕니다):

```
pass  rerun                schema_shape_rank_token_type
pass  rerun                symbol_declared
pass  rerun                expression_no_cycle
pass  rerun                substitution_nonneg_integer
pass  rerun                zero_axis_only_initial_residual
pass  rerun                batch_seq_head_axis_consistency
pass  rerun                head_scope_exclusive
pass  rerun                prefill_decode_structure
pass  rerun                layers_repeat_consistency
pass  rerun                row_metadata_preserved
pass  rerun                op_id_dag
pass  rerun                reshape_derivation          (원본 이견 0 -> 파생 0)
pass  inherited_unchanged  port_coverage               (사이드카 해시 + cell 키 불변)
n/a   not_applicable       moe_quotient_remainder_consistency  (MoE 철회로 대상 없음)
not_run not_evaluated      axis_class_consistency      <- **미평가. release 를 막는다**
```

## 6. 제가 아는 약점 (먼저 적습니다)

```
1  sweep_verified 거짓.  T·d_chunk·R_res 를 바꿔 재트레이스하지 않았다. 식이 이 한 점에서만
   확인됐다. 틀린 식도 한 점에서는 우연히 맞을 수 있다.
2  C 의 R_res sweep 은 이 치환으로 안 된다. R1 지적대로 attn_res_block_size 를 바꾸면
   경계 concat 이 생기는 층, 행 접힘, 심지어 op 존재 자체가 달라진다(control flow).
   축 문자를 식으로 바꾼 것은 **현재 DAG 를 정확히 기술한 것**이다. B sweep 에는 무해하다.
3  C 는 적용기와 독립 matcher 가 **같은 분류기를 공유한다**(plus_at_resid.py).
   A 와 달리 구현 독립이 없다. 대신 (a) 식을 R1 이 독립 도출, (b) 적용기가 접힌 행의 모든
   층에서 산술 재확인, (c) shape 기반 후보 완전성 검사, (d) 사람이 쓴 fixture.
4  선언 집합이 틀리면 V8 은 통과한다(선언을 그대로 믿는다).
5  axis_class_consistency 를 평가하지 않았다. 이 작업이 축 의미를 바꾸므로 직접 관련되는데
   등가류 재실행을 배선하지 않았다. not_applicable 이 아니라 **미검증**으로 적었다.
```

## 7. 물어보는 것

### Q1. stage 판별을 어떻게 깨뜨리겠습니까 (가장 중요)

`append` vs `mix` 를 "concat 출력이 혼합 체인 op 로 가는가" 로, `pre` vs `post` 를 "층 안
op_id 순서에서 뒤가 post" 로 가릅니다. 이 판별이 틀리는 구조가 있습니까? 특히:

```
- 한 층에 mix 그룹이 셋 이상 나오는 경우
- append 의 출력이 혼합 체인으로도 가는 경우
- 층 경계가 op_id 순서와 어긋나는 경우
- decode 에서 구조가 다른 경우 (지금 prefill/decode 가 각각 631 자리로 같다)
```

### Q2. A 의 분류가 `architecture` 입니까 `trace_artifact` 입니까

**R1 과 제가 갈린 지점입니다.** R1 은 `trace_artifact` 로 봤습니다 -- 이 축은 실제 GPU
Triton 커널의 축이 아니라 shim 이 노출한 torch reference 구현의 내부 축이라는 점입니다.
제 overlay 는 `kind: architecture` 로 적었고 근거는 `n_chunk = T/d_chunk` 가 config 값들로만
결정된다는 점입니다.

이것이 `symbols.yaml` 의 `kind` 에 들어가고 소비자가 아키텍처와 트레이스 산물을 가르는
기준이 됩니다. 어느 쪽입니까?

### Q3. 행 필드 `l` 을 축 토큰에 넣은 것이 옳습니까

`ceil(l/R_res)+1` 처럼 **행 필드를 참조하는 식**을 축에 적었습니다. 표가 자립하려면 소비자가
"`l` 은 그 행의 `layer_idx`" 를 알아야 합니다. `symbols.yaml` 과 MANIFEST 의 bundle 계약에
적었지만, 이 방식 자체가 위험합니까? 대안은 리터럴을 두고 사이드카에만 식을 적는 것입니다.

### Q4. MoE 철회로 충분합니까

R1 을 따라 4,784 자리를 리터럴로 뒀습니다. 대신 overlay 에 `moe_aggregate` 를 넣었습니다:

```
total_routed_tokens  prefill "B*T*k" (15360) / decode "B*k" (48)
traced_experts       4   (모델은 E = 896)
how                  전문가 행을 **합쳐서** 쓰라. 총 expert projection FLOPs 는 보존된다
                     (4 x 3840 = 15360 = B*T*k). 개별 행의 토큰 축은 균등분할 대체값이다
not_preserved        전문가별 실제 분포 / active expert 수 / weight traffic / cache·latency
```

**사용자의 목적이 roofline 이고 MoE 가 이 모델의 지배적 계산입니다.** 리터럴 + 집계 안내로
충분합니까, 아니면 별도 consumer view(`roofline_aggregate.yaml`)를 만들어야 합니까?

### Q5. `axis_class_consistency` 를 어떻게 합니까

미평가로 뒀고 그게 release 를 막습니다. 세 갈래로 보입니다.

```
(a) 등가류 재실행을 배선한다 -- 원시 원장 + 포트 사이드카 + crosswalk 가 저장소에 있다
(b) 불변 승계를 증명한다 -- 축 **이름**만 바꿨고 cell 키·구체값·포트가 불변이므로
    등가류 자체는 변하지 않는다. 다만 등가류 **안의 이름 일관성**은 영향을 받는다
(c) 미검증으로 두고 release 를 막은 채 남긴다
```

(b) 가 성립한다고 보십니까? 등가류는 구체 shape 으로 세우므로 이름 치환에 불변이고, 제가
바꾼 것은 한 등가류 안의 자리 **전부**(footprint 완전성으로 확인)라서 "한 축에 이름이 둘"
이 생기지 않는다고 봅니다 -- 다만 그걸 표만으로 증명할 수는 없습니다.

### Q6. 정식 publish 를 승인하십니까

승인하시면:

```
overlay 의 substitution status  proposed -> accepted
verification.semantic_evidence_verified  false -> true
develop/plus_at/v9_review.yaml 의 게이트 review_status  proposed -> accepted
MANIFEST status  provisional -> released
results-plus-at 브랜치 생성 (base results commit 기록)
```

조건이 남았다면 무엇인지 적어 주십시오. 특히 Q5 가 미해결이면 그 게이트만 남기고 나머지를
승인하는 것도 가능합니까?

### Q7. footprint 전수를 보고 이상한 자리가 있습니까

`actual_footprint.jsonl` 3,342 줄이 그대로 있습니다. 각 줄이
`{phase, op_id, field, shape_index, axis, before, after, module_path, op_type, sub_id}` 입니다.

특히 확인 부탁: residual 의 stage 배정입니다. 같은 값인데 `c_pre` / `c_post` 로 갈린 자리가
있고(층 5 에서 둘 다 3), 그 구분이 lineage 로만 정당화됩니다.

### Q8. 놓친 것

위 밖에서 제가 못 본 실패 방식이 있으면 지적해 주십시오.

## 8. 읽을 곳

```
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml    판정 원문 (근거·exemplar·limitations·
                                                    withdrawn·moe_aggregate)
develop/plus_at/expected-moonshotai__Kimi-K3.jsonl  사전 승인 입력 3,342 줄
develop/plus_at_resid.py                            stage 분류기 (C)
develop/plus_at_refmatch.py                         독립 matcher (A)
develop/plus_at_apply.py  plus_at_canon.py  plus_at_v9.py  plus_at_diff.py
develop/fixtures/plus_at/cases.yaml                 사람이 쓴 fixture
develop/test_plus_at.py                             10/10
develop/plus_at/v9_review.yaml                      게이트별 검토 상태
develop/reviews/R1-2026-09-27-codex-blind.md        R1 원문
develop/reviews/R3-2026-09-27-codex.md              R3 1 차 원문
models/moonshotai__Kimi-K3/plus_at/                 적용본·footprint·MANIFEST·DIFF.md
models/moonshotai__Kimi-K3/prefill.csv              원본 (대조용)
src/kda_shim.py:633-652   fla/ops/kda/naive.py:100-170
modeling_kimi_linear.py:995-998,1075-1087,1188-1192
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> Q1(stage 판별을 깨뜨리는 구조)과 Q6(publish 승인)이 가장 중요합니다.
> Q2(A 의 분류 -- R1 과 갈린 지점), Q3(`l` 을 축에 넣은 것), Q4(MoE 철회 충분성),
> Q5(`axis_class_consistency` 처분)도 정해 주십시오.
>
> 그리고 **R2/R3 를 합친 것이 부적절하면 그렇게 말씀해 주십시오** -- 다시 나눠 보냅니다.
