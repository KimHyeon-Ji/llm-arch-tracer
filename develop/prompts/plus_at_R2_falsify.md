# R2 — 이 판정들을 **깨뜨려 주십시오**

요청일 2026-09-27 (개정). `develop/PLAN_plus_at_layer.md` v3 의 R2.
**R1 에 답한 세션에는 보내지 마십시오.** R1 은 이미 끝났고 그 결과를 반영했습니다.

---

## 이 라운드의 규칙

R1 과 달리 **제 판정을 전부 공개합니다.** 부탁은 하나입니다.

> **이 판정들이 틀렸을 경우를 찾아 주십시오.** 맞다는 확인은 필요 없습니다.

## 0. R1 이 이미 바꿔 놓은 것 (배경)

R1(blind 독립 도출)의 결과로 판정이 하나 **철회**되고 하나 **추가**됐습니다.

```
KDA `5` -> n_chunk        R1 과 완전 일치. 선택자와 footprint 2,080 자리를 독립적으로 같게 도출
MoE `3840`/`12`           **철회.** R1 이 "변경 0 건" 을 권고했다 -- 실제 per-expert 축은
                          cannot determine 이고 식을 붙이면 shim 의 균등분할 대체값이
                          모델 아키텍처처럼 보인다
residual `2..9`           **추가.** R1 이 architecture 로 판정하고 식을 줬다
```

그래서 지금 활성 판정은 **둘**입니다.

## 1. 현재 상태

```
base_results_commit  d8fec245671e
tool_source_commit   ccc3660f7010
footprint digest     997726a9476fa6d21446358072d000c4...
바뀐 셀              3,342
status               provisional (release 차단 21 건 -- 이 라운드가 그중 하나입니다)

원본 맨정수 8,128  ->  적용본 4,786
  바꾼 것   3,342  (A 2,080 + C 1,262)
  남긴 것   MoE 4,784 (R1 판정으로 철회)  +  초기 빈 버퍼 `0` 2 자리
```

토큰별 분포:

```
2080  k3-kda-nchunk       n_chunk
 552  k3-residual-accum   ceil(l/R_res)+1          c_pre
 576  k3-residual-accum   ceil((l+1)/R_res)+1      c_post
  52  k3-residual-accum   ceil(l/R_res)            b_in
  56  k3-residual-accum   ceil((l+1)/R_res)        b_after
  24  k3-residual-accum   ceil(L_layers/R_res)+1   c_final
   2  k3-residual-accum   ceil(L_layers/R_res)     b_final
```

## 2. 판정 A — KDA 청크 축 (2,080 자리, prefill)

```
선택자   module `\.self_attn$`  /  op_type `exp`
         shape 전체 일치 ["B","n_h_kda","5","d_chunk","d_head_kda"]
         field input_shape·output_shape  /  shape_index 0  /  axis 2
식       n_chunk = T / d_chunk = 320 / 64 = 5
유효조건 T % d_chunk == 0,  prefill 의 chunk_kda reference 경로
```

근거 셋:

```
1  fla/ops/kda/naive.py:106,108-112,114-116
   T 를 읽고 BT=chunk_size, NT=T//BT, T%BT==0 을 정의하며 출력이 [B,H,NT,BT,...] 다
2  src/build_table.py:1250
   트레이서가 이미 T == d_chunk*n_chunk 를 등재하고 있다 (_MERGE_IDENTITY_EQUIVALENTS)
3  발행본 자신 -- 같은 축을 곱 안에서는 이미 `B*n_h_kda*n_chunk` 로 쓴다
   (3,168 자리 / 1,056 행). 단독일 때만 `5` 로 남았다. **표가 자기모순이었다.**
```

R1 도 같은 결론에 도달했고 근거 3 을 독립적으로 지적했습니다. 다만 R1 은 분류를
`trace_artifact` 로 봤습니다 -- Triton 커널이 아니라 shim 이 노출한 torch reference 내부
축이라는 점입니다. 제 overlay 는 `kind: architecture` 로 적었습니다. **이 불일치를 봐
주십시오** (아래 Q3).

## 3. 판정 C — residual 누적 (1,262 자리)

`R_res = attn_res_block_size = 12`, `L_layers = num_hidden_layers = 93`,
`l = 그 행의 layer_idx` (0-based):

```
b_in(l)    = ceil(l/R_res)          레이어 진입 시 저장된 residual stream 수
b_after(l) = ceil((l+1)/R_res)      경계(l % R == 0)에서 추가한 뒤
c_pre(l)   = ceil(l/R_res) + 1      pre-attention 혼합 후보 수   (l > 0)
c_post(l)  = ceil((l+1)/R_res) + 1  post-attention 혼합 후보 수
b_final    = ceil(L_layers/R_res)   전 층 종료 후
c_final    = ceil(L_layers/R_res)+1 최종 output 혼합 후보 수
```

근거: `modeling_kimi_linear.py:1188-1192`(저장소가 너비 0 으로 시작),
`:995-998`(l % R == 0 에서 stream 하나 추가), `:1075-1087`(`_apply_attn_res` 가 저장소와
현재 prefix 를 concat 하고 그 축으로 score/softmax/BMM 수행).
그리고 **R1 과 앞선 R3 세션이 별개로 같은 식에 도달**했습니다.

### 여기서 가장 중요한 것 — stage 를 값으로 고르면 안 됩니다

```
1,262 셀 중 식이 **유일하게** 결정되는 것은 216 셀뿐이다.
950 셀이 두 식에, 96 셀이 세 식에 맞는다.  (실측)
```

그래서 `develop/plus_at_resid.py` 가 **op lineage** 로 stage 를 판별합니다.

```
boundary_append  concat 의 출력이 혼합 체인으로 안 간다 (버퍼를 키운다)
mix_pre/post     concat 출력이 혼합 체인(elementwise_mul/sum/softmax/batched_matmul)으로
                 간다. 층 안에서 **뒤에서부터** 배정 -- 마지막이 post, 그 앞이 pre
mix_final        module == model
```

결과: **1,262 셀 전부 stage 가 유일하게 결정되고, 접힌 행(`layers` 범위)의 모든 층에서
식이 그 값을 낸다. 불일치 0, stage 없는 셀 0.**

층 0 에서 제 첫 구현이 틀렸습니다. 앞에서부터 pre/post 를 세면 층 0(버퍼가 비어 pre-mix 가
없다 -- R1 의 `c_pre` 는 `l>0`)에서 12 건이 어긋납니다. 뒤에서부터 배정해 0 이 됐습니다.

### `l` 은 심볼이 아닙니다

`l` 을 `kind: row_field` 로 선언했습니다 -- **그 행의 `layer_idx`** 입니다. 접힌 행에서도
그 행의 모든 층에서 식이 같은 값을 내므로 대표 `layer_idx` 로 대입해도 됩니다(적용기가
층마다 산술로 확인합니다).

## 4. 반증 조건 (제가 적어 둔 것)

```
A   T=384, d_chunk=64 로 재트레이스하면 이 축의 구체값이 6 이어야 한다. 5 면 틀렸다.
C   attn_res_block_size 를 다른 값으로 트레이스하면 경계 concat 이 생기는 층이 바뀌고
    이 축들의 값도 바뀌어야 한다. 안 바뀌면 식이 틀렸다.
```

**이 조건들이 충분합니까?** 더 쉽게 깨뜨릴 방법이 있으면 그것을 알려 주십시오.

## 5. negative control (제가 확인한 것)

```
A 가 건드리면 안 되는 자리
  발행본 residual 계열의 `5` (module 이 root/model.layers.N, op 이 concat/sum/softmax/…)
  self_attn 안이지만 op 이 exp 가 아닌 `5`
  원시 원장의 [B, n_h_kda, 5, d_chunk, 5] 축 4 (causal prefix)
  -- 이 shape 은 **발행본에 없다** (확인: n_h_kda + 맨정수 2 개 이상인 자리 0 건)

C 가 건드리면 안 되는 자리
  self_attn 안의 `5` (계열 A 가 맡는다)
  [B*T, 0, d_model] 의 `0` (초기 빈 버퍼 -- 리터럴이 맞다)
  [..., 1, ...] 의 `1` (singleton -- 리터럴이 맞다)
  MoE 의 3840/12 (철회했다)
```

## 6. 유효 범위의 한계 (제가 인정하는 것)

```
point_verified        참. 이 트레이스의 값에서 식에 대입하면 원본 구체값이 정확히 복원된다
                      (A: n_chunk=5;  C: 접힌 행의 **모든 층**에서 확인)
sweep_verified        **거짓.** T·d_chunk·R_res 를 바꿔 다시 트레이스하지 않았다
semantic_evidence     R2 가 끝나기 전에는 거짓으로 둔다
```

### C 의 `R_res` sweep 한계 (R1 이 지적했고 그대로 적었습니다)

`attn_res_block_size` 를 바꾸면 (a) 어느 층에서 경계 concat 이 생기는지가 바뀌고
(b) 지금 `repeat/layers` 로 합쳐진 층들이 서로 다른 shape 를 갖게 되며 (c) 일부 op 자체가
생기거나 사라집니다. **즉 control flow 가 달라집니다.** 축 문자를 식으로 바꾼 것은 현재
DAG 를 정확히 기술한 것이고, R sweep 을 하려면 행을 층별로 펼치거나 다시 트레이스해야
합니다. `B` 만 바꾸면 이 축의 값은 변하지 않습니다.

### C 의 독립성 한계 (제가 먼저 적습니다)

계열 A 는 적용기와 독립 matcher 가 규칙을 **따로** 구현해 대조했습니다. 계열 C 는 두 쪽이
**같은 분류기를 공유합니다**(`develop/plus_at_resid.py`) -- lineage 분류기를 두 번 쓰는
것이 실질적 독립이 아니라고 판단했습니다. 대신:

```
1  식 자체를 R1(blind)이 독립 도출했다
2  적용기가 셀마다 접힌 행의 **모든 층**에서 식이 성립하는지 산술로 재확인한다
   (선택이 아니라 주장의 검증 -- 실패 방식이 다르다)
3  후보 완전성: shape 로 뽑은 residual 후보 전부가 footprint 에 있어야 한다.
   분류기가 한 셀이라도 놓치면 적용기가 잡는다
4  fixture 에 stage 별 사례를 뒀다 (사람이 직접 작성, 10/10 통과)
```

## 7. 물어보는 것

### Q1. 어느 판정이 가장 약합니까

둘 중 먼저 깨질 것을 지목하고 왜 그런지 적어 주십시오.

### Q2. stage 판별을 어떻게 깨뜨리겠습니까

`boundary_append` vs `mix` 를 "concat 출력이 혼합 체인 op 로 가는가" 로 가르고,
pre/post 를 "층 안 op_id 순서에서 뒤가 post" 로 가릅니다. 이 두 판별이 틀리는 구조가
있습니까? 특히:

```
- 한 층에 mix 그룹이 셋 이상 나오는 경우
- append 의 출력이 혼합 체인으로도 가는 경우
- 층 경계가 op_id 순서와 어긋나는 경우
```

### Q3. A 의 분류가 `architecture` 입니까 `trace_artifact` 입니까

R1 은 `trace_artifact` 로 봤습니다 -- 이 축은 실제 GPU Triton 커널의 축이 아니라 shim 이
노출한 torch reference 구현의 내부 축입니다. 제 overlay 는 `kind: architecture` 로 적었고,
근거는 `n_chunk = T/d_chunk` 가 config 값들로만 결정된다는 점입니다.

**어느 쪽이 맞습니까?** 이것이 `symbols.yaml` 의 `kind` 에 들어가고, 소비자가 아키텍처
심볼과 트레이스 산물을 가르는 기준이 됩니다.

### Q4. `l` 을 표의 축 토큰에 넣은 것이 옳습니까

`ceil(l/R_res)+1` 처럼 **행 필드를 참조하는 식**을 축에 적었습니다. 표가 자립하려면
소비자가 "`l` 은 그 행의 `layer_idx`" 를 알아야 합니다. `symbols.yaml` 과 MANIFEST 의
bundle 계약에 적었지만, 이 방식 자체가 위험합니까? 대안은 리터럴을 두고 사이드카에만
식을 적는 것입니다.

### Q5. MoE 철회가 옳았습니까

R1 의 권고를 따랐습니다. 대신 overlay 에 `moe_aggregate` 를 넣어 "총량만 말한다" 로
적었습니다(`4*3840 = B*T*k = 15360`, 전문가별 분포·active 수·weight traffic·cache·
latency 는 미보존). **사용자의 목적이 roofline 인데 MoE 가 그 모델의 지배적 계산입니다.**
리터럴로 두고 집계 안내만 주는 것으로 충분합니까?

### Q6. 놓친 것

위 밖에서 제가 못 본 실패 방식이 있으면 지적해 주십시오.

## 8. 읽을 곳

```
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml   판정 원문 (근거·exemplar·limitations·withdrawn)
develop/plus_at/expected-moonshotai__Kimi-K3.jsonl 사전 승인된 변경 허용 목록 3,342 줄
develop/plus_at_resid.py                           stage 분류기
develop/fixtures/plus_at/cases.yaml                사람이 쓴 fixture
models/moonshotai__Kimi-K3/plus_at/                적용본 + footprint + MANIFEST + DIFF.md
models/moonshotai__Kimi-K3/prefill.csv             원본
develop/reviews/R1-2026-09-27-codex-blind.md       R1 원문
src/kda_shim.py:633-652                            moe_infer (철회 근거)
fla/ops/kda/naive.py:100-170
```

**읽기만 하고 수정하지 말아 주십시오.**
