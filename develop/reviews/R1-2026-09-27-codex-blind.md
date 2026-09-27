# R1 원문 기록 — blind 도출 라운드, 2026-09-27

라운드: R1 (현재 라벨·후보를 가리고 축의 의미와 식을 직접 도출)
prompt: `develop/prompts/plus_at_R1_derive.md` + `plus_at_R1_data.md`
검토자: Codex (제안식을 보지 않은 세션)
"읽기만 했고 파일은 수정하지 않았습니다" 명시.

## 결론 요약 (원문)

- KDA 의 `5` 는 기존 심볼 `n_chunk` 로 바꿔야 한다.
- residual 축의 `2..9` 는 레이어별 residual 슬롯 수다. 심볼화할 수 있지만
  `attn_res_block_size` 를 바꾸면 DAG 와 압축 그룹도 달라지므로 단순 값 치환만으로
  완전한 config sweep 은 불가능하다.
- MoE 의 `3840`/`12` 는 실제 모델 속성이 아니라 shim 이 만든 균등분할 값이다.
  **아키텍처 심볼로 바꾸면 안 된다.**
- `0`, singleton `1`, `2*d_moe` 의 계수 `2` 는 리터럴이 맞다.

## 축 1 — KDA 의 `5`

```
shape/축   [B, n_h_kda, 5, d_chunk, d_head_kda] 의 index 2
의미       chunk KDA reference 구현에서의 시퀀스 청크 개수
식         n_chunk = T / d_chunk
분류       trace_artifact  (단, reference trace 내부 일관성을 위해 기존 심볼 n_chunk 로
           고치는 것이 맞다)
```

근거:
- `fla/ops/kda/naive.py:106` 에서 T 를 읽고, `:108-112` 에서 `BT=chunk_size`,
  `NT=T//BT`, `T % BT == 0` 을 정의한다.
- `naive.py:114-116` 의 출력 레이아웃이 `[B,H,NT,BT,...]` 다.
- 같은 발행본에서 바로 다음 BMM 은 이미 `B*n_h_kda*n_chunk` 로 렌더된다
  (`prefill.csv:20` 의 `5` 와 `:21` 의 `n_chunk` 는 같은 chunk decomposition 의 축).

유효 조건: prefill 의 `chunk_kda` reference 경로, `T % d_chunk == 0` (320/64=5),
decode 는 recurrent 경로라 이 축이 없다. **실제 GPU Triton 내부 op 가 아니라 shim 이
노출한 torch reference 내부 축이다.**

Footprint (독립 도출):
```
phase=prefill
module_path =~ ^model\.layers\.\d+\.self_attn$
op_type=exp
shape 정확히 [B,n_h_kda,5,d_chunk,d_head_kda]
input_shape/output_shape 의 index 2 만
-> 2,080 shape 자리 / 2,080 CSV shape-field 셀
```

바꾸면 안 되는 자리: `[B*T,5,d_model]`, `[B*T,5]`, `[B*T,5,1]`, `[B*T,1,5]` 의 `5` 는
residual 후보 수다. 값만 보고 바꾸면 오염된다.

## 축 2 — MoE 의 `3840` / `12`

```
의미          shim 이 한 traced expert 에 잘라 준 routed-token assignment 수
실제 모델 식  cannot determine
shim 식       N = B*T*k (prefill), B*k (decode);  C = min(expert_cap,E) = 4
              현재 N%C=0 이므로 각 slice = N/C.  3840 = 3*320*16/4,  12 = 3*16/4
분류          trace_artifact
```

실제 모델에서는 "한 전문가가 받는 토큰 수" 가 라우팅 결과에 따라 달라지므로
`B, T, k, E` 만으로 결정되지 않는다.

근거: `modeling_kimi_linear.py:720-750` (토큰마다 top-k 선택),
`:841-858` (전문가별 수를 라우팅 값에서 계산), `src/kda_shim.py:623` (cap 4),
`:636-648` (flattened assignment 를 네 조각으로). 일반 shim 식도 항상 `N/C` 가 아니다 --
`per = max(1, floor(N/C))` 이고 마지막 expert 가 나머지를 받는다. `N%C=0` 일 때만 네
expert 가 같은 식이다.

**권고 footprint 는 변경 0 건이다.** 실제 per-expert 축은 cannot determine 이고,
`B*T*k/4` 같은 이름을 붙이면 shim 의 대체값이 모델 아키텍처처럼 보인다. 필요하다면
아키텍처 심볼이 아니라 명시적인 `trace_even_split_*` 주석으로만 표현해야 한다.

특히 decode 의 `12` 를 `attn_res_block_size` 로 바꾸면 안 된다 -- 숫자만 같고 의미가 다르다.

## 축 3 — residual 의 `2..9`

두 관련 축이 있다: `block_residual` 축(지금까지 저장된 block residual stream 수)과
`_apply_attn_res` 의 v/scores/probs 축(저장된 stream + 현재 prefix_sum 을 합친 혼합 후보 수).

`R = attn_res_block_size`, `ℓ = layer_idx`, 전체 레이어 수 `L`:
```
레이어 진입 시 저장 수      b_in(ℓ)    = ceil(ℓ/R)
경계 처리 후 저장 수        b_after(ℓ) = ceil((ℓ+1)/R)
pre-attention 혼합 후보 수   c_pre(ℓ)   = ceil(ℓ/R) + 1      (ℓ>0)
post-attention 혼합 후보 수  c_post(ℓ)  = ceil((ℓ+1)/R) + 1
최종 output 혼합 후보 수     c_final    = ceil(L/R) + 1
```
현재 R=12, L=93 이므로 최종값은 ceil(93/12)+1 = 9.

근거: `modeling_kimi_linear.py:1188-1192` (residual 저장소가 너비 0 으로 시작),
`:995-998` (ℓ % R == 0 일 때 stream 하나 추가),
`:1075-1087` (`_apply_attn_res` 가 저장소와 현재 prefix 를 concat 하고 그 축으로
score/softmax/BMM 을 수행).

분류: **architecture**

Footprint (독립 도출):
```
module_path =~ ^model(?:\.layers\.\d+)?$
op_type ∈ {concat, elementwise_mul, sum, softmax, batched_matmul}
대상 축   [tokens,r,d_model] axis 1 / [tokens,r] axis 1 / [tokens,r,1] axis 1
          / [tokens,1,r] axis 2     tokens = B*T(prefill) 또는 B(decode), r ∈ 2..9
-> prefill 631 shape 자리 / 535 CSV 셀,  decode 631 / 535,  합계 1,262 / 1,070
```

유효 조건 (중요): `attn_res_block_size` 를 바꾸면 (a) 어느 레이어에서 경계 concat 이
생기는지가 바뀌고, (b) 현재 `repeat/layers` 로 합쳐진 레이어들이 서로 다른 shape 를 가질
수 있으며, (c) 일부 op 자체가 생기거나 사라진다. 따라서 현재 DAG 에서 이 축을 layer/site
식으로 표시하는 것은 맞지만, **R 을 자유롭게 sweep 하려면 행을 레이어별로 펼치거나 다시
trace 해야 한다. 축 문자만 바꾸는 것으로는 충분하지 않다.**

## 바꾸지 말아야 할 축

- `[B*T,0,d_model]` / `[B,0,d_model]` 의 `0`: 초기 empty block_residual. 리터럴이 맞다.
- `[...,1,...]` 의 `1`: unsqueeze(1), singleton projection/query 축, broadcast 축.
- `2*d_moe` 의 `2`: 두 MLP projection 을 concat 한 폭. 이미 올바른 symbolic expression 이고
  residual 의 `2` 나 `E_shared=2` 와 무관하다.
- MoE 의 `3840`/`12`: 값 자체는 trace 결과로 맞지만 실제 모델의 symbolic axis 는 결정
  불가능. 그대로 두고 caveat 을 유지해야 한다.

## 표 자체의 모순

명확한 렌더링 불일치는 한 가지다 -- KDA rank-5 tensor 에서는 청크 수가 `5` 로 남았는데
바로 이어지는 flatten tensor 에서는 같은 축이 `n_chunk` 로 렌더된다. **고쳐야 할 내부
불일치다.**

나머지는 현재 config 에서 수치적으로 일관된다. 네 개의 `[3840,d_moe_lat]` 를 concat 해
`[B*k*T,d_moe_lat]` 가 되는 것은 `4*3840 = 3*16*320` 이므로 모순이 아니다 -- 다만 앞쪽
값은 shim 산물이다. residual 의 concat → sum/softmax → BMM 축도 같은 `2..9` 를 일관되게
보존한다. `attn_res_block_size` sweep 문제는 현재 값의 모순이라기보다, 압축된 현 발행본이
그 control-flow sweep 을 표현하지 못한다는 한계다.

## 최종 권고

```
KDA       2,080 자리   바꿀 것
residual  1,262 자리   바꿀 수 있음 (단 R sweep 한계 기록)
MoE       4,784 자리   **아키텍처 식으로 바꾸지 말 것**
합계      3,342 shape 자리
```
