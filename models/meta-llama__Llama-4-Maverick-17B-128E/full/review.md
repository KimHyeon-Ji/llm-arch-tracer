# 리뷰 패킷 — meta-llama/Llama-4-Maverick-17B-128E

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2` / 트레이스 seq_len(T) = 16
> 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 1. 이 산출물이 무엇인가

Hugging Face의 **공식 config + modeling 코드를 meta device에서 실제로 forward 실행**하고,
그 실행을 PyTorch dispatch(ATen) 레벨에서 가로채 op·shape·의존관계를 기록한 것입니다.
가중치는 없지만(shape 계산에 불필요) 연산 그래프는 실제로 실행된 것이며, 값을 지어내지
않습니다. shape은 아키텍처 심볼(`B, T, d_model, n_h, …`)로 렌더됩니다.

**따라서 트레이스 자체(어떤 op이 어떤 크기로 도는가)는 관측값이고, 검토 대상은
"그 축에 붙은 이름이 맞는가"입니다.**

## 2. 심볼표 (이 모델에서 각 이름이 갖는 값)

```
  L            = 48
  d_model      = 5120
  n_h          = 40
  n_kv         = 8
  d_head       = 128
  d_ff         = 16384
  V            = 202048
  ctx          = 262144
  E            = 128
  E_shared     = 1
  k            = 1
  d_moe        = 8192
  w_local      = None
  n_sink       = None
  layer_sched  = ['chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention']
  c_kv         = None
  d_nope       = None
  d_v          = None
  c_q          = None
  d_rope       = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  d_state      = None
  n_g_ssm      = None
  n_h_ssm      = None
  d_head_ssm   = None
  d_conv       = None
  n_mem        = None
  r_lora       = None
  d_attn       = None
  n_h_lin_k    = None
  n_h_lin_v    = None
  d_head_lin_k = None
  d_head_lin_v = None
  d_conv_lin   = None
```

## 3. 모델 요약 산출물

# Model Summary -- meta-llama/Llama-4-Maverick-17B-128E

## 기본 정보

- revision: `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 400.71B total, 17.18B active (4.3% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-04-02  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 36× chunked_attention, 12× GQA  (FFN: 48× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 192.0 KiB (High) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=128, top-1, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, NoPE, GQA, MoE, shared expert, sigmoid-gating |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `llama4_text` |
| attention | GQA — 40 query : 8 kv heads (repeat 5), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=500000.0) |
| FFN | MoE — 128 routed experts, top-1 + 1 shared, expert intermediate 8192, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; all 48 layers ⇒ 98304 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 48 |
| d_model | 5120 |
| n_h | 40 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 16384 |
| V | 202048 |
| ctx | 262144 |
| E | 128 |
| E_shared | 1 |
| k | 1 |
| d_moe | 8192 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 36× chunked_attention, 12× full_attention (총 48층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | —  _(해당 없음: 이 모델은 `partial_rope` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| d_state | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_g_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_h_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| d_head_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| d_conv | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_mem | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| r_lora | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| d_attn | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| n_h_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| n_h_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_conv_lin | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |

## 미등록 config 필드 (Tier 2 조사 대상)

이 아키텍처가 실제로 쓰는 config 필드 중 `rules/symbols.yaml`에 등록되지 않은 것들이다. 등록되지 않은 폭은 이름을 붙일 근거가 없으므로 shape 셀에 정수로 남는다. `02-new-module-handling.md` Tier 2 절차로 역할을 확인한 뒤 `aliases`(같은 개념의 다른 필드명) 또는 `derived_dims.yaml`(계산식)에 **출처와 함께** 등록하면 다음 모델부터 자동으로 잡힌다.

| config 필드 | 값 | 쓰는 모듈 수 |
|---|---|---|
| `floor_scale` | 8192 | 48 |
| `intermediate_size` | 8192 | 24 |
| `expert_dim` | 8192 | 24 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 5 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 2048 | 2·n_kv·d_head (K와 V 합친 투영 폭) | experts, feed_forward |

## 레이어 구조

- layer 0: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 1: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 2: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 3: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 4: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 5: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 6: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 7: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 8: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 9: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 10: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 11: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 12: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 13: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 14: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 15: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 16: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 17: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 18: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 19: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 20: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 21: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 22: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 23: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 24: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 25: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 26: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 27: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 28: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 29: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 30: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 31: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 32: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 33: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 34: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 35: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 36: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 37: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 38: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 39: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 40: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 41: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 42: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 43: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 44: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 45: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 46: feed_forward, input_layernorm, post_attention_layernorm, self_attn
- layer 47: feed_forward, input_layernorm, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 48 == 48 |
| C2 | WARN | 3 trace clusters vs 2 config-schedule signatures ['layer_types', 'no_rope_layers'] -- review (mas... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=5120 in 48/48 layers |
| C6 | PASS | hidden_size=5120 (heuristic check, 3156 flagged) |
| C7 | PASS | GQA 40:8 (repeat factor 5) |
| C8 | WARN | MoE trace-verified [router_dim(E=128):ok, top_k(1):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=202048, tie_word_embeddings=False |
| C10 | PASS | all 507 params covered |
| C11 | PASS | 96 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 4076 unmapped rows, 30 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `meta-llama/Llama-4-Maverick-17B-128E` config.json @ `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2` (sha256 `72572339104b…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_


## 4. 검증 체크리스트 결과

```
# Extraction Report -- meta-llama/Llama-4-Maverick-17B-128E @ 10751cb97a4d7c90f7ed89196b98eb8220cfa1c2

C1   PASS   48 == 48
C2   WARN   3 trace clusters vs 2 config-schedule signatures ['layer_types', 'no_rope_layers'] -- review (mask-only heterogeneity is op-invisible)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 48/48 layers
C6   PASS   hidden_size=5120 (heuristic check, 3156 flagged)
C7   PASS   GQA 40:8 (repeat factor 5)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(1):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=202048, tie_word_embeddings=False
C10  PASS   all 507 params covered
C11  PASS   96 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4076 unmapped rows, 30 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.div.Tensor', 'aten.expand.default', 'aten.floor.default', 'aten.full_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              lift_fresh       [] -> []
  model                                              zeros            [] -> [B]
  model.rotary_emb                                   unsqueeze        [d_head/2] -> [B,d_head/2]
  model.rotary_emb                                   unsqueeze        [B,d_head/2] -> [B,d_head/2,B]
  model.rotary_emb                                   expand           [B,d_head/2,B] -> [B,d_head/2,B]
  model.rotary_emb                                   unsqueeze        [B,T] -> [B,B,T]
  model.rotary_emb                                   _to_copy         [B,B,T] -> [B,B,T]
  model.rotary_emb                                   view             [B,d_head/2,B] -> [B,d_head/2,B]
  model.rotary_emb                                   expand           [B,B,T] -> [B,B,T]
  model.rotary_emb                                   view             [B,B,T] -> [B,B,T]
  model.rotary_emb                                   batched_matmul   [B,d_head/2,B]*[B,B,T] -> [B,d_head/2,T]
  model.rotary_emb                                   _unsafe_view     [B,d_head/2,T] -> [B,d_head/2,T]
  model.rotary_emb                                   transpose        [B,d_head/2,T] -> [B,T,d_head/2]
  model.rotary_emb                                   ones_like        [B,T,d_head/2] -> [B,T,d_head/2]
  model.rotary_emb                                   polar            [B,T,d_head/2]*[B,T,d_head/2] -> [B,T,d_head/2]
  model.rotary_emb                                   elementwise_mul  [B,T,d_head/2] -> [B,T,d_head/2]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,B]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,B] -> [B,T,B]
  model.layers.N.input_layernorm                     rsqrt            [B,T,B] -> [B,T,B]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,B] -> [B,T,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [n_h*d_head,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,n_h*d_head]
  model.layers.N.self_attn.k_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [n_h*d_head,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,n_h*d_head]
  model.layers.N.self_attn.v_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           view             [B,T,n_h,d_head] -> [B,T,n_h,d_head/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,T,n_h,d_head/2,2] -> [B,T,n_h,d_head/2]
  model.layers.N.self_attn                           _to_copy         [B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv,d_head] -> [B,T,n_kv,d_head/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,T,n_kv,d_head/2,2] -> [B,T,n_kv,d_head/2]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_head/2] -> [B,T,B,d_head/2]
  model.layers.N.self_attn                           elementwise_mul  [B,T,n_h,d_head/2]*[B,T,B,d_head/2] -> [B,T,n_h,d_head/2]
  model.layers.N.self_attn                           view_as_real     [B,T,n_h,d_head/2] -> [B,T,n_h,d_head/2,2]
  model.layers.N.self_attn                           view             [B,T,n_h,d_head/2,2] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,T,n_kv,d_head/2]*[B,T,B,d_head/2] -> [B,T,n_kv,d_head/2]
  model.layers.N.self_attn                           view_as_real     [B,T,n_kv,d_head/2] -> [B,T,n_kv,d_head/2,2]
  model.layers.N.self_attn                           view             [B,T,n_kv,d_head/2,2] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [] -> []
  model.layers.N.self_attn                           concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           ones             [] -> [T,T]
  model.layers.N.self_attn                           tril             [T,T] -> [T,T]
  model.layers.N.self_attn                           scalar_tensor    [] -> []
  model.layers.N.self_attn                           where            [T,T]*[]*[] -> [T,T]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T,d_head] -> [B,n_kv,B,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,B,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,B]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,B] -> [B,T,B]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,B] -> [B,T,B]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,B] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.feed_forward.gate_proj              t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.gate_proj              view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.gate_proj              matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.gate_proj              _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.activation_fn          silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.up_proj                t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.up_proj                view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.up_proj                matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.up_proj                _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.down_proj              t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.down_proj              view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.feed_forward.down_proj              matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.feed_forward.down_proj              _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     view             [B,T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.feed_forward                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.router                 t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.feed_forward.router                 matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.feed_forward.router                 topk             [T,E] -> [T,B]*[T,B]
  model.layers.N.feed_forward.router                 full_like        [T,E] -> [T,E]
  model.layers.N.feed_forward.router                 scatter_         [T,E]*[T,B]*[T,B] -> [T,E]
  model.layers.N.feed_forward.router                 _to_copy         [T,E] -> [T,E]
  model.layers.N.feed_forward.router                 sigmoid          [T,E] -> [T,E]
  model.layers.N.feed_forward                        repeat           [T,d_model] -> [E*T,d_model]
  model.layers.N.feed_forward                        transpose        [T,E] -> [E,T]
  model.layers.N.feed_forward                        clone            [E,T] -> [E,T]
  model.layers.N.feed_forward                        _unsafe_view     [E,T] -> [E*T,B]
  model.layers.N.feed_forward                        elementwise_mul  [E*T,d_model]*[E*T,B] -> [E*T,d_model]
  model.layers.N.feed_forward.experts                view             [E*T,d_model] -> [E,T,d_model]
  model.layers.N.feed_forward.experts                batched_matmul   [E,T,d_model]*[E,d_model,d_ff] -> w=[E,d_model,d_ff] [E,T,d_ff]
  model.layers.N.feed_forward.experts                split            [E,T,d_ff] -> [E,T,d_moe]*[E,T,d_moe]
  model.layers.N.feed_forward.experts.act_fn         silu             [E,T,d_moe] -> [E,T,d_moe]
  model.layers.N.feed_forward.experts                elementwise_mul  [E,T,d_moe]*[E,T,d_moe] -> [E,T,d_moe]
  model.layers.N.feed_forward.experts                batched_matmul   [E,T,d_moe]*[E,d_moe,d_model] -> w=[E,d_moe,d_model] [E,T,d_model]
  model.layers.N.feed_forward.experts                view             [E,T,d_model] -> [E*T,d_model]
  model.layers.N.feed_forward.shared_expert.gate_proj t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.feed_forward.shared_expert.gate_proj matmul           [T,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [T,d_moe]
  model.layers.N.feed_forward.shared_expert.activation_fn silu             [T,d_moe] -> [T,d_moe]
  model.layers.N.feed_forward.shared_expert.up_proj  t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.feed_forward.shared_expert.up_proj  matmul           [T,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [T,d_moe]
  model.layers.N.feed_forward.shared_expert          elementwise_mul  [T,d_moe]*[T,d_moe] -> [T,d_moe]
  model.layers.N.feed_forward.shared_expert.down_proj t                [d_model,d_moe] -> w=[d_model,d_moe] [d_moe,d_model]
  model.layers.N.feed_forward.shared_expert.down_proj matmul           [T,d_moe]*[d_moe,d_model] -> w=[d_model,d_moe] [T,d_model]
  model.layers.N.feed_forward                        view             [E*T,d_model] -> [E,T,d_model]
  model.layers.N.feed_forward                        sum              [E,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward                        add_             [T,d_model]*[T,d_model] -> [T,d_model]
  model.layers.1                                     view             [T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     view             [B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn                           arange           [] -> [T]
  model.layers.N.self_attn                           elementwise_add  [T] -> [T]
  model.layers.N.self_attn                           div              [T] -> [T]
  model.layers.N.self_attn                           floor            [T] -> [T]
  model.layers.N.self_attn                           log1p            [T] -> [T]
  model.layers.N.self_attn                           elementwise_mul  [T] -> [T]
  model.layers.N.self_attn                           expand           [B,T,B,B] -> [B,T,B,B]
  model.layers.N.self_attn                           elementwise_mul  [B,T,n_h,d_head]*[B,T,B,B] -> [B,T,n_h,d_head]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     view             [T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     view             [B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     view             [T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     view             [B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     view             [T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     view             [B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     view             [T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.24                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.38                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.38                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.40                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.40                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.42                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.42                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.44                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.44                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.45                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.45                                    view             [T,d_model] -> [B,T,d_model]
  model.layers.46                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.46                                    view             [B,T,d_model] -> [B,T,d_model]
  model.layers.47                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.47                                    view             [T,d_model] -> [B,T,d_model]
  model.norm                                         _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.norm                                         pow              [B,T,d_model] -> [B,T,d_model]
  model.norm                                         mean             [B,T,d_model] -> [B,T,B]
  model.norm                                         elementwise_add  [B,T,B] -> [B,T,B]
  model.norm                                         rsqrt            [B,T,B] -> [B,T,B]
  model.norm                                         elementwise_mul  [B,T,d_model]*[B,T,B] -> [B,T,d_model]
  model.norm                                         elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
                                                     alias            [B,T,d_model] -> [B,T,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,T,d_model] -> [T,d_model]
  lm_head                                            matmul           [T,d_model]*[d_model,V] -> w=[V,d_model] [T,V]
  lm_head                                            _unsafe_view     [T,V] -> [B,T,V]
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,B] -> w=[V,d_model] [B,B,d_model]
  model                                              zeros            [] -> [B]
  model.rotary_emb                                   unsqueeze        [d_head/2] -> [B,d_head/2]
  model.rotary_emb                                   unsqueeze        [B,d_head/2] -> [B,d_head/2,B]
  model.rotary_emb                                   expand           [B,d_head/2,B] -> [B,d_head/2,B]
  model.rotary_emb                                   unsqueeze        [B,B] -> [B,B,B]
  model.rotary_emb                                   _to_copy         [B,B,B] -> [B,B,B]
  model.rotary_emb                                   view             [B,d_head/2,B] -> [B,d_head/2,B]
  model.rotary_emb                                   expand           [B,B,B] -> [B,B,B]
  model.rotary_emb                                   view             [B,B,B] -> [B,B,B]
  model.rotary_emb                                   batched_matmul   [B,d_head/2,B]*[B,B,B] -> [B,d_head/2,B]
  model.rotary_emb                                   _unsafe_view     [B,d_head/2,B] -> [B,d_head/2,B]
  model.rotary_emb                                   transpose        [B,d_head/2,B] -> [B,B,d_head/2]
  model.rotary_emb                                   ones_like        [B,B,d_head/2] -> [B,B,d_head/2]
  model.rotary_emb                                   polar            [B,B,d_head/2]*[B,B,d_head/2] -> [B,B,d_head/2]
  model.rotary_emb                                   elementwise_mul  [B,B,d_head/2] -> [B,B,d_head/2]
  model.layers.N.input_layernorm                     _to_copy         [B,B,d_model] -> [B,B,d_model]
  model.layers.N.input_layernorm                     pow              [B,B,d_model] -> [B,B,d_model]
  model.layers.N.input_layernorm                     mean             [B,B,d_model] -> [B,B,B]
  model.layers.N.input_layernorm                     elementwise_add  [B,B,B] -> [B,B,B]
  model.layers.N.input_layernorm                     rsqrt            [B,B,B] -> [B,B,B]
  model.layers.N.input_layernorm                     elementwise_mul  [B,B,d_model]*[B,B,B] -> [B,B,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,B,d_model]*[d_model] -> [B,B,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,B,d_model] -> [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*d_head] -> [B,B,n_h*d_head]
  model.layers.N.self_attn                           view             [B,B,n_h*d_head] -> [B,B,n_h,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [n_h*d_head,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,B,d_model] -> [B,n_h*d_head]
  model.layers.N.self_attn.k_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,B,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,B,n_kv*d_head] -> [B,B,n_kv,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [n_h*d_head,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,B,d_model] -> [B,n_h*d_head]
  model.layers.N.self_attn.v_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,B,n_kv*d_head]
  model.layers.N.self_attn                           transpose        [B,B,n_kv,d_head] -> [B,n_kv,B,d_head]
  model.layers.N.self_attn                           _to_copy         [B,B,n_h,d_head] -> [B,B,n_h,d_head]
  model.layers.N.self_attn                           view             [B,B,n_h,d_head] -> [B,B,n_h,d_head/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,B,n_h,d_head/2,2] -> [B,B,n_h,d_head/2]
  model.layers.N.self_attn                           _to_copy         [B,B,n_kv,d_head] -> [B,B,n_kv,d_head]
  model.layers.N.self_attn                           view             [B,B,n_kv,d_head] -> [B,B,n_kv,d_head/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,B,n_kv,d_head/2,2] -> [B,B,n_kv,d_head/2]
  model.layers.N.self_attn                           unsqueeze        [B,B,d_head/2] -> [B,B,B,d_head/2]
  model.layers.N.self_attn                           elementwise_mul  [B,B,n_h,d_head/2]*[B,B,B,d_head/2] -> [B,B,n_h,d_head/2]
  model.layers.N.self_attn                           view_as_real     [B,B,n_h,d_head/2] -> [B,B,n_h,d_head/2,2]
  model.layers.N.self_attn                           view             [B,B,n_h,d_head/2,2] -> [B,B,n_h,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,B,n_kv,d_head/2]*[B,B,B,d_head/2] -> [B,B,n_kv,d_head/2]
  model.layers.N.self_attn                           view_as_real     [B,B,n_kv,d_head/2] -> [B,B,n_kv,d_head/2,2]
  model.layers.N.self_attn                           view             [B,B,n_kv,d_head/2,2] -> [B,B,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,B,n_h,d_head] -> [B,n_h,B,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,B,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,B,d_head] -> [B,n_h,B,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,B,d_head] -> [B,n_h,B,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T+1,d_head] -> [B,n_kv,B,T+1,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,B,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,B,d_head] -> [B,n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,B,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,B,T+1] -> [B,n_h,B,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,B,T+1] -> [B,n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,B,T+1] -> [B,n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_head] -> [B,n_h,B,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,B,d_head] -> [B,B,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.o_proj                    view             [B,B,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,n_h*d_head] -> [B,B,n_h*d_head]
  model.layers.0                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,B,d_model] -> [B,B,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,B,d_model] -> [B,B,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,B,d_model] -> [B,B,B]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,B,B] -> [B,B,B]
  model.layers.N.post_attention_layernorm            rsqrt            [B,B,B] -> [B,B,B]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,B,d_model]*[B,B,B] -> [B,B,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,B,d_model]*[d_model] -> [B,B,d_model]
  model.layers.N.feed_forward.gate_proj              t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.gate_proj              view             [B,B,d_model] -> [B,d_model]
  model.layers.N.feed_forward.gate_proj              matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.gate_proj              _unsafe_view     [B,d_ff] -> [B,B,d_ff]
  model.layers.N.feed_forward.activation_fn          silu             [B,B,d_ff] -> [B,B,d_ff]
  model.layers.N.feed_forward.up_proj                t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.up_proj                view             [B,B,d_model] -> [B,d_model]
  model.layers.N.feed_forward.up_proj                matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.up_proj                _unsafe_view     [B,d_ff] -> [B,B,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,B,d_ff]*[B,B,d_ff] -> [B,B,d_ff]
  model.layers.N.feed_forward.down_proj              t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.down_proj              view             [B,B,d_ff] -> [B,d_ff]
  model.layers.N.feed_forward.down_proj              matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.feed_forward.down_proj              _unsafe_view     [B,d_model] -> [B,B,d_model]
  model.layers.0                                     view             [B,B,d_model] -> [B,B,d_model]
  model.layers.1                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.N.feed_forward                        view             [B,B,d_model] -> [B,d_model]
  model.layers.N.feed_forward.router                 t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.feed_forward.router                 matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.feed_forward.router                 topk             [B,E] -> [B,B]*[B,B]
  model.layers.N.feed_forward.router                 full_like        [B,E] -> [B,E]
  model.layers.N.feed_forward.router                 scatter_         [B,E]*[B,B]*[B,B] -> [B,E]
  model.layers.N.feed_forward.router                 _to_copy         [B,E] -> [B,E]
  model.layers.N.feed_forward.router                 sigmoid          [B,E] -> [B,E]
  model.layers.N.feed_forward                        repeat           [B,d_model] -> [E,d_model]
  model.layers.N.feed_forward                        transpose        [B,E] -> [E,B]
  model.layers.N.feed_forward                        view             [E,B] -> [E,B]
  model.layers.N.feed_forward                        elementwise_mul  [E,d_model]*[E,B] -> [E,d_model]
  model.layers.N.feed_forward.experts                view             [E,d_model] -> [E,B,d_model]
  model.layers.N.feed_forward.experts                batched_matmul   [E,B,d_model]*[E,d_model,d_ff] -> w=[E,d_model,d_ff] [E,B,d_ff]
  model.layers.N.feed_forward.experts                split            [E,B,d_ff] -> [E,B,d_moe]*[E,B,d_moe]
  model.layers.N.feed_forward.experts.act_fn         silu             [E,B,d_moe] -> [E,B,d_moe]
  model.layers.N.feed_forward.experts                elementwise_mul  [E,B,d_moe]*[E,B,d_moe] -> [E,B,d_moe]
  model.layers.N.feed_forward.experts                batched_matmul   [E,B,d_moe]*[E,d_moe,d_model] -> w=[E,d_moe,d_model] [E,B,d_model]
  model.layers.N.feed_forward.experts                view             [E,B,d_model] -> [E,d_model]
  model.layers.N.feed_forward.shared_expert.gate_proj t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.feed_forward.shared_expert.gate_proj matmul           [B,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [B,d_moe]
  model.layers.N.feed_forward.shared_expert.activation_fn silu             [B,d_moe] -> [B,d_moe]
  model.layers.N.feed_forward.shared_expert.up_proj  t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.feed_forward.shared_expert.up_proj  matmul           [B,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [B,d_moe]
  model.layers.N.feed_forward.shared_expert          elementwise_mul  [B,d_moe]*[B,d_moe] -> [B,d_moe]
  model.layers.N.feed_forward.shared_expert.down_proj t                [d_model,d_moe] -> w=[d_model,d_moe] [d_moe,d_model]
  model.layers.N.feed_forward.shared_expert.down_proj matmul           [B,d_moe]*[d_moe,d_model] -> w=[d_model,d_moe] [B,d_model]
  model.layers.N.feed_forward                        view             [E,d_model] -> [E,B,d_model]
  model.layers.N.feed_forward                        sum              [E,B,d_model] -> [B,d_model]
  model.layers.N.feed_forward                        add_             [B,d_model]*[B,d_model] -> [B,d_model]
  model.layers.1                                     view             [B,d_model] -> [B,B,d_model]
  model.layers.2                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.2                                     view             [B,B,d_model] -> [B,B,d_model]
  model.layers.N.self_attn                           arange           [] -> [B]
  model.layers.N.self_attn                           elementwise_add  [B] -> [B]
  model.layers.N.self_attn                           _to_copy         [B] -> [B]
  model.layers.N.self_attn                           div              [B] -> [B]
  model.layers.N.self_attn                           floor            [B] -> [B]
  model.layers.N.self_attn                           log1p            [B] -> [B]
  model.layers.N.self_attn                           elementwise_mul  [B] -> [B]
  model.layers.N.self_attn                           expand           [B,B,B,B] -> [B,B,B,B]
  model.layers.N.self_attn                           elementwise_mul  [B,B,n_h,d_head]*[B,B,B,B] -> [B,B,n_h,d_head]
  model.layers.3                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.3                                     view             [B,d_model] -> [B,B,d_model]
  model.layers.4                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.4                                     view             [B,B,d_model] -> [B,B,d_model]
  model.layers.5                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.5                                     view             [B,d_model] -> [B,B,d_model]
  model.layers.6                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.6                                     view             [B,B,d_model] -> [B,B,d_model]
  model.layers.7                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.7                                     view             [B,d_model] -> [B,B,d_model]
  model.layers.8                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.8                                     view             [B,B,d_model] -> [B,B,d_model]
  model.layers.9                                     elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.9                                     view             [B,d_model] -> [B,B,d_model]
  model.layers.10                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.10                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.11                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.11                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.12                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.12                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.13                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.13                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.14                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.14                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.15                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.15                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.16                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.16                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.17                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.17                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.18                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.18                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.19                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.19                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.20                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.20                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.21                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.21                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.22                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.22                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.23                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.23                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.24                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.24                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.25                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.25                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.26                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.26                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.27                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.27                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.28                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.28                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.29                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.29                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.30                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.30                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.31                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.31                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.32                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.32                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.33                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.33                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.34                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.34                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.35                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.35                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.36                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.36                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.37                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.37                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.38                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.38                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.39                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.39                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.40                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.40                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.41                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.41                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.42                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.42                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.43                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.43                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.44                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.44                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.45                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.45                                    view             [B,d_model] -> [B,B,d_model]
  model.layers.46                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.46                                    view             [B,B,d_model] -> [B,B,d_model]
  model.layers.47                                    elementwise_add  [B,B,d_model]*[B,B,d_model] -> [B,B,d_model]
  model.layers.47                                    view             [B,d_model] -> [B,B,d_model]
  model.norm                                         _to_copy         [B,B,d_model] -> [B,B,d_model]
  model.norm                                         pow              [B,B,d_model] -> [B,B,d_model]
  model.norm                                         mean             [B,B,d_model] -> [B,B,B]
  model.norm                                         elementwise_add  [B,B,B] -> [B,B,B]
  model.norm                                         rsqrt            [B,B,B] -> [B,B,B]
  model.norm                                         elementwise_mul  [B,B,d_model]*[B,B,B] -> [B,B,d_model]
  model.norm                                         elementwise_mul  [B,B,d_model]*[d_model] -> [B,B,d_model]
                                                     alias            [B,B,d_model] -> [B,B,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,B,d_model] -> [B,d_model]
  lm_head                                            matmul           [B,d_model]*[d_model,V] -> w=[V,d_model] [B,V]
  lm_head                                            _unsafe_view     [B,V] -> [B,B,V]
```

## 6. 이미 알려진 한계 — 다시 보고하지 않아도 됨

- **값이 같은 서로 다른 개념은 자동 판별 불가.** 예: gpt-oss는 expert 블록 **안**에서
  `d_model`=`d_moe`=2880, Zamba2는 `n_h*d_head`=`2*d_model`=4096. module_path만으로는 어느
  개념인지 못 가른다. 이건 이미 `rules/structures/`에 명시해 뒀으므로 다시 보고하지 않아도 된다.
- **이미 고쳐서 다시 보고하지 않아도 되는 것**(2026-07-30~31):
  `post_attention_layernorm`의 잔차폭(→ `d_model`), MoE 라우터 입력폭(→ `d_model`),
  gpt-oss sliding 레이어의 KV 상한(→ `w_local`)과 attention sink 표기(→ `n_sink`),
  Llama-3.1-405B의 토큰축(재추적으로 T 충돌 해소), DeepSeek-V4 압축기의 블록 축(→ `m_csa`/`m_hca`),
  한 shape에 `n_h`와 `n_kv`가 동시에 나오던 문제(게이트가 자동 검사한다).
- **`d_model` ↔ `n_h*d_head`가 같은 텐서에 다르게 붙는 경우**는 표준 트랜스포머에서 두 값이
  정의상 같기 때문이며, 둘 다 참인 이름이다. 강제 통일하면 오히려 정보가 사라진다.
- **Qwen3-Next의 3,7,9,11… 같은 작은 정수**는 DeltaNet 청크 스캔의 언롤된 루프 경계다.
  아키텍처 상수가 아니므로 심볼을 붙이면 거짓이 된다.


---

## 리뷰 요청

당신은 이 산출물이 **실제 모델 아키텍처를 정확히 기술하는지** 판정해야 합니다.
규칙 체크리스트는 이미 전부 통과한 상태입니다. 그러니 규칙이 못 잡는 것을 찾아주세요.

### 반드시 대조할 것
1. 해당 모델의 **공식 HF modeling 코드**와 config 클래스
2. **논문 / 기술 리포트**, 벤더 공식 블로그
3. **vLLM · SGLang 등 독립 서빙 구현**의 같은 모델 코드·해설
4. 신뢰도 높은 아키텍처 정리 자료

### 특히 봐야 할 것
- 심볼 이름이 **그 위치에서 실제로 의미하는 것과 맞는가**
  (값이 맞아도 개념이 틀릴 수 있음 — 이게 지금까지 나온 오류의 거의 전부였다)
- attention 계열 판정(MHA/GQA/MQA/MLA/…)이 실제 구현과 맞는가.
  **config 필드를 그대로 믿지 말 것** — 필드가 있어도 실제 동작이 다를 수 있다
  (Falcon은 `num_kv_heads=71`이지만 `multi_query=True`라 실제 KV head는 1개였다)
- KV cache 계산의 **전제**가 맞는가 (어느 레이어가 캐시를 갖는지, K와 V가 별개인지)
- config에 없는데 코드에 하드코딩된 구조가 누락되지 않았는가
  (Llama-4는 shared expert 개수 필드가 없고 코드에 1개로 고정돼 있다)
- 이 아키텍처의 **핵심 특징 중 산출물에 아예 안 나타난 것**이 있는가
- **decode 표(5-2)를 반드시 보세요.** prefill에는 없는 축이 거기 있습니다 —
  sliding 레이어의 KV 상한, 캐시 길이, attention sink가 붙는 score 폭.
  실제로 이 표가 패킷에 없던 동안 gpt-oss의 sliding 컨텍스트 오라벨이 그대로 남아 있었습니다.
- **모듈 이름이 "무엇을 계산하는가"가 아니라 "블록 안 어디인가"를 뜻하는 곳**을 의심하세요.
  지금까지 나온 오류의 다수가 여기서 나왔습니다 — `post_attention_layernorm`은 attention이
  아니라 그 뒤의 잔차 정규화이고, `mlp.router`는 FFN 내부가 아니라 잔차를 읽는 라우터입니다.
- **같은 (모듈, op)인데 shape 표기가 갈리는 줄**을 찾으세요. 표본은 그 축을 일부러 접지
  않았습니다 — 라벨 오류는 정의상 거기서 드러납니다.

### 출력 형식 (반드시 지킬 것)

각 지적은 아래 표 형태로. **관찰과 가설을 반드시 분리**하세요.

| # | 관찰(사실) | 근거 | 내 가설(원인) | 확신도 | 검증 방법 |
|---|---|---|---|---|---|

- **관찰**: 패킷에서 직접 인용. "X라고 적혀 있다"
- **근거**: 공식 소스의 **파일명 + 함수/클래스명 + 인용문**, 또는 URL.
  `op_id`는 근거로 쓰지 마세요 — 재트레이싱하면 번호가 바뀝니다.
  대신 `module_path`와 shape 내용으로 지목하세요.
- **내 가설**: 왜 그렇게 됐다고 보는지. **틀려도 됩니다. 다만 관찰과 섞지 마세요.**
- **확신도**: 확실 / 아마도 / 추측
- **검증 방법**: 우리가 이 주장을 어떻게 확인하면 되는지 (구체적으로)

확실하지 않으면 "확실"이라고 쓰지 마세요. **틀린 지적보다 놓친 지적이 낫습니다** —
틀린 지적을 검증하는 비용이 더 큽니다.

