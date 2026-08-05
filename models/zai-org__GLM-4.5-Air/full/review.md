# 리뷰 패킷 — zai-org/GLM-4.5-Air

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `a24ceef6ce4f3536971efe9b778bdaa1bab18daa` / 트레이스 seq_len(T) = 16
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
  L            = 46
  d_model      = 4096
  n_h          = 96
  n_kv         = 8
  d_head       = 128
  d_ff         = 10944
  V            = 151552
  ctx          = 131072
  E            = 128
  E_shared     = 1
  k            = 8
  n_grp        = 1
  k_grp        = 1
  d_moe        = 1408
  w_local      = None
  n_sink       = None
  layer_sched  = None
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
  d_chunk      = None
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

# Model Summary -- zai-org/GLM-4.5-Air

## 기본 정보

- revision: `a24ceef6ce4f3536971efe9b778bdaa1bab18daa`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 106.85B total, 13.42B active (12.6% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-07-20  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 46× GQA  (FFN: 1 dense + 45 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 184.0 KiB (High) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=128, top-8, +1 shared, sigmoid gating/aux-loss-free); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, sigmoid-gating, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `glm4_moe` |
| attention | GQA — 96 query : 8 kv heads (repeat 12), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=1000000) |
| FFN | MoE — 128 routed experts, top-8 + 1 shared, expert intermediate 1408, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; all 46 layers ⇒ 94208 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 46 |
| d_model | 4096 |
| n_h | 96 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 10944 |
| V | 151552 |
| ctx | 131072 |
| E | 128 |
| E_shared | 1 |
| k | 8 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 1408 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
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
| d_chunk | —  _(해당 없음: 이 모델은 `ssm_chunk` 계열 구조를 쓰지 않음)_ |
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

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **158,209개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 48,049 | 30.37% |
| 스코프 없는 심볼 | 46,199 | 29.20% |
| 런타임 축 (B/T/1) | 45,442 | 28.72% |
| 이 모듈 스코프의 유도식 | 12,935 | 8.18% |
| 휴리스틱: 심볼+1 | 2,208 | 1.40% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,840 | 1.16% |
| 휴리스틱: 심볼의 배수 | 947 | 0.60% |
| 이름 없음 (정수 유지) | 589 | 0.37% |

등록된 규칙 **152,625축**, 약한 근거 1,840축, 휴리스틱 **3,155축 (1.99%)**, 이름 없음 589축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.0.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.1.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.2.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.3.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.4.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.5.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.6.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.7.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.8.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.9.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.10.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |
| `model.layers.11.self_attn` | `T+1` | 휴리스틱: 심볼+1 | 48 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 12 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 32 | d_rope/2 (partial_rotary_factor 기준 rotate_half 분할 축) | rotary_emb, self_attn |
| 64 | d_head − d_rope (부분 RoPE 비회전 통과분, partial_rotary_factor 기준) | rotary_emb, self_attn |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 12288 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_proj, self_attn |

## 레이어 구조

- layer 0: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 1-45: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 3개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 46 == 46 |
| C2 | WARN | 2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 46/46 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 5800 flagged) |
| C7 | PASS | GQA 96:8 (repeat factor 12) |
| C8 | WARN | MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=151552, tie_word_embeddings=False |
| C10 | PASS | all 690 params covered |
| C11 | PASS | 277 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=16 >= required=16 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 5267 unmapped rows, 31 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `zai-org/GLM-4.5-Air` config.json @ `a24ceef6ce4f3536971efe9b778bdaa1bab18daa` (sha256 `74819b0e4573…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_


## 4. 검증 체크리스트 결과

```
# Extraction Report -- zai-org/GLM-4.5-Air @ a24ceef6ce4f3536971efe9b778bdaa1bab18daa

C1   PASS   46 == 46
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 46/46 layers
C6   PASS   hidden_size=4096 (heuristic check, 5800 flagged)
C7   PASS   GQA 96:8 (repeat factor 12)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151552, tie_word_embeddings=False
C10  PASS   all 690 params covered
C11  PASS   277 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   5267 unmapped rows, 31 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [B,T] -> [B,1,T]
  model.rotary_emb                                   _to_copy         [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,1,T] -> [B,1,T]
  model.rotary_emb                                   batched_matmul   [B,d_rope/2,1]*[B,1,T] -> [B,d_rope/2,T]
  model.rotary_emb                                   _unsafe_view     [B,d_rope/2,T] -> [B,d_rope/2,T]
  model.rotary_emb                                   transpose        [B,d_rope/2,T] -> [B,T,d_rope/2]
  model.rotary_emb                                   concat           [B,T,d_rope/2]*[B,T,d_rope/2] -> [B,T,d_head-d_rope]
  model.rotary_emb                                   cos              [B,T,d_head-d_rope] -> [B,T,d_head-d_rope]
  model.rotary_emb                                   elementwise_mul  [B,T,d_head-d_rope] -> [B,T,d_head-d_rope]
  model.rotary_emb                                   sin              [B,T,d_head-d_rope] -> [B,T,d_head-d_rope]
  model.rotary_emb                                   _to_copy         [B,T,d_head-d_rope] -> [B,T,d_head-d_rope]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    linear           [n_h*d_head]*[T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    linear           [n_kv*d_head]*[T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    linear           [n_kv*d_head]*[T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_head-d_rope] -> [B,1,T,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head-d_rope]*[B,1,T,d_head-d_rope] -> [B,n_h,T,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head-d_rope] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_rope/2] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_rope/2]*[B,n_h,T,d_rope/2] -> [B,n_h,T,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_head-d_rope]*[B,n_h,T,d_head-d_rope] -> [B,n_h,T,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head-d_rope]*[B,1,T,d_head-d_rope] -> [B,n_kv,T,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head-d_rope] -> [B,n_kv,T,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_rope/2] -> [B,n_kv,T,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_rope/2]*[B,n_kv,T,d_rope/2] -> [B,n_kv,T,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,T,d_head-d_rope]*[B,n_kv,T,d_head-d_rope] -> [B,n_kv,T,d_head-d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head-d_rope]*[B,n_h,T,d_head-d_rope] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head-d_rope]*[B,n_kv,T,d_head-d_rope] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           ones             [] -> [T,T]
  model.layers.N.self_attn                           tril             [T,T] -> [T,T]
  model.layers.N.self_attn                           scalar_tensor    [] -> []
  model.layers.N.self_attn                           where            [T,T]*[]*[] -> [T,T]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T,d_head] -> [B,n_kv,1,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.act_fn                          silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.up_proj                         t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.up_proj                         view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.up_proj                         matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.mlp.up_proj                         _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mlp.gate                            view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mlp.gate                            sigmoid          [T,E] -> [T,E]
  model.layers.N.mlp.gate                            elementwise_add  [T,E]*[E] -> [T,E]
  model.layers.N.mlp.gate                            view             [T,E] -> [T,1,E]
  model.layers.N.mlp.gate                            topk             [T,1,E] -> [T,1,2]*[T,1,2]
  model.layers.N.mlp.gate                            sum              [T,1,2] -> [T,1]
  model.layers.N.mlp.gate                            topk             [T,1] -> [T,1]*[T,1]
  model.layers.N.mlp.gate                            zeros_like       [T,1] -> [T,1]
  model.layers.N.mlp.gate                            scatter_         [T,1]*[T,1] -> [T,1]
  model.layers.N.mlp.gate                            unsqueeze        [T,1] -> [T,1,1]
  model.layers.N.mlp.gate                            expand           [T,1,1] -> [T,1,E]
  model.layers.N.mlp.gate                            view             [T,1,E] -> [T,E]
  model.layers.N.mlp.gate                            _to_copy         [T,E] -> [T,E]
  model.layers.N.mlp.gate                            bitwise_not      [T,E] -> [T,E]
  model.layers.N.mlp.gate                            masked_fill      [T,E]*[T,E] -> [T,E]
  model.layers.N.mlp.gate                            topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mlp.gate                            gather           [T,E]*[T,k] -> [T,k]
  model.layers.N.mlp.gate                            sum              [T,k] -> [T,1]
  model.layers.N.mlp.gate                            elementwise_add  [T,1] -> [T,1]
  model.layers.N.mlp.gate                            div_             [T,k]*[T,1] -> [T,k]
  model.layers.N.mlp.gate                            elementwise_mul  [T,k] -> [T,k]
  model.layers.N.mlp                                 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.experts                         view             [T,k] -> [E]
  model.layers.N.mlp.experts                         sort             [E] -> [E]*[E]
  model.layers.N.mlp.experts                         floor_divide     [E] -> [E]
  model.layers.N.mlp.experts                         index            [T,d_model]*[E] -> [E,d_model]
  model.layers.N.mlp.experts                         index            [E]*[E] -> [E]
  model.layers.N.mlp.experts                         _to_copy         [E] -> [E]
  model.layers.N.mlp.experts                         histc            [E] -> [E]
  model.layers.N.mlp.experts                         cumsum           [E] -> [E]
  model.layers.N.mlp.experts                         ge               [E] -> [E]
  model.layers.N.mlp.experts                         unsqueeze        [E] -> [E,B]
  model.layers.N.mlp.experts                         clamp_           [E] -> [E]
  model.layers.N.mlp.experts                         masked_fill_     [E,d_model]*[E,B] -> [E,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [E,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [E,2*d_moe]
  model.layers.N.mlp.experts                         split            [E,2*d_moe] -> [E,d_moe]*[E,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [E,d_moe] -> [E,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [E,d_moe]*[E,d_moe] -> [E,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [E,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [E,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [E,d_model]*[E,B] -> [E,d_model]
  model.layers.N.mlp.experts                         empty_like       [E] -> [E]
  model.layers.N.mlp.experts                         arange           [] -> [E]
  model.layers.N.mlp.experts                         index_put_       [E]*[E]*[E] -> [E]
  model.layers.N.mlp.experts                         index            [E,d_model]*[E] -> [E,d_model]
  model.layers.N.mlp.experts                         view             [E,d_model] -> [T,k,d_model]
  model.layers.N.mlp.experts                         sum              [T,k,d_model] -> [T,d_model]
  model.layers.N.mlp.experts                         _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mlp                                 view             [T,d_model] -> [B,T,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [T,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [T,d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        _unsafe_view     [T,d_moe] -> [B,T,d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [B,T,d_moe] -> [B,T,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [T,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [T,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          _unsafe_view     [T,d_moe] -> [B,T,d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [B,T,d_moe]*[B,T,d_moe] -> [B,T,d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,d_moe] -> w=[d_model,d_moe] [d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        view             [B,T,d_moe] -> [T,d_moe]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [T,d_moe]*[d_moe,d_model] -> w=[d_model,d_moe] [T,d_model]
  model.layers.N.mlp.shared_experts.down_proj        _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.38                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.40                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.42                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.44                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.45                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.norm                                         _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.norm                                         pow              [B,T,d_model] -> [B,T,d_model]
  model.norm                                         mean             [B,T,d_model] -> [B,T,1]
  model.norm                                         elementwise_add  [B,T,1] -> [B,T,1]
  model.norm                                         rsqrt            [B,T,1] -> [B,T,1]
  model.norm                                         elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.embed_tokens                                 embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [B,1] -> [B,1,1]
  model.rotary_emb                                   _to_copy         [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,1,1] -> [B,1,1]
  model.rotary_emb                                   batched_matmul   [B,d_rope/2,1]*[B,1,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   _unsafe_view     [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   transpose        [B,d_rope/2,1] -> [B,1,d_rope/2]
  model.rotary_emb                                   concat           [B,1,d_rope/2]*[B,1,d_rope/2] -> [B,1,d_head-d_rope]
  model.rotary_emb                                   cos              [B,1,d_head-d_rope] -> [B,1,d_head-d_rope]
  model.rotary_emb                                   elementwise_mul  [B,1,d_head-d_rope] -> [B,1,d_head-d_rope]
  model.rotary_emb                                   sin              [B,1,d_head-d_rope] -> [B,1,d_head-d_rope]
  model.rotary_emb                                   _to_copy         [B,1,d_head-d_rope] -> [B,1,d_head-d_rope]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    linear           [n_h*d_head]*[B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    linear           [n_kv*d_head]*[B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    linear           [n_kv*d_head]*[B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_head-d_rope] -> [B,1,1,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head-d_rope]*[B,1,1,d_head-d_rope] -> [B,n_h,1,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head-d_rope] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_rope/2] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_rope/2]*[B,n_h,1,d_rope/2] -> [B,n_h,1,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_head-d_rope]*[B,n_h,1,d_head-d_rope] -> [B,n_h,1,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head-d_rope]*[B,1,1,d_head-d_rope] -> [B,n_kv,1,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head-d_rope] -> [B,n_kv,1,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_rope/2] -> [B,n_kv,1,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_rope/2]*[B,n_kv,1,d_rope/2] -> [B,n_kv,1,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,1,d_head-d_rope]*[B,n_kv,1,d_head-d_rope] -> [B,n_kv,1,d_head-d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head-d_rope]*[B,n_h,1,d_head-d_rope] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_head-d_rope]*[B,n_kv,1,d_head-d_rope] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T+1,d_head] -> [B,n_kv,1,T+1,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.act_fn                          silu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.up_proj                         t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.up_proj                         view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.up_proj                         matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.mlp.up_proj                         _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mlp.gate                            view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mlp.gate                            sigmoid          [B,E] -> [B,E]
  model.layers.N.mlp.gate                            elementwise_add  [B,E]*[E] -> [B,E]
  model.layers.N.mlp.gate                            view             [B,E] -> [B,1,E]
  model.layers.N.mlp.gate                            topk             [B,1,E] -> [B,1,2]*[B,1,2]
  model.layers.N.mlp.gate                            sum              [B,1,2] -> [B,1]
  model.layers.N.mlp.gate                            topk             [B,1] -> [B,1]*[B,1]
  model.layers.N.mlp.gate                            zeros_like       [B,1] -> [B,1]
  model.layers.N.mlp.gate                            scatter_         [B,1]*[B,1] -> [B,1]
  model.layers.N.mlp.gate                            unsqueeze        [B,1] -> [B,1,1]
  model.layers.N.mlp.gate                            expand           [B,1,1] -> [B,1,E]
  model.layers.N.mlp.gate                            view             [B,1,E] -> [B,E]
  model.layers.N.mlp.gate                            _to_copy         [B,E] -> [B,E]
  model.layers.N.mlp.gate                            bitwise_not      [B,E] -> [B,E]
  model.layers.N.mlp.gate                            masked_fill      [B,E]*[B,E] -> [B,E]
  model.layers.N.mlp.gate                            topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mlp.gate                            gather           [B,E]*[B,k] -> [B,k]
  model.layers.N.mlp.gate                            sum              [B,k] -> [B,1]
  model.layers.N.mlp.gate                            elementwise_add  [B,1] -> [B,1]
  model.layers.N.mlp.gate                            div_             [B,k]*[B,1] -> [B,k]
  model.layers.N.mlp.gate                            elementwise_mul  [B,k] -> [B,k]
  model.layers.N.mlp                                 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.experts                         view             [B,k] -> [k]
  model.layers.N.mlp.experts                         sort             [k] -> [k]*[k]
  model.layers.N.mlp.experts                         floor_divide     [k] -> [k]
  model.layers.N.mlp.experts                         index            [B,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         index            [k]*[k] -> [k]
  model.layers.N.mlp.experts                         _to_copy         [k] -> [k]
  model.layers.N.mlp.experts                         histc            [k] -> [E]
  model.layers.N.mlp.experts                         cumsum           [E] -> [E]
  model.layers.N.mlp.experts                         ge               [k] -> [k]
  model.layers.N.mlp.experts                         unsqueeze        [k] -> [k,B]
  model.layers.N.mlp.experts                         clamp_           [k] -> [k]
  model.layers.N.mlp.experts                         masked_fill_     [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k,2*d_moe]
  model.layers.N.mlp.experts                         split            [k,2*d_moe] -> [k,d_moe]*[k,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_moe]*[k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         empty_like       [k] -> [k]
  model.layers.N.mlp.experts                         arange           [] -> [k]
  model.layers.N.mlp.experts                         index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.mlp.experts                         index            [k,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         view             [k,d_model] -> [B,k,d_model]
  model.layers.N.mlp.experts                         sum              [B,k,d_model] -> [B,d_model]
  model.layers.N.mlp.experts                         _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mlp                                 view             [B,d_model] -> [B,1,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [B,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [B,d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        _unsafe_view     [B,d_moe] -> [B,1,d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [B,1,d_moe] -> [B,1,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [d_moe,d_model] -> w=[d_moe,d_model] [d_model,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [B,d_model]*[d_model,d_moe] -> w=[d_moe,d_model] [B,d_moe]
  model.layers.N.mlp.shared_experts.up_proj          _unsafe_view     [B,d_moe] -> [B,1,d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [B,1,d_moe]*[B,1,d_moe] -> [B,1,d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,d_moe] -> w=[d_model,d_moe] [d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        view             [B,1,d_moe] -> [B,d_moe]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [B,d_moe]*[d_moe,d_model] -> w=[d_model,d_moe] [B,d_model]
  model.layers.N.mlp.shared_experts.down_proj        _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.11                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.17                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.23                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.24                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.35                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.38                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.39                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.40                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.41                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.42                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.43                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.44                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.45                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.norm                                         _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.norm                                         pow              [B,1,d_model] -> [B,1,d_model]
  model.norm                                         mean             [B,1,d_model] -> [B,1,1]
  model.norm                                         elementwise_add  [B,1,1] -> [B,1,1]
  model.norm                                         rsqrt            [B,1,1] -> [B,1,1]
  model.norm                                         elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
                                                     alias            [B,1,d_model] -> [B,1,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,1,d_model] -> [B,d_model]
  lm_head                                            matmul           [B,d_model]*[d_model,V] -> w=[V,d_model] [B,V]
  lm_head                                            _unsafe_view     [B,V] -> [B,1,V]
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

