# Model Summary -- moonshotai/Kimi-K3

## 기본 정보

- revision: `a590ce090cb049c93a33dfe8c208ec652aa20503`
- capture backend: fake (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 320
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 2779.48B total (dense) |
| 2 | Context (tokens) | 1,048,576  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-06-13  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MLA |
| 6 | LAYER MIX | 93× MLA  (FFN: 1 dense + 92 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 104.6 KiB (Moderate) |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=896, top-16, +2 shared, sigmoid gating/aux-loss-free); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, RoPE, MLA, MoE, shared expert, sigmoid-gating, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `kimi_linear` |
| attention | MLA — KV latent compression (kv_lora_rank=512, q_lora_rank=1536); 헤드 q/k = nope(128)+rope(64)=192, v=128, n_h=96 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000.0) |
| FFN | MoE — 896 routed experts, top-16 + 2 shared, expert intermediate 3072, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | compressed MLA latent ≈ kv_lora_rank=512 (+decoupled RoPE dim) / token / layer |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 93 |
| d_model | 7168 |
| n_h | 96 |
| n_kv | 96 |
| d_head | 74 |
| d_ff | 33792 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 163840 |
| ctx | 1048576 |
| E | 896 |
| E_shared | 2 |
| k | 16 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | 1 |
| d_moe | 3072 |
| d_moe_lat | 3584 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
| c_kv | 512 |
| d_nope | 128 |
| d_v | 128 |
| c_q | 1536 |
| d_rope | 64 |
| n_h_kda | 96 |
| d_head_kda | 128 |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | 12 |
| d_state | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_g_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_h_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| d_chunk | 64 |
| d_head_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| d_conv | 4 |
| n_mem | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| r_lora | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| d_attn | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| n_h_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| n_h_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_conv_lin | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **10,965,681개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 6,389,105 | 58.26% |
| 런타임 축 (B/T/1) | 3,344,497 | 30.50% |
| 이름 없음 (정수 유지) | 902,505 | 8.23% |
| 스코프 없는 심볼 | 133,503 | 1.22% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 121,326 | 1.11% |
| 이 모듈 스코프의 유도식 | 54,689 | 0.50% |
| 휴리스틱: 심볼의 배수 | 18,676 | 0.17% |
| 휴리스틱: 심볼의 절반 | 1,380 | 0.01% |

등록된 규칙 **9,921,794축**, 약한 근거 121,326축, 휴리스틱 **20,056축 (0.18%)**, 이름 없음 902,505축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.1.block_sparse_moe.experts.0.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.1.block_sparse_moe.experts.1.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.1.block_sparse_moe.experts.2.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.1.block_sparse_moe.experts.3.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.2.block_sparse_moe.experts.0.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.2.block_sparse_moe.experts.1.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.2.block_sparse_moe.experts.2.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.2.block_sparse_moe.experts.3.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.3.block_sparse_moe.experts.0.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.3.block_sparse_moe.experts.1.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.3.block_sparse_moe.experts.2.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |
| `model.layers.3.block_sparse_moe.experts.3.act_fn` | `2*E_shared` | 휴리스틱: 심볼의 배수 | 30 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 5 | d_conv+1 (decode 의 conv 캐시 — 캐시 d_conv 개 + 새 토큰 1개) | 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, self_attn |
| 10 | d_head − d_rope (부분 RoPE 비회전 통과분) | self_attn |
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | self_attn |
| 37 | d_head/2 (RoPE rotate_half 분할 축) | self_attn |
| 192 | d_nope + d_rope (MLA q/k head 폭) | self_attn |
| 256 | d_nope+d_v | self_attn |
| 323 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv, k_conv1d, q_conv1d, v_conv1d |
| 576 | c_kv+d_rope (MLA kv_a_proj_with_mqa 출력) | kv_a_proj_with_mqa, self_attn |
| 5120 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | block_sparse_moe |
| 6144 | n_h·d_rope | 0, 1, 2, 3, act_fn, down_proj, gate_proj, shared_experts, up_proj |
| 12288 | n_h·d_v (attention 출력, o_proj 직전) | act_fn, conv, f_b_proj, g_proj, k_conv1d, k_proj, o_proj, q_conv1d, q_proj, self_attn, shared_experts, v_conv1d, v_proj |
| 18432 | n_h·(d_nope+d_rope) (MLA q_b_proj 출력) | q_b_proj, self_attn |
| 24576 | n_h·(d_nope+d_v) (MLA kv_b_proj 출력) | kv_b_proj, self_attn |
| 67584 | 2·d_ff (dense FFN gate+up 융합 투영 폭) | act_fn, mlp |
| 480 | **미해결 — 아래 Tier 3 확인 필요** | self_attn |
| 1280 | **미해결 — 아래 Tier 3 확인 필요** | 0, 1, 2, 3, act_fn, block_sparse_moe, w1, w2, w3 |

### ⚠ 미해결 유도 상수 — 신규 모듈 조사 필요 (Tier 3)

아래 2개 값은 `rules/derived_dims.yaml`의 어떤 식으로도 설명되지 않는다. 거의 항상 **아직 조사하지 않은 모듈**이 있다는 뜻이다. `02-new-module-handling.md`의 「신규 모듈 조사 절차」대로 1차 소스(현재 실행 중인 modeling 코드) → 독립 서빙 구현(vLLM/SGLang/TensorRT-LLM) → 공식 문서·논문 → 아키텍처 갤러리 순으로 확인한 뒤, `rules/symbols.yaml`(별칭) 또는 `rules/derived_dims.yaml`(식)에 **출처와 함께** 등록할 것. 확인되지 않으면 추측해서 채우지 말고 사람에게 확인을 요청한다(P1).

| 값 | 나타나는 모듈 | 조사 착안점 |
|---|---|---|
| 480 | self_attn | 해당 모듈의 `__init__` 투영 폭과 forward의 concat/slice 축을 config 필드 조합으로 역산 |
| 1280 | 0, 1, 2, 3, act_fn, block_sparse_moe, w1, w2, w3 | 해당 모듈의 `__init__` 투영 폭과 forward의 concat/slice 축을 config 필드 조합으로 역산 |

## 레이어 구조

- layer 0: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 1-2: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 3: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 4-6: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 7: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 8-10: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 11: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 12: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 13-14: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 15: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 16-18: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 19: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 20-22: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 23: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 24: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 25-26: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 27: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 28-30: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 31: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 32-34: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 35: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 36: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 37-38: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 39: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 40-42: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 43: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 44-46: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 47: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 48: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 49-50: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 51: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 52-54: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 55: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 56-58: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 59: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 60: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 61-62: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 63: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 64-66: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 67: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 68-70: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 71: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 72: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 73-74: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 75: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 76-78: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 79: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 80-82: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 83: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 84: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 85-86: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 87: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 88-90: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn
- layer 91-92: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: FAIL** (WARN 3개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 93 == 93 |
| C2 | WARN | 4 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=7168 in 93/93 layers |
| C6 | PASS | hidden_size=7168 (heuristic check, 610829 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | WARN | MoE trace-verified [router_dim(E=896):ok, top_k(None):n/a, expert_weight:grouped]; routed-token c... |
| C9 | PASS | vocab_size=163840, tie_word_embeddings=False |
| C10 | FAIL | 2 param(s) with no contributing op, e.g. ['model.layers.0.self_attention_res_norm.weight', 'model... |
| C11 | PASS | 843 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=320 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 526420 unmapped rows, 43 distinct raw ops: ['aten._local_scalar_dense.default', 'aten._to_copy.de... |
| C17 | WARN | 미해결 유도 상수 2개 [480, 1280] -- rules/derived_dims.yaml에 식+출처 등록 필요; 남은 축별 안건은 models/<model>/researc... |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `moonshotai/Kimi-K3` config.json @ `a590ce090cb049c93a33dfe8c208ec652aa20503` (sha256 `711d6a903faf…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=320 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.
