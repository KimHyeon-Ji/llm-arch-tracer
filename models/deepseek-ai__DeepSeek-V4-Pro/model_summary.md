# Model Summary -- deepseek-ai/DeepSeek-V4-Pro

## 기본 정보

- revision: `b5968e9190ef611bbf34a7229255be88a0e937c1`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 2048
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 1573B total, 49.78B active (3.2% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 1,048,576  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-04-22  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MQA + HCA/CSA |
| 6 | LAYER MIX | 31× heavily_compressed_attention, 30× compressed_sparse_attention  (FFN: 61× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 7.7 KiB (Very low)  _(증가하는 압축 엔트리만 계산, K==V 단일 텐서; 제외: 고정 크기 sliding 버퍼(window 128, 전 61층) / Lightning Indexer 캐시(+1.88 KiB/token))_ |
| 8 | KEY DETAIL | MQA + HCA/CSA attention; Sparse MoE (E=384, top-6, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, MQA, HCA, CSA, mHC, MoE, shared expert, sigmoid-gating, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `deepseek_v4` |
| attention | MQA — 128 query heads, 1 kv head, d_head=512; sliding window 128 + 블록 압축 분기(HCA m=128, CSA m=4); sliding window는 전 레이어 적용 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | MoE — 384 routed experts, top-6 + 1 shared, expert intermediate 3072, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 블록 압축 — 압축 레이어당 d_head/m = 512/m elems / token (HCA m=128, CSA m=4), K==V 단일 텐서 ⇒ 7,928 B/token 전체 (7.74 KiB). sliding 분기는 window=128로 상한이 있어 컨텍스트에 따라 증가하지 않음 |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 61 |
| d_model | 7168 |
| n_h | 128 |
| n_kv | 1 |
| d_head | 512 |
| d_ff | 3072 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 129280 |
| ctx | 1048576 |
| E | 384 |
| E_shared | 1 |
| k | 6 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 3072 |
| w_local | 128 |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 31× heavily_compressed_attention, 30× compressed_sparse_attention (총 61층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | 1536 |
| d_rope | 64 |
| m_csa | 4 |
| m_hca | 128 |
| g_o | 16 |
| d_g | 1024 |
| n_h_I | 64 |
| c_I | 128 |
| k_I | 1024 |
| n_hc | 4 |
| t_sinkhorn | 20 |
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

shape 축 **1,021,289개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 422,808 | 41.40% |
| 이 모듈 스코프의 심볼 | 266,165 | 26.06% |
| 스코프 없는 심볼 | 184,517 | 18.07% |
| 이 모듈 스코프의 유도식 | 86,460 | 8.47% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 50,652 | 4.96% |
| 이름 없음 (정수 유지) | 9,426 | 0.92% |
| 휴리스틱: 심볼의 배수 | 1,261 | 0.12% |

등록된 규칙 **959,950축**, 약한 근거 50,652축, 휴리스틱 **1,261축 (0.12%)**, 이름 없음 9,426축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.2.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.4.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.6.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.8.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.10.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.12.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.14.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.16.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.18.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.20.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.22.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |
| `model.layers.24.self_attn.compressor` | `2*m_csa` | 휴리스틱: 심볼의 배수 | 40 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 8 | 2·m_csa (Indexer 겹침 창 슬롯 수: Ca⊕Cb) | compressor, indexer |
| 24 | (2+n_hc)·n_hc (mHC 게이트 파라미터 수: pre n_hc + post n_hc + comb n_hc²) | attn_hc, ffn_hc |
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | compressor, indexer, rotary_emb, self_attn |
| 127 | w_local − 1 (sliding window mask 밴드 폭) | self_attn |
| 256 | 2·c^I (Indexer kv_proj / gate_proj 폭: Ca⊕Cb 겹침 레이아웃) | gate_proj, indexer, kv_proj |
| 448 | d_head − d_rope (부분 RoPE 비회전 통과분) | compressor, self_attn |
| 511 | T/m_csa − 1 (CSA Ca/Cb 겹침 shift: 이전 윈도우 기여분 슬라이스) | compressor, indexer |
| 513 | T/m_csa + 1 (CSA block-bias 버퍼 = 압축 엔트리 수 + 무효 인덱스 슬롯 1) | compressor |
| 2064 | T + T/m_hca (HCA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 2065 | T + T/m_hca + 1 (HCA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 2560 | T + T/m_csa (CSA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 2561 | T + T/m_csa + 1 (CSA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 4096 | n_h·d_head/g_o (grouped output projection 그룹당 입력 폭) | o_a_proj, self_attn |
| 6144 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |
| 8192 | n_h^I·c^I (Lightning Indexer 쿼리 투영 폭) | indexer, q_b_proj |
| 12288 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 16384 | g_o·d_g (grouped output projection 합친 폭 → o_b_proj 입력) | o_a_proj, o_b_proj, self_attn |
| 28672 | n_hc·d_model (mHC: n_hc개 잔차 스트림을 편 폭) | attn_hc, ffn_hc, hc_head, input_norm |
| 65536 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | q_b_proj, self_attn |

## 레이어 구조

- layer 0-1: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 2: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 3: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 4: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 5: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 6: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 7: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 8: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 9: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 10: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 11: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 12: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 13: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 14: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 15: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 16: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 17: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 18: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 19: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 20: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 21: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 22: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 23: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 24: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 25: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 26: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 27: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 28: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 29: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 30: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 31: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 32: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 33: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 34: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 35: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 36: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 37: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 38: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 39: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 40: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 41: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 42: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 43: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 44: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 45: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 46: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 47: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 48: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 49: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 50: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 51: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 52: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 53: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 54: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 55: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 56: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 57: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 58: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 59: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 60: attn_hc, ffn_hc, input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 61 == 61 |
| C2 | PASS | 4 clusters == 4 from config schedule ['layer_types', 'mlp_layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=7168 in 61/61 layers |
| C6 | PASS | hidden_size=7168 (heuristic check, 24299 flagged) |
| C7 | PASS | MQA (128 query heads : 1 kv head) |
| C8 | WARN | MoE trace-verified [router_dim(E=384):ok, top_k(6):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=129280, tie_word_embeddings=False |
| C10 | PASS | all 1772 params covered |
| C11 | PASS | 426 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=2048 >= required=2048 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 31440 unmapped rows, 48 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `deepseek-ai/DeepSeek-V4-Pro` config.json @ `b5968e9190ef611bbf34a7229255be88a0e937c1` (sha256 `f1b521a7962e…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=2048 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)

의뢰서 1건 — 같은 op 의 입력과 출력이 다르게 렌더되던 것을 찾아 교정 완료.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 교정 필요 | 9 |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.self_attn.compressor.kv_norm` | [B, 512, 512] 의 축 순서 | `[B, d_head, d_head]` | `[B, T/m_csa, d_head]` | `modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)` — RMSNorm 은 마지막 축을 정규화하므로 마지막이 `d_head`(512)이고 가운데가 압축 KV 길이다. **부분 교정(2026-08-09)**: rank-1 norm 앵커를 그 모듈 전체로 확장해  … |
| `model.layers.*.self_attn` | grouped output projection 그룹 축 (16) | `T/m_hca` | `g_o` | `clone [B,T,T/m_hca,d_g] -> _unsafe_view -> [B,T,g_o*d_g]` (실측 `[1,2048,16,1024]` → `[1,2048,16384]`). 합쳐진 축이 `g_o*d_g` 이므로 셋째 축은 `g_o` 여야 하는데 g_o = T/m_hca = 16 이라 압축 엔트리 수의 이름이 붙었다. `d_g` 자체는 맞다. 고치 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
