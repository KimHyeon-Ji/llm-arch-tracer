# Model Summary -- Qwen/Qwen3.5-397B-A17B

## 기본 정보

- revision: `8472618112abcbd45acbcdc58436aff4233c23f7`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 396.35B total, 17.35B active (4.4% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-02-16  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 45× linear_attention, 15× GQA  (FFN: 60× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 30.0 KiB (Low) over 15 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=512, top-10, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, sigmoid-gating, QK-Norm, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `qwen3_5_moe_text` |
| attention | GQA — 32 query : 2 kv heads (repeat 16), d_head=256 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000000) |
| FFN | MoE — 512 routed experts, top-10 + 1 shared, expert intermediate 1024, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·2·256 = 1024 elems / token / layer; all 60 layers ⇒ 61440 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 60 |
| d_model | 4096 |
| n_h | 32 |
| n_kv | 2 |
| d_head | 256 |
| d_ff | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 248320 |
| ctx | 262144 |
| E | 512 |
| E_shared | 1 |
| k | 10 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 1024 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 45× linear_attention, 15× full_attention (총 60층) |
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
| n_h_lin_k | 16 |
| n_h_lin_v | 64 |
| d_head_lin_k | 128 |
| d_head_lin_v | 128 |
| d_conv_lin | 4 |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **1,020,605개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 429,763 | 42.11% |
| 이 모듈 스코프의 심볼 | 269,631 | 26.42% |
| 이름 없음 (정수 유지) | 132,866 | 13.02% |
| 이 모듈 스코프의 유도식 | 122,772 | 12.03% |
| 스코프 없는 심볼 | 55,623 | 5.45% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 6,170 | 0.60% |
| 휴리스틱: 심볼의 배수 | 3,780 | 0.37% |

등록된 규칙 **877,789축**, 약한 근거 6,170축, 휴리스틱 **3,780축 (0.37%)**, 이름 없음 132,866축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.0.linear_attn` | `2*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.0.linear_attn` | `3*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.0.linear_attn` | `3*n_h_lin_k` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.1.linear_attn` | `2*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.1.linear_attn` | `3*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.1.linear_attn` | `3*n_h_lin_k` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.2.linear_attn` | `2*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.2.linear_attn` | `3*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.2.linear_attn` | `3*n_h_lin_k` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.4.linear_attn` | `2*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.4.linear_attn` | `3*d_conv_lin` | 휴리스틱: 심볼의 배수 | 28 |
| `model.layers.4.linear_attn` | `3*n_h_lin_k` | 휴리스틱: 심볼의 배수 | 28 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 18 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | linear_attn |
| 36 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | linear_attn |
| 170 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 192 | d_head − d_rope (부분 RoPE 비회전 통과분, partial_rotary_factor 기준) | self_attn |
| 1088 | T·n_h_lin_v (value head 축까지 flatten — gated norm 입력) | linear_attn, norm |
| 2048 | n_k·d_k (DeltaNet key_dim — q/k 조각 폭) | experts, linear_attn |
| 8192 | n_v·d_v (DeltaNet value_dim — v/z 조각 폭) | in_proj_z, linear_attn, o_proj, out_proj, self_attn |
| 12288 | 2·key_dim + value_dim (gated delta net conv1d 채널 폭) | conv1d, in_proj_qkv, linear_attn |
| 16384 | 2·n_h·d_head (게이트 어텐션 Q 투영: query ⊕ gate 묶음 폭) | q_proj, self_attn |

## 레이어 구조

- layer 0-2: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 3: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 4-6: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 7: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 8-10: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 11: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 12-14: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 15: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 16-18: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 19: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 20-22: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 23: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 24-26: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 27: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 28-30: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 31: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 32-34: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 35: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 36-38: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 39: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 40-42: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 43: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 44-46: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 47: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 48-50: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 51: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 52-54: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 55: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 56-58: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 59: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 60 == 60 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 60/60 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 3645 flagged) |
| C7 | PASS | GQA 32:2 (repeat factor 16) |
| C8 | WARN | MoE trace-verified [router_dim(E=512):ok, top_k(10):ok, expert_weight:grouped]; routed-token coun... |
| C9 | PASS | vocab_size=248320, tie_word_embeddings=False |
| C10 | PASS | all 1038 params covered |
| C11 | PASS | 136 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 41679 unmapped rows, 38 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Qwen/Qwen3.5-397B-A17B` config.json @ `8472618112abcbd45acbcdc58436aff4233c23f7` (sha256 `edf7ec82e7c4…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 행 단위 전건 — 검토자 방식)



| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 이름 없음이 정답 | 1 |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
