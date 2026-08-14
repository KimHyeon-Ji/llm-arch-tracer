# Model Summary -- Qwen/Qwen3.6-27B

## 기본 정보

- revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 26.9B total (dense) |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-04-21  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 48× linear_attention, 16× full_attention  (attention: GQA) |
| 7 | KV CACHE / TOKEN (BF16) | 64.0 KiB (Low) over 16 attn layers |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA, QK-Norm, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `qwen3_5_text` |
| attention | GQA — 24 query : 4 kv heads (repeat 6), d_head=256 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000000) |
| FFN | dense FFN — intermediate 17408, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·4·256 = 2048 elems / token / layer; all 64 layers ⇒ 131072 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 64 |
| d_model | 5120 |
| n_h | 24 |
| n_kv | 4 |
| d_head | 256 |
| d_ff | 17408 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 248320 |
| ctx | 262144 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 48× linear_attention, 16× full_attention (총 64층) |
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
| n_h_lin_v | 48 |
| d_head_lin_k | 128 |
| d_head_lin_v | 128 |
| d_conv_lin | 4 |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **1,055,673개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 455,516 | 43.15% |
| 이 모듈 스코프의 심볼 | 274,796 | 26.03% |
| 이름 없음 (정수 유지) | 139,039 | 13.17% |
| 이 모듈 스코프의 유도식 | 125,985 | 11.93% |
| 스코프 없는 심볼 | 48,465 | 4.59% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 7,744 | 0.73% |
| 휴리스틱: 심볼의 배수 | 2,784 | 0.26% |
| 휴리스틱: 심볼+1 | 1,344 | 0.13% |

등록된 규칙 **904,762축**, 약한 근거 7,744축, 휴리스틱 **4,128축 (0.39%)**, 이름 없음 139,039축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.3.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.7.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.11.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.15.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.19.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.23.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.27.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.31.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.35.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.39.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.43.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.47.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 3 | n_v/n_k (DeltaNet key head 하나가 담당하는 value head 수) | linear_attn, model, rotary_emb |
| 6 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | linear_attn, self_attn |
| 18 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | linear_attn |
| 32 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | linear_attn, rotary_emb, self_attn |
| 64 | d_rope (partial_rotary_factor 기준 회전 차원) | linear_attn, rotary_emb, self_attn |
| 192 | d_head − d_rope (부분 RoPE 비회전 통과분, partial_rotary_factor 기준) | self_attn |
| 512 | 2·d_head (CSA/HCA 압축기 kv_proj·gate_proj 폭: Ca⊕Cb) | self_attn |
| 816 | T·n_h_lin_v (value head 축까지 flatten — gated norm 입력) | linear_attn, norm |
| 1024 | 2·d_k + 2·(n_v/n_k)·d_v (DeltaNet qkvz 를 key head 별로 접은 폭) | k_proj, self_attn, v_proj |
| 2048 | n_k·d_k (DeltaNet key_dim — q/k 조각 폭) | linear_attn |
| 6144 | n_v·d_v (DeltaNet value_dim — v/z 조각 폭) | in_proj_z, linear_attn, o_proj, out_proj, self_attn |
| 10240 | 2·key_dim + value_dim (gated delta net conv1d 채널 폭) | conv1d, in_proj_qkv, linear_attn |
| 12288 | 2·n_h·d_head (게이트 어텐션 Q 투영: query ⊕ gate 묶음 폭) | q_proj, self_attn |

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
- layer 60-62: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 63: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 64 == 64 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=5120 in 64/64 layers |
| C6 | PASS | hidden_size=5120 (heuristic check, 2208 flagged) |
| C7 | PASS | GQA 24:4 (repeat factor 6) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=248320, tie_word_embeddings=False |
| C10 | PASS | all 851 params covered |
| C11 | PASS | 145 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 42962 unmapped rows, 27 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Qwen/Qwen3.6-27B` config.json @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (sha256 `9860c561806a…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 1건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 이름 없음이 정답 | 3 |
| 교정 필요 | 6 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `linear_attn\.norm$` | `d_head_lin_k` | `d_head_lin_v` | 2976 | modeling_qwen3_next.py:552 / :519 — 같은 블록. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.linear_attn` | in_proj_qkvz 조각 폭 (27B 에서 2048) | `2*n_kv*d_head` | `key_dim (= n_h_lin_k · d_head_lin_k)` | `modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다 … |
| `model.layers.*.linear_attn` | matmul 수축 축 (128) | `d_head_lin_k / d_head_lin_v 혼용` | `(소스가 가리키는 쪽 — 근거 참조)` | `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없 … |
| `model.layers.*.linear_attn.norm` | 정규화 폭 128 | `d_head_lin_k` | `d_head_lin_v` | `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head  … |
| `model.layers.*.linear_attn` | gated delta rule 청크 길이 64 (chunk_size) | `d_rope` | `d_chunk` | `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
