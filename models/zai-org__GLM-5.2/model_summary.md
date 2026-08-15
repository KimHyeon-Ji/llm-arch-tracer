# Model Summary -- zai-org/GLM-5.2

## 기본 정보

- revision: `b4734de4facf877f85769a911abafc5283eab3d9`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 2049
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 743.38B total, 41.25B active (5.5% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 1,048,576  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-06-16  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MLA |
| 6 | LAYER MIX | 78× deepseek_sparse_attention  (FFN: 3 dense + 75 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 87.8 KiB (Moderate) |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=256, top-8, +1 shared, sigmoid gating/aux-loss-free); dense-prefix 3 layer(s) |
| 9 | Related concepts | RMSNorm, LayerNorm, RoPE, MLA, MoE, shared expert, sigmoid-gating, QK-Norm, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `glm_moe_dsa` |
| attention | MLA — KV latent compression (kv_lora_rank=512, q_lora_rank=2048); 헤드 q/k = nope(192)+rope(64)=256, v=256, n_h=64 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=8000000) |
| FFN | MoE — 256 routed experts, top-8 + 1 shared, expert intermediate 2048, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm + LayerNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | compressed MLA latent ≈ kv_lora_rank=512 (+decoupled RoPE dim) / token / layer |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 78 |
| d_model | 6144 |
| n_h | 64 |
| n_kv | 64 |
| d_head | 64 |
| d_ff | 12288 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 154880 |
| ctx | 1048576 |
| E | 256 |
| E_shared | 1 |
| k | 8 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 2048 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 78× deepseek_sparse_attention |
| c_kv | 512 |
| d_nope | 192 |
| d_v | 256 |
| c_q | 2048 |
| d_rope | 64 |
| m_csa | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| m_hca | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| g_o | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| d_g | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| n_h_I | 32 |
| c_I | 128 |
| k_I | 2048 |
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

shape 축 **326,319개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 116,708 | 35.77% |
| 스코프 없는 심볼 | 87,378 | 26.78% |
| 이 모듈 스코프의 심볼 | 80,580 | 24.69% |
| 이 모듈 스코프의 유도식 | 35,268 | 10.81% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 5,356 | 1.64% |
| 이름 없음 (정수 유지) | 1,029 | 0.32% |

등록된 규칙 **319,934축**, 약한 근거 5,356축, 휴리스틱 **0축 (0.0%)**, 이름 없음 1,029축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 448 | d_nope+d_v | self_attn |
| 576 | c_kv+d_rope (MLA kv_a_proj_with_mqa 출력) | kv_a_proj_with_mqa, self_attn |
| 4096 | n_h^I·c^I (Lightning Indexer 쿼리 투영 폭) | experts, indexer, wq_b |
| 16384 | n_h·d_v (attention 출력, o_proj 직전) | o_proj, q_b_proj, self_attn |
| 16392 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 28672 | n_h·(d_nope+d_v) (MLA kv_b_proj 출력) | kv_b_proj, self_attn |

## 레이어 구조

- layer 0-2: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 3-5: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 6: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 7-9: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 10: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 11-13: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 14: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 15-17: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 18: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 19-21: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 22: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 23-25: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 26: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 27-29: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 30: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 31-33: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 34: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 35-37: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 38: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 39-41: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 42: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 43-45: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 46: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 47-49: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 50: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 51-53: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 54: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 55-57: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 58: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 59-61: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 62: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 63-65: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 66: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 67-69: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 70: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 71-73: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 74: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 75-77: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 78 == 78 |
| C2 | PASS | 3 clusters == 3 from config schedule ['indexer_types', 'layer_types', 'mlp_layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=6144 in 78/78 layers |
| C6 | PASS | hidden_size=6144 (heuristic check, 12435 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | WARN | MoE trace-verified [router_dim(E=256):ok, top_k(8):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=154880, tie_word_embeddings=False |
| C10 | PASS | all 1269 params covered |
| C11 | PASS | 574 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=2049 >= required=2048 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 9729 unmapped rows, 37 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `zai-org/GLM-5.2` config.json @ `b4734de4facf877f85769a911abafc5283eab3d9` (sha256 `98ddd1161773…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=2049 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 2건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 4 |
| 교정 필요 | 4 |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.self_attn.indexer` | rope 슬라이스 폭 64 | `d_head / n_h` | `d_rope` | `modeling_glm_moe_dsa.py:225-229`: `q = q.view(B, S, self.n_heads, self.head_dim)` 뒤 `q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)`, `k = self … |
| `model.layers.*.self_attn.indexer` | interleaved rope 절반 32 | `n_h_I` | `d_rope/2` | 위 항목과 같은 자리의 짝이다. `apply_rotary_pos_emb_interleave`(`modeling_glm_moe_dsa.py:232`)가 rope 슬라이스를 짝/홀로 갈라 32 를 만든다. `index_n_heads`(=32)와 값이 같아 head 개수 이름이 붙었으나, `[B, T, 1, ·]` 의 마지막 축은 feature 다 — 같은 행의 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
