# Model Summary -- meta-llama/Llama-4-Maverick-17B-128E

## 기본 정보

- revision: `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 400.71B total, 17.18B active (4.3% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings; **공급자 공개값 1,048,576** (둘 다 사실이다 — 공개값에 맞추려고 config 값을 고치지 않는다))_ |
| 3 | DATE | 2025-04-02  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 36× chunked_attention, 12× full_attention  (attention: GQA)  (FFN: 24 dense + 24 MoE) |
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
| 위치 인코딩 | RoPE (θ=500000.0); 12/48개 레이어는 NoPE(위치 인코딩 없음) — 4번째마다 |
| FFN | MoE — 128 routed experts, top-1 + 1 shared, expert intermediate 8192, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; 48 attention layer(s) ⇒ 98304 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 48 |
| d_model | 5120 |
| n_h | 40 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 16384 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 202048 |
| ctx | 262144 |
| E | 128 |
| E_shared | 1 |
| k | 1 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 8192 |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding_window` 계열 구조를 쓰지 않음)_ |
| chunk_size | 8192 |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 36× chunked_attention, 12× full_attention (총 48층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | —  _(해당 없음: 이 모델은 `partial_rope` 계열 구조를 쓰지 않음)_ |
| n_h_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| d_head_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
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

shape 축 **112,229개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 39,266 | 34.99% |
| 이 모듈 스코프의 심볼 | 28,398 | 25.30% |
| 스코프 없는 심볼 | 25,277 | 22.52% |
| 이 모듈 스코프의 유도식 | 16,760 | 14.93% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,472 | 1.31% |
| 이름 없음 (정수 유지) | 1,056 | 0.94% |

등록된 규칙 **109,701축**, 약한 근거 1,472축, 휴리스틱 **0축 (0.0%)**, 이름 없음 1,056축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 5 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 17 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | (root), 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 5, 6, 7, 8, 9, activation_fn, down_proj, embed_tokens, feed_forward, gate_proj, input_layernorm, k_proj, lm_head, model, norm, o_proj, post_attention_layernorm, q_proj, rotary_emb, self_attn, up_proj, v_proj |
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |

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

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 48 == 48 |
| C2 | WARN | 3 trace clusters vs 2 config-schedule signatures ['layer_types', 'no_rope_layers'] -- review (mas... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=5120 in 48/48 layers |
| C6 | PASS | hidden_size=5120 (heuristic check, 2772 flagged) |
| C7 | PASS | GQA 40:8 (repeat factor 5) |
| C8 | WARN | MoE trace-verified [router_dim(E=128):ok, top_k(1):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=202048, tie_word_embeddings=False |
| C10 | PASS | all 507 params covered |
| C11 | PASS | 96 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 3694 unmapped rows, 35 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `meta-llama/Llama-4-Maverick-17B-128E` config.json @ `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2` (sha256 `72572339104b…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-09-11 · codex(외부) — 산출물 4개(prefill/decode의 csv/jsonl)를 공식 모델 카드와 설치본 소스로 대조

차원 매핑은 정확. decode attention 의 가짜 `B` 축은 **좁은 범위 교정**으로 고쳤고(공개 CSV 12칸), 공개 표기 3건도 고쳤다. 표 생성 구조 2건은 `accepted_limit` 로 공개 요약에 명시한다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 교정 필요 | 5 |
| table_omits_computation | 1 |
| 미확정 | 3 |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `(root)` | E_shared | `1` | `shared_expert_modules=1` | `num_shared_experts` 같은 config 필드는 없다. shared MLP 모듈이 하나 있다는 구조 사실이므로 축 심볼표가 아니라 구조 메타데이터에 두는 편이 정확하다. Meta 공식 표기도 '128 experts' 이지 129 가 아니다. [2026-09-11 조치] 이 모델에서 E_shared 는 축 라벨로 쓰이지 않는다(다른 모델에서는  … |
| `model.layers.*` | block_type | `attn+MoE` | `chunked+RoPE+MoE / full+NoPE+temp+MoE` | 레이어 접기 자체는 옳다(3그룹: 0,2,..=dense / 1,5,..=chunked+MoE / 3,7,..=full+NoPE+MoE). 그런데 뒤 두 그룹이 같은 `attn+MoE` 로 찍혀 독자가 왜 같아 보이는 블록이 두 번 나오는지 알 수 없다. chunked/full 과 RoPE/NoPE 차이가 표에 드러나지 않는다. `no_rope_layers … |
| `feed_forward` | MoE 결합 | `elementwise_add 한 행` | `shared+routed add 와 residual add 를 분리` | 소스는 `shared_out += routed_sum` 다음에 `residual + combined` 두 단계다(modeling_llama4.py:166-174, 450-458). 지금 표는 add 한 행이 의존성 셋을 물어 두 단계를 하나로 뭉갠다. |
| `(전체)` | 표에 없는 연산 | `` | 미확정 | major 표에 chunked/full 마스크 생성과 score 합산, RoPE, NoPE 층의 temperature tuning, GQA 의 KV head 8->40 반복, router 의 topk/scatter, KV cache update/concat 이 안 보인다. QK-norm 이 없는 것은 누락이 아니다 -- 이 checkpoint 는 `use_ … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
