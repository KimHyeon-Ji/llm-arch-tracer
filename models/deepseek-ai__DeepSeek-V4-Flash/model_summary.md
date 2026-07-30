# Model Summary -- deepseek-ai/DeepSeek-V4-Flash

## 기본 정보

- revision: `60d8d70770c6776ff598c94bb586a859a38244f1`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 1032
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 284.33B total, 13.8B active (4.9% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 1,048,576  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-04-22  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MQA + HCA/CSA |
| 6 | LAYER MIX | 21× compressed_sparse_attention, 20× heavily_compressed_attention, 2× sliding_attention  (FFN: 43× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 5.4 KiB (Very low)  _(증가하는 압축 엔트리만 계산, K==V 단일 텐서; 제외: 고정 크기 sliding 버퍼(window 128, 전 43층, 그중 2층은 이것만 보유) / Lightning Indexer 캐시(+1.31 KiB/token))_ |
| 8 | KEY DETAIL | MQA + HCA/CSA attention; Sparse MoE (E=256, top-6, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, MQA, HCA, CSA, mHC, MoE, shared expert, sigmoid-gating, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `deepseek_v4` |
| attention | MQA — 64 query heads, 1 kv head, d_head=512; sliding window 128 on part of layers (hybrid local/global) + 블록 압축 분기(HCA m=128, CSA m=4); sliding window는 전 레이어 적용 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | MoE — 256 routed experts, top-6 + 1 shared, expert intermediate 2048, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 블록 압축 — 압축 레이어당 d_head/m = 512/m elems / token (HCA m=128, CSA m=4), K==V 단일 텐서 ⇒ 5,536 B/token 전체 (5.41 KiB). sliding 분기는 window=128로 상한이 있어 컨텍스트에 따라 증가하지 않음 |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 43 |
| d_model | 4096 |
| n_h | 64 |
| n_kv | 1 |
| d_head | 512 |
| d_ff | 2048 |
| V | 129280 |
| ctx | 1048576 |
| E | 256 |
| E_shared | 1 |
| k | 6 |
| d_moe | 2048 |
| w_local | 128 |
| layer_sched | 21× compressed_sparse_attention, 20× heavily_compressed_attention, 2× sliding_attention (총 43층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | 1024 |
| d_rope | 64 |
| m_csa | 4 |
| m_hca | 128 |
| g_o | 8 |
| d_g | 1024 |
| n_h_I | 64 |
| c_I | 128 |
| k_I | 512 |
| n_hc | 4 |
| t_sinkhorn | 20 |
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

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 127 | w_local − 1 (sliding window mask 밴드 폭) | self_attn |
| 257 | T/m_csa − 1 (CSA Ca/Cb 겹침 shift: 이전 윈도우 기여분 슬라이스) | compressor, indexer |
| 258 | T/m_csa (CSA 압축 엔트리 수) | compressor, indexer, kv_norm, rotary_emb, scorer, self_attn |
| 259 | T/m_csa + 1 (CSA block-bias 버퍼 = 압축 엔트리 수 + 무효 인덱스 슬롯 1) | compressor |
| 448 | d_head − d_rope (부분 RoPE 비회전 통과분) | compressor, self_attn |
| 1040 | T + T/m_hca (HCA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 1041 | T + T/m_hca + 1 (HCA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 1290 | T + T/m_csa (CSA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 1291 | T + T/m_csa + 1 (CSA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 8192 | g_o·d_g (grouped output projection 합친 폭 → o_b_proj 입력) | indexer, o_a_proj, o_b_proj, q_b_proj, self_attn |
| 32768 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | q_b_proj, self_attn |

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

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 43 == 43 |
| C2 | PASS | 4 clusters == 4 from config schedule ['layer_types', 'mlp_layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 43/43 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 16671 flagged) |
| C7 | PASS | GQA 64:1 (repeat factor 64) |
| C8 | WARN | MoE trace-verified [router_dim(E=256):ok, top_k(6):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=129280, tie_word_embeddings=False |
| C10 | PASS | all 1242 params covered |
| C11 | PASS | 340 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=1032 >= required=1032 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 22037 unmapped rows, 48 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `deepseek-ai/DeepSeek-V4-Flash` config.json @ `60d8d70770c6776ff598c94bb586a859a38244f1` (sha256 `7ae6ab3cca56…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=1032 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_
