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
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 129280 |
| ctx | 1048576 |
| E | 256 |
| E_shared | 1 |
| k | 6 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 2048 |
| w_local | 128 |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
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

shape 축 **717,735개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 297,119 | 41.40% |
| 이 모듈 스코프의 심볼 | 176,088 | 24.53% |
| 스코프 없는 심볼 | 131,939 | 18.38% |
| 이 모듈 스코프의 유도식 | 69,928 | 9.74% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 36,157 | 5.04% |
| 이름 없음 (정수 유지) | 6,504 | 0.91% |

등록된 규칙 **675,074축**, 약한 근거 36,157축, 휴리스틱 **0축 (0.0%)**, 이름 없음 6,504축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 16 | n_hc² (mHC comb 행렬 원소 수) | attn_hc, ffn_hc |
| 24 | (2+n_hc)·n_hc (mHC 게이트 파라미터 수: pre n_hc + post n_hc + comb n_hc²) | attn_hc, ffn_hc |
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | compressor, indexer, rotary_emb, self_attn |
| 127 | w_local − 1 (sliding window mask 밴드 폭) | self_attn |
| 257 | T/m_csa − 1 (CSA Ca/Cb 겹침 shift: 이전 윈도우 기여분 슬라이스) | compressor, indexer |
| 258 | T/m_csa (CSA 압축 엔트리 수) | compressor, indexer, kv_norm, rotary_emb, scorer, self_attn |
| 259 | T/m_csa + 1 (CSA block-bias 버퍼 = 압축 엔트리 수 + 무효 인덱스 슬롯 1) | compressor |
| 448 | d_head − d_rope (부분 RoPE 비회전 통과분) | compressor, self_attn |
| 1033 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | self_attn |
| 1040 | T + T/m_hca (HCA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 1041 | T + T/m_hca + 1 (HCA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 1290 | T + T/m_csa (CSA 레이어 KV 길이: sliding ⊕ 압축 엔트리) | self_attn |
| 1291 | T + T/m_csa + 1 (CSA 레이어 score 폭: sliding KV ⊕ 압축 KV ⊕ attention sink) | self_attn |
| 6192 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 8192 | n_h^I·c^I (Lightning Indexer 쿼리 투영 폭) | indexer, o_a_proj, o_b_proj, q_b_proj, self_attn |
| 16384 | n_hc·d_model (mHC: n_hc개 잔차 스트림을 편 폭) | attn_hc, ffn_hc, hc_head, input_norm |
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
| C7 | PASS | MQA (64 query heads : 1 kv head) |
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

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 4건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 6 |
| 교정 필요 | 10 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `o_a_proj$` | `g_o` | `g_o` | 86 | V4-Pro 와 같은 코드. modeling_deepseek_v4.py:783-785 / :317-323. o_groups=8. |
| `self_attn$` | `n_h` | `d_rope` | 1118 | modeling_deepseek_v4.py:338-350 `cos`/`sin` 의 d_rope/2 항목을 `repeat_interleave(2, dim=-1)`로 전체 `rope_dim`까지 늘리고, 그 폭으로 `x[..., -rope_dim:]`을 회전한다. 따라서 `[B,T,d_rope/2,2]`를 평탄화한 nth 5 view의 마지막 축은 head 수가 아니라 d_rope다. |
| `self_attn$` | `n_h` | `d_rope` | 1118 | decode의 같은 자리. modeling_deepseek_v4.py:338-350에서 d_rope/2인 cos/sin을 `repeat_interleave(2, dim=-1)`해 `rope_dim`을 만들므로 `[B,1,d_rope/2,2]`를 평탄화한 nth 5 view의 마지막 축은 d_rope다. |
| `compressor$` | `n_h` | `d_rope` | 546 | modeling_deepseek_v4.py:338-350의 `apply_rotary_pos_emb`는 d_rope/2인 cos/sin을 `repeat_interleave(2, dim=-1)`해 전체 `rope_dim`으로 만든다. :670-671에서 CSA compressor의 `[B,T/m_csa,d_rope/2]` cos/sin에 이 함수를 호출하므로, `[B,T/m_csa,d_rope/2,2]`를 평탄화한 nth 2 view의 마지막 축은 n_h가 아니라 d_rope다. |
| `indexer$` | `n_h_I` | `d_rope` | 525 | modeling_deepseek_v4.py:338-350의 `apply_rotary_pos_emb`는 d_rope/2인 cos/sin을 `repeat_interleave(2, dim=-1)`해 전체 `rope_dim`으로 만든다. :542-546에서 indexer의 `[B,T/m_csa,d_rope/2]` cos/sin에 이 함수를 호출하므로, `[B,T/m_csa,d_rope/2,2]`를 평탄화한 nth 2 view의 마지막 축은 n_h_I가 아니라 d_rope다. |
| `compressor$` | `n_h` | `d_rope` | 520 | transformers 5.14.1 installed source modeling_deepseek_v4.py:342-359 expands the half-width cos/sin pairs to the full trailing rope_dim; :384-422 applies that RoPE to the HCA compressor output. Therefore the nth-2 view produced while flattening the repeated pairs has trailing width d_rope, not n_h. |
| `indexer$` | `n_h` | `d_rope` | 357 | transformers 5.14.1 modeling_deepseek_v4.py:345-359 defines rope_dim from the repeated cos width and takes rope=x[..., -rope_dim:]. Applied to indexer q created at :563-565, slice nth21 therefore has trailing d_rope, not the equal-valued n_h. |
| `indexer$` | `n_h` | `d_rope` | 357 | transformers 5.14.1 modeling_deepseek_v4.py:345-359 defines the trailing slice as rope=x[..., -rope_dim:], with rope_dim obtained from the full repeated cos width. On decode indexer q from :563-565, slice nth3 therefore ends in d_rope, not n_h. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mlp.experts` | [E, d_model, d_model] 의 가운데 축 (4096) | `d_model` | `2*d_moe` | `modeling_deepseek_v4.py:992` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. d_moe=2048 이라 2·2048=4096=d_model 로 겹친다. OLMoE 와 같은 경로로 부분  … |
| `model.layers.*.self_attn.compressor.indexer` | [B, T/m_csa, 4, c_I] 의 셋째 축 | `4 (이름 없음)` | `m_csa` | indexer 안의 `[B, T/m_csa, 4, c_I]` 는 압축 엔트리마다 그것이 덮는 원본 토큰 m_csa 개다(m_csa=4). 그런데 `m_csa` 의 스코프가 `compressor(?!\.indexer)` 라 이름이 안 붙고 정수로 남는다. 그 배제는 원래 **m_hca(=128)가 c_I(=128)를 뺏는 것**을 막으려고 넣은 것이라, 값이 … |
| `model.layers.*.self_attn` | 복소수 되접기 축 (64) | `n_h` | `d_rope` | `view [B, T, d_rope/2, 2] -> [B, T, n_h]` — 뒤 두 축을 합치면 d_rope/2 × 2 = **d_rope**(64)다. RoPE 를 복소수 곱으로 구현할 때 실수부·허수부를 되접는 자리이고, attention head 수와는 아무 관계가 없다. n_h 도 64 라 값으로는 안 보인다. **반박 프레임으로 찾았다** — ' … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
