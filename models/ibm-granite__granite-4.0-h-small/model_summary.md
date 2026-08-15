# Model Summary -- ibm-granite/granite-4.0-h-small

## 기본 정보

- revision: `b8c0982bab7fde4eb48110f5a069527c008fab39`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 32.21B total, 8.8B active (27.3% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-09-16  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 36× linear_attention, 4× full_attention  (attention: GQA)  (FFN: 40× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 16.0 KiB (Very low) over 4 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=72, top-10) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `granitemoehybrid` |
| attention | GQA — 32 query : 8 kv heads (repeat 4), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | MoE — 72 routed experts, top-10, expert intermediate 768, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; all 40 layers ⇒ 81920 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 40 |
| d_model | 4096 |
| n_h | 32 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 768 |
| d_shared | 1536 |
| V | 100352 |
| ctx | 131072 |
| E | 72 |
| E_shared | 0 |
| k | 10 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 768 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 36× linear_attention, 4× full_attention (총 40층) |
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
| d_state | 128 |
| n_g_ssm | 1 |
| n_h_ssm | 128 |
| d_chunk | 256 |
| d_head_ssm | 64 |
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

shape 축 **155,627개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 54,181 | 34.81% |
| 이 모듈 스코프의 심볼 | 52,336 | 33.63% |
| 스코프 없는 심볼 | 26,917 | 17.30% |
| 이 모듈 스코프의 유도식 | 14,023 | 9.01% |
| 이름 없음 (정수 유지) | 5,128 | 3.30% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,042 | 1.95% |

등록된 규칙 **147,457축**, 약한 근거 3,042축, 휴리스틱 **0축 (0.0%)**, 이름 없음 5,128축.

## 미등록 config 필드 (Tier 2 조사 대상)

이 아키텍처가 실제로 쓰는 config 필드 중 `rules/symbols.yaml`에 등록되지 않은 것들이다. 등록되지 않은 폭은 이름을 붙일 근거가 없으므로 shape 셀에 정수로 남는다. `02-new-module-handling.md` Tier 2 절차로 역할을 확인한 뒤 `aliases`(같은 개념의 다른 필드명) 또는 `derived_dims.yaml`(계산식)에 **출처와 함께** 등록하면 다음 모델부터 자동으로 잡힌다.

| config 필드 | 값 | 쓰는 모듈 수 |
|---|---|---|
| `logits_scaling` | 16 | 1 |
| `embedding_multiplier` | 12 | 1 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 19 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mamba |
| 160 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 3072 | 2·d_shared (공유 MLP gate+up 융합 투영 폭) | input_linear, shared_mlp |
| 8192 | d_inner (Mamba 내부 폭 = n_h_ssm · d_head_ssm) | mamba, norm, out_proj |
| 8448 | d_inner + 2·n_g·d_state (conv1d 입력 폭) | act, conv1d, mamba |
| 16768 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mamba |

## 레이어 구조

- layer 0-4: block_sparse_moe, input_layernorm, mamba, post_attention_layernorm, shared_mlp
- layer 5: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn, shared_mlp
- layer 6-14: block_sparse_moe, input_layernorm, mamba, post_attention_layernorm, shared_mlp
- layer 15: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn, shared_mlp
- layer 16-24: block_sparse_moe, input_layernorm, mamba, post_attention_layernorm, shared_mlp
- layer 25: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn, shared_mlp
- layer 26-34: block_sparse_moe, input_layernorm, mamba, post_attention_layernorm, shared_mlp
- layer 35: block_sparse_moe, input_layernorm, post_attention_layernorm, self_attn, shared_mlp
- layer 36-39: block_sparse_moe, input_layernorm, mamba, post_attention_layernorm, shared_mlp

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 40 == 40 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 40/40 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 168 flagged) |
| C7 | PASS | GQA 32:8 (repeat factor 4) |
| C8 | WARN | MoE trace-verified [router_dim(E=72):ok, top_k(10):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=100352, tie_word_embeddings=True |
| C10 | PASS | all 586 params covered |
| C11 | PASS | 44 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 5459 unmapped rows, 39 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `ibm-granite/granite-4.0-h-small` config.json @ `b8c0982bab7fde4eb48110f5a069527c008fab39` (sha256 `781d74c171aa…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 2건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 이름 없음이 정답 | 2 |
| 교정 필요 | 3 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `mamba$` | `d_state` | `n_h_ssm` | 324 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-414에서 B/C는 `[B,T,num_heads,state_size]`이고 sequence 축만 pad된다. 앞의 마지막 state 축 교정 이후 앵커에서 축 2는 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 360 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:416-421에서 A는 `[B,num_heads,c,l]`이고 segment_sum 출력은 `[B,num_heads,c,l,l]`이다. 축 1은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 360 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:416-417,458에서 inter-chunk decay의 prefix는 `[B,num_heads]`이고 뒤 두 축만 chunk 경계다. 축 1은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 324 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:645-648에서 dt는 `[B,self.num_heads,self.head_dim]`으로 expand된다. 축 1은 n_h_ssm이다. |
| `mamba$` | `n_h_ssm` | `d_state` | 288 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:339-355에서 B는 `[B,num_groups,num_heads//num_groups,state_size]`로 expand된다. 마지막 축은 n_h_ssm이 아니라 d_state다. |
| `mamba$` | `n_h_ssm` | `d_state` | 792 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:339-359은 state-update 곱을 `[batch_size,num_heads,head_dim,state_size]`로 만든다. 마지막 축은 n_h_ssm이 아니라 d_state다. |
| `mamba$` | `d_state` | `n_h_ssm` | 720 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:339-359의 `batch_size, num_heads, head_dim` 해체에 따라 축 1은 state_size가 아니라 num_heads, 즉 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 576 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-426은 hidden_states를 `[B,sequence_length,num_heads,head_dim]`으로 읽고 :205-221이 sequence 축만 pad한다. 따라서 축 2의 128은 d_state가 아니라 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 504 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-428에서 A는 chunk reshape 뒤 `permute(0,3,1,2)`를 거쳐 `[B,num_heads,n_chunks,chunk_size]`가 된다. 축 1은 d_state가 아니라 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 396 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:509,534,606-607에서 projected_states의 마지막 split 크기는 `self.num_heads`이고 그 출력이 dt다. shape_index 2의 마지막 축은 d_state가 아니라 n_h_ssm이다. |
| `mamba$` | `n_h_ssm` | `d_state` | 396 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-426에서 B/C는 먼저 num_heads로 repeat되어 `[B,T,num_heads,state_size]`가 된 뒤 :205-221에서 sequence 축만 pad된다. 따라서 마지막 축은 n_h_ssm이 아니라 d_state다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-426에서 A와 dt는 `[B,T,num_heads]`이고 singleton을 붙여도 축 2는 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:403-433에서 chunk A의 순서는 `[B,num_heads,n_chunks,chunk_size]`다. 출력 축 1은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:444-460에서 states 순서는 chunk 뒤 num_heads, head_dim, state_size다. 이 broadcast 출력 축 3은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:339-359에서 recurrent hidden_states는 `[B,num_heads,head_dim]`이다. singleton 출력 축 1은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:339-362의 두 번째 recurrent state broadcast도 `[B,num_heads,head_dim,1]` 순서다. 축 1은 n_h_ssm이다. |
| `mamba$` | `d_state` | `n_h_ssm` | 216 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:366-373에서 C는 `[B,num_heads,state_size]`다. bmm용 view 입력 축 1은 n_h_ssm이다. |
| `mamba$` | `n_h_ssm` | `d_state` | 216 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:366-373의 C 순서는 `[B,num_heads,state_size]`다. 앞 교정 뒤 축 2는 d_state다. |
| `mamba$` | `d_state` | `n_h_ssm` | 72 | transformers 5.14.1 installed source modeling_granitemoehybrid.py:209-714; revalidated this axis verdict unchanged. modeling_granitemoehybrid.py:372-374에서 C_reshaped는 `[batch_size*num_heads,state_size,1]`이다. B=1 표본의 출력 축 0은 n_h_ssm이다. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mamba` | n_h_ssm vs d_state 축 (둘 다 128) | `d_state / n_h_ssm 혼용` | `(소스가 가리키는 쪽 — 근거 참조)` | `view [.., n_g_ssm, n_h_ssm/n_g_ssm, ?] -> [.., ?, ?]` 의 두 축이 값으로 구별되지 않는다(ssm_state_size == mamba_n_heads == 128). Nemotron-3-Super 와 **같은 막힘**이고, 합쳐진 축이 무엇인지는 reshape 자체가 알지만 그걸 채택하려면 개명을 데이터플로우 끝까지 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
