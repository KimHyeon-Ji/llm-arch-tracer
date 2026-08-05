# Model Summary -- nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

## 기본 정보

- revision: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 24
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 3.97B total (dense) |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-03-07  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 21× linear_attention, 17× mlp, 4× GQA |
| 7 | KV CACHE / TOKEN (BF16) | 16.0 KiB (Very low) over 4 attn layers |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, GQA, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `nemotron_h` |
| attention | GQA — 40 query : 8 kv heads (repeat 5), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | none observed (NoPE, or position handled implicitly) |
| FFN | dense FFN — intermediate 12544, SwiGLU (silu·gate)  _(config는 8 expert를 선언하지만 트레이스에 expert 연산·파라미터가 전혀 없음 — vestigial 필드, C8 WARN 참고)_ |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; all 42 layers ⇒ 86016 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 42 |
| d_model | 3136 |
| n_h | 40 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 12544 |
| V | 131072 |
| ctx | 262144 |
| E | 8 |
| E_shared | 1 |
| k | 2 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 7688 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 21× linear_attention, 17× mlp, 4× full_attention (총 42층) |
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
| n_g_ssm | 8 |
| n_h_ssm | 96 |
| d_chunk | 256 |
| d_head_ssm | 80 |
| d_conv | 4 |
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
| 5 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | mixer |
| 27 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mixer |
| 960 | d_inner/n_g_ssm (Mamba gated RMSNorm의 그룹당 폭) | norm |
| 1024 | n_g_ssm·d_state (B/C 하나의 폭) | k_proj, mixer, v_proj |
| 5120 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | mixer, o_proj, q_proj |
| 7680 | n_h_ssm·d_head_ssm (Mamba d_inner) | mixer, norm, out_proj |
| 9728 | d_inner + 2·n_g·d_state (Mamba causal conv1d 폭: x, B, C) | act, conv1d, mixer |
| 17504 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mixer |

## 레이어 구조

- layer 0: mixer, norm
- layer 1: mixer, norm
- layer 2: mixer, norm
- layer 3: mixer, norm
- layer 4: mixer, norm
- layer 5: mixer, norm
- layer 6-7: mixer, norm
- layer 8: mixer, norm
- layer 9: mixer, norm
- layer 10: mixer, norm
- layer 11: mixer, norm
- layer 12: mixer, norm
- layer 13: mixer, norm
- layer 14: mixer, norm
- layer 15: mixer, norm
- layer 16: mixer, norm
- layer 17: mixer, norm
- layer 18: mixer, norm
- layer 19: mixer, norm
- layer 20: mixer, norm
- layer 21: mixer, norm
- layer 22: mixer, norm
- layer 23: mixer, norm
- layer 24: mixer, norm
- layer 25: mixer, norm
- layer 26: mixer, norm
- layer 27: mixer, norm
- layer 28: mixer, norm
- layer 29: mixer, norm
- layer 30-31: mixer, norm
- layer 32: mixer, norm
- layer 33: mixer, norm
- layer 34-36: mixer, norm
- layer 37: mixer, norm
- layer 38: mixer, norm
- layer 39: mixer, norm
- layer 40: mixer, norm
- layer 41: mixer, norm

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 42 == 42 |
| C2 | PASS | 3 clusters == 3 from config schedule ['layers_block_type'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=3136 in 42/42 layers |
| C6 | PASS | hidden_size=3136 (heuristic check, 0 flagged) |
| C7 | PASS | GQA 40:8 (repeat factor 5) |
| C8 | WARN | config has num_experts=8 but NO expert params or router/expert ops in trace -- model is dense her... |
| C9 | PASS | vocab_size=131072, tie_word_embeddings=False |
| C10 | PASS | all 263 params covered |
| C11 | PASS | 29 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=24 >= required=24 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 2741 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` config.json @ `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` (sha256 `52013ed7f7a5…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=24 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_
