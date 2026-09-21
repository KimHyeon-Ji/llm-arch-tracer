# Model Summary -- moonshotai/Kimi-K3

## 기본 정보

- revision: `f831ab66814297da540d832a5235f8e904f29d06`
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
| 6 | LAYER MIX | 69× KDA, 24× MLA  (FFN: 1 dense + 92 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 27.0 KiB (Low) over 24 attn layers |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=896, top-16, +2 shared, sigmoid gating/aux-loss-free); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, MLA, MoE, shared expert, sigmoid-gating, short-conv |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `kimi_linear` |
| attention | MLA — KV latent compression (kv_lora_rank=512, q_lora_rank=1536); 헤드 q/k = nope(128)+rope(64)=192, v=128, n_h=96 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | none observed (NoPE, or position handled implicitly) |
| FFN | MoE — 896 routed experts, top-16 + 2 shared, expert intermediate 3072, SiTU-GLU (tanh+sigmoid gate, β=4.0, β_linear=25.0) |
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
| w_local | —  _(해당 없음: 이 모델은 `sliding_window` 계열 구조를 쓰지 않음)_ |
| chunk_size | —  _(해당 없음: 이 모델은 `chunked_attention` 계열 구조를 쓰지 않음)_ |
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

shape 축 **10,975,890개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 6,108,592 | 55.65% |
| 런타임 축 (B/T/1) | 3,386,194 | 30.85% |
| 이름 없음 (정수 유지) | 874,539 | 7.97% |
| 이 모듈 스코프의 유도식 | 394,458 | 3.59% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 113,458 | 1.03% |
| 스코프 없는 심볼 | 98,649 | 0.90% |

등록된 규칙 **9,987,893축**, 약한 근거 113,458축, 휴리스틱 **0축 (0.0%)**, 이름 없음 874,539축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 10 | d_head − d_rope (부분 RoPE 비회전 통과분) | self_attn |
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | self_attn |
| 37 | d_head/2 (RoPE rotate_half 분할 축) | self_attn |
| 192 | d_nope + d_rope (MLA q/k head 폭) | self_attn |
| 256 | d_nope+d_v | self_attn |
| 288 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | self_attn |
| 323 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv, k_conv1d, q_conv1d, v_conv1d |
| 576 | c_kv+d_rope (MLA kv_a_proj_with_mqa 출력) | kv_a_proj_with_mqa, self_attn |
| 3840 | (비-아키텍처 상수, 의도적으로 이름 없음 -- Kimi-K3 의 MoE even-split shim 이 만드는 **전문가 하나가 받는 토큰 수** 3,840 = B(3)·k(16)·T(320) / expert_cap(4). 아키텍처 폭이 아니라 shim 의 산술 부산물이다 -- MoE 라우팅이 값 의존적이라 meta 텐서로는 추적할) | 0, 1, 2, 3, act_fn, block_sparse_moe, w1, w2, w3 |
| 6144 | E_shared·d_moe (공유 전문가 FFN 폭 — 공유 전문가 수만큼 넓힌 하나의 MLP) | 0, 1, 2, 3, act_fn, down_proj, gate_proj, shared_experts, up_proj |
| 12288 | n_h·d_v (attention 출력, o_proj 직전) | act_fn, conv, f_b_proj, g_proj, k_conv1d, k_proj, o_proj, q_conv1d, q_proj, self_attn, shared_experts, v_conv1d, v_proj |
| 18432 | n_h·(d_nope+d_rope) (MLA q_b_proj 출력) | q_b_proj, self_attn |
| 24576 | n_h·(d_nope+d_v) (MLA kv_b_proj 출력) | kv_b_proj, self_attn |
| 67584 | 2·d_ff (dense FFN gate+up 융합 투영 폭) | act_fn, mlp |

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

- **종합: PASS** (WARN 3개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 93 == 93 |
| C2 | WARN | 4 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=7168 in 93/93 layers |
| C6 | PASS | hidden_size=7168 (heuristic check, 611129 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | WARN | MoE trace-verified [router_dim(E=896):ok, top_k(None):n/a, expert_weight:grouped]; routed-token c... |
| C9 | PASS | vocab_size=163840, tie_word_embeddings=False |
| C10 | WARN | all params covered except 246194 excluded by a documented remedy (see adaptation_log) -- not a co... |
| C11 | PASS | 1050 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=320 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 526099 unmapped rows, 40 distinct raw ops: ['aten._local_scalar_dense.default', 'aten._to_copy.de... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `moonshotai/Kimi-K3` config.json @ `f831ab66814297da540d832a5235f8e904f29d06` (sha256 `711d6a903faf…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=320 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-09-02 · llm(claude) + codex(외부, 2026-09-02, 파이프라인 코드 미접근)

7건 중 4건(square 축, n_h_kda tie, d_head_kda tie, MoE 캡 1280)은 이미 맞게 렌더되고 있음을 원본 소스로 재확인. 나머지 2건(2*d_conv류 3개, n_h_kda/2 1개, 전부 KDA 청크 스캔의 루프 인덱스)은 2026-08-25에 이미 no_name_exists로 판정됐지만 한 번도 산출물에 반영되지 못했다 -- label_no_name.yaml로 닫으려 시도했으나 그 메커니즘이 stub_ambiguous 축만 인식한다는 것을 게이트 FAIL로 확인(8건 dead verdict)하고 되돌렸다. `_unname_loop_indices`(src/build_table.py)를 직접 고치는 것만이 실제 경로인데, 그 함수는 오늘 이미 두 번의 정교화 시도가 전부 함대 회귀로 되돌아간 이력이 있어(git log 참고) 이번에도 손대지 않았다. review/06-open-renames.md A62로 기록. [2026-09-02 추가] 값충돌 4건(96/128/64/6144) 전부 Codex 판정을 독립 검증 후 반영(96/128은 n_h*d_v→n_h_kda*d_head_kda 실제 버그 수정 ~390축, 64/6144는 이미 정답이었음 확인만). Codex 아키텍처 검토로 model_summary.md의 RoPE/활성함수/KV캐시/LAYER MIX 4건 추가 수정(src/summarize.py). d_head=74(값 10=d_head-d_rope, 37=d_head/2로 오표시)는 Codex가 KDA naive_chunk_kda의 청크 내부 루프 인덱스(fla/ops/kda/naive.py:101-125, i in range(1,BT))라고 특정 -- 기존 「2*d_conv류」와 같은 부류(A62)로 합류, 근본 수정은 여전히 _unname_loop_indices(두 번 회귀 이력)뿐이라 이번에도 보류.

| 판정 | 건수 |
|---|---|
| 맞음 | 4 |
| different_lowering_verified | 1 |
| 이름 없음이 정답 | 3 |
| should_be_no_name | 1 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `self_attn$` | `d_nope` | `d_v` | 96 | modeling_kimi_linear.py:432-468 -- value_states (v_head_dim-wide) reaches this reshape before o_proj; same value as the split override above, one op further downstream (prefill: [B,T,n_h,d_nope] -> [B,T,n_h*d_v]). |
| `self_attn$` | `d_nope` | `d_v` | 48 | modeling_kimi_linear.py:432-468 -- same axis as the prefill entry above, decode's size-1 T axis (decode: [B,1,n_h,d_nope] -> [B,1,n_h*d_v]). |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=29. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=29. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=73. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=1 axis=3 nth=73. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=73. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=12. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=1 axis=3 nth=12. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=12. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=91. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=93. |
| `self_attn$` | `d_head-d_rope` | `10` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=11. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=110. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=110. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=100. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=1 axis=3 nth=100. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=100. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=0 axis=3 nth=39. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=i si=1 axis=3 nth=39. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=39. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=172. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=174. |
| `self_attn$` | `d_head/2` | `37` | 69 | fla/ops/kda/naive.py:101-125 -- see block comment above; op field=o si=0 axis=3 nth=38. |
| `self_attn$` | `n_h*d_v` | `n_h_kda*d_head_kda` | 552 | modeling_kimi_linear.py:495,541,658 -- KDA's own (h d) flatten feeding o_proj; see block comment above. |
| `self_attn$` | `n_h*d_v` | `n_h_kda*d_head_kda` | 414 | modeling_kimi_linear.py:495,541,658 -- same axis as the prefill entry above, decode's size-1 T axis. |
| `self_attn$` | `n_h*d_v` | `n_h_kda*d_head_kda` | 138 | modeling_kimi_linear.py:495,541,658 -- KDA qkv 투영 직후 언플래튼, prefill nth=4; 외부 검토 2026-09-01. |
| `self_attn$` | `5` | `n_chunk` | 138 | fla/ops/kda/naive.py:108-109,166 -- see block comment above. |
| `self_attn\.(q|k|v)_conv1d` | `n_h*d_v` | `n_h_kda*d_head_kda` | 8901 | KDA(Kimi Delta Attention) 경로의 depthwise conv 다. 채널 축은 KDA 헤드 폭 (n_h_kda * d_head_kda)이고, 같은 행의 출력이 이미 그렇게 렌더링돼 있다. MLA 쪽 심볼(n_h*d_v)은 값이 같아 가려졌을 뿐이다. |
| `self_attn$` | `d_conv` | `4` | 759 | fla/ops/kda/naive.py:134-136 -- for i in range(1, BT) 안의 A[..., i, :i] 로, 이 축은 루프 prefix 길이다. d_conv(=4)와는 값만 같다. |
| `f_b_proj$` | `n_h*d_v` | `n_h_kda*d_head_kda` | 1242 | modeling_kimi_linear.py:523-524,601-607 -- f_b_proj(..., projection_size) 이후 KDA 가 h, d = self.head_dim 으로 재배열한다. 같은 모듈의 view 가 이미 [B,1,n_h_kda,d_head_kda] 로 되접는다. |
| `g_proj$` | `n_h*d_v` | `n_h_kda*d_head_kda` | 1242 | modeling_kimi_linear.py:651-658 -- KDA gate 를 d = self.head_dim 으로 재배열한 뒤 KDA 출력에 적용한다. |
| `self_attn$` | `d_nope` | `d_v` | 48 | modeling_kimi_linear.py:454-458 / eager_attention_forward:330 -- 확률 x value 의 결과는 value 의 마지막 축을 보존한다. 같은 행의 출력이 이미 d_v 다. |
| `self_attn$` | `d_nope` | `d_v` | 48 | modeling_kimi_linear.py:430-433 -- attention 출력은 value 폭을 보존한다. 같은 행의 출력이 이미 d_v 다. |
| `self_attn$` | `d_nope` | `d_v` | 48 | modeling_kimi_linear.py:430-433 -- 위와 같다 (decode). |
| `self_attn$` | `d_nope` | `d_v` | 48 | 같은 행의 입력이 [B, n_h, 1, d_v, 1] 이고 view 는 마지막 두 축만 합친다. modeling_kimi_linear.py:430-433 -- attention 출력은 value 폭이다. |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
