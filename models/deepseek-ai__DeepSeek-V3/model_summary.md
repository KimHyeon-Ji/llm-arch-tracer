# Model Summary -- deepseek-ai/DeepSeek-V3

## 기본 정보

- revision: `e815299b0bcbac849fa540c768ef21845365c9eb`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 24
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 671.03B total, 37.55B active (5.6% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 163,840  _(config max_position_embeddings; yarn 스케일(원본 4096×40) — 벤더 광고 컨텍스트와 다를 수 있음)_ |
| 3 | DATE | 2024-12-25  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MLA |
| 6 | LAYER MIX | 61× MLA  (FFN: 3 dense + 58 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 68.6 KiB (Low) |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=256, top-8, +1 shared, sigmoid gating/aux-loss-free); dense-prefix 3 layer(s) |
| 9 | Related concepts | RMSNorm, RoPE, MLA, MoE, shared expert, sigmoid-gating, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `deepseek_v3` |
| attention | MLA — KV latent compression (kv_lora_rank=512, q_lora_rank=1536); 헤드 q/k = nope(128)+rope(64)=192, v=128, n_h=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000), yarn scaling |
| FFN | MoE — 256 routed experts, top-8 + 1 shared, expert intermediate 2048, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | compressed MLA latent ≈ kv_lora_rank=512 (+decoupled RoPE dim) / token / layer |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 61 |
| d_model | 7168 |
| n_h | 128 |
| n_kv | 128 |
| d_head | 64 |
| d_ff | 18432 |
| V | 129280 |
| ctx | 163840 |
| E | 256 |
| E_shared | 1 |
| k | 8 |
| n_grp | 8 |
| k_grp | 4 |
| d_moe | 2048 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
| c_kv | 512 |
| d_nope | 128 |
| d_v | 128 |
| c_q | 1536 |
| d_rope | 64 |
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
| n_h_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| n_h_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_conv_lin | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **226,166개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 73,511 | 32.50% |
| 스코프 없는 심볼 | 62,721 | 27.73% |
| 이 모듈 스코프의 심볼 | 55,391 | 24.49% |
| 이 모듈 스코프의 유도식 | 31,337 | 13.86% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 2,440 | 1.08% |
| 이름 없음 (정수 유지) | 766 | 0.34% |

등록된 규칙 **222,960축**, 약한 근거 2,440축, 휴리스틱 **0축 (0.0%)**, 이름 없음 766축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | gate, rotary_emb, self_attn |
| 192 | d_nope + d_rope (MLA q/k head 폭) | act_fn, experts, self_attn |
| 576 | c_kv+d_rope (MLA kv_a_proj_with_mqa 출력) | kv_a_proj_with_mqa, self_attn |
| 4096 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |
| 16384 | n_h·d_v (attention 출력, o_proj 직전) | o_proj, self_attn |
| 24576 | (n_h + 2·n_kv)·d_head (fused QKV 투영 폭 — Q·K·V 한 행렬) | q_b_proj, self_attn |
| 32768 | n_h·(d_nope+d_v) (MLA kv_b_proj 출력) | kv_b_proj, self_attn |

## 레이어 구조

- layer 0-2: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 3-60: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 3개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 61 == 61 |
| C2 | WARN | 2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=7168 in 61/61 layers |
| C6 | PASS | hidden_size=7168 (heuristic check, 8777 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | WARN | MoE trace-verified [router_dim(E=256):ok, top_k(8):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=129280, tie_word_embeddings=False |
| C10 | PASS | all 909 params covered |
| C11 | PASS | 367 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=24 >= required=24 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 7088 unmapped rows, 33 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `deepseek-ai/DeepSeek-V3` config.json @ `e815299b0bcbac849fa540c768ef21845365c9eb` (sha256 `82d3503724d6…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=24 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 자기모순 추적 + 소스 대조)

의뢰서의 `2*d_moe` 는 이름이 옳았다 — 산술 휴리스틱이 내던 것을 규칙으로 승격했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 교정 필요 | 3 |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.self_attn` | value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지 | `d_nope` | `d_v` | 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 … |
| `model.layers.*.self_attn` | q/k split 둘째 조각 (64) | `d_head` | `d_rope` | `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
