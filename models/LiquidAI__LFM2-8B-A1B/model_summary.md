# Model Summary -- LiquidAI/LFM2-8B-A1B

## 기본 정보

- revision: `c1c44ff9fc00db3ebf4516970563f5f383d23670`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 8.34B total, 1.56B active (18.7% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 128,000  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-10-07  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 18× conv, 6× GQA  (FFN: 24× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 12.0 KiB (Very low) over 6 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=32, top-4, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, sigmoid-gating, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `lfm2_moe` |
| attention | GQA — 32 query : 8 kv heads (repeat 4), d_head=64 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=1000000.0) |
| FFN | MoE — 32 routed experts, top-4, expert intermediate 1792, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·64 = 1024 elems / token / layer; all 24 layers ⇒ 24576 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 24 |
| d_model | 2048 |
| n_h | 32 |
| n_kv | 8 |
| d_head | 64 |
| d_ff | 7168 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 65536 |
| ctx | 128000 |
| E | 32 |
| E_shared | 0 |
| k | 4 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 1792 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 18× conv, 6× full_attention (총 24층) |
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
| d_state | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| n_g_ssm | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| n_h_ssm | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| d_chunk | —  _(해당 없음: 이 모델은 `ssm_chunk` 계열 구조를 쓰지 않음)_ |
| d_head_ssm | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| d_conv | 3 |
| n_mem | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| r_lora | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| d_attn | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| n_h_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| n_h_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_conv_lin | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **42,520개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 스코프 없는 심볼 | 13,785 | 32.42% |
| 런타임 축 (B/T/1) | 13,041 | 30.67% |
| 이 모듈 스코프의 심볼 | 9,888 | 23.25% |
| 이 모듈 스코프의 유도식 | 4,655 | 10.95% |
| 휴리스틱: 심볼의 배수 | 524 | 1.23% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 444 | 1.04% |
| 이름 없음 (정수 유지) | 144 | 0.34% |
| 스코프가 배제한 심볼 | 39 | 0.09% |

등록된 규칙 **41,369축**, 약한 근거 483축, 휴리스틱 **524축 (1.23%)**, 이름 없음 144축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.0.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 26 |
| `model.layers.1.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.3.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.4.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.5.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.7.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.8.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.9.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.11.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.12.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.13.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |
| `model.layers.15.conv.in_proj` | `3*d_model` | 휴리스틱: 심볼의 배수 | 24 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 18 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv |
| 512 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 3584 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |

## 레이어 구조

- layer 0-1: conv, feed_forward, ffn_norm, operator_norm
- layer 2: feed_forward, ffn_norm, operator_norm, self_attn
- layer 3-5: conv, feed_forward, ffn_norm, operator_norm
- layer 6: feed_forward, ffn_norm, operator_norm, self_attn
- layer 7-9: conv, feed_forward, ffn_norm, operator_norm
- layer 10: feed_forward, ffn_norm, operator_norm, self_attn
- layer 11-13: conv, feed_forward, ffn_norm, operator_norm
- layer 14: feed_forward, ffn_norm, operator_norm, self_attn
- layer 15-17: conv, feed_forward, ffn_norm, operator_norm
- layer 18: feed_forward, ffn_norm, operator_norm, self_attn
- layer 19-20: conv, feed_forward, ffn_norm, operator_norm
- layer 21: feed_forward, ffn_norm, operator_norm, self_attn
- layer 22-23: conv, feed_forward, ffn_norm, operator_norm

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 24 == 24 |
| C2 | WARN | 3 trace clusters vs 2 config-schedule signatures ['layer_types'] -- review (mask-only heterogenei... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2048 in 24/24 layers |
| C6 | PASS | hidden_size=2048 (heuristic check, 444 flagged) |
| C7 | PASS | GQA 32:8 (repeat factor 4) |
| C8 | WARN | MoE trace-verified [router_dim(E=32):ok, top_k(4):ok, expert_weight:grouped]; routed-token count ... |
| C9 | PASS | vocab_size=65536, tie_word_embeddings=True |
| C10 | PASS | all 212 params covered |
| C11 | PASS | 43 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1351 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `LiquidAI/LFM2-8B-A1B` config.json @ `c1c44ff9fc00db3ebf4516970563f5f383d23670` (sha256 `8c4f4b8e6d5d…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 외부 검토 지적 반영 + 미답변 4건 판정)



| 판정 | 건수 |
|---|---|
| 교정 필요 | 2 |
| 미확정 | 1 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `^model\.pos_emb$` | `E` | `d_head/2` | 3 | 실측 `[1, 16, 32]` 이고 바로 옆 concat 이 `[1, 16, 64]`(=d_head) 다 — rotary 의 inv_freq 절반 축이다. 전문가 수 E(=32)가 값이 같아 그 이름이 붙었다. rotary 모듈에 MoE 심볼이 있을 수 없다. (C절 전수 점검 2026-08-12) |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.conv` | 설명되지 않는 정수 18·32 | `정수` | 미확정 | `model.layers.*.conv` 안에서만 나타나고 config 어느 필드와도 대응되지 않는다. 커널 폭 3 은 `conv_L_cache` 로 접지했지만 이 둘은 소스에서 근거를 못 찾았다. `develop/verify/references.yaml` 에 사유와 함께 등재했다 — 이름을 지어내지 않는다. |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
