# 리뷰 패킷 — LiquidAI/LFM2-8B-A1B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `c1c44ff9fc00db3ebf4516970563f5f383d23670` / 트레이스 seq_len(T) = 16
> 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 1. 이 산출물이 무엇인가

Hugging Face의 **공식 config + modeling 코드를 meta device에서 실제로 forward 실행**하고,
그 실행을 PyTorch dispatch(ATen) 레벨에서 가로채 op·shape·의존관계를 기록한 것입니다.
가중치는 없지만(shape 계산에 불필요) 연산 그래프는 실제로 실행된 것이며, 값을 지어내지
않습니다. shape은 아키텍처 심볼(`B, T, d_model, n_h, …`)로 렌더됩니다.

**따라서 트레이스 자체(어떤 op이 어떤 크기로 도는가)는 관측값이고, 검토 대상은
"그 축에 붙은 이름이 맞는가"입니다.**

## 2. 심볼표 (이 모델에서 각 이름이 갖는 값)

```
  L            = 24
  d_model      = 2048
  n_h          = 32
  n_kv         = 8
  d_head       = 64
  d_ff         = 7168
  d_shared     = None
  V            = 65536
  ctx          = 128000
  E            = 32
  E_shared     = 0
  k            = 4
  n_grp        = None
  k_grp        = None
  d_moe        = 1792
  w_local      = None
  n_sink       = None
  layer_sched  = None
  c_kv         = None
  d_nope       = None
  d_v          = None
  c_q          = None
  d_rope       = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  d_state      = None
  n_g_ssm      = None
  n_h_ssm      = None
  d_chunk      = None
  d_head_ssm   = None
  d_conv       = 3
  n_mem        = None
  r_lora       = None
  d_attn       = None
  n_h_lin_k    = None
  n_h_lin_v    = None
  d_head_lin_k = None
  d_head_lin_v = None
  d_conv_lin   = None
```

## 3. 모델 요약 산출물

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
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
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

2026-08-12 · llm(claude, A/B/C절 기계 선별 + 소스 대조)



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


## 4. 검증 체크리스트 결과

```
# Extraction Report -- LiquidAI/LFM2-8B-A1B @ c1c44ff9fc00db3ebf4516970563f5f383d23670

C1   PASS   24 == 24
C2   WARN   3 trace clusters vs 2 config-schedule signatures ['layer_types'] -- review (mask-only heterogeneity is op-invisible)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 24/24 layers
C6   PASS   hidden_size=2048 (heuristic check, 444 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   WARN   MoE trace-verified [router_dim(E=32):ok, top_k(4):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=65536, tie_word_embeddings=True
C10  PASS   all 212 params covered
C11  PASS   43 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1351 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.div.Tensor', 'aten.empty_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              arange           [] -> [B]
  model                                              arange           [] -> [T]
  model                                              elementwise_add  [T] -> [T]
  model                                              unsqueeze        [B] -> [B,1]
  model                                              unsqueeze        [B,1] -> [B,1,1]
  model                                              unsqueeze        [B,1,1] -> [B,1,1,1]
  model                                              unsqueeze        [T] -> [B,T]
  model                                              unsqueeze        [B,T] -> [B,1,T]
  model                                              unsqueeze        [B,1,T] -> [B,1,T,1]
  model                                              le               [B,1,1,T]*[B,1,T,1] -> [B,1,T,T]
  model                                              expand           [B,1,T,T] -> [B,1,T,T]
  model                                              scalar_tensor    [] -> []
  model                                              where            [B,1,T,T]*[]*[] -> [B,1,T,T]
  model.pos_emb                                      unsqueeze        [32] -> [B,32]
  model.pos_emb                                      unsqueeze        [B,32] -> [B,32,1]
  model.pos_emb                                      expand           [B,32,1] -> [B,32,1]
  model.pos_emb                                      unsqueeze        [B,T] -> [B,1,T]
  model.pos_emb                                      _to_copy         [B,1,T] -> [B,1,T]
  model.pos_emb                                      view             [B,32,1] -> [B,32,1]
  model.pos_emb                                      expand           [B,1,T] -> [B,1,T]
  model.pos_emb                                      view             [B,1,T] -> [B,1,T]
  model.pos_emb                                      batched_matmul   [B,32,1]*[B,1,T] -> [B,32,T]
  model.pos_emb                                      _unsafe_view     [B,32,T] -> [B,32,T]
  model.pos_emb                                      transpose        [B,32,T] -> [B,T,d_head/2]
  model.pos_emb                                      concat           [B,T,d_head/2]*[B,T,d_head/2] -> [B,T,d_head]
  model.pos_emb                                      cos              [B,T,d_head] -> [B,T,d_head]
  model.pos_emb                                      elementwise_mul  [B,T,d_head] -> [B,T,d_head]
  model.pos_emb                                      sin              [B,T,d_head] -> [B,T,d_head]
  model.pos_emb                                      _to_copy         [B,T,d_head] -> [B,T,d_head]
  model.layers.N.operator_norm                       _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.operator_norm                       pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.operator_norm                       mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.operator_norm                       elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.operator_norm                       rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.operator_norm                       elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.operator_norm                       elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.conv.in_proj                        t                [3*d_model,d_model] -> w=[3*d_model,d_model] [d_model,3*d_model]
  model.layers.N.conv.in_proj                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.conv.in_proj                        matmul           [T,d_model]*[d_model,3*d_model] -> w=[3*d_model,d_model] [T,3*d_model]
  model.layers.N.conv.in_proj                        _unsafe_view     [T,3*d_model] -> [B,T,3*d_model]
  model.layers.N.conv                                transpose        [B,T,3*d_model] -> [B,3*d_model,T]
  model.layers.N.conv                                split            [B,3*d_model,T] -> [B,d_model,T]*[B,d_model,T]*[B,d_model,T]
  model.layers.N.conv                                elementwise_mul  [B,d_model,T]*[B,d_model,T] -> [B,d_model,T]
  model.layers.N.conv                                constant_pad_nd  [B,d_model,T] -> [B,d_model,d_conv]
  model.layers.N.conv                                zeros            [] -> [B,d_model,d_conv]
  model.layers.N.conv                                slice            [B,d_model,d_conv] -> [B,d_model,d_conv]
  model.layers.N.conv                                copy_            [B,d_model,d_conv]*[B,d_model,d_conv] -> [B,d_model,d_conv]
  model.layers.N.conv.conv                           conv1d           [B,d_model,T]*[d_model,B,d_conv] -> w=[d_model,1,d_conv] [B,d_model,18]
  model.layers.N.conv                                slice            [B,d_model,18] -> [B,d_model,T]
  model.layers.N.conv                                transpose        [B,d_model,T] -> [B,T,d_model]
  model.layers.N.conv.out_proj                       t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.conv.out_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.conv.out_proj                       matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  model.layers.N.conv.out_proj                       _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.ffn_norm                            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.ffn_norm                            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.ffn_norm                            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.ffn_norm                            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.ffn_norm                            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.ffn_norm                            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.ffn_norm                            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.feed_forward.w1                     t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.w1                     view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.w1                     matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.w1                     _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward                        silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.w3                     t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.w3                     view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.w3                     matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.w3                     _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.w2                     t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.w2                     view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.feed_forward.w2                     matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.feed_forward.w2                     _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               _to_copy         [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               pow              [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               mean             [B,T,n_h,d_head] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_layernorm               elementwise_add  [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_layernorm               rsqrt            [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_layernorm               elementwise_mul  [B,T,n_h,d_head]*[B,T,n_h,1] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               elementwise_mul  [d_head]*[B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               _to_copy         [B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               pow              [B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               mean             [B,T,n_kv,d_head] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_layernorm               elementwise_add  [B,T,n_kv,1] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_layernorm               rsqrt            [B,T,n_kv,1] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_layernorm               elementwise_mul  [B,T,n_kv,d_head]*[B,T,n_kv,1] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               elementwise_mul  [d_head]*[B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,T,d_head]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T,d_head] -> [B,n_kv,1,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           view             [B,T,n_h,d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.out_proj                  t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.out_proj                  view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.out_proj                  matmul           [T,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [T,n_h*d_head]
  model.layers.N.self_attn.out_proj                  _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.feed_forward                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.gate                   t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.feed_forward.gate                   matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.feed_forward.gate                   sigmoid          [T,E] -> [T,E]
  model.layers.N.feed_forward.gate                   elementwise_add  [T,E]*[E] -> [T,E]
  model.layers.N.feed_forward.gate                   topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.feed_forward.gate                   gather           [T,E]*[T,k] -> [T,k]
  model.layers.N.feed_forward.gate                   sum              [T,k] -> [T,1]
  model.layers.N.feed_forward.gate                   elementwise_add  [T,1] -> [T,1]
  model.layers.N.feed_forward.gate                   div              [T,k]*[T,1] -> [T,k]
  model.layers.N.feed_forward.gate                   elementwise_mul  [T,k] -> [T,k]
  model.layers.N.feed_forward.experts                view             [T,k] -> [k*T]
  model.layers.N.feed_forward.experts                sort             [k*T] -> [k*T]*[k*T]
  model.layers.N.feed_forward.experts                floor_divide     [k*T] -> [k*T]
  model.layers.N.feed_forward.experts                index            [T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.feed_forward.experts                index            [k*T]*[k*T] -> [k*T]
  model.layers.N.feed_forward.experts                _to_copy         [k*T] -> [k*T]
  model.layers.N.feed_forward.experts                histc            [k*T] -> [E]
  model.layers.N.feed_forward.experts                cumsum           [E] -> [E]
  model.layers.N.feed_forward.experts                ge               [k*T] -> [k*T]
  model.layers.N.feed_forward.experts                unsqueeze        [k*T] -> [k*T,B]
  model.layers.N.feed_forward.experts                clamp_           [k*T] -> [k*T]
  model.layers.N.feed_forward.experts                masked_fill_     [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.feed_forward.experts                transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.feed_forward.experts                grouped_matmul   [k*T,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k*T,2*d_moe]
  model.layers.N.feed_forward.experts                split            [k*T,2*d_moe] -> [k*T,d_moe]*[k*T,d_moe]
  model.layers.N.feed_forward.experts                silu             [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.feed_forward.experts                elementwise_mul  [k*T,d_moe]*[k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.feed_forward.experts                transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.feed_forward.experts                grouped_matmul   [k*T,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k*T,d_model]
  model.layers.N.feed_forward.experts                elementwise_mul  [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.feed_forward.experts                empty_like       [k*T] -> [k*T]
  model.layers.N.feed_forward.experts                arange           [] -> [k*T]
  model.layers.N.feed_forward.experts                index_put_       [k*T]*[k*T]*[k*T] -> [k*T]
  model.layers.N.feed_forward.experts                index            [k*T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.feed_forward.experts                view             [k*T,d_model] -> [T,k,d_model]
  model.layers.N.feed_forward.experts                sum              [T,k,d_model] -> [T,d_model]
  model.layers.N.feed_forward                        view             [T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.embedding_norm                               _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.embedding_norm                               pow              [B,T,d_model] -> [B,T,d_model]
  model.embedding_norm                               mean             [B,T,d_model] -> [B,T,1]
  model.embedding_norm                               elementwise_add  [B,T,1] -> [B,T,1]
  model.embedding_norm                               rsqrt            [B,T,1] -> [B,T,1]
  model.embedding_norm                               elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.embedding_norm                               elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
                                                     alias            [B,T,d_model] -> [B,T,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,T,d_model] -> [T,d_model]
  lm_head                                            matmul           [T,d_model]*[d_model,V] -> w=[V,d_model] [T,V]
  lm_head                                            _unsafe_view     [T,V] -> [B,T,V]
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  model                                              arange           [] -> [B]
  model                                              elementwise_add  [B] -> [B]
  model                                              arange           [] -> [T+1]
  model                                              elementwise_add  [T+1] -> [T+1]
  model                                              unsqueeze        [B] -> [B,1]
  model                                              unsqueeze        [B,1] -> [B,1,1]
  model                                              unsqueeze        [B,1,1] -> [B,1,1,1]
  model                                              unsqueeze        [T+1] -> [B,T+1]
  model                                              unsqueeze        [B,T+1] -> [B,1,T+1]
  model                                              unsqueeze        [B,1,T+1] -> [B,1,1,T+1]
  model                                              le               [B,1,1,T+1]*[B,1,1,1] -> [B,1,1,T+1]
  model                                              expand           [B,1,1,T+1] -> [B,1,1,T+1]
  model                                              scalar_tensor    [] -> []
  model                                              where            [B,1,1,T+1]*[]*[] -> [B,1,1,T+1]
  model.pos_emb                                      unsqueeze        [32] -> [B,32]
  model.pos_emb                                      unsqueeze        [B,32] -> [B,32,1]
  model.pos_emb                                      expand           [B,32,1] -> [B,32,1]
  model.pos_emb                                      unsqueeze        [B,1] -> [B,1,1]
  model.pos_emb                                      _to_copy         [B,1,1] -> [B,1,1]
  model.pos_emb                                      view             [B,32,1] -> [B,32,1]
  model.pos_emb                                      expand           [B,1,1] -> [B,1,1]
  model.pos_emb                                      view             [B,1,1] -> [B,1,1]
  model.pos_emb                                      batched_matmul   [B,32,1]*[B,1,1] -> [B,32,1]
  model.pos_emb                                      _unsafe_view     [B,32,1] -> [B,32,1]
  model.pos_emb                                      transpose        [B,32,1] -> [B,1,32]
  model.pos_emb                                      concat           [B,1,32]*[B,1,32] -> [B,1,d_head]
  model.pos_emb                                      cos              [B,1,d_head] -> [B,1,d_head]
  model.pos_emb                                      elementwise_mul  [B,1,d_head] -> [B,1,d_head]
  model.pos_emb                                      sin              [B,1,d_head] -> [B,1,d_head]
  model.pos_emb                                      _to_copy         [B,1,d_head] -> [B,1,d_head]
  model.layers.N.operator_norm                       _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.operator_norm                       pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.operator_norm                       mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.operator_norm                       elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.operator_norm                       rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.operator_norm                       elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.operator_norm                       elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.conv.in_proj                        t                [3*d_model,d_model] -> w=[3*d_model,d_model] [d_model,3*d_model]
  model.layers.N.conv.in_proj                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.conv.in_proj                        matmul           [B,d_model]*[d_model,3*d_model] -> w=[3*d_model,d_model] [B,3*d_model]
  model.layers.N.conv.in_proj                        _unsafe_view     [B,3*d_model] -> [B,1,3*d_model]
  model.layers.N.conv                                transpose        [B,1,3*d_model] -> [B,3*d_model,1]
  model.layers.N.conv                                split            [B,3*d_model,1] -> [B,d_model,1]*[B,d_model,1]*[B,d_model,1]
  model.layers.N.conv                                elementwise_mul  [B,d_model,1]*[B,d_model,1] -> [B,d_model,1]
  model.layers.N.conv                                concat           [B,d_model,d_conv]*[B,d_model,1] -> [B,d_model,d_conv+1]
  model.layers.N.conv                                slice            [B,d_model,d_conv+1] -> [B,d_model,d_conv]
  model.layers.N.conv                                copy_            [B,d_model,d_conv]*[B,d_model,d_conv] -> [B,d_model,d_conv]
  model.layers.N.conv                                select           [d_model,B,d_conv] -> w=[d_model,1,d_conv] [d_model,d_conv]
  model.layers.N.conv                                elementwise_mul  [B,d_model,d_conv]*[d_model,d_conv] -> [B,d_model,d_conv]
  model.layers.N.conv                                sum              [B,d_model,d_conv] -> [B,d_model]
  model.layers.N.conv                                unsqueeze        [B,d_model] -> [B,d_model,1]
  model.layers.N.conv                                transpose        [B,d_model,1] -> [B,1,d_model]
  model.layers.N.conv.out_proj                       t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.conv.out_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.conv.out_proj                       matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  model.layers.N.conv.out_proj                       _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.ffn_norm                            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.ffn_norm                            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.ffn_norm                            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.ffn_norm                            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.ffn_norm                            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.ffn_norm                            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.ffn_norm                            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.feed_forward.w1                     t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.w1                     view             [B,1,d_model] -> [B,d_model]
  model.layers.N.feed_forward.w1                     matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.w1                     _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward                        silu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward.w3                     t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.w3                     view             [B,1,d_model] -> [B,d_model]
  model.layers.N.feed_forward.w3                     matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.w3                     _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward.w2                     t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.w2                     view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.feed_forward.w2                     matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.feed_forward.w2                     _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               _to_copy         [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               pow              [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               mean             [B,1,n_h,d_head] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_layernorm               elementwise_add  [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_layernorm               rsqrt            [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_layernorm               elementwise_mul  [B,1,n_h,d_head]*[B,1,n_h,1] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_layernorm               elementwise_mul  [d_head]*[B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               _to_copy         [B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               pow              [B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               mean             [B,1,n_kv,d_head] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_layernorm               elementwise_add  [B,1,n_kv,1] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_layernorm               rsqrt            [B,1,n_kv,1] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_layernorm               elementwise_mul  [B,1,n_kv,d_head]*[B,1,n_kv,1] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_layernorm               elementwise_mul  [d_head]*[B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,n_h*d_head] -> w=[n_kv*d_head,n_h*d_head] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,n_h*d_head] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,1,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T+1,d_head] -> [B,n_kv,1,T+1,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           _unsafe_view     [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T+1,d_head] -> [n_h,T+1,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.out_proj                  t                [n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [n_h*d_head,n_h*d_head]
  model.layers.N.self_attn.out_proj                  view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.out_proj                  matmul           [B,n_h*d_head]*[n_h*d_head,n_h*d_head] -> w=[n_h*d_head,n_h*d_head] [B,n_h*d_head]
  model.layers.N.self_attn.out_proj                  _unsafe_view     [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.feed_forward                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.feed_forward.gate                   t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.feed_forward.gate                   matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.feed_forward.gate                   sigmoid          [B,E] -> [B,E]
  model.layers.N.feed_forward.gate                   elementwise_add  [B,E]*[E] -> [B,E]
  model.layers.N.feed_forward.gate                   topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.feed_forward.gate                   gather           [B,E]*[B,k] -> [B,k]
  model.layers.N.feed_forward.gate                   sum              [B,k] -> [B,1]
  model.layers.N.feed_forward.gate                   elementwise_add  [B,1] -> [B,1]
  model.layers.N.feed_forward.gate                   div              [B,k]*[B,1] -> [B,k]
  model.layers.N.feed_forward.gate                   elementwise_mul  [B,k] -> [B,k]
  model.layers.N.feed_forward.experts                view             [B,k] -> [k]
  model.layers.N.feed_forward.experts                sort             [k] -> [k]*[k]
  model.layers.N.feed_forward.experts                floor_divide     [k] -> [k]
  model.layers.N.feed_forward.experts                index            [B,d_model]*[k] -> [k,d_model]
  model.layers.N.feed_forward.experts                index            [k]*[k] -> [k]
  model.layers.N.feed_forward.experts                _to_copy         [k] -> [k]
  model.layers.N.feed_forward.experts                histc            [k] -> [E]
  model.layers.N.feed_forward.experts                cumsum           [E] -> [E]
  model.layers.N.feed_forward.experts                ge               [k] -> [k]
  model.layers.N.feed_forward.experts                unsqueeze        [k] -> [k,B]
  model.layers.N.feed_forward.experts                clamp_           [k] -> [k]
  model.layers.N.feed_forward.experts                masked_fill_     [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.feed_forward.experts                transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.feed_forward.experts                grouped_matmul   [k,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k,2*d_moe]
  model.layers.N.feed_forward.experts                split            [k,2*d_moe] -> [k,d_moe]*[k,d_moe]
  model.layers.N.feed_forward.experts                silu             [k,d_moe] -> [k,d_moe]
  model.layers.N.feed_forward.experts                elementwise_mul  [k,d_moe]*[k,d_moe] -> [k,d_moe]
  model.layers.N.feed_forward.experts                transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.feed_forward.experts                grouped_matmul   [k,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k,d_model]
  model.layers.N.feed_forward.experts                elementwise_mul  [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.feed_forward.experts                empty_like       [k] -> [k]
  model.layers.N.feed_forward.experts                arange           [] -> [k]
  model.layers.N.feed_forward.experts                index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.feed_forward.experts                index            [k,d_model]*[k] -> [k,d_model]
  model.layers.N.feed_forward.experts                view             [k,d_model] -> [B,k,d_model]
  model.layers.N.feed_forward.experts                sum              [B,k,d_model] -> [B,d_model]
  model.layers.N.feed_forward                        view             [B,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.11                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.17                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.23                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.embedding_norm                               _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.embedding_norm                               pow              [B,1,d_model] -> [B,1,d_model]
  model.embedding_norm                               mean             [B,1,d_model] -> [B,1,1]
  model.embedding_norm                               elementwise_add  [B,1,1] -> [B,1,1]
  model.embedding_norm                               rsqrt            [B,1,1] -> [B,1,1]
  model.embedding_norm                               elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.embedding_norm                               elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
                                                     alias            [B,1,d_model] -> [B,1,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,1,d_model] -> [B,d_model]
  lm_head                                            matmul           [B,d_model]*[d_model,V] -> w=[V,d_model] [B,V]
  lm_head                                            _unsafe_view     [B,V] -> [B,1,V]
```

## 6. 이미 알려진 한계 — 다시 보고하지 않아도 됨

- **값이 같은 서로 다른 개념은 자동 판별 불가.** 예: gpt-oss는 expert 블록 **안**에서
  `d_model`=`d_moe`=2880, Zamba2는 `n_h*d_head`=`2*d_model`=4096. module_path만으로는 어느
  개념인지 못 가른다. 이건 이미 `rules/structures/`에 명시해 뒀으므로 다시 보고하지 않아도 된다.
- **이미 고쳐서 다시 보고하지 않아도 되는 것**(2026-07-30~31):
  `post_attention_layernorm`의 잔차폭(→ `d_model`), MoE 라우터 입력폭(→ `d_model`),
  gpt-oss sliding 레이어의 KV 상한(→ `w_local`)과 attention sink 표기(→ `n_sink`),
  Llama-3.1-405B의 토큰축(재추적으로 T 충돌 해소), DeepSeek-V4 압축기의 블록 축(→ `m_csa`/`m_hca`),
  한 shape에 `n_h`와 `n_kv`가 동시에 나오던 문제(게이트가 자동 검사한다).
- **`d_model` ↔ `n_h*d_head`가 같은 텐서에 다르게 붙는 경우**는 표준 트랜스포머에서 두 값이
  정의상 같기 때문이며, 둘 다 참인 이름이다. 강제 통일하면 오히려 정보가 사라진다.
- **Qwen3-Next의 3,7,9,11… 같은 작은 정수**는 DeltaNet 청크 스캔의 언롤된 루프 경계다.
  아키텍처 상수가 아니므로 심볼을 붙이면 거짓이 된다.


---

## 리뷰 요청

당신은 이 산출물이 **실제 모델 아키텍처를 정확히 기술하는지** 판정해야 합니다.
규칙 체크리스트는 이미 전부 통과한 상태입니다. 그러니 규칙이 못 잡는 것을 찾아주세요.

### 반드시 대조할 것
1. 해당 모델의 **공식 HF modeling 코드**와 config 클래스
2. **논문 / 기술 리포트**, 벤더 공식 블로그
3. **vLLM · SGLang 등 독립 서빙 구현**의 같은 모델 코드·해설
4. 신뢰도 높은 아키텍처 정리 자료

### 특히 봐야 할 것
- 심볼 이름이 **그 위치에서 실제로 의미하는 것과 맞는가**
  (값이 맞아도 개념이 틀릴 수 있음 — 이게 지금까지 나온 오류의 거의 전부였다)
- attention 계열 판정(MHA/GQA/MQA/MLA/…)이 실제 구현과 맞는가.
  **config 필드를 그대로 믿지 말 것** — 필드가 있어도 실제 동작이 다를 수 있다
  (Falcon은 `num_kv_heads=71`이지만 `multi_query=True`라 실제 KV head는 1개였다)
- KV cache 계산의 **전제**가 맞는가 (어느 레이어가 캐시를 갖는지, K와 V가 별개인지)
- config에 없는데 코드에 하드코딩된 구조가 누락되지 않았는가
  (Llama-4는 shared expert 개수 필드가 없고 코드에 1개로 고정돼 있다)
- 이 아키텍처의 **핵심 특징 중 산출물에 아예 안 나타난 것**이 있는가
- **decode 표(5-2)를 반드시 보세요.** prefill에는 없는 축이 거기 있습니다 —
  sliding 레이어의 KV 상한, 캐시 길이, attention sink가 붙는 score 폭.
  실제로 이 표가 패킷에 없던 동안 gpt-oss의 sliding 컨텍스트 오라벨이 그대로 남아 있었습니다.
- **모듈 이름이 "무엇을 계산하는가"가 아니라 "블록 안 어디인가"를 뜻하는 곳**을 의심하세요.
  지금까지 나온 오류의 다수가 여기서 나왔습니다 — `post_attention_layernorm`은 attention이
  아니라 그 뒤의 잔차 정규화이고, `mlp.router`는 FFN 내부가 아니라 잔차를 읽는 라우터입니다.
- **같은 (모듈, op)인데 shape 표기가 갈리는 줄**을 찾으세요. 표본은 그 축을 일부러 접지
  않았습니다 — 라벨 오류는 정의상 거기서 드러납니다.

### 출력 형식 (반드시 지킬 것)

각 지적은 아래 표 형태로. **관찰과 가설을 반드시 분리**하세요.

| # | 관찰(사실) | 근거 | 내 가설(원인) | 확신도 | 검증 방법 |
|---|---|---|---|---|---|

- **관찰**: 패킷에서 직접 인용. "X라고 적혀 있다"
- **근거**: 공식 소스의 **파일명 + 함수/클래스명 + 인용문**, 또는 URL.
  `op_id`는 근거로 쓰지 마세요 — 재트레이싱하면 번호가 바뀝니다.
  대신 `module_path`와 shape 내용으로 지목하세요.
- **내 가설**: 왜 그렇게 됐다고 보는지. **틀려도 됩니다. 다만 관찰과 섞지 마세요.**
- **확신도**: 확실 / 아마도 / 추측
- **검증 방법**: 우리가 이 주장을 어떻게 확인하면 되는지 (구체적으로)

확실하지 않으면 "확실"이라고 쓰지 마세요. **틀린 지적보다 놓친 지적이 낫습니다** —
틀린 지적을 검증하는 비용이 더 큽니다.

