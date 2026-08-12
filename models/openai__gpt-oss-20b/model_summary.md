# Model Summary -- openai/gpt-oss-20b

## 기본 정보

- revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 264
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 20.91B total, 4.19B active (20.0% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings; yarn 스케일(원본 4096×32.0) — 벤더 광고 컨텍스트와 다를 수 있음)_ |
| 3 | DATE | 2025-08-04  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 12× sliding_attention, 12× GQA  (FFN: 24× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 48.0 KiB (Low) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=32, top-4, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, sigmoid-gating |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `gpt_oss` |
| attention | GQA — 64 query : 8 kv heads (repeat 8), d_head=64; sliding window 128 on part of layers (hybrid local/global); attention sink (1개 학습형 로짓 열이 softmax 분모에 추가 — KV는 늘지 않고 score 폭만 +1) |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=150000), yarn scaling |
| FFN | MoE — 32 routed experts, top-4, expert intermediate 2880, ? [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·64 = 1024 elems / token / layer; all 24 layers ⇒ 24576 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 24 |
| d_model | 2880 |
| n_h | 64 |
| n_kv | 8 |
| d_head | 64 |
| d_ff | 2880 |
| V | 201088 |
| ctx | 131072 |
| E | 32 |
| E_shared | 0 |
| k | 4 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 2880 |
| w_local | 128 |
| n_sink | 1 |
| layer_sched | 12× sliding_attention, 12× full_attention (총 24층) |
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

shape 축 **70,969개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 21,117 | 29.76% |
| 이 모듈 스코프의 심볼 | 21,012 | 29.61% |
| 스코프 없는 심볼 | 15,797 | 22.26% |
| 이 모듈 스코프의 유도식 | 12,195 | 17.18% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 752 | 1.06% |
| 이름 없음 (정수 유지) | 96 | 0.14% |

등록된 규칙 **70,121축**, 약한 근거 752축, 휴리스틱 **0축 (0.0%)**, 이름 없음 96축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 127 | w_local − 1 (sliding window mask 밴드 폭) | self_attn |
| 265 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | self_attn |
| 512 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 1056 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | experts |
| 4096 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_proj, self_attn |
| 5760 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |

## 레이어 구조

- layer 0: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 1: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 2: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 3: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 4: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 5: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 6: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 7: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 8: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 9: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 10: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 11: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 12: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 13: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 14: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 15: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 16: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 17: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 18: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 19: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 20: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 21: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 22: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 23: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 24 == 24 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2880 in 24/24 layers |
| C6 | PASS | hidden_size=2880 (heuristic check, 2208 flagged) |
| C7 | PASS | GQA 64:8 (repeat factor 8) |
| C8 | WARN | MoE trace-verified [router_dim(E=32):ok, top_k(4):ok, expert_weight:grouped]; routed-token count ... |
| C9 | PASS | vocab_size=201088, tie_word_embeddings=False |
| C10 | PASS | all 411 params covered |
| C11 | PASS | 120 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=264 >= required=264 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 2152 unmapped rows, 32 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `openai/gpt-oss-20b` config.json @ `6cee5e81ee83917806bbde320786a8fb61efebee` (sha256 `183c7aa726f3…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=264 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 자기모순 추적 + 소스 대조)

의뢰서의 `2*d_moe` 는 이름이 옳았다 — 산술 휴리스틱이 내던 것을 규칙으로 승격했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
