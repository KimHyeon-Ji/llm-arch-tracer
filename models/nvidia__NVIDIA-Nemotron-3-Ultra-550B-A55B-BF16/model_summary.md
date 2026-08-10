# Model Summary -- nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16

## 기본 정보

- revision: `624ba927cfbef0427354998700de3d51173c8c04`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 24
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 549.31B total, 56.06B active (10.2% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-06-03  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 48× linear_attention, 48× moe, 12× GQA  (FFN: 108× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 12.0 KiB (Very low) over 12 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=512, top-22, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, sigmoid-gating, MTP, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `nemotron_h` |
| attention | GQA — 64 query : 2 kv heads (repeat 32), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | MoE — 512 routed experts, top-22 + 1 shared, expert intermediate 5120, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·2·128 = 512 elems / token / layer; all 108 layers ⇒ 55296 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 108 |
| d_model | 8192 |
| n_h | 64 |
| n_kv | 2 |
| d_head | 128 |
| d_ff | 5120 |
| V | 131072 |
| ctx | 262144 |
| E | 512 |
| E_shared | 1 |
| k | 22 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 5120 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 48× linear_attention, 48× moe, 12× full_attention (총 108층) |
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
| n_h_ssm | 256 |
| d_chunk | 128 |
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

shape 축 **234,827개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 81,852 | 34.86% |
| 런타임 축 (B/T/1) | 78,685 | 33.51% |
| 스코프 없는 심볼 | 38,317 | 16.32% |
| 이 모듈 스코프의 유도식 | 26,843 | 11.43% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 4,882 | 2.08% |
| 이름 없음 (정수 유지) | 4,056 | 1.73% |
| 스코프가 배제한 심볼 | 192 | 0.08% |

등록된 규칙 **225,697축**, 약한 근거 5,074축, 휴리스틱 **0축 (0.0%)**, 이름 없음 4,056축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 27 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mixer |
| 32 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | mixer |
| 528 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 1024 | n_g_ssm·d_state (B/C 하나의 폭) | mixer |
| 2048 | d_inner/n_g_ssm (Mamba gated RMSNorm의 그룹당 폭) | experts, fc1_latent_proj, fc2_latent_proj, norm |
| 10240 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | act_fn, down_proj, up_proj |
| 16384 | d_inner (Mamba 내부 폭 = n_h_ssm · d_head_ssm) | mixer, norm, out_proj |
| 18432 | d_inner + 2·n_g·d_state (conv1d 입력 폭) | act, conv1d, mixer |
| 35072 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mixer |

## 레이어 구조

- layer 0: mixer, norm
- layer 1: mixer, norm
- layer 2: mixer, norm
- layer 3: mixer, norm
- layer 4: mixer, norm
- layer 5: mixer, norm
- layer 6: mixer, norm
- layer 7: mixer, norm
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
- layer 30: mixer, norm
- layer 31: mixer, norm
- layer 32: mixer, norm
- layer 33: mixer, norm
- layer 34: mixer, norm
- layer 35: mixer, norm
- layer 36: mixer, norm
- layer 37: mixer, norm
- layer 38: mixer, norm
- layer 39: mixer, norm
- layer 40: mixer, norm
- layer 41: mixer, norm
- layer 42: mixer, norm
- layer 43: mixer, norm
- layer 44: mixer, norm
- layer 45: mixer, norm
- layer 46: mixer, norm
- layer 47: mixer, norm
- layer 48: mixer, norm
- layer 49: mixer, norm
- layer 50: mixer, norm
- layer 51: mixer, norm
- layer 52: mixer, norm
- layer 53: mixer, norm
- layer 54: mixer, norm
- layer 55: mixer, norm
- layer 56: mixer, norm
- layer 57: mixer, norm
- layer 58: mixer, norm
- layer 59: mixer, norm
- layer 60: mixer, norm
- layer 61: mixer, norm
- layer 62: mixer, norm
- layer 63: mixer, norm
- layer 64: mixer, norm
- layer 65: mixer, norm
- layer 66: mixer, norm
- layer 67: mixer, norm
- layer 68: mixer, norm
- layer 69: mixer, norm
- layer 70: mixer, norm
- layer 71: mixer, norm
- layer 72: mixer, norm
- layer 73: mixer, norm
- layer 74: mixer, norm
- layer 75: mixer, norm
- layer 76: mixer, norm
- layer 77: mixer, norm
- layer 78: mixer, norm
- layer 79: mixer, norm
- layer 80: mixer, norm
- layer 81: mixer, norm
- layer 82: mixer, norm
- layer 83: mixer, norm
- layer 84: mixer, norm
- layer 85: mixer, norm
- layer 86: mixer, norm
- layer 87: mixer, norm
- layer 88: mixer, norm
- layer 89: mixer, norm
- layer 90: mixer, norm
- layer 91: mixer, norm
- layer 92: mixer, norm
- layer 93: mixer, norm
- layer 94: mixer, norm
- layer 95: mixer, norm
- layer 96: mixer, norm
- layer 97: mixer, norm
- layer 98: mixer, norm
- layer 99: mixer, norm
- layer 100: mixer, norm
- layer 101: mixer, norm
- layer 102: mixer, norm
- layer 103: mixer, norm
- layer 104: mixer, norm
- layer 105: mixer, norm
- layer 106: mixer, norm
- layer 107: mixer, norm

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 108 == 108 |
| C2 | PASS | 3 clusters == 3 from config schedule ['layers_block_type'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=8192 in 108/108 layers |
| C6 | PASS | hidden_size=8192 (heuristic check, 0 flagged) |
| C7 | PASS | GQA 64:2 (repeat factor 32) |
| C8 | WARN | MoE trace-verified [router_dim(E=512):ok, top_k(22):ok, expert_weight:grouped]; routed-token coun... |
| C9 | PASS | vocab_size=131072, tie_word_embeddings=False |
| C10 | PASS | all 879 params covered |
| C11 | PASS | 72 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=24 >= required=24 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 8523 unmapped rows, 41 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` config.json @ `624ba927cfbef0427354998700de3d51173c8c04` (sha256 `00242512bc24…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=24 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-10 · llm(claude, 소스 직접 대조)

의뢰서 2건 → 1건. L=108, d=8192 의 최상위 모델이 **새 규칙 0개**로 들어왔다 — '규칙은 모델마다 늘지 않는다'가 대규모에서도 성립함을 보여준다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 이름 없음이 정답 | 1 |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
