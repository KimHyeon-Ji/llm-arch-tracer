# Model Summary -- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

## 기본 정보

- revision: `d51eab0d1f979ebc26b546e634a04f450d99158e`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 24
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 120.67B total, 12.77B active (10.6% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-03-10  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 40× linear_attention, 40× moe, 8× full_attention  (attention: GQA)  (FFN: 88× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 8.0 KiB (Very low) over 8 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=512, top-22, +1 shared, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, sigmoid-gating, MTP, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `nemotron_h` |
| attention | GQA — 32 query : 2 kv heads (repeat 16), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | MoE — 512 routed experts, top-22 + 1 shared, expert intermediate 2688, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·2·128 = 512 elems / token / layer; all 88 layers ⇒ 45056 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 88 |
| d_model | 4096 |
| n_h | 32 |
| n_kv | 2 |
| d_head | 128 |
| d_ff | 2688 |
| d_shared | 5376 |
| V | 131072 |
| ctx | 262144 |
| E | 512 |
| E_shared | 1 |
| k | 22 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 2688 |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 40× linear_attention, 40× moe, 8× full_attention (총 88층) |
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
| n_h_ssm | 128 |
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

shape 축 **193,087개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 69,228 | 35.85% |
| 런타임 축 (B/T/1) | 64,825 | 33.57% |
| 스코프 없는 심볼 | 31,377 | 16.25% |
| 이 모듈 스코프의 유도식 | 20,231 | 10.48% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,978 | 2.06% |
| 이름 없음 (정수 유지) | 3,240 | 1.68% |
| 스코프가 배제한 심볼 | 160 | 0.08% |
| 휴리스틱: 심볼의 배수 | 48 | 0.02% |

등록된 규칙 **185,661축**, 약한 근거 4,138축, 휴리스틱 **48축 (0.02%)**, 이름 없음 3,240축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 16 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | mixer |
| 27 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mixer |
| 256 | 2·d_head (CSA/HCA 압축기 kv_proj·gate_proj 폭: Ca⊕Cb) | k_proj, mixer, v_proj |
| 528 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 1024 | d_inner/n_g_ssm (Mamba gated RMSNorm의 그룹당 폭) | experts, fc1_latent_proj, fc2_latent_proj, mixer, norm |
| 8192 | d_inner (Mamba 내부 폭 = n_h_ssm · d_head_ssm) | mixer, norm, out_proj |
| 10240 | d_inner + 2·n_g·d_state (conv1d 입력 폭) | act, conv1d, mixer |
| 18560 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mixer |

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

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 88 == 88 |
| C2 | PASS | 3 clusters == 3 from config schedule ['layers_block_type'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 88/88 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 0 flagged) |
| C7 | PASS | GQA 32:2 (repeat factor 16) |
| C8 | WARN | MoE trace-verified [router_dim(E=512):ok, top_k(22):ok, expert_weight:grouped]; routed-token coun... |
| C9 | PASS | vocab_size=131072, tie_word_embeddings=False |
| C10 | PASS | all 723 params covered |
| C11 | PASS | 56 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=24 >= required=24 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 7019 unmapped rows, 41 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` config.json @ `d51eab0d1f979ebc26b546e634a04f450d99158e` (sha256 `ac7dddf84baf…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=24 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 1건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 3 |
| 이름 없음이 정답 | 2 |
| 교정 필요 | 5 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `mixer` | `n_kv` | `2` | 1800 | modeling_nemotron_h.py:320 `decay_chunk = torch.exp(segment_sum(F.pad( A_cumsum[:, :, :, -1], (1, 0))))` — 실측 `[1, 128, 2, 2]`. n_chunks+1 이고 여기서는 2 다. GQA 의 KV head 수와 무관하다. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mixer` | n_h_ssm vs d_state 축 순서 (둘 다 128) | `n_h_ssm / d_state 혼용` | `(소스가 가리키는 쪽 — 근거 참조)` | 남은 128건은 Mamba 내부의 진짜 값 충돌이다: n_h_ssm(128) == d_state(128) 이라 `view [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,T,?,?]` 의 두 출력 축을 우선순위로만 가르면 순서가 뒤집힌다. 합쳐진 축이 무엇인지는 reshape 자체가 알고 있지만(파생 계산), 그걸 채택하려면  … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
