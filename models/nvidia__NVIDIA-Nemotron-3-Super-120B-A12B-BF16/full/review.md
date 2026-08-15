# 리뷰 패킷 — nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `d51eab0d1f979ebc26b546e634a04f450d99158e` / 트레이스 seq_len(T) = 24
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
  L            = 88
  d_model      = 4096
  n_h          = 32
  n_kv         = 2
  d_head       = 128
  d_ff         = 2688
  d_shared     = 5376
  V            = 131072
  ctx          = 262144
  E            = 512
  E_shared     = 1
  k            = 22
  n_grp        = 1
  k_grp        = 1
  d_moe        = 2688
  w_local      = None
  n_sink       = None
  layer_sched  = ['linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'full_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe', 'linear_attention', 'moe']
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
  d_state      = 128
  n_g_ssm      = 8
  n_h_ssm      = 128
  d_chunk      = 128
  d_head_ssm   = 64
  d_conv       = 4
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


## 4. 검증 체크리스트 결과

```
# Extraction Report -- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ d51eab0d1f979ebc26b546e634a04f450d99158e

C1   PASS   88 == 88
C2   PASS   3 clusters == 3 from config schedule ['layers_block_type']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 88/88 layers
C6   PASS   hidden_size=4096 (heuristic check, 0 flagged)
C7   PASS   GQA 32:2 (repeat factor 16)
C8   WARN   MoE trace-verified [router_dim(E=512):ok, top_k(22):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=131072, tie_word_embeddings=False
C10  PASS   all 723 params covered
C11  PASS   56 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=24 >= required=24
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   7019 unmapped rows, 41 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embeddings                                   embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
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
  model.layers.N.norm                                _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.norm                                pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.norm                                mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.norm                                elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.norm                                rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.norm                                elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.norm                                elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mixer.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer.in_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.in_proj                       matmul           [T,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer.in_proj                       _unsafe_view     [T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer                               split_with_sizes [B,T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,0]*[B,T,0]*[B,T,d_inner]*[B,T,d_inner+2*n_g*d_state]*[B,T,d_state]
  model.layers.N.mixer                               transpose        [B,T,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mixer                               constant_pad_nd  [B,d_inner+2*n_g*d_state,T] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               zeros            [] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               slice            [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer.conv1d                        conv1d           [B,d_inner+2*n_g*d_state,T]*[d_inner+2*n_g*d_state,1,d_conv]*[d_inner+2*n_g*d_state] -> w=[d_inner+2*n_g*d_state,1,d_conv] [B,d_inner+2*n_g*d_state,T+d_conv-1]
  model.layers.N.mixer                               slice            [B,d_inner+2*n_g*d_state,T+d_conv-1] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mixer                               transpose        [B,d_inner+2*n_g*d_state,T] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mixer.act                           silu             [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mixer                               split_with_sizes [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner]*[B,T,n_g*d_state]*[B,T,n_g*d_state]
  model.layers.N.mixer                               _to_copy         [d_state] -> [d_state]
  model.layers.N.mixer                               exp              [d_state] -> [d_state]
  model.layers.N.mixer                               neg              [d_state] -> [d_state]
  model.layers.N.mixer                               elementwise_add  [B,T,d_state]*[d_state] -> [B,T,d_state]
  model.layers.N.mixer                               softplus         [B,T,d_state] -> [B,T,d_state]
  model.layers.N.mixer                               clamp            [B,T,d_state] -> [B,T,d_state]
  model.layers.N.mixer                               view             [B,T,d_inner] -> [B,T,d_state,d_head_ssm]
  model.layers.N.mixer                               _to_copy         [B,T,d_state,d_head_ssm] -> [B,T,d_state,d_head_ssm]
  model.layers.N.mixer                               view             [B,T,n_g*d_state] -> [B,T,n_g_ssm,d_state]
  model.layers.N.mixer                               _to_copy         [B,T,n_g_ssm,d_state] -> [B,T,n_g_ssm,d_state]
  model.layers.N.mixer                               unsqueeze        [B,T,n_g_ssm,d_state] -> [B,T,n_g_ssm,1,d_state]
  model.layers.N.mixer                               expand           [B,T,n_g_ssm,1,d_state] -> [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state]
  model.layers.N.mixer                               clone            [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state]
  model.layers.N.mixer                               view             [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,T,d_state,d_state]
  model.layers.N.mixer                               unsqueeze        [d_state] -> [d_state,B]
  model.layers.N.mixer                               constant_pad_nd  [B,T,d_state,d_head_ssm] -> [B,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               elementwise_mul  [d_state,B]*[B,d_state,n_h_ssm,d_head_ssm] -> [B,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               unsqueeze        [B,T,d_state] -> [B,T,d_state,1]
  model.layers.N.mixer                               elementwise_mul  [B,T,d_state,d_head_ssm]*[B,T,d_state,1] -> [B,T,d_state,d_head_ssm]
  model.layers.N.mixer                               elementwise_mul  [d_state]*[B,T,d_state] -> [B,T,d_state]
  model.layers.N.mixer                               view             [B,d_state,n_h_ssm,d_head_ssm] -> [B,1,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               constant_pad_nd  [B,T,d_state] -> [B,d_state,n_h_ssm]
  model.layers.N.mixer                               view             [B,d_state,n_h_ssm] -> [B,1,d_state,n_h_ssm]
  model.layers.N.mixer                               constant_pad_nd  [B,T,d_state,d_state] -> [B,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               view             [B,d_state,n_h_ssm,d_chunk] -> [B,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               permute          [B,1,d_state,n_h_ssm] -> [B,d_state,1,n_h_ssm]
  model.layers.N.mixer                               cumsum           [B,d_state,1,n_h_ssm] -> [B,d_state,1,n_h_ssm]
  model.layers.N.mixer                               unsqueeze        [B,d_state,1,n_h_ssm] -> [B,d_state,1,n_h_ssm,1]
  model.layers.N.mixer                               expand           [B,d_state,1,n_h_ssm,1] -> [B,d_state,1,n_h_ssm,d_chunk]
  model.layers.N.mixer                               ones             [] -> [d_state,n_h_ssm]
  model.layers.N.mixer                               tril             [d_state,n_h_ssm] -> [d_state,n_h_ssm]
  model.layers.N.mixer                               bitwise_not      [d_state,n_h_ssm] -> [d_state,n_h_ssm]
  model.layers.N.mixer                               masked_fill      [B,d_state,1,n_h_ssm,d_chunk]*[d_state,n_h_ssm] -> [B,d_state,1,n_h_ssm,d_chunk]
  model.layers.N.mixer                               cumsum           [B,d_state,1,n_h_ssm,d_chunk] -> [B,d_state,1,n_h_ssm,d_chunk]
  model.layers.N.mixer                               exp              [B,d_state,1,n_h_ssm,d_chunk] -> [B,d_state,1,n_h_ssm,d_chunk]
  model.layers.N.mixer                               unsqueeze        [B,1,d_state,n_h_ssm,d_chunk] -> [B,1,d_state,1,n_h_ssm,d_chunk]
  model.layers.N.mixer                               unsqueeze        [B,1,d_state,n_h_ssm,d_chunk] -> [B,1,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               elementwise_mul  [B,1,d_state,1,n_h_ssm,d_chunk]*[B,1,1,d_state,n_h_ssm,d_chunk] -> [B,1,d_state,n_h_ssm,d_chunk,d_head]
  model.layers.N.mixer                               sum              [B,1,d_state,n_h_ssm,d_chunk,d_head] -> [B,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               permute          [B,d_state,1,n_h_ssm,d_chunk] -> [B,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               elementwise_mul  [B,1,d_state,n_h_ssm,d_chunk,1]*[B,1,d_state,n_h_ssm,d_chunk,1] -> [B,1,d_state,n_h_ssm,d_chunk,1]
  model.layers.N.mixer                               sum              [B,1,d_state,n_h_ssm,d_chunk,1] -> [B,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               elementwise_mul  [B,1,d_state,n_h_ssm,d_chunk,1]*[B,1,1,d_state,n_h_ssm,d_head_ssm] -> [B,1,d_state,n_h_ssm,d_chunk,d_head_ssm]
  model.layers.N.mixer                               sum              [B,1,d_state,n_h_ssm,d_chunk,d_head_ssm] -> [B,1,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               slice            [B,d_state,1,n_h_ssm] -> [B,d_state,1,1]
  model.layers.N.mixer                               sub              [B,d_state,1,1]*[B,d_state,1,n_h_ssm] -> [B,d_state,1,n_h_ssm]
  model.layers.N.mixer                               exp              [B,d_state,1,n_h_ssm] -> [B,d_state,1,n_h_ssm]
  model.layers.N.mixer                               permute          [B,d_state,1,n_h_ssm] -> [B,1,d_state,n_h_ssm]
  model.layers.N.mixer                               permute          [B,1,d_state,n_h_ssm,d_chunk] -> [B,1,d_state,n_h_ssm,d_chunk]
  model.layers.N.mixer                               permute          [B,1,d_state,n_h_ssm,d_head_ssm] -> [B,1,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               permute          [B,1,d_state,n_h_ssm,d_head_ssm] -> [B,1,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               alias            [B,1,d_state,d_head_ssm,n_h_ssm] -> [B,1,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               zeros_like       [B,1,d_state,d_head_ssm,n_h_ssm] -> [B,1,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               concat           [B,1,d_state,d_head_ssm,n_h_ssm]*[B,1,d_state,d_head_ssm,n_h_ssm] -> [B,2,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               select           [B,d_state,1,n_h_ssm] -> [B,d_state,1]
  model.layers.N.mixer                               constant_pad_nd  [B,d_state,1] -> [B,d_state,2]
  model.layers.N.mixer                               expand           [B,d_state,2,1] -> [B,d_state,2,2]
  model.layers.N.mixer                               ones             [] -> [2,2]
  model.layers.N.mixer                               tril             [2,2] -> [2,2]
  model.layers.N.mixer                               bitwise_not      [2,2] -> [2,2]
  model.layers.N.mixer                               masked_fill      [B,d_state,2,2]*[2,2] -> [B,d_state,2,2]
  model.layers.N.mixer                               cumsum           [B,d_state,2,2] -> [B,d_state,2,2]
  model.layers.N.mixer                               exp              [B,d_state,2,2] -> [B,d_state,2,2]
  model.layers.N.mixer                               sum              [B,d_state,2,2,d_head_ssm,n_h_ssm] -> [B,d_state,2,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               slice            [B,2,d_state,d_head_ssm,n_h_ssm] -> [B,1,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               select           [B,2,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               sum              [B,1,d_state,n_h_ssm,d_head_ssm,d_chunk] -> [B,1,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               elementwise_add  [B,1,d_state,n_h_ssm,d_head_ssm]*[B,1,d_state,n_h_ssm,d_head_ssm] -> [B,1,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               elementwise_add  [B,d_state,n_h_ssm,d_head_ssm]*[B,d_state,n_h_ssm,d_head_ssm] -> [B,d_state,n_h_ssm,d_head_ssm]
  model.layers.N.mixer                               slice            [B,d_state,n_h_ssm,d_head_ssm] -> [B,T,d_state,d_head_ssm]
  model.layers.N.mixer                               zeros_like       [B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               copy_            [B,d_state,d_head_ssm,n_h_ssm]*[B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer.norm                          _to_copy         [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mixer.norm                          silu             [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mixer.norm                          elementwise_mul  [B,T,d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mixer.norm                          view             [B,T,d_inner] -> [B,T,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          pow              [B,T,n_g_ssm,n_g*d_state] -> [B,T,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          mean             [B,T,n_g_ssm,n_g*d_state] -> [B,T,n_g_ssm,1]
  model.layers.N.mixer.norm                          elementwise_add  [B,T,n_g_ssm,1] -> [B,T,n_g_ssm,1]
  model.layers.N.mixer.norm                          rsqrt            [B,T,n_g_ssm,1] -> [B,T,n_g_ssm,1]
  model.layers.N.mixer.norm                          elementwise_mul  [B,T,n_g_ssm,n_g*d_state]*[B,T,n_g_ssm,1] -> [B,T,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          view             [B,T,n_g_ssm,n_g*d_state] -> [B,T,d_inner]
  model.layers.N.mixer.norm                          elementwise_mul  [d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mixer                               _to_copy         [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mixer.out_proj                      t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mixer.out_proj                      view             [B,T,d_inner] -> [T,d_inner]
  model.layers.N.mixer.out_proj                      matmul           [T,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [T,d_model]
  model.layers.N.mixer.out_proj                      _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mixer.gate                          view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.gate                          _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mixer.gate                          _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mixer.gate                          t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mixer.gate                          matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mixer.gate                          sigmoid          [T,E] -> [T,E]
  model.layers.N.mixer.gate                          elementwise_add  [T,E]*[E] -> [T,E]
  model.layers.N.mixer.gate                          view             [T,E] -> [T,1,E]
  model.layers.N.mixer.gate                          topk             [T,1,E] -> [T,1,2]*[T,1,2]
  model.layers.N.mixer.gate                          sum              [T,1,2] -> [T,1]
  model.layers.N.mixer.gate                          topk             [T,1] -> [T,1]*[T,1]
  model.layers.N.mixer.gate                          zeros_like       [T,1] -> [T,1]
  model.layers.N.mixer.gate                          scatter_         [T,1]*[T,1] -> [T,1]
  model.layers.N.mixer.gate                          unsqueeze        [T,1] -> [T,1,1]
  model.layers.N.mixer.gate                          expand           [T,1,1] -> [T,1,E]
  model.layers.N.mixer.gate                          view             [T,1,E] -> [T,E]
  model.layers.N.mixer.gate                          _to_copy         [T,E] -> [T,E]
  model.layers.N.mixer.gate                          bitwise_not      [T,E] -> [T,E]
  model.layers.N.mixer.gate                          masked_fill      [T,E]*[T,E] -> [T,E]
  model.layers.N.mixer.gate                          topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mixer.gate                          gather           [T,E]*[T,k] -> [T,k]
  model.layers.N.mixer.gate                          sum              [T,k] -> [T,1]
  model.layers.N.mixer.gate                          elementwise_add  [T,1] -> [T,1]
  model.layers.N.mixer.gate                          div_             [T,k]*[T,1] -> [T,k]
  model.layers.N.mixer.gate                          elementwise_mul  [T,k] -> [T,k]
  model.layers.N.mixer.fc1_latent_proj               t                [n_g*d_state,d_model] -> w=[n_g*d_state,d_model] [d_model,n_g*d_state]
  model.layers.N.mixer.fc1_latent_proj               matmul           [T,d_model]*[d_model,n_g*d_state] -> w=[n_g*d_state,d_model] [T,n_g*d_state]
  model.layers.N.mixer.experts                       view             [T,k] -> [k*T]
  model.layers.N.mixer.experts                       sort             [k*T] -> [k*T]*[k*T]
  model.layers.N.mixer.experts                       floor_divide     [k*T] -> [k*T]
  model.layers.N.mixer.experts                       index            [T,n_g*d_state]*[k*T] -> [k*T,n_g*d_state]
  model.layers.N.mixer.experts                       index            [k*T]*[k*T] -> [k*T]
  model.layers.N.mixer.experts                       _to_copy         [k*T] -> [k*T]
  model.layers.N.mixer.experts                       histc            [k*T] -> [E]
  model.layers.N.mixer.experts                       cumsum           [E] -> [E]
  model.layers.N.mixer.experts                       ge               [k*T] -> [k*T]
  model.layers.N.mixer.experts                       unsqueeze        [k*T] -> [k*T,B]
  model.layers.N.mixer.experts                       clamp_           [k*T] -> [k*T]
  model.layers.N.mixer.experts                       masked_fill_     [k*T,n_g*d_state]*[k*T,B] -> [k*T,n_g*d_state]
  model.layers.N.mixer.experts                       transpose        [E,d_moe,n_g*d_state] -> w=[E,d_moe,n_g*d_state] [E,n_g*d_state,d_moe]
  model.layers.N.mixer.experts                       grouped_matmul   [k*T,n_g*d_state]*[E,n_g*d_state,d_moe]*[E] -> w=[E,d_moe,n_g*d_state] [k*T,d_moe]
  model.layers.N.mixer.experts.act_fn                relu             [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mixer.experts.act_fn                pow              [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mixer.experts                       transpose        [E,n_g*d_state,d_moe] -> w=[E,n_g*d_state,d_moe] [E,d_moe,n_g*d_state]
  model.layers.N.mixer.experts                       grouped_matmul   [k*T,d_moe]*[E,d_moe,n_g*d_state]*[E] -> w=[E,n_g*d_state,d_moe] [k*T,n_g*d_state]
  model.layers.N.mixer.experts                       elementwise_mul  [k*T,n_g*d_state]*[k*T,B] -> [k*T,n_g*d_state]
  model.layers.N.mixer.experts                       empty_like       [k*T] -> [k*T]
  model.layers.N.mixer.experts                       arange           [] -> [k*T]
  model.layers.N.mixer.experts                       index_put_       [k*T]*[k*T]*[k*T] -> [k*T]
  model.layers.N.mixer.experts                       index            [k*T,n_g*d_state]*[k*T] -> [k*T,n_g*d_state]
  model.layers.N.mixer.experts                       view             [k*T,n_g*d_state] -> [T,k,n_g*d_state]
  model.layers.N.mixer.experts                       sum              [T,k,n_g*d_state] -> [T,n_g*d_state]
  model.layers.N.mixer.experts                       _to_copy         [T,n_g*d_state] -> [T,n_g*d_state]
  model.layers.N.mixer.fc2_latent_proj               t                [d_model,n_g*d_state] -> w=[d_model,n_g*d_state] [n_g*d_state,d_model]
  model.layers.N.mixer.fc2_latent_proj               matmul           [T,n_g*d_state]*[n_g*d_state,d_model] -> w=[d_model,n_g*d_state] [T,d_model]
  model.layers.N.mixer.shared_experts.up_proj        t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mixer.shared_experts.up_proj        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.shared_experts.up_proj        matmul           [T,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [T,d_shared]
  model.layers.N.mixer.shared_experts.up_proj        _unsafe_view     [T,d_shared] -> [B,T,d_shared]
  model.layers.N.mixer.shared_experts.act_fn         relu             [B,T,d_shared] -> [B,T,d_shared]
  model.layers.N.mixer.shared_experts.act_fn         pow              [B,T,d_shared] -> [B,T,d_shared]
  model.layers.N.mixer.shared_experts.down_proj      t                [d_model,d_shared] -> w=[d_model,d_shared] [d_shared,d_model]
  model.layers.N.mixer.shared_experts.down_proj      view             [B,T,d_shared] -> [T,d_shared]
  model.layers.N.mixer.shared_experts.down_proj      matmul           [T,d_shared]*[d_shared,d_model] -> w=[d_model,d_shared] [T,d_model]
  model.layers.N.mixer.shared_experts.down_proj      _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.mixer                               elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mixer.q_proj                        t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.mixer.q_proj                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.q_proj                        matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  model.layers.N.mixer.q_proj                        _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.mixer                               transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.mixer.k_proj                        t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.mixer.k_proj                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.k_proj                        matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.mixer.k_proj                        _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.mixer                               transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.mixer.v_proj                        t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.mixer.v_proj                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.v_proj                        matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.mixer.v_proj                        _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.mixer                               concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.mixer                               expand           [B,2,1,T,d_head] -> [B,2,n_h_ssm/n_g_ssm,T,d_head]
  model.layers.N.mixer                               clone            [B,2,n_h_ssm/n_g_ssm,T,d_head] -> [B,2,n_h_ssm/n_g_ssm,T,d_head]
  model.layers.N.mixer                               _unsafe_view     [B,2,n_h_ssm/n_g_ssm,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.mixer                               transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.mixer                               expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.mixer                               expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.mixer                               batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.mixer                               _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.mixer                               elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.mixer                               _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.mixer                               softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.mixer                               batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.mixer                               _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.mixer                               transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.mixer                               clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.mixer.o_proj                        t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.mixer.o_proj                        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mixer.o_proj                        matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  model.layers.N.mixer.o_proj                        _unsafe_view     [T,d_model] -> [B,T,d_model]
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
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.38                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.40                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.42                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.44                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.45                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.46                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.47                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.48                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.49                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.50                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.51                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.52                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.53                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.54                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.55                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.56                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.57                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.58                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.59                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.60                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.61                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.62                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.63                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.64                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.65                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.66                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.67                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.68                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.69                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.70                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.71                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.72                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.73                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.74                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.75                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.76                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.77                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.78                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.79                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.80                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.81                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.82                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.83                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.84                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.85                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.86                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.87                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.norm_f                                       _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.norm_f                                       pow              [B,T,d_model] -> [B,T,d_model]
  model.norm_f                                       mean             [B,T,d_model] -> [B,T,1]
  model.norm_f                                       elementwise_add  [B,T,1] -> [B,T,1]
  model.norm_f                                       rsqrt            [B,T,1] -> [B,T,1]
  model.norm_f                                       elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.norm_f                                       elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
                                                     alias            [B,T,d_model] -> [B,T,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,T,d_model] -> [T,d_model]
  lm_head                                            matmul           [T,d_model]*[d_model,V] -> w=[V,d_model] [T,V]
  lm_head                                            _unsafe_view     [T,V] -> [B,T,V]
                                                     _to_copy         [B,T,V] -> [B,T,V]
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
  model.embeddings                                   embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
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
  model.layers.N.norm                                _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.norm                                pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.norm                                mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.norm                                elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.norm                                rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.norm                                elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.norm                                elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mixer.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer.in_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.in_proj                       matmul           [B,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [B,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer.in_proj                       _unsafe_view     [B,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mixer                               split_with_sizes [B,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,0]*[B,1,0]*[B,1,d_inner]*[B,1,d_inner+2*n_g*d_state]*[B,1,d_state]
  model.layers.N.mixer                               transpose        [B,1,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,1]
  model.layers.N.mixer                               concat           [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,1] -> [B,d_inner+2*n_g*d_state,d_conv+1]
  model.layers.N.mixer                               slice            [B,d_inner+2*n_g*d_state,d_conv+1] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               select           [d_inner+2*n_g*d_state,1,d_conv] -> w=[d_inner+2*n_g*d_state,1,d_conv] [d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               elementwise_mul  [B,d_inner+2*n_g*d_state,d_conv]*[d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mixer                               sum              [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mixer                               add_             [B,d_inner+2*n_g*d_state]*[d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mixer.act                           silu             [B,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mixer                               unsqueeze        [B,d_inner+2*n_g*d_state] -> [B,1,d_inner+2*n_g*d_state]
  model.layers.N.mixer                               split_with_sizes [B,1,d_inner+2*n_g*d_state] -> [B,1,d_inner]*[B,1,n_g*d_state]*[B,1,n_g*d_state]
  model.layers.N.mixer                               _to_copy         [d_state] -> [d_state]
  model.layers.N.mixer                               exp              [d_state] -> [d_state]
  model.layers.N.mixer                               neg              [d_state] -> [d_state]
  model.layers.N.mixer                               select           [B,1,d_state] -> [B,d_state]
  model.layers.N.mixer                               unsqueeze        [B,d_state] -> [B,1,d_state]
  model.layers.N.mixer                               transpose        [B,1,d_state] -> [B,d_state,1]
  model.layers.N.mixer                               expand           [B,d_state,1] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               unsqueeze        [d_state] -> [d_state,B]
  model.layers.N.mixer                               expand           [d_state,B] -> [d_state,d_head_ssm]
  model.layers.N.mixer                               elementwise_add  [B,d_state,d_head_ssm]*[d_state,d_head_ssm] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               softplus         [B,d_state,d_head_ssm] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               clamp            [B,d_state,d_head_ssm] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               unsqueeze        [d_state,B] -> [d_state,B,1]
  model.layers.N.mixer                               expand           [d_state,B,1] -> [d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               unsqueeze        [B,d_state,d_head_ssm] -> [B,d_state,d_head_ssm,1]
  model.layers.N.mixer                               elementwise_mul  [B,d_state,d_head_ssm,1]*[d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               exp              [B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               view             [B,1,n_g*d_state] -> [B,n_g_ssm,d_state]
  model.layers.N.mixer                               unsqueeze        [B,n_g_ssm,d_state] -> [B,n_g_ssm,1,d_state]
  model.layers.N.mixer                               expand           [B,n_g_ssm,1,d_state] -> [B,n_g_ssm,n_h_ssm/n_g_ssm,d_state]
  model.layers.N.mixer                               clone            [B,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,n_g_ssm,n_h_ssm/n_g_ssm,d_state]
  model.layers.N.mixer                               view             [B,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,d_state,d_state]
  model.layers.N.mixer                               elementwise_mul  [B,d_state,d_head_ssm,1]*[B,d_state,1,d_state] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               view             [B,1,d_inner] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               elementwise_mul  [B,d_state,d_head_ssm,n_h_ssm]*[B,d_state,d_head_ssm,1] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               clone            [B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               elementwise_mul  [B,d_state,d_head_ssm,n_h_ssm]*[B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               elementwise_add  [B,d_state,d_head_ssm,n_h_ssm]*[B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               copy_            [B,d_state,d_head_ssm,n_h_ssm]*[B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               _to_copy         [B,d_state,d_head_ssm,n_h_ssm] -> [B,d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               view             [B,d_state,d_head_ssm,n_h_ssm] -> [d_state,d_head_ssm,n_h_ssm]
  model.layers.N.mixer                               view             [B,d_state,d_state] -> [d_state,n_h_ssm,B]
  model.layers.N.mixer                               batched_matmul   [d_state,d_head_ssm,n_h_ssm]*[d_state,n_h_ssm,B] -> [d_state,d_head_ssm,B]
  model.layers.N.mixer                               view             [d_state,d_head_ssm,B] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               elementwise_mul  [B,d_state,d_head_ssm]*[d_state,d_head_ssm] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer                               elementwise_add  [B,d_state,d_head_ssm]*[B,d_state,d_head_ssm] -> [B,d_state,d_head_ssm]
  model.layers.N.mixer.norm                          _to_copy         [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mixer.norm                          silu             [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mixer.norm                          elementwise_mul  [B,1,d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mixer.norm                          view             [B,1,d_inner] -> [B,1,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          pow              [B,1,n_g_ssm,n_g*d_state] -> [B,1,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          mean             [B,1,n_g_ssm,n_g*d_state] -> [B,1,n_g_ssm,1]
  model.layers.N.mixer.norm                          elementwise_add  [B,1,n_g_ssm,1] -> [B,1,n_g_ssm,1]
  model.layers.N.mixer.norm                          rsqrt            [B,1,n_g_ssm,1] -> [B,1,n_g_ssm,1]
  model.layers.N.mixer.norm                          elementwise_mul  [B,1,n_g_ssm,n_g*d_state]*[B,1,n_g_ssm,1] -> [B,1,n_g_ssm,n_g*d_state]
  model.layers.N.mixer.norm                          view             [B,1,n_g_ssm,n_g*d_state] -> [B,1,d_inner]
  model.layers.N.mixer.norm                          elementwise_mul  [d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mixer.out_proj                      t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mixer.out_proj                      view             [B,1,d_inner] -> [B,d_inner]
  model.layers.N.mixer.out_proj                      matmul           [B,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [B,d_model]
  model.layers.N.mixer.out_proj                      _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mixer.gate                          view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.gate                          _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mixer.gate                          _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mixer.gate                          t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mixer.gate                          matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mixer.gate                          sigmoid          [B,E] -> [B,E]
  model.layers.N.mixer.gate                          elementwise_add  [B,E]*[E] -> [B,E]
  model.layers.N.mixer.gate                          view             [B,E] -> [B,1,E]
  model.layers.N.mixer.gate                          topk             [B,1,E] -> [B,1,2]*[B,1,2]
  model.layers.N.mixer.gate                          sum              [B,1,2] -> [B,1]
  model.layers.N.mixer.gate                          topk             [B,1] -> [B,1]*[B,1]
  model.layers.N.mixer.gate                          zeros_like       [B,1] -> [B,1]
  model.layers.N.mixer.gate                          scatter_         [B,1]*[B,1] -> [B,1]
  model.layers.N.mixer.gate                          unsqueeze        [B,1] -> [B,1,1]
  model.layers.N.mixer.gate                          expand           [B,1,1] -> [B,1,E]
  model.layers.N.mixer.gate                          view             [B,1,E] -> [B,E]
  model.layers.N.mixer.gate                          _to_copy         [B,E] -> [B,E]
  model.layers.N.mixer.gate                          bitwise_not      [B,E] -> [B,E]
  model.layers.N.mixer.gate                          masked_fill      [B,E]*[B,E] -> [B,E]
  model.layers.N.mixer.gate                          topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mixer.gate                          gather           [B,E]*[B,k] -> [B,k]
  model.layers.N.mixer.gate                          sum              [B,k] -> [B,1]
  model.layers.N.mixer.gate                          elementwise_add  [B,1] -> [B,1]
  model.layers.N.mixer.gate                          div_             [B,k]*[B,1] -> [B,k]
  model.layers.N.mixer.gate                          elementwise_mul  [B,k] -> [B,k]
  model.layers.N.mixer.fc1_latent_proj               t                [n_g*d_state,d_model] -> w=[n_g*d_state,d_model] [d_model,n_g*d_state]
  model.layers.N.mixer.fc1_latent_proj               matmul           [B,d_model]*[d_model,n_g*d_state] -> w=[n_g*d_state,d_model] [B,n_g*d_state]
  model.layers.N.mixer.experts                       view             [B,k] -> [k]
  model.layers.N.mixer.experts                       sort             [k] -> [k]*[k]
  model.layers.N.mixer.experts                       floor_divide     [k] -> [k]
  model.layers.N.mixer.experts                       index            [B,n_g*d_state]*[k] -> [k,n_g*d_state]
  model.layers.N.mixer.experts                       index            [k]*[k] -> [k]
  model.layers.N.mixer.experts                       _to_copy         [k] -> [k]
  model.layers.N.mixer.experts                       histc            [k] -> [E]
  model.layers.N.mixer.experts                       cumsum           [E] -> [E]
  model.layers.N.mixer.experts                       ge               [k] -> [k]
  model.layers.N.mixer.experts                       unsqueeze        [k] -> [k,B]
  model.layers.N.mixer.experts                       clamp_           [k] -> [k]
  model.layers.N.mixer.experts                       masked_fill_     [k,n_g*d_state]*[k,B] -> [k,n_g*d_state]
  model.layers.N.mixer.experts                       transpose        [E,d_moe,n_g*d_state] -> w=[E,d_moe,n_g*d_state] [E,n_g*d_state,d_moe]
  model.layers.N.mixer.experts                       grouped_matmul   [k,n_g*d_state]*[E,n_g*d_state,d_moe]*[E] -> w=[E,d_moe,n_g*d_state] [k,d_moe]
  model.layers.N.mixer.experts.act_fn                relu             [k,d_moe] -> [k,d_moe]
  model.layers.N.mixer.experts.act_fn                pow              [k,d_moe] -> [k,d_moe]
  model.layers.N.mixer.experts                       transpose        [E,n_g*d_state,d_moe] -> w=[E,n_g*d_state,d_moe] [E,d_moe,n_g*d_state]
  model.layers.N.mixer.experts                       grouped_matmul   [k,d_moe]*[E,d_moe,n_g*d_state]*[E] -> w=[E,n_g*d_state,d_moe] [k,n_g*d_state]
  model.layers.N.mixer.experts                       elementwise_mul  [k,n_g*d_state]*[k,B] -> [k,n_g*d_state]
  model.layers.N.mixer.experts                       empty_like       [k] -> [k]
  model.layers.N.mixer.experts                       arange           [] -> [k]
  model.layers.N.mixer.experts                       index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.mixer.experts                       index            [k,n_g*d_state]*[k] -> [k,n_g*d_state]
  model.layers.N.mixer.experts                       view             [k,n_g*d_state] -> [B,k,n_g*d_state]
  model.layers.N.mixer.experts                       sum              [B,k,n_g*d_state] -> [B,n_g*d_state]
  model.layers.N.mixer.experts                       _to_copy         [B,n_g*d_state] -> [B,n_g*d_state]
  model.layers.N.mixer.fc2_latent_proj               t                [d_model,n_g*d_state] -> w=[d_model,n_g*d_state] [n_g*d_state,d_model]
  model.layers.N.mixer.fc2_latent_proj               matmul           [B,n_g*d_state]*[n_g*d_state,d_model] -> w=[d_model,n_g*d_state] [B,d_model]
  model.layers.N.mixer.shared_experts.up_proj        t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mixer.shared_experts.up_proj        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.shared_experts.up_proj        matmul           [B,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [B,d_shared]
  model.layers.N.mixer.shared_experts.up_proj        _unsafe_view     [B,d_shared] -> [B,1,d_shared]
  model.layers.N.mixer.shared_experts.act_fn         relu             [B,1,d_shared] -> [B,1,d_shared]
  model.layers.N.mixer.shared_experts.act_fn         pow              [B,1,d_shared] -> [B,1,d_shared]
  model.layers.N.mixer.shared_experts.down_proj      t                [d_model,d_shared] -> w=[d_model,d_shared] [d_shared,d_model]
  model.layers.N.mixer.shared_experts.down_proj      view             [B,1,d_shared] -> [B,d_shared]
  model.layers.N.mixer.shared_experts.down_proj      matmul           [B,d_shared]*[d_shared,d_model] -> w=[d_model,d_shared] [B,d_model]
  model.layers.N.mixer.shared_experts.down_proj      _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.mixer                               elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mixer.q_proj                        t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.mixer.q_proj                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.q_proj                        matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  model.layers.N.mixer.q_proj                        _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.mixer                               transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.mixer.k_proj                        t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.mixer.k_proj                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.k_proj                        matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.mixer.k_proj                        _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.mixer                               transpose        [B,1,2,d_head] -> [B,2,1,d_head]
  model.layers.N.mixer.v_proj                        t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.mixer.v_proj                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.v_proj                        matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.mixer.v_proj                        _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.mixer                               concat           [B,2,T,d_head]*[B,2,1,d_head] -> [B,2,T+1,d_head]
  model.layers.N.mixer                               expand           [B,2,1,T+1,d_head] -> [B,2,n_h_ssm/n_g_ssm,T+1,d_head]
  model.layers.N.mixer                               clone            [B,2,n_h_ssm/n_g_ssm,T+1,d_head] -> [B,2,n_h_ssm/n_g_ssm,T+1,d_head]
  model.layers.N.mixer                               _unsafe_view     [B,2,n_h_ssm/n_g_ssm,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.mixer                               transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.mixer                               expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.mixer                               batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.mixer                               _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.mixer                               elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.mixer                               _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.mixer                               softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.mixer                               batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.mixer                               _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.mixer                               transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.mixer.o_proj                        t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.mixer.o_proj                        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mixer.o_proj                        matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  model.layers.N.mixer.o_proj                        _unsafe_view     [B,d_model] -> [B,1,d_model]
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
  model.layers.24                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.35                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.38                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.39                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.40                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.41                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.42                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.43                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.44                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.45                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.46                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.47                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.48                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.49                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.50                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.51                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.52                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.53                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.54                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.55                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.56                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.57                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.58                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.59                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.60                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.61                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.62                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.63                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.64                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.65                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.66                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.67                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.68                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.69                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.70                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.71                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.72                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.73                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.74                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.75                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.76                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.77                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.78                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.79                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.80                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.81                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.82                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.83                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.84                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.85                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.86                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.87                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.norm_f                                       _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.norm_f                                       pow              [B,1,d_model] -> [B,1,d_model]
  model.norm_f                                       mean             [B,1,d_model] -> [B,1,1]
  model.norm_f                                       elementwise_add  [B,1,1] -> [B,1,1]
  model.norm_f                                       rsqrt            [B,1,1] -> [B,1,1]
  model.norm_f                                       elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.norm_f                                       elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
                                                     alias            [B,1,d_model] -> [B,1,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,1,d_model] -> [B,d_model]
  lm_head                                            matmul           [B,d_model]*[d_model,V] -> w=[V,d_model] [B,V]
  lm_head                                            _unsafe_view     [B,V] -> [B,1,V]
                                                     _to_copy         [B,1,V] -> [B,1,V]
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

