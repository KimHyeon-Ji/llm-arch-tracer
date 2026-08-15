# 리뷰 패킷 — Qwen/Qwen3.5-4B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` / 트레이스 seq_len(T) = 17
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
  L            = 32
  d_model      = 2560
  n_h          = 16
  n_kv         = 4
  d_head       = 256
  d_ff         = 9216
  d_shared     = None
  V            = 248320
  ctx          = 262144
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
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
  d_chunk      = 64
  d_head_ssm   = None
  d_conv       = None
  n_mem        = None
  r_lora       = None
  d_attn       = None
  n_h_lin_k    = 16
  n_h_lin_v    = 32
  d_head_lin_k = 128
  d_head_lin_v = 128
  d_conv_lin   = 4
```

## 3. 모델 요약 산출물

# Model Summary -- Qwen/Qwen3.5-4B

## 기본 정보

- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 4.21B total (dense) |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2026-02-27  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 24× linear_attention, 8× full_attention  (attention: GQA) |
| 7 | KV CACHE / TOKEN (BF16) | 32.0 KiB (Low) over 8 attn layers |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA, QK-Norm, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `qwen3_5_text` |
| attention | GQA — 16 query : 4 kv heads (repeat 4), d_head=256 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000000) |
| FFN | dense FFN — intermediate 9216, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·4·256 = 2048 elems / token / layer; all 32 layers ⇒ 65536 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 32 |
| d_model | 2560 |
| n_h | 16 |
| n_kv | 4 |
| d_head | 256 |
| d_ff | 9216 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 248320 |
| ctx | 262144 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 24× linear_attention, 8× full_attention (총 32층) |
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
| d_chunk | 64 |
| d_head_ssm | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| d_conv | —  _(해당 없음: 이 모델은 `ssm` 계열 구조를 쓰지 않음)_ |
| n_mem | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| r_lora | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| d_attn | —  _(해당 없음: 이 모델은 `shared_block` 계열 구조를 쓰지 않음)_ |
| n_h_lin_k | 16 |
| n_h_lin_v | 32 |
| d_head_lin_k | 128 |
| d_head_lin_v | 128 |
| d_conv_lin | 4 |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **528,337개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 227,972 | 43.15% |
| 이 모듈 스코프의 심볼 | 175,236 | 33.17% |
| 이름 없음 (정수 유지) | 71,543 | 13.54% |
| 이 모듈 스코프의 유도식 | 25,577 | 4.84% |
| 스코프 없는 심볼 | 24,377 | 4.61% |
| 휴리스틱: 심볼의 배수 | 2,064 | 0.39% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 896 | 0.17% |
| 휴리스틱: 심볼+1 | 672 | 0.13% |

등록된 규칙 **453,162축**, 약한 근거 896축, 휴리스틱 **2,736축 (0.52%)**, 이름 없음 71,543축.

지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):

| 모듈 | 라벨 | 규칙 | 축 수 |
|---|---|---|---:|
| `model.layers.3.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.7.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.11.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.15.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.19.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.23.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.27.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |
| `model.layers.31.self_attn` | `2*d_head` | 휴리스틱: 심볼의 배수 | 6 |

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 2 | n_v/n_k (DeltaNet key head 하나가 담당하는 value head 수) | linear_attn |
| 18 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | linear_attn |
| 24 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | linear_attn |
| 192 | d_head − d_rope (부분 RoPE 비회전 통과분, partial_rotary_factor 기준) | self_attn |
| 512 | 2·d_head (CSA/HCA 압축기 kv_proj·gate_proj 폭: Ca⊕Cb) | self_attn |
| 544 | T·n_h_lin_v (value head 축까지 flatten — gated norm 입력) | linear_attn, norm |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 2048 | n_k·d_k (DeltaNet key_dim — q/k 조각 폭) | linear_attn |
| 4096 | n_v·d_v (DeltaNet value_dim — v/z 조각 폭) | in_proj_z, linear_attn, o_proj, out_proj, self_attn |
| 8192 | 2·key_dim + value_dim (gated delta net conv1d 채널 폭) | conv1d, in_proj_qkv, linear_attn, q_proj, self_attn |

## 레이어 구조

- layer 0-2: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 3: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 4-6: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 7: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 8-10: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 11: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 12-14: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 15: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 16-18: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 19: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 20-22: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 23: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 24-26: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 27: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 28-30: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 31: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 32 == 32 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2560 in 32/32 layers |
| C6 | PASS | hidden_size=2560 (heuristic check, 1104 flagged) |
| C7 | PASS | GQA 16:4 (repeat factor 4) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=248320, tie_word_embeddings=True |
| C10 | PASS | all 426 params covered |
| C11 | PASS | 73 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 21498 unmapped rows, 27 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Qwen/Qwen3.5-4B` config.json @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (sha256 `92c14622dda4…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 1건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 이름 없음이 정답 | 3 |
| 교정 필요 | 6 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `linear_attn\.norm$` | `d_head_lin_k` | `d_head_lin_v` | 1488 | transformers 5.14.1 installed source modeling_qwen3_5.py:248-558; revalidated this axis verdict unchanged. modeling_qwen3_next.py:552 `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, ...)`, :519 `self.head_v_dim = config.linear_value_head_dim`. Qwen3.5 는 같은 블록을 쓴다. |
| `linear_attn$` | `d_head_lin_k` | `d_head_lin_v` | 240 | transformers 5.14.1 installed source modeling_qwen3_5.py:248-558; revalidated this axis verdict unchanged. modeling_qwen3_5.py:350-351은 key와 value의 마지막 폭을 각각 k_head_dim과 v_head_dim으로 읽고, :364-367의 세 번째 select는 `v_t = value[:, :, i]`다. 또한 :493-495에서 value는 `self.head_v_dim`으로 reshape된다. 따라서 nth 2 select 출력의 마지막 축은 값이 같은 d_head_lin_k가 아니라 d_head_lin_v다. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.linear_attn` | in_proj_qkvz 조각 폭 (27B 에서 2048) | `2*n_kv*d_head` | `key_dim (= n_h_lin_k · d_head_lin_k)` | `modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다 … |
| `model.layers.*.linear_attn` | matmul 수축 축 (128) | `d_head_lin_k / d_head_lin_v 혼용` | `(소스가 가리키는 쪽 — 근거 참조)` | `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없 … |
| `model.layers.*.linear_attn.norm` | 정규화 폭 128 | `d_head_lin_k` | `d_head_lin_v` | `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head  … |
| `model.layers.*.linear_attn` | gated delta rule 청크 길이 64 (chunk_size) | `d_rope` | `d_chunk` | `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

C1   PASS   32 == 32
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2560 in 32/32 layers
C6   PASS   hidden_size=2560 (heuristic check, 1104 flagged)
C7   PASS   GQA 16:4 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=248320, tie_word_embeddings=True
C10  PASS   all 426 params covered
C11  PASS   73 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   21498 unmapped rows, 27 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.eye.default', 'aten.masked_fill.Scalar', 'aten.ones.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              unsqueeze        [B,T] -> [B,1,T]
  model                                              expand           [B,1,T] -> [4,B,T]
  model                                              select           [4,B,T] -> [B,T]
  model                                              slice            [4,B,T] -> [3,B,T]
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,1,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,1,d_rope/2] -> [B,1,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,d_rope/2,1] -> [3,B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [3,B,T] -> [3,B,1,T]
  model.rotary_emb                                   _to_copy         [3,B,1,T] -> [3,B,1,T]
  model.rotary_emb                                   expand           [3,B,d_rope/2,1] -> [3,B,d_rope/2,1]
  model.rotary_emb                                   view             [3,B,d_rope/2,1] -> [3,d_rope/2,B]
  model.rotary_emb                                   expand           [3,B,1,T] -> [3,B,1,T]
  model.rotary_emb                                   view             [3,B,1,T] -> [3,B,T]
  model.rotary_emb                                   batched_matmul   [3,d_rope/2,B]*[3,B,T] -> [3,d_rope/2,T]
  model.rotary_emb                                   _unsafe_view     [3,d_rope/2,T] -> [3,B,d_rope/2,T]
  model.rotary_emb                                   transpose        [3,B,d_rope/2,T] -> [3,B,T,d_rope/2]
  model.rotary_emb                                   select           [3,B,T,d_rope/2] -> [B,T,d_rope/2]
  model.rotary_emb                                   slice            [B,T,d_rope/2] -> [B,T,11]
  model.rotary_emb                                   copy_            [B,T,11]*[B,T,11] -> [B,T,11]
  model.rotary_emb                                   slice            [B,T,d_rope/2] -> [B,T,10]
  model.rotary_emb                                   copy_            [B,T,10]*[B,T,10] -> [B,T,10]
  model.rotary_emb                                   concat           [B,T,d_rope/2]*[B,T,d_rope/2] -> [B,T,d_rope]
  model.rotary_emb                                   cos              [B,T,d_rope] -> [B,T,d_rope]
  model.rotary_emb                                   elementwise_mul  [B,T,d_rope] -> [B,T,d_rope]
  model.rotary_emb                                   sin              [B,T,d_rope] -> [B,T,d_rope]
  model.rotary_emb                                   _to_copy         [B,T,d_rope] -> [B,T,d_rope]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.input_layernorm                     _to_copy         [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_add  [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.linear_attn.in_proj_qkv             t                [2*n_h*d_head,d_model] -> w=[2*n_h*d_head,d_model] [d_model,2*n_h*d_head]
  model.layers.N.linear_attn.in_proj_qkv             view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_qkv             matmul           [T,d_model]*[d_model,2*n_h*d_head] -> w=[2*n_h*d_head,d_model] [T,2*n_h*d_head]
  model.layers.N.linear_attn.in_proj_qkv             _unsafe_view     [T,2*n_h*d_head] -> [B,T,2*n_h*d_head]
  model.layers.N.linear_attn                         transpose        [B,T,2*n_h*d_head] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn.in_proj_z               t                [n_v*d_v,d_model] -> w=[n_v*d_v,d_model] [d_model,n_v*d_v]
  model.layers.N.linear_attn.in_proj_z               view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_z               matmul           [T,d_model]*[d_model,n_v*d_v] -> w=[n_v*d_v,d_model] [T,n_v*d_v]
  model.layers.N.linear_attn.in_proj_z               _unsafe_view     [T,n_v*d_v] -> [B,T,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,T,n_v*d_v] -> [B,T,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.in_proj_b               t                [n_h_lin_v,d_model] -> w=[n_h_lin_v,d_model] [d_model,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_b               view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_b               matmul           [T,d_model]*[d_model,n_h_lin_v] -> w=[n_h_lin_v,d_model] [T,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_b               _unsafe_view     [T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               t                [n_h_lin_v,d_model] -> w=[n_h_lin_v,d_model] [d_model,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_a               matmul           [T,d_model]*[d_model,n_h_lin_v] -> w=[n_h_lin_v,d_model] [T,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               _unsafe_view     [T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         constant_pad_nd  [B,2*n_h*d_head,T] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         zeros            [] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         copy_            [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn.conv1d                  conv1d           [B,2*n_h*d_head,T]*[2*n_h*d_head,1,d_conv_lin] -> w=[2*n_h*d_head,1,d_conv_lin] [B,2*n_h*d_head,20]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,20] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn                         silu             [B,2*n_h*d_head,T] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn                         transpose        [B,2*n_h*d_head,T] -> [B,T,2*n_h*d_head]
  model.layers.N.linear_attn                         split_with_sizes [B,T,2*n_h*d_head] -> [B,T,n_k*d_k]*[B,T,n_k*d_k]*[B,T,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,T,n_k*d_k] -> [B,T,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,T,n_v*d_v] -> [B,T,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         sigmoid          [B,T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         _to_copy         [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         exp              [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         neg              [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         elementwise_add  [B,T,n_h_lin_v]*[n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         softplus         [B,T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         elementwise_mul  [n_h_lin_v]*[B,T,n_h_lin_v] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         unsqueeze        [B,T,n_h_lin_k,d_head_lin_k] -> [B,T,n_h_lin_k,1,d_head_lin_k]
  model.layers.N.linear_attn                         expand           [B,T,n_h_lin_k,1,d_head_lin_k] -> [B,T,n_h_lin_k,n_v/n_k,d_head_lin_k]
  model.layers.N.linear_attn                         clone            [B,T,n_h_lin_k,n_v/n_k,d_head_lin_k] -> [B,T,n_h_lin_k,n_v/n_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,T,n_h_lin_k,n_v/n_k,d_head_lin_k] -> [B,T,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         elementwise_mul  [B,T,n_h_lin_v,d_head_lin_k]*[B,T,n_h_lin_v,d_head_lin_k] -> [B,T,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         sum              [B,T,n_h_lin_v,d_head_lin_k] -> [B,T,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_add  [B,T,n_h_lin_v,1] -> [B,T,n_h_lin_v,1]
  model.layers.N.linear_attn                         rsqrt            [B,T,n_h_lin_v,1] -> [B,T,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,T,n_h_lin_v,d_head_lin_k]*[B,T,n_h_lin_v,1] -> [B,T,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         transpose        [B,T,n_h_lin_v,d_head_lin_k] -> [B,n_h_lin_v,T,d_head_lin_k]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,T,d_head_lin_k] -> [B,n_h_lin_v,T,d_head_lin_k]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,T,d_head_lin_k] -> [B,n_h_lin_v,T,d_head_lin_k]
  model.layers.N.linear_attn                         transpose        [B,T,n_h_lin_v] -> [B,n_h_lin_v,T]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,T] -> [B,n_h_lin_v,T]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,T] -> [B,n_h_lin_v,T]
  model.layers.N.linear_attn                         constant_pad_nd  [B,n_h_lin_v,T,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         constant_pad_nd  [B,n_h_lin_v,T] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_chunk,d_head_lin_k]*[B,n_h_lin_v,d_chunk,1] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         ones             [] -> [d_chunk,2*n_v]
  model.layers.N.linear_attn                         triu             [d_chunk,2*n_v] -> [d_chunk,2*n_v]
  model.layers.N.linear_attn                         cumsum           [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,d_chunk,1]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,1,d_chunk]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,1,d_chunk,1]*[B,n_h_lin_v,1,1,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         tril             [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         transpose        [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_head_lin_k,d_chunk]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,1,d_head_lin_k,d_chunk] -> [B,n_h_lin_v,1,d_head_lin_k,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_head_lin_k]*[n_h_lin_v,d_head_lin_k,d_chunk] -> [n_h_lin_v,d_chunk,d_chunk]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,1,d_chunk,d_chunk]*[B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         masked_fill      [B,n_h_lin_v,1,d_chunk,d_chunk]*[d_chunk,2*n_v] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         neg              [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,1,64]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,1,64] -> [B,n_h_lin_v,1,1,1]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,1,1,1] -> [B,n_h_lin_v,1,1,1]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1,1]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,1,1]*[B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,1]*[B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,2]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,1,2] -> [B,n_h_lin_v,1,2]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1,2] -> [B,n_h_lin_v,1,2,1]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,2,2] -> [B,n_h_lin_v,1,2]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,1,2]*[B,n_h_lin_v,1,2] -> [B,n_h_lin_v,1,2]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,2]*[B,n_h_lin_v,1,2] -> [B,n_h_lin_v,1,2]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,3,3] -> [B,n_h_lin_v,1,3]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,1,3]*[B,n_h_lin_v,1,3] -> [B,n_h_lin_v,1,3]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,3]*[B,n_h_lin_v,1,3] -> [B,n_h_lin_v,1,3]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,4,4] -> [B,n_h_lin_v,1,4]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,1,4]*[B,n_h_lin_v,1,4] -> [B,n_h_lin_v,1,4]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,4]*[B,n_h_lin_v,1,4] -> [B,n_h_lin_v,1,4]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,5,5] -> [B,n_h_lin_v,1,5]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,5]*[B,n_h_lin_v,1,5] -> [B,n_h_lin_v,1,5]
  model.layers.N.linear_attn                         eye              [] -> [d_chunk,2*n_v]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_k] -> [n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         zeros            [] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         zeros_like       [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         transpose        [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_k,d_chunk]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,d_head_lin_k,d_chunk] -> [B,n_h_lin_v,d_head_lin_k,d_chunk]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_chunk] -> [B,n_h_lin_v,d_chunk,d_chunk]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,d_chunk,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_head_lin_k]*[n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,d_chunk,d_head_lin_v]*[B,n_h_lin_v,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,d_chunk,1] -> [B,n_h_lin_v,d_chunk,1]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_v] -> [n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,1]*[B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_head_lin_k,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_v] -> [n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,T,n_h_lin_v,d_head_lin_v] -> [B,T,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         zeros_like       [B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn.norm                    _to_copy         [n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.norm                    pow              [n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.norm                    mean             [n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,B]
  model.layers.N.linear_attn.norm                    elementwise_add  [n_h_lin_v*T,B] -> [n_h_lin_v*T,B]
  model.layers.N.linear_attn.norm                    rsqrt            [n_h_lin_v*T,B] -> [n_h_lin_v*T,B]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v*T,d_head_lin_v]*[n_h_lin_v*T,B] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [d_head_lin_v]*[n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.norm                    silu             [n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v*T,d_head_lin_v]*[n_h_lin_v*T,d_head_lin_v] -> [n_h_lin_v*T,d_head_lin_v]
  model.layers.N.linear_attn.out_proj                t                [d_model,n_v*d_v] -> w=[d_model,n_v*d_v] [n_v*d_v,d_model]
  model.layers.N.linear_attn.out_proj                view             [B,T,n_v*d_v] -> [T,n_v*d_v]
  model.layers.N.linear_attn.out_proj                matmul           [T,n_v*d_v]*[n_v*d_v,d_model] -> w=[d_model,n_v*d_v] [T,d_model]
  model.layers.N.linear_attn.out_proj                _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.act_fn                          silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.up_proj                         t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.up_proj                         view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.up_proj                         matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.mlp.up_proj                         _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    t                [2*n_h*d_head,d_model] -> w=[2*n_h*d_head,d_model] [d_model,2*n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,2*n_h*d_head] -> w=[2*n_h*d_head,d_model] [T,2*n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,2*n_h*d_head] -> [B,T,2*n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,2*n_h*d_head] -> [B,T,n_h,2*d_head]
  model.layers.N.self_attn                           split            [B,T,n_h,2*d_head] -> [B,T,n_h,d_head]*[B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           _unsafe_view     [B,T,n_h,d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,T,n_h,d_head] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,T,n_h,d_head]*[B,T,n_h,1] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,T,n_h,d_head]*[d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,T,n_kv,d_head] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,T,n_kv,1] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,T,n_kv,1] -> [B,T,n_kv,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,T,n_kv,d_head]*[B,T,n_kv,1] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,T,n_kv,d_head]*[d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_rope] -> [B,1,T,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_rope]*[B,1,T,d_rope] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_rope] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_rope/2] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_rope/2]*[B,n_h,T,d_rope/2] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_rope]*[B,n_h,T,d_rope] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_rope]*[B,1,T,d_rope] -> [B,n_kv,T,d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_rope] -> [B,n_kv,T,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_rope/2] -> [B,n_kv,T,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_rope/2]*[B,n_kv,T,d_rope/2] -> [B,n_kv,T,d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,T,d_rope]*[B,n_kv,T,d_rope] -> [B,n_kv,T,d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_rope]*[B,n_h,T,d_head-d_rope] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_rope]*[B,n_kv,T,d_head-d_rope] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           ones             [] -> [T,T]
  model.layers.N.self_attn                           tril             [T,T] -> [T,T]
  model.layers.N.self_attn                           scalar_tensor    [] -> []
  model.layers.N.self_attn                           where            [T,T]*[]*[] -> [T,T]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T,d_head] -> [B,n_kv,1,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_kv,n_h/n_kv,T,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,n_h/n_kv,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           sigmoid          [B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,T,n_h*d_head]*[B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,d_model] -> [B,T,d_model]
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
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.norm                                         _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.norm                                         pow              [B,T,d_model] -> [B,T,d_model]
  model.norm                                         mean             [B,T,d_model] -> [B,T,1]
  model.norm                                         elementwise_add  [B,T,1] -> [B,T,1]
  model.norm                                         rsqrt            [B,T,1] -> [B,T,1]
  model.norm                                         elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.norm                                         _to_copy         [d_model] -> [d_model]
  model.norm                                         elementwise_add  [d_model] -> [d_model]
  model.norm                                         elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
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
  model                                              unsqueeze        [B,1] -> [B,1,1]
  model                                              expand           [B,1,1] -> [4,B,1]
  model                                              select           [4,B,1] -> [B,1]
  model                                              slice            [4,B,1] -> [3,B,1]
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,1,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,1,d_rope/2] -> [B,1,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,d_rope/2,1] -> [3,B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [3,B,1] -> [3,B,1,1]
  model.rotary_emb                                   _to_copy         [3,B,1,1] -> [3,B,1,1]
  model.rotary_emb                                   expand           [3,B,d_rope/2,1] -> [3,B,d_rope/2,1]
  model.rotary_emb                                   view             [3,B,d_rope/2,1] -> [3,d_rope/2,B]
  model.rotary_emb                                   expand           [3,B,1,1] -> [3,B,1,1]
  model.rotary_emb                                   view             [3,B,1,1] -> [3,B,1]
  model.rotary_emb                                   batched_matmul   [3,d_rope/2,B]*[3,B,1] -> [3,d_rope/2,B]
  model.rotary_emb                                   _unsafe_view     [3,d_rope/2,B] -> [3,B,d_rope/2,1]
  model.rotary_emb                                   transpose        [3,B,d_rope/2,1] -> [3,B,1,d_rope/2]
  model.rotary_emb                                   select           [3,B,1,d_rope/2] -> [B,1,d_rope/2]
  model.rotary_emb                                   slice            [B,1,d_rope/2] -> [B,1,11]
  model.rotary_emb                                   copy_            [B,1,11]*[B,1,11] -> [B,1,11]
  model.rotary_emb                                   slice            [B,1,d_rope/2] -> [B,1,10]
  model.rotary_emb                                   copy_            [B,1,10]*[B,1,10] -> [B,1,10]
  model.rotary_emb                                   concat           [B,1,d_rope/2]*[B,1,d_rope/2] -> [B,1,d_rope]
  model.rotary_emb                                   cos              [B,1,d_rope] -> [B,1,d_rope]
  model.rotary_emb                                   elementwise_mul  [B,1,d_rope] -> [B,1,d_rope]
  model.rotary_emb                                   sin              [B,1,d_rope] -> [B,1,d_rope]
  model.rotary_emb                                   _to_copy         [B,1,d_rope] -> [B,1,d_rope]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.input_layernorm                     _to_copy         [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_add  [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.N.linear_attn.in_proj_qkv             t                [2*n_h*d_head,d_model] -> w=[2*n_h*d_head,d_model] [d_model,2*n_h*d_head]
  model.layers.N.linear_attn.in_proj_qkv             view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_qkv             matmul           [B,d_model]*[d_model,2*n_h*d_head] -> w=[2*n_h*d_head,d_model] [B,2*n_h*d_head]
  model.layers.N.linear_attn.in_proj_qkv             _unsafe_view     [B,2*n_h*d_head] -> [B,1,2*n_h*d_head]
  model.layers.N.linear_attn                         transpose        [B,1,2*n_h*d_head] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn.in_proj_z               t                [n_v*d_v,d_model] -> w=[n_v*d_v,d_model] [d_model,n_v*d_v]
  model.layers.N.linear_attn.in_proj_z               view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_z               matmul           [B,d_model]*[d_model,n_v*d_v] -> w=[n_v*d_v,d_model] [B,n_v*d_v]
  model.layers.N.linear_attn.in_proj_z               _unsafe_view     [B,n_v*d_v] -> [B,1,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,1,n_v*d_v] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.in_proj_b               t                [n_h_lin_v,d_model] -> w=[n_h_lin_v,d_model] [d_model,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_b               view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_b               matmul           [B,d_model]*[d_model,n_h_lin_v] -> w=[n_h_lin_v,d_model] [B,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_b               _unsafe_view     [B,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               t                [n_h_lin_v,d_model] -> w=[n_h_lin_v,d_model] [d_model,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_a               matmul           [B,d_model]*[d_model,n_h_lin_v] -> w=[n_h_lin_v,d_model] [B,n_h_lin_v]
  model.layers.N.linear_attn.in_proj_a               _unsafe_view     [B,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         squeeze          [2*n_h*d_head,1,d_conv_lin] -> w=[2*n_h*d_head,1,d_conv_lin] [2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         concat           [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,1] -> [B,2*n_h*d_head,5]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,5] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         copy_            [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         unsqueeze        [2*n_h*d_head,d_conv_lin] -> [2*n_h*d_head,B,d_conv_lin]
  model.layers.N.linear_attn                         conv1d           [B,2*n_h*d_head,5]*[2*n_h*d_head,B,d_conv_lin] -> [B,2*n_h*d_head,n_v/n_k]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,n_v/n_k] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn                         silu             [B,2*n_h*d_head,1] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn                         transpose        [B,2*n_h*d_head,1] -> [B,1,2*n_h*d_head]
  model.layers.N.linear_attn                         split_with_sizes [B,1,2*n_h*d_head] -> [B,1,n_k*d_k]*[B,1,n_k*d_k]*[B,1,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,1,n_k*d_k] -> [B,1,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,1,n_v*d_v] -> [B,1,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         sigmoid          [B,1,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         _to_copy         [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         exp              [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         neg              [n_h_lin_v] -> [n_h_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,1,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         elementwise_add  [B,1,n_h_lin_v]*[n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         softplus         [B,1,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         elementwise_mul  [n_h_lin_v]*[B,1,n_h_lin_v] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         unsqueeze        [B,1,n_h_lin_k,d_head_lin_k] -> [B,1,n_h_lin_k,1,d_head_lin_k]
  model.layers.N.linear_attn                         expand           [B,1,n_h_lin_k,1,d_head_lin_k] -> [B,1,n_h_lin_k,n_v/n_k,d_head_lin_k]
  model.layers.N.linear_attn                         clone            [B,1,n_h_lin_k,n_v/n_k,d_head_lin_k] -> [B,1,n_h_lin_k,n_v/n_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,1,n_h_lin_k,n_v/n_k,d_head_lin_k] -> [B,1,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         elementwise_mul  [B,1,n_h_lin_v,d_head_lin_k]*[B,1,n_h_lin_v,d_head_lin_k] -> [B,1,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         sum              [B,1,n_h_lin_v,d_head_lin_k] -> [B,1,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_add  [B,1,n_h_lin_v,1] -> [B,1,n_h_lin_v,1]
  model.layers.N.linear_attn                         rsqrt            [B,1,n_h_lin_v,1] -> [B,1,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,1,n_h_lin_v,d_head_lin_k]*[B,1,n_h_lin_v,1] -> [B,1,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         transpose        [B,1,n_h_lin_v,d_head_lin_k] -> [B,n_h_lin_v,1,d_head_lin_k]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,1,d_head_lin_k]
  model.layers.N.linear_attn                         transpose        [B,1,n_h_lin_v] -> [B,n_h_lin_v,1]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,1] -> [B,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,1,d_head_lin_k]
  model.layers.N.linear_attn                         zeros            [] -> [B,n_h_lin_v,1,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1] -> [B,n_h_lin_v]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v] -> [B,n_h_lin_v]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v] -> [B,n_h_lin_v,1]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]*[B,n_h_lin_v,1,1] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_k,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_k,1] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,d_head_lin_v] -> [B,n_h_lin_v,1,d_head_lin_v]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,d_head_lin_k]*[B,n_h_lin_v,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         transpose        [B,n_h_lin_v,1,d_head_lin_k] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,1,n_h_lin_v,d_head_lin_v] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         view             [B,1,n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    _to_copy         [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    pow              [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    mean             [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    elementwise_add  [n_h_lin_v,B] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    rsqrt            [n_h_lin_v,B] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v,d_head_lin_v]*[n_h_lin_v,B] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [d_head_lin_v]*[n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    silu             [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v,d_head_lin_v]*[n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         view             [n_h_lin_v,d_head_lin_v] -> [B,1,n_v*d_v]
  model.layers.N.linear_attn.out_proj                t                [d_model,n_v*d_v] -> w=[d_model,n_v*d_v] [n_v*d_v,d_model]
  model.layers.N.linear_attn.out_proj                view             [B,1,n_v*d_v] -> [B,n_v*d_v]
  model.layers.N.linear_attn.out_proj                matmul           [B,n_v*d_v]*[n_v*d_v,d_model] -> w=[d_model,n_v*d_v] [B,d_model]
  model.layers.N.linear_attn.out_proj                _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.act_fn                          silu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.up_proj                         t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.up_proj                         view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.up_proj                         matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.mlp.up_proj                         _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn.q_proj                    t                [2*n_h*d_head,d_model] -> w=[2*n_h*d_head,d_model] [d_model,2*n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,2*n_h*d_head] -> w=[2*n_h*d_head,d_model] [B,2*n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,2*n_h*d_head] -> [B,1,2*n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,2*n_h*d_head] -> [B,1,n_h,2*d_head]
  model.layers.N.self_attn                           split            [B,1,n_h,2*d_head] -> [B,1,n_h,d_head]*[B,1,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           _unsafe_view     [B,1,n_h,d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,1,n_h,d_head] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,1,n_h,d_head]*[B,1,n_h,1] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,1,n_h,d_head]*[d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,1,n_kv,d_head] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,1,n_kv,1] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,1,n_kv,1] -> [B,1,n_kv,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,n_kv,d_head]*[B,1,n_kv,1] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,n_kv,d_head]*[d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_rope] -> [B,1,1,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head-d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head-d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_rope]*[B,1,1,d_rope] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_rope] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_rope/2] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_rope/2]*[B,n_h,1,d_rope/2] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_rope]*[B,n_h,1,d_rope] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_rope]*[B,1,1,d_rope] -> [B,n_kv,1,d_rope]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_rope] -> [B,n_kv,1,d_rope/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_rope/2] -> [B,n_kv,1,d_rope/2]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_rope/2]*[B,n_kv,1,d_rope/2] -> [B,n_kv,1,d_rope]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,1,d_rope]*[B,n_kv,1,d_rope] -> [B,n_kv,1,d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_rope]*[B,n_h,1,d_head-d_rope] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_rope]*[B,n_kv,1,d_head-d_rope] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,n_kv,T+1,d_head] -> [B,n_kv,1,T+1,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,1,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           clone            [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_kv,n_h/n_kv,T+1,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,n_h/n_kv,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           sigmoid          [B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,1,n_h*d_head]*[B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,d_model] -> [B,1,d_model]
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
  model.layers.24                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.norm                                         _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.norm                                         pow              [B,1,d_model] -> [B,1,d_model]
  model.norm                                         mean             [B,1,d_model] -> [B,1,1]
  model.norm                                         elementwise_add  [B,1,1] -> [B,1,1]
  model.norm                                         rsqrt            [B,1,1] -> [B,1,1]
  model.norm                                         elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.norm                                         _to_copy         [d_model] -> [d_model]
  model.norm                                         elementwise_add  [d_model] -> [d_model]
  model.norm                                         elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
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

