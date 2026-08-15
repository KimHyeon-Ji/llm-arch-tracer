# 리뷰 패킷 — Zyphra/Zamba2-1.2B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe` / 트레이스 seq_len(T) = 16
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
  L            = 38
  d_model      = 2048
  n_h          = 32
  n_kv         = 32
  d_head       = 128
  d_ff         = 8192
  d_shared     = None
  V            = 32000
  ctx          = 4096
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention']
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
  n_g_ssm      = 1
  n_h_ssm      = 64
  d_chunk      = 256
  d_head_ssm   = 64
  d_conv       = 4
  n_mem        = 1
  r_lora       = 128
  d_attn       = 4096
  n_h_lin_k    = None
  n_h_lin_v    = None
  d_head_lin_k = None
  d_head_lin_v = None
  d_conv_lin   = None
```

## 3. 모델 요약 산출물

# Model Summary -- Zyphra/Zamba2-1.2B

## 기본 정보

- revision: `6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 1.22B total (dense) |
| 2 | Context (tokens) | 4,096  _(config max_position_embeddings)_ |
| 3 | DATE | 2024-08-16  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MHA |
| 6 | LAYER MIX | 32× linear_attention, 6× hybrid |
| 7 | KV CACHE / TOKEN (BF16) | 96.0 KiB (Moderate) over 6 attn layers |
| 8 | KEY DETAIL | MHA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, MHA, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `zamba2` |
| attention | MHA — 32 heads (no GQA/MQA), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000) |
| FFN | dense FFN — intermediate 8192, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·32·128 = 8192 elems / token / layer; all 38 layers ⇒ 311296 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 38 |
| d_model | 2048 |
| n_h | 32 |
| n_kv | 32 |
| d_head | 128 |
| d_ff | 8192 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 32000 |
| ctx | 4096 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 32× linear_attention, 6× hybrid (총 38층) |
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
| n_g_ssm | 1 |
| n_h_ssm | 64 |
| d_chunk | 256 |
| d_head_ssm | 64 |
| d_conv | 4 |
| n_mem | 1 |
| r_lora | 128 |
| d_attn | 4096 |
| n_h_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| n_h_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_k | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_head_lin_v | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |
| d_conv_lin | —  _(해당 없음: 이 모델은 `linear_attn` 계열 구조를 쓰지 않음)_ |

## 라벨 출처 (이 표의 이름들이 어디서 왔나)

shape 축 **136,178개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 52,482 | 38.54% |
| 이 모듈 스코프의 심볼 | 51,047 | 37.49% |
| 스코프 없는 심볼 | 15,345 | 11.27% |
| 이 모듈 스코프의 유도식 | 8,332 | 6.12% |
| 이름 없음 (정수 유지) | 5,648 | 4.15% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,324 | 2.44% |

등록된 규칙 **127,206축**, 약한 근거 3,324축, 휴리스틱 **0축 (0.0%)**, 이름 없음 5,648축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 19 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mamba |
| 4352 | d_inner + 2·n_g·d_state (conv1d 입력 폭) | act, conv1d, mamba |
| 8512 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mamba |
| 16384 | 2·d_ff (dense FFN gate+up 융합 투영 폭) | 1, feed_forward, gate_up_proj |

## 레이어 구조

- layer 0-4: input_layernorm, mamba
- layer 5: linear, mamba_decoder, shared_transformer
- layer 6-10: input_layernorm, mamba
- layer 11: linear, mamba_decoder, shared_transformer
- layer 12-16: input_layernorm, mamba
- layer 17: linear, mamba_decoder, shared_transformer
- layer 18-22: input_layernorm, mamba
- layer 23: linear, mamba_decoder, shared_transformer
- layer 24-28: input_layernorm, mamba
- layer 29: linear, mamba_decoder, shared_transformer
- layer 30-34: input_layernorm, mamba
- layer 35: linear, mamba_decoder, shared_transformer
- layer 36-37: input_layernorm, mamba

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 38 == 38 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layers_block_type'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2048 in 38/38 layers |
| C6 | PASS | hidden_size=2048 (heuristic check, 0 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=32000, tie_word_embeddings=True |
| C10 | PASS | all 406 params covered |
| C11 | PASS | 69 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 4468 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Zyphra/Zamba2-1.2B` config.json @ `6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe` (sha256 `025b74b13fef…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 3건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 8 |
| 교정 필요 | 3 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:518-520에서 C를 `[batch_size, num_groups, num_heads // num_groups, state_size]`로 expand한다. 이 모델은 num_groups=1이므로 출력 축 2는 head 폭이 아니라 펼쳐진 SSM head 수 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:657,693,697,731에서 A_log는 `self.num_heads` 길이의 파라미터이고 `A = -torch.exp(self.A_log.float())`로 사용된다. 이 벡터에서 시작한 unsqueeze 입력 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 32 | modeling_zamba2.py:701,803-811에서 D는 `self.num_heads` 길이이고 recurrent 경로에서 `self.D[:, None]`로 unsqueeze한 뒤 head_dim으로 expand한다. 따라서 unsqueeze 입력의 유일한 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 24 | modeling_zamba2.py:555,573-580에서 A는 `[B,T,num_heads]`이고 reshape_into_chunks는 sequence 축만 pad해 `[B,n_chunks,chunk_size,num_heads]`로 만든다. 한 chunk 표본의 마지막 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 24 | modeling_zamba2.py:555,578,600에서 hidden_states는 chunk 뒤 `[B,n_chunks,chunk_size,num_heads,head_dim]`이고 곱셈을 위해 `[B,n_chunks,num_heads,chunk_size,head_dim]`으로 놓인다. 마지막 축은 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 24 | modeling_zamba2.py:596-609에서 states와 previous_states는 모두 `[B,n_chunks,num_heads,head_dim,state_size]` 순서로 concat된다. 입력 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 30 | modeling_zamba2.py:596-600에서 chunk state를 `[batch_size, num_chunks, num_heads, head_dim, state_size]` 순서로 만든다. num_chunks=1인 이 출력에서 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 30 | modeling_zamba2.py:596-600에서 chunk state 순서는 `[batch_size, num_chunks, num_heads, head_dim, state_size]`다. 앞선 축 2 교정 뒤에도 축 3은 head 수가 아니라 d_head_ssm이다. |
| `self_attn$` | `n_h` | `n_kv` | 18 | modeling_zamba2.py:255,312-318에서 key_states는 num_key_value_heads로 reshape되고 RoPE는 마지막 d_head 축만 절반으로 slice한다. key의 nth 3 slice 출력 축 1은 n_h가 아니라 n_kv다. |
| `self_attn$` | `n_h` | `n_kv` | 18 | modeling_zamba2.py:255,312-318의 같은 key RoPE slice를 decode 길이 1에서 본 앵커다. 마지막 head_dim만 절반으로 자르므로 축 1은 n_kv다. |
| `self_attn$` | `n_h` | `n_kv` | 12 | modeling_zamba2.py:255,300-314에서 두 번째 view는 k_proj 출력을 `[batch_size,sequence_length,num_key_value_heads,head_dim]`으로 만든다. 축 2는 n_h가 아니라 n_kv다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 24 | modeling_zamba2.py:555,573-580에서 A는 num_heads 폭이고 chunk 계산 전에 head 축을 앞으로 옮긴다. 그 경로의 `[num_heads,1]` unsqueeze 출력 축 0은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:555,573-575에서 A와 dt는 `[B,T,num_heads]`이고 dt에 singleton을 붙이는 동안 head 축은 보존된다. `[B,T,num_heads,1]`의 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 18 | modeling_zamba2.py:600,609-612에서 recurrence 뒤 state를 다시 `[B,n_chunks+1,num_heads,head_dim,state_size]` 순서로 되돌린다. n_chunks+1=2인 출력의 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 18 | modeling_zamba2.py:600,609-612에서 state를 `[B,n_chunks+1,num_heads,head_dim,state_size]` 순서로 되돌린다. 앞선 축 2 교정 뒤 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 24 | modeling_zamba2.py:596-609에서 concat 입력은 `[B,n_chunks,num_heads,head_dim,state_size]` 순서다. 앞선 축 2 교정 뒤 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 24 | modeling_zamba2.py:600,609-611에서 누적 state는 계산을 위해 `[B,num_heads,n_chunks+1,head_dim,state_size]`로 permute된다. num_chunks+1=2인 출력의 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 24 | modeling_zamba2.py:600,609-612에서 `new_states[:, :-1]`는 chunk 축만 자르며 `[B,n_chunks,num_heads,head_dim,state_size]` 순서를 유지한다. 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 24 | modeling_zamba2.py:600,609-612에서 `new_states[:, :-1]`는 `[B,n_chunks,num_heads,head_dim,state_size]` 순서를 유지한다. 앞선 축 2 교정 뒤 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 24 | modeling_zamba2.py:758-760에서 dt의 마지막 폭은 `self.num_heads`이고 decode에서 sequence 축 하나를 선택하면 `[B,num_heads]`가 된다. 따라서 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 320 | modeling_zamba2.py:580-585에서 A는 `[B,num_heads,n_chunks,chunk_size]`이고 `segment_sum(A)`도 그 prefix를 보존한다. :377-394의 expand가 만드는 뒤 두 축만 chunk_size이므로 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 320 | modeling_zamba2.py:580-581에서 A_cumsum은 `[B,num_heads,n_chunks,chunk_size]`이고 :610은 마지막 chunk 위치를 pad한 뒤 segment_sum한다. 따라서 `[B,num_heads,2,2]`의 축 1은 n_h_ssm이며 뒤의 2만 chunk 경계 축이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 288 | modeling_zamba2.py:555-566에서 B/C는 `[B,T,num_groups,state_size]`에서 head 축을 repeat_interleave해 `[B,T,num_heads,state_size]`가 된다. :568,578의 padding은 sequence 축만 늘리므로 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 288 | modeling_zamba2.py:805-807은 hidden_states를 `[B,num_heads,head_dim]`으로 view하고 dt를 `[B,num_heads,1] -> [B,num_heads,head_dim]`으로 expand한다. 따라서 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 288 | modeling_zamba2.py:805-807의 같은 expand 출력은 `[B,num_heads,head_dim]`이다. 앞 교정으로 축 1을 n_h_ssm으로 고친 뒤의 앵커 shape를 썼고, 축 2는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 224 | modeling_zamba2.py:555와 :568-574에서 hidden_states는 `[B,T,num_heads,head_dim]`이고 padding은 sequence 축만 늘린다. 따라서 `[B,d_chunk,num_heads,head_dim]`의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 288 | modeling_zamba2.py:803-806은 recurrent hidden_states를 명시적으로 `[batch_size,self.num_heads,self.head_dim]`으로 view한다. 따라서 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 288 | modeling_zamba2.py:803-806의 같은 view 출력은 `[B,num_heads,head_dim]`이다. 앞 교정 이후 shape에서 축 2는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 224 | modeling_zamba2.py:555,568-574에서 이 padding 출력은 `[B,d_chunk,num_heads,head_dim]`이다. 앞 교정으로 축 2를 n_h_ssm으로 고친 앵커에서 마지막 축은 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 224 | modeling_zamba2.py:555-566은 B와 C를 모두 `[B,T,num_heads,state_size]`로 repeat하고 :568,578은 sequence 축만 pad한다. nth 5의 C 경로에서도 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:555-566에서 B는 `[B,T,num_groups,state_size]`의 group 축을 num_heads까지 repeat_interleave한다. 그 분해 expand의 축 3은 head 폭이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:587-591에서 G는 `(b,c,l,s,h,n)`의 state 축 n을 합쳐 `[B,n_chunks,l,s,num_heads]`가 되고 `G[...,None]`이 마지막 singleton을 붙인다. 따라서 `[B,1,d_chunk,d_chunk,num_heads,1]`의 축 4는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:555-566에서 C도 B와 똑같이 group 축을 num_heads까지 repeat_interleave한다. 그 분해 expand의 축 3은 head 폭이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:596-612에서 states는 `[B,n_chunks,num_heads,head_dim,state_size]`, new_states도 `[B,n_chunks+1,num_heads,head_dim,state_size]`이고 `new_states[:,-1]`은 `[B,num_heads,head_dim,state_size]`다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 192 | modeling_zamba2.py:596-612의 final_state는 `[B,num_heads,head_dim,state_size]`다. 앞 교정 이후 앵커에서 축 2는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:491-507에서 B는 `[B,num_groups,1,state_size]`에서 `[B,num_groups,num_heads//num_groups,state_size]`로 expand된다. n_groups=1인 이 앵커의 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 320 | modeling_zamba2.py:657,693,697,731에서 A_log는 `self.num_heads` 길이의 파라미터이고 `A = -torch.exp(self.A_log.float())`로 읽힌다. exp 입력 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 160 | modeling_zamba2.py:831-836은 hidden_states를 명시적으로 `[batch_size,seq_len,self.num_heads,self.head_dim]`으로 view한다. 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 192 | modeling_zamba2.py:518-520에서 C는 `[B,num_groups,1,state_size] -> [B,num_groups,num_heads//num_groups,state_size]`로 expand된다. n_groups=1인 이 앵커의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 160 | modeling_zamba2.py:831-836의 hidden_states view 출력은 `[B,T,self.num_heads,self.head_dim]`이다. 앞 교정 이후 shape에서 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 160 | modeling_zamba2.py:596-600에서 chunk state는 `[B,n_chunks,num_heads,head_dim,state_size]` 순서다. n_chunks=1인 출력의 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 160 | modeling_zamba2.py:596-600의 같은 states 출력에서 축 3은 head_dim이다. 앞 교정 이후 앵커 shape를 썼고, n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 132 | modeling_zamba2.py:491-511의 동일한 Zamba2MambaMixer selective update는 `[B,num_heads,head_dim,state_size]`를 만든다. 중첩 mixer에서도 축 2는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 128 | modeling_zamba2.py:555,573-580에서 A는 `[B,T,num_heads]`이고 reshape helper는 sequence 축만 pad해 `[B,n_chunks,d_chunk,num_heads]`로 만든다. 접힌 `[B,d_chunk,num_heads]` 출력의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 128 | modeling_zamba2.py:555,578은 hidden_states를 `[B,c,l,num_heads,head_dim]`으로 만들고 :600의 broadcast가 내부적으로 `[B,c,num_heads,l,head_dim]` permute를 낸다. 따라서 그 출력의 마지막 축은 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 128 | modeling_zamba2.py:596-609에서 states와 previous_states는 모두 `[B,n_chunks,num_heads,head_dim,state_size]` 순서이고 chunk 축으로 concat한다. concat 첫 입력의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 128 | modeling_zamba2.py:596-609의 같은 state concat 입력에서 축 3은 head_dim이다. 앞 교정 이후 앵커 shape를 썼고, n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 128 | modeling_zamba2.py:600,609-611의 recurrence state는 `[B,n_chunks+1,num_heads,head_dim,state_size]`이고 decay와 곱하기 위해 내부적으로 `[B,num_heads,n_chunks+1,head_dim,state_size]`가 된다. 축 3은 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 128 | modeling_zamba2.py:758-760에서 dt의 마지막 폭은 self.num_heads이고 :803-806에서 단일 token dt를 transpose하기 전에 sequence 축을 select한다. 남은 `[B,num_heads]` 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 128 | modeling_zamba2.py:600,609-612에서 new_states는 `[B,n_chunks+1,num_heads,head_dim,state_size]`이고 `new_states[:,:-1]`은 chunk 축만 자른다. 축 2는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 128 | modeling_zamba2.py:600,609-612의 같은 states slice에서 축 3은 head_dim이다. 앞 교정 이후 앵커 shape를 썼고, n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 120 | modeling_zamba2.py:491-511의 state update 출력은 중첩 mixer에서도 `[B,num_heads,head_dim,state_size]`다. 앞 교정 이후 shape에서 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 96 | modeling_zamba2.py:609-612에서 recurrence 결과를 다시 `[B,n_chunks+1,num_heads,head_dim,state_size]` 순서로 돌린다. n_chunks+1=2인 출력의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 96 | modeling_zamba2.py:609-612의 recurrence 출력은 `[B,n_chunks+1,num_heads,head_dim,state_size]`다. 앞 교정 이후 shape에서 축 3은 n_h_ssm이 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 96 | modeling_zamba2.py:555-578,357-373의 같은 Zamba2MambaMixer chunk helper는 hidden_states를 `[B,n_chunks,d_chunk,num_heads,head_dim]`으로 만든다. 중첩 mixer padding 출력의 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 96 | modeling_zamba2.py:555-578,357-373의 중첩 mixer hidden_states padding 출력에서 마지막 축은 head_dim이다. 앞 교정 이후 앵커 shape에서 축 3은 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 84 | modeling_zamba2.py:580에서 A를 `[B,num_heads,n_chunks,chunk_size]`로 permute한다. 같은 클래스를 쓰는 중첩 mixer에서도 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `self_attn$` | `n_h` | `n_kv` | 102 | modeling_zamba2.py:255,300-314에서 두 번째 transpose는 key_states이고 k_proj 폭은 `num_key_value_heads * head_dim`이다. decode 출력의 축 1은 n_h가 아니라 n_kv다. |
| `self_attn$` | `n_h` | `n_kv` | 96 | modeling_zamba2.py:255,300-314에서 두 번째 transpose는 key_states이고 k_proj 폭은 `num_key_value_heads * head_dim`이다. prefill 출력의 축 1은 n_h가 아니라 n_kv다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 66 | modeling_zamba2.py:657,684,758-760에서 projected_states의 마지막 split 크기는 self.num_heads이고 그 출력이 dt다. 중첩 mixer의 shape_index 4 마지막 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 128 | modeling_zamba2.py:657,697,731,807에서 A는 self.num_heads 길이이고 recurrent update를 위해 두 singleton 축을 붙인다. nth 2 unsqueeze의 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:555,573-575에서 dt와 A의 곱은 `[B,T,num_heads]`이고 hidden_states와 곱하기 위해 마지막 singleton을 붙인다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-585에서 A와 segment_sum 입력은 `[B,num_heads,n_chunks,chunk_size]`이고 expand 전에 마지막 singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:587-588의 broadcast 곱은 합산 전 `(b,c,l,s,h,n)`, 즉 `[B,n_chunks,d_chunk,d_chunk,num_heads,state_size]`다. 축 4는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:587-588에서 `(b,c,l,s,h,n)`의 마지막 state 축 n을 합친 G는 `[B,n_chunks,l,s,num_heads]`다. sum 출력의 축 4는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:587-594에서 M은 `[B,c,l,s,num_heads]`이고 `M[...,None]`이 마지막 singleton을 붙인다. 축 4는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:591-594에서 M과 hidden_states의 broadcast 곱은 `[B,c,l,s,num_heads,head_dim]`이다. 축 4는 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:591-594의 같은 곱에서 마지막 축은 hidden_states의 head_dim이다. 앞 교정 이후 앵커 shape에서 축 5는 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-581,598에서 A_cumsum은 `[B,num_heads,n_chunks,chunk_size]`이고 `A_cumsum[:,:,:,-1:]`은 마지막 chunk 위치만 자른다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:598-600에서 decay_states는 `[B,num_heads,c,l]`이고 `permute(0,-2,-1,1)` 출력은 `[B,c,l,num_heads]`다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-585,591에서 L은 `[B,num_heads,c,l,l]`이고 `L.permute(0,2,3,4,1)`은 `[B,c,l,l,num_heads]`다. 축 4는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:598-600에서 permuted decay_states는 `[B,c,l,num_heads]`이고 B와 곱하기 위해 마지막 singleton을 붙인다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:599-600의 B_decay는 `[B,c,l,num_heads,state_size]`이고 broadcast 정렬용 내부 permute는 `[B,c,num_heads,l,state_size]`다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:599-600의 B_decay broadcast 텐서는 내부 정렬 후 `[B,c,num_heads,l,state_size,1]`이다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:555,578,600에서 hidden_states는 `[B,c,l,num_heads,head_dim]`이고 broadcast 정렬용 내부 permute는 `[B,c,num_heads,l,head_dim]`다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:555,578,600의 hidden_states broadcast 텐서는 내부 정렬 후 `[B,c,num_heads,l,1,head_dim]`이다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:590-591에서 M은 G와 L의 곱에서 마지막 singleton만 합친 `[B,c,l,s,num_heads]`다. sum 출력의 축 4는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:599-600의 B_decay와 hidden_states 곱은 내부 정렬에서 `[B,c,num_heads,l,state_size,head_dim]`이다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:599-600의 같은 곱에서 마지막 축은 hidden_states의 head_dim이다. 앞 교정 이후 앵커 shape에서 축 5는 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:599-600에서 chunk 위치를 합친 내부 state는 `[B,c,num_heads,state_size,head_dim]`이고 다음 permute가 `[B,c,num_heads,head_dim,state_size]`로 만든다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:599-600의 같은 내부 state에서 마지막 축은 head_dim이다. 앞 교정 이후 앵커 shape에서 축 4는 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-581,610에서 A_cumsum은 `[B,num_heads,c,l]`이고 `A_cumsum[:,:,:,-1]`은 마지막 chunk 위치만 select한다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-581,610에서 `[B,num_heads,n_chunks]`인 `A_cumsum[:,:,:,-1]`을 chunk 경계 축만 pad한다. `[B,num_heads,2]`의 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:610과 :377-384에서 segment_sum은 `[B,num_heads,2]`에 마지막 singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:600,609-611에서 states를 decay와 곱하기 위한 내부 순서는 `[B,num_heads,n_chunks+1,head_dim,state_size]`다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:610-611의 decay_chunk는 transpose 전 `[B,num_heads,2,2]`이고 state 축과 곱하기 위해 singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:610-611의 같은 decay_chunk broadcast가 singleton을 하나 더 붙여도 축 1은 계속 num_heads, 즉 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:600,609-611의 states broadcast는 내부적으로 `[B,num_heads,1,n_chunks+1,head_dim,state_size]`다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:609-611의 decay_chunk와 states 곱은 `[B,num_heads,c+1,c+1,head_dim,state_size]`다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:609-611의 같은 recurrence 곱에서 축 4는 state의 head_dim이다. 앞 교정 이후 앵커 shape에서 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:611의 sum(dim=1)은 이전 chunk 축만 합치며 내부 출력은 `[B,num_heads,c+1,head_dim,state_size]`다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:611의 같은 recurrence sum 출력에서 축 3은 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:555-566,578,617에서 C는 `[B,c,l,num_heads,state_size]`이고 `C[...,None,:]`이 head_dim 자리에 singleton을 붙인다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:617의 C와 states broadcast 곱은 `[B,c,l,num_heads,head_dim,state_size]`다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:617의 C와 states 곱은 `[B,c,l,num_heads,head_dim,state_size]`다. 앞 교정 이후 앵커 shape에서 축 4는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:580-581,616-618에서 state_decay_out은 `[B,num_heads,c,l]`이고 `permute(0,2,3,1)`은 `[B,c,l,num_heads]`다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:758-760,803-806에서 dt는 `[B,1,num_heads]`이고 transpose 후 `[B,num_heads,1]`이다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:657,693,701,810-811에서 D와 dt_bias는 self.num_heads 길이이고 `[num_heads,head_dim]`으로 expand된다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:810-811의 같은 `[num_heads,head_dim]` expand에서 축 1은 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:657,697,731,807에서 A는 self.num_heads 길이이고 `A[:,None,None]`의 첫 unsqueeze도 축 0을 보존한다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:807의 `A[:,None,None]`은 두 singleton을 붙여도 첫 축이 self.num_heads다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:807은 A를 `[self.num_heads,self.head_dim,self.ssm_state_size]`로 expand한다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:807의 A expand 출력은 `[num_heads,head_dim,state_size]`다. 앞 교정 이후 앵커 shape에서 축 1은 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:491,499,805-806에서 dt는 `[B,num_heads,head_dim]`이고 selective update가 마지막 state singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:491,499,805-806의 같은 dt 텐서에서 축 2는 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:518-520에서 C는 repeat 후 `[B,num_heads,state_size]`이고 batched matmul용 singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:524-525는 C를 `[batch_size*num_heads,state_size,1]`로 view한다. B=1인 펼친 앵커의 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:524-527의 bmm 출력은 `[batch_size*num_heads,head_dim,1]`이다. B=1인 앵커의 축 0은 n_h_ssm이고 축 1만 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:491,505-511에서 hidden_states는 `[B,num_heads,head_dim]`이고 state_size와 곱하기 위해 마지막 singleton을 붙인다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:491,505-511의 같은 hidden_states에서 축 2는 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:491,517-531에서 recurrent output과 hidden_states는 `[B,num_heads,head_dim]` 순서이고 마지막 singleton을 붙이는 내부 텐서도 이를 보존한다. 축 1은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:491,517-531의 같은 recurrent output 텐서에서 축 2는 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 60 | modeling_zamba2.py:580-585에서 A와 segment_sum 출력의 prefix는 `[B,num_heads,n_chunks]`다. 같은 클래스를 쓰는 중첩 mixer의 축 1도 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 60 | modeling_zamba2.py:580-581,610과 :377-394에서 inter-chunk segment_sum의 prefix는 `[B,num_heads]`다. 중첩 mixer `[B,num_heads,2,2]`의 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 54 | modeling_zamba2.py:555-566에서 B/C는 `[B,T,num_heads,state_size]`로 repeat되고 :568,578은 sequence 축만 pad한다. 중첩 mixer nth 4의 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 54 | modeling_zamba2.py:805-807은 dt를 `[B,num_heads,head_dim]`으로 expand한다. 중첩 mixer decode 앵커의 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 54 | modeling_zamba2.py:805-807의 같은 dt expand에서 축 2는 head_dim이다. 앞 교정 이후 중첩 mixer 앵커 shape에서 d_head_ssm이다. |
| `self_attn$` | `n_h` | `n_kv` | 54 | modeling_zamba2.py:256,300-314에서 세 번째 transpose는 value_states이고 v_proj 폭은 `num_key_value_heads * head_dim`이다. decode 출력의 축 1은 n_kv다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 54 | modeling_zamba2.py:803-806의 recurrent hidden_states view는 `[B,self.num_heads,self.head_dim]`이다. 중첩 mixer 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 54 | modeling_zamba2.py:803-806의 같은 view에서 축 2는 head_dim이다. 앞 교정 이후 중첩 mixer 앵커 shape에서 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 42 | modeling_zamba2.py:555,568-578에서 hidden_states padding 출력은 `[B,d_chunk,num_heads,head_dim]`이다. 중첩 mixer nth 2의 축 2는 n_h_ssm이다. |
| `self_attn$` | `n_h` | `n_kv` | 48 | modeling_zamba2.py:256,300-314에서 세 번째 transpose는 value_states이고 v_proj 폭은 `num_key_value_heads * head_dim`이다. prefill 출력의 축 1은 n_kv다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 42 | modeling_zamba2.py:555,568-578의 hidden_states padding 출력은 `[B,d_chunk,num_heads,head_dim]`이다. 앞 교정 이후 중첩 mixer 앵커에서 축 3은 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 42 | modeling_zamba2.py:555-566에서 C는 `[B,T,num_heads,state_size]`로 repeat되고 sequence 축만 pad된다. 중첩 mixer nth 5의 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:555-566의 B repeat_interleave 분해 expand는 `[B,T,1,num_heads,state_size]`다. 중첩 mixer 축 3은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:587-591에서 G/M의 순서는 `[B,c,l,s,num_heads]`이고 마지막 singleton을 붙인다. 중첩 mixer 축 4는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:600,609-612의 final_state는 `[B,num_heads,head_dim,state_size]`다. 중첩 mixer 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 36 | modeling_zamba2.py:600,609-612의 같은 final_state에서 축 2는 head_dim이다. 앞 교정 이후 중첩 mixer 앵커 shape에서 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:555-566의 C repeat_interleave 분해 expand는 `[B,T,1,num_heads,state_size]`다. 중첩 mixer 축 3은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 36 | modeling_zamba2.py:491-507에서 B는 `[B,num_groups,num_heads//num_groups,state_size]`로 expand된다. n_groups=1인 중첩 mixer 앵커의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 32 | modeling_zamba2.py:657,693,495-498에서 dt_bias는 self.num_heads 길이이고 dt에 더해진다. elementwise_add 둘째 입력의 축은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 60 | modeling_zamba2.py:657,697,731에서 A_log와 그 exp 입력은 self.num_heads 길이다. 중첩 mixer 축 0은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 30 | modeling_zamba2.py:831-836의 hidden_states view는 `[B,T,self.num_heads,self.head_dim]`이다. 중첩 mixer 축 2는 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 30 | modeling_zamba2.py:831-836의 같은 view에서 축 3은 head_dim이다. 앞 교정 이후 중첩 mixer 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:657,701,810에서 D는 self.num_heads 길이이고 `D[:,None]`으로 singleton을 붙인다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:701,810에서 D는 `[num_heads,head_dim]`으로 expand된다. 축 0은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:701,810의 같은 D expand에서 축 1은 head_dim이다. 앞 교정 이후 앵커 shape에서 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:616-618의 permuted state_decay_out은 `[B,c,l,num_heads]`이고 Y_off와 곱하기 위해 마지막 singleton을 붙인다. 축 3은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:621-629에서 output은 `[batch_size,padded_sequence,num_heads,head_dim]`으로 reshape된 뒤 sequence 축만 자른다. 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 64 | modeling_zamba2.py:621-629의 같은 output slice에서 마지막 축은 head_dim이다. 앞 교정 이후 앵커 shape에서 축 3은 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 64 | modeling_zamba2.py:657,684,758-760에서 projected_states의 마지막 split 크기는 self.num_heads이고 그 출력이 dt다. decode shape_index 4의 마지막 축은 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 704 | modeling_zamba2.py:491-511은 hidden_states를 `[batch_size,num_heads,head_dim]`으로 해체하고 `hidden_states[...,None]`을 state_size 축과 곱한다. 따라서 `[B,num_heads,head_dim,state_size]`의 축 2는 n_h_ssm이 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 640 | modeling_zamba2.py:491-511의 `batch_size, num_heads, head_dim = hidden_states.shape`과 `dB * hidden_states[..., None]`이 출력 순서를 `[B,num_heads,head_dim,state_size]`로 강제한다. 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 512 | modeling_zamba2.py:555-578은 hidden_states 입력을 `[batch_size,sequence_length,num_heads,head_dim]`으로 선언하고 :357-373이 sequence 축만 pad/chunk한다. 따라서 `[B,d_chunk,num_heads,head_dim]`의 축 2는 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `n_h_ssm` | `d_head_ssm` | 512 | modeling_zamba2.py:555-578,357-373에서 hidden_states는 `[B,sequence_length,num_heads,head_dim] -> [B,n_chunks,chunk_size,num_heads,head_dim]`이다. padding 출력의 마지막 축은 head 개수가 아니라 d_head_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 448 | modeling_zamba2.py:555-580에서 A는 `[B,T,num_heads]`에서 chunk된 뒤 `A.permute(0,3,1,2)`로 `[B,num_heads,n_chunks,chunk_size]`가 된다. 축 1은 d_head_ssm이 아니라 n_h_ssm이다. |
| `^model\.layers\.\*\.mamba$` | `d_head_ssm` | `n_h_ssm` | 352 | modeling_zamba2.py:657,684,758-759에서 projected_states의 마지막 split 크기는 `self.num_heads` 이고 그 출력이 dt다. 따라서 shape_index 4의 마지막 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:555,580-581에서 A는 num_heads 축을 앞으로 옮긴다. 이 경로의 단일 폭은 head_dim이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:555,580-585에서 segment 합의 A 순서는 `[B,num_heads,n_chunks,chunk]`다. singleton을 붙인 출력의 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:587-591에서 G와 M은 state 축을 합친 뒤 `(b,c,l,s,h)` 순서다. 마지막 축은 d_head_ssm이 아니라 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:587-594의 두 번째 G/M broadcast 경로도 `(b,c,l,s,h)`를 유지한다. 마지막 축은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:580-581,609-611에서 decay는 A의 num_heads 축을 보존한다. chunk 경계 행렬에 singleton을 붙인 출력 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:614-618에서 C와 states의 순서는 chunk 뒤 num_heads, state_size다. broadcast singleton 출력의 축 3은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:657,693,697,731에서 A_log와 A는 self.num_heads 길이다. decode의 두 번째 head 벡터 unsqueeze 출력 축 0은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:491-511에서 recurrent hidden_states는 `[B,num_heads,head_dim]`이다. 마지막 singleton을 붙여도 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 12 | modeling_zamba2.py:491-511의 같은 `[B,num_heads,head_dim,1]` 출력에서 앞 교정 뒤 축 2는 head 개수가 아니라 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:491-511에서 recurrent state 곱의 hidden_states 순서는 `[B,num_heads,head_dim]`이다. 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 12 | modeling_zamba2.py:491-511의 같은 recurrent broadcast에서 앞 교정 뒤 축 2는 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:505-508에서 B는 `[B,num_heads,1,state_size]`로 reshape된다. 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:491-514에서 state update의 hidden_states 순서는 `[B,num_heads,head_dim]`이다. 축 1은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `n_h_ssm` | `d_head_ssm` | 12 | modeling_zamba2.py:491-514의 같은 state update broadcast에서 앞 교정 뒤 축 2는 d_head_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:518-526에서 C는 num_heads로 펼쳐지고 state_size와 곱한다. bmm용 `[num_heads,state_size,1]` view의 축 0은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:521-526에서 bmm 출력은 `[num_heads,head_dim,1]`이고 이를 `[B,num_heads,head_dim]`으로 되돌린다. 입력 축 0은 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:580-581,609-611에서 recurrence decay는 A의 num_heads 축을 보존한다. 두 번째 singleton 출력의 축 1도 n_h_ssm이다. |
| `mamba_decoder\.mamba$` | `d_head_ssm` | `n_h_ssm` | 12 | modeling_zamba2.py:803-811에서 A, D, dt_bias는 모두 self.num_heads에서 시작해 singleton을 붙인다. `[num_heads,1,1]` 출력의 축 0은 n_h_ssm이다. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mamba` | num_heads vs head_dim (둘 다 64) | `d_head_ssm / n_h_ssm (순서 뒤바뀜)` | `n_h_ssm 이 앞, d_head_ssm 이 뒤` | `modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * nu … |
| `model.layers.*.mamba_decoder.mamba` | num_heads vs head_dim (둘 다 64) | `d_head_ssm / n_h_ssm (순서 뒤바뀜)` | `n_h_ssm 이 앞, d_head_ssm 이 뒤` | `modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * nu … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- Zyphra/Zamba2-1.2B @ 6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe

C1   PASS   38 == 38
C2   PASS   2 clusters == 2 from config schedule ['layers_block_type']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 38/38 layers
C6   PASS   hidden_size=2048 (heuristic check, 0 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=32000, tie_word_embeddings=True
C10  PASS   all 406 params covered
C11  PASS   69 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4468 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.masked_fill.Scalar']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              clone            [B,T,d_model] -> [B,T,d_model]
  model.rotary_emb                                   unsqueeze        [d_head/2] -> [B,d_head/2]
  model.rotary_emb                                   unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  model.rotary_emb                                   expand           [B,d_head/2,1] -> [B,d_head/2,1]
  model.rotary_emb                                   unsqueeze        [B,T] -> [B,1,T]
  model.rotary_emb                                   _to_copy         [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,d_head/2,1] -> [B,d_head/2,1]
  model.rotary_emb                                   expand           [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,1,T] -> [B,1,T]
  model.rotary_emb                                   batched_matmul   [B,d_head/2,1]*[B,1,T] -> [B,d_head/2,T]
  model.rotary_emb                                   _unsafe_view     [B,d_head/2,T] -> [B,d_head/2,T]
  model.rotary_emb                                   transpose        [B,d_head/2,T] -> [B,T,d_head/2]
  model.rotary_emb                                   concat           [B,T,d_head/2]*[B,T,d_head/2] -> [B,T,d_head]
  model.rotary_emb                                   cos              [B,T,d_head] -> [B,T,d_head]
  model.rotary_emb                                   elementwise_mul  [B,T,d_head] -> [B,T,d_head]
  model.rotary_emb                                   sin              [B,T,d_head] -> [B,T,d_head]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mamba.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mamba.in_proj                       matmul           [T,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       _unsafe_view     [T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               split_with_sizes [B,T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,0]*[B,T,0]*[B,T,d_inner]*[B,T,d_inner+2*n_g*d_state]*[B,T,n_h_ssm]
  model.layers.N.mamba                               transpose        [B,T,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba                               constant_pad_nd  [B,d_inner+2*n_g*d_state,T] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               zeros            [] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba.conv1d                        conv1d           [B,d_inner+2*n_g*d_state,T]*[d_inner+2*n_g*d_state,1,d_conv]*[d_inner+2*n_g*d_state] -> w=[d_inner+2*n_g*d_state,1,d_conv] [B,d_inner+2*n_g*d_state,T+d_conv-1]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,T+d_conv-1] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba                               transpose        [B,d_inner+2*n_g*d_state,T] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mamba.act                           silu             [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               split_with_sizes [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner]*[B,T,d_state]*[B,T,d_state]
  model.layers.N.mamba                               exp              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               neg              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               elementwise_add  [B,T,n_h_ssm]*[n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               softplus         [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               clamp            [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               view             [B,T,d_inner] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               view             [B,T,d_state] -> [B,T,1,d_state]
  model.layers.N.mamba                               unsqueeze        [B,T,1,d_state] -> [B,T,1,1,d_state]
  model.layers.N.mamba                               expand           [B,T,1,1,d_state] -> [B,T,1,n_h_ssm,d_state]
  model.layers.N.mamba                               clone            [B,T,1,n_h_ssm,d_state] -> [B,T,1,n_h_ssm,d_state]
  model.layers.N.mamba                               view             [B,T,1,n_h_ssm,d_state] -> [B,T,n_h_ssm,d_state]
  model.layers.N.mamba                               unsqueeze        [n_h_ssm] -> [n_h_ssm,B]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [n_h_ssm,B]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               unsqueeze        [B,T,n_h_ssm] -> [B,T,n_h_ssm,1]
  model.layers.N.mamba                               elementwise_mul  [B,T,n_h_ssm,d_head_ssm]*[B,T,n_h_ssm,1] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [n_h_ssm]*[B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               view             [B,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm] -> [B,d_chunk,n_h_ssm]
  model.layers.N.mamba                               view             [B,d_chunk,n_h_ssm] -> [B,1,d_chunk,n_h_ssm]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm,d_state] -> [B,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               view             [B,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               permute          [B,1,d_chunk,n_h_ssm] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               unsqueeze        [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk,1]
  model.layers.N.mamba                               expand           [B,n_h_ssm,1,d_chunk,1] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               ones             [] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               tril             [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               bitwise_not      [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               masked_fill      [B,n_h_ssm,1,d_chunk,d_chunk]*[d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               exp              [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               unsqueeze        [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,1,n_h_ssm,d_state]
  model.layers.N.mamba                               unsqueeze        [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               elementwise_mul  [B,1,d_chunk,1,n_h_ssm,d_state]*[B,1,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               sum              [B,1,d_chunk,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba                               permute          [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,1,d_chunk,d_chunk,n_h_ssm,1]*[B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,n_h_ssm,1]
  model.layers.N.mamba                               sum              [B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,1,d_chunk,d_chunk,n_h_ssm,1]*[B,1,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               sum              [B,1,d_chunk,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               slice            [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,1]
  model.layers.N.mamba                               sub              [B,n_h_ssm,1,1]*[B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               exp              [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               permute          [B,n_h_ssm,1,d_chunk] -> [B,1,d_chunk,n_h_ssm]
  model.layers.N.mamba                               permute          [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,n_h_ssm,d_chunk,d_state]
  model.layers.N.mamba                               permute          [B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,n_h_ssm,d_chunk,d_head_ssm]
  model.layers.N.mamba                               sum              [B,1,n_h_ssm,d_chunk,d_state,d_head_ssm] -> [B,1,n_h_ssm,d_state,d_head_ssm]
  model.layers.N.mamba                               permute          [B,1,n_h_ssm,d_state,d_head_ssm] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               alias            [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               zeros_like       [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               concat           [B,1,n_h_ssm,d_head_ssm,d_state]*[B,1,n_h_ssm,d_head_ssm,d_state] -> [B,2,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               select           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1]
  model.layers.N.mamba                               constant_pad_nd  [B,n_h_ssm,1] -> [B,n_h_ssm,2]
  model.layers.N.mamba                               expand           [B,n_h_ssm,2,1] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba                               ones             [] -> [2,2]
  model.layers.N.mamba                               tril             [2,2] -> [2,2]
  model.layers.N.mamba                               bitwise_not      [2,2] -> [2,2]
  model.layers.N.mamba                               masked_fill      [B,n_h_ssm,2,2]*[2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba                               exp              [B,n_h_ssm,2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba                               sum              [B,n_h_ssm,2,2,d_head_ssm,d_state] -> [B,n_h_ssm,2,d_head_ssm,d_state]
  model.layers.N.mamba                               slice            [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               select           [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               sum              [B,1,d_chunk,n_h_ssm,d_head_ssm,d_state] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,1,d_chunk,n_h_ssm,d_head_ssm]*[B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,d_chunk,n_h_ssm,d_head_ssm]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               slice            [B,d_chunk,n_h_ssm,d_head_ssm] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               zeros_like       [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba.norm                          silu             [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,T,d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          view             [B,T,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          pow              [B,T,1,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          mean             [B,T,1,d_inner] -> [B,T,1,1]
  model.layers.N.mamba.norm                          elementwise_add  [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba.norm                          rsqrt            [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba.norm                          elementwise_mul  [B,T,1,d_inner]*[B,T,1,1] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,T,1,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.out_proj                      t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mamba.out_proj                      view             [B,T,d_inner] -> [T,d_inner]
  model.layers.N.mamba.out_proj                      matmul           [T,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [T,d_model]
  model.layers.N.mamba.out_proj                      _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.shared_transformer                  concat           [B,T,d_model]*[B,T,d_model] -> [B,T,d_attn]
  model.layers.N.shared_transformer.input_layernorm  pow              [B,T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.input_layernorm  mean             [B,T,d_attn] -> [B,T,1]
  model.layers.N.shared_transformer.input_layernorm  elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.shared_transformer.input_layernorm  rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.shared_transformer.input_layernorm  elementwise_mul  [B,T,d_attn]*[B,T,1] -> [B,T,d_attn]
  model.layers.N.shared_transformer.input_layernorm  elementwise_mul  [d_attn]*[B,T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj matmul           [T,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj matmul           [T,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj matmul           [T,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 matmul           [T,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 _unsafe_view     [T,r_lora] -> [B,T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 view             [B,T,r_lora] -> [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 matmul           [T,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,T,d_attn]*[B,T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 matmul           [T,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 _unsafe_view     [T,r_lora] -> [B,T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 view             [B,T,r_lora] -> [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 matmul           [T,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 matmul           [T,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 _unsafe_view     [T,r_lora] -> [B,T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 view             [B,T,r_lora] -> [T,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 matmul           [T,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [T,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 _unsafe_view     [T,d_attn] -> [B,T,d_attn]
  model.layers.N.shared_transformer.self_attn        view             [B,T,d_attn] -> [B,T,n_h,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,T,d_attn] -> [B,T,n_kv,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,T,n_h,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_kv,T,d_head]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        ones             [] -> [T,T]
  model.layers.N.shared_transformer.self_attn        tril             [T,T] -> [T,T]
  model.layers.N.shared_transformer.self_attn        scalar_tensor    [] -> []
  model.layers.N.shared_transformer.self_attn        where            [T,T]*[]*[] -> [T,T]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_kv,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,T,T]*[T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_kv,T,d_head] -> [n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,T,T]*[n_kv,T,d_head] -> [n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.shared_transformer.self_attn        clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.shared_transformer.self_attn.o_proj t                [d_model,d_attn] -> w=[d_model,d_attn] [d_attn,d_model]
  model.layers.N.shared_transformer.self_attn.o_proj view             [B,T,d_attn] -> [T,d_attn]
  model.layers.N.shared_transformer.self_attn.o_proj matmul           [T,d_attn]*[d_attn,d_model] -> w=[d_model,d_attn] [T,d_model]
  model.layers.N.shared_transformer.self_attn.o_proj _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.shared_transformer.pre_ff_layernorm rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj t                [2*d_ff,d_model] -> w=[2*d_ff,d_model] [d_model,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj view             [B,T,d_model] -> [T,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj matmul           [T,d_model]*[d_model,2*d_ff] -> w=[2*d_ff,d_model] [T,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj _unsafe_view     [T,2*d_ff] -> [B,T,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 t                [r_lora,d_model] -> w=[r_lora,d_model] [d_model,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 matmul           [T,d_model]*[d_model,r_lora] -> w=[r_lora,d_model] [T,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 _unsafe_view     [T,r_lora] -> [B,T,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 t                [2*d_ff,r_lora] -> w=[2*d_ff,r_lora] [r_lora,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 view             [B,T,r_lora] -> [T,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 matmul           [T,r_lora]*[r_lora,2*d_ff] -> w=[2*d_ff,r_lora] [T,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 _unsafe_view     [T,2*d_ff] -> [B,T,2*d_ff]
  model.layers.N.shared_transformer.feed_forward     elementwise_add  [B,T,2*d_ff]*[B,T,2*d_ff] -> [B,T,2*d_ff]
  model.layers.N.shared_transformer.feed_forward     split            [B,T,2*d_ff] -> [B,T,d_ff]*[B,T,d_ff]
  model.layers.N.shared_transformer.feed_forward.act_fn gelu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.shared_transformer.feed_forward     elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.shared_transformer.feed_forward.down_proj t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.shared_transformer.feed_forward.down_proj view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.shared_transformer.feed_forward.down_proj matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.shared_transformer.feed_forward.down_proj _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.linear                              t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.linear                              view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear                              matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  model.layers.N.linear                              _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.mamba_decoder                       elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mamba_decoder.input_layernorm       pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.mamba_decoder.input_layernorm       mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.mamba_decoder.input_layernorm       rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mamba_decoder.mamba.in_proj         t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba.in_proj         view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mamba_decoder.mamba.in_proj         matmul           [T,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba.in_proj         _unsafe_view     [T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 split_with_sizes [B,T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,0]*[B,T,0]*[B,T,d_inner]*[B,T,d_inner+2*n_g*d_state]*[B,T,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 transpose        [B,T,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,d_inner+2*n_g*d_state,T] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 zeros            [] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 slice            [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba.conv1d          conv1d           [B,d_inner+2*n_g*d_state,T]*[d_inner+2*n_g*d_state,1,d_conv]*[d_inner+2*n_g*d_state] -> w=[d_inner+2*n_g*d_state,1,d_conv] [B,d_inner+2*n_g*d_state,T+d_conv-1]
  model.layers.N.mamba_decoder.mamba                 slice            [B,d_inner+2*n_g*d_state,T+d_conv-1] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba_decoder.mamba                 transpose        [B,d_inner+2*n_g*d_state,T] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba.act             silu             [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba                 split_with_sizes [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner]*[B,T,d_state]*[B,T,d_state]
  model.layers.N.mamba_decoder.mamba                 exp              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 neg              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,T,n_h_ssm]*[d_head_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 softplus         [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 clamp            [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 view             [B,T,d_inner] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 view             [B,T,d_state] -> [B,T,1,d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,T,1,d_state] -> [B,T,1,1,d_state]
  model.layers.N.mamba_decoder.mamba                 expand           [B,T,1,1,d_state] -> [B,T,1,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 clone            [B,T,1,n_h_ssm,d_state] -> [B,T,1,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,T,1,n_h_ssm,d_state] -> [B,T,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm] -> [n_h_ssm,B]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,T,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [n_h_ssm,B]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,T,n_h_ssm] -> [B,T,n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,T,n_h_ssm,d_head_ssm]*[B,T,n_h_ssm,1] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [n_h_ssm]*[B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 view             [B,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,T,n_h_ssm] -> [B,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 view             [B,d_chunk,n_h_ssm] -> [B,1,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,T,n_h_ssm,d_state] -> [B,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_chunk,n_h_ssm] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 cumsum           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk,1]
  model.layers.N.mamba_decoder.mamba                 expand           [B,n_h_ssm,1,d_chunk,1] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 ones             [] -> [d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 tril             [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 bitwise_not      [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 masked_fill      [B,n_h_ssm,1,d_chunk,d_chunk]*[d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 cumsum           [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 exp              [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,1,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,1,n_h_ssm,d_state]*[B,1,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,d_chunk,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,d_head_ssm,d_state] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,d_chunk,n_h_ssm,1]*[B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,d_chunk,d_head_ssm,1]*[B,1,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,d_chunk,d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,d_head_ssm,n_h_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 slice            [B,n_h_ssm,1,d_chunk] -> [B,d_head_ssm,1,1]
  model.layers.N.mamba_decoder.mamba                 sub              [B,d_head_ssm,1,1]*[B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 exp              [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 permute          [B,n_h_ssm,1,d_chunk] -> [B,1,d_chunk,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_head_ssm,d_chunk,d_state]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_head_ssm,d_chunk,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_head_ssm,d_chunk,d_state,n_h_ssm] -> [B,1,d_head_ssm,d_state,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_head_ssm,d_state,n_h_ssm] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 alias            [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 zeros_like       [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 concat           [B,1,n_h_ssm,d_head_ssm,d_state]*[B,1,n_h_ssm,d_head_ssm,d_state] -> [B,2,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 select           [B,n_h_ssm,1,d_chunk] -> [B,d_head_ssm,1]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,d_head_ssm,1] -> [B,d_head_ssm,2]
  model.layers.N.mamba_decoder.mamba                 expand           [B,d_head_ssm,2,1] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba_decoder.mamba                 ones             [] -> [2,2]
  model.layers.N.mamba_decoder.mamba                 tril             [2,2] -> [2,2]
  model.layers.N.mamba_decoder.mamba                 bitwise_not      [2,2] -> [2,2]
  model.layers.N.mamba_decoder.mamba                 masked_fill      [B,n_h_ssm,2,2]*[2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba_decoder.mamba                 cumsum           [B,n_h_ssm,2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba_decoder.mamba                 exp              [B,n_h_ssm,2,2] -> [B,n_h_ssm,2,2]
  model.layers.N.mamba_decoder.mamba                 sum              [B,d_head_ssm,2,2,n_h_ssm,d_state] -> [B,d_head_ssm,2,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 slice            [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 select           [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_head_ssm,n_h_ssm,d_state] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,1,d_chunk,n_h_ssm,d_head_ssm]*[B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,d_chunk,n_h_ssm,d_head_ssm]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 slice            [B,d_chunk,n_h_ssm,d_head_ssm] -> [B,T,d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 zeros_like       [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba.norm            silu             [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [B,T,d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            view             [B,T,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            pow              [B,T,1,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            mean             [B,T,1,d_inner] -> [B,T,1,1]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_add  [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba_decoder.mamba.norm            rsqrt            [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [B,T,1,d_inner]*[B,T,1,1] -> [B,T,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            view             [B,T,1,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba_decoder.mamba.out_proj        t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mamba_decoder.mamba.out_proj        view             [B,T,d_inner] -> [T,d_inner]
  model.layers.N.mamba_decoder.mamba.out_proj        matmul           [T,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [T,d_model]
  model.layers.N.mamba_decoder.mamba.out_proj        _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.final_layernorm                              pow              [B,T,d_model] -> [B,T,d_model]
  model.final_layernorm                              mean             [B,T,d_model] -> [B,T,1]
  model.final_layernorm                              elementwise_add  [B,T,1] -> [B,T,1]
  model.final_layernorm                              rsqrt            [B,T,1] -> [B,T,1]
  model.final_layernorm                              elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.final_layernorm                              elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model                                              clone            [B,1,d_model] -> [B,1,d_model]
  model.rotary_emb                                   unsqueeze        [d_head/2] -> [B,d_head/2]
  model.rotary_emb                                   unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  model.rotary_emb                                   expand           [B,d_head/2,1] -> [B,d_head/2,1]
  model.rotary_emb                                   unsqueeze        [B,1] -> [B,1,1]
  model.rotary_emb                                   _to_copy         [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,d_head/2,1] -> [B,d_head/2,1]
  model.rotary_emb                                   expand           [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,1,1] -> [B,1,1]
  model.rotary_emb                                   batched_matmul   [B,d_head/2,1]*[B,1,1] -> [B,d_head/2,1]
  model.rotary_emb                                   _unsafe_view     [B,d_head/2,1] -> [B,d_head/2,1]
  model.rotary_emb                                   transpose        [B,d_head/2,1] -> [B,1,d_head/2]
  model.rotary_emb                                   concat           [B,1,d_head/2]*[B,1,d_head/2] -> [B,1,d_head]
  model.rotary_emb                                   cos              [B,1,d_head] -> [B,1,d_head]
  model.rotary_emb                                   elementwise_mul  [B,1,d_head] -> [B,1,d_head]
  model.rotary_emb                                   sin              [B,1,d_head] -> [B,1,d_head]
  model.layers.N.input_layernorm                     pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mamba.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mamba.in_proj                       matmul           [B,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [B,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       _unsafe_view     [B,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               split_with_sizes [B,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,0]*[B,1,0]*[B,1,d_inner]*[B,1,d_inner+2*n_g*d_state]*[B,1,n_h_ssm]
  model.layers.N.mamba                               transpose        [B,1,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,1]
  model.layers.N.mamba                               concat           [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,1] -> [B,d_inner+2*n_g*d_state,d_conv+1]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,d_conv+1] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               select           [d_inner+2*n_g*d_state,1,d_conv] -> w=[d_inner+2*n_g*d_state,1,d_conv] [d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               elementwise_mul  [B,d_inner+2*n_g*d_state,d_conv]*[d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               sum              [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               add_             [B,d_inner+2*n_g*d_state]*[d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba.act                           silu             [B,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               unsqueeze        [B,d_inner+2*n_g*d_state] -> [B,1,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               split_with_sizes [B,1,d_inner+2*n_g*d_state] -> [B,1,d_inner]*[B,1,d_state]*[B,1,d_state]
  model.layers.N.mamba                               exp              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               neg              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               select           [B,1,n_h_ssm] -> [B,n_h_ssm]
  model.layers.N.mamba                               unsqueeze        [B,n_h_ssm] -> [B,1,n_h_ssm]
  model.layers.N.mamba                               transpose        [B,1,n_h_ssm] -> [B,n_h_ssm,1]
  model.layers.N.mamba                               expand           [B,n_h_ssm,1] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               unsqueeze        [n_h_ssm] -> [n_h_ssm,B]
  model.layers.N.mamba                               expand           [n_h_ssm,B] -> [n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm]*[n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               softplus         [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               clamp            [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               unsqueeze        [n_h_ssm,B] -> [n_h_ssm,B,1]
  model.layers.N.mamba                               expand           [n_h_ssm,B,1] -> [n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               unsqueeze        [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm,1]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,1]*[n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               exp              [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,1,d_state] -> [B,1,d_state]
  model.layers.N.mamba                               unsqueeze        [B,1,d_state] -> [B,1,1,d_state]
  model.layers.N.mamba                               expand           [B,1,1,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba                               clone            [B,1,n_h_ssm,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba                               view             [B,1,n_h_ssm,d_state] -> [B,n_h_ssm,d_state]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,1]*[B,n_h_ssm,1,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,1,d_inner] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,1] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               clone            [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,n_h_ssm,d_head_ssm,d_state] -> [n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,n_h_ssm,d_state] -> [n_h_ssm,d_state,B]
  model.layers.N.mamba                               batched_matmul   [n_h_ssm,d_head_ssm,d_state]*[n_h_ssm,d_state,B] -> [n_h_ssm,d_head_ssm,B]
  model.layers.N.mamba                               view             [n_h_ssm,d_head_ssm,B] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm]*[n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm]*[B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba.norm                          silu             [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,1,d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          pow              [B,1,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          mean             [B,1,1,d_inner] -> [B,1,1,1]
  model.layers.N.mamba.norm                          elementwise_add  [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba.norm                          rsqrt            [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba.norm                          elementwise_mul  [B,1,1,d_inner]*[B,1,1,1] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,1,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.out_proj                      t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mamba.out_proj                      view             [B,1,d_inner] -> [B,d_inner]
  model.layers.N.mamba.out_proj                      matmul           [B,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [B,d_model]
  model.layers.N.mamba.out_proj                      _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.shared_transformer                  concat           [B,1,d_model]*[B,1,d_model] -> [B,1,d_attn]
  model.layers.N.shared_transformer.input_layernorm  pow              [B,1,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.input_layernorm  mean             [B,1,d_attn] -> [B,1,1]
  model.layers.N.shared_transformer.input_layernorm  elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.shared_transformer.input_layernorm  rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.shared_transformer.input_layernorm  elementwise_mul  [B,1,d_attn]*[B,1,1] -> [B,1,d_attn]
  model.layers.N.shared_transformer.input_layernorm  elementwise_mul  [d_attn]*[B,1,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj matmul           [B,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.q_proj _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj matmul           [B,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.k_proj _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj t                [d_attn,d_attn] -> w=[d_attn,d_attn] [d_attn,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj matmul           [B,d_attn]*[d_attn,d_attn] -> w=[d_attn,d_attn] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.v_proj _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 matmul           [B,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.0 _unsafe_view     [B,r_lora] -> [B,1,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 view             [B,1,r_lora] -> [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 matmul           [B,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_q_adapter_list.N.1 _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,1,d_attn]*[B,1,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 matmul           [B,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.0 _unsafe_view     [B,r_lora] -> [B,1,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 view             [B,1,r_lora] -> [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 matmul           [B,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_k_adapter_list.N.1 _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 t                [r_lora,d_attn] -> w=[r_lora,d_attn] [d_attn,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 matmul           [B,d_attn]*[d_attn,r_lora] -> w=[r_lora,d_attn] [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.0 _unsafe_view     [B,r_lora] -> [B,1,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 t                [d_attn,r_lora] -> w=[d_attn,r_lora] [r_lora,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 view             [B,1,r_lora] -> [B,r_lora]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 matmul           [B,r_lora]*[r_lora,d_attn] -> w=[d_attn,r_lora] [B,d_attn]
  model.layers.N.shared_transformer.self_attn.linear_v_adapter_list.N.1 _unsafe_view     [B,d_attn] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn        view             [B,1,d_attn] -> [B,1,n_h,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,1,n_h,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_kv,1,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_kv,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_kv,T+1,d_head] -> [n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,B,T+1]*[n_kv,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,1,n_h,d_head] -> [B,1,d_attn]
  model.layers.N.shared_transformer.self_attn.o_proj t                [d_model,d_attn] -> w=[d_model,d_attn] [d_attn,d_model]
  model.layers.N.shared_transformer.self_attn.o_proj view             [B,1,d_attn] -> [B,d_attn]
  model.layers.N.shared_transformer.self_attn.o_proj matmul           [B,d_attn]*[d_attn,d_model] -> w=[d_model,d_attn] [B,d_model]
  model.layers.N.shared_transformer.self_attn.o_proj _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.shared_transformer.pre_ff_layernorm rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.shared_transformer.pre_ff_layernorm elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj t                [2*d_ff,d_model] -> w=[2*d_ff,d_model] [d_model,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj view             [B,1,d_model] -> [B,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj matmul           [B,d_model]*[d_model,2*d_ff] -> w=[2*d_ff,d_model] [B,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj _unsafe_view     [B,2*d_ff] -> [B,1,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 t                [r_lora,d_model] -> w=[r_lora,d_model] [d_model,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 matmul           [B,d_model]*[d_model,r_lora] -> w=[r_lora,d_model] [B,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.0 _unsafe_view     [B,r_lora] -> [B,1,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 t                [2*d_ff,r_lora] -> w=[2*d_ff,r_lora] [r_lora,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 view             [B,1,r_lora] -> [B,r_lora]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 matmul           [B,r_lora]*[r_lora,2*d_ff] -> w=[2*d_ff,r_lora] [B,2*d_ff]
  model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.N.1 _unsafe_view     [B,2*d_ff] -> [B,1,2*d_ff]
  model.layers.N.shared_transformer.feed_forward     elementwise_add  [B,1,2*d_ff]*[B,1,2*d_ff] -> [B,1,2*d_ff]
  model.layers.N.shared_transformer.feed_forward     split            [B,1,2*d_ff] -> [B,1,d_ff]*[B,1,d_ff]
  model.layers.N.shared_transformer.feed_forward.act_fn gelu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.shared_transformer.feed_forward     elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.shared_transformer.feed_forward.down_proj t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.shared_transformer.feed_forward.down_proj view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.shared_transformer.feed_forward.down_proj matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.shared_transformer.feed_forward.down_proj _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.linear                              t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  model.layers.N.linear                              view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear                              matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  model.layers.N.linear                              _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.mamba_decoder                       elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mamba_decoder.input_layernorm       pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.mamba_decoder.input_layernorm       mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.mamba_decoder.input_layernorm       rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.mamba_decoder.input_layernorm       elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mamba_decoder.mamba.in_proj         t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba.in_proj         view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mamba_decoder.mamba.in_proj         matmul           [B,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [B,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba.in_proj         _unsafe_view     [B,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 split_with_sizes [B,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,0]*[B,1,0]*[B,1,d_inner]*[B,1,d_inner+2*n_g*d_state]*[B,1,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 transpose        [B,1,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,1]
  model.layers.N.mamba_decoder.mamba                 concat           [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,1] -> [B,d_inner+2*n_g*d_state,d_conv+1]
  model.layers.N.mamba_decoder.mamba                 slice            [B,d_inner+2*n_g*d_state,d_conv+1] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 select           [d_inner+2*n_g*d_state,1,d_conv] -> w=[d_inner+2*n_g*d_state,1,d_conv] [d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,d_inner+2*n_g*d_state,d_conv]*[d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba_decoder.mamba                 sum              [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba                 add_             [B,d_inner+2*n_g*d_state]*[d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba.act             silu             [B,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,d_inner+2*n_g*d_state] -> [B,1,d_inner+2*n_g*d_state]
  model.layers.N.mamba_decoder.mamba                 split_with_sizes [B,1,d_inner+2*n_g*d_state] -> [B,1,d_inner]*[B,1,d_state]*[B,1,d_state]
  model.layers.N.mamba_decoder.mamba                 exp              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 neg              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 select           [B,1,d_head_ssm] -> [B,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,n_h_ssm] -> [B,1,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 transpose        [B,1,n_h_ssm] -> [B,d_head_ssm,1]
  model.layers.N.mamba_decoder.mamba                 expand           [B,d_head_ssm,1] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm] -> [n_h_ssm,B]
  model.layers.N.mamba_decoder.mamba                 expand           [n_h_ssm,B] -> [d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,n_h_ssm,d_head_ssm]*[d_head_ssm,n_h_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 softplus         [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 clamp            [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm,B] -> [n_h_ssm,B,1]
  model.layers.N.mamba_decoder.mamba                 expand           [n_h_ssm,B,1] -> [d_head_ssm,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm,1]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,n_h_ssm,d_head_ssm,1]*[d_head_ssm,n_h_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 exp              [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,1,d_state] -> [B,1,d_state]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [B,1,d_state] -> [B,1,1,d_state]
  model.layers.N.mamba_decoder.mamba                 expand           [B,1,1,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 clone            [B,1,n_h_ssm,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,1,n_h_ssm,d_state] -> [B,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,n_h_ssm,d_head_ssm,1]*[B,n_h_ssm,1,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,1,d_inner] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,1] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 clone            [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,n_h_ssm,d_head_ssm,d_state] -> [n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 view             [B,n_h_ssm,d_state] -> [n_h_ssm,d_state,B]
  model.layers.N.mamba_decoder.mamba                 batched_matmul   [n_h_ssm,d_head_ssm,d_state]*[n_h_ssm,d_state,B] -> [n_h_ssm,d_head_ssm,B]
  model.layers.N.mamba_decoder.mamba                 view             [n_h_ssm,d_head_ssm,B] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 expand           [d_head_ssm,B] -> [d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,n_h_ssm,d_head_ssm]*[d_head_ssm,n_h_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,n_h_ssm,d_head_ssm]*[B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba.norm            silu             [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [B,1,d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            view             [B,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            pow              [B,1,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            mean             [B,1,1,d_inner] -> [B,1,1,1]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_add  [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba_decoder.mamba.norm            rsqrt            [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [B,1,1,d_inner]*[B,1,1,1] -> [B,1,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            view             [B,1,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba_decoder.mamba.norm            elementwise_mul  [d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba_decoder.mamba.out_proj        t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mamba_decoder.mamba.out_proj        view             [B,1,d_inner] -> [B,d_inner]
  model.layers.N.mamba_decoder.mamba.out_proj        matmul           [B,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [B,d_model]
  model.layers.N.mamba_decoder.mamba.out_proj        _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.24                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.final_layernorm                              pow              [B,1,d_model] -> [B,1,d_model]
  model.final_layernorm                              mean             [B,1,d_model] -> [B,1,1]
  model.final_layernorm                              elementwise_add  [B,1,1] -> [B,1,1]
  model.final_layernorm                              rsqrt            [B,1,1] -> [B,1,1]
  model.final_layernorm                              elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.final_layernorm                              elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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

