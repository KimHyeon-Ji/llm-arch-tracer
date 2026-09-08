# 검토 요청 — CSV/JSONL 축 라벨 정확도만 본다 (요약 카드 아님)

## 이 검토가 확인하려는 것, 정확히 한 문장으로

**`full/<phase>.csv` / `full/<phase>.trace.raw.jsonl` 안에서, 텐서 축 하나하나에 붙은
심볼 이름(`n_h`, `d_moe`, `n_kv*d_head` 등)이 그 축이 실제로 무엇인지와 맞는가.**

이 프로젝트의 목적은 이 CSV/JSONL을 이용해 정확한 roofline(FLOPs/바이트) 계산을 하는
것입니다. 그러니 다른 무엇(요약 문서 문구, 집계 숫자, 서술 정확도)보다 **이 파일 안의
축 이름 하나하나가 진짜 맞는지**가 압도적으로 중요합니다. 그것만 확인해 주세요.

## 절대 하지 말아야 할 것 — 지난 라운드 실수

`model_summary.md` 안에 "유도 상수" 표라는 게 있습니다 (예시):

```
| 128 | 2·d_head (CSA/HCA 압축기 kv_proj·gate_proj 폭: Ca⊕Cb) | k_proj, self_attn, v_proj |
```

**이 표는 증거가 아닙니다.** 이 표는 정수 값 하나당 대표 식 **하나만** 고정으로 보여주고,
그 옆에는 "이 값이 나타나는 다른 모듈들"을 그냥 나열합니다 — 그 모듈들이 실제로 그 식을
쓴다는 뜻이 **전혀 아닙니다**. 지난 라운드에 정확히 이 표를 근거로 3건을 "오라벨"이라고
지적받았는데, 실제 `full/prefill.csv`를 직접 열어보니 셋 다 이미 정확했습니다:

- Qwen2.5-0.5B: `128`이 `k_proj/v_proj`에서 이미 `n_kv*d_head`로 정확히 렌더링되어
  있었음. 표에 뜬 `2*d_head`는 이 모델에 존재하지도 않는 압축기 모듈(CSA/HCA, DeepSeek
  전용)용 식이 우연히 같은 숫자라서 표에 같이 뜬 것뿐.
- ERNIE-4.5-21B-A3B: `3072`가 routed experts에서는 `2*d_moe`, shared_experts에서는
  `E_shared*d_moe`로 **이미 정확히 구분되어** 렌더링되어 있었음. 표는 둘 중 하나만 보여줌.
- Llama-4-Maverick: `2048`이 KV 캐시 폭(`2*n_kv*d_head`)과 `E*T`(전문가 수 × 토큰 수,
  이번 트레이스의 `T` 값 때문에 우연히 같은 숫자)가 겹친 것뿐. 실제 CSV에는 `k*T`가
  단 한 곳도 없고 `E*T`가 전부 정확히 쓰여 있었음.

**그러므로: "유도 상수" 표에 어떤 값이 여러 모듈에 걸쳐 나열돼 있다는 사실 자체는
아무것도 증명하지 않습니다.** 그 표는 아예 보지 마세요 — 무시하세요. 대신 아래 절차를
따르세요.

## 무엇을 봐야 하는가

각 모델 패킷의 **"5. 대표 트레이스 표본"** 절만 보세요. 이건 `full/prefill.csv` /
`full/decode.csv` (그리고 `.trace.raw.jsonl`)에서 (모듈, op) 조합마다 **서로 다른 shape는
전부** 뽑아 만든 것이라, 표 안의 모든 줄이 실제 산출물의 실제 행입니다 — 요약도 근사도
아닙니다. 이 표 밖의 어떤 것도(모델 요약 카드, 집계 숫자, 서술 문단) 판단 근거로 쓰지
마세요.

각 행에는 다음이 있습니다: `module_path`(레이어 번호는 `.N.`으로 정규화됨),
`op_type`, `phase`(prefill/decode), `input_shape` / `weight_shape` / `output_shape`
(심볼로 렌더됨), 그리고 그 옆 괄호 안에 **실제 정수 값**. 축 이름이 틀렸다고 판단하려면
**정확히 이 행 안에서** 어느 필드의 몇 번째 피연산자의 몇 번째 축이 틀렸는지 짚을 수
있어야 합니다.

## 확인 절차 (모델마다)

1. 공식 HF modeling/config 소스(패킷에 리비전과 설치된 transformers 버전이 적혀 있습니다
   — **그 버전 기준으로** 읽으세요, 최신 GitHub main이 아닙니다)를 열어 그 모듈이 실제로
   어떤 이름의 config 필드로 어떤 폭을 만드는지 확인
2. **반박 프레임으로 판단하세요**: 먼저 "이 라벨이 틀렸다는 증거가 있는가"를 찾고, 못
   찾으면 맞는 것으로 적으세요. "맞을 수도, 틀릴 수도 있어 보인다"는 사유로 지적하지
   마세요 — 소스로 직접 확인 가능한 것만 지적하세요.
3. attention 계열(MHA/GQA/MQA/MLA), MoE 라우팅/전문가 폭, KV 캐시 전제(어느 레이어가
   캐시를 갖는지, K/V가 별개 텐서인지)를 특히 주의 깊게 볼 것 — config 필드 이름만 보고
   판단하지 말고 실제 코드가 그 필드로 무엇을 만드는지 확인
4. 두 config 필드가 우연히 같은 숫자인 지점(값 충돌)은 특히 위험합니다 — 값이 맞아도
   개념이 틀릴 수 있는 자리이므로 소스를 반드시 열어서 "선언 순서"나 "실제 사용처"로
   판단하세요
5. decode 표본도 꼭 보세요 — prefill에 없는 축(캐시 길이, sliding/chunk 상한 등)이
   거기 있습니다
6. 같은 (모듈, op)인데 shape 표기가 갈리는 줄이 있는지도 확인

## 답변 형식

**틀렸다고 확신하는 것만** 아래 표로 적어주세요. 각 지적은 반드시 CSV/JSONL의 정확한
자리를 가리켜야 합니다 (요약 문서 문구나 집계 수치에 대한 지적은 이번 검토 범위 밖입니다
— 받지 않습니다):

| # | module_path | op_type | phase | field(i/w/o) | axis 위치 | 지금 라벨 | 실제 정수 값 | 제안 라벨 | 근거(공식 소스 파일명+줄+인용) | 확신도 |
|---|---|---|---|---|---|---|---|---|---|---|

전부 맞는 모델은 짧게 "이상 없음" 한 줄로 충분합니다. **확신 없는 지적은 아예 적지
마세요** — "유도 상수" 표만 보고 낸 지적은 이번엔 전부 오탐으로 되돌려보낼 것입니다.

이 요청 하나에 여러 모델의 전체 리뷰 패킷을 이어붙였습니다. 모델별로 독립적으로
판단해 주세요. 이 저장소의 파이프라인/추출 코드(`src/`, `rules/`, `develop/`)는 전혀
보지 마세요 — 애초에 첨부되어 있지 않습니다, 아래 각 모델의 리뷰 패킷이 전부입니다.

---


# 배치 D — 대상 모델: Zyphra__Zamba2-1.2B, deepseek-ai__DeepSeek-V2-Lite, openai-community__gpt2-xl, hf-internal-testing__tiny-random-LlamaForCausalLM


# ============================================================
# 모델: Zyphra__Zamba2-1.2B
# ============================================================

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
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'hybrid', 'linear_attention', 'linear_attention']
  c_kv         = None
  d_nope       = None
  d_v          = None
  c_q          = None
  d_rope       = None
  n_h_kda      = None
  d_head_kda   = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  n_attn_res_block = None
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
| KV cache 크기 | 2·n_kv·d_head = 2·32·128 = 8192 elems / token / layer; 6 attention layer(s) ⇒ 49152 / token |

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
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 32× linear_attention, 6× hybrid (총 38층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | —  _(해당 없음: 이 모델은 `partial_rope` 계열 구조를 쓰지 않음)_ |
| n_h_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| d_head_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
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

shape 축 **136,298개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 52,725 | 38.68% |
| 이 모듈 스코프의 심볼 | 50,951 | 37.38% |
| 스코프 없는 심볼 | 15,311 | 11.23% |
| 이 모듈 스코프의 유도식 | 8,365 | 6.14% |
| 이름 없음 (정수 유지) | 5,648 | 4.14% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,298 | 2.42% |

등록된 규칙 **127,352축**, 약한 근거 3,298축, 휴리스틱 **0축 (0.0%)**, 이름 없음 5,648축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

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

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

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
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 4458 unmapped rows, 30 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
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

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


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
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4458 unmapped rows, 30 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              clone            [B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.mamba                               unsqueeze        [n_h_ssm] -> [n_h_ssm,1]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [n_h_ssm,1]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
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
  model.layers.N.shared_transformer.self_attn        unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_kv,T,d_head]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_kv,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
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
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm] -> [n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,T,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [n_h_ssm,1]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
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
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,1,n_h_ssm,d_state]*[B,1,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,d_chunk,n_h_ssm,1]*[B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_chunk,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_mul  [B,1,d_chunk,d_chunk,n_h_ssm,1]*[B,1,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,d_chunk,d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_chunk,d_chunk,d_head_ssm,n_h_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 slice            [B,n_h_ssm,1,d_chunk] -> [B,d_head_ssm,1,1]
  model.layers.N.mamba_decoder.mamba                 sub              [B,d_head_ssm,1,1]*[B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 exp              [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba_decoder.mamba                 permute          [B,n_h_ssm,1,d_chunk] -> [B,1,d_chunk,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,n_h_ssm,d_chunk,d_state]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,n_h_ssm,d_chunk,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 sum              [B,1,d_head_ssm,d_chunk,d_state,n_h_ssm] -> [B,1,d_head_ssm,d_state,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 permute          [B,1,d_head_ssm,d_state,n_h_ssm] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 alias            [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 zeros_like       [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 concat           [B,1,n_h_ssm,d_head_ssm,d_state]*[B,1,n_h_ssm,d_head_ssm,d_state] -> [B,2,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba_decoder.mamba                 select           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 constant_pad_nd  [B,n_h_ssm,1] -> [B,d_head_ssm,2]
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
  model.layers.N.mamba                               unsqueeze        [n_h_ssm] -> [n_h_ssm,1]
  model.layers.N.mamba                               expand           [n_h_ssm,1] -> [n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm]*[n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               softplus         [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               clamp            [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               unsqueeze        [n_h_ssm,1] -> [n_h_ssm,1,1]
  model.layers.N.mamba                               expand           [n_h_ssm,1,1] -> [n_h_ssm,d_head_ssm,d_state]
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
  model.layers.N.shared_transformer.self_attn        view             [B,1,d_attn] -> [B,1,n_kv,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_kv,1,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.shared_transformer.self_attn        concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_kv,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.shared_transformer.self_attn        view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.shared_transformer.self_attn        expand           [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        view             [B,n_kv,T+1,d_head] -> [n_kv,T+1,d_head]
  model.layers.N.shared_transformer.self_attn        batched_matmul   [n_h,B,T+1]*[n_kv,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.shared_transformer.self_attn        _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.shared_transformer.self_attn        transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
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
  model.layers.N.mamba_decoder.mamba                 transpose        [B,1,n_h_ssm] -> [B,n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 expand           [B,n_h_ssm,1] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm] -> [n_h_ssm,1]
  model.layers.N.mamba_decoder.mamba                 expand           [n_h_ssm,1] -> [d_head_ssm,n_h_ssm]
  model.layers.N.mamba_decoder.mamba                 elementwise_add  [B,n_h_ssm,d_head_ssm]*[d_head_ssm,n_h_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 softplus         [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 clamp            [B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba_decoder.mamba                 unsqueeze        [n_h_ssm,1] -> [n_h_ssm,1,1]
  model.layers.N.mamba_decoder.mamba                 expand           [n_h_ssm,1,1] -> [d_head_ssm,n_h_ssm,d_state]
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
  model.layers.N.mamba_decoder.mamba                 expand           [d_head_ssm,1] -> [d_head_ssm,n_h_ssm]
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



# ============================================================
# 모델: deepseek-ai__DeepSeek-V2-Lite
# ============================================================

# 리뷰 패킷 — deepseek-ai/DeepSeek-V2-Lite

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `604d5664dddd88a0433dbae533b7fe9472482de0` / 트레이스 seq_len(T) = 17
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
  L            = 27
  d_model      = 2048
  n_h          = 16
  n_kv         = 16
  d_head       = 64
  d_ff         = 10944
  d_shared     = None
  V            = 102400
  ctx          = 163840
  E            = 64
  E_shared     = 2
  k            = 6
  n_grp        = 1
  k_grp        = 1
  d_moe        = 1408
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = None
  c_kv         = 512
  d_nope       = 128
  d_v          = 128
  c_q          = None
  d_rope       = 64
  n_h_kda      = None
  d_head_kda   = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  n_attn_res_block = None
  d_state      = None
  n_g_ssm      = None
  n_h_ssm      = None
  d_chunk      = None
  d_head_ssm   = None
  d_conv       = None
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

# Model Summary -- deepseek-ai/DeepSeek-V2-Lite

## 기본 정보

- revision: `604d5664dddd88a0433dbae533b7fe9472482de0`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 15.71B total, 2.66B active (16.9% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 163,840  _(config max_position_embeddings; yarn 스케일(원본 4096×40) — 벤더 광고 컨텍스트와 다를 수 있음)_ |
| 3 | DATE | 2024-05-15  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | MLA |
| 6 | LAYER MIX | 27× MLA  (FFN: 1 dense + 26 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 30.4 KiB (Low) |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=64, top-6, +2 shared, topk-then-softmax routing); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, RoPE, MLA, MoE, shared expert, topk-softmax routing |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `deepseek_v2` |
| attention | MLA — KV latent compression (kv_lora_rank=512); 헤드 q/k = nope(128)+rope(64)=192, v=128, n_h=16 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000), yarn scaling |
| FFN | MoE — 64 routed experts, top-6 + 2 shared, expert intermediate 1408, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | compressed MLA latent ≈ kv_lora_rank=512 (+decoupled RoPE dim) / token / layer |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 27 |
| d_model | 2048 |
| n_h | 16 |
| n_kv | 16 |
| d_head | 64 |
| d_ff | 10944 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 102400 |
| ctx | 163840 |
| E | 64 |
| E_shared | 2 |
| k | 6 |
| n_grp | 1 |
| k_grp | 1 |
| d_moe | 1408 |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
| c_kv | 512 |
| d_nope | 128 |
| d_v | 128 |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | 64 |
| n_h_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| d_head_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
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

shape 축 **77,485개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 24,377 | 31.46% |
| 스코프 없는 심볼 | 20,861 | 26.92% |
| 이 모듈 스코프의 심볼 | 16,407 | 21.17% |
| 이 모듈 스코프의 유도식 | 14,048 | 18.13% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,036 | 1.34% |
| 이름 없음 (정수 유지) | 756 | 0.98% |

등록된 규칙 **75,693축**, 약한 근거 1,036축, 휴리스틱 **0축 (0.0%)**, 이름 없음 756축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 32 | d_rope/2 (부분/decoupled RoPE의 rotate_half 분할 축) | rotary_emb, self_attn |
| 102 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 192 | d_nope + d_rope (MLA q/k head 폭) | self_attn |
| 256 | d_nope+d_v | self_attn |
| 576 | c_kv+d_rope (MLA kv_a_proj_with_mqa 출력) | kv_a_proj_with_mqa, self_attn |
| 2816 | E_shared·d_moe (공유 전문가 FFN 폭 — 공유 전문가 수만큼 넓힌 하나의 MLP) | act_fn, down_proj, experts, gate_proj, shared_experts, up_proj |
| 3072 | n_h·(d_nope+d_rope) (MLA q_b_proj 출력) | q_proj, self_attn |
| 4096 | n_h·(d_nope+d_v) (MLA kv_b_proj 출력) | kv_b_proj, self_attn |

## 레이어 구조

- layer 0: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 1-26: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 2개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 27 == 27 |
| C2 | WARN | 2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2048 in 27/27 layers |
| C6 | PASS | hidden_size=2048 (heuristic check, 2675 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | WARN | MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count ... |
| C9 | PASS | vocab_size=102400, tie_word_embeddings=False |
| C10 | PASS | all 351 params covered |
| C11 | PASS | 108 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 2548 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `deepseek-ai/DeepSeek-V2-Lite` config.json @ `604d5664dddd88a0433dbae533b7fe9472482de0` (sha256 `948d543e0e22…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- deepseek-ai/DeepSeek-V2-Lite @ 604d5664dddd88a0433dbae533b7fe9472482de0

C1   PASS   27 == 27
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 27/27 layers
C6   PASS   hidden_size=2048 (heuristic check, 2675 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   WARN   MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=102400, tie_word_embeddings=False
C10  PASS   all 351 params covered
C11  PASS   108 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2548 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default', 'aten.ge.Scalar']
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
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [B,T] -> [B,1,T]
  model.rotary_emb                                   _to_copy         [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,T] -> [B,1,T]
  model.rotary_emb                                   view             [B,1,T] -> [B,1,T]
  model.rotary_emb                                   batched_matmul   [B,d_rope/2,1]*[B,1,T] -> [B,d_rope/2,T]
  model.rotary_emb                                   _unsafe_view     [B,d_rope/2,T] -> [B,d_rope/2,T]
  model.rotary_emb                                   transpose        [B,d_rope/2,T] -> [B,T,d_rope/2]
  model.rotary_emb                                   ones_like        [B,T,d_rope/2] -> [B,T,d_rope/2]
  model.rotary_emb                                   polar            [B,T,d_rope/2]*[B,T,d_rope/2] -> [B,T,d_rope/2]
  model.rotary_emb                                   elementwise_mul  [B,T,d_rope/2] -> [B,T,d_rope/2]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*(d_nope+d_rope),d_model] -> w=[n_h*(d_nope+d_rope),d_model] [d_model,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*(d_nope+d_rope)] -> w=[n_h*(d_nope+d_rope),d_model] [T,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*(d_nope+d_rope)] -> [B,T,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn                           view             [B,T,n_h*(d_nope+d_rope)] -> [B,T,n_h,d_nope+d_rope]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_nope+d_rope] -> [B,n_h,T,d_nope+d_rope]
  model.layers.N.self_attn                           split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope]*[B,n_h,T,d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        t                [c_kv+d_rope,d_model] -> w=[c_kv+d_rope,d_model] [d_model,c_kv+d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.kv_a_proj_with_mqa        matmul           [T,d_model]*[d_model,c_kv+d_rope] -> w=[c_kv+d_rope,d_model] [T,c_kv+d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        _unsafe_view     [T,c_kv+d_rope] -> [B,T,c_kv+d_rope]
  model.layers.N.self_attn                           split_with_sizes [B,T,c_kv+d_rope] -> [B,T,c_kv]*[B,T,d_rope]
  model.layers.N.self_attn.kv_a_layernorm            _to_copy         [B,T,c_kv] -> [B,T,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            pow              [B,T,c_kv] -> [B,T,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            mean             [B,T,c_kv] -> [B,T,1]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.kv_a_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_mul  [B,T,c_kv]*[B,T,1] -> [B,T,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_mul  [c_kv]*[B,T,c_kv] -> [B,T,c_kv]
  model.layers.N.self_attn.kv_b_proj                 t                [n_h*(d_nope+d_v),c_kv] -> w=[n_h*(d_nope+d_v),c_kv] [c_kv,n_h*(d_nope+d_v)]
  model.layers.N.self_attn.kv_b_proj                 view             [B,T,c_kv] -> [T,c_kv]
  model.layers.N.self_attn.kv_b_proj                 matmul           [T,c_kv]*[c_kv,n_h*(d_nope+d_v)] -> w=[n_h*(d_nope+d_v),c_kv] [T,n_h*(d_nope+d_v)]
  model.layers.N.self_attn.kv_b_proj                 _unsafe_view     [T,n_h*(d_nope+d_v)] -> [B,T,n_h*(d_nope+d_v)]
  model.layers.N.self_attn                           view             [B,T,n_h*(d_nope+d_v)] -> [B,T,n_h,d_nope+d_v]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_nope+d_v] -> [B,n_h,T,d_nope+d_v]
  model.layers.N.self_attn                           split_with_sizes [B,n_h,T,d_nope+d_v] -> [B,n_h,T,d_nope]*[B,n_h,T,d_v]
  model.layers.N.self_attn                           view             [B,T,d_rope] -> [B,1,T,d_rope]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_rope] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           view             [B,n_h,T,d_rope] -> [B,n_h,T,d_rope/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,n_h,T,d_rope/2,2] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           _to_copy         [B,1,T,d_rope] -> [B,1,T,d_rope]
  model.layers.N.self_attn                           view             [B,1,T,d_rope] -> [B,1,T,d_rope/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,1,T,d_rope/2,2] -> [B,1,T,d_rope/2]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_rope/2] -> [B,1,T,d_rope/2]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_rope/2]*[B,1,T,d_rope/2] -> [B,n_h,T,d_rope/2]
  model.layers.N.self_attn                           view_as_real     [B,n_h,T,d_rope/2] -> [B,n_h,T,d_rope/2,2]
  model.layers.N.self_attn                           view             [B,n_h,T,d_rope/2,2] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,1,T,d_rope/2]*[B,1,T,d_rope/2] -> [B,1,T,d_rope/2]
  model.layers.N.self_attn                           view_as_real     [B,1,T,d_rope/2] -> [B,1,T,d_rope/2,2]
  model.layers.N.self_attn                           expand           [B,1,T,d_rope] -> [B,n_h,T,d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_nope]*[B,n_h,T,d_rope] -> [B,n_h,T,d_nope+d_rope]
  model.layers.N.self_attn                           concat           [0]*[B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope+d_rope]
  model.layers.N.self_attn                           concat           [0]*[B,n_h,T,d_v] -> [B,n_h,T,d_v]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_nope+d_rope] -> [B,n_h,d_nope+d_rope,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope+d_rope]
  model.layers.N.self_attn                           expand           [B,n_h,d_nope+d_rope,T] -> [B,n_h,d_nope+d_rope,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_nope+d_rope]*[n_h,d_nope+d_rope,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_v] -> [B,n_h,T,d_v]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_v] -> [n_h,T,d_v]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_v] -> [B,n_h,T,d_v]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_v] -> [B,T,n_h,d_v]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_v] -> [B,T,n_h,d_v]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_v] -> w=[d_model,n_h*d_v] [n_h*d_v,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_v] -> [T,n_h*d_v]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_v]*[n_h*d_v,d_model] -> w=[d_model,n_h*d_v] [T,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.mlp.gate                            view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mlp.gate                            softmax          [T,E] -> [T,E]
  model.layers.N.mlp.gate                            topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mlp.gate                            elementwise_mul  [T,k] -> [T,k]
  model.layers.N.mlp                                 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.experts                         view             [T,k] -> [k*T]
  model.layers.N.mlp.experts                         sort             [k*T] -> [k*T]*[k*T]
  model.layers.N.mlp.experts                         floor_divide     [k*T] -> [k*T]
  model.layers.N.mlp.experts                         index            [T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.mlp.experts                         index            [k*T]*[k*T] -> [k*T]
  model.layers.N.mlp.experts                         _to_copy         [k*T] -> [k*T]
  model.layers.N.mlp.experts                         histc            [k*T] -> [E]
  model.layers.N.mlp.experts                         cumsum           [E] -> [E]
  model.layers.N.mlp.experts                         ge               [k*T] -> [k*T]
  model.layers.N.mlp.experts                         unsqueeze        [k*T] -> [k*T,1]
  model.layers.N.mlp.experts                         clamp_           [k*T] -> [k*T]
  model.layers.N.mlp.experts                         masked_fill_     [k*T,d_model]*[k*T,1] -> [k*T,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k*T,2*d_moe]
  model.layers.N.mlp.experts                         split            [k*T,2*d_moe] -> [k*T,d_moe]*[k*T,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_moe]*[k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k*T,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_model]*[k*T,1] -> [k*T,d_model]
  model.layers.N.mlp.experts                         empty_like       [k*T] -> [k*T]
  model.layers.N.mlp.experts                         arange           [] -> [k*T]
  model.layers.N.mlp.experts                         index_put_       [k*T]*[k*T]*[k*T] -> [k*T]
  model.layers.N.mlp.experts                         index            [k*T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.mlp.experts                         view             [k*T,d_model] -> [T,k,d_model]
  model.layers.N.mlp.experts                         sum              [T,k,d_model] -> [T,d_model]
  model.layers.N.mlp.experts                         _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mlp                                 view             [T,d_model] -> [B,T,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [T,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        _unsafe_view     [T,E_shared*d_moe] -> [B,T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [B,T,E_shared*d_moe] -> [B,T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [T,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          _unsafe_view     [T,E_shared*d_moe] -> [B,T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [B,T,E_shared*d_moe]*[B,T,E_shared*d_moe] -> [B,T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,E_shared*d_moe] -> w=[d_model,E_shared*d_moe] [E_shared*d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        view             [B,T,E_shared*d_moe] -> [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [T,E_shared*d_moe]*[E_shared*d_moe,d_model] -> w=[d_model,E_shared*d_moe] [T,d_model]
  model.layers.N.mlp.shared_experts.down_proj        _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.norm                                         _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.norm                                         pow              [B,T,d_model] -> [B,T,d_model]
  model.norm                                         mean             [B,T,d_model] -> [B,T,1]
  model.norm                                         elementwise_add  [B,T,1] -> [B,T,1]
  model.norm                                         rsqrt            [B,T,1] -> [B,T,1]
  model.norm                                         elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.rotary_emb                                   unsqueeze        [d_rope/2] -> [B,d_rope/2]
  model.rotary_emb                                   unsqueeze        [B,d_rope/2] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   unsqueeze        [B,1] -> [B,1,1]
  model.rotary_emb                                   _to_copy         [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   expand           [B,1,1] -> [B,1,1]
  model.rotary_emb                                   view             [B,1,1] -> [B,1,1]
  model.rotary_emb                                   batched_matmul   [B,d_rope/2,1]*[B,1,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   _unsafe_view     [B,d_rope/2,1] -> [B,d_rope/2,1]
  model.rotary_emb                                   transpose        [B,d_rope/2,1] -> [B,1,d_rope/2]
  model.rotary_emb                                   ones_like        [B,1,d_rope/2] -> [B,1,d_rope/2]
  model.rotary_emb                                   polar            [B,1,d_rope/2]*[B,1,d_rope/2] -> [B,1,d_rope/2]
  model.rotary_emb                                   elementwise_mul  [B,1,d_rope/2] -> [B,1,d_rope/2]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*(d_nope+d_rope),d_model] -> w=[n_h*(d_nope+d_rope),d_model] [d_model,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*(d_nope+d_rope)] -> w=[n_h*(d_nope+d_rope),d_model] [B,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*(d_nope+d_rope)] -> [B,1,n_h*(d_nope+d_rope)]
  model.layers.N.self_attn                           view             [B,1,n_h*(d_nope+d_rope)] -> [B,1,n_h,d_nope+d_rope]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_nope+d_rope] -> [B,n_h,1,d_nope+d_rope]
  model.layers.N.self_attn                           split_with_sizes [B,n_h,1,d_nope+d_rope] -> [B,n_h,1,d_nope]*[B,n_h,1,d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        t                [c_kv+d_rope,d_model] -> w=[c_kv+d_rope,d_model] [d_model,c_kv+d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.kv_a_proj_with_mqa        matmul           [B,d_model]*[d_model,c_kv+d_rope] -> w=[c_kv+d_rope,d_model] [B,c_kv+d_rope]
  model.layers.N.self_attn.kv_a_proj_with_mqa        _unsafe_view     [B,c_kv+d_rope] -> [B,1,c_kv+d_rope]
  model.layers.N.self_attn                           split_with_sizes [B,1,c_kv+d_rope] -> [B,1,c_kv]*[B,1,d_rope]
  model.layers.N.self_attn.kv_a_layernorm            _to_copy         [B,1,c_kv] -> [B,1,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            pow              [B,1,c_kv] -> [B,1,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            mean             [B,1,c_kv] -> [B,1,1]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.kv_a_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_mul  [B,1,c_kv]*[B,1,1] -> [B,1,c_kv]
  model.layers.N.self_attn.kv_a_layernorm            elementwise_mul  [c_kv]*[B,1,c_kv] -> [B,1,c_kv]
  model.layers.N.self_attn.kv_b_proj                 t                [n_h*(d_nope+d_v),c_kv] -> w=[n_h*(d_nope+d_v),c_kv] [c_kv,n_h*(d_nope+d_v)]
  model.layers.N.self_attn.kv_b_proj                 view             [B,1,c_kv] -> [B,c_kv]
  model.layers.N.self_attn.kv_b_proj                 matmul           [B,c_kv]*[c_kv,n_h*(d_nope+d_v)] -> w=[n_h*(d_nope+d_v),c_kv] [B,n_h*(d_nope+d_v)]
  model.layers.N.self_attn.kv_b_proj                 _unsafe_view     [B,n_h*(d_nope+d_v)] -> [B,1,n_h*(d_nope+d_v)]
  model.layers.N.self_attn                           view             [B,1,n_h*(d_nope+d_v)] -> [B,1,n_h,d_nope+d_v]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_nope+d_v] -> [B,n_h,1,d_nope+d_v]
  model.layers.N.self_attn                           split_with_sizes [B,n_h,1,d_nope+d_v] -> [B,n_h,1,d_nope]*[B,n_h,1,d_v]
  model.layers.N.self_attn                           view             [B,1,d_rope] -> [B,1,1,d_rope]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_rope] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           view             [B,n_h,1,d_rope] -> [B,n_h,1,d_rope/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,n_h,1,d_rope/2,2] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           _to_copy         [B,1,1,d_rope] -> [B,1,1,d_rope]
  model.layers.N.self_attn                           view             [B,1,1,d_rope] -> [B,1,1,d_rope/2,2]
  model.layers.N.self_attn                           view_as_complex  [B,1,1,d_rope/2,2] -> [B,1,1,d_rope/2]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_rope/2] -> [B,1,1,d_rope/2]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_rope/2]*[B,1,1,d_rope/2] -> [B,n_h,1,d_rope/2]
  model.layers.N.self_attn                           view_as_real     [B,n_h,1,d_rope/2] -> [B,n_h,1,d_rope/2,2]
  model.layers.N.self_attn                           view             [B,n_h,1,d_rope/2,2] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           elementwise_mul  [B,1,1,d_rope/2]*[B,1,1,d_rope/2] -> [B,1,1,d_rope/2]
  model.layers.N.self_attn                           view_as_real     [B,1,1,d_rope/2] -> [B,1,1,d_rope/2,2]
  model.layers.N.self_attn                           expand           [B,1,1,d_rope] -> [B,n_h,1,d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_nope]*[B,n_h,1,d_rope] -> [B,n_h,1,d_nope+d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_nope+d_rope]*[B,n_h,1,d_nope+d_rope] -> [B,n_h,T+1,d_nope+d_rope]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_v]*[B,n_h,1,d_v] -> [B,n_h,T+1,d_v]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_nope+d_rope] -> [B,n_h,d_nope+d_rope,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_nope+d_rope] -> [B,n_h,1,d_nope+d_rope]
  model.layers.N.self_attn                           expand           [B,n_h,d_nope+d_rope,T+1] -> [B,n_h,d_nope+d_rope,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_nope+d_rope]*[n_h,d_nope+d_rope,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_v] -> [B,n_h,T+1,d_v]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_v] -> [n_h,B,d_v]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_v] -> [B,n_h,1,d_v]
  model.layers.N.self_attn                           transpose        [B,n_h,1,d_v] -> [B,1,n_h,d_v]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_v] -> w=[d_model,n_h*d_v] [n_h*d_v,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_v] -> [B,n_h*d_v]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_v]*[n_h*d_v,d_model] -> w=[d_model,n_h*d_v] [B,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.mlp.gate                            view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            _to_copy         [E,d_model] -> w=[E,d_model] [E,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mlp.gate                            softmax          [B,E] -> [B,E]
  model.layers.N.mlp.gate                            topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mlp.gate                            elementwise_mul  [B,k] -> [B,k]
  model.layers.N.mlp                                 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.experts                         view             [B,k] -> [k]
  model.layers.N.mlp.experts                         sort             [k] -> [k]*[k]
  model.layers.N.mlp.experts                         floor_divide     [k] -> [k]
  model.layers.N.mlp.experts                         index            [B,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         index            [k]*[k] -> [k]
  model.layers.N.mlp.experts                         _to_copy         [k] -> [k]
  model.layers.N.mlp.experts                         histc            [k] -> [E]
  model.layers.N.mlp.experts                         cumsum           [E] -> [E]
  model.layers.N.mlp.experts                         ge               [k] -> [k]
  model.layers.N.mlp.experts                         unsqueeze        [k] -> [k,1]
  model.layers.N.mlp.experts                         clamp_           [k] -> [k]
  model.layers.N.mlp.experts                         masked_fill_     [k,d_model]*[k,1] -> [k,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k,2*d_moe]
  model.layers.N.mlp.experts                         split            [k,2*d_moe] -> [k,d_moe]*[k,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_moe]*[k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_model]*[k,1] -> [k,d_model]
  model.layers.N.mlp.experts                         empty_like       [k] -> [k]
  model.layers.N.mlp.experts                         arange           [] -> [k]
  model.layers.N.mlp.experts                         index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.mlp.experts                         index            [k,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         view             [k,d_model] -> [B,k,d_model]
  model.layers.N.mlp.experts                         sum              [B,k,d_model] -> [B,d_model]
  model.layers.N.mlp.experts                         _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mlp                                 view             [B,d_model] -> [B,1,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [B,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        _unsafe_view     [B,E_shared*d_moe] -> [B,1,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [B,1,E_shared*d_moe] -> [B,1,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [B,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          _unsafe_view     [B,E_shared*d_moe] -> [B,1,E_shared*d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [B,1,E_shared*d_moe]*[B,1,E_shared*d_moe] -> [B,1,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,E_shared*d_moe] -> w=[d_model,E_shared*d_moe] [E_shared*d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        view             [B,1,E_shared*d_moe] -> [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [B,E_shared*d_moe]*[E_shared*d_moe,d_model] -> w=[d_model,E_shared*d_moe] [B,d_model]
  model.layers.N.mlp.shared_experts.down_proj        _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
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
  model.norm                                         _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.norm                                         pow              [B,1,d_model] -> [B,1,d_model]
  model.norm                                         mean             [B,1,d_model] -> [B,1,1]
  model.norm                                         elementwise_add  [B,1,1] -> [B,1,1]
  model.norm                                         rsqrt            [B,1,1] -> [B,1,1]
  model.norm                                         elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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



# ============================================================
# 모델: openai-community__gpt2-xl
# ============================================================

# 리뷰 패킷 — openai-community/gpt2-xl

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `15ea56dee5df4983c59b2538573817e1667135e2` / 트레이스 seq_len(T) = 16
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
  L            = 48
  d_model      = 1600
  n_h          = 25
  n_kv         = 25
  d_head       = 64
  d_ff         = 6400
  d_shared     = None
  V            = 50257
  ctx          = 1024
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = None
  c_kv         = None
  d_nope       = None
  d_v          = None
  c_q          = None
  d_rope       = None
  n_h_kda      = None
  d_head_kda   = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  n_attn_res_block = None
  d_state      = None
  n_g_ssm      = None
  n_h_ssm      = None
  d_chunk      = None
  d_head_ssm   = None
  d_conv       = None
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

# Model Summary -- openai-community/gpt2-xl

## 기본 정보

- revision: `15ea56dee5df4983c59b2538573817e1667135e2`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 1.56B total (dense) |
| 2 | Context (tokens) | 1,024  _(config max_position_embeddings)_ |
| 3 | DATE | 2022-03-02  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MHA |
| 6 | LAYER MIX | 48× MHA |
| 7 | KV CACHE / TOKEN (BF16) | 300.0 KiB (High) |
| 8 | KEY DETAIL | MHA attention; dense FFN |
| 9 | Related concepts | LayerNorm, learned-pos, MHA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `gpt2` |
| attention | MHA — 25 heads (no GQA/MQA), d_head=64 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | learned absolute position embeddings |
| FFN | dense FFN — intermediate 6400, ? |
| 정규화 | LayerNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·25·64 = 3200 elems / token / layer; 48 attention layer(s) ⇒ 153600 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 48 |
| d_model | 1600 |
| n_h | 25 |
| n_kv | 25 |
| d_head | 64 |
| d_ff | 6400 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 50257 |
| ctx | 1024 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | —  _(해당 없음: 이 모델은 `partial_rope` 계열 구조를 쓰지 않음)_ |
| n_h_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| d_head_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
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

shape 축 **57,573개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 18,683 | 32.45% |
| 이 모듈 스코프의 심볼 | 16,966 | 29.47% |
| 스코프 없는 심볼 | 14,931 | 25.93% |
| 이 모듈 스코프의 유도식 | 5,351 | 9.29% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,450 | 2.52% |
| 이름 없음 (정수 유지) | 192 | 0.33% |

등록된 규칙 **55,931축**, 약한 근거 1,450축, 휴리스틱 **0축 (0.0%)**, 이름 없음 192축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 4800 | (n_h + 2·n_kv)·d_head (fused QKV 투영 폭 — Q·K·V 한 행렬) | attn, c_attn |

## 레이어 구조

- layer 0-47: attn, ln_1, ln_2, mlp

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 48 == 48 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=1600 in 48/48 layers |
| C6 | PASS | hidden_size=1600 (heuristic check, 528 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=50257, tie_word_embeddings=True |
| C10 | PASS | all 580 params covered |
| C11 | PASS | 96 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1418 unmapped rows, 13 distinct raw ops: ['aten._unsafe_view.default', 'aten.alias.default', 'ate... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `openai-community/gpt2-xl` config.json @ `15ea56dee5df4983c59b2538573817e1667135e2` (sha256 `dffbb746a29c…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- openai-community/gpt2-xl @ 15ea56dee5df4983c59b2538573817e1667135e2

C1   PASS   48 == 48
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=1600 in 48/48 layers
C6   PASS   hidden_size=1600 (heuristic check, 528 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=50257, tie_word_embeddings=True
C10  PASS   all 580 params covered
C11  PASS   96 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1418 unmapped rows, 13 distinct raw ops: ['aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.split.Tensor', 'aten.t.default', 'aten.transpose.int']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  transformer                                        view             [B,T] -> [B,T]
  transformer.wte                                    embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  transformer.wpe                                    embedding        [ctx,d_model]*[B,T] -> w=[ctx,d_model] [B,T,d_model]
  transformer                                        elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer                                        arange           [] -> [B]
  transformer                                        arange           [] -> [T]
  transformer                                        elementwise_add  [T] -> [T]
  transformer                                        unsqueeze        [B] -> [B,1]
  transformer                                        unsqueeze        [B,1] -> [B,1,1]
  transformer                                        unsqueeze        [B,1,1] -> [B,1,1,1]
  transformer                                        unsqueeze        [T] -> [B,T]
  transformer                                        unsqueeze        [B,T] -> [B,1,T]
  transformer                                        unsqueeze        [B,1,T] -> [B,1,T,1]
  transformer                                        le               [B,1,1,T]*[B,1,T,1] -> [B,1,T,T]
  transformer                                        expand           [B,1,T,T] -> [B,1,T,T]
  transformer                                        scalar_tensor    [] -> []
  transformer                                        where            [B,1,T,T]*[]*[] -> [B,1,T,T]
  transformer.h.N.ln_1                               layernorm        [B,T,d_model]*[d_model]*[d_model] -> [B,T,d_model]*[B,T,1]*[B,T,1]
  transformer.h.N.attn.c_attn                        view             [B,T,d_model] -> [T,d_model]
  transformer.h.N.attn.c_attn                        linear           [(n_h+2*n_kv)*d_head]*[T,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[d_model,(n_h+2*n_kv)*d_head] [T,(n_h+2*n_kv)*d_head]
  transformer.h.N.attn.c_attn                        view             [T,(n_h+2*n_kv)*d_head] -> [B,T,(n_h+2*n_kv)*d_head]
  transformer.h.N.attn                               split            [B,T,(n_h+2*n_kv)*d_head] -> [B,T,n_h*d_head]*[B,T,n_h*d_head]*[B,T,n_h*d_head]
  transformer.h.N.attn                               view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  transformer.h.N.attn                               transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.attn                               concat           [0]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.attn                               transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  transformer.h.N.attn                               expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.attn                               view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  transformer.h.N.attn                               expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  transformer.h.N.attn                               view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  transformer.h.N.attn                               batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  transformer.h.N.attn                               _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.attn                               elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.attn                               elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  transformer.h.N.attn                               softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.attn                               expand           [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.attn                               view             [B,n_h,T,T] -> [n_h,T,T]
  transformer.h.N.attn                               batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  transformer.h.N.attn                               _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.attn                               transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  transformer.h.N.attn                               clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  transformer.h.N.attn                               _unsafe_view     [B,T,n_h,d_head] -> [B,T,n_h*d_head]
  transformer.h.N.attn.c_proj                        view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  transformer.h.N.attn.c_proj                        linear           [n_h*d_head]*[T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  transformer.h.N.attn.c_proj                        view             [T,d_model] -> [B,T,d_model]
  transformer.h.0                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.N.ln_2                               layernorm        [B,T,d_model]*[d_model]*[d_model] -> [B,T,d_model]*[B,T,1]*[B,T,1]
  transformer.h.N.mlp.c_fc                           view             [B,T,d_model] -> [T,d_model]
  transformer.h.N.mlp.c_fc                           linear           [d_ff]*[T,d_model]*[d_model,d_ff] -> w=[d_model,d_ff] [T,d_ff]
  transformer.h.N.mlp.c_fc                           view             [T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            elementwise_mul  [B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            pow              [B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            elementwise_add  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            tanh             [B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            elementwise_add  [B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.c_proj                         view             [B,T,d_ff] -> [T,d_ff]
  transformer.h.N.mlp.c_proj                         linear           [d_model]*[T,d_ff]*[d_ff,d_model] -> w=[d_ff,d_model] [T,d_model]
  transformer.h.N.mlp.c_proj                         view             [T,d_model] -> [B,T,d_model]
  transformer.h.1                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.2                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.3                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.4                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.5                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.6                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.7                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.8                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.9                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.10                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.11                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.12                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.13                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.14                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.15                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.16                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.17                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.18                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.19                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.20                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.21                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.22                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.23                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.24                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.25                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.26                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.27                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.28                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.29                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.30                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.31                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.32                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.33                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.34                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.35                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.36                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.37                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.38                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.39                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.40                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.41                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.42                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.43                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.44                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.45                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.46                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.47                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.ln_f                                   layernorm        [B,T,d_model]*[d_model]*[d_model] -> [B,T,d_model]*[B,T,1]*[B,T,1]
  transformer                                        view             [B,T,d_model] -> [B,T,d_model]
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
  transformer                                        view             [B,1] -> [B,1]
  transformer.wte                                    embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  transformer.wpe                                    embedding        [ctx,d_model]*[B,1] -> w=[ctx,d_model] [B,1,d_model]
  transformer                                        elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer                                        arange           [] -> [B]
  transformer                                        elementwise_add  [B] -> [B]
  transformer                                        arange           [] -> [T+1]
  transformer                                        elementwise_add  [T+1] -> [T+1]
  transformer                                        unsqueeze        [B] -> [B,1]
  transformer                                        unsqueeze        [B,1] -> [B,1,1]
  transformer                                        unsqueeze        [B,1,1] -> [B,1,1,1]
  transformer                                        unsqueeze        [T+1] -> [B,T+1]
  transformer                                        unsqueeze        [B,T+1] -> [B,1,T+1]
  transformer                                        unsqueeze        [B,1,T+1] -> [B,1,1,T+1]
  transformer                                        le               [B,1,1,T+1]*[B,1,1,1] -> [B,1,1,T+1]
  transformer                                        expand           [B,1,1,T+1] -> [B,1,1,T+1]
  transformer                                        scalar_tensor    [] -> []
  transformer                                        where            [B,1,1,T+1]*[]*[] -> [B,1,1,T+1]
  transformer.h.N.ln_1                               layernorm        [B,1,d_model]*[d_model]*[d_model] -> [B,1,d_model]*[B,1,1]*[B,1,1]
  transformer.h.N.attn.c_attn                        view             [B,1,d_model] -> [B,d_model]
  transformer.h.N.attn.c_attn                        linear           [(n_h+2*n_kv)*d_head]*[B,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[d_model,(n_h+2*n_kv)*d_head] [B,(n_h+2*n_kv)*d_head]
  transformer.h.N.attn.c_attn                        view             [B,(n_h+2*n_kv)*d_head] -> [B,1,(n_h+2*n_kv)*d_head]
  transformer.h.N.attn                               split            [B,1,(n_h+2*n_kv)*d_head] -> [B,1,n_h*d_head]*[B,1,n_h*d_head]*[B,1,n_h*d_head]
  transformer.h.N.attn                               view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  transformer.h.N.attn                               transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.attn                               concat           [B,n_h,T,d_head]*[B,n_h,1,d_head] -> [B,n_h,T+1,d_head]
  transformer.h.N.attn                               transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  transformer.h.N.attn                               expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.attn                               view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  transformer.h.N.attn                               expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  transformer.h.N.attn                               view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  transformer.h.N.attn                               batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  transformer.h.N.attn                               _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.attn                               elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.attn                               elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.attn                               softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.attn                               expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.attn                               view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  transformer.h.N.attn                               expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
  transformer.h.N.attn                               view             [B,n_h,T+1,d_head] -> [n_h,T+1,d_head]
  transformer.h.N.attn                               batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  transformer.h.N.attn                               _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.attn                               transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  transformer.h.N.attn                               view             [B,1,n_h,d_head] -> [B,1,n_h*d_head]
  transformer.h.N.attn.c_proj                        view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  transformer.h.N.attn.c_proj                        linear           [n_h*d_head]*[B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  transformer.h.N.attn.c_proj                        view             [B,d_model] -> [B,1,d_model]
  transformer.h.0                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.N.ln_2                               layernorm        [B,1,d_model]*[d_model]*[d_model] -> [B,1,d_model]*[B,1,1]*[B,1,1]
  transformer.h.N.mlp.c_fc                           view             [B,1,d_model] -> [B,d_model]
  transformer.h.N.mlp.c_fc                           linear           [d_ff]*[B,d_model]*[d_model,d_ff] -> w=[d_model,d_ff] [B,d_ff]
  transformer.h.N.mlp.c_fc                           view             [B,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            elementwise_mul  [B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            pow              [B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            elementwise_add  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            tanh             [B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            elementwise_add  [B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.c_proj                         view             [B,1,d_ff] -> [B,d_ff]
  transformer.h.N.mlp.c_proj                         linear           [d_model]*[B,d_ff]*[d_ff,d_model] -> w=[d_ff,d_model] [B,d_model]
  transformer.h.N.mlp.c_proj                         view             [B,d_model] -> [B,1,d_model]
  transformer.h.1                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.2                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.3                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.4                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.5                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.6                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.7                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.8                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.9                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.10                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.11                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.12                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.13                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.14                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.15                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.16                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.17                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.18                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.19                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.20                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.21                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.22                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.23                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.24                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.25                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.26                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.27                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.28                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.29                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.30                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.31                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.32                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.33                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.34                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.35                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.36                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.37                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.38                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.39                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.40                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.41                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.42                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.43                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.44                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.45                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.46                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.47                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.ln_f                                   layernorm        [B,1,d_model]*[d_model]*[d_model] -> [B,1,d_model]*[B,1,1]*[B,1,1]
  transformer                                        view             [B,1,d_model] -> [B,1,d_model]
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



# ============================================================
# 모델: hf-internal-testing__tiny-random-LlamaForCausalLM
# ============================================================

# 리뷰 패킷 — hf-internal-testing/tiny-random-LlamaForCausalLM

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `9fb191250dd56d0ba7ec9785a025ed29c03d5998` / 트레이스 seq_len(T) = 17
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
  L            = 2
  d_model      = 16
  n_h          = 4
  n_kv         = 4
  d_head       = 4
  d_ff         = 64
  d_shared     = None
  V            = 32000
  ctx          = 2048
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = None
  c_kv         = None
  d_nope       = None
  d_v          = None
  c_q          = None
  d_rope       = None
  n_h_kda      = None
  d_head_kda   = None
  m_csa        = None
  m_hca        = None
  g_o          = None
  d_g          = None
  n_h_I        = None
  c_I          = None
  k_I          = None
  n_hc         = None
  t_sinkhorn   = None
  n_attn_res_block = None
  d_state      = None
  n_g_ssm      = None
  n_h_ssm      = None
  d_chunk      = None
  d_head_ssm   = None
  d_conv       = None
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

# Model Summary -- hf-internal-testing/tiny-random-LlamaForCausalLM

## 기본 정보

- revision: `9fb191250dd56d0ba7ec9785a025ed29c03d5998`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 1.03M total (dense) |
| 2 | Context (tokens) | 2,048  _(config max_position_embeddings)_ |
| 3 | DATE | 2024-02-19  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MHA |
| 6 | LAYER MIX | 2× MHA |
| 7 | KV CACHE / TOKEN (BF16) | 0.1 KiB (Very low) |
| 8 | KEY DETAIL | MHA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, MHA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `llama` |
| attention | MHA — 4 heads (no GQA/MQA), d_head=4 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000.0) |
| FFN | dense FFN — intermediate 64, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·4·4 = 32 elems / token / layer; 2 attention layer(s) ⇒ 64 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 2 |
| d_model | 16 |
| n_h | 4 |
| n_kv | 4 |
| d_head | 4 |
| d_ff | 64 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 32000 |
| ctx | 2048 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | —  _(해당 없음: 이 모델은 `sched` 계열 구조를 쓰지 않음)_ |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| c_q | —  _(해당 없음: 이 모델은 `lowrank_q` 계열 구조를 쓰지 않음)_ |
| d_rope | —  _(해당 없음: 이 모델은 `partial_rope` 계열 구조를 쓰지 않음)_ |
| n_h_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| d_head_kda | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| m_csa | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| m_hca | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| g_o | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| d_g | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_h_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| c_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| k_I | —  _(해당 없음: 이 모델은 `v4_compress` 계열 구조를 쓰지 않음)_ |
| n_hc | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| t_sinkhorn | —  _(해당 없음: 이 모델은 `mhc` 계열 구조를 쓰지 않음)_ |
| n_attn_res_block | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
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

shape 축 **4,670개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 1,739 | 37.24% |
| 스코프 없는 심볼 | 1,201 | 25.72% |
| 이 모듈 스코프의 심볼 | 1,093 | 23.40% |
| 이 모듈 스코프의 유도식 | 559 | 11.97% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 70 | 1.50% |
| 이름 없음 (정수 유지) | 8 | 0.17% |

등록된 규칙 **4,592축**, 약한 근거 70축, 휴리스틱 **0축 (0.0%)**, 이름 없음 8축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 2 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |

## 레이어 구조

- layer 0-1: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 2 == 2 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=16 in 2/2 layers |
| C6 | PASS | hidden_size=16 (heuristic check, 102 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=32000, tie_word_embeddings=False |
| C10 | PASS | all 21 params covered |
| C11 | PASS | 9 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 129 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'a... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `hf-internal-testing/tiny-random-LlamaForCausalLM` config.json @ `9fb191250dd56d0ba7ec9785a025ed29c03d5998` (sha256 `c640311e2c38…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- hf-internal-testing/tiny-random-LlamaForCausalLM @ 9fb191250dd56d0ba7ec9785a025ed29c03d5998

C1   PASS   2 == 2
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=16 in 2/2 layers
C6   PASS   hidden_size=16 (heuristic check, 102 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=32000, tie_word_embeddings=False
C10  PASS   all 21 params covered
C11  PASS   9 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   129 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default']
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
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
  model.layers.N.self_attn                           transpose        [B,n_kv,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,T,d_head] -> [n_kv,T,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_kv,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.norm                                         pow              [B,T,d_model] -> [B,T,d_model]
  model.norm                                         mean             [B,T,d_model] -> [B,T,1]
  model.norm                                         elementwise_add  [B,T,1] -> [B,T,1]
  model.norm                                         rsqrt            [B,T,1] -> [B,T,1]
  model.norm                                         elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
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
  model.layers.N.self_attn                           transpose        [B,n_kv,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_kv,T+1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           view             [B,n_kv,T+1,d_head] -> [n_kv,T+1,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_kv,T+1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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
  model.norm                                         pow              [B,1,d_model] -> [B,1,d_model]
  model.norm                                         mean             [B,1,d_model] -> [B,1,1]
  model.norm                                         elementwise_add  [B,1,1] -> [B,1,1]
  model.norm                                         rsqrt            [B,1,1] -> [B,1,1]
  model.norm                                         elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.norm                                         elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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

