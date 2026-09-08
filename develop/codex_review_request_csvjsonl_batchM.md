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


# 배치 M — 대상 모델: meta-llama__Llama-3.1-8B, meta-llama__Llama-3.1-70B, microsoft__Phi-4, tiiuae__falcon-7b


# ============================================================
# 모델: meta-llama__Llama-3.1-8B
# ============================================================

# 리뷰 패킷 — meta-llama/Llama-3.1-8B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` / 트레이스 seq_len(T) = 16
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
  d_model      = 4096
  n_h          = 32
  n_kv         = 8
  d_head       = 128
  d_ff         = 14336
  d_shared     = None
  V            = 128256
  ctx          = 131072
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

# Model Summary -- meta-llama/Llama-3.1-8B

## 기본 정보

- revision: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 8.03B total (dense) |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings; llama3 스케일(원본 8192×8.0) — 벤더 광고 컨텍스트와 다를 수 있음)_ |
| 3 | DATE | 2024-07-14  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 32× GQA |
| 7 | KV CACHE / TOKEN (BF16) | 128.0 KiB (Moderate) |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `llama` |
| attention | GQA — 32 query : 8 kv heads (repeat 4), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=500000.0), llama3 scaling |
| FFN | dense FFN — intermediate 14336, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; 32 attention layer(s) ⇒ 65536 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 32 |
| d_model | 4096 |
| n_h | 32 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 14336 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 128256 |
| ctx | 131072 |
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

shape 축 **72,908개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 23,977 | 32.89% |
| 이 모듈 스코프의 심볼 | 20,311 | 27.86% |
| 스코프 없는 심볼 | 18,189 | 24.95% |
| 이 모듈 스코프의 유도식 | 9,077 | 12.45% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,226 | 1.68% |
| 이름 없음 (정수 유지) | 128 | 0.18% |

등록된 규칙 **71,554축**, 약한 근거 1,226축, 휴리스틱 **0축 (0.0%)**, 이름 없음 128축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 4 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |

## 레이어 구조

- layer 0-31: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 32 == 32 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 2144 flagged) |
| C7 | PASS | GQA 32:8 (repeat factor 4) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=128256, tie_word_embeddings=False |
| C10 | PASS | all 291 params covered |
| C11 | PASS | 129 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1991 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `meta-llama/Llama-3.1-8B` config.json @ `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (sha256 `ba80fc9821ca…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- meta-llama/Llama-3.1-8B @ d04e592bb4f6aa9cfee91e2e20afa771667e1d4b

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers
C6   PASS   hidden_size=4096 (heuristic check, 2144 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=128256, tie_word_embeddings=False
C10  PASS   all 291 params covered
C11  PASS   129 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1991 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default']
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
  model.rotary_emb                                   _to_copy         [B,T,d_head] -> [B,T,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
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
  model.rotary_emb                                   _to_copy         [B,1,d_head] -> [B,1,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
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
# 모델: meta-llama__Llama-3.1-70B
# ============================================================

# 리뷰 패킷 — meta-llama/Llama-3.1-70B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `349b2ddb53ce8f2849a6c168a81980ab25258dac` / 트레이스 seq_len(T) = 16
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
  L            = 80
  d_model      = 8192
  n_h          = 64
  n_kv         = 8
  d_head       = 128
  d_ff         = 28672
  d_shared     = None
  V            = 128256
  ctx          = 131072
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

# Model Summary -- meta-llama/Llama-3.1-70B

## 기본 정보

- revision: `349b2ddb53ce8f2849a6c168a81980ab25258dac`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 70.55B total (dense) |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings; llama3 스케일(원본 8192×8.0) — 벤더 광고 컨텍스트와 다를 수 있음)_ |
| 3 | DATE | 2024-07-14  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 80× GQA |
| 7 | KV CACHE / TOKEN (BF16) | 320.0 KiB (Very high) |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `llama` |
| attention | GQA — 64 query : 8 kv heads (repeat 8), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=500000.0), llama3 scaling |
| FFN | dense FFN — intermediate 28672, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; 80 attention layer(s) ⇒ 163840 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 80 |
| d_model | 8192 |
| n_h | 64 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 28672 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 128256 |
| ctx | 131072 |
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

shape 축 **180,860개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 59,161 | 32.71% |
| 이 모듈 스코프의 심볼 | 50,695 | 28.03% |
| 스코프 없는 심볼 | 45,069 | 24.92% |
| 이 모듈 스코프의 유도식 | 22,565 | 12.48% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,050 | 1.69% |
| 이름 없음 (정수 유지) | 320 | 0.18% |

등록된 규칙 **177,490축**, 약한 근거 3,050축, 휴리스틱 **0축 (0.0%)**, 이름 없음 320축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 1024 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |

## 레이어 구조

- layer 0-79: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 80 == 80 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=8192 in 80/80 layers |
| C6 | PASS | hidden_size=8192 (heuristic check, 5360 flagged) |
| C7 | PASS | GQA 64:8 (repeat factor 8) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=128256, tie_word_embeddings=False |
| C10 | PASS | all 723 params covered |
| C11 | PASS | 321 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 4919 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `meta-llama/Llama-3.1-70B` config.json @ `349b2ddb53ce8f2849a6c168a81980ab25258dac` (sha256 `e5d40b5959f2…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- meta-llama/Llama-3.1-70B @ 349b2ddb53ce8f2849a6c168a81980ab25258dac

C1   PASS   80 == 80
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=8192 in 80/80 layers
C6   PASS   hidden_size=8192 (heuristic check, 5360 flagged)
C7   PASS   GQA 64:8 (repeat factor 8)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=128256, tie_word_embeddings=False
C10  PASS   all 723 params covered
C11  PASS   321 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4919 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default']
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
  model.rotary_emb                                   _to_copy         [B,T,d_head] -> [B,T,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
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
  model.rotary_emb                                   _to_copy         [B,1,d_head] -> [B,1,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
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
# 모델: microsoft__Phi-4
# ============================================================

# 리뷰 패킷 — microsoft/Phi-4

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` / 트레이스 seq_len(T) = 16
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
  L            = 40
  d_model      = 5120
  n_h          = 40
  n_kv         = 10
  d_head       = 128
  d_ff         = 17920
  d_shared     = None
  V            = 100352
  ctx          = 16384
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

# Model Summary -- microsoft/Phi-4

## 기본 정보

- revision: `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 14.66B total (dense) |
| 2 | Context (tokens) | 16,384  _(config max_position_embeddings)_ |
| 3 | DATE | 2024-12-11  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 40× GQA |
| 7 | KV CACHE / TOKEN (BF16) | 200.0 KiB (High) |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `phi3` |
| attention | GQA — 40 query : 10 kv heads (repeat 4), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=250000) |
| FFN | dense FFN — intermediate 17920, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·10·128 = 2560 elems / token / layer; 40 attention layer(s) ⇒ 102400 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 40 |
| d_model | 5120 |
| n_h | 40 |
| n_kv | 10 |
| d_head | 128 |
| d_ff | 17920 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 100352 |
| ctx | 16384 |
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

shape 축 **93,168개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 32,081 | 34.43% |
| 이 모듈 스코프의 심볼 | 26,891 | 28.86% |
| 스코프 없는 심볼 | 22,267 | 23.90% |
| 이 모듈 스코프의 유도식 | 9,759 | 10.47% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,530 | 1.64% |
| 이름 없음 (정수 유지) | 640 | 0.69% |

등록된 규칙 **90,998축**, 약한 근거 1,530축, 휴리스틱 **0축 (0.0%)**, 이름 없음 640축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 4 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 1280 | n_kv·d_head (KV 투영 폭) | self_attn |
| 7680 | (n_h + 2·n_kv)·d_head (fused QKV 투영 폭 — Q·K·V 한 행렬) | qkv_proj, self_attn |
| 35840 | 2·d_ff (dense FFN gate+up 융합 투영 폭) | gate_up_proj, mlp |

## 레이어 구조

- layer 0-39: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 40 == 40 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=5120 in 40/40 layers |
| C6 | PASS | hidden_size=5120 (heuristic check, 2800 flagged) |
| C7 | PASS | GQA 40:10 (repeat factor 4) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=100352, tie_word_embeddings=False |
| C10 | PASS | all 243 params covered |
| C11 | PASS | 241 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 2439 unmapped rows, 15 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `microsoft/Phi-4` config.json @ `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` (sha256 `6a2d84f6511b…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- microsoft/Phi-4 @ 2db69c1c3e91a05d2c64a3185acfbaf36f744e25

C1   PASS   40 == 40
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 40/40 layers
C6   PASS   hidden_size=5120 (heuristic check, 2800 flagged)
C7   PASS   GQA 40:10 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=100352, tie_word_embeddings=False
C10  PASS   all 243 params covered
C11  PASS   241 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2439 unmapped rows, 15 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.split.Tensor']
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
  model.rotary_emb                                   _to_copy         [B,T,d_head] -> [B,T,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.input_layernorm                     mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn.qkv_proj                  t                [(n_h+2*n_kv)*d_head,d_model] -> w=[(n_h+2*n_kv)*d_head,d_model] [d_model,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn.qkv_proj                  view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.qkv_proj                  matmul           [T,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[(n_h+2*n_kv)*d_head,d_model] [T,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn.qkv_proj                  _unsafe_view     [T,(n_h+2*n_kv)*d_head] -> [B,T,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn                           slice            [B,T,(n_h+2*n_kv)*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           slice            [B,T,(n_h+2*n_kv)*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           alias            [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,0]
  model.layers.N.self_attn                           alias            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,0]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head]*[B,n_h,T,0] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,T,d_head]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,T,0] -> [B,n_kv,T,d_head]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.mlp.gate_up_proj                    t                [2*d_ff,d_model] -> w=[2*d_ff,d_model] [d_model,2*d_ff]
  model.layers.N.mlp.gate_up_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate_up_proj                    matmul           [T,d_model]*[d_model,2*d_ff] -> w=[2*d_ff,d_model] [T,2*d_ff]
  model.layers.N.mlp.gate_up_proj                    _unsafe_view     [T,2*d_ff] -> [B,T,2*d_ff]
  model.layers.N.mlp                                 split            [B,T,2*d_ff] -> [B,T,d_ff]*[B,T,d_ff]
  model.layers.N.mlp.activation_fn                   silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.rotary_emb                                   _to_copy         [B,1,d_head] -> [B,1,d_head]
  model.layers.N.input_layernorm                     _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.input_layernorm                     mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn.qkv_proj                  t                [(n_h+2*n_kv)*d_head,d_model] -> w=[(n_h+2*n_kv)*d_head,d_model] [d_model,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn.qkv_proj                  view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.qkv_proj                  matmul           [B,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[(n_h+2*n_kv)*d_head,d_model] [B,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn.qkv_proj                  _unsafe_view     [B,(n_h+2*n_kv)*d_head] -> [B,1,(n_h+2*n_kv)*d_head]
  model.layers.N.self_attn                           slice            [B,1,(n_h+2*n_kv)*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           slice            [B,1,(n_h+2*n_kv)*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           alias            [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,0]
  model.layers.N.self_attn                           alias            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,0]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head]*[B,n_h,1,0] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,1,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_head]*[B,n_kv,1,0] -> [B,n_kv,1,d_head]
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
  model.layers.N.self_attn.o_proj                    t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.o_proj                    view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.o_proj                    matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  model.layers.N.self_attn.o_proj                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.mlp.gate_up_proj                    t                [2*d_ff,d_model] -> w=[2*d_ff,d_model] [d_model,2*d_ff]
  model.layers.N.mlp.gate_up_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate_up_proj                    matmul           [B,d_model]*[d_model,2*d_ff] -> w=[2*d_ff,d_model] [B,2*d_ff]
  model.layers.N.mlp.gate_up_proj                    _unsafe_view     [B,2*d_ff] -> [B,1,2*d_ff]
  model.layers.N.mlp                                 split            [B,1,2*d_ff] -> [B,1,d_ff]*[B,1,d_ff]
  model.layers.N.mlp.activation_fn                   silu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp                                 elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.down_proj                       t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.mlp.down_proj                       view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.mlp.down_proj                       matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.mlp.down_proj                       _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
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
# 모델: tiiuae__falcon-7b
# ============================================================

# 리뷰 패킷 — tiiuae/falcon-7b

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `ec89142b67d748a1865ea4451372db8313ada0d8` / 트레이스 seq_len(T) = 16
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
  d_model      = 4544
  n_h          = 71
  n_kv         = 1
  d_head       = 64
  d_ff         = 18176
  d_shared     = None
  V            = 65024
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

# Model Summary -- tiiuae/falcon-7b

## 기본 정보

- revision: `ec89142b67d748a1865ea4451372db8313ada0d8`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 6.92B total (dense) |
| 2 | Context (tokens) | 2,048  _(config max_position_embeddings)_ |
| 3 | DATE | 2023-04-24  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MQA |
| 6 | LAYER MIX | 32× MQA |
| 7 | KV CACHE / TOKEN (BF16) | 8.0 KiB (Very low) |
| 8 | KEY DETAIL | MQA attention; dense FFN |
| 9 | Related concepts | LayerNorm, RoPE, MQA |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `falcon` |
| attention | MQA — 71 query heads, 1 kv head, d_head=64 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000.0) |
| FFN | dense FFN — intermediate 18176, GELU |
| 정규화 | LayerNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·1·64 = 128 elems / token / layer; 32 attention layer(s) ⇒ 4096 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 32 |
| d_model | 4544 |
| n_h | 71 |
| n_kv | 1 |
| d_head | 64 |
| d_ff | 18176 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 65024 |
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

shape 축 **52,402개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 19,801 | 37.79% |
| 이 모듈 스코프의 심볼 | 14,517 | 27.70% |
| 스코프 없는 심볼 | 11,675 | 22.28% |
| 이 모듈 스코프의 유도식 | 5,295 | 10.10% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 986 | 1.88% |
| 이름 없음 (정수 유지) | 128 | 0.24% |

등록된 규칙 **51,288축**, 약한 근거 986축, 휴리스틱 **0축 (0.0%)**, 이름 없음 128축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 32 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attention |
| 73 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | self_attention |
| 4672 | (n_h + 2·n_kv)·d_head (fused QKV 투영 폭 — Q·K·V 한 행렬) | query_key_value, self_attention |

## 레이어 구조

- layer 0-31: input_layernorm, mlp, self_attention

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 32 == 32 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4544 in 32/32 layers |
| C6 | PASS | hidden_size=4544 (heuristic check, 160 flagged) |
| C7 | PASS | MQA (71 query heads : 1 kv head) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=65024, tie_word_embeddings=True |
| C10 | PASS | all 195 params covered |
| C11 | PASS | 129 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1476 unmapped rows, 22 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `tiiuae/falcon-7b` config.json @ `ec89142b67d748a1865ea4451372db8313ada0d8` (sha256 `dfcdbab01746…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- tiiuae/falcon-7b @ ec89142b67d748a1865ea4451372db8313ada0d8

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4544 in 32/32 layers
C6   PASS   hidden_size=4544 (heuristic check, 160 flagged)
C7   PASS   MQA (71 query heads : 1 kv head)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=65024, tie_word_embeddings=True
C10  PASS   all 195 params covered
C11  PASS   129 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1476 unmapped rows, 22 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clone.default', 'aten.div_.Tensor', 'aten.expand.default', 'aten.index.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  transformer.word_embeddings                        embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  transformer                                        arange           [] -> [B]
  transformer                                        arange           [] -> [T]
  transformer                                        elementwise_add  [T] -> [T]
  transformer                                        new_ones         [T] -> [T]
  transformer                                        view             [T] -> [T,1]
  transformer                                        le               [T]*[T,1] -> [T,T]
  transformer                                        select           [T,T] -> [T]
  transformer                                        select           [T] -> []
  transformer                                        bitwise_and      [T,1]*[T,T] -> [T,T]
  transformer                                        lift_fresh       [] -> []
  transformer                                        _to_copy         [] -> []
  transformer                                        bitwise_and      [T,T]*[] -> [T,T]
  transformer                                        expand           [T,T] -> [B,T,T]
  transformer                                        expand           [B,T,T] -> [B,1,T,T]
  transformer                                        scalar_tensor    [] -> []
  transformer                                        where            [B,1,T,T]*[]*[] -> [B,1,T,T]
  transformer.rotary_emb                             unsqueeze        [d_head/2] -> [B,d_head/2]
  transformer.rotary_emb                             unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  transformer.rotary_emb                             expand           [B,d_head/2,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             unsqueeze        [B,T] -> [B,1,T]
  transformer.rotary_emb                             _to_copy         [B,1,T] -> [B,1,T]
  transformer.rotary_emb                             view             [B,d_head/2,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             expand           [B,1,T] -> [B,1,T]
  transformer.rotary_emb                             view             [B,1,T] -> [B,1,T]
  transformer.rotary_emb                             batched_matmul   [B,d_head/2,1]*[B,1,T] -> [B,d_head/2,T]
  transformer.rotary_emb                             _unsafe_view     [B,d_head/2,T] -> [B,d_head/2,T]
  transformer.rotary_emb                             transpose        [B,d_head/2,T] -> [B,T,d_head/2]
  transformer.rotary_emb                             concat           [B,T,d_head/2]*[B,T,d_head/2] -> [B,T,d_head]
  transformer.rotary_emb                             cos              [B,T,d_head] -> [B,T,d_head]
  transformer.rotary_emb                             elementwise_mul  [B,T,d_head] -> [B,T,d_head]
  transformer.rotary_emb                             sin              [B,T,d_head] -> [B,T,d_head]
  transformer.rotary_emb                             _to_copy         [B,T,d_head] -> [B,T,d_head]
  transformer.h.N.input_layernorm                    layernorm        [B,T,d_model]*[d_model]*[d_model] -> [B,T,d_model]*[B,T,1]*[B,T,1]
  transformer.h.N.self_attention.query_key_value     permute          [(n_h+2*n_kv)*d_head,d_model] -> w=[(n_h+2*n_kv)*d_head,d_model] [d_model,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention.query_key_value     view             [B,T,d_model] -> [T,d_model]
  transformer.h.N.self_attention.query_key_value     matmul           [T,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[(n_h+2*n_kv)*d_head,d_model] [T,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention.query_key_value     _unsafe_view     [T,(n_h+2*n_kv)*d_head] -> [B,T,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention                     view             [B,T,(n_h+2*n_kv)*d_head] -> [B,T,n_h+2*n_kv,d_head]
  transformer.h.N.self_attention                     slice            [B,T,n_h+2*n_kv,d_head] -> [B,T,n_h,d_head]
  transformer.h.N.self_attention                     index            [B,T,n_h+2*n_kv,d_head]*[B] -> [B,T,1,d_head]
  transformer.h.N.self_attention                     transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     view             [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     transpose        [B,T,1,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     view             [B,1,T,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  transformer.h.N.self_attention                     neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  transformer.h.N.self_attention                     concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     elementwise_mul  [B,1,T,d_head]*[B,1,T,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     slice            [B,1,T,d_head] -> [B,1,T,d_head/2]
  transformer.h.N.self_attention                     neg              [B,1,T,d_head/2] -> [B,1,T,d_head/2]
  transformer.h.N.self_attention                     concat           [B,1,T,d_head/2]*[B,1,T,d_head/2] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     elementwise_add  [B,1,T,d_head]*[B,1,T,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     concat           [0]*[B,1,T,d_head] -> [B,1,T,d_head]
  transformer.h.N.self_attention                     transpose        [B,1,T,d_head] -> [B,1,d_head,T]
  transformer.h.N.self_attention                     expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  transformer.h.N.self_attention                     expand           [B,1,d_head,T] -> [B,n_h,d_head,T]
  transformer.h.N.self_attention                     view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  transformer.h.N.self_attention                     batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  transformer.h.N.self_attention                     _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.self_attention                     div_             [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.self_attention                     elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  transformer.h.N.self_attention                     softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.self_attention                     expand           [B,n_h,T,T] -> [B,n_h,T,T]
  transformer.h.N.self_attention                     view             [B,n_h,T,T] -> [n_h,T,T]
  transformer.h.N.self_attention                     expand           [B,1,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  transformer.h.N.self_attention                     _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  transformer.h.N.self_attention                     permute          [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  transformer.h.N.self_attention                     clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  transformer.h.N.self_attention                     _unsafe_view     [B,T,n_h,d_head] -> [B,T,n_h*d_head]
  transformer.h.N.self_attention.dense               permute          [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  transformer.h.N.self_attention.dense               view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  transformer.h.N.self_attention.dense               matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  transformer.h.N.self_attention.dense               _unsafe_view     [T,d_model] -> [B,T,d_model]
  transformer.h.N.mlp.dense_h_to_4h                  permute          [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  transformer.h.N.mlp.dense_h_to_4h                  view             [B,T,d_model] -> [T,d_model]
  transformer.h.N.mlp.dense_h_to_4h                  matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  transformer.h.N.mlp.dense_h_to_4h                  _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.act                            gelu             [B,T,d_ff] -> [B,T,d_ff]
  transformer.h.N.mlp.dense_4h_to_h                  permute          [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  transformer.h.N.mlp.dense_4h_to_h                  view             [B,T,d_ff] -> [T,d_ff]
  transformer.h.N.mlp.dense_4h_to_h                  matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  transformer.h.N.mlp.dense_4h_to_h                  _unsafe_view     [T,d_model] -> [B,T,d_model]
  transformer.h.0                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.0                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.1                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.1                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.2                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.2                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.3                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.3                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.4                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.4                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.5                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.5                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.6                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.6                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.7                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.7                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.8                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.8                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.9                                    add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.9                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.10                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.10                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.11                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.11                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.12                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.12                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.13                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.13                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.14                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.14                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.15                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.15                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.16                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.16                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.17                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.17                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.18                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.18                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.19                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.19                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.20                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.20                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.21                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.21                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.22                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.22                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.23                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.23                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.24                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.24                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.25                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.25                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.26                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.26                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.27                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.27                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.28                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.28                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.29                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.29                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.30                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.30                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.31                                   add_             [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.h.31                                   elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  transformer.ln_f                                   layernorm        [B,T,d_model]*[d_model]*[d_model] -> [B,T,d_model]*[B,T,1]*[B,T,1]
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
  transformer.word_embeddings                        embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  transformer                                        arange           [] -> [B]
  transformer                                        elementwise_add  [B] -> [B]
  transformer                                        arange           [] -> [T+1]
  transformer                                        elementwise_add  [T+1] -> [T+1]
  transformer                                        new_ones         [B] -> [B]
  transformer                                        view             [B] -> [B,1]
  transformer                                        le               [T+1]*[B,1] -> [B,T+1]
  transformer                                        select           [B,T+1] -> [B]
  transformer                                        select           [B] -> []
  transformer                                        bitwise_and      [B,1]*[B,T+1] -> [B,T+1]
  transformer                                        lift_fresh       [] -> []
  transformer                                        _to_copy         [] -> []
  transformer                                        bitwise_and      [B,T+1]*[] -> [B,T+1]
  transformer                                        expand           [B,T+1] -> [B,1,T+1]
  transformer                                        expand           [B,1,T+1] -> [B,1,1,T+1]
  transformer                                        scalar_tensor    [] -> []
  transformer                                        where            [B,1,1,T+1]*[]*[] -> [B,1,1,T+1]
  transformer.rotary_emb                             unsqueeze        [d_head/2] -> [B,d_head/2]
  transformer.rotary_emb                             unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  transformer.rotary_emb                             expand           [B,d_head/2,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             unsqueeze        [B,1] -> [B,1,1]
  transformer.rotary_emb                             _to_copy         [B,1,1] -> [B,1,1]
  transformer.rotary_emb                             view             [B,d_head/2,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             expand           [B,1,1] -> [B,1,1]
  transformer.rotary_emb                             view             [B,1,1] -> [B,1,1]
  transformer.rotary_emb                             batched_matmul   [B,d_head/2,1]*[B,1,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             _unsafe_view     [B,d_head/2,1] -> [B,d_head/2,1]
  transformer.rotary_emb                             transpose        [B,d_head/2,1] -> [B,1,d_head/2]
  transformer.rotary_emb                             concat           [B,1,d_head/2]*[B,1,d_head/2] -> [B,1,d_head]
  transformer.rotary_emb                             cos              [B,1,d_head] -> [B,1,d_head]
  transformer.rotary_emb                             elementwise_mul  [B,1,d_head] -> [B,1,d_head]
  transformer.rotary_emb                             sin              [B,1,d_head] -> [B,1,d_head]
  transformer.rotary_emb                             _to_copy         [B,1,d_head] -> [B,1,d_head]
  transformer.h.N.input_layernorm                    layernorm        [B,1,d_model]*[d_model]*[d_model] -> [B,1,d_model]*[B,1,1]*[B,1,1]
  transformer.h.N.self_attention.query_key_value     permute          [(n_h+2*n_kv)*d_head,d_model] -> w=[(n_h+2*n_kv)*d_head,d_model] [d_model,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention.query_key_value     view             [B,1,d_model] -> [B,d_model]
  transformer.h.N.self_attention.query_key_value     matmul           [B,d_model]*[d_model,(n_h+2*n_kv)*d_head] -> w=[(n_h+2*n_kv)*d_head,d_model] [B,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention.query_key_value     _unsafe_view     [B,(n_h+2*n_kv)*d_head] -> [B,1,(n_h+2*n_kv)*d_head]
  transformer.h.N.self_attention                     view             [B,1,(n_h+2*n_kv)*d_head] -> [B,1,n_h+2*n_kv,d_head]
  transformer.h.N.self_attention                     slice            [B,1,n_h+2*n_kv,d_head] -> [B,1,n_h,d_head]
  transformer.h.N.self_attention                     index            [B,1,n_h+2*n_kv,d_head]*[B] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     view             [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     transpose        [B,1,1,d_head] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     view             [B,1,1,d_head] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  transformer.h.N.self_attention                     neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  transformer.h.N.self_attention                     concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     elementwise_mul  [B,1,1,d_head]*[B,1,1,d_head] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     slice            [B,1,1,d_head] -> [B,1,1,d_head/2]
  transformer.h.N.self_attention                     neg              [B,1,1,d_head/2] -> [B,1,1,d_head/2]
  transformer.h.N.self_attention                     concat           [B,1,1,d_head/2]*[B,1,1,d_head/2] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     elementwise_add  [B,1,1,d_head]*[B,1,1,d_head] -> [B,1,1,d_head]
  transformer.h.N.self_attention                     concat           [B,1,T,d_head]*[B,1,1,d_head] -> [B,1,T+1,d_head]
  transformer.h.N.self_attention                     transpose        [B,1,T+1,d_head] -> [B,1,d_head,T+1]
  transformer.h.N.self_attention                     expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  transformer.h.N.self_attention                     expand           [B,1,d_head,T+1] -> [B,n_h,d_head,T+1]
  transformer.h.N.self_attention                     view             [B,n_h,d_head,T+1] -> [n_h,d_head,T+1]
  transformer.h.N.self_attention                     batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  transformer.h.N.self_attention                     _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.self_attention                     div_             [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.self_attention                     elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.self_attention                     softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.self_attention                     expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  transformer.h.N.self_attention                     view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  transformer.h.N.self_attention                     expand           [B,1,T+1,d_head] -> [B,n_h,T+1,d_head]
  transformer.h.N.self_attention                     batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
  transformer.h.N.self_attention                     _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  transformer.h.N.self_attention                     permute          [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  transformer.h.N.self_attention.dense               permute          [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  transformer.h.N.self_attention.dense               view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  transformer.h.N.self_attention.dense               matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  transformer.h.N.self_attention.dense               _unsafe_view     [B,d_model] -> [B,1,d_model]
  transformer.h.N.mlp.dense_h_to_4h                  permute          [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  transformer.h.N.mlp.dense_h_to_4h                  view             [B,1,d_model] -> [B,d_model]
  transformer.h.N.mlp.dense_h_to_4h                  matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  transformer.h.N.mlp.dense_h_to_4h                  _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.act                            gelu             [B,1,d_ff] -> [B,1,d_ff]
  transformer.h.N.mlp.dense_4h_to_h                  permute          [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  transformer.h.N.mlp.dense_4h_to_h                  view             [B,1,d_ff] -> [B,d_ff]
  transformer.h.N.mlp.dense_4h_to_h                  matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  transformer.h.N.mlp.dense_4h_to_h                  _unsafe_view     [B,d_model] -> [B,1,d_model]
  transformer.h.0                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.0                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.1                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.1                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.2                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.2                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.3                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.3                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.4                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.4                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.5                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.5                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.6                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.6                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.7                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.7                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.8                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.8                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.9                                    add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.9                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.10                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.10                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.11                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.11                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.12                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.12                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.13                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.13                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.14                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.14                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.15                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.15                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.16                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.16                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.17                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.17                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.18                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.18                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.19                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.19                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.20                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.20                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.21                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.21                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.22                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.22                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.23                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.23                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.24                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.24                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.25                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.25                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.26                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.26                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.27                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.27                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.28                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.28                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.29                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.29                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.30                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.30                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.31                                   add_             [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.h.31                                   elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  transformer.ln_f                                   layernorm        [B,1,d_model]*[d_model]*[d_model] -> [B,1,d_model]*[B,1,1]*[B,1,1]
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

