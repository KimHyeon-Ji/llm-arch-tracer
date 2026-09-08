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


# 배치 A — 대상 모델: MiniMaxAI__MiniMax-M2, NX-AI__xLSTM-7b, Qwen__Qwen3-Next-80B-A3B-Instruct, allenai__OLMo-2-1124-7B-Instruct


# ============================================================
# 모델: MiniMaxAI__MiniMax-M2
# ============================================================

# 리뷰 패킷 — MiniMaxAI/MiniMax-M2

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `757303d492a50514c312788b5247a4f696a4c6a3` / 트레이스 seq_len(T) = 16
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
  L            = 62
  d_model      = 3072
  n_h          = 48
  n_kv         = 8
  d_head       = 128
  d_ff         = 1536
  d_shared     = 0
  V            = 200064
  ctx          = 196608
  E            = 256
  E_shared     = 0
  k            = 8
  n_grp        = None
  k_grp        = None
  d_moe        = 1536
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

# Model Summary -- MiniMaxAI/MiniMax-M2

## 기본 정보

- revision: `757303d492a50514c312788b5247a4f696a4c6a3`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 228.69B total, 11.03B active (4.8% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 196,608  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-10-22  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 62× GQA  (FFN: 62× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 248.0 KiB (High) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=256, top-8, sigmoid gating/aux-loss-free) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, sigmoid-gating, QK-Norm |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `minimax_m2` |
| attention | GQA — 48 query : 8 kv heads (repeat 6), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=5000000) |
| FFN | MoE — 256 routed experts, top-8, expert intermediate 1536, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·8·128 = 2048 elems / token / layer; 62 attention layer(s) ⇒ 126976 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 62 |
| d_model | 3072 |
| n_h | 48 |
| n_kv | 8 |
| d_head | 128 |
| d_ff | 1536 |
| d_shared | 0 |
| V | 200064 |
| ctx | 196608 |
| E | 256 |
| E_shared | 0 |
| k | 8 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 1536 |
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

shape 축 **191,784개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 60,351 | 31.47% |
| 이 모듈 스코프의 심볼 | 52,323 | 27.28% |
| 스코프 없는 심볼 | 50,993 | 26.59% |
| 이 모듈 스코프의 유도식 | 23,827 | 12.42% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,298 | 1.72% |
| 이름 없음 (정수 유지) | 992 | 0.52% |

등록된 규칙 **187,494축**, 약한 근거 3,298축, 휴리스틱 **0축 (0.0%)**, 이름 없음 992축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 6 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 64 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | rotary_emb, self_attn |
| 1024 | n_kv·d_head (KV 투영 폭) | k_norm, k_proj, self_attn, v_proj |
| 6144 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_norm, q_proj, self_attn |

## 레이어 구조

- layer 0-61: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 62 == 62 |
| C2 | PASS | 1 clusters == 1 from config schedule ['attn_type_list'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=3072 in 62/62 layers |
| C6 | PASS | hidden_size=3072 (heuristic check, 6944 flagged) |
| C7 | PASS | GQA 48:8 (repeat factor 6) |
| C8 | WARN | MoE trace-verified [router_dim(E=256):ok, top_k(8):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=200064, tie_word_embeddings=False |
| C10 | PASS | all 685 params covered |
| C11 | PASS | 373 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 5681 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `MiniMaxAI/MiniMax-M2` config.json @ `757303d492a50514c312788b5247a4f696a4c6a3` (sha256 `0bed4c1d99bb…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- MiniMaxAI/MiniMax-M2 @ 757303d492a50514c312788b5247a4f696a4c6a3

C1   PASS   62 == 62
C2   PASS   1 clusters == 1 from config schedule ['attn_type_list']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=3072 in 62/62 layers
C6   PASS   hidden_size=3072 (heuristic check, 6944 flagged)
C7   PASS   GQA 48:8 (repeat factor 6)
C8   WARN   MoE trace-verified [router_dim(E=256):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=200064, tie_word_embeddings=False
C10  PASS   all 685 params covered
C11  PASS   373 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   5681 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default']
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
  model.layers.N.self_attn.q_norm                    _to_copy         [B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,T,n_h*d_head] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,T,n_h*d_head]*[B,T,1] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [n_h*d_head]*[B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,T,n_kv*d_head] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,T,n_kv*d_head]*[B,T,1] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [n_kv*d_head]*[B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
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
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,T,d_head]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head]*[B,n_h,T,0] -> [B,n_h,T,d_head]
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
  model.layers.N.mlp                                 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            view             [T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mlp.gate                            _to_copy         [T,E] -> [T,E]
  model.layers.N.mlp.gate                            sigmoid          [T,E] -> [T,E]
  model.layers.N.mlp.gate                            elementwise_add  [T,E]*[E] -> [T,E]
  model.layers.N.mlp.gate                            topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mlp.gate                            gather           [T,E]*[T,k] -> [T,k]
  model.layers.N.mlp.gate                            sum              [T,k] -> [T,1]
  model.layers.N.mlp.gate                            div_             [T,k]*[T,1] -> [T,k]
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
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_model] -> w=[E,d_model,d_model] [E,d_model,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_model]*[E,d_model,d_model]*[E] -> w=[E,d_model,d_model] [k*T,2*d_moe]
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
  model.layers.N.self_attn.q_norm                    _to_copy         [B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,1,n_h*d_head] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,1,n_h*d_head]*[B,1,1] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [n_h*d_head]*[B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,1,n_kv*d_head] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,n_kv*d_head]*[B,1,1] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [n_kv*d_head]*[B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
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
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_kv,1,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head]*[B,n_h,1,0] -> [B,n_h,1,d_head]
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
  model.layers.N.mlp                                 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            view             [B,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mlp.gate                            _to_copy         [B,E] -> [B,E]
  model.layers.N.mlp.gate                            sigmoid          [B,E] -> [B,E]
  model.layers.N.mlp.gate                            elementwise_add  [B,E]*[E] -> [B,E]
  model.layers.N.mlp.gate                            topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mlp.gate                            gather           [B,E]*[B,k] -> [B,k]
  model.layers.N.mlp.gate                            sum              [B,k] -> [B,1]
  model.layers.N.mlp.gate                            div_             [B,k]*[B,1] -> [B,k]
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
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_model] -> w=[E,d_model,d_model] [E,d_model,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_model]*[E,d_model,d_model]*[E] -> w=[E,d_model,d_model] [k,2*d_moe]
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
# 모델: NX-AI__xLSTM-7b
# ============================================================

# 리뷰 패킷 — NX-AI/xLSTM-7b

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `9dc507bd0939cf372a4a4f667335651d8e49dddb` / 트레이스 seq_len(T) = 16
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
  n_h          = 8
  n_kv         = 8
  d_head       = 512
  d_ff         = None
  d_shared     = None
  V            = 50304
  ctx          = None
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
  d_v          = 512
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
  d_chunk      = 64
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

# Model Summary -- NX-AI/xLSTM-7b

## 기본 정보

- revision: `9dc507bd0939cf372a4a4f667335651d8e49dddb`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 6.87B total (dense) |
| 2 | Context (tokens) | ?  _(config max_position_embeddings)_ |
| 3 | DATE | 2024-12-11  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | attention-free |
| 6 | LAYER MIX | 32× attention-free |
| 7 | KV CACHE / TOKEN (BF16) | N/A — recurrent/SSM state, not KV cache |
| 8 | KEY DETAIL | attention-free (recurrent/mLSTM or SSM); dense FFN |
| 9 | Related concepts | RMSNorm, attention-free |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `xlstm` |
| attention | MHA — 8 heads (no GQA/MQA), d_head=512 |
| attention 커널 | ? (no softmax/sdpa op — non-softmax attention?) |
| 위치 인코딩 | none observed (NoPE, or position handled implicitly) |
| FFN | dense FFN — intermediate None, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | recurrent/SSM state (no KV cache) |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 32 |
| d_model | 4096 |
| n_h | 8 |
| n_kv | 8 |
| d_head | 512 |
| d_ff | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 50304 |
| ctx | _(미확인 -- config 별칭 없음, Tier 2 대상)_ |
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
| d_v | 512 |
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
| d_chunk | 64 |
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

shape 축 **519,911개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 237,112 | 45.61% |
| 이 모듈 스코프의 심볼 | 183,970 | 35.38% |
| 이 모듈 스코프의 유도식 | 68,330 | 13.14% |
| 스코프 없는 심볼 | 27,133 | 5.22% |
| 이름 없음 (정수 유지) | 1,920 | 0.37% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,062 | 0.20% |
| 스코프가 배제한 심볼 | 384 | 0.07% |

등록된 규칙 **516,545축**, 약한 근거 1,446축, 휴리스틱 **0축 (0.0%)**, 이름 없음 1,920축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 256 | d_head/2 (RoPE rotate_half 분할 축) | backbone, mlstm_backend, mlstm_layer |
| 2048 | d_model·qk_dim_factor (xLSTM q/k 투영 폭) | k, mlstm_layer, q |
| 10944 | round_up(d_model·ffn_proj_factor, ffn_round_up_to_multiple_of) (xLSTM FFN 폭) | act_fn, ffn, proj_down, proj_up, proj_up_gate |

## 레이어 구조

- layer 0-31: ffn, mlstm_layer, norm_ffn, norm_mlstm

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 32 == 32 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 0 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=50304, tie_word_embeddings=False |
| C10 | PASS | all 483 params covered |
| C11 | WARN | no concat/cache-touching op found in decode trace -- verify cache is actually being reused |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 22501 unmapped rows, 20 distinct raw ops: ['aten._unsafe_view.default', 'aten.abs.default', 'aten... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `NX-AI/xLSTM-7b` config.json @ `9dc507bd0939cf372a4a4f667335651d8e49dddb` (sha256 `de4625828325…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- NX-AI/xLSTM-7b @ 9dc507bd0939cf372a4a4f667335651d8e49dddb

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers
C6   PASS   hidden_size=4096 (heuristic check, 0 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=50304, tie_word_embeddings=False
C10  PASS   all 483 params covered
C11  WARN   no concat/cache-touching op found in decode trace -- verify cache is actually being reused
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   22501 unmapped rows, 20 distinct raw ops: ['aten._unsafe_view.default', 'aten.abs.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.clone.default', 'aten.copy_.default', 'aten.div.Tensor', 'aten.expand.default', 'aten.log_sigmoid_forward.default', 'aten.maximum.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  backbone.embeddings                                embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  backbone                                           zeros            [] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone                                           zeros            [] -> [B,n_h,d_model*qk_f/n_h]
  backbone                                           zeros            [] -> [B,n_h,1]
  backbone.blocks.N.norm_mlstm                       pow              [B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.norm_mlstm                       mean             [B,T,d_model] -> [B,T,1]
  backbone.blocks.N.norm_mlstm                       elementwise_add  [B,T,1] -> [B,T,1]
  backbone.blocks.N.norm_mlstm                       rsqrt            [B,T,1] -> [B,T,1]
  backbone.blocks.N.norm_mlstm                       elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  backbone.blocks.N.norm_mlstm                       elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.q                    t                [d_model*qk_f,d_model] -> w=[d_model*qk_f,d_model] [d_model,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.q                    view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.q                    matmul           [T,d_model]*[d_model,d_model*qk_f] -> w=[d_model*qk_f,d_model] [T,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.q                    _unsafe_view     [T,d_model*qk_f] -> [B,T,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    t                [d_model*qk_f,d_model] -> w=[d_model*qk_f,d_model] [d_model,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.k                    matmul           [T,d_model]*[d_model,d_model*qk_f] -> w=[d_model*qk_f,d_model] [T,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    _unsafe_view     [T,d_model*qk_f] -> [B,T,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.v                    t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.v                    view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.v                    matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  backbone.blocks.N.mlstm_layer.v                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         _unsafe_view     [T,d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.igate_preact         view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.igate_preact         t                [n_h,d_model] -> w=[n_h,d_model] [d_model,n_h]
  backbone.blocks.N.mlstm_layer.igate_preact         linear           [n_h]*[T,d_model]*[d_model,n_h] -> w=[n_h,d_model] [T,n_h]
  backbone.blocks.N.mlstm_layer.igate_preact         view             [T,n_h] -> [B,T,n_h]
  backbone.blocks.N.mlstm_layer                      div              [B,T,n_h] -> [B,T,n_h]
  backbone.blocks.N.mlstm_layer                      tanh             [B,T,n_h] -> [B,T,n_h]
  backbone.blocks.N.mlstm_layer                      elementwise_mul  [B,T,n_h] -> [B,T,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.fgate_preact         t                [n_h,d_model] -> w=[n_h,d_model] [d_model,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         linear           [n_h]*[T,d_model]*[d_model,n_h] -> w=[n_h,d_model] [T,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         view             [T,n_h] -> [B,T,n_h]
  backbone.blocks.N.mlstm_layer                      view             [B,T,d_model*qk_f] -> [B,T,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer                      transpose        [B,T,n_h,d_model*qk_f/n_h] -> [B,n_h,T,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer                      view             [B,T,d_model] -> [B,T,n_h,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,T,n_h] -> [B,n_h,T]
  backbone.blocks.N.mlstm_layer.mlstm_backend        alias            [B,n_h,T,d_model*qk_f/n_h] -> [B,n_h,T,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,T,d_model*qk_f/n_h] -> [B,n_h,T,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        alias            [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        alias            [B,n_h,T] -> [B,n_h,T]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,T] -> [B,n_h,T]
  backbone.blocks.N.mlstm_layer.mlstm_backend        select           [B,n_h,T] -> [B,n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        select           [B,n_h,T,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        select           [B,n_h,T,d_head] -> [B,n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        log_sigmoid_forward [B,n_h,1] -> [B,n_h,1]*[B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        maximum          [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        sub              [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        exp              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,1] -> [B,n_h,1,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,1,1]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,d_model*qk_f/n_h,1] -> [B,n_h,d_model*qk_f/n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,d_model*qk_f/n_h,1] -> [n_h,d_model*qk_f/n_h,B]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,d_model*qk_f/n_h,B]*[n_h,B,d_head] -> [n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,d_model*qk_f/n_h,d_head]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,1]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,d_model*qk_f/n_h]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_model*qk_f/n_h] -> [B,n_h,1,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,1,d_model*qk_f/n_h] -> [B,n_h,1,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,1,d_model*qk_f/n_h] -> [n_h,B,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,d_model*qk_f/n_h,d_head] -> [n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,B,d_model*qk_f/n_h]*[n_h,d_model*qk_f/n_h,d_head] -> [n_h,B,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        squeeze          [B,n_h,1,d_head] -> [B,n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,B,d_model*qk_f/n_h]*[n_h,d_model*qk_f/n_h,B] -> [n_h,B,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,B,1] -> [B,n_h,1,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        squeeze          [B,n_h,1,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        neg              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        abs              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        div              [B,n_h,d_head]*[B,n_h,1] -> [B,n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        stack            [B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head]*[B,n_h,d_head] -> [B,n_h,T,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        concat           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       mean             [B,T,n_h,d_head] -> [B,T,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       sub              [B,T,n_h,d_head]*[B,T,n_h,1] -> [B,T,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       var              [B,T,n_h,d_head] -> [B,T,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_add  [B,T,n_h,1] -> [B,T,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       rsqrt            [B,T,n_h,1] -> [B,T,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_mul  [B,T,n_h,d_head]*[B,T,n_h,1] -> [B,T,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       _unsafe_view     [B,T,n_h,d_head] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer                      view             [B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.ogate_act_fn         sigmoid          [B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer                      elementwise_mul  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             matmul           [T,d_model]*[d_model,d_model] -> w=[d_model,d_model] [T,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             _unsafe_view     [T,d_model] -> [B,T,d_model]
  backbone.blocks.0                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.norm_ffn                         pow              [B,T,d_model] -> [B,T,d_model]
  backbone.blocks.N.norm_ffn                         mean             [B,T,d_model] -> [B,T,1]
  backbone.blocks.N.norm_ffn                         elementwise_add  [B,T,1] -> [B,T,1]
  backbone.blocks.N.norm_ffn                         rsqrt            [B,T,1] -> [B,T,1]
  backbone.blocks.N.norm_ffn                         elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  backbone.blocks.N.norm_ffn                         elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  backbone.blocks.N.ffn.proj_up_gate                 t                [roundup(d_model*ffn_f,ffn_r),d_model] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [d_model,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up_gate                 view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.ffn.proj_up_gate                 matmul           [T,d_model]*[d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up_gate                 _unsafe_view     [T,roundup(d_model*ffn_f,ffn_r)] -> [B,T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.act_fn                       silu             [B,T,roundup(d_model*ffn_f,ffn_r)] -> [B,T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      t                [roundup(d_model*ffn_f,ffn_r),d_model] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [d_model,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      view             [B,T,d_model] -> [T,d_model]
  backbone.blocks.N.ffn.proj_up                      matmul           [T,d_model]*[d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      _unsafe_view     [T,roundup(d_model*ffn_f,ffn_r)] -> [B,T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn                              elementwise_mul  [B,T,roundup(d_model*ffn_f,ffn_r)]*[B,T,roundup(d_model*ffn_f,ffn_r)] -> [B,T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_down                    t                [d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[d_model,roundup(d_model*ffn_f,ffn_r)] [roundup(d_model*ffn_f,ffn_r),d_model]
  backbone.blocks.N.ffn.proj_down                    view             [B,T,roundup(d_model*ffn_f,ffn_r)] -> [T,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_down                    matmul           [T,roundup(d_model*ffn_f,ffn_r)]*[roundup(d_model*ffn_f,ffn_r),d_model] -> w=[d_model,roundup(d_model*ffn_f,ffn_r)] [T,d_model]
  backbone.blocks.N.ffn.proj_down                    _unsafe_view     [T,d_model] -> [B,T,d_model]
  backbone                                           copy_            [B,n_h,d_model*qk_f/n_h,d_head]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone                                           copy_            [B,n_h,d_model*qk_f/n_h]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone                                           copy_            [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.1                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.2                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.3                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.4                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.5                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.6                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.7                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.8                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.9                                  elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.10                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.11                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.12                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.13                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.14                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.15                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.16                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.17                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.18                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.19                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.20                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.21                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.22                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.23                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.24                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.25                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.26                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.27                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.28                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.29                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.30                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone.blocks.31                                 elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  backbone                                           add_             [B] -> [B]
  backbone.out_norm                                  pow              [B,T,d_model] -> [B,T,d_model]
  backbone.out_norm                                  mean             [B,T,d_model] -> [B,T,1]
  backbone.out_norm                                  elementwise_add  [B,T,1] -> [B,T,1]
  backbone.out_norm                                  rsqrt            [B,T,1] -> [B,T,1]
  backbone.out_norm                                  elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  backbone.out_norm                                  elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,T,d_model] -> [T,d_model]
  lm_head                                            matmul           [T,d_model]*[d_model,V] -> w=[V,d_model] [T,V]
  lm_head                                            _unsafe_view     [T,V] -> [B,T,V]
                                                     div              [B,T,V] -> [B,T,V]
                                                     tanh             [B,T,V] -> [B,T,V]
                                                     elementwise_mul  [B,T,V] -> [B,T,V]
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
  backbone.embeddings                                embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  backbone                                           zeros            [] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone                                           zeros            [] -> [B,n_h,d_model*qk_f/n_h]
  backbone                                           zeros            [] -> [B,n_h,1]
  backbone.blocks.N.norm_mlstm                       pow              [B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.norm_mlstm                       mean             [B,1,d_model] -> [B,1,1]
  backbone.blocks.N.norm_mlstm                       elementwise_add  [B,1,1] -> [B,1,1]
  backbone.blocks.N.norm_mlstm                       rsqrt            [B,1,1] -> [B,1,1]
  backbone.blocks.N.norm_mlstm                       elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  backbone.blocks.N.norm_mlstm                       elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.q                    t                [d_model*qk_f,d_model] -> w=[d_model*qk_f,d_model] [d_model,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.q                    view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.q                    matmul           [B,d_model]*[d_model,d_model*qk_f] -> w=[d_model*qk_f,d_model] [B,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.q                    _unsafe_view     [B,d_model*qk_f] -> [B,1,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    t                [d_model*qk_f,d_model] -> w=[d_model*qk_f,d_model] [d_model,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.k                    matmul           [B,d_model]*[d_model,d_model*qk_f] -> w=[d_model*qk_f,d_model] [B,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.k                    _unsafe_view     [B,d_model*qk_f] -> [B,1,d_model*qk_f]
  backbone.blocks.N.mlstm_layer.v                    t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.v                    view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.v                    matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  backbone.blocks.N.mlstm_layer.v                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  backbone.blocks.N.mlstm_layer.ogate_preact         _unsafe_view     [B,d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.igate_preact         view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.igate_preact         t                [n_h,d_model] -> w=[n_h,d_model] [d_model,n_h]
  backbone.blocks.N.mlstm_layer.igate_preact         linear           [n_h]*[B,d_model]*[d_model,n_h] -> w=[n_h,d_model] [B,n_h]
  backbone.blocks.N.mlstm_layer.igate_preact         view             [B,n_h] -> [B,1,n_h]
  backbone.blocks.N.mlstm_layer                      div              [B,1,n_h] -> [B,1,n_h]
  backbone.blocks.N.mlstm_layer                      tanh             [B,1,n_h] -> [B,1,n_h]
  backbone.blocks.N.mlstm_layer                      elementwise_mul  [B,1,n_h] -> [B,1,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.fgate_preact         t                [n_h,d_model] -> w=[n_h,d_model] [d_model,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         linear           [n_h]*[B,d_model]*[d_model,n_h] -> w=[n_h,d_model] [B,n_h]
  backbone.blocks.N.mlstm_layer.fgate_preact         view             [B,n_h] -> [B,1,n_h]
  backbone.blocks.N.mlstm_layer                      view             [B,1,d_model*qk_f] -> [B,1,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer                      transpose        [B,1,n_h,d_model*qk_f/n_h] -> [B,n_h,1,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer                      view             [B,1,d_model] -> [B,1,n_h,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,1,n_h] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        squeeze          [B,n_h,1,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        squeeze          [B,n_h,1,d_head] -> [B,n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        log_sigmoid_forward [B,n_h,1] -> [B,n_h,1]*[B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        maximum          [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        sub              [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        exp              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,1] -> [B,n_h,1,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,1,1]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,d_model*qk_f/n_h,1] -> [B,n_h,d_model*qk_f/n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,d_model*qk_f/n_h,1] -> [n_h,d_model*qk_f/n_h,B]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,d_model*qk_f/n_h,B]*[n_h,B,d_head] -> [n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,d_model*qk_f/n_h,d_head]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        clone            [B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_mul  [B,n_h,1]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,d_model*qk_f/n_h]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        unsqueeze        [B,n_h,d_model*qk_f/n_h] -> [B,n_h,1,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,1,d_model*qk_f/n_h] -> [B,n_h,1,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,1,d_model*qk_f/n_h] -> [n_h,B,d_model*qk_f/n_h]
  backbone.blocks.N.mlstm_layer.mlstm_backend        expand           [B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        view             [B,n_h,d_model*qk_f/n_h,d_head] -> [n_h,d_model*qk_f/n_h,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,B,d_model*qk_f/n_h]*[n_h,d_model*qk_f/n_h,d_head] -> [n_h,B,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,B,d_head] -> [B,n_h,1,d_head]
  backbone.blocks.N.mlstm_layer.mlstm_backend        batched_matmul   [n_h,B,d_model*qk_f/n_h]*[n_h,d_model*qk_f/n_h,B] -> [n_h,B,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        _unsafe_view     [n_h,B,1] -> [B,n_h,1,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        squeeze          [B,n_h,1,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        neg              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        abs              [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        elementwise_add  [B,n_h,1] -> [B,n_h,1]
  backbone.blocks.N.mlstm_layer.mlstm_backend        div              [B,n_h,d_head]*[B,n_h,1] -> [B,n_h,d_head]
  backbone.blocks.N.mlstm_layer                      transpose        [B,n_h,1,d_head] -> [B,1,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       mean             [B,1,n_h,d_head] -> [B,1,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       sub              [B,1,n_h,d_head]*[B,1,n_h,1] -> [B,1,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       var              [B,1,n_h,d_head] -> [B,1,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_add  [B,1,n_h,1] -> [B,1,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       rsqrt            [B,1,n_h,1] -> [B,1,n_h,1]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_mul  [B,1,n_h,d_head]*[B,1,n_h,1] -> [B,1,n_h,d_head]
  backbone.blocks.N.mlstm_layer.multihead_norm       view             [B,1,n_h,d_head] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.multihead_norm       elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer                      view             [B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.ogate_act_fn         sigmoid          [B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer                      elementwise_mul  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             t                [d_model,d_model] -> w=[d_model,d_model] [d_model,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             matmul           [B,d_model]*[d_model,d_model] -> w=[d_model,d_model] [B,d_model]
  backbone.blocks.N.mlstm_layer.out_proj             _unsafe_view     [B,d_model] -> [B,1,d_model]
  backbone.blocks.0                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.norm_ffn                         pow              [B,1,d_model] -> [B,1,d_model]
  backbone.blocks.N.norm_ffn                         mean             [B,1,d_model] -> [B,1,1]
  backbone.blocks.N.norm_ffn                         elementwise_add  [B,1,1] -> [B,1,1]
  backbone.blocks.N.norm_ffn                         rsqrt            [B,1,1] -> [B,1,1]
  backbone.blocks.N.norm_ffn                         elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  backbone.blocks.N.norm_ffn                         elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  backbone.blocks.N.ffn.proj_up_gate                 t                [roundup(d_model*ffn_f,ffn_r),d_model] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [d_model,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up_gate                 view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.ffn.proj_up_gate                 matmul           [B,d_model]*[d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [B,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up_gate                 _unsafe_view     [B,roundup(d_model*ffn_f,ffn_r)] -> [B,1,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.act_fn                       silu             [B,1,roundup(d_model*ffn_f,ffn_r)] -> [B,1,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      t                [roundup(d_model*ffn_f,ffn_r),d_model] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [d_model,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      view             [B,1,d_model] -> [B,d_model]
  backbone.blocks.N.ffn.proj_up                      matmul           [B,d_model]*[d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[roundup(d_model*ffn_f,ffn_r),d_model] [B,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_up                      _unsafe_view     [B,roundup(d_model*ffn_f,ffn_r)] -> [B,1,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn                              elementwise_mul  [B,1,roundup(d_model*ffn_f,ffn_r)]*[B,1,roundup(d_model*ffn_f,ffn_r)] -> [B,1,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_down                    t                [d_model,roundup(d_model*ffn_f,ffn_r)] -> w=[d_model,roundup(d_model*ffn_f,ffn_r)] [roundup(d_model*ffn_f,ffn_r),d_model]
  backbone.blocks.N.ffn.proj_down                    view             [B,1,roundup(d_model*ffn_f,ffn_r)] -> [B,roundup(d_model*ffn_f,ffn_r)]
  backbone.blocks.N.ffn.proj_down                    matmul           [B,roundup(d_model*ffn_f,ffn_r)]*[roundup(d_model*ffn_f,ffn_r),d_model] -> w=[d_model,roundup(d_model*ffn_f,ffn_r)] [B,d_model]
  backbone.blocks.N.ffn.proj_down                    _unsafe_view     [B,d_model] -> [B,1,d_model]
  backbone                                           copy_            [B,n_h,d_model*qk_f/n_h,d_head]*[B,n_h,d_model*qk_f/n_h,d_head] -> [B,n_h,d_model*qk_f/n_h,d_head]
  backbone                                           copy_            [B,n_h,d_model*qk_f/n_h]*[B,n_h,d_model*qk_f/n_h] -> [B,n_h,d_model*qk_f/n_h]
  backbone                                           copy_            [B,n_h,1]*[B,n_h,1] -> [B,n_h,1]
  backbone.blocks.1                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.2                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.3                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.4                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.5                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.6                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.7                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.8                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.9                                  elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.10                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.11                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.12                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.13                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.14                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.15                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.16                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.17                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.18                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.19                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.20                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.21                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.22                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.23                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.24                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.25                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.26                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.27                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.28                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.29                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.30                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone.blocks.31                                 elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  backbone                                           add_             [B] -> [B]
  backbone.out_norm                                  pow              [B,1,d_model] -> [B,1,d_model]
  backbone.out_norm                                  mean             [B,1,d_model] -> [B,1,1]
  backbone.out_norm                                  elementwise_add  [B,1,1] -> [B,1,1]
  backbone.out_norm                                  rsqrt            [B,1,1] -> [B,1,1]
  backbone.out_norm                                  elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  backbone.out_norm                                  elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  lm_head                                            t                [V,d_model] -> w=[V,d_model] [d_model,V]
  lm_head                                            view             [B,1,d_model] -> [B,d_model]
  lm_head                                            matmul           [B,d_model]*[d_model,V] -> w=[V,d_model] [B,V]
  lm_head                                            _unsafe_view     [B,V] -> [B,1,V]
                                                     div              [B,1,V] -> [B,1,V]
                                                     tanh             [B,1,V] -> [B,1,V]
                                                     elementwise_mul  [B,1,V] -> [B,1,V]
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
# 모델: Qwen__Qwen3-Next-80B-A3B-Instruct
# ============================================================

# 리뷰 패킷 — Qwen/Qwen3-Next-80B-A3B-Instruct

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` / 트레이스 seq_len(T) = 17
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
  d_model      = 2048
  n_h          = 16
  n_kv         = 2
  d_head       = 256
  d_ff         = 5120
  d_shared     = None
  V            = 151936
  ctx          = 262144
  E            = 512
  E_shared     = 1
  k            = 10
  n_grp        = None
  k_grp        = None
  d_moe        = 512
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
  c_kv         = None
  d_nope       = None
  d_v          = None
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

# Model Summary -- Qwen/Qwen3-Next-80B-A3B-Instruct

## 기본 정보

- revision: `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 79.67B total, 3.87B active (4.9% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-09-09  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 36× linear_attention, 12× full_attention  (attention: GQA)  (FFN: 48× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 24.0 KiB (Very low) over 12 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=512, top-10, +1 shared, topk-then-softmax routing) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, topk-softmax routing, QK-Norm, short-conv |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `qwen3_next` |
| attention | GQA — 16 query : 2 kv heads (repeat 8), d_head=256 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=10000000) |
| FFN | MoE — 512 routed experts, top-10 + 1 shared, expert intermediate 512, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·2·256 = 1024 elems / token / layer; 12 attention layer(s) ⇒ 12288 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 48 |
| d_model | 2048 |
| n_h | 16 |
| n_kv | 2 |
| d_head | 256 |
| d_ff | 5120 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 151936 |
| ctx | 262144 |
| E | 512 |
| E_shared | 1 |
| k | 10 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 512 |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 36× linear_attention, 12× full_attention (총 48층) |
| c_kv | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_nope | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
| d_v | —  _(해당 없음: 이 모델은 `mla` 계열 구조를 쓰지 않음)_ |
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

shape 축 **828,876개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 349,005 | 42.11% |
| 이 모듈 스코프의 심볼 | 276,697 | 33.38% |
| 이름 없음 (정수 유지) | 105,996 | 12.79% |
| 이 모듈 스코프의 유도식 | 46,297 | 5.59% |
| 스코프 없는 심볼 | 46,095 | 5.56% |
| 휴리스틱: 심볼의 배수 | 2,016 | 0.24% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,762 | 0.21% |
| 휴리스틱: 심볼+1 | 1,008 | 0.12% |

등록된 규칙 **718,094축**, 약한 근거 1,762축, 휴리스틱 **3,024축 (0.36%)**, 이름 없음 105,996축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 8 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | linear_attn, self_attn |
| 18 | T+1 (decode 의 KV 캐시 길이 — 캐시 T개 + 새 토큰 1개) | linear_attn |
| 20 | n_h + 2·n_kv (fused QKV를 head 축으로 편 총 head 수: Q + K + V) | conv1d, linear_attn |
| 170 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 192 | d_head − d_rope (부분 RoPE 비회전 통과분) | self_attn |
| 544 | T·n_h_lin_v (value head 축까지 flatten — gated norm 입력) | linear_attn, norm |
| 768 | 2·d_k + 2·(n_v/n_k)·d_v (DeltaNet qkvz 를 key head 별로 접은 폭) | linear_attn |
| 1024 | n_h·d_rope | experts |
| 4096 | n_v·d_v (DeltaNet value_dim — v/z 조각 폭) | linear_attn, o_proj, out_proj, self_attn |
| 8192 | 2·key_dim + value_dim (gated delta net conv1d 채널 폭) | conv1d, linear_attn, q_proj, self_attn |
| 12288 | 2·(n_k·d_k) + 2·(n_v·d_v) (DeltaNet in_proj_qkvz 출력: q,k,v,z) | in_proj_qkvz, linear_attn |

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
- layer 32-34: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 35: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 36-38: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 39: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 40-42: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 43: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 44-46: input_layernorm, linear_attn, mlp, post_attention_layernorm
- layer 47: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 48 == 48 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers |
| C6 | PASS | hidden_size=2048 (heuristic check, 2916 flagged) |
| C7 | PASS | GQA 16:2 (repeat factor 8) |
| C8 | WARN | MoE trace-verified [router_dim(E=512):ok, top_k(10):ok, expert_weight:grouped]; routed-token coun... |
| C9 | PASS | vocab_size=151936, tie_word_embeddings=False |
| C10 | PASS | all 759 params covered |
| C11 | PASS | 145 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 33736 unmapped rows, 38 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', ... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Qwen/Qwen3-Next-80B-A3B-Instruct` config.json @ `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` (sha256 `181c31bdb4c1…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- Qwen/Qwen3-Next-80B-A3B-Instruct @ 9c7f2fbe84465e40164a94cc16cd30b6999b0cc7

C1   PASS   48 == 48
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers
C6   PASS   hidden_size=2048 (heuristic check, 2916 flagged)
C7   PASS   GQA 16:2 (repeat factor 8)
C8   WARN   MoE trace-verified [router_dim(E=512):ok, top_k(10):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151936, tie_word_embeddings=False
C10  PASS   all 759 params covered
C11  PASS   145 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   33736 unmapped rows, 38 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.div_.Tensor', 'aten.empty_like.default']
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
  model.layers.N.linear_attn.in_proj_qkvz            t                [2*n_k*d_k+2*n_v*d_v,d_model] -> w=[2*n_k*d_k+2*n_v*d_v,d_model] [d_model,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_qkvz            view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_qkvz            matmul           [T,d_model]*[d_model,2*n_k*d_k+2*n_v*d_v] -> w=[2*n_k*d_k+2*n_v*d_v,d_model] [T,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_qkvz            _unsafe_view     [T,2*n_k*d_k+2*n_v*d_v] -> [B,T,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_ba              t                [2*n_h_lin_v,d_model] -> w=[2*n_h_lin_v,d_model] [d_model,2*n_h_lin_v]
  model.layers.N.linear_attn.in_proj_ba              view             [B,T,d_model] -> [T,d_model]
  model.layers.N.linear_attn.in_proj_ba              matmul           [T,d_model]*[d_model,2*n_h_lin_v] -> w=[2*n_h_lin_v,d_model] [T,2*n_h_lin_v]
  model.layers.N.linear_attn.in_proj_ba              _unsafe_view     [T,2*n_h_lin_v] -> [B,T,2*n_h_lin_v]
  model.layers.N.linear_attn                         view             [B,T,2*n_k*d_k+2*n_v*d_v] -> [B,T,n_h_lin_k,2*d_k+2*(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         view             [B,T,2*n_h_lin_v] -> [B,T,n_h_lin_k,d_conv_lin]
  model.layers.N.linear_attn                         split_with_sizes [B,T,n_h_lin_k,2*d_k+2*(n_v/n_k)*d_v] -> [B,T,n_h_lin_k,d_head_lin_k]*[B,T,n_h_lin_k,d_head_lin_k]*[B,T,n_h_lin_k,(n_v/n_k)*d_v]*[B,T,n_h_lin_k,(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         split_with_sizes [B,T,n_h_lin_k,d_conv_lin] -> [B,T,n_h_lin_k,n_v/n_k]*[B,T,n_h_lin_k,n_v/n_k]
  model.layers.N.linear_attn                         clone            [B,T,n_h_lin_k,(n_v/n_k)*d_v] -> [B,T,n_h_lin_k,(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         _unsafe_view     [B,T,n_h_lin_k,(n_v/n_k)*d_v] -> [B,T,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         clone            [B,T,n_h_lin_k,n_v/n_k] -> [B,T,n_h_lin_k,n_v/n_k]
  model.layers.N.linear_attn                         _unsafe_view     [B,T,n_h_lin_k,n_v/n_k] -> [B,T,n_h_lin_v]
  model.layers.N.linear_attn                         clone            [B,T,n_h_lin_k,d_head_lin_k] -> [B,T,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         _unsafe_view     [B,T,n_h_lin_k,d_head_lin_k] -> [B,T,d_model]
  model.layers.N.linear_attn                         view             [B,T,n_h_lin_v,d_head_lin_v] -> [B,T,n_v*d_v]
  model.layers.N.linear_attn                         concat           [B,T,n_h_lin_k*d_head_lin_k]*[B,T,n_h_lin_k*d_head_lin_k]*[B,T,n_v*d_v] -> [B,T,2*n_h*d_head]
  model.layers.N.linear_attn                         transpose        [B,T,2*n_h*d_head] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn                         constant_pad_nd  [B,2*n_h*d_head,T] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         zeros            [] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         copy_            [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn.conv1d                  conv1d           [B,2*n_h*d_head,T]*[2*n_h*d_head,1,d_conv_lin] -> w=[2*n_h*d_head,1,d_conv_lin] [B,2*n_h*d_head,T+d_conv_lin-1]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,T+d_conv_lin-1] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn                         silu             [B,2*n_h*d_head,T] -> [B,2*n_h*d_head,T]
  model.layers.N.linear_attn                         transpose        [B,2*n_h*d_head,T] -> [B,T,2*n_h*d_head]
  model.layers.N.linear_attn                         split_with_sizes [B,T,2*n_h*d_head] -> [B,T,n_h_lin_k*d_head_lin_k]*[B,T,n_h_lin_k*d_head_lin_k]*[B,T,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,T,d_model] -> [B,T,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,T,n_v*d_v] -> [B,T,n_h_lin_v,d_head_lin_v]
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
  model.layers.N.linear_attn                         transpose        [B,T,n_h_lin_v,d_head_lin_v] -> [B,n_h_lin_v,T,d_head_lin_v]
  model.layers.N.linear_attn                         clone            [B,n_h_lin_v,T,d_head_lin_v] -> [B,n_h_lin_v,T,d_head_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,T,d_head_lin_v] -> [B,n_h_lin_v,T,d_head_lin_v]
  model.layers.N.linear_attn                         transpose        [B,T,n_h_lin_v] -> [B,n_h_lin_v,T]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,T] -> [B,n_h_lin_v,T]
  model.layers.N.linear_attn                         constant_pad_nd  [B,n_h_lin_v,T,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         constant_pad_nd  [B,n_h_lin_v,T,d_head_lin_v] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         constant_pad_nd  [B,n_h_lin_v,T] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_chunk,d_head_lin_v]*[B,n_h_lin_v,d_chunk,1] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,d_chunk,d_head_lin_k]*[B,n_h_lin_v,d_chunk,1] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         ones             [] -> [d_chunk,d_chunk]
  model.layers.N.linear_attn                         triu             [d_chunk,d_chunk] -> [d_chunk,d_chunk]
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
  model.layers.N.linear_attn                         masked_fill      [B,n_h_lin_v,1,d_chunk,d_chunk]*[d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         neg              [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,1,64]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,1,64] -> [B,n_h_lin_v,1,1,1]
  model.layers.N.linear_attn                         unsqueeze        [B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1,1]
  model.layers.N.linear_attn                         sum              [B,n_h_lin_v,1,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         elementwise_add  [B,n_h_lin_v,1,1]*[B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,1,1]*[B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         slice            [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,2]
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
  model.layers.N.linear_attn                         eye              [] -> [d_chunk,d_chunk]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,1,d_chunk,d_chunk]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,1,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_v] -> [n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,1,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_k] -> [n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         _unsafe_view     [n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         zeros            [] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn                         zeros_like       [B,n_h_lin_v,1,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,1,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         expand           [B,n_h_lin_v,d_chunk,d_head_lin_k] -> [B,n_h_lin_v,d_chunk,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk,d_chunk] -> [B,n_h_lin_v,d_chunk,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_chunk,d_head_lin_k]*[n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,d_chunk,d_head_lin_v]*[B,n_h_lin_v,d_chunk,d_head_lin_v] -> [B,n_h_lin_v,d_chunk,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,d_chunk,1] -> [B,n_h_lin_v,d_chunk,1]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,1,1] -> [B,n_h_lin_v,1,1]
  model.layers.N.linear_attn                         sub              [B,n_h_lin_v,1]*[B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         exp              [B,n_h_lin_v,d_chunk] -> [B,n_h_lin_v,d_chunk]
  model.layers.N.linear_attn                         batched_matmul   [n_h_lin_v,d_head_lin_k,d_chunk]*[n_h_lin_v,d_chunk,d_head_lin_v] -> [n_h_lin_v,d_head_lin_k,d_head_lin_v]
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
  model.layers.N.mlp                                 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_expert.gate_proj         t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mlp.shared_expert.gate_proj         matmul           [T,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [T,d_shared]
  model.layers.N.mlp.shared_expert.act_fn            silu             [T,d_shared] -> [T,d_shared]
  model.layers.N.mlp.shared_expert.up_proj           t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mlp.shared_expert.up_proj           matmul           [T,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [T,d_shared]
  model.layers.N.mlp.shared_expert                   elementwise_mul  [T,d_shared]*[T,d_shared] -> [T,d_shared]
  model.layers.N.mlp.shared_expert.down_proj         t                [d_model,d_shared] -> w=[d_model,d_shared] [d_shared,d_model]
  model.layers.N.mlp.shared_expert.down_proj         matmul           [T,d_shared]*[d_shared,d_model] -> w=[d_model,d_shared] [T,d_model]
  model.layers.N.mlp.gate                            view             [T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mlp.gate                            _to_copy         [T,E] -> [T,E]
  model.layers.N.mlp.gate                            softmax          [T,E] -> [T,E]
  model.layers.N.mlp.gate                            topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mlp.gate                            sum              [T,k] -> [T,1]
  model.layers.N.mlp.gate                            div_             [T,k]*[T,1] -> [T,k]
  model.layers.N.mlp.gate                            _to_copy         [T,k] -> [T,k]
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
  model.layers.N.mlp.shared_expert_gate              t                [1,d_model] -> w=[1,d_model] [d_model,1]
  model.layers.N.mlp.shared_expert_gate              matmul           [T,d_model]*[d_model,1] -> w=[1,d_model] [T,1]
  model.layers.N.mlp                                 sigmoid          [T,1] -> [T,1]
  model.layers.N.mlp                                 elementwise_mul  [T,1]*[T,d_model] -> [T,d_model]
  model.layers.N.mlp                                 elementwise_add  [T,d_model]*[T,d_model] -> [T,d_model]
  model.layers.N.mlp                                 view             [T,d_model] -> [B,T,d_model]
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
  model.layers.N.linear_attn.in_proj_qkvz            t                [2*n_k*d_k+2*n_v*d_v,d_model] -> w=[2*n_k*d_k+2*n_v*d_v,d_model] [d_model,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_qkvz            view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_qkvz            matmul           [B,d_model]*[d_model,2*n_k*d_k+2*n_v*d_v] -> w=[2*n_k*d_k+2*n_v*d_v,d_model] [B,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_qkvz            _unsafe_view     [B,2*n_k*d_k+2*n_v*d_v] -> [B,1,2*n_k*d_k+2*n_v*d_v]
  model.layers.N.linear_attn.in_proj_ba              t                [2*n_h_lin_v,d_model] -> w=[2*n_h_lin_v,d_model] [d_model,2*n_h_lin_v]
  model.layers.N.linear_attn.in_proj_ba              view             [B,1,d_model] -> [B,d_model]
  model.layers.N.linear_attn.in_proj_ba              matmul           [B,d_model]*[d_model,2*n_h_lin_v] -> w=[2*n_h_lin_v,d_model] [B,2*n_h_lin_v]
  model.layers.N.linear_attn.in_proj_ba              _unsafe_view     [B,2*n_h_lin_v] -> [B,1,2*n_h_lin_v]
  model.layers.N.linear_attn                         view             [B,1,2*n_k*d_k+2*n_v*d_v] -> [B,1,n_h_lin_k,2*d_k+2*(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         view             [B,1,2*n_h_lin_v] -> [B,1,n_h_lin_k,d_conv_lin]
  model.layers.N.linear_attn                         split_with_sizes [B,1,n_h_lin_k,2*d_k+2*(n_v/n_k)*d_v] -> [B,1,n_h_lin_k,d_head_lin_k]*[B,1,n_h_lin_k,d_head_lin_k]*[B,1,n_h_lin_k,(n_v/n_k)*d_v]*[B,1,n_h_lin_k,(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         split_with_sizes [B,1,n_h_lin_k,d_conv_lin] -> [B,1,n_h_lin_k,n_v/n_k]*[B,1,n_h_lin_k,n_v/n_k]
  model.layers.N.linear_attn                         clone            [B,1,n_h_lin_k,(n_v/n_k)*d_v] -> [B,1,n_h_lin_k,(n_v/n_k)*d_v]
  model.layers.N.linear_attn                         _unsafe_view     [B,1,n_h_lin_k,(n_v/n_k)*d_v] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         clone            [B,1,n_h_lin_k,n_v/n_k] -> [B,1,n_h_lin_k,n_v/n_k]
  model.layers.N.linear_attn                         _unsafe_view     [B,1,n_h_lin_k,n_v/n_k] -> [B,1,n_h_lin_v]
  model.layers.N.linear_attn                         clone            [B,1,n_h_lin_k,d_head_lin_k] -> [B,1,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         _unsafe_view     [B,1,n_h_lin_k,d_head_lin_k] -> [B,1,d_model]
  model.layers.N.linear_attn                         view             [B,1,n_h_lin_v,d_head_lin_v] -> [B,1,n_v*d_v]
  model.layers.N.linear_attn                         concat           [B,1,n_h_lin_k*d_head_lin_k]*[B,1,n_h_lin_k*d_head_lin_k]*[B,1,n_v*d_v] -> [B,1,2*n_h*d_head]
  model.layers.N.linear_attn                         transpose        [B,1,2*n_h*d_head] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn                         squeeze          [2*n_h*d_head,1,d_conv_lin] -> w=[2*n_h*d_head,1,d_conv_lin] [2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         concat           [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,1] -> [B,2*n_h*d_head,5]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,5] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         copy_            [B,2*n_h*d_head,d_conv_lin]*[B,2*n_h*d_head,d_conv_lin] -> [B,2*n_h*d_head,d_conv_lin]
  model.layers.N.linear_attn                         unsqueeze        [2*n_h*d_head,d_conv_lin] -> [2*n_h*d_head,1,d_conv_lin]
  model.layers.N.linear_attn                         conv1d           [B,2*n_h*d_head,5]*[2*n_h*d_head,1,d_conv_lin] -> [B,2*n_h*d_head,2]
  model.layers.N.linear_attn                         slice            [B,2*n_h*d_head,2] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn                         silu             [B,2*n_h*d_head,1] -> [B,2*n_h*d_head,1]
  model.layers.N.linear_attn                         transpose        [B,2*n_h*d_head,1] -> [B,1,2*n_h*d_head]
  model.layers.N.linear_attn                         split_with_sizes [B,1,2*n_h*d_head] -> [B,1,n_h_lin_k*d_head_lin_k]*[B,1,n_h_lin_k*d_head_lin_k]*[B,1,n_v*d_v]
  model.layers.N.linear_attn                         view             [B,1,d_model] -> [B,1,n_h_lin_k,d_head_lin_k]
  model.layers.N.linear_attn                         view             [B,1,n_v*d_v] -> [B,1,n_h_lin_v,d_head_lin_v]
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
  model.layers.N.linear_attn                         transpose        [B,1,n_h_lin_v,d_head_lin_v] -> [B,n_h_lin_v,1,d_head_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,1,d_head_lin_v] -> [B,n_h_lin_v,1,d_head_lin_v]
  model.layers.N.linear_attn                         transpose        [B,1,n_h_lin_v] -> [B,n_h_lin_v,1]
  model.layers.N.linear_attn                         _to_copy         [B,n_h_lin_v,1] -> [B,n_h_lin_v,1]
  model.layers.N.linear_attn                         elementwise_mul  [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,1,d_head_lin_k]
  model.layers.N.linear_attn                         zeros            [] -> [B,n_h_lin_v,1,d_head_lin_v]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_head_lin_k] -> [B,n_h_lin_v,d_head_lin_k]
  model.layers.N.linear_attn                         select           [B,n_h_lin_v,1,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_v]
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
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         transpose        [B,n_h_lin_v,1,d_head_lin_v] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         _to_copy         [B,1,n_h_lin_v,d_head_lin_v] -> [B,1,n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn                         copy_            [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]*[B,n_h_lin_v,d_head_lin_k,d_head_lin_v] -> [B,n_h_lin_v,d_head_lin_k,d_head_lin_v]
  model.layers.N.linear_attn.norm                    _to_copy         [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    pow              [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    mean             [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    elementwise_add  [n_h_lin_v,B] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    rsqrt            [n_h_lin_v,B] -> [n_h_lin_v,B]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v,d_head_lin_v]*[n_h_lin_v,B] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [d_head_lin_v]*[n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    silu             [n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
  model.layers.N.linear_attn.norm                    elementwise_mul  [n_h_lin_v,d_head_lin_v]*[n_h_lin_v,d_head_lin_v] -> [n_h_lin_v,d_head_lin_v]
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
  model.layers.N.mlp                                 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_expert.gate_proj         t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mlp.shared_expert.gate_proj         matmul           [B,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [B,d_shared]
  model.layers.N.mlp.shared_expert.act_fn            silu             [B,d_shared] -> [B,d_shared]
  model.layers.N.mlp.shared_expert.up_proj           t                [d_shared,d_model] -> w=[d_shared,d_model] [d_model,d_shared]
  model.layers.N.mlp.shared_expert.up_proj           matmul           [B,d_model]*[d_model,d_shared] -> w=[d_shared,d_model] [B,d_shared]
  model.layers.N.mlp.shared_expert                   elementwise_mul  [B,d_shared]*[B,d_shared] -> [B,d_shared]
  model.layers.N.mlp.shared_expert.down_proj         t                [d_model,d_shared] -> w=[d_model,d_shared] [d_shared,d_model]
  model.layers.N.mlp.shared_expert.down_proj         matmul           [B,d_shared]*[d_shared,d_model] -> w=[d_model,d_shared] [B,d_model]
  model.layers.N.mlp.gate                            view             [B,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mlp.gate                            _to_copy         [B,E] -> [B,E]
  model.layers.N.mlp.gate                            softmax          [B,E] -> [B,E]
  model.layers.N.mlp.gate                            topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mlp.gate                            sum              [B,k] -> [B,1]
  model.layers.N.mlp.gate                            div_             [B,k]*[B,1] -> [B,k]
  model.layers.N.mlp.gate                            _to_copy         [B,k] -> [B,k]
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
  model.layers.N.mlp.shared_expert_gate              t                [1,d_model] -> w=[1,d_model] [d_model,1]
  model.layers.N.mlp.shared_expert_gate              matmul           [B,d_model]*[d_model,1] -> w=[1,d_model] [B,1]
  model.layers.N.mlp                                 sigmoid          [B,1] -> [B,1]
  model.layers.N.mlp                                 elementwise_mul  [B,1]*[B,d_model] -> [B,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,d_model]*[B,d_model] -> [B,d_model]
  model.layers.N.mlp                                 view             [B,d_model] -> [B,1,d_model]
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



# ============================================================
# 모델: allenai__OLMo-2-1124-7B-Instruct
# ============================================================

# 리뷰 패킷 — allenai/OLMo-2-1124-7B-Instruct

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `470b1fba1ae01581f270116362ee4aa1b97f4c84` / 트레이스 seq_len(T) = 16
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
  n_kv         = 32
  d_head       = 128
  d_ff         = 11008
  d_shared     = None
  V            = 100352
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

# Model Summary -- allenai/OLMo-2-1124-7B-Instruct

## 기본 정보

- revision: `470b1fba1ae01581f270116362ee4aa1b97f4c84`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 7.3B total (dense) |
| 2 | Context (tokens) | 4,096  _(config max_position_embeddings)_ |
| 3 | DATE | 2024-12-18  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MHA |
| 6 | LAYER MIX | 32× MHA |
| 7 | KV CACHE / TOKEN (BF16) | 512.0 KiB (Very high) |
| 8 | KEY DETAIL | MHA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, MHA, QK-Norm |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `olmo2` |
| attention | MHA — 32 heads (no GQA/MQA), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=500000) |
| FFN | dense FFN — intermediate 11008, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·32·128 = 8192 elems / token / layer; 32 attention layer(s) ⇒ 262144 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 32 |
| d_model | 4096 |
| n_h | 32 |
| n_kv | 32 |
| d_head | 128 |
| d_ff | 11008 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 100352 |
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

shape 축 **77,100개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 28,057 | 36.39% |
| 스코프 없는 심볼 | 19,589 | 25.41% |
| 이 모듈 스코프의 심볼 | 17,995 | 23.34% |
| 이 모듈 스코프의 유도식 | 10,105 | 13.11% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,226 | 1.59% |
| 이름 없음 (정수 유지) | 128 | 0.17% |

등록된 규칙 **75,746축**, 약한 근거 1,226축, 휴리스틱 **0축 (0.0%)**, 이름 없음 128축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

> ⚠ **이 표는 값 하나당 대표 식 하나만 보여준다.** 서로 다른 모듈이 우연히 같은 값을 가지면(예: `n_kv*d_head`와 `2*d_head`가 이 체크포인트에서 같은 128) 이 표에는 둘 중 스코프가 먼저 걸린 식 하나만 뜨고, 그 값이 나타나는 다른 모듈들도 전부 그 옆에 나열된다 — 그 모듈들의 **실제** 라벨이 그 식이라는 뜻은 아니다. 축 하나하나에 정확히 붙은 이름은 이 표가 아니라 `full/<phase>.csv`/`.jsonl`(모듈별로 이미 정확히 구분됨)을 봐야 한다. (외부 검토, 2026-09-02 -- 재추적 없이는 이 표 자체를 모듈별로 쪼갤 수 없다.)

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |

## 레이어 구조

- layer 0-31: mlp, post_attention_layernorm, post_feedforward_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 32 == 32 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers |
| C6 | PASS | hidden_size=4096 (heuristic check, 1952 flagged) |
| C7 | PASS | MHA (kv_heads == heads, not GQA) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=100352, tie_word_embeddings=False |
| C10 | PASS | all 355 params covered |
| C11 | PASS | 129 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1925 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `allenai/OLMo-2-1124-7B-Instruct` config.json @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` (sha256 `fca81fdc3155…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토

**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 구별 불가능한 축)이 여기서만 걸러진다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- allenai/OLMo-2-1124-7B-Instruct @ 470b1fba1ae01581f270116362ee4aa1b97f4c84

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers
C6   PASS   hidden_size=4096 (heuristic check, 1952 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=100352, tie_word_embeddings=False
C10  PASS   all 355 params covered
C11  PASS   129 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1925 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default']
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,T,n_h*d_head] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,T,n_h*d_head]*[B,T,1] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [n_h*d_head]*[B,T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,T,n_kv*d_head] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,T,n_kv*d_head]*[B,T,1] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [n_kv*d_head]*[B,T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,n_kv*d_head] -> [B,T,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,T,n_kv*d_head] -> [B,T,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_kv,d_head] -> [B,n_kv,T,d_head]
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
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           concat           [0]*[B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_kv,T,d_head] -> [B,n_kv,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,d_head,T] -> [B,n_kv,d_head,T]
  model.layers.N.self_attn                           view             [B,n_kv,d_head,T] -> [n_kv,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_kv,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
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
  model.layers.N.post_attention_layernorm            _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.post_feedforward_layernorm          _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_feedforward_layernorm          pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.post_feedforward_layernorm          mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.post_feedforward_layernorm          elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.post_feedforward_layernorm          rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,1,n_h*d_head] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,1,n_h*d_head]*[B,1,1] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [n_h*d_head]*[B,1,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,1,n_kv*d_head] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,n_kv*d_head]*[B,1,1] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [n_kv*d_head]*[B,1,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,n_kv*d_head] -> [B,1,n_kv*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,1,n_kv*d_head] -> [B,1,n_kv,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_kv,d_head] -> [B,n_kv,1,d_head]
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
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           concat           [B,n_kv,T,d_head]*[B,n_kv,1,d_head] -> [B,n_kv,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_kv,T+1,d_head] -> [B,n_kv,d_head,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_kv,d_head,T+1] -> [B,n_kv,d_head,T+1]
  model.layers.N.self_attn                           view             [B,n_kv,d_head,T+1] -> [n_kv,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_kv,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
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
  model.layers.N.post_attention_layernorm            _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.post_feedforward_layernorm          _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_feedforward_layernorm          pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.post_feedforward_layernorm          mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.post_feedforward_layernorm          elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.post_feedforward_layernorm          rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
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

