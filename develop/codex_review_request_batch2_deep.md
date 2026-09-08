# 검토 요청 — 배치 2/4: LFM2-8B-A1B, Qwen3-30B-A3B, ERNIE-4.5-21B-A3B-PT, gemma-3-270m, 실제 레이어별 심볼 표현 검증

## 반드시 지켜줄 것

**이 저장소의 파이프라인/추출 코드는 전혀 보지 마세요.** `src/`, `rules/`, `develop/` 등
"어떻게 이 결과가 나왔는지"를 설명하는 어떤 내부 코드도 참고하지 마세요 (애초에 첨부돼 있지
않습니다 -- 아래 각 모델의 **리뷰 패킷**이 전부입니다).

**이전 검토와 다른 점**: 지난 라운드는 요약 카드(`model_summary.md`, 집계 숫자)만 봤습니다.
이번엔 **레이어별 실제 op 표본**(각 패킷의 "5. 대표 트레이스 표본", prefill+decode)까지
전부 봐주세요 -- "hidden_size=X, head 96개" 같은 집계 숫자가 아니라, **그 축 하나하나에
붙은 심볼 이름이 그 위치에서 실제로 맞는 개념인가**를 확인하는 게 목적입니다. 값이 맞아도
개념이 틀릴 수 있습니다(예: 두 config 필드가 우연히 같은 값이라 이름이 뒤바뀐 경우).

이 요청 하나에 4개 모델의 전체 리뷰 패킷을 이어붙였습니다. 모델별로 독립적으로
판단해 주세요.

## 확인해줄 것 (모델마다, 패킷 안 "리뷰 요청" 절차를 따르되)

1. 공식 HF modeling 코드/config, 논문/기술 리포트, vLLM/SGLang 독립 구현, 신뢰도 높은
   아키텍처 정리 자료와 대조
2. 심볼 이름이 **그 위치에서 실제로 의미하는 것과 맞는가** (값이 맞아도 개념이 틀릴 수 있음)
3. attention 계열 판정(MHA/GQA/MQA/MLA 등)이 실제 구현과 맞는가 -- config 필드를 그대로
   믿지 말 것
4. KV cache 계산의 전제(어느 레이어가 캐시를 갖는지, K/V가 별개인지)가 맞는가
5. decode 표본을 꼭 보세요 -- prefill에 없는 축(캐시 길이, sliding 상한 등)이 거기 있습니다
6. 같은 (모듈, op)인데 shape 표기가 갈리는 줄이 있는가

## 답변 형식

모델별로, 각 지적은:

| # | 관찰(사실) | 근거(공식 소스 파일명+함수/클래스명+인용, op_id는 근거로 쓰지 말 것 -- module_path와 shape로 지목) | 가설(원인) | 확신도 | 검증 방법 |
|---|---|---|---|---|---|

전부 맞는 모델은 짧게 한 줄로 충분합니다. **틀린 지적보다 놓친 지적이 낫습니다.**

---



# ============================================================
# 모델: LiquidAI__LFM2-8B-A1B
# ============================================================

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
  d_moe_lat    = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['conv', 'conv', 'full_attention', 'conv', 'conv', 'conv', 'full_attention', 'conv', 'conv', 'conv', 'full_attention', 'conv', 'conv', 'conv', 'full_attention', 'conv', 'conv', 'conv', 'full_attention', 'conv', 'conv', 'full_attention', 'conv', 'conv']
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
| 6 | LAYER MIX | 18× conv, 6× full_attention  (attention: GQA)  (FFN: 2 dense + 22 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 12.0 KiB (Very low) over 6 attn layers |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=32, top-4, sigmoid gating/aux-loss-free); dense-prefix 2 layer(s) |
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
| KV cache 크기 | 2·n_kv·d_head = 2·8·64 = 1024 elems / token / layer; 6 attention layer(s) ⇒ 6144 / token |

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
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 18× conv, 6× full_attention (총 24층) |
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
| 이 모듈 스코프의 유도식 | 5,251 | 12.35% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 438 | 1.03% |
| 이름 없음 (정수 유지) | 78 | 0.18% |
| 스코프가 배제한 심볼 | 39 | 0.09% |

등록된 규칙 **41,965축**, 약한 근거 477축, 휴리스틱 **0축 (0.0%)**, 이름 없음 78축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 18 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv |
| 512 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 3584 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |
| 6144 | 3·d_model (LFM2 short-conv in_proj 융합 투영 폭: B⊕C⊕x) | conv, in_proj |

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

2026-08-13 · llm(claude, 반박 프레임 전건 판정)

미답 항목 1건을 소스로 판정했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 교정 필요 | 4 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `^model\.pos_emb$` | `32` | `d_head/2` | 36 | 위 항목과 같은 축이다. rotary 의 inv_freq 절반 폭(=d_head/2=32)이며, 이름이 없던 행까지 함께 맞춘다. (전건 순회 2026-08-12) |

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
  model.pos_emb                                      unsqueeze        [d_head/2] -> [B,d_head/2]
  model.pos_emb                                      unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  model.pos_emb                                      expand           [B,d_head/2,1] -> [B,d_head/2,1]
  model.pos_emb                                      unsqueeze        [B,T] -> [B,1,T]
  model.pos_emb                                      _to_copy         [B,1,T] -> [B,1,T]
  model.pos_emb                                      view             [B,d_head/2,1] -> [B,d_head/2,1]
  model.pos_emb                                      expand           [B,1,T] -> [B,1,T]
  model.pos_emb                                      view             [B,1,T] -> [B,1,T]
  model.pos_emb                                      batched_matmul   [B,d_head/2,1]*[B,1,T] -> [B,d_head/2,T]
  model.pos_emb                                      _unsafe_view     [B,d_head/2,T] -> [B,d_head/2,T]
  model.pos_emb                                      transpose        [B,d_head/2,T] -> [B,T,d_head/2]
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
  model.layers.N.conv.conv                           conv1d           [B,d_model,T]*[d_model,1,d_conv] -> w=[d_model,1,d_conv] [B,d_model,T+d_conv-1]
  model.layers.N.conv                                slice            [B,d_model,T+d_conv-1] -> [B,d_model,T]
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [T,n_h*d_head]
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
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [T,n_kv*d_head]
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
  model.layers.N.self_attn.out_proj                  t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.out_proj                  view             [B,T,n_h*d_head] -> [T,n_h*d_head]
  model.layers.N.self_attn.out_proj                  matmul           [T,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [T,d_model]
  model.layers.N.self_attn.out_proj                  _unsafe_view     [T,d_model] -> [B,T,d_model]
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
  model.pos_emb                                      unsqueeze        [d_head/2] -> [B,d_head/2]
  model.pos_emb                                      unsqueeze        [B,d_head/2] -> [B,d_head/2,1]
  model.pos_emb                                      expand           [B,d_head/2,1] -> [B,d_head/2,1]
  model.pos_emb                                      unsqueeze        [B,1] -> [B,1,1]
  model.pos_emb                                      _to_copy         [B,1,1] -> [B,1,1]
  model.pos_emb                                      view             [B,d_head/2,1] -> [B,d_head/2,1]
  model.pos_emb                                      expand           [B,1,1] -> [B,1,1]
  model.pos_emb                                      view             [B,1,1] -> [B,1,1]
  model.pos_emb                                      batched_matmul   [B,d_head/2,1]*[B,1,1] -> [B,d_head/2,1]
  model.pos_emb                                      _unsafe_view     [B,d_head/2,1] -> [B,d_head/2,1]
  model.pos_emb                                      transpose        [B,d_head/2,1] -> [B,1,d_head/2]
  model.pos_emb                                      concat           [B,1,d_head/2]*[B,1,d_head/2] -> [B,1,d_head]
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
  model.layers.N.conv                                select           [d_model,1,d_conv] -> w=[d_model,1,d_conv] [d_model,d_conv]
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
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [B,n_h*d_head]
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
  model.layers.N.self_attn.k_proj                    t                [n_kv*d_head,d_model] -> w=[n_kv*d_head,d_model] [d_model,n_kv*d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,n_kv*d_head] -> w=[n_kv*d_head,d_model] [B,n_kv*d_head]
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
  model.layers.N.self_attn.out_proj                  t                [d_model,n_h*d_head] -> w=[d_model,n_h*d_head] [n_h*d_head,d_model]
  model.layers.N.self_attn.out_proj                  view             [B,1,n_h*d_head] -> [B,n_h*d_head]
  model.layers.N.self_attn.out_proj                  matmul           [B,n_h*d_head]*[n_h*d_head,d_model] -> w=[d_model,n_h*d_head] [B,d_model]
  model.layers.N.self_attn.out_proj                  _unsafe_view     [B,d_model] -> [B,1,d_model]
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



# ============================================================
# 모델: Qwen__Qwen3-30B-A3B
# ============================================================

# 리뷰 패킷 — Qwen/Qwen3-30B-A3B

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` / 트레이스 seq_len(T) = 104
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
  n_h          = 32
  n_kv         = 4
  d_head       = 128
  d_ff         = 6144
  d_shared     = None
  V            = 151936
  ctx          = 40960
  E            = 128
  E_shared     = 0
  k            = 8
  n_grp        = None
  k_grp        = None
  d_moe        = 768
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

# Model Summary -- Qwen/Qwen3-30B-A3B

## 기본 정보

- revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 104
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 30.53B total, 3.35B active (11.0% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 40,960  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-04-27  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 48× GQA  (FFN: 48× MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 96.0 KiB (Moderate) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=128, top-8) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, QK-Norm |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `qwen3_moe` |
| attention | GQA — 32 query : 4 kv heads (repeat 8), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=1000000.0) |
| FFN | MoE — 128 routed experts, top-8, expert intermediate 768, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·4·128 = 1024 elems / token / layer; 48 attention layer(s) ⇒ 49152 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 48 |
| d_model | 2048 |
| n_h | 32 |
| n_kv | 4 |
| d_head | 128 |
| d_ff | 6144 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 151936 |
| ctx | 40960 |
| E | 128 |
| E_shared | 0 |
| k | 8 |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | 768 |
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

shape 축 **148,452개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 이 모듈 스코프의 심볼 | 47,629 | 32.08% |
| 런타임 축 (B/T/1) | 44,420 | 29.92% |
| 스코프 없는 심볼 | 38,339 | 25.83% |
| 이 모듈 스코프의 유도식 | 15,952 | 10.75% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,920 | 1.29% |
| 이름 없음 (정수 유지) | 192 | 0.13% |

등록된 규칙 **146,340축**, 약한 근거 1,920축, 휴리스틱 **0축 (0.0%)**, 이름 없음 192축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 512 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 832 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 1536 | 2·d_moe (라우팅 전문가 gate+up 융합 투영 폭) | experts |
| 4096 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_proj, self_attn |

## 레이어 구조

- layer 0-47: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 1개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 48 == 48 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers |
| C6 | PASS | hidden_size=2048 (heuristic check, 5472 flagged) |
| C7 | PASS | GQA 32:4 (repeat factor 8) |
| C8 | WARN | MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count... |
| C9 | PASS | vocab_size=151936, tie_word_embeddings=False |
| C10 | PASS | all 531 params covered |
| C11 | PASS | 193 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=104 >= required=104 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 4531 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `Qwen/Qwen3-30B-A3B` config.json @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (sha256 `3843fde337e2…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=104 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 반박 프레임 전건 판정)

의뢰서의 `2*d_moe` 는 이름이 옳았다 — 산술 휴리스틱이 내던 것을 규칙으로 승격했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 2 |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- Qwen/Qwen3-30B-A3B @ ad44e777bcd18fa416d9da3bd8f70d33ebb85d39

C1   PASS   48 == 48
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers
C6   PASS   hidden_size=2048 (heuristic check, 5472 flagged)
C7   PASS   GQA 32:4 (repeat factor 8)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151936, tie_word_embeddings=False
C10  PASS   all 531 params covered
C11  PASS   193 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=104 >= required=104
C15  PASS   all discovered entrypoints traced
C16  INFO   4531 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
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
  model.layers.N.self_attn.q_norm                    _to_copy         [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,T,n_h,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,T,n_h,d_head] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,T,n_h,1] -> [B,T,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,T,n_h,d_head]*[B,T,n_h,1] -> [B,T,n_h,d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [d_head]*[B,T,n_h,d_head] -> [B,T,n_h,d_head]
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
  model.layers.N.self_attn.k_norm                    elementwise_mul  [d_head]*[B,T,n_kv,d_head] -> [B,T,n_kv,d_head]
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
  model.layers.N.self_attn                           view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
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
  model.layers.N.mlp.experts                         unsqueeze        [k*T] -> [k*T,B]
  model.layers.N.mlp.experts                         clamp_           [k*T] -> [k*T]
  model.layers.N.mlp.experts                         masked_fill_     [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k*T,2*d_moe]
  model.layers.N.mlp.experts                         split            [k*T,2*d_moe] -> [k*T,d_moe]*[k*T,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_moe]*[k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k*T,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.mlp.experts                         empty_like       [k*T] -> [k*T]
  model.layers.N.mlp.experts                         arange           [] -> [k*T]
  model.layers.N.mlp.experts                         index_put_       [k*T]*[k*T]*[k*T] -> [k*T]
  model.layers.N.mlp.experts                         index            [k*T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.mlp.experts                         view             [k*T,d_model] -> [T,k,d_model]
  model.layers.N.mlp.experts                         sum              [T,k,d_model] -> [T,d_model]
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
  model.layers.N.self_attn.q_norm                    _to_copy         [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,1,n_h,d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,1,n_h,d_head] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,1,n_h,1] -> [B,1,n_h,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,1,n_h,d_head]*[B,1,n_h,1] -> [B,1,n_h,d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [d_head]*[B,1,n_h,d_head] -> [B,1,n_h,d_head]
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
  model.layers.N.self_attn.k_norm                    elementwise_mul  [d_head]*[B,1,n_kv,d_head] -> [B,1,n_kv,d_head]
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
  model.layers.N.self_attn                           view             [B,n_h,1,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
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
  model.layers.N.mlp.experts                         unsqueeze        [k] -> [k,B]
  model.layers.N.mlp.experts                         clamp_           [k] -> [k]
  model.layers.N.mlp.experts                         masked_fill_     [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k,2*d_moe]
  model.layers.N.mlp.experts                         split            [k,2*d_moe] -> [k,d_moe]*[k,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_moe]*[k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         empty_like       [k] -> [k]
  model.layers.N.mlp.experts                         arange           [] -> [k]
  model.layers.N.mlp.experts                         index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.mlp.experts                         index            [k,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         view             [k,d_model] -> [B,k,d_model]
  model.layers.N.mlp.experts                         sum              [B,k,d_model] -> [B,d_model]
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



# ============================================================
# 모델: baidu__ERNIE-4.5-21B-A3B-PT
# ============================================================

# 리뷰 패킷 — baidu/ERNIE-4.5-21B-A3B-PT

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `87db95487941cb39592ee0abca3b9155a6d19c5c` / 트레이스 seq_len(T) = 16
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
  L            = 28
  d_model      = 2560
  n_h          = 20
  n_kv         = 4
  d_head       = 128
  d_ff         = 12288
  d_shared     = None
  V            = 103424
  ctx          = 131072
  E            = 64
  E_shared     = 2
  k            = 6
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

# Model Summary -- baidu/ERNIE-4.5-21B-A3B-PT

## 기본 정보

- revision: `87db95487941cb39592ee0abca3b9155a6d19c5c`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 21.83B total, 3.35B active (15.4% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 131,072  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-06-28  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 28× GQA  (FFN: 1 dense + 27 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 56.0 KiB (Low) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=64, top-6, +2 shared); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, RoPE, GQA, MoE, shared expert, MTP |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `ernie4_5_moe` |
| attention | GQA — 20 query : 4 kv heads (repeat 5), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=500000.0) |
| FFN | MoE — 64 routed experts, top-6 + 2 shared, expert intermediate 1536, SwiGLU (silu·gate) [grouped_mm] |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·4·128 = 1024 elems / token / layer; 28 attention layer(s) ⇒ 28672 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 28 |
| d_model | 2560 |
| n_h | 20 |
| n_kv | 4 |
| d_head | 128 |
| d_ff | 12288 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 103424 |
| ctx | 131072 |
| E | 64 |
| E_shared | 2 |
| k | 6 |
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

shape 축 **91,218개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 27,889 | 30.57% |
| 이 모듈 스코프의 심볼 | 24,434 | 26.79% |
| 스코프 없는 심볼 | 21,984 | 24.10% |
| 이 모듈 스코프의 유도식 | 14,717 | 16.13% |
| 이름 없음 (정수 유지) | 1,120 | 1.23% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 1,074 | 1.18% |

등록된 규칙 **89,024축**, 약한 근거 1,074축, 휴리스틱 **0축 (0.0%)**, 이름 없음 1,120축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 5 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 96 | k·T (라우팅된 (토큰, 슬롯) 쌍 수 — 토큰마다 expert k개) | act_fn, experts |
| 512 | n_kv·d_head (KV 투영 폭) | k_proj, self_attn, v_proj |
| 3072 | E_shared·d_moe (공유 전문가 FFN 폭 — 공유 전문가 수만큼 넓힌 하나의 MLP) | act_fn, down_proj, experts, gate_proj, shared_experts, up_proj |

## 레이어 구조

- layer 0: input_layernorm, mlp, post_attention_layernorm, self_attn
- layer 1-27: input_layernorm, mlp, post_attention_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 3개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 28 == 28 |
| C2 | WARN | 2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=2560 in 28/28 layers |
| C6 | PASS | hidden_size=2560 (heuristic check, 3136 flagged) |
| C7 | PASS | GQA 20:4 (repeat factor 5) |
| C8 | WARN | MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count ... |
| C9 | PASS | vocab_size=103424, tie_word_embeddings=True |
| C10 | PASS | all 362 params covered |
| C11 | PASS | 57 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=16 >= required=16 |
| C15 | WARN | config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers i... |
| C16 | INFO | 3007 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `baidu/ERNIE-4.5-21B-A3B-PT` config.json @ `87db95487941cb39592ee0abca3b9155a6d19c5c` (sha256 `a44a07dbeab6…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=16 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 반박 프레임 전건 판정)

의뢰서가 비어 있고 자기모순 0건이다. 기존 규칙만으로 전부 설명됐다 — 전용 규칙 0개.

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- baidu/ERNIE-4.5-21B-A3B-PT @ 87db95487941cb39592ee0abca3b9155a6d19c5c

C1   PASS   28 == 28
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2560 in 28/28 layers
C6   PASS   hidden_size=2560 (heuristic check, 3136 flagged)
C7   PASS   GQA 20:4 (repeat factor 5)
C8   WARN   MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=103424, tie_word_embeddings=True
C10  PASS   all 362 params covered
C11  PASS   57 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   3007 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div.Tensor', 'aten.empty_like.default', 'aten.expand.default']
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
  model.layers.N.self_attn                           slice            [B,1,T,d_head] -> [B,1,T,d_head/2]
  model.layers.N.self_attn                           unsqueeze        [B,1,T,d_head/2] -> [B,1,T,d_head/2,1]
  model.layers.N.self_attn                           expand           [B,1,T,d_head/2,1] -> [B,1,T,d_head/2,2]
  model.layers.N.self_attn                           clone            [B,1,T,d_head/2,2] -> [B,1,T,d_head/2,2]
  model.layers.N.self_attn                           view             [B,1,T,d_head/2,2] -> [B,1,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           stack            [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2,2]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head/2,2] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head]*[B,1,T,d_head] -> [B,n_kv,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,T,d_head] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2]
  model.layers.N.self_attn                           stack            [B,n_kv,T,d_head/2]*[B,n_kv,T,d_head/2] -> [B,n_kv,T,d_head/2,2]
  model.layers.N.self_attn                           view             [B,n_kv,T,d_head/2,2] -> [B,n_kv,T,d_head]
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
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           clone            [B,T,n_h,d_head] -> [B,T,n_h,d_head]
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
  model.layers.N.mlp                                 view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [T,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [T,E_shared*d_moe] -> [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [T,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [T,E_shared*d_moe]*[T,E_shared*d_moe] -> [T,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,E_shared*d_moe] -> w=[d_model,E_shared*d_moe] [E_shared*d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [T,E_shared*d_moe]*[E_shared*d_moe,d_model] -> w=[d_model,E_shared*d_moe] [T,d_model]
  model.layers.N.mlp.gate                            _to_copy         [T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [T,d_model]*[d_model,E] -> w=[E,d_model] [T,E]
  model.layers.N.mlp.gate                            softmax          [T,E] -> [T,E]
  model.layers.N.mlp.gate.moe_statics                squeeze          [1,E] -> w=[1,E] [E]
  model.layers.N.mlp.gate.moe_statics                elementwise_add  [T,E]*[E] -> [T,E]
  model.layers.N.mlp.gate                            topk             [T,E] -> [T,k]*[T,k]
  model.layers.N.mlp.gate                            gather           [T,E]*[T,k] -> [T,k]
  model.layers.N.mlp.gate                            sum              [T,k] -> [T,1]
  model.layers.N.mlp.gate                            clamp            [T,1] -> [T,1]
  model.layers.N.mlp.gate                            div              [T,k]*[T,1] -> [T,k]
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
  model.layers.N.mlp.experts                         unsqueeze        [k*T] -> [k*T,B]
  model.layers.N.mlp.experts                         clamp_           [k*T] -> [k*T]
  model.layers.N.mlp.experts                         masked_fill_     [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k*T,2*d_moe]
  model.layers.N.mlp.experts                         split            [k*T,2*d_moe] -> [k*T,d_moe]*[k*T,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_moe]*[k*T,d_moe] -> [k*T,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k*T,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k*T,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k*T,d_model]*[k*T,B] -> [k*T,d_model]
  model.layers.N.mlp.experts                         empty_like       [k*T] -> [k*T]
  model.layers.N.mlp.experts                         arange           [] -> [k*T]
  model.layers.N.mlp.experts                         index_put_       [k*T]*[k*T]*[k*T] -> [k*T]
  model.layers.N.mlp.experts                         index            [k*T,d_model]*[k*T] -> [k*T,d_model]
  model.layers.N.mlp.experts                         view             [k*T,d_model] -> [T,k,d_model]
  model.layers.N.mlp.experts                         sum              [T,k,d_model] -> [T,d_model]
  model.layers.N.mlp                                 elementwise_add  [T,d_model]*[T,d_model] -> [T,d_model]
  model.layers.N.mlp                                 view             [T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn                           slice            [B,1,1,d_head] -> [B,1,1,d_head/2]
  model.layers.N.self_attn                           unsqueeze        [B,1,1,d_head/2] -> [B,1,1,d_head/2,1]
  model.layers.N.self_attn                           expand           [B,1,1,d_head/2,1] -> [B,1,1,d_head/2,2]
  model.layers.N.self_attn                           clone            [B,1,1,d_head/2,2] -> [B,1,1,d_head/2,2]
  model.layers.N.self_attn                           view             [B,1,1,d_head/2,2] -> [B,1,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           stack            [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2,2]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head/2,2] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head]*[B,1,1,d_head] -> [B,n_kv,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_kv,1,d_head] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2]
  model.layers.N.self_attn                           stack            [B,n_kv,1,d_head/2]*[B,n_kv,1,d_head/2] -> [B,n_kv,1,d_head/2,2]
  model.layers.N.self_attn                           view             [B,n_kv,1,d_head/2,2] -> [B,n_kv,1,d_head]
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
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,T+1]*[B,1,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           expand           [B,n_h,T+1,d_head] -> [B,n_h,T+1,d_head]
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
  model.layers.N.mlp                                 view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.shared_experts.gate_proj        t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.gate_proj        matmul           [B,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.act_fn           silu             [B,E_shared*d_moe] -> [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          t                [E_shared*d_moe,d_model] -> w=[E_shared*d_moe,d_model] [d_model,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.up_proj          matmul           [B,d_model]*[d_model,E_shared*d_moe] -> w=[E_shared*d_moe,d_model] [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts                  elementwise_mul  [B,E_shared*d_moe]*[B,E_shared*d_moe] -> [B,E_shared*d_moe]
  model.layers.N.mlp.shared_experts.down_proj        t                [d_model,E_shared*d_moe] -> w=[d_model,E_shared*d_moe] [E_shared*d_moe,d_model]
  model.layers.N.mlp.shared_experts.down_proj        matmul           [B,E_shared*d_moe]*[E_shared*d_moe,d_model] -> w=[d_model,E_shared*d_moe] [B,d_model]
  model.layers.N.mlp.gate                            _to_copy         [B,d_model] -> [B,d_model]
  model.layers.N.mlp.gate                            t                [E,d_model] -> w=[E,d_model] [d_model,E]
  model.layers.N.mlp.gate                            matmul           [B,d_model]*[d_model,E] -> w=[E,d_model] [B,E]
  model.layers.N.mlp.gate                            softmax          [B,E] -> [B,E]
  model.layers.N.mlp.gate.moe_statics                squeeze          [1,E] -> w=[1,E] [E]
  model.layers.N.mlp.gate.moe_statics                elementwise_add  [B,E]*[E] -> [B,E]
  model.layers.N.mlp.gate                            topk             [B,E] -> [B,k]*[B,k]
  model.layers.N.mlp.gate                            gather           [B,E]*[B,k] -> [B,k]
  model.layers.N.mlp.gate                            sum              [B,k] -> [B,1]
  model.layers.N.mlp.gate                            clamp            [B,1] -> [B,1]
  model.layers.N.mlp.gate                            div              [B,k]*[B,1] -> [B,k]
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
  model.layers.N.mlp.experts                         unsqueeze        [k] -> [k,B]
  model.layers.N.mlp.experts                         clamp_           [k] -> [k]
  model.layers.N.mlp.experts                         masked_fill_     [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         transpose        [E,2*d_moe,d_model] -> w=[E,2*d_moe,d_model] [E,d_model,2*d_moe]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_model]*[E,d_model,2*d_moe]*[E] -> w=[E,2*d_moe,d_model] [k,2*d_moe]
  model.layers.N.mlp.experts                         split            [k,2*d_moe] -> [k,d_moe]*[k,d_moe]
  model.layers.N.mlp.experts.act_fn                  silu             [k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_moe]*[k,d_moe] -> [k,d_moe]
  model.layers.N.mlp.experts                         transpose        [E,d_model,d_moe] -> w=[E,d_model,d_moe] [E,d_moe,d_model]
  model.layers.N.mlp.experts                         grouped_matmul   [k,d_moe]*[E,d_moe,d_model]*[E] -> w=[E,d_model,d_moe] [k,d_model]
  model.layers.N.mlp.experts                         elementwise_mul  [k,d_model]*[k,B] -> [k,d_model]
  model.layers.N.mlp.experts                         empty_like       [k] -> [k]
  model.layers.N.mlp.experts                         arange           [] -> [k]
  model.layers.N.mlp.experts                         index_put_       [k]*[k]*[k] -> [k]
  model.layers.N.mlp.experts                         index            [k,d_model]*[k] -> [k,d_model]
  model.layers.N.mlp.experts                         view             [k,d_model] -> [B,k,d_model]
  model.layers.N.mlp.experts                         sum              [B,k,d_model] -> [B,d_model]
  model.layers.N.mlp                                 elementwise_add  [B,d_model]*[B,d_model] -> [B,d_model]
  model.layers.N.mlp                                 view             [B,d_model] -> [B,1,d_model]
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



# ============================================================
# 모델: google__gemma-3-270m
# ============================================================

# 리뷰 패킷 — google/gemma-3-270m

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` / 트레이스 seq_len(T) = 1032
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
  L            = 18
  d_model      = 640
  n_h          = 4
  n_kv         = 1
  d_head       = 256
  d_ff         = 2048
  d_shared     = None
  V            = 262144
  ctx          = 32768
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  d_moe_lat    = None
  w_local      = 512
  n_sink       = None
  layer_sched  = ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention']
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

# Model Summary -- google/gemma-3-270m

## 기본 정보

- revision: `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 1032
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 268.1M total (dense) |
| 2 | Context (tokens) | 32,768  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-08-05  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | MQA |
| 6 | LAYER MIX | 15× sliding_attention, 3× full_attention  (attention: MQA) |
| 7 | KV CACHE / TOKEN (BF16) | 18.0 KiB (Very low) |
| 8 | KEY DETAIL | MQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, MQA, QK-Norm |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `gemma3_text` |
| attention | MQA — 4 query heads, 1 kv head, d_head=256; sliding window 512 on part of layers (hybrid local/global) |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE |
| FFN | dense FFN — intermediate 2048, GELU |
| 정규화 | RMSNorm |
| tie embeddings | True |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·1·256 = 512 elems / token / layer; 18 attention layer(s) ⇒ 9216 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 18 |
| d_model | 640 |
| n_h | 4 |
| n_kv | 1 |
| d_head | 256 |
| d_ff | 2048 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 262144 |
| ctx | 32768 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| d_moe_lat | —  _(해당 없음: 이 모델은 `kda_attn` 계열 구조를 쓰지 않음)_ |
| w_local | 512 |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 15× sliding_attention, 3× full_attention (총 18층) |
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

shape 축 **58,189개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 23,981 | 41.21% |
| 스코프 없는 심볼 | 16,787 | 28.85% |
| 이 모듈 스코프의 심볼 | 14,849 | 25.52% |
| 이 모듈 스코프의 유도식 | 1,852 | 3.18% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 648 | 1.11% |
| 이름 없음 (정수 유지) | 72 | 0.12% |

등록된 규칙 **57,469축**, 약한 근거 648축, 휴리스틱 **0축 (0.0%)**, 이름 없음 72축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 128 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 511 | w_local − 1 (sliding window mask 밴드 폭) | self_attn |
| 1024 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_proj, self_attn |

## 레이어 구조

- layer 0-4: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn
- layer 5: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn
- layer 6-10: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn
- layer 11: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn
- layer 12-16: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn
- layer 17: input_layernorm, mlp, post_attention_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=PASS)

| check | status | detail |
|---|---|---|
| C1 | PASS | 18 == 18 |
| C2 | PASS | 2 clusters == 2 from config schedule ['layer_types'] |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=640 in 18/18 layers |
| C6 | PASS | hidden_size=640 (heuristic check, 1752 flagged) |
| C7 | PASS | MQA (4 query heads : 1 kv head) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=262144, tie_word_embeddings=True |
| C10 | PASS | all 236 params covered |
| C11 | PASS | 74 cache-related op(s) found, new-token seq dim confirmed |
| C13 | PASS | identical across two runs |
| C14 | PASS | used=1032 >= required=1032 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 1550 unmapped rows, 21 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `google/gemma-3-270m` config.json @ `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` (sha256 `d9879f533459…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=1032 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-12 · llm(claude, 반박 프레임 전건 판정)

의뢰서가 비어 있었다 — 이 모델의 축은 전부 등록된 규칙이 이름을 냈고 소스 대조에서도 어긋난 곳이 없다.

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- google/gemma-3-270m @ 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1

C1   PASS   18 == 18
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=640 in 18/18 layers
C6   PASS   hidden_size=640 (heuristic check, 1752 flagged)
C7   PASS   MQA (4 query heads : 1 kv head)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=262144, tie_word_embeddings=True
C10  PASS   all 236 params covered
C11  PASS   74 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=1032 >= required=1032
C15  PASS   all discovered entrypoints traced
C16  INFO   1550 unmapped rows, 21 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clone.default', 'aten.expand.default', 'aten.gt.Tensor', 'aten.le.Tensor', 'aten.lift_fresh.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model.embed_tokens                                 elementwise_mul  [B,T,d_model]*[] -> [B,T,d_model]
  model                                              lift_fresh       [] -> []
  model                                              arange           [] -> [B]
  model                                              arange           [] -> [T]
  model                                              elementwise_add  [T] -> [T]
  model                                              unsqueeze        [B] -> [B,1]
  model                                              unsqueeze        [B,1] -> [B,1,1]
  model                                              unsqueeze        [B,1,1] -> [B,1,1,1]
  model                                              unsqueeze        [T] -> [B,T]
  model                                              unsqueeze        [B,T] -> [B,1,T]
  model                                              unsqueeze        [B,1,T] -> [B,1,T,1]
  model                                              new_ones         [B,1,T,1] -> []
  model                                              sub              [B,1,T,1] -> [B,1,T,1]
  model                                              gt               [B,1,1,T]*[B,1,T,1] -> [B,1,T,T]
  model                                              bitwise_and      []*[B,1,T,T] -> [B,1,T,T]
  model                                              le               [B,1,1,T]*[B,1,T,1] -> [B,1,T,T]
  model                                              bitwise_and      [B,1,T,T]*[B,1,T,T] -> [B,1,T,T]
  model                                              expand           [B,1,T,T] -> [B,1,T,T]
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
  model.layers.N.input_layernorm                     _to_copy         [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_add  [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [T,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [T,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [T,n_h*d_head] -> [B,T,n_h*d_head]
  model.layers.N.self_attn                           view             [B,T,n_h*d_head] -> [B,T,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,T,n_h,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.k_proj                    t                [d_head,d_model] -> w=[d_head,d_model] [d_model,d_head]
  model.layers.N.self_attn.k_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [T,d_model]*[d_model,d_head] -> w=[d_head,d_model] [T,d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [T,d_head] -> [B,T,d_head]
  model.layers.N.self_attn                           view             [B,T,d_head] -> [B,T,1,d_head]
  model.layers.N.self_attn                           transpose        [B,T,1,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn.v_proj                    t                [d_head,d_model] -> w=[d_head,d_model] [d_model,d_head]
  model.layers.N.self_attn.v_proj                    view             [B,T,d_model] -> [T,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [T,d_model]*[d_model,d_head] -> w=[d_head,d_model] [T,d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [T,d_head] -> [B,T,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,n_h,T,d_head] -> [B,n_h,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,n_h,T,1] -> [B,n_h,T,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,n_h,T,1] -> [B,n_h,T,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,n_h,T,d_head]*[B,n_h,T,1] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,n_h,T,d_head]*[d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,1,T,d_head] -> [B,1,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,1,T,1] -> [B,1,T,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,1,T,1] -> [B,1,T,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,T,d_head]*[B,1,T,1] -> [B,1,T,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,T,d_head]*[d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head]*[B,1,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,T,d_head] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,T,d_head/2] -> [B,n_h,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,T,d_head/2]*[B,n_h,T,d_head/2] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,d_head]*[B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,1,T,d_head]*[B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           slice            [B,1,T,d_head] -> [B,1,T,d_head/2]
  model.layers.N.self_attn                           neg              [B,1,T,d_head/2] -> [B,1,T,d_head/2]
  model.layers.N.self_attn                           concat           [B,1,T,d_head/2]*[B,1,T,d_head/2] -> [B,1,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,1,T,d_head]*[B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           _to_copy         [] -> []
  model.layers.N.self_attn                           concat           [0]*[B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           slice            [B,1,T,d_head] -> [B,1,w_local-1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,T,d_head] -> [B,1,1,T,d_head]
  model.layers.N.self_attn                           expand           [B,1,1,T,d_head] -> [B,1,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,1,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           scalar_tensor    [] -> []
  model.layers.N.self_attn                           where            [B,1,T,T]*[]*[] -> [B,1,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T,d_head] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,d_head] -> [B,n_h,T,d_head]
  model.layers.N.self_attn                           view             [B,n_h,T,d_head] -> [n_h,T,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,T] -> [B,n_h,d_head,T]
  model.layers.N.self_attn                           view             [B,n_h,d_head,T] -> [n_h,d_head,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,d_head]*[n_h,d_head,T] -> [n_h,T,T]
  model.layers.N.self_attn                           _unsafe_view     [n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[B,1,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           softmax          [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           _to_copy         [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           expand           [B,n_h,T,T] -> [B,n_h,T,T]
  model.layers.N.self_attn                           view             [B,n_h,T,T] -> [n_h,T,T]
  model.layers.N.self_attn                           batched_matmul   [n_h,T,T]*[n_h,T,d_head] -> [n_h,T,d_head]
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
  model.layers.N.post_attention_layernorm            _to_copy         [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.pre_feedforward_layernorm           _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.pre_feedforward_layernorm           pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.pre_feedforward_layernorm           mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.pre_feedforward_layernorm           elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.pre_feedforward_layernorm           rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.pre_feedforward_layernorm           elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.pre_feedforward_layernorm           _to_copy         [d_model] -> [d_model]
  model.layers.N.pre_feedforward_layernorm           elementwise_add  [d_model] -> [d_model]
  model.layers.N.pre_feedforward_layernorm           elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.mlp.act_fn                          gelu             [B,T,d_ff] -> [B,T,d_ff]
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
  model.layers.N.post_feedforward_layernorm          _to_copy         [d_model] -> [d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [B,T,d_model]*[d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.self_attn                           _to_copy         [B,1,T,d_head] -> [B,1,T,d_head]
  model.layers.N.self_attn                           ones             [] -> [T,T]
  model.layers.N.self_attn                           tril             [T,T] -> [T,T]
  model.layers.N.self_attn                           where            [T,T]*[]*[] -> [T,T]
  model.layers.N.self_attn                           clone            [B,1,n_h,T,d_head] -> [B,1,n_h,T,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,T,T]*[T,T] -> [B,n_h,T,T]
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
  model.embed_tokens                                 elementwise_mul  [B,1,d_model]*[] -> [B,1,d_model]
  model                                              arange           [] -> [B]
  model                                              elementwise_add  [B] -> [B]
  model                                              arange           [] -> [w_local]
  model                                              elementwise_add  [w_local] -> [w_local]
  model                                              unsqueeze        [B] -> [B,1]
  model                                              unsqueeze        [B,1] -> [B,1,1]
  model                                              unsqueeze        [B,1,1] -> [B,1,1,1]
  model                                              unsqueeze        [w_local] -> [B,w_local]
  model                                              unsqueeze        [B,w_local] -> [B,1,w_local]
  model                                              unsqueeze        [B,1,w_local] -> [B,1,1,w_local]
  model                                              new_ones         [B,1,1,1] -> []
  model                                              sub              [B,1,1,1] -> [B,1,1,1]
  model                                              gt               [B,1,1,w_local]*[B,1,1,1] -> [B,1,1,w_local]
  model                                              bitwise_and      []*[B,1,1,w_local] -> [B,1,1,w_local]
  model                                              le               [B,1,1,w_local]*[B,1,1,1] -> [B,1,1,w_local]
  model                                              bitwise_and      [B,1,1,w_local]*[B,1,1,w_local] -> [B,1,1,w_local]
  model                                              expand           [B,1,1,w_local] -> [B,1,1,w_local]
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
  model.layers.N.input_layernorm                     _to_copy         [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_add  [d_model] -> [d_model]
  model.layers.N.input_layernorm                     elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.N.self_attn.q_proj                    t                [n_h*d_head,d_model] -> w=[n_h*d_head,d_model] [d_model,n_h*d_head]
  model.layers.N.self_attn.q_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.q_proj                    matmul           [B,d_model]*[d_model,n_h*d_head] -> w=[n_h*d_head,d_model] [B,n_h*d_head]
  model.layers.N.self_attn.q_proj                    _unsafe_view     [B,n_h*d_head] -> [B,1,n_h*d_head]
  model.layers.N.self_attn                           view             [B,1,n_h*d_head] -> [B,1,n_h,d_head]
  model.layers.N.self_attn                           transpose        [B,1,n_h,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.k_proj                    t                [d_head,d_model] -> w=[d_head,d_model] [d_model,d_head]
  model.layers.N.self_attn.k_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.k_proj                    matmul           [B,d_model]*[d_model,d_head] -> w=[d_head,d_model] [B,d_head]
  model.layers.N.self_attn.k_proj                    _unsafe_view     [B,d_head] -> [B,1,d_head]
  model.layers.N.self_attn                           view             [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           transpose        [B,1,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn.v_proj                    t                [d_head,d_model] -> w=[d_head,d_model] [d_model,d_head]
  model.layers.N.self_attn.v_proj                    view             [B,1,d_model] -> [B,d_model]
  model.layers.N.self_attn.v_proj                    matmul           [B,d_model]*[d_model,d_head] -> w=[d_head,d_model] [B,d_head]
  model.layers.N.self_attn.v_proj                    _unsafe_view     [B,d_head] -> [B,1,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.q_norm                    pow              [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.q_norm                    mean             [B,n_h,1,d_head] -> [B,n_h,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_add  [B,n_h,1,1] -> [B,n_h,1,1]
  model.layers.N.self_attn.q_norm                    rsqrt            [B,n_h,1,1] -> [B,n_h,1,1]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,n_h,1,d_head]*[B,n_h,1,1] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.q_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.q_norm                    elementwise_mul  [B,n_h,1,d_head]*[d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [B,1,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn.k_norm                    pow              [B,1,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn.k_norm                    mean             [B,1,1,d_head] -> [B,1,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_add  [B,1,1,1] -> [B,1,1,1]
  model.layers.N.self_attn.k_norm                    rsqrt            [B,1,1,1] -> [B,1,1,1]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,1,d_head]*[B,1,1,1] -> [B,1,1,d_head]
  model.layers.N.self_attn.k_norm                    _to_copy         [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_add  [d_head] -> [d_head]
  model.layers.N.self_attn.k_norm                    elementwise_mul  [B,1,1,d_head]*[d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head]*[B,1,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           slice            [B,n_h,1,d_head] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,n_h,1,d_head/2] -> [B,n_h,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,n_h,1,d_head/2]*[B,n_h,1,d_head/2] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,d_head]*[B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,1,1,d_head]*[B,1,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           slice            [B,1,1,d_head] -> [B,1,1,d_head/2]
  model.layers.N.self_attn                           neg              [B,1,1,d_head/2] -> [B,1,1,d_head/2]
  model.layers.N.self_attn                           concat           [B,1,1,d_head/2]*[B,1,1,d_head/2] -> [B,1,1,d_head]
  model.layers.N.self_attn                           elementwise_add  [B,1,1,d_head]*[B,1,1,d_head] -> [B,1,1,d_head]
  model.layers.N.self_attn                           concat           [B,1,w_local-1,d_head]*[B,1,1,d_head] -> [B,1,w_local,d_head]
  model.layers.N.self_attn                           slice            [B,1,w_local,d_head] -> [B,1,w_local-1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,w_local,d_head] -> [B,1,1,w_local,d_head]
  model.layers.N.self_attn                           expand           [B,1,1,w_local,d_head] -> [B,1,n_h,w_local,d_head]
  model.layers.N.self_attn                           view             [B,1,n_h,w_local,d_head] -> [B,n_h,w_local,d_head]
  model.layers.N.self_attn                           scalar_tensor    [] -> []
  model.layers.N.self_attn                           where            [B,1,1,w_local]*[]*[] -> [B,1,1,w_local]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,n_h,w_local,d_head] -> [B,n_h,w_local,d_head]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,w_local,d_head] -> [B,n_h,d_head,w_local]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,w_local] -> [B,n_h,d_head,w_local]
  model.layers.N.self_attn                           expand           [B,n_h,1,d_head] -> [B,n_h,1,d_head]
  model.layers.N.self_attn                           view             [B,n_h,1,d_head] -> [n_h,B,d_head]
  model.layers.N.self_attn                           expand           [B,n_h,d_head,w_local] -> [B,n_h,d_head,w_local]
  model.layers.N.self_attn                           view             [B,n_h,d_head,w_local] -> [n_h,d_head,w_local]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,w_local] -> [n_h,B,w_local]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,w_local] -> [B,n_h,1,w_local]
  model.layers.N.self_attn                           elementwise_add  [B,n_h,1,w_local]*[B,1,1,w_local] -> [B,n_h,1,w_local]
  model.layers.N.self_attn                           softmax          [B,n_h,1,w_local] -> [B,n_h,1,w_local]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,w_local] -> [B,n_h,1,w_local]
  model.layers.N.self_attn                           expand           [B,n_h,1,w_local] -> [B,n_h,1,w_local]
  model.layers.N.self_attn                           view             [B,n_h,1,w_local] -> [n_h,B,w_local]
  model.layers.N.self_attn                           expand           [B,n_h,w_local,d_head] -> [B,n_h,w_local,d_head]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,w_local]*[n_h,w_local,d_head] -> [n_h,B,d_head]
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
  model.layers.N.post_attention_layernorm            _to_copy         [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_attention_layernorm            elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_feedforward_layernorm           _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_feedforward_layernorm           pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_feedforward_layernorm           mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.pre_feedforward_layernorm           elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.pre_feedforward_layernorm           rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.pre_feedforward_layernorm           elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.pre_feedforward_layernorm           _to_copy         [d_model] -> [d_model]
  model.layers.N.pre_feedforward_layernorm           elementwise_add  [d_model] -> [d_model]
  model.layers.N.pre_feedforward_layernorm           elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.N.mlp.gate_proj                       t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.mlp.gate_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mlp.gate_proj                       matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.mlp.gate_proj                       _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.mlp.act_fn                          gelu             [B,1,d_ff] -> [B,1,d_ff]
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
  model.layers.N.post_feedforward_layernorm          _to_copy         [d_model] -> [d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_add  [d_model] -> [d_model]
  model.layers.N.post_feedforward_layernorm          elementwise_mul  [B,1,d_model]*[d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.self_attn                           concat           [B,1,T,d_head]*[B,1,1,d_head] -> [B,1,T+1,d_head]
  model.layers.N.self_attn                           _to_copy         [B,1,T+1,d_head] -> [B,1,T+1,d_head]
  model.layers.N.self_attn                           unsqueeze        [B,1,T+1,d_head] -> [B,1,1,T+1,d_head]
  model.layers.N.self_attn                           expand           [B,1,1,T+1,d_head] -> [B,1,n_h,T+1,d_head]
  model.layers.N.self_attn                           clone            [B,1,n_h,T+1,d_head] -> [B,1,n_h,T+1,d_head]
  model.layers.N.self_attn                           transpose        [B,n_h,T+1,d_head] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           elementwise_mul  [B,n_h,d_head,T+1] -> [B,n_h,d_head,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,d_head]*[n_h,d_head,T+1] -> [n_h,B,T+1]
  model.layers.N.self_attn                           _unsafe_view     [n_h,B,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           softmax          [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           _to_copy         [B,n_h,1,T+1] -> [B,n_h,1,T+1]
  model.layers.N.self_attn                           batched_matmul   [n_h,B,T+1]*[n_h,T+1,d_head] -> [n_h,B,d_head]
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

