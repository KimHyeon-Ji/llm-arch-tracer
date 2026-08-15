# 리뷰 패킷 — tiiuae/Falcon-H1-7B-Instruct

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `41e72f27effbab80cd45b6e884688452253a3686` / 트레이스 seq_len(T) = 17
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
  L            = 44
  d_model      = 3072
  n_h          = 12
  n_kv         = 2
  d_head       = 128
  d_ff         = 12288
  d_shared     = None
  V            = 130049
  ctx          = 262144
  E            = None
  E_shared     = None
  k            = None
  n_grp        = None
  k_grp        = None
  d_moe        = None
  w_local      = None
  n_sink       = None
  layer_sched  = ['hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid', 'hybrid']
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
  d_state      = 256
  n_g_ssm      = 1
  n_h_ssm      = 24
  d_chunk      = 256
  d_head_ssm   = 128
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

# Model Summary -- tiiuae/Falcon-H1-7B-Instruct

## 기본 정보

- revision: `41e72f27effbab80cd45b6e884688452253a3686`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 17
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 7.59B total (dense) |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings)_ |
| 3 | DATE | 2025-05-01  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Dense |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 44× GQA |
| 7 | KV CACHE / TOKEN (BF16) | 44.0 KiB (Low) |
| 8 | KEY DETAIL | GQA attention; dense FFN |
| 9 | Related concepts | RMSNorm, RoPE, GQA, short-conv (SSM/DeltaNet) |

_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo 메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 보강._

ref) 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. (7)은 같은 갤러리의 [KV cache 계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** 합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.

## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)

| 항목 | 값 |
|---|---|
| 모델 타입 (config) | `falcon_h1` |
| attention | GQA — 12 query : 2 kv heads (repeat 6), d_head=128 |
| attention 커널 | eager (explicit softmax) |
| 위치 인코딩 | RoPE (θ=100000000000.0) |
| FFN | dense FFN — intermediate 12288, SwiGLU (silu·gate) |
| 정규화 | RMSNorm |
| tie embeddings | False |
| decode 방식 | autoregressive, 1 token/step, reuses KV cache (prefill builds it) |
| KV cache 크기 | 2·n_kv·d_head = 2·2·128 = 512 elems / token / layer; all 44 layers ⇒ 22528 / token |

## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)

| symbol | value |
|---|---|
| L | 44 |
| d_model | 3072 |
| n_h | 12 |
| n_kv | 2 |
| d_head | 128 |
| d_ff | 12288 |
| d_shared | —  _(해당 없음: 이 모델은 `moe_shared_width` 계열 구조를 쓰지 않음)_ |
| V | 130049 |
| ctx | 262144 |
| E | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| E_shared | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| k | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| n_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| k_grp | —  _(해당 없음: 이 모델은 `moe_grouped` 계열 구조를 쓰지 않음)_ |
| d_moe | —  _(해당 없음: 이 모델은 `moe` 계열 구조를 쓰지 않음)_ |
| w_local | —  _(해당 없음: 이 모델은 `sliding` 계열 구조를 쓰지 않음)_ |
| n_sink | —  _(해당 없음: 이 모델은 `attn_sink` 계열 구조를 쓰지 않음)_ |
| layer_sched | 44× hybrid |
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
| d_state | 256 |
| n_g_ssm | 1 |
| n_h_ssm | 24 |
| d_chunk | 256 |
| d_head_ssm | 128 |
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

shape 축 **239,874개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. 위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. 후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, `02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.

| 근거 | 축 수 | 비율 |
|---|---:|---:|
| 런타임 축 (B/T/1) | 87,663 | 36.55% |
| 이 모듈 스코프의 심볼 | 82,743 | 34.49% |
| 스코프 없는 심볼 | 37,195 | 15.51% |
| 이 모듈 스코프의 유도식 | 23,991 | 10.00% |
| 이름 없음 (정수 유지) | 4,312 | 1.80% |
| 같은 shape에서 이미 쓴 심볼 재사용 | 3,970 | 1.66% |

등록된 규칙 **231,592축**, 약한 근거 3,970축, 휴리스틱 **0축 (0.0%)**, 이름 없음 4,312축.

## 유도 상수 (합성 차원 범례)

심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. 표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). 설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).

| 값 | 유래 | 나타나는 모듈 |
|---|---|---|
| 6 | n_h/n_kv (GQA repeat 계수 — repeat_kv의 expand 축) | self_attn |
| 20 | T + d_conv − 1 (causal conv1d 좌측 패딩 포함 길이) | conv1d, mamba |
| 64 | d_head/2 (RoPE rotate_half 분할 축) | rotary_emb, self_attn |
| 1536 | n_h·d_head (Q 투영 폭 / attention 출력 폭) | o_proj, q_proj, self_attn |
| 3584 | d_inner + 2·n_g·d_state (conv1d 입력 폭) | act, conv1d, mamba |
| 6680 | 2·d_inner + 2·n_g·d_state + n_h_ssm (Mamba in_proj 출력: gate+x, B+C, dt) | in_proj, mamba |

## 레이어 구조

- layer 0-43: feed_forward, input_layernorm, mamba, pre_ff_layernorm, self_attn

## 검증 로그 (01-main.md §9 체크리스트)

- **종합: PASS** (WARN 0개, 재현성 C13=SKIP)

| check | status | detail |
|---|---|---|
| C1 | PASS | 44 == 44 |
| C2 | PASS | 1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like f... |
| C3 | PASS | acyclic, 0 orphan(s) |
| C4 | PASS | embedding reachable from lm_head |
| C5 | PASS | matmul contraction dims consistent; residual stream at d_model=3072 in 44/44 layers |
| C6 | PASS | hidden_size=3072 (heuristic check, 2816 flagged) |
| C7 | PASS | GQA 12:2 (repeat factor 6) |
| C8 | SKIP | no MoE-related fields found on config (likely a dense model) |
| C9 | PASS | vocab_size=130049, tie_word_embeddings=False |
| C10 | PASS | all 751 params covered |
| C11 | PASS | 221 cache-related op(s) found, new-token seq dim confirmed |
| C13 | SKIP | pass --check-repro to actually run twice and verify |
| C14 | PASS | used=17 >= required=16 |
| C15 | PASS | all discovered entrypoints traced |
| C16 | INFO | 7431 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', '... |
| C17 | PASS | 유도 상수 전부 설명됨, 구조 라이브러리에 등재됨 |

## 추출 방법

01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward 실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). 아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다.

## 구성 근거 / 소스

이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):

| 구분 | 소스 | 역할 |
|---|---|---|
| config (1차) | HF `tiiuae/Falcon-H1-7B-Instruct` config.json @ `41e72f27effbab80cd45b6e884688452253a3686` (sha256 `8d1f1e8fcc45…`) | 심볼 값의 출처 |
| modeling code (1차) | transformers 5.14.1 공식 modeling forward (meta device) | op·shape·dependency 캡처 |
| trace (1차) | dispatch(ATen) 레벨, seq_len(T)=17 | 표·그래프 생성 근거 |

교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):

_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), 공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_

## ③ 라벨 검토 — 소스와 대조한 결과

2026-08-15 · llm(claude, 블라인드 온보딩 테스트의 ③ 소스 대조)

3건 전부 판정했다. 2건은 오라벨로 확정(1건 교정 완료, 1건은 대체할 심볼이 없어 open), 1건은 관례 선택이 맞았음을 확인했다.

| 판정 | 건수 |
|---|---|
| 맞음 | 1 |
| 교정 필요 | 2 |

### 소스 판정으로 교정된 라벨

규칙으로는 도달할 수 없는 축이다(두 config 값이 같아 값으로 결정할 게 없다). 소스를 읽어 확정하고 **표에 반영했다** — 근거는 `rules/label_overrides.yaml`, 적용 내역은 `full/label_overrides.json`. 게이트가 매 실행마다 이 교정이 실제로 발화하는지 확인한다.

| 모듈 | 이전 | 이후 | 축 | 근거 |
|---|---|---|---|---|
| `mamba$` | `d_state` | `d_chunk` | 264 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py `segment_sum`: `mask = torch.tril(torch.ones(chunk_size, chunk_size, ...), diagonal=-1)` — 양 축 모두 chunk_size 다. 이 모델은 mamba_chunk_size == mamba_d_state == 256 이라 값으로는 원리적으로 못 가리고, 정사각 마스크를 무엇으로 짓는지가 유일한 근거다. |
| `mamba$` | `d_state` | `d_chunk` | 264 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. 같은 `segment_sum` 의 둘째 `torch.ones(chunk_size, chunk_size)` (`diagonal=0` 마스크). 첫째와 같은 근거다. |
| `mamba$` | `d_state` | `d_chunk` | 396 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 B/C는 `[B,T,num_heads,state_size]`이고 :299-316은 sequence 축만 pad한다. 축 1은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 308 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-520에서 hidden_states는 `[B,T,num_heads,head_dim]`이고 :299-316은 sequence 축만 pad한다. nth 2 출력의 축 1은 d_chunk다. |
| `mamba$` | `d_chunk` | `d_state` | 396 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 C는 `[B,T,num_heads,state_size]`이고 sequence 축만 pad된다. nth 5 출력의 마지막 축은 d_state다. |
| `mamba$` | `d_state` | `d_chunk` | 264 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:529-533의 G/M 경로는 `(b,c,l,s,h,n)`에서 state 축을 합친 뒤 singleton을 붙인다. `[B,c,l,s,h,1]`의 축 2는 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 704 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-520은 hidden_states를 `[B,T,num_heads,head_dim]`으로 읽고 :299-316이 sequence 축을 chunk_size 배수로 pad한다. 축 1의 256은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 616 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-523에서 A는 chunk reshape 뒤 `permute(0,3,1,2)`로 `[B,num_heads,n_chunks,chunk_size]`가 된다. 마지막 축은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_chunk` | `d_state` | 484 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 B/C는 num_heads로 repeat된 `[B,T,num_heads,state_size]`이고 :299-316은 sequence 축만 pad한다. 따라서 `[B,d_chunk,n_h_ssm,state_size]`의 마지막 축은 d_chunk이 아니라 d_state다. |
| `mamba$` | `d_state` | `d_chunk` | 440 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:319-335의 `segment_sum` expand는 마지막 chunk_size 축을 하나 더 만들어 `[...,chunk_size,chunk_size]`를 구성한다. :527의 첫 intra-chunk 호출에서 축 3은 d_state가 아니라 d_chunk다. |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mamba` | [B, n_h_ssm, n_kv, n_kv] (실제 [1, 24, 2, 2]) | `n_kv` | `(청크 개수 축 -- 이름 없는 정수로 남겨야 한다)` | `n_kv` 가 아니다. 이 축은 **inter-chunk 재귀의 청크 개수**다: 트레이스 op143 이 `[B, n_h_ssm, 1] -> [B, n_h_ssm, 2]` 로 pad 하는데(초기 상태 1칸을 앞에 붙임), 그 결과 폭 2 = 청크 1개 + 초기 상태 1개다. 그 뒤 `segment_sum` 이 다시 불려 `[2, 2]` 감쇠 행렬을 만든다 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.


## 4. 검증 체크리스트 결과

```
# Extraction Report -- tiiuae/Falcon-H1-7B-Instruct @ 41e72f27effbab80cd45b6e884688452253a3686

C1   PASS   44 == 44
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=3072 in 44/44 layers
C6   PASS   hidden_size=3072 (heuristic check, 2816 flagged)
C7   PASS   GQA 12:2 (repeat factor 6)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=130049, tie_word_embeddings=False
C10  PASS   all 751 params covered
C11  PASS   221 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   7431 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨

```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,T] -> w=[V,d_model] [B,T,d_model]
  model                                              elementwise_mul  [B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.mamba                               elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.N.mamba.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       view             [B,T,d_model] -> [T,d_model]
  model.layers.N.mamba.in_proj                       matmul           [T,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       _unsafe_view     [T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]*[1,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[1,1,2*d_inner+2*n_g*d_state+n_h_ssm] [B,T,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               split_with_sizes [B,T,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,T,d_inner]*[B,T,d_inner+2*n_g*d_state]*[B,T,n_h_ssm]
  model.layers.N.mamba                               transpose        [B,T,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba                               constant_pad_nd  [B,d_inner+2*n_g*d_state,T] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               zeros            [] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba.conv1d                        conv1d           [B,d_inner+2*n_g*d_state,T]*[d_inner+2*n_g*d_state,1,d_conv]*[d_inner+2*n_g*d_state] -> w=[d_inner+2*n_g*d_state,1,d_conv] [B,d_inner+2*n_g*d_state,T+d_conv-1]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,T+d_conv-1] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba.act                           silu             [B,d_inner+2*n_g*d_state,T] -> [B,d_inner+2*n_g*d_state,T]
  model.layers.N.mamba                               transpose        [B,d_inner+2*n_g*d_state,T] -> [B,T,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               split_with_sizes [B,T,d_inner+2*n_g*d_state] -> [B,T,d_inner]*[B,T,d_state]*[B,T,d_state]
  model.layers.N.mamba                               _to_copy         [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               exp              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               neg              [n_h_ssm] -> [n_h_ssm]
  model.layers.N.mamba                               elementwise_add  [B,T,n_h_ssm]*[n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               softplus         [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               clamp            [B,T,n_h_ssm] -> [B,T,n_h_ssm]
  model.layers.N.mamba                               view             [B,T,d_inner] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               _to_copy         [B,T,n_h_ssm,d_head_ssm] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               view             [B,T,d_state] -> [B,T,1,d_state]
  model.layers.N.mamba                               _to_copy         [B,T,1,d_state] -> [B,T,1,d_state]
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
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm] -> [B,d_state,n_h_ssm]
  model.layers.N.mamba                               view             [B,d_state,n_h_ssm] -> [B,1,d_state,n_h_ssm]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm,d_state] -> [B,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               view             [B,d_chunk,n_h_ssm,d_state] -> [B,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               constant_pad_nd  [B,T,n_h_ssm,d_state] -> [B,d_state,n_h_ssm,d_state]
  model.layers.N.mamba                               permute          [B,1,d_state,n_h_ssm] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               unsqueeze        [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_state,1]
  model.layers.N.mamba                               expand           [B,n_h_ssm,1,d_state,1] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               ones             [] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               tril             [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               bitwise_not      [d_chunk,d_chunk] -> [d_chunk,d_chunk]
  model.layers.N.mamba                               masked_fill      [B,n_h_ssm,1,d_chunk,d_chunk]*[d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               exp              [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,n_h_ssm,1,d_chunk,d_chunk]
  model.layers.N.mamba                               unsqueeze        [B,1,d_state,n_h_ssm,d_state] -> [B,1,d_state,1,n_h_ssm,d_state]
  model.layers.N.mamba                               unsqueeze        [B,1,d_chunk,n_h_ssm,d_state] -> [B,1,1,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               elementwise_mul  [B,1,d_state,1,n_h_ssm,d_state]*[B,1,1,d_chunk,n_h_ssm,d_state] -> [B,1,d_state,d_chunk,n_h_ssm,d_state]
  model.layers.N.mamba                               sum              [B,1,d_state,d_chunk,n_h_ssm,d_state] -> [B,1,d_state,d_chunk,n_h_ssm]
  model.layers.N.mamba                               permute          [B,n_h_ssm,1,d_chunk,d_chunk] -> [B,1,d_state,d_chunk,n_h_ssm]
  model.layers.N.mamba                               sum              [B,1,d_chunk,d_chunk,n_h_ssm,1] -> [B,1,d_state,d_chunk,n_h_ssm]
  model.layers.N.mamba                               sum              [B,1,d_state,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               slice            [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,1]
  model.layers.N.mamba                               sub              [B,n_h_ssm,1,1]*[B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               exp              [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1,d_chunk]
  model.layers.N.mamba                               permute          [B,n_h_ssm,1,d_chunk] -> [B,1,d_state,n_h_ssm]
  model.layers.N.mamba                               sum              [B,1,d_state,n_h_ssm,d_head_ssm,d_chunk] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               alias            [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               zeros_like       [B,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               concat           [B,1,n_h_ssm,d_head_ssm,d_state]*[B,1,n_h_ssm,d_head_ssm,d_state] -> [B,2,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               select           [B,n_h_ssm,1,d_chunk] -> [B,n_h_ssm,1]
  model.layers.N.mamba                               constant_pad_nd  [B,n_h_ssm,1] -> [B,n_h_ssm,n_kv]
  model.layers.N.mamba                               expand           [B,n_h_ssm,n_kv,1] -> [B,n_h_ssm,n_kv,n_kv]
  model.layers.N.mamba                               ones             [] -> [2,2]
  model.layers.N.mamba                               tril             [2,2] -> [2,2]
  model.layers.N.mamba                               bitwise_not      [2,2] -> [2,2]
  model.layers.N.mamba                               masked_fill      [B,n_h_ssm,n_kv,n_kv]*[2,2] -> [B,n_h_ssm,n_kv,n_kv]
  model.layers.N.mamba                               cumsum           [B,n_h_ssm,n_kv,n_kv] -> [B,n_h_ssm,n_kv,n_kv]
  model.layers.N.mamba                               exp              [B,n_h_ssm,n_kv,n_kv] -> [B,n_h_ssm,n_kv,n_kv]
  model.layers.N.mamba                               transpose        [B,n_h_ssm,n_kv,n_kv] -> [B,2,2,n_h_ssm]
  model.layers.N.mamba                               sum              [B,2,2,n_h_ssm,d_head_ssm,d_state] -> [B,2,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               slice            [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,1,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               select           [B,2,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               sum              [B,1,d_state,n_h_ssm,d_head_ssm,d_chunk] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,1,d_chunk,n_h_ssm,d_head_ssm]*[B,1,d_chunk,n_h_ssm,d_head_ssm] -> [B,1,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,d_chunk,n_h_ssm,d_head_ssm]*[B,d_chunk,n_h_ssm,d_head_ssm] -> [B,d_chunk,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               slice            [B,d_chunk,n_h_ssm,d_head_ssm] -> [B,T,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               zeros_like       [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba.norm                          _to_copy         [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          silu             [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,T,d_inner]*[B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.norm                          view             [B,T,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          pow              [B,T,1,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          mean             [B,T,1,d_inner] -> [B,T,1,1]
  model.layers.N.mamba.norm                          elementwise_add  [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba.norm                          rsqrt            [B,T,1,1] -> [B,T,1,1]
  model.layers.N.mamba.norm                          elementwise_mul  [B,T,1,d_inner]*[B,T,1,1] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          view             [d_inner] -> [B,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,d_inner]*[B,T,1,d_inner] -> [B,T,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,T,1,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba                               _to_copy         [B,T,d_inner] -> [B,T,d_inner]
  model.layers.N.mamba.out_proj                      t                [d_model,d_inner] -> w=[d_model,d_inner] [d_inner,d_model]
  model.layers.N.mamba.out_proj                      view             [B,T,d_inner] -> [T,d_inner]
  model.layers.N.mamba.out_proj                      matmul           [T,d_inner]*[d_inner,d_model] -> w=[d_model,d_inner] [T,d_model]
  model.layers.N.mamba.out_proj                      _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.0                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
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
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,T,d_head] -> [B,n_kv,T,d_head]
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
  model.layers.N.pre_ff_layernorm                    _to_copy         [B,T,d_model] -> [B,T,d_model]
  model.layers.N.pre_ff_layernorm                    pow              [B,T,d_model] -> [B,T,d_model]
  model.layers.N.pre_ff_layernorm                    mean             [B,T,d_model] -> [B,T,1]
  model.layers.N.pre_ff_layernorm                    elementwise_add  [B,T,1] -> [B,T,1]
  model.layers.N.pre_ff_layernorm                    rsqrt            [B,T,1] -> [B,T,1]
  model.layers.N.pre_ff_layernorm                    elementwise_mul  [B,T,d_model]*[B,T,1] -> [B,T,d_model]
  model.layers.N.pre_ff_layernorm                    elementwise_mul  [d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.N.feed_forward.up_proj                t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.up_proj                view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.up_proj                matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.up_proj                _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.gate_proj              t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.gate_proj              view             [B,T,d_model] -> [T,d_model]
  model.layers.N.feed_forward.gate_proj              matmul           [T,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [T,d_ff]
  model.layers.N.feed_forward.gate_proj              _unsafe_view     [T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.act_fn                 silu             [B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,T,d_ff]*[B,T,d_ff] -> [B,T,d_ff]
  model.layers.N.feed_forward.down_proj              t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.down_proj              view             [B,T,d_ff] -> [T,d_ff]
  model.layers.N.feed_forward.down_proj              matmul           [T,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [T,d_model]
  model.layers.N.feed_forward.down_proj              _unsafe_view     [T,d_model] -> [B,T,d_model]
  model.layers.N.feed_forward                        elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.1                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.2                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.3                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.4                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.5                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.6                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.7                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.8                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.9                                     elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.10                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.11                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.12                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.13                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.14                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.15                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.16                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.17                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.18                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.19                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.20                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.21                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.22                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.23                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.24                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.24                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.25                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.26                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.27                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.28                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.29                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.30                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.31                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.32                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.33                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.34                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.35                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.36                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.37                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.38                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.38                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.39                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.40                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.40                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.41                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.42                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.42                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    elementwise_mul  [B,T,d_model] -> [B,T,d_model]
  model.layers.43                                    elementwise_add  [B,T,d_model]*[B,T,d_model] -> [B,T,d_model]
  model.final_layernorm                              _to_copy         [B,T,d_model] -> [B,T,d_model]
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
                                                     elementwise_mul  [B,T,V] -> [B,T,V]
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
  model.embed_tokens                                 embedding        [V,d_model]*[B,1] -> w=[V,d_model] [B,1,d_model]
  model                                              elementwise_mul  [B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.mamba                               elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.N.mamba.in_proj                       t                [2*d_inner+2*n_g*d_state+n_h_ssm,d_model] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [d_model,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       view             [B,1,d_model] -> [B,d_model]
  model.layers.N.mamba.in_proj                       matmul           [B,d_model]*[d_model,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[2*d_inner+2*n_g*d_state+n_h_ssm,d_model] [B,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba.in_proj                       _unsafe_view     [B,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]*[1,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> w=[1,1,2*d_inner+2*n_g*d_state+n_h_ssm] [B,1,2*d_inner+2*n_g*d_state+n_h_ssm]
  model.layers.N.mamba                               split_with_sizes [B,1,2*d_inner+2*n_g*d_state+n_h_ssm] -> [B,1,d_inner]*[B,1,d_inner+2*n_g*d_state]*[B,1,n_h_ssm]
  model.layers.N.mamba                               transpose        [B,1,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state,1]
  model.layers.N.mamba                               concat           [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,1] -> [B,d_inner+2*n_g*d_state,d_conv+1]
  model.layers.N.mamba                               slice            [B,d_inner+2*n_g*d_state,d_conv+1] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               copy_            [B,d_inner+2*n_g*d_state,d_conv]*[B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               squeeze          [d_inner+2*n_g*d_state,1,d_conv] -> w=[d_inner+2*n_g*d_state,1,d_conv] [d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               elementwise_mul  [B,d_inner+2*n_g*d_state,d_conv]*[d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state,d_conv]
  model.layers.N.mamba                               sum              [B,d_inner+2*n_g*d_state,d_conv] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               elementwise_add  [B,d_inner+2*n_g*d_state]*[d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba.act                           silu             [B,d_inner+2*n_g*d_state] -> [B,d_inner+2*n_g*d_state]
  model.layers.N.mamba                               split_with_sizes [B,d_inner+2*n_g*d_state] -> [B,d_inner]*[B,d_state]*[B,d_state]
  model.layers.N.mamba                               _to_copy         [n_h_ssm] -> [n_h_ssm]
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
  model.layers.N.mamba                               view             [B,d_state] -> [B,1,d_state]
  model.layers.N.mamba                               unsqueeze        [B,1,d_state] -> [B,1,1,d_state]
  model.layers.N.mamba                               expand           [B,1,1,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba                               clone            [B,1,n_h_ssm,d_state] -> [B,1,n_h_ssm,d_state]
  model.layers.N.mamba                               view             [B,1,n_h_ssm,d_state] -> [B,n_h_ssm,d_state]
  model.layers.N.mamba                               unsqueeze        [B,n_h_ssm,d_state] -> [B,n_h_ssm,1,d_state]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,1]*[B,n_h_ssm,1,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,d_inner] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_mul  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,1] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               copy_            [B,n_h_ssm,d_head_ssm,d_state]*[B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               _to_copy         [B,n_h_ssm,d_head_ssm,d_state] -> [B,n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,n_h_ssm,d_head_ssm,d_state] -> [n_h_ssm,d_head_ssm,d_state]
  model.layers.N.mamba                               view             [B,n_h_ssm,d_state] -> [n_h_ssm,d_state,B]
  model.layers.N.mamba                               batched_matmul   [n_h_ssm,d_head_ssm,d_state]*[n_h_ssm,d_state,B] -> [n_h_ssm,d_head_ssm,B]
  model.layers.N.mamba                               view             [n_h_ssm,d_head_ssm,B] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba                               elementwise_add  [B,n_h_ssm,d_head_ssm]*[B,n_h_ssm,d_head_ssm] -> [B,n_h_ssm,d_head_ssm]
  model.layers.N.mamba.norm                          _to_copy         [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          silu             [B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,1,d_inner]*[B,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          pow              [B,1,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          mean             [B,1,1,d_inner] -> [B,1,1,1]
  model.layers.N.mamba.norm                          elementwise_add  [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba.norm                          rsqrt            [B,1,1,1] -> [B,1,1,1]
  model.layers.N.mamba.norm                          elementwise_mul  [B,1,1,d_inner]*[B,1,1,1] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          view             [d_inner] -> [B,d_inner]
  model.layers.N.mamba.norm                          elementwise_mul  [B,d_inner]*[B,1,1,d_inner] -> [B,1,1,d_inner]
  model.layers.N.mamba.norm                          view             [B,1,1,d_inner] -> [B,1,d_inner]
  model.layers.N.mamba.norm                          squeeze          [B,1,d_inner] -> [B,d_inner]
  model.layers.N.mamba.norm                          _to_copy         [B,d_inner] -> [B,d_inner]
  model.layers.N.mamba.out_proj                      t                [d_inner,d_inner] -> w=[d_inner,d_inner] [d_inner,d_inner]
  model.layers.N.mamba.out_proj                      matmul           [B,d_inner]*[d_inner,d_inner] -> w=[d_inner,d_inner] [B,d_inner]
  model.layers.0                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.0                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
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
  model.layers.N.self_attn                           elementwise_mul  [B,n_kv,1,d_head] -> [B,n_kv,1,d_head]
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
  model.layers.0                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.0                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_ff_layernorm                    _to_copy         [B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_ff_layernorm                    pow              [B,1,d_model] -> [B,1,d_model]
  model.layers.N.pre_ff_layernorm                    mean             [B,1,d_model] -> [B,1,1]
  model.layers.N.pre_ff_layernorm                    elementwise_add  [B,1,1] -> [B,1,1]
  model.layers.N.pre_ff_layernorm                    rsqrt            [B,1,1] -> [B,1,1]
  model.layers.N.pre_ff_layernorm                    elementwise_mul  [B,1,d_model]*[B,1,1] -> [B,1,d_model]
  model.layers.N.pre_ff_layernorm                    elementwise_mul  [d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.N.feed_forward.up_proj                t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.up_proj                view             [B,1,d_model] -> [B,d_model]
  model.layers.N.feed_forward.up_proj                matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.up_proj                _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward.gate_proj              t                [d_ff,d_model] -> w=[d_ff,d_model] [d_model,d_ff]
  model.layers.N.feed_forward.gate_proj              view             [B,1,d_model] -> [B,d_model]
  model.layers.N.feed_forward.gate_proj              matmul           [B,d_model]*[d_model,d_ff] -> w=[d_ff,d_model] [B,d_ff]
  model.layers.N.feed_forward.gate_proj              _unsafe_view     [B,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward.act_fn                 silu             [B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward                        elementwise_mul  [B,1,d_ff]*[B,1,d_ff] -> [B,1,d_ff]
  model.layers.N.feed_forward.down_proj              t                [d_model,d_ff] -> w=[d_model,d_ff] [d_ff,d_model]
  model.layers.N.feed_forward.down_proj              view             [B,1,d_ff] -> [B,d_ff]
  model.layers.N.feed_forward.down_proj              matmul           [B,d_ff]*[d_ff,d_model] -> w=[d_model,d_ff] [B,d_model]
  model.layers.N.feed_forward.down_proj              _unsafe_view     [B,d_model] -> [B,1,d_model]
  model.layers.N.feed_forward                        elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.1                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.1                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.2                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.2                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.3                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.3                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.4                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.4                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.5                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.5                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.6                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.6                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.7                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.7                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.8                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.8                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.9                                     elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.9                                     elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.10                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.10                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.11                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.11                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.11                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.11                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.12                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.12                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.13                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.13                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.14                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.14                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.15                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.15                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.16                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.16                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.17                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.17                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.17                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.17                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.18                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.18                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.19                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.19                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.20                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.20                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.21                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.21                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.22                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.22                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.23                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.23                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.23                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.23                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.24                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.24                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.24                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.24                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.25                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.25                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.26                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.26                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.27                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.27                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.28                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.28                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.29                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.29                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.30                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.30                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.31                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.31                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.32                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.32                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.33                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.33                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.34                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.34                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.35                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.35                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.35                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.35                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.36                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.36                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.37                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.37                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.38                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.38                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.38                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.38                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.39                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.39                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.39                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.39                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.40                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.40                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.40                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.40                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.41                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.41                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.41                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.41                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.42                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.42                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.42                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.42                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.layers.43                                    elementwise_mul  [B,d_inner] -> [B,d_inner]
  model.layers.43                                    elementwise_mul  [B,1,d_model] -> [B,1,d_model]
  model.layers.43                                    elementwise_add  [B,d_inner]*[B,1,d_model] -> [B,1,d_model]
  model.layers.43                                    elementwise_add  [B,1,d_model]*[B,1,d_model] -> [B,1,d_model]
  model.final_layernorm                              _to_copy         [B,1,d_model] -> [B,1,d_model]
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

