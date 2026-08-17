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
| `mamba$` | `d_state` | `d_chunk` | 484 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 B/C는 `[B,T,num_heads,state_size]`이고 :299-316은 sequence 축만 pad한다. 축 1은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 396 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-520에서 hidden_states는 `[B,T,num_heads,head_dim]`이고 :299-316은 sequence 축만 pad한다. nth 2 출력의 축 1은 d_chunk다. |
| `mamba$` | `d_chunk` | `d_state` | 396 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 C는 `[B,T,num_heads,state_size]`이고 sequence 축만 pad된다. nth 5 출력의 마지막 축은 d_state다. |
| `mamba$` | `d_state` | `d_chunk` | 440 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:529-533의 G/M 경로는 `(b,c,l,s,h,n)`에서 state 축을 합친 뒤 singleton을 붙인다. `[B,c,l,s,h,1]`의 축 2는 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 704 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-520은 hidden_states를 `[B,T,num_heads,head_dim]`으로 읽고 :299-316이 sequence 축을 chunk_size 배수로 pad한다. 축 1의 256은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 704 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:497-523에서 A는 chunk reshape 뒤 `permute(0,3,1,2)`로 `[B,num_heads,n_chunks,chunk_size]`가 된다. 마지막 축은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_chunk` | `d_state` | 484 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:507-520에서 B/C는 num_heads로 repeat된 `[B,T,num_heads,state_size]`이고 :299-316은 sequence 축만 pad한다. 따라서 `[B,d_chunk,n_h_ssm,state_size]`의 마지막 축은 d_chunk이 아니라 d_state다. |
| `mamba$` | `d_state` | `d_chunk` | 440 | transformers 5.14.1 installed source modeling_falcon_h1.py:312-858; revalidated this axis verdict unchanged. modeling_falcon_h1.py:319-335의 `segment_sum` expand는 마지막 chunk_size 축을 하나 더 만들어 `[...,chunk_size,chunk_size]`를 구성한다. :527의 첫 intra-chunk 호출에서 축 3은 d_state가 아니라 d_chunk다. |
| `mamba$` | `d_state` | `d_chunk` | 88 | transformers 5.14.1 modeling_falcon_h1.py:818 `G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :] # shape: (b, c, l, s, h, n)` — 축 2 는 l 로 chunk_size 다. 이 모델은 mamba_chunk_size == mamba_d_state == 256 이라 값으로는 못 가린다. (Nemotron 과 달리 Falcon-H1 은 permute 가 끼지 않아 정준 순서 그대로다.) |

### 이 표를 읽을 때 유의할 것

소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.

| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |
|---|---|---|---|---|
| `model.layers.*.mamba` | [B, n_h_ssm, n_kv, n_kv] (실제 [1, 24, 2, 2]) | `n_kv` | `(청크 개수 축 -- 이름 없는 정수로 남겨야 한다)` | `n_kv` 가 아니다. 이 축은 **inter-chunk 재귀의 청크 개수**다: 트레이스 op143 이 `[B, n_h_ssm, 1] -> [B, n_h_ssm, 2]` 로 pad 하는데(초기 상태 1칸을 앞에 붙임), 그 결과 폭 2 = 청크 1개 + 초기 상태 1개다. 그 뒤 `segment_sum` 이 다시 불려 `[2, 2]` 감쇠 행렬을 만든다 … |

전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 `develop/sources/` 에 있다.
