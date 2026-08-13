# 라벨 검토 결과 — zai-org/GLM-5.2

- 검토일: 2026-08-12
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 **모든** 질문에 답한다(기계가 개수를 맞춘다). 확인 프레임이 아니라 반박 프레임으로 — 각 라벨에 대해 '틀렸다는 증거'를 먼저 찾고, 못 찾은 것만 맞다고 적었다. 외부 검토가 준 팁 3가지(op 내부 필드 상호 대조 / 요청·응답 개수 diff / 반박 프레임)를 그대로 적용했다.
- 요약: 의뢰서가 비어 있었다. **다른 벤더의 새 아키텍처가 기존 규칙만으로 전부 설명된 첫 사례**다 — 새 규칙 0개, 휴리스틱 0.00%, 미등록 config 필드 0.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(전체)` |
| 축 | DSA indexer + MLA 축 |
| 현재 라벨 | `n_h_I / c_I / k_I / c_q / c_kv / d_nope / d_v / d_rope` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`model_type: glm_moe_dsa` 는 Zhipu 가 DeepSeek Sparse Attention 을 채택한 것으로, config 의 `index_head_dim`(128) / `index_n_heads`(32) / `index_topk`(2048) 가 DeepSeek-V4 의 indexer 와 같은 자리를 차지한다. V4 용으로 등록해 둔 심볼이 그대로 맞았고 (실측 n_h_I=32, c_I=128, k_I=2048), MLA 쪽도 c_q=2048 / c_kv=512 / d_nope=192 / d_v=256 / d_rope=64 로 전부 해결됐다. 게이트 FAIL 0, 축 326,319개 중 지어낸 이름 0개.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.indexer.wq_b` |
| 축 | 입력 폭 (q_lora_rank) |
| 현재 라벨 | `k_I` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `c_q` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

GLM-5.2 는 index_topk == q_lora_rank == d_moe == 2048 인 삼중 충돌이다. 위 V4 수정으로 '안쪽 스코프가 이긴다'를 넣자 `k_I`(scope `indexer`)가 `c_q`(scope `attn`)를 눌러 압축 Q latent 폭을 가져갔다 — 축 156건 퇴행으로 게이트가 잡았다. **선택 개수는 폭이 아니다**: 어떤 단계가 몇 개를 남기는지는 그 앞 파라미터가 몇 폭인지와 무관하다. `_SELECTION_SYMS`(k, k_I)는 값 동률에서 규칙이 더 위로 매긴 심볼을 이길 수 없게 했다(`symbolic_shape._pick`). DeepSeek-V3 의 라우터에서는 `k`(우선순위 22)가 `n_grp`(38)를 여전히 이긴다.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | o_proj 직전 축 (16384) |
| 현재 라벨 | `n_h*(d_nope+d_rope)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h*d_v` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`view [.., n_h*(d_nope+d_rope)] `가 o_proj 직전 축에 붙어 있었다. o_proj 의 선언 폭은 `num_heads * v_head_dim` 이므로 `n_h*d_v` 가 맞다. GLM-5.2 는 d_v = 256 = d_nope+d_rope = 192+64 라 값이 같고, `rules/derived_dims.yaml` 의 파일 순서에서 q_b_proj 쪽 식이 앞에 있어 이겼다. `n_h*d_v` 를 앞으로 옮겼다 — DeepSeek-V2/V3 계열은 d_v(128) != d_nope+d_rope(192) 라 영향이 없다. 자기모순 78 → 0.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 4 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn*` |
| 축 | d_head vs d_rope vs n_h vs n_kv (64) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

head **개수**와 head **폭**이 같은 값이라 값으로는 못 가른다. 결정은 값이 아니라 `src/anchors.py` 가 한다 — `nn.Linear.weight == [out, in]` 으로 모듈이 선언한 폭을 읽고, 그 이름을 그 모듈의 모든 op 에 고정한다.

**반박 시도**: 실제로 틀리면 어떤 모습인가? head-개수 이름이 head-폭 축을 가져가면 한 shape 안에 `n_h` 와 `n_kv` 가 함께 나온다(2026-07-30 에 8개 모델 16,859축이 그랬다). 그걸 잡는 `head_excl` 불변식이 현재 함대 전체 · 양쪽 phase 에서 **0** 이다. 또한 `[..., 개수, 폭]` 순서 규약을 어기면 `matmul_compose` 가 걸리는데 그것도 **0** 이다. 틀렸다는 증거를 찾지 못했다.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 5 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.indexer.wq_b` |
| 축 | c_q vs k_I (2048) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

index_topk == q_lora_rank == 2048 이다. `wq_b` 의 입력은 **투영의 입력 폭**이므로 압축 Q latent(`c_q`)이고, indexer 가 몇 개를 남기는지(k_I)는 그 앞 파라미터의 폭과 무관하다. `_SELECTION_SYMS` 가드가 이 자리를 이미 결정한다(선택 개수는 값 동률에서 규칙이 더 위로 매긴 심볼을 못 이긴다).

**반박 시도**: 틀렸다면 `wq_b` 가중치의 in-features 가 선택 개수를 달아야 하는데, 그건 `weight_operand`/`matmul_compose` 에 걸린다. 둘 다 0 이다.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 6 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.rotary_emb` |
| 축 | 회전 차원 (64) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`model.rotary_emb` 의 64 — `d_head`(64)와 `d_rope`(64)가 같은 값이다. rotary 모듈이 만드는 것은 **회전 차원**이므로 `d_rope` 가 뜻으로는 더 정확하지만, GLM-5.2 는 partial rotary 가 아니라 head 전체를 회전시키므로 두 이름이 같은 축을 가리킨다.

**반박 시도**: 두 값이 다른 모델(부분 RoPE)에서 rotary_emb 가 head 폭을 달면 틀린 것인데, 그런 모델(DeepSeek-V4: d_head=512, d_rope=64)에서는 `d_rope` 가 붙는다. 이 모델에서만 겹치는 것이고 어느 쪽도 거짓이 아니다.

**근거 소스**: 이 판정은 `develop/sources/modeling_glm_moe_dsa.py`, `develop/sources/configuration_glm_moe_dsa.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)
