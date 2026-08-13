# 라벨 검토 결과 — openai/gpt-oss-120b

- 검토일: 2026-08-12
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 **모든** 질문에 답한다(기계가 개수를 맞춘다). 확인 프레임이 아니라 반박 프레임으로 — 각 라벨에 대해 '틀렸다는 증거'를 먼저 찾고, 못 찾은 것만 맞다고 적었다. 외부 검토가 준 팁 3가지(op 내부 필드 상호 대조 / 요청·응답 개수 diff / 반박 프레임)를 그대로 적용했다.
- 요약: 의뢰서의 `2*d_moe` 는 이름이 옳았다 — 산술 휴리스틱이 내던 것을 규칙으로 승격했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | gate+up 융합 투영 폭 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))` — deepseek_v3.py:181 / qwen3_moe.py:218 / glm4_moe.py:350, gpt_oss.py:75 는 축 순서만 다르다 `(num_experts, hidden_size, 2 * intermediate_size)`. gate 와 up 을 파라미터 하나에 이어 붙인 폭이므로 2·d_moe 가 맞다.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn / (root)` |
| 축 | d_head vs n_h (64), d_model vs d_moe (2880) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

head **개수**와 head **폭**이 같은 값이라 값으로는 못 가른다. 결정은 값이 아니라 `src/anchors.py` 가 한다 — `nn.Linear.weight == [out, in]` 으로 모듈이 선언한 폭을 읽고, 그 이름을 그 모듈의 모든 op 에 고정한다.

**반박 시도**: 실제로 틀리면 어떤 모습인가? head-개수 이름이 head-폭 축을 가져가면 한 shape 안에 `n_h` 와 `n_kv` 가 함께 나온다(2026-07-30 에 8개 모델 16,859축이 그랬다). 그걸 잡는 `head_excl` 불변식이 현재 함대 전체 · 양쪽 phase 에서 **0** 이다. 또한 `[..., 개수, 폭]` 순서 규약을 어기면 `matmul_compose` 가 걸리는데 그것도 **0** 이다. 틀렸다는 증거를 찾지 못했다.

**근거 소스**: 이 판정은 `develop/sources/modeling_gpt_oss.py`, `develop/sources/configuration_gpt_oss.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | 라우팅 게더의 feature 축 2880 (d_model vs d_moe) |
| 현재 라벨 | `d_moe` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_model` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_gpt_oss.py:75-78` — `gate_up_proj = nn.Parameter(num_experts, hidden_size, 2 * intermediate_size)`, `down_proj = nn.Parameter(num_experts, intermediate_size, hidden_size)`. 즉 전문가에 **들어가는** 폭은 `hidden_size`(=`d_model`)이고 `intermediate_size`(=`d_moe`)는 그 안에서만 쓰인다. gpt-oss 는 둘 다 2880 이라 값으로는 구별되지 않는다.

결함은 라우팅 게더에 있었다: `index([T, d_model], [k*T]) -> [k*T, d_moe]`(실측 `[1056, 2880]`). **게더는 행을 고르는 연산이라 뒤 축의 이름을 바꿀 수 없다** — 전치·view 와 같은 부류다. 여기서 잔차 스트림이 전문가 폭 이름을 얻어 `masked_fill_` / `clamp` / `elementwise_mul` 로 이어지는 체인 전체가 그 이름을 물려받았다.

**교정 완료**: `src/build_table._gather_keeps_features` 를 넣어 dim-0 게더의 뒤 축 이름을 입력에서 그대로 복사한다(랭크가 같고 뒤 축의 실측 폭이 동일할 때만). 값이 아니라 연산의 정의에서 나오는 규칙이라 다른 모델에도 그대로 적용된다. 결과: `index([T, d_model], [k*T]) -> [k*T, d_model]`, 전문가 가중치 게더는 `[E, 2*d_moe], [k*T] -> [k*T, 2*d_moe]` 로 유지된다.
