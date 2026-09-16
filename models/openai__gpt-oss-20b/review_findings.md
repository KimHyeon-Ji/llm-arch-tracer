# 라벨 검토 결과 — openai/gpt-oss-20b

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

## 발견 4 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | down_proj 가중치 [E, X, d_model] 의 가운데 축 X (2880) |
| 현재 라벨 | `d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_moe` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

위 finding #2(2026-08-12)가 'd_head vs n_h, d_model vs d_moe' 를 뭉뚱그려 '현재 라벨이 맞다'고 판정했는데, 그건 self_attn 의 module-declared-anchor 메커니즘(nn.Linear.weight == [out,in])이 잘 맞은 경우만 확인한 것이었다. 이 자리는 다르다 -- down_proj(modeling_gpt_oss.py:82)는 nn.Linear 가 아니라 융합된 nn.Parameter(torch.empty((num_experts, intermediate_size, hidden_size))) 라 module-declared-anchor 가 못 미치고 값-동률 우선순위로 넘어가 있었다. 값(2880=2880)으로는 못 가리지만 선언 순서는 명확하다: 가운데 축은 intermediate_size(d_moe), 마지막이 hidden_size(d_model). 대조: 같은 클래스의 gate_up_proj(80줄, [num_experts, hidden_size, 2*intermediate_size])는 가운데 축이 진짜 d_model이라 2*d_moe != d_model 로 값이 안 겹쳐서 처음부터 맞게 렌더되고 있었다 -- down_proj 만 두 config 필드가 우연히 같아 새는 자리였다.

교정: rules/label_overrides.yaml 에 spread: class 로 등록(2026-08-31), gpt-oss-20b/120b 둘 다.

## 발견 5 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | d_head vs n_h (64) 전체 앵커 114개 (위 finding #2 는 존재만 확인, 앵커별 검증은 안 했었다) |
| 현재 라벨 | `위치별로 이미 정확 (개수 자리는 n_h, 폭 자리는 d_head)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

review_request.md 0절의 self_attn 앵커 전부(4D/3D/5D/1D 형태 합쳐 114개, prefill+decode)를 develop/canonical_axis_rules.yaml 스타일 위치 규칙 6개로 측정(develop/rule_coverage.py) -- head 개수는 항상 axis1(4D 후치환)/axis2(4D 전치환)/axis0(3D matmul, bare [n_h]), head 폭은 항상 axis3(4D 어느 형태든)/axis2(3D matmul)/axis4(5D GQA-expand). agree=114/differ=0/clash=0. modeling_gpt_oss.py:320-324 hidden_shape=(*input_shape,-1,head_dim); .view(hidden_shape).transpose(1,2) 가 이 위치 규약의 근거 -- transpose(1,2)는 axis 1/2만 바꾸고 axis3(마지막)은 절대 안 건드리므로 head_dim==num_attention_heads 값 동률과 무관하게 위치로 항상 구별된다. rules/label_confirmed.yaml에 114개 앵커 전부 등록(2026-08-31).

## 발견 6 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | review_request.md 2절: [..., d_head, d_head] 또는 [..., n_h, n_h] 같은 진짜 정사각이 있는가 |
| 현재 라벨 | `(정사각 없음)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

modeling_gpt_oss.py 전체를 확인 -- head_dim 과 num_attention_heads 어느 쪽도 같은 view/reshape 호출 안에서 두 번 쓰이지 않는다(q/k/v_proj 는 항상 (*input_shape,-1,head_dim), sinks 는 torch.empty(num_attention_heads) 뿐). 위 finding에서 114개 앵커를 전수 확인했을 때도 axis1/2(개수)와 axis3/4(폭)가 같은 텐서 안에서 뒤섞인 자리는 하나도 없었다(그랬다면 head_excl 불변식이 잡았을 것 -- 현재 0). 소스에 진짜 정사각 reshape이 없다는 것이 정답이고, 정적 소스 대조가 '확인 불가'로 남긴 것은 그 대조기가 view() 문자열 안에서 같은 필드를 두 번 찾는 방식이라 애초에 없는 걸 못 찾는 게 정상 동작이다 -- 결함이 아니다.

## 발견 7 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | expert GLU clamp |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16 지적에서 출발해 원인을 찾았다. gate 상한 7 / up [-7,7] clamp 가 요약 표에 없다. `clamp` 는 major-op 선별의 KEEP 목록에 없다 -- 비용을 지배하는 연산이 아니라는 판단이고 그 범위는 structure.yaml 이 밝힌다. 원시 trace 에는 있다(20b prefill op 198,199). modeling_gpt_oss.py:113-116.

## 발견 8 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | d_model vs d_moe (둘 다 2880) |
| 현재 라벨 | `현행 라벨` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. gate_up weight [E,d_model,2*d_moe], down weight [E,d_moe,d_model]; GLU 중간은 d_moe, down 이후와 down bias 는 d_model. 값이 같아도 소스 역할로 확정된다. 요약 op 13/19/35/41 의 weight 라벨은 맞다. modeling_gpt_oss.py:77-92,113-116.

## 발견 9 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | prefill softmax 의 T+1 / sliding window |
| 현재 라벨 | `T+1` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. T+1 은 KV 264 + sink 1. sink 확률은 PV GEMM 전에 버려지므로 PV 수축축은 여전히 T. sliding window 는 prefill 에서 마스크로 적용되어 dense T×T 가 정상이며 별도 op 가 필요하지 않다. 다만 마스크 종류 메타데이터가 없으면 shape 만으로 full/local 을 구분할 수 없다. modeling_gpt_oss.py:261-278,264-265,308,485-496.

## 발견 10 — corrected (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | expert bias add 가 요약 표에서 빠졌던 것 |
| 현재 라벨 | `` |
| 판정 | `corrected` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex) 2026-09-16 지적에서 출발해 원인을 찾았다. `aten.add_.Tensor` 의 op_type 이 `add_` 로 찍히는데 그 문자열이 `elementwise_add` 가 아니라는 이유만으로 크기 게이트를 못 타고 무조건 빠졌다. 넓은 축(`[B*k*T, d_model]`)을 만지는 덧셈이라 선별 기준상 **남아야 했다**. major_ops._keep 이 in-place 변종을 기본형으로 정규화하도록 고쳤다(판정에만 쓰고 표의 op_type 은 `add_` 그대로 둔다). 함대 실측 294건: gpt-oss 120b 72 / 20b 48, Kimi-K3 92, falcon-7b 32, Kimi-Linear 26, Llama-4 24. modeling_gpt_oss.py:80-92 / integrations/moe.py:435-468.
