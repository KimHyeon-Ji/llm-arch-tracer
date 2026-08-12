# 라벨 검토 결과 — Qwen/Qwen3.6-27B

- 검토일: 2026-08-12
- 검토자: llm(claude, 자기모순 추적 + 소스 대조)
- 본 것: 게이트가 센 자기모순(같은 텐서를 한 행/한 엣지 안에서 두 이름으로 부르는 곳)을 출발점으로 linear_attn 전건 추적. 1·2회차 판정 유지. C절(모듈별 출력 shape) 미수행.
- 요약: 의뢰서 4건 — 전부 linear_attn 의 청크 루프 인덱스였다. 새 규칙은 게이트 어텐션 Q 폭 하나뿐이었고 미등록 config 필드는 0이다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | 작은 정수 축들 (2*n_kv, 3*n_kv, n_h+1, n_h_lin_v+1, 3*n_h, 3*d_conv_lin, n_kv*T 로 렌더된 것들) |
| 현재 라벨 | `산술로 지은 이름` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

값이 6·8·12·25·33·48·49 처럼 작은 정수인데, 같은 `linear_attn` 모듈의 다른 축 위치에는 1,2,3,…,64 사다리가 65종 그대로 관측된다(실측). chunked gated-delta 스캔이 청크 내부를 파이썬 루프로 돌며 잘라낸 조각들이고, 그중 config 값과 우연히 같은 정수만 이름을 받았다 — Qwen3-Next 에서 확인한 것과 같은 부류다(`modeling_qwen3_5.py` 의 chunk 루프). **루프 인덱스는 이름이 없는 것이 정답이다.** `build_table._unname_loop_indices` 가 이미 사다리를 정수로 되돌리지만, 키가 (module, op_type, field, shape_index, axis, rank) 라 피연산자별로 쪼개져 이 위치들은 8종 문턱을 못 넘었다. shape_index 를 키에서 빼면 같은 op 의 피연산자들이 합쳐져 잡힌다 — 다음 회차 과제.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_proj` |
| 축 | Q 투영 폭 (n_h·2·d_head) |
| 현재 라벨 | `이름 없는 정수` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*n_h*d_head` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_qwen3_5.py:641-643` `self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)` — Q 투영이 query 와 gate 를 한 번에 낸다(게이트 어텐션). 뒤따르는 view 는 이미 `[B, T, n_h, 2*d_head]` 로 맞게 렌더되고 있었고 묶인 폭만 이름이 없었다. `rules/derived_dims.yaml` 에 등록 완료.

## 발견 3 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | in_proj_qkvz 조각 폭 (27B 에서 2048) |
| 현재 라벨 | `2*n_kv*d_head` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `key_dim (= n_h_lin_k · d_head_lin_k)` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다(실측 [2048, 2048, 6144]). 어텐션 head 수와 무관한 축인데 2·n_kv·d_head 와 값이 같아 그쪽으로 붙었다 — 확인된 오라벨. `n_h_lin_k * d_head_lin_k` 로 등록해봤으나 Qwen3-Next 의 flow_ambig 가 0 -> 72 로 퇴행해 보류했다(2026-08-10): 새 이름이 붙은 축의 하류 소비자가 옛 이름을 그대로 들고 있어 한 텐서가 두 이름을 갖는다. 라벨이 아니라 전파 쪽 과제다.

## 발견 4 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model` |
| 축 | [4, B, T] 의 첫 축 |
| 현재 라벨 | `4 (이름 없음)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 4` |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

루트 모듈(`model`)에서 `expand`/`select`/`slice` 가 `[4, B, T]` 를 다룬다(6축). 레이어가 아니라 모델 루트의 작은 열거이고, config 차원으로 볼 근거가 없다. `d_conv_lin`(=4)·`n_kv`(=4)와 값이 같은 것은 우연이다. 정수로 둔다 — 다만 무엇을 세는 열거인지까지는 소스에서 확정하지 않았다(축이 6개뿐이라 우선순위를 낮췄다).

## 발견 5 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | 청크 루프 인덱스 축 (elementwise·copy 양쪽) |
| 현재 라벨 | `n_kv / d_conv_lin / 3*n_kv / n_h/n_kv / k / T (입력) vs 정수 (출력)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

루프 계단에서 지어낸 이름을 떼는 `build_table._unname_loop_indices` 는 `_propagate_labels` **앞**에 있어야 하는데(뒤로 옮기면 데이터플로우 불일치가 43,000행으로 폭증한다), 그 전파가 monotone 이라 비운 정수를 이웃에서 다시 채웠다. 그래서 한 `elementwise_add` 가 들어갈 때는 `n_kv`, 나올 때는 `2` 였다 — 1,008행. `clone` 에서 504행이 더 있었다.

**교정(2026-08-12)**: `build_table._unname_refilled_operands` 를 전파 **뒤**에 고정점으로 돌린다. 두 방향만 허용한다 — (1) shape 을 보존하는 elementwise·copy op 에서 출력 축이 이미 정수인데 같은 concrete shape 의 피연산자가 이름을 달고 있으면 떼고, (2) 같은 텐서를 만든 상류 op 도 같이 뗀다(`depends_on` + concrete shape 일치). **이름을 지어내는 방향으로는 절대 가지 않는다.** 값으로 쓸어내는 것이 위험한 이유 — linear_attn 안에서 4 는 루프 계단이면서 진짜 `d_conv_lin` 이기도 하다 — 는 그대로지만, 텐서 신원을 따라가면 그 둘이 구별된다. 자기모순 1,512행 → 0.

## 발견 6 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | head 수 축 (16) |
| 현재 라벨 | `n_h` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_lin_k` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`attn` 이 **`linear_attn` 안에서도 매치**한다. 그래서 softmax attention 심볼 전체가 Gated DeltaNet 모듈을 덮었고, 정작 그 모듈의 이름은 전역 우선순위에서 졌다(n_h=10 vs n_h_lin_k=29). 값도 정확히 겹친다 — num_attention_heads == linear_num_key_heads == 16. `n_h`/`n_kv`/`d_head` 스코프를 `(?<!linear_)attn` 으로 바꿔 선형 attention 을 뺐다. 선형 attention 은 자기 head 수·head 폭을 따로 선언한다(`modeling_qwen3_next.py:516-521`). 축 1,584건이 `n_h_lin_k` 로 교정됐다.

## 발견 7 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | matmul 수축 축 (128) |
| 현재 라벨 | `d_head_lin_k / d_head_lin_v 혼용` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없다.
