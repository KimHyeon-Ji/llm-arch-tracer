# 라벨 검토 결과 — Qwen/Qwen3.6-35B-A3B

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
- 요약: 의뢰서 5건 — 전부 linear_attn 의 청크 루프 인덱스였다. 새 규칙은 게이트 어텐션 Q 폭 하나뿐이었고 미등록 config 필드는 0이다.

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
