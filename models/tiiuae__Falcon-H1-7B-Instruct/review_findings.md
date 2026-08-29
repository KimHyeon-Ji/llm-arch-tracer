# 라벨 검토 결과 — tiiuae/Falcon-H1-7B-Instruct

- 검토일: 2026-08-29
- 검토자: llm(claude, 반박 프레임 전건 판정 -- 2026-08-15 블라인드 온보딩 판정을 실측 재확인 + 나머지 d_chunk/d_state 충돌 전수 완결)
- 본 것: review_ledger가 이 모델을 (Kimi-K3와 함께) '미수행'으로 보고해서 처음부터 다시 봤다. 2026-08-15 판정 3건이 지금 산출물에 그대로 반영돼 있는지 트레이스로 재확인한 뒤, review_request.md 0절에 여전히 남아 있던 `d_chunk vs d_state`(18행, config.mamba_chunk_size==config.mamba_d_state==256인 값 충돌) 전체를 소스(torch_forward의 SSD naive 구현, modeling_falcon_h1.py:674-878)와 실측 트레이스 op_id 대조로 끝까지 판정했다.
- 요약: 2026-08-15의 3건: 2건(정사각 축 교정, 관례 확인)은 지금도 정확히 그 상태로 렌더되고 있음을 재확인. 1건(inter-chunk 재귀 카운트 축의 n_kv 오라벨)도 이미 별도로 고쳐져 지금은 정직하게 bare 정수로 렌더됨을 확인(레이어0 기준 값 2, references.yaml의 기존 주석과 일치). 그 위에 review_request.md 0절에 남아 있던 d_chunk/d_state tie 18행을 전수 판정 -- 9곳은 값-매칭이 실제로 틀렸던 자리(주로 seq_len 청크 패딩을 state 축으로 오인, 그 오염이 sum/permute/elementwise_mul을 타고 하위 3~4곳으로 더 퍼짐)라 rules/label_overrides.yaml에 교정, 나머지는 이미 맞는 렌더라 rules/label_confirmed.yaml에 확인으로 기록. review_request.md의 0절(실제 조치 필요 표)이 0행으로 수렴, develop/verify_all.py FAIL 0 / 퇴행 0. **Codex 교차검증(2026-08-29)**: 다단계 추론이 가장 깊었던 두 자리(elementwise_mul nth=9/nth=11)를 짚어 확인 요청했고, 실제로 2건의 결함을 찾아냄 -- nth=9는 축 배정은 맞았지만 출처 인용이 틀린 소스 줄(850행 C_times_states)을 가리키고 있었고(진짜는 832행 states 계산), nth=11은 축5는 맞았지만 축2를 confirm(label: d_state)으로 잘못 기록해서 실제로는 override(d_state→d_chunk)여야 하는 자리를 놓치고 있었다. 둘 다 Codex 지시대로 수정(잘못된 confirm 삭제, override 추가, 이어지는 axis5 규칙의 shape 시그니처를 축2 수정 이후 상태로 업데이트) 후 재검증 -- FAIL 0/퇴행 0 유지, 나머지 7개 자리는 Codex도 전부 맞다고 확인.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [d_state, d_chunk] (ones/tril, 실제 [256, 256]) |
| 현재 라벨 | `d_state (축 0)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_chunk` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`segment_sum` 이 짓는 청크 내 인과 마스크다: `chunk_size = input_tensor.size(-1)` 로 받아 `mask = torch.tril(torch.ones(chunk_size, chunk_size, ...), diagonal=-1)` 를 만든다 -- **양 축 모두 chunk_size** 이므로 축 0 이 `d_state` 일 수 없다. 이 모델은 `mamba_chunk_size`(256) == `mamba_d_state`(256) 이라 값으로는 못 가리고, 정사각 마스크를 무엇으로 짓는지가 유일한 근거다.

이 자리를 규칙이 놓친 이유도 확인했다: 정사각 탐지기가 렌더된 **이름**이 같은지를 보고 있어서, 이름이 이미 `d_state`/`d_chunk` 로 갈린 정사각은 검사 자체가 건너뛰었다 -- 묻고 있는 질문의 답을 이미 안다고 가정한 셈이다. `source_check.square_labels` 를 **실제 크기**로 판정하도록 고쳤다(2026-08-15).

**근거 소스**: `develop/sources/modeling_falcon_h1.py` (`segment_sum`, `reshape_into_chunks`), `develop/sources/configuration_falcon_h1.py` (`mamba_chunk_size`, `mamba_d_state`).

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [B, n_h_ssm, n_kv, n_kv] (실제 [1, 24, 2, 2]) |
| 현재 라벨 | `n_kv` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `(청크 개수 축 -- 이름 없는 정수로 남겨야 한다)` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`n_kv` 가 아니다. 이 축은 **inter-chunk 재귀의 청크 개수**다: 트레이스 op143 이 `[B, n_h_ssm, 1] -> [B, n_h_ssm, 2]` 로 pad 하는데(초기 상태 1칸을 앞에 붙임), 그 결과 폭 2 = 청크 1개 + 초기 상태 1개다. 그 뒤 `segment_sum` 이 다시 불려 `[2, 2]` 감쇠 행렬을 만든다(op146 ones -> op147 tril). T 에 따라 변하는 값이라 config 심볼일 수 없다.

`num_key_value_heads` 가 2 라서 값이 겹쳤을 뿐이다. `n_kv` 는 `not_layer_types: [linear_attention, mamba, ...]` 로 SSM 을 제외하고 있는데, **병렬 하이브리드에서는 그 제외가 무력하다** -- 이 층의 유형은 `mamba` 가 아니라 `hybrid`(attention 과 SSM 이 한 층에 동시에 있음)이기 때문이다. 층 유형이 아니라 모듈 경로(`.mamba` vs `.self_attn`)로 갈라야 하는 자리다.

대체할 등록 심볼이 없어(`n_chunks` 는 T 파생이라 config 심볼이 아니다) 교정은 넣지 않고 `open` 으로 남긴다 -- 지어낸 이름을 넣는 것이 더 나쁘다.

**근거 소스**: `develop/sources/modeling_falcon_h1.py` (`torch_forward` 의 inter-chunk 구간, `segment_sum`), `rules/symbols.yaml:60` (`n_kv` 의 scope/not_layer_types).

**재확인(2026-08-29)**: 지금 트레이스에서 이 축은 더 이상 `n_kv`가 아니라 정직하게 bare 정수(레이어0 기준 `2`)로 렌더되고 있음을 확인했다(prefill.trace.raw.jsonl op_id 145/149 등, `model.layers.0.mamba`). 이 세션 밖에서 이미 별도로 고쳐졌고, `develop/verify/references.yaml`의 기존 주석("우리가 트레이스하는 seq_len은 항상 d_chunk보다 작아 n_chunks가 늘 1이고, 축은 말 그대로 정수 2다")과 정확히 일치한다.

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [d_chunk, d_chunk] 축 1 (ones/tril) |
| 현재 라벨 | `d_chunk` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

같은 `torch.ones(chunk_size, chunk_size, ...)` 의 둘째 축이다. 관례로 고른 이름이 맞았다 -- 축 0 만 틀렸다. `rules/label_confirmed.yaml` 에 기록해 인계 목록에서 종결한다.

**근거 소스**: `develop/sources/modeling_falcon_h1.py` (`segment_sum`).

## 발견 4 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | SSD naive scan(torch_forward) 전체에 걸친 d_chunk/d_state 값 충돌 -- 남은 18행(review_request.md 0절) |
| 현재 라벨 | `여러 자리에 걸쳐 d_state <-> d_chunk 혼용` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `각 자리마다 다름 -- rules/label_overrides.yaml/label_confirmed.yaml 참고` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

2026-08-15 판정(위 3건)은 segment_sum이 짓는 첫 정사각 마스크만 다뤘다. torch_forward(modeling_falcon_h1.py:674-878) 전체를 op_id 단위로 실측 대조하니 같은 값 충돌이 그 뒤로도 계속 퍼져 있었다 -- 근본 원인은 pad_tensor_by_size(:301-309, 자기 docstring: "Padding x tensor with pad_size on the seq_len dim")가 A*dt/B/C를 패딩하는 두 자리(op_id 96, 100)가 값-매칭으로 `d_state`를 받은 것: 이 fleet의 모든 트레이스는 seq_len < chunk_size라 패딩된 길이가 항상 정확히 한 청크(`d_chunk`)와 같기 때문에 실제로는 `d_chunk`여야 한다. sum/permute/elementwise_mul은 축 정체성을 새로 만들 수 없고 줄이거나(sum) 순서만 바꾸거나(permute) 브로드캐스트만 하므로(elementwise_mul), 이 두 자리의 오염이 그 뒤 7개 자리(op_id 124,127,132,137×2,167,168)까지 그대로 흘러갔다.

각 자리를 두 피연산자 중 브로드캐스트가 아닌(non-1) 쪽의 기존 라벨로 역추적해 판정: 값이 그대로 흘러온 자리는 `d_chunk`로 교정(9건, rules/label_overrides.yaml), 이미 맞게 렌더되던 형제 축은 확인으로 기록(16건, rules/label_confirmed.yaml, 기존 2026-08-15/그 이전 24건에 추가). segment_sum의 진짜 SSM 상태 축(recurrent state, C/B의 투영 폭, split_with_sizes 출력)은 전부 이미 정확히 `d_state`였다.

review_request.md의 0절(실제 조치 필요 표)이 18행 -> 0행으로 수렴했다. `develop/verify_all.py`가 각 rule의 (a) 매치 0건이면 dead verdict FAIL, (b) 등가류 개수가 어긋나면 FAIL, (c) 확인 기록에 소스 인용이 없으면 FAIL 을 강제해서 검증됨 -- 전부 통과, 함대 전체 퇴행 0건.

**근거 소스**: `develop/sources/modeling_falcon_h1.py:301-309`(pad_tensor_by_size), `:674-878`(torch_forward 전체), 실측 트레이스(prefill/decode.trace.raw.jsonl) op_id 67,68,87,96,100,101,103,118,119,123,124,127,131,132,137,160,161,163,167,168 대조.

**Codex 교차검증으로 발견/수정(2026-08-29)**: 다단계 역추론이 가장 깊었던 두 자리(op_id 137의 nth=9, op_id 167의 nth=11)를 짚어 확인 요청했더니 실제 결함 2건이 나왔다. (1) `nth=9`(op_id 137)는 축 배정 자체(2→d_chunk, 5→d_state)는 맞았지만 `source`가 엉뚱한 소스 줄(850행 `C_times_states`)을 인용하고 있었다 -- 진짜 근거는 832행 `states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)`. (2) `nth=11`(op_id 167, 진짜 `C_times_states`, 850행)은 축5(d_chunk→d_state)는 맞았는데, 축2를 `label_confirmed.yaml`에 `label: d_state`(현재 라벨이 맞다는 확인)로 잘못 기록해 뒀다 -- 실제로는 `C`의 축2 자체가 `l`(=`d_chunk`, 청크 내부 위치)이므로 `label_overrides.yaml`에 `d_state→d_chunk` 교정이 필요한 자리였다. 잘못된 confirm을 지우고 올바른 override를 추가했다(순서상 축5 규칙보다 먼저 적용돼야 축5 규칙의 shape 시그니처가 맞음 -- Codex가 순서까지 지정). 나머지 7개 자리(sum nth=1, elementwise_mul nth=7, permute nth=2/3, constant_pad_nd nth=3/5)는 Codex도 전부 맞다고 확인. 수정 후 재검증 -- develop/verify_all.py FAIL 0 / 퇴행 0 유지.
