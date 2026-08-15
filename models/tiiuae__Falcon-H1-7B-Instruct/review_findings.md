# 라벨 검토 결과 — tiiuae/Falcon-H1-7B-Instruct

- 검토일: 2026-08-15
- 검토자: llm(claude, 블라인드 온보딩 테스트의 ③ 소스 대조)
- 본 것: 이 모델은 블라인드 테스트 대상이다 -- 사전 조사 없이 프로파일 한 줄로 트레이스하고, 의뢰서가 낸 항목만 소스로 판정했다. 항목마다 `develop/sources/modeling_falcon_h1.py` 의 해당 함수를 열어 축 위치를 대조했다. 이 아키텍처의 함정은 `mamba_chunk_size == mamba_d_state == 256` 이라 두 심볼을 값으로는 원리적으로 못 가린다는 것이다.
- 요약: 3건 전부 판정했다. 2건은 오라벨로 확정(1건 교정 완료, 1건은 대체할 심볼이 없어 open), 1건은 관례 선택이 맞았음을 확인했다.

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

## 발견 2 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [B, n_h_ssm, n_kv, n_kv] (실제 [1, 24, 2, 2]) |
| 현재 라벨 | `n_kv` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `(청크 개수 축 -- 이름 없는 정수로 남겨야 한다)` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`n_kv` 가 아니다. 이 축은 **inter-chunk 재귀의 청크 개수**다: 트레이스 op143 이 `[B, n_h_ssm, 1] -> [B, n_h_ssm, 2]` 로 pad 하는데(초기 상태 1칸을 앞에 붙임), 그 결과 폭 2 = 청크 1개 + 초기 상태 1개다. 그 뒤 `segment_sum` 이 다시 불려 `[2, 2]` 감쇠 행렬을 만든다(op146 ones -> op147 tril). T 에 따라 변하는 값이라 config 심볼일 수 없다.

`num_key_value_heads` 가 2 라서 값이 겹쳤을 뿐이다. `n_kv` 는 `not_layer_types: [linear_attention, mamba, ...]` 로 SSM 을 제외하고 있는데, **병렬 하이브리드에서는 그 제외가 무력하다** -- 이 층의 유형은 `mamba` 가 아니라 `hybrid`(attention 과 SSM 이 한 층에 동시에 있음)이기 때문이다. 층 유형이 아니라 모듈 경로(`.mamba` vs `.self_attn`)로 갈라야 하는 자리다.

대체할 등록 심볼이 없어(`n_chunks` 는 T 파생이라 config 심볼이 아니다) 교정은 넣지 않고 `open` 으로 남긴다 -- 지어낸 이름을 넣는 것이 더 나쁘다.

**근거 소스**: `develop/sources/modeling_falcon_h1.py` (`torch_forward` 의 inter-chunk 구간, `segment_sum`), `rules/symbols.yaml:60` (`n_kv` 의 scope/not_layer_types).

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
