# 라벨 검토 결과 — nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

- 검토일: 2026-08-12
- 검토자: llm(claude, C절 전수 + 소스 대조)
- 본 것: **A·B·C절 전건 수행 완료.** C절은 (모듈, 라벨) 쌍 8,886건을 모집단으로 삼고, 심볼 자신의 scope 가 그 모듈을 덮지 않는 경우를 기계로 선별해(등록 유도식이 그 모듈 스코프로 설명하는 라벨은 제외) 20건을 전건 판정했다. 9건은 규칙 교정으로 닫았고 11건은 판정과 함께 남는다. 모집단·선별 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 3건 → 2건. `nemotron_h` 계열이라 **새 규칙 0개**로 들어왔고, T+1 스코프만 넓혔다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | [..., 2, 2] 의 뒤 두 축 |
| 현재 라벨 | `n_kv` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

n_kv=2 인데 `[..., 2, 2]` 로 렌더된다. 이 축은 Mamba2 청크 간 재귀의 청크 경계 상태 수다 — `modeling_nemotron_h.py:320` `decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)` → `[B, n_h_ssm, n_chunks+1, n_chunks+1]`. 우리가 트레이스하는 seq_len 은 항상 d_chunk(256)보다 작아 n_chunks=1 이므로 축은 말 그대로 2 이고, n_kv 와 우연히 같다. Nemotron-3-Nano 에서 같은 자리가 `k`(=2)로 붙었던 것과 동일한 건이다. **일반형(ceil(T/d_chunk)+1)으로 등록하지 않는다**: 관측한 적 없는 regime 을 주장하게 되고, 게이트의 라벨 검사가 `/` 를 floor division 으로 다뤄 그 식이 성립하지도 않는다. n_h/n_kv 에 `group: attn` 을 달아 계열 밖 폴백은 막았지만, 이 자리는 스코프가 아니라 값 충돌이라 남는다 — 정수로 두는 것이 정답이다.

## 발견 2 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | [B, T, 256] |
| 현재 라벨 | `2*d_state` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | low |
| 산출물 반영 | 미반영 |

**근거**

d_state=128 이라 2·d_state 와 값이 같지만, n_g_ssm=8 이므로 B/C 묶음(n_g·d_state=1024)은 아니다. Mamba2 in_proj 분할의 어느 조각인지 modeling 소스에서 확정하지 못했다 — 무엇을 봤는지만 남긴다. 값으로 우기지 않는다.

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | decode KV concat `[B, n_kv, T+1, d_state]` |
| 현재 라벨 | `T+1` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

nemotron_h 는 어텐션 층과 Mamba 층을 **모두 `mixer`** 라고 부른다. 그래서 어텐션 층의 KV concat 이 `attn` 스코프에 안 걸려 등록된 `T+1` 규칙 대신 산술로 다시 지어지고 있었다. 규칙 스코프에 `mixer` 를 추가해 해소했다.

## 발견 4 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer.shared_experts.{up,down}_proj` |
| 축 | 공유 전문가 FFN 폭 |
| 현재 라벨 | `d_moe (필드 미접지)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_nemotron_h.py:689` `self.shared_experts = NemotronHMLP(config=config, intermediate_size=config.moe_shared_expert_intermediate_size)`. config 는 `moe_intermediate_size` 와 `moe_shared_expert_intermediate_size` 를 따로 두고 이 체크포인트에서 둘의 값이 같다. 이름 `d_moe`(전문가 FFN 중간 폭)의 뜻은 맞지만 **별칭 표에 그 필드가 없어** 소속 검사가 접지 실패로 잡아냈다 — 값으로는 영원히 안 보였을 자리다. `rules/symbols.yaml` 의 d_moe 별칭에 추가했다.

## 발견 5 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer.{up,down}_proj` |
| 축 | FFN 중간 폭 |
| 현재 라벨 | `d_ff (스코프 밖 폴백)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Nemotron-H 는 **모든 블록을 `mixer` 라 부른다** — FFN 블록도 그렇다. 그래서 `mixer.up_proj` 의 가중치 `[d_ff, d_model]` 이 d_ff 스코프의 어떤 철자에도 안 걸렸고, 이름은 맞는데 근거가 '스코프 밖 폴백'이었다. 스코프에 `up_proj|down_proj` 를 추가했다 — expert/router 를 막는 음의 전방탐색은 그대로다.

## 발견 6 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer[.gate]` |
| 축 | Mamba2 chunk-scan decay 행렬의 뒤 두 축 (2) |
| 현재 라벨 | `n_kv / k` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:,:,:,-1], (1,0))))` → `[B, n_h_ssm, n_chunks+1, n_chunks+1]` (실측 `[1,128,2,2]`). 우리가 트레이스하는 regime 에서 T < d_chunk 라 n_chunks = 1 이고 그 축은 말 그대로 2 다. Nano 는 k(=2, num_experts_per_tok), Super/Ultra 는 n_kv(=2, GQA KV head 수)가 값이 같아 들어왔다 — 둘 다 Mamba mixer 와 아무 관계가 없다.

**일반형 `ceil(T/d_chunk)+1` 로 등록하지 않는다**: 관측한 적 없는 것을 주장하게 된다. 두 심볼 다 `group` 이 있어 스코프 밖 폴백에서는 배제되므로 재사용·전파 경로로 들어온 것이고, 그 경로를 막는 건 값이 겹치는 축 전반에 영향을 준다. 정수가 정답이라고 판정하고 남긴다.
