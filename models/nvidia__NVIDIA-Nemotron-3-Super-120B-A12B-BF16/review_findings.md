# 라벨 검토 결과 — nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

- 검토일: 2026-08-10
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 신규 온보딩 검토 — 의뢰서 전수 + 소스 대조
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
