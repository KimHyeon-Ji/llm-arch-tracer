# 라벨 검토 결과 — Qwen/Qwen3-Next-80B-A3B-Instruct

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 4건 — 전부 루프 인덱스에 config 이름이 붙은 것이었다. 이번 검토에서 가장 큰 발견.

## 발견 1 — 이름 없음이 정답

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | [B, n_h_lin_v, 1, X, X] 의 뒤 두 축 (X=2,4,10 …) |
| 현재 라벨 | `n_kv / d_conv_lin / k` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수` |
| 확신도 | high |

**근거**

`modeling_qwen3_next.py:420` `for i in range(1, chunk_size):` — chunked gated-delta 스캔이 청크 내부를 파이썬 루프로 돌며 `[..., i, i]` 를 잘라낸다. 트레이스에 X = 1,3,5,7,9,11,…,63 사다리가 그대로 남아 있고, 그중 config 값과 우연히 같은 정수만 이름을 받았다(2→n_kv, 4→d_conv_lin, 10→k, 6→3*n_kv, 8→n_h/n_kv, 12→3*d_conv_lin, 48→3*n_h …). **루프 인덱스는 이름이 없는 것이 정답이다. 교정 완료(2026-08-09)**: `build_table._unname_loop_indices` — 같은 (module, op_type, 필드, 축위치)에서 관측된 정수가 8개 이상 연속 사다리를 이루면 그 자리는 반복 인덱스이므로 이름을 떼고 정수로 되돌린다. 사다리 63종 전부 정수가 됐고(지어낸 이름 10,656축 제거), 함대에서 이 패턴에 걸리는 모델은 이 하나뿐이다.

## 발견 2 — 교정 필요

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | conv1d 채널 폭 8192 (744축) |
| 현재 라벨 | `4*d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_k_lin+d_v_lin` |
| 확신도 | high |

**근거**

**확정: conv1d 채널 폭이다(2026-08-09).** `modeling_qwen3_next.py:520-529` `key_dim = head_k_dim * num_k_heads` (128·16=2048), `value_dim = head_v_dim * num_v_heads` (128·32=4096), `conv_dim = key_dim * 2 + value_dim` = 8192. d_model=2048 이라 4·d_model 과 값이 같았을 뿐이다. 같은 줄들의 `projection_size_qkvz = key_dim*2 + value_dim*2` 와 `projection_size_ba = num_v_heads*2` 도 함께 규칙으로 등록했다.
