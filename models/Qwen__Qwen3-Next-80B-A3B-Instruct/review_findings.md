# 라벨 검토 결과 — Qwen/Qwen3-Next-80B-A3B-Instruct

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
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

`modeling_qwen3_next.py:420` `for i in range(1, chunk_size):` — chunked gated-delta 스캔이 청크 내부를 파이썬 루프로 돌며 `[..., i, i]` 를 잘라낸다. 트레이스에 X = 1,3,5,7,9,11,…,63 사다리가 그대로 남아 있고, 그중 config 값과 우연히 같은 정수만 이름을 받았다(2→n_kv, 4→d_conv_lin, 10→k, 6→3*n_kv, 8→n_h/n_kv, 12→3*d_conv_lin, 48→3*n_h …). **루프 인덱스는 이름이 없는 것이 정답이다.** 자동 탐지 방법도 확인했다: 같은 (module, op_type, 필드, 축위치)에서 관측된 정수들이 연속 사다리를 이루면 그 자리는 루프 인덱스다 — 함대 전체에서 이 패턴은 이 모델에만 있다.

## 발견 2 — 미확정

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | 4*d_model (744축) |
| 현재 라벨 | `4*d_model` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | low |

**근거**

linear_attn 의 in_proj 계열 폭으로 보이나 HF 소스의 torch 경로만으로는 확정하지 못했다. Qwen3-Next 의 linear attention 은 실제로는 fla/Triton 커널로 도는 부분이 있어 vLLM·커널 저장소 대조가 필요하다 — 이번에는 보지 못했다.
