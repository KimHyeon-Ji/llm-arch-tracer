# 라벨 검토 결과 — Qwen/Qwen2.5-0.5B

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 1건 — 오탐이다.

## 발견 1 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_proj` |
| 축 | [d_model, d_model] |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

n_h·d_head = 14·64 = 896 = hidden_size 라 q_proj 가중치가 정사각 `[896, 896]` 이다. 정사각 가중치이지 reshape 이 아니다.
