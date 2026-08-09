# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 2건 — 미등록으로 보고됐지만 실제로는 이미 이름이 있는 값이다.

## 발견 1 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | intermediate_size=8192 / expert_dim=8192 |
| 현재 라벨 | `(미등록)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |

**근거**

두 필드 모두 전문가 FFN 폭이고 우리 심볼 d_moe 가 이미 8192 로 해석하고 있다(dense FFN 은 `intermediate_size_mlp`=16384 → d_ff). 즉 '이름이 없는 값'이 아니라 **한 값에 config 필드가 둘**인 경우다. 탐지기(`src/symbolic_dims.probe`)가 '이미 등록된 심볼이 그 값을 설명하는가'를 안 보는 것이 오탐의 원인 — 다음 개선 대상으로 남긴다.
