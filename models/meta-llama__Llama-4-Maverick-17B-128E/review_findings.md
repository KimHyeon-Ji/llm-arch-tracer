# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
- 요약: 의뢰서 2건 — 미등록으로 보고됐지만 실제로는 이미 이름이 있는 값이다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | intermediate_size=8192 / expert_dim=8192 |
| 현재 라벨 | `(미등록)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

두 필드 모두 전문가 FFN 폭이고 우리 심볼 d_moe 가 이미 8192 로 해석하고 있다(dense FFN 은 `intermediate_size_mlp`=16384 → d_ff). 즉 '이름이 없는 값'이 아니라 **한 값에 config 필드가 둘**인 경우다. 탐지기(`src/symbolic_dims.probe`)가 '이미 등록된 심볼이 그 값을 설명하는가'를 안 보는 것이 오탐의 원인 — 다음 개선 대상으로 남긴다.
