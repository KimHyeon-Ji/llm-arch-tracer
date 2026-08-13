# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-08-12
- 검토자: llm(claude, 행 단위 전건 — 검토자 방식)
- 본 것: 의뢰서에 **행 단위 전건 절**을 앞에 세우고 그 뷰로 다시 봤다. 접힌 표의 고유 행은 모델당 중앙값 62개(최대 136)뿐이라 전부 읽을 수 있다 — A/B/C 절보다 작으면서 한 행 안의 어긋남까지 보인다.
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

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.feed_forward.experts / shared_expert.*` |
| 축 | 전문가 FFN 폭 8192 |
| 현재 라벨 | `d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_llama4.py:59-61` `self.intermediate_size = config.intermediate_size; self.expert_dim = self.intermediate_size` — Llama-4 는 전문가 폭에 `moe_intermediate_size` 가 아니라 그냥 `intermediate_size` 를 쓰고, dense 쪽은 `intermediate_size_mlp`(:411)를 쓴다. 이름 `d_moe` 의 뜻은 정확하다. 소속 검사가 처음에 이걸 지적했는데 **탐지기 쪽 한계**였다 — 이 심볼이 어느 필드에서 값을 읽었는지 기록이 없으면 무엇과 대조할지 알 수 없다. 그런 심볼에 대해서는 아무 주장도 하지 않도록 고쳤다(침묵은 근거가 아니다).
