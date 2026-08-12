# 라벨 검토 결과 — Qwen/Qwen3.5-397B-A17B

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | DeltaNet 청크 루프·head 폭 |
| 현재 라벨 | `(Qwen3.5/3.6 계열과 동일)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Qwen3-Next 가 만든 Gated DeltaNet 규칙이 그대로 적용됐다 — **전용 규칙 0개**. 남은 reshape 90 / matmul 135 건은 형제 모델들과 같은 원인이다(`linear_key_head_dim == linear_value_head_dim`, 개명 전파 막힘). 그 판정은 Qwen3.5-4B / 3.6-27B / 3.6-35B 의 findings 에 이미 기록돼 있다.

## 발견 2 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_ff 미확인 |
| 현재 라벨 | `—` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

이 체크포인트는 `intermediate_size` 가 없다 — dense FFN 이 아예 없는 순수 MoE 다. `d_ff` 는 group 이 없어서 '해당 없음' 대신 '미확인'으로 표시된다. 라벨 오류가 아니라 표기 문제이고, d_ff 에 group 을 다는 것은 전 함대에 영향을 주므로 하지 않았다.
