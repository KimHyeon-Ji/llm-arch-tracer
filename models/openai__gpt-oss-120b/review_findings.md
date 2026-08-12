# 라벨 검토 결과 — openai/gpt-oss-120b

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 의뢰서의 `2*d_moe` 는 이름이 옳았다 — 산술 휴리스틱이 내던 것을 규칙으로 승격했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | gate+up 융합 투영 폭 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))` — deepseek_v3.py:181 / qwen3_moe.py:218 / glm4_moe.py:350, gpt_oss.py:75 는 축 순서만 다르다 `(num_experts, hidden_size, 2 * intermediate_size)`. gate 와 up 을 파라미터 하나에 이어 붙인 폭이므로 2·d_moe 가 맞다.
