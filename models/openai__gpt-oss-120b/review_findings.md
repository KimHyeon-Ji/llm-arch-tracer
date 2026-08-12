# 라벨 검토 결과 — openai/gpt-oss-120b

- 검토일: 2026-08-12
- 검토자: llm(claude, 자기모순 추적 + 소스 대조)
- 본 것: A·B·C절 전건 수행 완료 + 게이트가 센 자기모순을 출발점으로 하이브리드 스택 추적. 모집단·선별 기준은 review/04-full-inventory.md.
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
