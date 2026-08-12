# 라벨 검토 결과 — zai-org/GLM-4.5-Air

- 검토일: 2026-08-12
- 검토자: llm(claude, 양쪽 phase 전건 + 통과군 무작위 표본 감사)
- 본 것: **게이트가 이제 prefill·decode 양쪽을 본다**(그전까지 decode 는 한 번도 검사된 적이 없었다). A·B·C절 전건 + 통과군 무작위 표본 30건 감사. 기준은 review/04-full-inventory.md.
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

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head 가 읽은 head_dim |
| 현재 라벨 | `head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_glm4_moe.py:192` 가 Qwen3-MoE 와 같은 `getattr(config, "head_dim", ...)` 패턴이다.
