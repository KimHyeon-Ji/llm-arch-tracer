# 라벨 검토 결과 — Qwen/Qwen3-30B-A3B

- 검토일: 2026-08-12
- 검토자: llm(claude, C절 전수 + 소스 대조)
- 본 것: **A·B·C절 전건 수행 완료.** C절은 (모듈, 라벨) 쌍 8,886건을 모집단으로 삼고, 심볼 자신의 scope 가 그 모듈을 덮지 않는 경우를 기계로 선별해(등록 유도식이 그 모듈 스코프로 설명하는 라벨은 제외) 20건을 전건 판정했다. 9건은 규칙 교정으로 닫았고 11건은 판정과 함께 남는다. 모집단·선별 기준은 review/04-full-inventory.md.
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

`modeling_qwen3_moe.py:129` `self.head_dim = getattr(config, "head_dim", ...)` — 체크포인트가 주면 그 값을 쓰는 선택적 override 라 클래스가 선언하지 않는 것이 정상이다. `src/source_check.py.optional_config_reads` 로 이 패턴을 접지로 인정하게 했다.
