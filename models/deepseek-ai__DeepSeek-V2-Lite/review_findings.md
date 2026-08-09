# 라벨 검토 결과 — deepseek-ai/DeepSeek-V2-Lite

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소
- 요약: 의뢰서 3건 — 산술은 맞지만 이름이 틀렸다. 규칙으로 교정 완료.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.shared_experts.{gate,up,down}_proj` |
| 축 | 공유 전문가 FFN 폭 2816 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `E_shared*d_moe` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v2.py:127` `DeepseekV2MLP(config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)`. n_shared_experts=2, d_moe=1408 이라 2·1408 과 값이 같지만 그 2 는 gate+up 의 2 가 아니라 **공유 전문가 수**다. `rules/derived_dims.yaml` 에 `E_shared*d_moe` 를 shared_expert 스코프로 등록했고, gate+up 규칙보다 앞에 두어 순서로 이기게 했다.
