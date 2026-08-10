# 라벨 검토 결과 — deepseek-ai/DeepSeek-V2-Lite

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
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

## 발견 2 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | [B, 1, 1, d_rope/2, 2] 의 마지막 축 |
| 현재 라벨 | `2 (이름 없음)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

바로 그 축을 `view_as_complex` / `view_as_real` 이 소비한다 — 복소수 한 개의 실수부·허수부 쌍이지 아키텍처 차원이 아니다. RoPE 를 복소수 곱으로 구현하는 표준 형태이고, 이 모델의 `E_shared`(=2)와 값이 같은 것은 우연이다. **정수로 두는 것이 정답이다.** B절이 지목했고 여기서 종결한다.
