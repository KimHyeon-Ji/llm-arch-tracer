# 라벨 검토 결과 — Qwen/Qwen2.5-0.5B

- 검토일: 2026-08-12
- 검토자: llm(claude, 양쪽 phase 전건 + 통과군 무작위 표본 감사)
- 본 것: **게이트가 이제 prefill·decode 양쪽을 본다**(그전까지 decode 는 한 번도 검사된 적이 없었다). A·B·C절 전건 + 통과군 무작위 표본 30건 감사. 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 1건 — 정사각 자체는 정상이지만, 파고드니 같은 파라미터가 두 이름으로 렌더되는 진짜 오류가 나왔다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_proj` |
| 축 | [d_model, d_model] (정사각 여부) |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

n_h·d_head = 14·64 = 896 = hidden_size 라 q_proj 가중치가 정사각 `[896, 896]` 이다. 정사각 가중치이지 reshape 이 아니다 — 탐지기가 가중치를 묻지 않도록 고쳤다.

## 발견 2 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_proj` |
| 축 | 가중치 축 이름 |
| 현재 라벨 | `weight_shape=[n_h*d_head, n_h*d_head] / 피연산자=[d_model, d_model]` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[n_h*d_head, d_model]` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`Qwen2Attention.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim)` — out=n_h·d_head, in=d_model 이다. 그런데 **한 파라미터가 두 이름으로 나온다**: weight_shape 열은 `[n_h*d_head, n_h*d_head]`, 같은 파라미터가 `t`/`linear` 의 피연산자로 나올 때는 `[d_model, d_model]`. 어느 쪽도 `[out, in]` 이 아니다. **원인 진단**: `build_table._canonical_weight_labels` 가 앵커 적용 **전에** 계산된다 — `_contraction_pin` 이 활성화의 마지막 축 라벨을 가져오는데, `self_attn` 스코프 안에서 896 은 `n_h*d_head`(스코프 있는 유도식)로 먼저 해석되어 d_model 을 이긴다. 그 값이 가중치의 in 축에 박히고 canon 으로 굳는다. 고치려면 canon 을 앵커의 선언 in/out 으로 계산해야 한다 — 이번 회차에서는 진단까지만 하고 남긴다.
