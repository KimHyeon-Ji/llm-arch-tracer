# 라벨 검토 결과 — bzantium/tiny-deepseek-v3

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
- 요약: 의뢰서 2건 — 하나는 이름이 틀렸고 하나는 맞았다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.gate` |
| 축 | [T, n_grp, 4] 의 마지막 축 |
| 현재 라벨 | `2*n_grp` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `E/n_grp` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v3.py:149-155` 게이트가 점수를 그룹으로 접어 그룹 단위 top-k 를 한다: `[T,E] -> [T, n_group, E//n_group] -> topk(topk_group)`. E=8, n_grp=2 라 E/n_grp=4 가 2·n_grp 와 값이 같았다. 실측 `[[16, 8]] -> [[16, 2, 4]]` 가 그대로 보여준다. 규칙 등록 완료.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | gate+up 융합 폭 4096 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v3.py:181` `gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))`. 이름이 맞아서 규칙으로 등록만 했다(전에는 산술 휴리스틱이 냈다).
