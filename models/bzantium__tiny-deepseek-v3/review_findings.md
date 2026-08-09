# 라벨 검토 결과 — bzantium/tiny-deepseek-v3

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 2건 — 하나는 이름이 틀렸고 하나는 맞았다.

## 발견 1 — 교정 필요

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.gate` |
| 축 | [T, n_grp, 4] 의 마지막 축 |
| 현재 라벨 | `2*n_grp` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `E/n_grp` |
| 확신도 | high |

**근거**

`modeling_deepseek_v3.py:149-155` 게이트가 점수를 그룹으로 접어 그룹 단위 top-k 를 한다: `[T,E] -> [T, n_group, E//n_group] -> topk(topk_group)`. E=8, n_grp=2 라 E/n_grp=4 가 2·n_grp 와 값이 같았다. 실측 `[[16, 8]] -> [[16, 2, 4]]` 가 그대로 보여준다. 규칙 등록 완료.

## 발견 2 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | gate+up 융합 폭 4096 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

`modeling_deepseek_v3.py:181` `gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))`. 이름이 맞아서 규칙으로 등록만 했다(전에는 산술 휴리스틱이 냈다).
