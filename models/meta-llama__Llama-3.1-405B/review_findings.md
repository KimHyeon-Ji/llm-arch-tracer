# 라벨 검토 결과 — meta-llama/Llama-3.1-405B

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 의뢰서의 질문에 전건 답했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | 128 을 두고 두 후보 |
| 현재 라벨 | `d_head vs n_h 동률` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

num_attention_heads == head_dim == 128 이다. **이 충돌이 `src/anchors.py` 가 존재하는 이유** — 모듈이 선언한 폭(`nn.Linear.weight == [out, in]`)으로 결정하지 값으로 결정하지 않는다. 예전에 값으로 결정하던 시절 KV head-size 축 16,859개가 `n_h` 로 오라벨됐고, 그래서 `head_excl`(한 shape 에 n_h·n_kv 공존 금지) 불변식이 추가됐다. 현재 그 지표는 0 이다.

**주의**: 외부 검토가 지적한 이 모델의 진짜 문제는 이 동률이 아니라 q/k/v_proj 의 `weight_shape` 가 `input_shape` 의 같은 텐서와 다른 이름을 쓰던 것이었고, 그건 새 불변식 `weight_operand` 로 잡아 교정했다(함대 4,406건 → 0).
