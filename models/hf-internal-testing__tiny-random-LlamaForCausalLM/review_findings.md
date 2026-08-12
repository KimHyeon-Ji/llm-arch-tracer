# 라벨 검토 결과 — hf-internal-testing/tiny-random-LlamaForCausalLM

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 의뢰서의 질문에 전건 답했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | 4 를 두고 세 후보 |
| 현재 라벨 | `d_head vs n_h vs n_kv 동률` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

**테스트 픽스처**다(L=2, d_ff=64). head 수·KV head 수·head 폭이 전부 4 로 맞춰져 있어 값으로는 셋을 가를 수 없다 — 실제 아키텍처의 성질이 아니라 이 더미 config 가 그렇게 만들어진 것이다. 라벨은 앵커가 모듈 단위로 결정하며, 이 모델은 정확도 지표의 대상이 아니라 파이프라인 회귀 테스트용이다.
