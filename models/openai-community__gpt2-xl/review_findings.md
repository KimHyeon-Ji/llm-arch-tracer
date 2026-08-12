# 라벨 검토 결과 — openai-community/gpt2-xl

- 검토일: 2026-08-12
- 검토자: llm(claude, 자기모순 추적 + 소스 대조)
- 본 것: A·B·C절 전건 수행 완료 + 게이트가 센 자기모순을 출발점으로 하이브리드 스택 추적. 모집단·선별 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 1건 — FFN 폭이 이름 대신 산술로 지어져 있었다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `transformer.h.*.mlp.act` |
| 축 | FFN 중간 폭 6400 |
| 현재 라벨 | `4*d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_ff` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_gpt2.py:250` `inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size`. gpt2-xl 은 `n_inner=null` 이라 모델이 4·hidden 을 계산해 쓴다 — 값은 맞지만 축의 이름은 FFN 중간 폭이다. `summarize.resolve_symbols` 에 modeling 과 같은 폴백을 넣었다(model_type=gpt2 한정).
