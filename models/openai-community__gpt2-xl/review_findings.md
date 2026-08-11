# 라벨 검토 결과 — openai-community/gpt2-xl

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
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
