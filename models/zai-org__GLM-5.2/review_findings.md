# 라벨 검토 결과 — zai-org/GLM-5.2

- 검토일: 2026-08-10
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 신규 온보딩 검토 — 의뢰서 전수 + 소스 대조
- 요약: 의뢰서가 비어 있었다. **다른 벤더의 새 아키텍처가 기존 규칙만으로 전부 설명된 첫 사례**다 — 새 규칙 0개, 휴리스틱 0.00%, 미등록 config 필드 0.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(전체)` |
| 축 | DSA indexer + MLA 축 |
| 현재 라벨 | `n_h_I / c_I / k_I / c_q / c_kv / d_nope / d_v / d_rope` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`model_type: glm_moe_dsa` 는 Zhipu 가 DeepSeek Sparse Attention 을 채택한 것으로, config 의 `index_head_dim`(128) / `index_n_heads`(32) / `index_topk`(2048) 가 DeepSeek-V4 의 indexer 와 같은 자리를 차지한다. V4 용으로 등록해 둔 심볼이 그대로 맞았고 (실측 n_h_I=32, c_I=128, k_I=2048), MLA 쪽도 c_q=2048 / c_kv=512 / d_nope=192 / d_v=256 / d_rope=64 로 전부 해결됐다. 게이트 FAIL 0, 축 326,319개 중 지어낸 이름 0개.
