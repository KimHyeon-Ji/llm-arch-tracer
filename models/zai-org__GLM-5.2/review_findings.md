# 라벨 검토 결과 — zai-org/GLM-5.2

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
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

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.indexer.wq_b` |
| 축 | 입력 폭 (q_lora_rank) |
| 현재 라벨 | `k_I` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `c_q` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

GLM-5.2 는 index_topk == q_lora_rank == d_moe == 2048 인 삼중 충돌이다. 위 V4 수정으로 '안쪽 스코프가 이긴다'를 넣자 `k_I`(scope `indexer`)가 `c_q`(scope `attn`)를 눌러 압축 Q latent 폭을 가져갔다 — 축 156건 퇴행으로 게이트가 잡았다. **선택 개수는 폭이 아니다**: 어떤 단계가 몇 개를 남기는지는 그 앞 파라미터가 몇 폭인지와 무관하다. `_SELECTION_SYMS`(k, k_I)는 값 동률에서 규칙이 더 위로 매긴 심볼을 이길 수 없게 했다(`symbolic_shape._pick`). DeepSeek-V3 의 라우터에서는 `k`(우선순위 22)가 `n_grp`(38)를 여전히 이긴다.
