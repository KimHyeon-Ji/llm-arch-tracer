# 라벨 검토 결과 — allenai/OLMo-2-1124-7B-Instruct

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 의뢰서의 질문에 전건 답했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | head 수 축 (32) |
| 현재 라벨 | `n_h vs n_kv 동률` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

OLMo-2-7B 은 GQA 가 아니라 **MHA** 다 — num_attention_heads == num_key_value_heads == 32. 두 이름이 같은 값을 갖는 것이 구조 그 자체이지 충돌이 아니다. 한 shape 에 둘이 동시에 나오는 것은 `head_excl` 불변식이 이미 막고 있고(현재 0건), Q 쪽 텐서는 n_h, KV 쪽은 n_kv 로 앵커가 모듈 단위로 가른다.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | 루트의 4096 |
| 현재 라벨 | `ctx vs d_model 동률` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

max_position_embeddings == hidden_size == 4096 이다. `ctx` 는 스코프가 `wpe|embed_positions|position_embeddings` 라 루트를 덮지 않으므로 d_model 이 이긴다 — 그리고 그게 맞다. OLMo-2 는 RoPE 라 학습형 위치 임베딩 테이블이 아예 없다.
