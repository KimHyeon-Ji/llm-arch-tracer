# 라벨 검토 결과 — allenai/OLMo-2-1124-7B-Instruct

- 검토일: 2026-08-12
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 **모든** 질문에 답한다(기계가 개수를 맞춘다). 확인 프레임이 아니라 반박 프레임으로 — 각 라벨에 대해 '틀렸다는 증거'를 먼저 찾고, 못 찾은 것만 맞다고 적었다. 외부 검토가 준 팁 3가지(op 내부 필드 상호 대조 / 요청·응답 개수 diff / 반박 프레임)를 그대로 적용했다.
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

**근거 소스**: 이 판정은 `develop/sources/modeling_olmo2.py`, `develop/sources/configuration_olmo2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

**재확인 + 반영(2026-08-30)**: review_ledger가 STALE로 보고해 재확인. 실측 트레이스를 다시 보니 이 모델은 `n_kv`가 어디에도 렌더되지 않고 self_attn 전체가 `n_h`로만 나온다(2026-08-12 서술의 'KV 쪽은 n_kv'는 이 체크포인트의 실제 렌더와는 다름 -- MHA라 K/V도 Q와 같은 head 축을 그대로 쓰는 걸로 통일돼 있었다). 28개 앵커를 `rules/label_confirmed.yaml`에 전부 등록했다(2026-08-13 당시 등록 누락).

**정정(Codex 외부 검토, 2026-08-30 같은 날 늦게)**: 위 재정정이 틀렸다. MHA에서 `num_key_value_heads == num_attention_heads`인 것은 값이 같다는 것뿐, K/V 텐서의 계보(어느 projection이 만들었는지)까지 같아지는 게 아니다 -- `repeat_kv(..., n_rep=1)`이 항등 연산으로 축약돼도 원 텐서가 k_proj/v_proj에서 나왔다는 사실은 변하지 않는다. OLMoE와 동일한 14개 shape/axis/op_type/nth 조합(K·V쪽 transpose#1·#2·#3, view#1·#2, slice#2·#3의 prefill/decode 대응)을 `label_confirmed.yaml`에서 제거하고 `label_overrides.yaml`에 `n_h→n_kv`로 재등록했다. 2026-08-12 원래 판정('Q는 n_h, KV는 n_kv로 모듈 단위로 갈린다')이 맞았다.

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

**근거 소스**: 이 판정은 `develop/sources/modeling_olmo2.py`, `develop/sources/configuration_olmo2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)
