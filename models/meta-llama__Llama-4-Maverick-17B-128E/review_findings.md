# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-09-11
- 검토자: codex(외부) — 산출물 4개(prefill/decode의 csv/jsonl)를 공식 모델 카드와 설치본 소스로 대조
- 본 것: 심볼 값 / 축 이름 / 빠진 구조 / 이름 없는 축 / 표기 형식 다섯 갈래로 표를 전수 검토했다. 값이 겹치는 자리(d_head=E=128, d_moe=w_local=8192, E_shared=k=1)를 특히 봤다.
- 요약: 차원 매핑은 정확. decode attention 의 가짜 `B` 축은 **좁은 범위 교정**으로 고쳤고(공개 CSV 12칸), 공개 표기 3건도 고쳤다. 표 생성 구조 2건은 `accepted_limit` 로 공개 요약에 명시한다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `self_attn` |
| 축 | decode bmm 축 1 |
| 현재 라벨 | `B` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `1` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

구체 shape 이 `[40, 1, 128]` 인데 가운데 축이 배치가 아니라 decode 의 query 길이 1 이다. 실행이 B=1 이라 값이 같아 잘못 골랐다.

**전역 규칙으로 올리려다 물렸다.** 처음에 '축 0 이 아닌 B 294,408개가 전부 구체값 1' 을 반례 검사라고 내세웠는데, `symbolic_shape` 이 `if n == 1: return _r("runtime", "B")` 이라 `B` 는 크기 1 축에만 붙는다 -- 반례가 나올 수 없는 관찰이었다(외부 검토가 짚었다).

진짜 판별식은 B=2 재트레이스다(`develop/batch_axis_probe.py`):
    decode op130  B=1 [40, 1, 128] -> B=2 [80, 1, 128]  (축 0 만 2배, 축 1 은 1 그대로)
    MoE           [E, B*T, d_moe] 의 축 1 은 배치를 따라 2배가 된다
즉 축 1 이 배치 의존일 수 있으므로 전역 규칙은 틀렸다. `rules/label_overrides.yaml` 에 **소스로 확인된 self_attn 의 batched_matmul 에만** 걸었다(192축 발화).

공개 CSV 변경은 decode 12칸, prefill 무변화.
  [인용] modeling_llama4.py:362 `hidden_shape = (*input_shape, -1, self.head_dim)` — `input_shape` 가 `(B, T_q)` 이고 decode 는 `T_q = 1` 이다.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | ctx |
| 현재 라벨 | `262144` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `ctx_config=262144 / ctx_public=1048576` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

이 checkpoint 의 config 값은 262,144 지만 Meta 공식 모델 카드는 Maverick 의 컨텍스트를 1M 으로 공개한다. 두 값을 한 칸에 담을 수 없으므로 분리해야 한다.

[2026-09-11 조치] `structure.yaml` 의 `context` 블록에 `config_max_position_embeddings: 262144` 와 `public_context_length: 1048576` 을 함께 싣는다. `references.yaml` 은 출고되지 않으므로 공개 파일에 직접 넣었다.
  [인용] configuration_llama4.py 의 `max_position_embeddings` = 262,144. 공개값은 https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md (Maverick: 128 experts / 1M context).

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | w_local |
| 현재 라벨 | `w_local` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `chunk_size` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

값 8192 는 맞지만 sliding window 가 아니다. `masking_utils.py` 는 query 와 key 의 `index // chunk_size` 가 같은지 검사하는 **고정 청크**다. `w_local` 이라는 이름이 sliding window 를 뜻하므로 오해를 부른다.

[2026-09-11 조치] `chunk_size` 를 갈라내고 **group 도 분리**했다(`sliding_window` / `chunked_attention`). 같은 group 에 두었더니 Llama-4 의 `w_local` 이 '해당 없음' 이 아니라 '미확인 -- Tier 2 대상' 으로 떠서 새 오해를 만들었다.
  [인용] `attention_chunk_size` 는 configuration_llama4.py 에만 2회 나오고 modeling_llama4.py 에는 없다. 청킹은 masking_utils.py:104 가 query/key 의 `index // chunk_size` 로 구현한다 — sliding window 가 아니다.

## 발견 4 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | E_shared |
| 현재 라벨 | `1` |
| 판정 | `undetermined` |
| 제안 라벨 | `shared_expert_modules=1` |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`num_shared_experts` 같은 config 필드는 없다. shared MLP 모듈이 하나 있다는 구조 사실이므로 축 심볼표가 아니라 구조 메타데이터에 두는 편이 정확하다. Meta 공식 표기도 '128 experts' 이지 129 가 아니다.  [2026-09-11 조치] 이 모델에서 E_shared 는 축 라벨로 쓰이지 않는다(다른 모델에서는 쓰인다). structure.yaml 의 known_limits 에 "텐서 축이 아니라 shared expert 모듈 수" 로 명시했다.
  [인용] https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md 는 Maverick 을 '128 experts' 로 적는다(shared MLP 를 더해 129 라 하지 않는다). config 에 `num_shared_experts` 같은 필드도 없다.

## 발견 5 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*` |
| 축 | block_type |
| 현재 라벨 | `attn+MoE` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `chunked+RoPE+MoE / full+NoPE+temp+MoE` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

레이어 접기 자체는 옳다(3그룹: 0,2,..=dense / 1,5,..=chunked+MoE / 3,7,..=full+NoPE+MoE). 그런데 뒤 두 그룹이 같은 `attn+MoE` 로 찍혀 독자가 왜 같아 보이는 블록이 두 번 나오는지 알 수 없다. chunked/full 과 RoPE/NoPE 차이가 표에 드러나지 않는다. `no_rope_layers` 는 이름과 달리 1 이 RoPE 사용이다.
  [인용] configuration_llama4.py:175-200 의 `no_rope_layers` 는 이름과 달리 1 이 RoPE 사용이고, modeling_llama4.py:327-385 가 그 값을 `self.use_rope` 에 넣고 같은 값으로 chunked/full 을 정한다.

## 발견 6 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `feed_forward` |
| 축 | MoE 결합 |
| 현재 라벨 | `elementwise_add 한 행` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `shared+routed add 와 residual add 를 분리` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

소스는 `shared_out += routed_sum` 다음에 `residual + combined` 두 단계다(modeling_llama4.py:166-174, 450-458). 지금 표는 add 한 행이 의존성 셋을 물어 두 단계를 하나로 뭉갠다.

## 발견 7 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(전체)` |
| 축 | 표에 없는 연산 |
| 현재 라벨 | `` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

major 표에 chunked/full 마스크 생성과 score 합산, RoPE, NoPE 층의 temperature tuning, GQA 의 KV head 8->40 반복, router 의 topk/scatter, KV cache update/concat 이 안 보인다. QK-norm 이 없는 것은 누락이 아니다 -- 이 checkpoint 는 `use_qk_norm=false` 다. temperature tuning 은 T=16 이라 scale 이 수치상 1 이고, 8192 청크 경계도 T=16 으로는 검증할 수 없다.
  [인용] modeling_llama4.py:549-579 와 :305-314 가 full/chunked 마스크를 따로 만들어 레이어마다 고르고 eager attention 에서 score 에 더한다. QK-norm 이 없는 것은 누락이 아니다 -- 이 checkpoint 는 `use_qk_norm=false` 이고 소스도 128E 모델에는 없다고 적는다. 모델 카드: https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md

## 발견 8 — 미확정 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | 산출물 범위 |
| 현재 라벨 | `` |
| 판정 | `undetermined` |
| 제안 라벨 | `text-only 명시` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

공식 Maverick 은 vision tower 와 multimodal projector 를 가진 native multimodal 모델인데 이 트레이스는 text-only forward 다. 메타데이터에 `text-only / no pixel_values`, `B=1`, `prefill_len=16`, `decode_cache_len=16` 을 적어야 오해가 없다.

[2026-09-11 조치] `structure.yaml` 의 `scope` 블록에 text-only / batch / 길이 / 표의 성격을 명시했다.
  [인용] https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md 는 Maverick 을 vision tower 와 multimodal projector 를 가진 native multimodal 모델로 적는다. 이 트레이스는 `input_ids` 만 넣는 text-only forward 다.

## 발견 9 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.feed_forward.experts` |
| 축 | routed 입력 폭 B*E*T |
| 현재 라벨 | `B*E*T / [E, B*T, ...]` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 설치된 구현은 routed 입력을 [E,B*T,d_model] 로 확장하고 top-1 이외 expert 입력을 0 으로 만들어 **전체 expert 에 bmm** 한다. 따라서 표의 B*E*T 는 맞다. **이걸 B*k*T 로 바꾸면 현재 trace 를 잘못 표현한다** -- sparse 최적화 구현의 실제 연산량과는 구분해야 한다. modeling_llama4.py:79-83,147-173.

## 발견 10 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | chunked attention 이 shape 에 안 보임 |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. T=17, decode 전체 길이 18 은 chunk_size=8192 안이라 chunked/full 양쪽의 eager attention shape 가 같은 것이 정상이다. 차이는 마스크에 있다. RoPE 와 NoPE 층의 temperature scaling 도 attention shape 만으로는 알 수 없다. modeling_llama4.py:560-573,368-385.
