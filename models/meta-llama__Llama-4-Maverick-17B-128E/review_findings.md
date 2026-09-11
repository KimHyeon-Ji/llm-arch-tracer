# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-09-11
- 검토자: codex(외부) — 산출물 4개(prefill/decode의 csv/jsonl)를 공식 모델 카드와 설치본 소스로 대조
- 본 것: 심볼 값 / 축 이름 / 빠진 구조 / 이름 없는 축 / 표기 형식 다섯 갈래로 표를 전수 검토했다. 값이 겹치는 자리(d_head=E=128, d_moe=w_local=8192, E_shared=k=1)를 특히 봤다.
- 요약: 차원 매핑은 대체로 정확. decode attention 의 가짜 `B` 축 1건은 고쳤고, 표현 누락 6건은 열어 둔다.

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

구체 shape 이 `[40, 1, 128]` 인데 가운데 축이 배치가 아니라 decode 의 query 길이 1 이다. 실행이 B=1 이라 값이 같아 잘못 골랐다. 함대를 세어보니 축 0 이 아닌 자리의 `B` 294,408개가 **전부 구체값 1** 이라 반례가 없어, `symbolic_shape` 의 배치 불변식을 '배치는 맨 앞에만' 으로 강화했다. decode.csv 30칸이 `B` -> `1` 로 바뀌었고 prefill 은 무변화.

## 발견 2 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | ctx |
| 현재 라벨 | `262144` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `ctx_config=262144 / ctx_public=1048576` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

이 checkpoint 의 config 값은 262,144 지만 Meta 공식 모델 카드는 Maverick 의 컨텍스트를 1M 으로 공개한다. 두 값을 한 칸에 담을 수 없으므로 분리해야 한다.

## 발견 3 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | w_local |
| 현재 라벨 | `w_local` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `chunk_size` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

값 8192 는 맞지만 sliding window 가 아니다. `masking_utils.py` 는 query 와 key 의 `index // chunk_size` 가 같은지 검사하는 **고정 청크**다. `w_local` 이라는 이름이 sliding window 를 뜻하므로 오해를 부른다.

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

`num_shared_experts` 같은 config 필드는 없다. shared MLP 모듈이 하나 있다는 구조 사실이므로 축 심볼표가 아니라 구조 메타데이터에 두는 편이 정확하다. Meta 공식 표기도 '128 experts' 이지 129 가 아니다.

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

## 발견 8 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | 산출물 범위 |
| 현재 라벨 | `` |
| 판정 | `undetermined` |
| 제안 라벨 | `text-only 명시` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

공식 Maverick 은 vision tower 와 multimodal projector 를 가진 native multimodal 모델인데 이 트레이스는 text-only forward 다. 메타데이터에 `text-only / no pixel_values`, `B=1`, `prefill_len=16`, `decode_cache_len=16` 을 적어야 오해가 없다.
