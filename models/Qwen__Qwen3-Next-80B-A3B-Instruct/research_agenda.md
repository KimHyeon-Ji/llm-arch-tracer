# 조사 안건 — Qwen/Qwen3-Next-80B-A3B-Instruct

라벨러가 **혼자 결정하지 못한 축**만 모았다. 각 항목은 `02-new-module-handling.md` Tier 2 절차로 확인한 뒤 근거와 함께 등록하면 다음 실행부터 자동으로 잡힌다. 비어 있으면 조사할 것이 없다는 뜻이다.

> 값이 같아서 못 가리는 것과 규칙이 없어서 못 붙이는 것은 다르다. 앞의 것은 **코드를 읽어야** 풀리고, 뒤의 것은 **등록만** 하면 된다. 아래에서 구분해 둔다.

## 판정

**코드 조사 필요**

| 성격 | 조치 | 해당 축 |
|---|---|---:|
| 값이 겹쳐 어느 쪽인지 미결 | modeling 소스를 읽어야 함 | 21,420 |
| 규칙이 없어 이름을 못 붙임 | 확인 후 규칙 등록 | 744 |
| 이름이 존재하지 않음 | 정수로 두는 것이 정직 | 48,960 |
| 정사각 투영 (알려진 패턴) | 조사 불필요 — 축 순서만의 문제 | 324 |

## 1. 한 shape 에 같은 이름이 두 번 — 어느 한쪽은 다른 이름이다

텐서의 두 축이 같은 이름을 받았다. 두 축의 크기가 우연히 같아 값으로는 못 가린다. **어느 축이 무엇인지는 모델 코드를 읽어야 안다.**

| 모듈 | 중복된 이름 | 렌더된 shape | 실제 크기 | 축 수 |
|---|---|---|---|---:|
| `model.layers.*.linear_attn` | `d_rope` | `[B, n_h_lin_v, 1, d_rope, d_rope]` | `[1, 32, 1, 64, 64]` | 7668 |
| `model.layers.*.linear_attn` | `d_rope` | `[n_h_lin_v, d_rope, d_rope]` | `[32, 64, 64]` | 360 |
| `model.layers.*.linear_attn` | `d_rope` | `[d_rope, d_rope]` | `[64, 64]` | 324 |
| `model.layers.*.linear_attn` | `d_rope` | `[B, n_h_lin_v, d_rope, d_rope]` | `[1, 32, 64, 64]` | 288 |
| `model.layers.*.linear_attn` | `n_kv` | `[B, n_h_lin_v, 1, n_kv, n_kv]` | `[1, 32, 1, 2, 2]` | 216 |
| `model.layers.*.linear_attn` | `3` | `[B, n_h_lin_v, 1, 3, 3]` | `[1, 32, 1, 3, 3]` | 216 |
| `model.layers.*.linear_attn` | `d_conv_lin` | `[B, n_h_lin_v, 1, d_conv_lin, d_conv_lin]` | `[1, 32, 1, 4, 4]` | 216 |
| `model.layers.*.linear_attn` | `5` | `[B, n_h_lin_v, 1, 5, 5]` | `[1, 32, 1, 5, 5]` | 216 |
| `model.layers.*.linear_attn` | `3*n_kv` | `[B, n_h_lin_v, 1, 3*n_kv, 3*n_kv]` | `[1, 32, 1, 6, 6]` | 216 |
| `model.layers.*.linear_attn` | `7` | `[B, n_h_lin_v, 1, 7, 7]` | `[1, 32, 1, 7, 7]` | 216 |

## 2. reshape 자기 유도와 라벨이 불일치 — 같은 텐서에 설명이 둘

reshape 는 자기 입력 축에서 출력 축을 유도할 수 있다. 그 유도와 붙어 있는 이름이 다르면 둘 중 하나가 틀렸다. **검사는 어느 쪽이 틀렸는지는 말해주지 않는다.**

| 모듈 | 현재 라벨 | 유도된 이름 | 축 수 |
|---|---|---|---:|
| `model.layers.*.linear_attn` | `d_model` | `n_h*d_head_lin_k` | 72 |
| `model.layers.*.linear_attn` | `n_h*d_head` | `n_h_lin_v*d_head_lin_k` | 72 |

## 3. 규칙 없이 산술로 지은 이름 — 등록하면 해결된다

값은 맞지만 근거가 규칙이 아니라 산술이다. 이번 트레이스의 seq_len 에서만 참일 수 있으므로, 확인 후 `rules/derived_dims.yaml` 에 **식과 출처**를 등록한다.

| 모듈 | 붙은 이름 | 방식 | 축 수 |
|---|---|---|---:|
| `model.layers.0.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.1.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.2.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.4.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.5.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.6.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.8.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.9.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.10.linear_attn` | `4*d_model` | heur_multiple | 62 |
| `model.layers.12.linear_attn` | `4*d_model` | heur_multiple | 62 |

## 5. 설명 없는 정수 (상위)

이름이 붙지 않아 정수로 남은 축이다. 루프 인덱스나 데이터 의존 크기라면 **정수로 두는 것이 정직하다** — 전부 이름을 붙일 대상은 아니다.

| 모듈 | 값 | 축 수 |
|---|---:|---:|
| `model.layers.*.linear_attn` | 3 | 1008 |
| `model.layers.*.linear_attn` | 5 | 1008 |
| `model.layers.*.linear_attn` | 7 | 1008 |
| `model.layers.*.linear_attn` | 9 | 1008 |
| `model.layers.*.linear_attn` | 11 | 1008 |
| `model.layers.*.linear_attn` | 13 | 1008 |

## 확인할 소스 (신뢰도 순 — 위에서 답이 나오면 아래는 생략)

1. 실행 중인 modeling 소스의 주석·변수명·docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
2. 같은 저장소 config 클래스의 docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py
3. 독립 서빙 구현 (vLLM / SGLang / TensorRT-LLM) — https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
4. 저장소 README / 공식 model card — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
5. model card 가 링크한 논문·기술 리포트
6. 아키텍처 갤러리 (2차 자료 — 원 소스로 재확인) — https://sebastianraschka.com/llm-architecture-gallery/

## 답을 적는 곳

| 알아낸 것 | 적는 파일 |
|---|---|
| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |
| 이 값은 이런 식으로 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + 출처 주석) |
| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17 이 등재를 확인한다) |
| 둘 중 어느 쪽인지 사람이 정해야 한다 | `02-new-module-handling.md` Tier 3 |
