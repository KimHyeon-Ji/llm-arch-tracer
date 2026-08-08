# 조사 안건 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

라벨러가 **혼자 결정하지 못한 축**만 모았다. 각 항목은 `02-new-module-handling.md` Tier 2 절차로 확인한 뒤 근거와 함께 등록하면 다음 실행부터 자동으로 잡힌다. 비어 있으면 조사할 것이 없다는 뜻이다.

> 값이 같아서 못 가리는 것과 규칙이 없어서 못 붙이는 것은 다르다. 앞의 것은 **코드를 읽어야** 풀리고, 뒤의 것은 **등록만** 하면 된다. 아래에서 구분해 둔다.

## 판정

**코드 조사 필요**

| 성격 | 조치 | 해당 축 |
|---|---|---:|
| 값이 겹쳐 어느 쪽인지 미결 | modeling 소스를 읽어야 함 | 928 |
| 규칙이 없어 이름을 못 붙임 | 확인 후 규칙 등록 | 384 |
| 이름이 존재하지 않음 | 정수로 두는 것이 정직 | 0 |
| 정사각 투영 (알려진 패턴) | 조사 불필요 — 축 순서만의 문제 | 504 |

## 1. 한 shape 에 같은 이름이 두 번 — 어느 한쪽은 다른 이름이다

텐서의 두 축이 같은 이름을 받았다. 두 축의 크기가 우연히 같아 값으로는 못 가린다. **어느 축이 무엇인지는 모델 코드를 읽어야 안다.**

| 모듈 | 중복된 이름 | 렌더된 shape | 실제 크기 | 축 수 |
|---|---|---|---|---:|
| `model.layers.*.mixer` | `d_chunk` | `[d_chunk, d_chunk]` | `[256, 256]` | 252 |
| `model.layers.*.mixer` | `k` | `[k, k]` | `[2, 2]` | 252 |
| `model.layers.*.mixer` | `d_chunk` | `[B, n_h_ssm, 1, d_chunk, d_chunk]` | `[1, 96, 1, 256, 256]` | 210 |
| `model.layers.*.mixer` | `k` | `[B, n_h_ssm, k, k]` | `[1, 96, 2, 2]` | 210 |
| `model.layers.*.mixer` | `d_chunk` | `[B, 1, d_chunk, d_chunk, n_h_ssm, 1]` | `[1, 1, 256, 256, 96, 1]` | 168 |
| `model.layers.*.mixer` | `d_chunk` | `[B, 1, d_chunk, d_chunk, n_h_ssm]` | `[1, 1, 256, 256, 96]` | 126 |
| `model.layers.*.mixer` | `d_chunk` | `[B, 1, d_chunk, d_chunk, n_h_ssm, d_state]` | `[1, 1, 256, 256, 96, 128]` | 42 |
| `model.layers.*.mixer` | `d_chunk` | `[B, 1, d_chunk, d_chunk, n_h_ssm, d_head_ssm]` | `[1, 1, 256, 256, 96, 80]` | 42 |
| `model.layers.*.mixer` | `k` | `[B, n_h_ssm, k, k, 1]` | `[1, 96, 2, 2, 1]` | 42 |
| `model.layers.*.mixer` | `k` | `[B, n_h_ssm, k, k, 1, 1]` | `[1, 96, 2, 2, 1, 1]` | 42 |

## 2. reshape 자기 유도와 라벨이 불일치 — 같은 텐서에 설명이 둘

reshape 는 자기 입력 축에서 출력 축을 유도할 수 있다. 그 유도와 붙어 있는 이름이 다르면 둘 중 하나가 틀렸다. **검사는 어느 쪽이 틀렸는지는 말해주지 않는다.**

| 모듈 | 현재 라벨 | 유도된 이름 | 축 수 |
|---|---|---|---:|
| `model.layers.*.mixer` | `n_h*d_head` | `n_h*d_state` | 4 |

## 3. 규칙 없이 산술로 지은 이름 — 등록하면 해결된다

값은 맞지만 근거가 규칙이 아니라 산술이다. 이번 트레이스의 seq_len 에서만 참일 수 있으므로, 확인 후 `rules/derived_dims.yaml` 에 **식과 출처**를 등록한다.

| 모듈 | 붙은 이름 | 방식 | 축 수 |
|---|---|---|---:|
| `model.layers.12.mixer` | `T+1` | heur_plus1 | 48 |
| `model.layers.17.mixer` | `T+1` | heur_plus1 | 48 |
| `model.layers.24.mixer` | `T+1` | heur_plus1 | 48 |
| `model.layers.32.mixer` | `T+1` | heur_plus1 | 48 |
| `model.layers.0.mixer` | `3*d_conv` | heur_multiple | 24 |
| `model.layers.2.mixer` | `3*d_conv` | heur_multiple | 24 |
| `model.layers.4.mixer` | `3*d_conv` | heur_multiple | 24 |
| `model.layers.6.mixer` | `3*d_conv` | heur_multiple | 24 |
| `model.layers.7.mixer` | `3*d_conv` | heur_multiple | 24 |
| `model.layers.9.mixer` | `3*d_conv` | heur_multiple | 24 |

## 확인할 소스 (신뢰도 순 — 위에서 답이 나오면 아래는 생략)

1. 실행 중인 modeling 소스의 주석·변수명·docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py
2. 같은 저장소 config 클래스의 docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/configuration_nemotron_h.py
3. 독립 서빙 구현 (vLLM / SGLang / TensorRT-LLM) — https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
4. 저장소 README / 공식 model card — https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
5. model card 가 링크한 논문·기술 리포트
6. 아키텍처 갤러리 (2차 자료 — 원 소스로 재확인) — https://sebastianraschka.com/llm-architecture-gallery/

## 답을 적는 곳

| 알아낸 것 | 적는 파일 |
|---|---|
| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |
| 이 값은 이런 식으로 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + 출처 주석) |
| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17 이 등재를 확인한다) |
| 둘 중 어느 쪽인지 사람이 정해야 한다 | `02-new-module-handling.md` Tier 3 |
