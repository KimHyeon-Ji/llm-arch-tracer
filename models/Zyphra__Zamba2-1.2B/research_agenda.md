# 조사 안건 — Zyphra/Zamba2-1.2B

라벨러가 **혼자 결정하지 못한 축**만 모았다. 각 항목은 `02-new-module-handling.md` Tier 2 절차로 확인한 뒤 근거와 함께 등록하면 다음 실행부터 자동으로 잡힌다. 비어 있으면 조사할 것이 없다는 뜻이다.

> 값이 같아서 못 가리는 것과 규칙이 없어서 못 붙이는 것은 다르다. 앞의 것은 **코드를 읽어야** 풀리고, 뒤의 것은 **등록만** 하면 된다. 아래에서 구분해 둔다.

## 판정

**코드 조사 필요**

| 성격 | 조치 | 해당 축 |
|---|---|---:|
| 값이 겹쳐 어느 쪽인지 미결 | modeling 소스를 읽어야 함 | 918 |
| 규칙이 없어 이름을 못 붙임 | 확인 후 규칙 등록 | 0 |
| 이름이 존재하지 않음 | 정수로 두는 것이 정직 | 2,698 |
| 정사각 투영 (알려진 패턴) | 조사 불필요 — 축 순서만의 문제 | 0 |

## 1. 한 shape 에 같은 이름이 두 번 — 어느 한쪽은 다른 이름이다

텐서의 두 축이 같은 이름을 받았다. 두 축의 크기가 우연히 같아 값으로는 못 가린다. **어느 축이 무엇인지는 모델 코드를 읽어야 안다.**

| 모듈 | 중복된 이름 | 렌더된 shape | 실제 크기 | 축 수 |
|---|---|---|---|---:|
| `model.layers.*.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm, 1]` | `[1, 1, 256, 256, 64, 1]` | 256 |
| `model.layers.*.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm]` | `[1, 1, 256, 256, 64]` | 192 |
| `model.layers.*.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm, d_state]` | `[1, 1, 256, 256, 64, 128]` | 64 |
| `model.layers.*.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm, n_h_ssm]` | `[1, 1, 256, 256, 64, 64]` | 64 |
| `model.layers.*.mamba` | `2` | `[B, d_head_ssm, 2, 2, 1]` | `[1, 64, 2, 2, 1]` | 64 |
| `model.layers.*.mamba` | `2` | `[B, d_head_ssm, 2, 2, 1, 1]` | `[1, 64, 2, 2, 1, 1]` | 64 |
| `model.layers.*.mamba` | `2` | `[B, d_head_ssm, 2, 2, n_h_ssm, d_state]` | `[1, 64, 2, 2, 64, 128]` | 64 |
| `model.layers.*.mamba_decoder.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm, 1]` | `[1, 1, 256, 256, 64, 1]` | 48 |
| `model.layers.*.mamba_decoder.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm]` | `[1, 1, 256, 256, 64]` | 36 |
| `model.layers.*.mamba_decoder.mamba` | `d_chunk` | `[B, 1, d_chunk, d_chunk, d_head_ssm, d_state]` | `[1, 1, 256, 256, 64, 128]` | 12 |

## 2. reshape 자기 유도와 라벨이 불일치 — 같은 텐서에 설명이 둘

reshape 는 자기 입력 축에서 출력 축을 유도할 수 있다. 그 유도와 붙어 있는 이름이 다르면 둘 중 하나가 틀렸다. **검사는 어느 쪽이 틀렸는지는 말해주지 않는다.**

| 모듈 | 현재 라벨 | 유도된 이름 | 축 수 |
|---|---|---|---:|
| `model.layers.*.shared_transformer.self_attn` | `d_attn` | `n_h*d_head` | 6 |

## 5. 설명 없는 정수 (상위)

이름이 붙지 않아 정수로 남은 축이다. 루프 인덱스나 데이터 의존 크기라면 **정수로 두는 것이 정직하다** — 전부 이름을 붙일 대상은 아니다.

| 모듈 | 값 | 축 수 |
|---|---:|---:|
| `model.layers.*.mamba` | 2 | 2272 |
| `model.layers.*.mamba_decoder.mamba` | 2 | 426 |

## 확인할 소스 (신뢰도 순 — 위에서 답이 나오면 아래는 생략)

1. 실행 중인 modeling 소스의 주석·변수명·docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py
2. 같은 저장소 config 클래스의 docstring — https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/configuration_zamba2.py
3. 독립 서빙 구현 (vLLM / SGLang / TensorRT-LLM) — https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
4. 저장소 README / 공식 model card — https://huggingface.co/Zyphra/Zamba2-1.2B
5. model card 가 링크한 논문·기술 리포트
6. 아키텍처 갤러리 (2차 자료 — 원 소스로 재확인) — https://sebastianraschka.com/llm-architecture-gallery/

## 답을 적는 곳

| 알아낸 것 | 적는 파일 |
|---|---|
| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |
| 이 값은 이런 식으로 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + 출처 주석) |
| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17 이 등재를 확인한다) |
| 둘 중 어느 쪽인지 사람이 정해야 한다 | `02-new-module-handling.md` Tier 3 |
