# 검토 의뢰서 — NX-AI/xLSTM-7b

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `xlstm`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_xlstm.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_xlstm.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/xlstm

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 1. 이 config 필드가 정말 이 뜻인가

값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 (체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.

- `d_head ← head_dim`

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 1건이 클래스 선언 밖 — 위 1절 참고
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 7개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (10종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `backbone.blocks.*.mlstm_layer.mlstm_backend`, `backbone.blocks.*.mlstm_layer`, `backbone.blocks.*.mlstm_layer.multihead_norm`, `backbone.blocks.*.norm_mlstm` 외 51개 | 79478 |
| `n_h` | 8 | `backbone.blocks.*.mlstm_layer.mlstm_backend`, `backbone.blocks.*.mlstm_layer`, `backbone.blocks.*.mlstm_layer.multihead_norm`, `backbone` 외 2개 | 76992 |
| `d_model*qk_f/n_h` |  | `backbone.blocks.*.mlstm_layer.mlstm_backend`, `backbone`, `backbone.blocks.*.mlstm_layer` | 33792 |
| `d_head` | 512 | `backbone.blocks.*.mlstm_layer.mlstm_backend`, `backbone.blocks.*.mlstm_layer.multihead_norm`, `backbone.blocks.*.mlstm_layer`, `backbone` | 19840 |
| `d_model` | 4096 | `backbone.blocks.*.mlstm_layer.v`, `backbone.blocks.*.mlstm_layer.ogate_preact`, `backbone.blocks.*.mlstm_layer.out_proj`, `backbone.blocks.*.norm_mlstm` 외 46개 | 8870 |
| `T` |  | `backbone.blocks.*.mlstm_layer.mlstm_backend`, `backbone.blocks.*.mlstm_layer`, `backbone.blocks.*.mlstm_layer.multihead_norm`, `backbone.blocks.*.norm_mlstm` 외 50개 | 8251 |
| `roundup(d_model*ffn_f` |  | `backbone.blocks.*.ffn.proj_up_gate`, `backbone.blocks.*.ffn.proj_up`, `backbone.blocks.*.ffn.proj_down`, `backbone.blocks.*.ffn` 외 1개 | 1856 |
| `ffn_r)` |  | `backbone.blocks.*.ffn.proj_up_gate`, `backbone.blocks.*.ffn.proj_up`, `backbone.blocks.*.ffn.proj_down`, `backbone.blocks.*.ffn` 외 1개 | 1856 |
| `d_model*qk_f` |  | `backbone.blocks.*.mlstm_layer.q`, `backbone.blocks.*.mlstm_layer.k`, `backbone.blocks.*.mlstm_layer` | 1152 |
| `V` | 50304 | `lm_head`, `(root)`, `backbone.embeddings` | 32 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (55개 모듈 / 199종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

- `(root)`
  - `[[B, 1, V]]`
  - `[[B, T, V]]`
- `backbone`
  - `[[B, n_h, 1]]`
  - `[[B, n_h, d_model*qk_f/n_h, d_head]]`
  - `[[B, n_h, d_model*qk_f/n_h]]`
  - `[[B]]`
- `backbone.blocks.*.ffn`
  - `[[B, 1, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, T, roundup(d_model*ffn_f,ffn_r)]]`
- `backbone.blocks.*.ffn.act_fn`
  - `[[B, 1, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, T, roundup(d_model*ffn_f,ffn_r)]]`
- `backbone.blocks.*.ffn.proj_down`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[T, d_model]]`
  - `[[T, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[roundup(d_model*ffn_f,ffn_r), d_model]]`
- `backbone.blocks.*.ffn.proj_up`
  - `[[B, 1, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, T, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, d_model]]`
  - `[[B, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[T, d_model]]`
  - `[[T, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[d_model, roundup(d_model*ffn_f,ffn_r)]]`
- `backbone.blocks.*.ffn.proj_up_gate`
  - `[[B, 1, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, T, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[B, d_model]]`
  - `[[B, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[T, d_model]]`
  - `[[T, roundup(d_model*ffn_f,ffn_r)]]`
  - `[[d_model, roundup(d_model*ffn_f,ffn_r)]]`
- `backbone.blocks.*.mlstm_layer`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, d_model*qk_f/n_h]]`
  - `[[B, 1, n_h]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_h, d_model*qk_f/n_h]]`
  - `[[B, T, n_h]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_model*qk_f/n_h]]`
  - `[[B, n_h, 1]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_model*qk_f/n_h]]`
  - `[[B, n_h, T]]`
- `backbone.blocks.*.mlstm_layer.fgate_preact`
  - `[[B, 1, n_h]]`
  - `[[B, T, n_h]]`
  - `[[B, d_model]]`
  - `[[B, n_h]]`
  - `[[T, d_model]]`
  - `[[T, n_h]]`
  - `[[d_model, n_h]]`
- `backbone.blocks.*.mlstm_layer.igate_preact`
  - `[[B, 1, n_h]]`
  - `[[B, T, n_h]]`
  - `[[B, d_model]]`
  - `[[B, n_h]]`
  - `[[T, d_model]]`
  - `[[T, n_h]]`
  - `[[d_model, n_h]]`
- `backbone.blocks.*.mlstm_layer.k`
  - `[[B, 1, d_model*qk_f]]`
  - `[[B, T, d_model*qk_f]]`
  - `[[B, d_model*qk_f]]`
  - `[[B, d_model]]`
  - `[[T, d_model*qk_f]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model*qk_f]]`
- `backbone.blocks.*.mlstm_layer.mlstm_backend`
  - `[[B, n_h, 1, 1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_model*qk_f/n_h]]`
  - `[[B, n_h, 1], [B, n_h, 1]]`
  - `[[B, n_h, 1]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_model*qk_f/n_h]]`
  - `[[B, n_h, T]]`
  - `[[B, n_h, d_head]]`
  - `[[B, n_h, d_model*qk_f/n_h, 1]]`
  - `[[B, n_h, d_model*qk_f/n_h, d_head]]`
  - `[[B, n_h, d_model*qk_f/n_h]]`
  - `[[B, n_h]]`
  - `[[n_h, B, 1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, B, d_model*qk_f/n_h]]`
  - `[[n_h, d_model*qk_f/n_h, B]]`
  - `[[n_h, d_model*qk_f/n_h, d_head]]`
- `backbone.blocks.*.mlstm_layer.multihead_norm`
  - `[[B, 1, d_model]]`
  - `[[B, 1, n_h, 1]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, d_model]]`
  - `[[B, T, n_h, 1]]`
  - `[[B, T, n_h, d_head]]`
- `backbone.blocks.*.mlstm_layer.ogate_act_fn`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.*.mlstm_layer.ogate_preact`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `backbone.blocks.*.mlstm_layer.out_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `backbone.blocks.*.mlstm_layer.q`
  - `[[B, 1, d_model*qk_f]]`
  - `[[B, T, d_model*qk_f]]`
  - `[[B, d_model*qk_f]]`
  - `[[B, d_model]]`
  - `[[T, d_model*qk_f]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model*qk_f]]`
- `backbone.blocks.*.mlstm_layer.v`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
  - `[[d_model, d_model]]`
- `backbone.blocks.*.norm_ffn`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.*.norm_mlstm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.1`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.10`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.11`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.12`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.13`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.14`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.15`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.16`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.17`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.18`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.19`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.2`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.20`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.21`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.22`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.23`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.24`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.25`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.26`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.27`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.28`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.29`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.3`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.30`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.31`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.4`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.5`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.6`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.7`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.8`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.blocks.9`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.embeddings`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `backbone.out_norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `lm_head`
  - `[[B, 1, V]]`
  - `[[B, T, V]]`
  - `[[B, V]]`
  - `[[B, d_model]]`
  - `[[T, V]]`
  - `[[T, d_model]]`
  - `[[d_model, V]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
