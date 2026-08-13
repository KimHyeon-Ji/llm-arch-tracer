# 검토 의뢰서 — tiiuae/falcon-7b

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `falcon`
- 판단 필요: **0건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_falcon.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_falcon.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.

그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 26개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `transformer.word_embeddings` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `transformer.h.*.input_layernorm` | layernorm | `[['B', 'T', 'd_model'], ['d_model'], ['d_model']]` | `['d_model']` | `[['B', 'T', 'd_model'], ['B', 'T', '1'], ['B', 'T', '1']]` |
| prefill | `transformer.h.*.self_attention.query_key_value` | matmul | `[['T', 'd_model'], ['d_model', '(n_h+2*n_kv)*d_head']]` | `['(n_h+2*n_kv)*d_head', 'd_model']` | `[['T', '(n_h+2*n_kv)*d_head']]` |
| prefill | `transformer.h.*.self_attention` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `transformer.h.*.self_attention` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `transformer.h.*.self_attention` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `transformer.h.*.self_attention.dense` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['T', 'd_model']]` |
| prefill | `transformer.h.*.mlp.dense_h_to_4h` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `transformer.h.*.mlp.act` | gelu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `transformer.h.*.mlp.dense_4h_to_h` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `transformer.h.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `transformer.ln_f` | layernorm | `[['B', 'T', 'd_model'], ['d_model'], ['d_model']]` | `['d_model']` | `[['B', 'T', 'd_model'], ['B', 'T', '1'], ['B', 'T', '1']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `transformer.word_embeddings` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `transformer.h.*.input_layernorm` | layernorm | `[['B', '1', 'd_model'], ['d_model'], ['d_model']]` | `['d_model']` | `[['B', '1', 'd_model'], ['B', '1', '1'], ['B', '1', '1']]` |
| decode | `transformer.h.*.self_attention.query_key_value` | matmul | `[['B', 'd_model'], ['d_model', '(n_h+2*n_kv)*d_head']]` | `['(n_h+2*n_kv)*d_head', 'd_model']` | `[['B', '(n_h+2*n_kv)*d_head']]` |
| decode | `transformer.h.*.self_attention` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `transformer.h.*.self_attention` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `transformer.h.*.self_attention` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `transformer.h.*.self_attention.dense` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['B', 'd_model']]` |
| decode | `transformer.h.*.mlp.dense_h_to_4h` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `transformer.h.*.mlp.act` | gelu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `transformer.h.*.mlp.dense_4h_to_h` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `transformer.h.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `transformer.ln_f` | layernorm | `[['B', '1', 'd_model'], ['d_model'], ['d_model']]` | `['d_model']` | `[['B', '1', 'd_model'], ['B', '1', '1'], ['B', '1', '1']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (12종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `transformer.h.*.self_attention`, `transformer.h.*.input_layernorm`, `transformer.h.*.self_attention.query_key_value`, `transformer.h.*.self_attention.dense` 외 41개 | 9286 |
| `d_head` | 64 | `transformer.h.*.self_attention`, `transformer.rotary_emb` | 5786 |
| `T` |  | `transformer.h.*.self_attention`, `transformer.h.*.self_attention.query_key_value`, `transformer.h.*.self_attention.dense`, `transformer.h.*.mlp.dense_h_to_4h` 외 41개 | 5715 |
| `n_h` | 71 | `transformer.h.*.self_attention` | 4032 |
| `d_model` | 4544 | `transformer.h.*.self_attention.query_key_value`, `transformer.h.*.self_attention.dense`, `transformer.h.*.mlp.dense_h_to_4h`, `transformer.h.*.mlp.dense_4h_to_h` 외 37개 | 2722 |
| `T+1` |  | `transformer.h.*.self_attention`, `transformer` | 1199 |
| `d_ff` | 18176 | `transformer.h.*.mlp.dense_h_to_4h`, `transformer.h.*.mlp.dense_4h_to_h`, `transformer.h.*.mlp.act` | 1152 |
| `d_head/2` |  | `transformer.h.*.self_attention`, `transformer.rotary_emb` | 804 |
| `(n_h+2*n_kv)*d_head` |  | `transformer.h.*.self_attention.query_key_value`, `transformer.h.*.self_attention` | 576 |
| `n_h*d_head` |  | `transformer.h.*.self_attention.dense`, `transformer.h.*.self_attention` | 576 |
| `n_h+2*n_kv` |  | `transformer.h.*.self_attention` | 256 |
| `V` | 65024 | `lm_head`, `transformer.word_embeddings` | 20 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (45개 모듈 / 163종)

모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 self_attn 안에).

- `(root)`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `lm_head`
  - `[[B, 1, V]]`
  - `[[B, T, V]]`
  - `[[B, V]]`
  - `[[B, d_model]]`
  - `[[T, V]]`
  - `[[T, d_model]]`
  - `[[d_model, V]]`
- `transformer`
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, T+1]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1]]`
  - `[[B, T+1]]`
  - `[[B, T, T]]`
  - `[[B]]`
  - `[[T+1]]`
  - `[[T, 1]]`
  - `[[T, T]]`
  - `[[T]]`
  - `[[]]`
- `transformer.h.*.input_layernorm`
  - `[[B, 1, d_model], [B, 1, 1], [B, 1, 1]]`
  - `[[B, T, d_model], [B, T, 1], [B, T, 1]]`
- `transformer.h.*.mlp.act`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `transformer.h.*.mlp.dense_4h_to_h`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_ff, d_model]]`
- `transformer.h.*.mlp.dense_h_to_4h`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `transformer.h.*.self_attention`
  - `[[B, 1, 1, T+1]]`
  - `[[B, 1, 1, d_head/2]]`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T+1, d_head]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T, d_head/2]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, d_head, T+1]]`
  - `[[B, 1, d_head, T]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h+2*n_kv, d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, T, 1, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h+2*n_kv, d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head/2]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head/2]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
- `transformer.h.*.self_attention.dense`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
- `transformer.h.*.self_attention.query_key_value`
  - `[[B, (n_h+2*n_kv)*d_head]]`
  - `[[B, 1, (n_h+2*n_kv)*d_head]]`
  - `[[B, T, (n_h+2*n_kv)*d_head]]`
  - `[[B, d_model]]`
  - `[[T, (n_h+2*n_kv)*d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, (n_h+2*n_kv)*d_head]]`
- `transformer.h.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.1`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.10`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.11`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.12`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.13`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.14`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.15`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.16`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.17`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.18`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.19`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.2`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.20`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.21`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.22`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.23`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.24`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.25`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.26`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.27`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.28`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.29`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.3`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.30`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.31`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.4`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.5`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.6`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.7`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.8`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.h.9`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `transformer.ln_f`
  - `[[B, 1, d_model], [B, 1, 1], [B, 1, 1]]`
  - `[[B, T, d_model], [B, T, 1], [B, T, 1]]`
- `transformer.rotary_emb`
  - `[[B, 1, 1]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_head/2]]`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head/2]]`
  - `[[B, T, d_head]]`
  - `[[B, d_head/2, 1]]`
  - `[[B, d_head/2, T]]`
  - `[[B, d_head/2]]`
- `transformer.word_embeddings`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
