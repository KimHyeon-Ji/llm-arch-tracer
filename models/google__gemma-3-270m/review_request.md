# 검토 의뢰서 — google/gemma-3-270m

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `gemma3_text`
- 판단 필요: **0건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_gemma3_text.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_gemma3_text.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3_text

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.

그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 9개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 49개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.embed_tokens` | elementwise_mul | `[['B', 'T', 'd_model'], []]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.q_norm` | rmsnorm | `[['B', 'n_h', 'T', 'd_head']]` | `['d_head']` | `[['B', 'n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.k_norm` | rmsnorm | `[['B', '1', 'T', 'd_head']]` | `['d_head']` | `[['B', '1', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.pre_feedforward_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | gelu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.down_proj` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.post_feedforward_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.embed_tokens` | elementwise_mul | `[['B', '1', 'd_model'], []]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['B', 'd_head']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_head']]` | `['d_head', 'd_model']` | `[['B', 'd_head']]` |
| decode | `model.layers.*.self_attn.q_norm` | rmsnorm | `[['B', 'n_h', '1', 'd_head']]` | `['d_head']` | `[['B', 'n_h', '1', 'd_head']]` |
| decode | `model.layers.*.self_attn.k_norm` | rmsnorm | `[['B', '1', '1', 'd_head']]` | `['d_head']` | `[['B', '1', '1', 'd_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'w_local']]` | `None` | `[['n_h', 'B', 'w_local']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'w_local']]` | `None` | `[['B', 'n_h', '1', 'w_local']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'w_local'], ['n_h', 'w_local', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'd_model']]` | `['d_model', 'n_h*d_head']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.pre_feedforward_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | gelu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.post_feedforward_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (12종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` 외 36개 | 9630 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm` 외 36개 | 5506 |
| `d_head` | 256 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_norm`, `model.layers.*.self_attn.k_norm`, `model.layers.*.self_attn.k_proj` 외 2개 | 5236 |
| `d_model` | 640 | `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.pre_feedforward_layernorm`, `model.layers.*.post_feedforward_layernorm` 외 29개 | 4598 |
| `n_h` | 4 | `model.layers.*.self_attn`, `model.layers.*.self_attn.q_norm` | 3306 |
| `d_ff` | 2048 | `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj`, `model.layers.*.mlp` 외 1개 | 1044 |
| `w_local` | 512 | `model.layers.*.self_attn`, `model` | 785 |
| `n_h*d_head` |  | `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 648 |
| `d_head/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 504 |
| `T+1` |  | `model.layers.*.self_attn` | 144 |
| `w_local-1` |  | `model.layers.*.self_attn` | 90 |
| `V` | 262144 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (40개 모듈 / 203종)

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
- `model`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, T]]`
  - `[[B, 1, 1, w_local]]`
  - `[[B, 1, 1]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T]]`
  - `[[B, 1, w_local]]`
  - `[[B, 1]]`
  - `[[B, T]]`
  - `[[B, w_local]]`
  - `[[B]]`
  - `[[T]]`
  - `[[]]`
  - `[[w_local]]`
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.layers.*.mlp`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.mlp.act_fn`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
- `model.layers.*.mlp.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_ff, d_model]]`
- `model.layers.*.mlp.gate_proj`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.mlp.up_proj`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.post_attention_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.layers.*.post_feedforward_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.layers.*.pre_feedforward_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.layers.*.self_attn`
  - `[[B, 1, 1, T+1, d_head]]`
  - `[[B, 1, 1, T, d_head]]`
  - `[[B, 1, 1, d_head/2]]`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, 1, w_local, d_head]]`
  - `[[B, 1, 1, w_local]]`
  - `[[B, 1, T+1, d_head]]`
  - `[[B, 1, T, T]]`
  - `[[B, 1, T, d_head/2]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, T+1, d_head]]`
  - `[[B, 1, n_h, T, d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_h, w_local, d_head]]`
  - `[[B, 1, w_local, d_head]]`
  - `[[B, 1, w_local-1, d_head]]`
  - `[[B, T, 1, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head/2]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, w_local]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head/2]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_h, d_head, w_local]]`
  - `[[B, n_h, w_local, d_head]]`
  - `[[T, T]]`
  - `[[]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, B, w_local]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
  - `[[n_h, d_head, w_local]]`
  - `[[n_h, w_local, d_head]]`
- `model.layers.*.self_attn.k_norm`
  - `[[B, 1, 1, 1]]`
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, T, 1]]`
  - `[[B, 1, T, d_head]]`
  - `[[d_head]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_head]]`
  - `[[B, d_model]]`
  - `[[T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, d_head]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.self_attn.q_norm`
  - `[[B, n_h, 1, 1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T, 1]]`
  - `[[B, n_h, T, d_head]]`
  - `[[d_head]]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[d_model, n_h*d_head]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head]]`
  - `[[B, d_head]]`
  - `[[B, d_model]]`
  - `[[T, d_head]]`
  - `[[T, d_model]]`
  - `[[d_model, d_head]]`
- `model.layers.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.1`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.10`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.11`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.12`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.13`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.14`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.15`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.16`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.17`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.2`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.3`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.4`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.5`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.6`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.7`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.8`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.9`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
  - `[[d_model]]`
- `model.rotary_emb`
  - `[[B, 1, 1]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_head/2]]`
  - `[[B, 1, d_head]]`
  - `[[B, T, d_head/2]]`
  - `[[B, T, d_head]]`
  - `[[B, d_head/2, 1]]`
  - `[[B, d_head/2, T]]`
  - `[[B, d_head/2]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
