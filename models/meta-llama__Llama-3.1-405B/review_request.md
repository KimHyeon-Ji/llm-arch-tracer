# 검토 의뢰서 — meta-llama/Llama-3.1-405B

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `llama`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_llama.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_llama.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_head vs n_h` in `model.layers.*.self_attn` — 값 128 를 두고 후보가 2개, 18648축

## 기계적으로 이미 확인된 것 — 다시 묻지 말 것

- **심볼이 읽은 config 필드**: 전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다
- **정사각 축**: 소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음
- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 6개를 소스에서 확인했다. 그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.
- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.

## 행 단위 전건 — 여기부터 읽는다

접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.

**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**

- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 (가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)
- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가
- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`
- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)

고유 행 36개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*.self_attn.k_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn.v_proj` | matmul | `[['T', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['T', 'n_kv*d_head']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_head'], ['n_h', 'd_head', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_head']]` | `None` | `[['n_h', 'T', 'd_head']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_head'], ['n_h*d_head', 'n_h*d_head']]` | `['n_h*d_head', 'n_h*d_head']` | `[['T', 'n_h*d_head']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | silu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.down_proj` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_h*d_head']]` | `['n_h*d_head', 'd_model']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*.self_attn.k_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn.v_proj` | matmul | `[['B', 'd_model'], ['d_model', 'n_kv*d_head']]` | `['n_kv*d_head', 'd_model']` | `[['B', 'n_kv*d_head']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_head'], ['n_h', 'd_head', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_head']]` | `None` | `[['n_h', 'B', 'd_head']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_head'], ['n_h*d_head', 'n_h*d_head']]` | `['n_h*d_head', 'n_h*d_head']` | `[['B', 'n_h*d_head']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | silu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (13종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.q_proj` 외 139개 | 47504 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.q_proj` 외 139개 | 30170 |
| `d_head` | 128 | `model.layers.*.self_attn`, `model.rotary_emb` | 21698 |
| `d_model` | 16384 | `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.k_proj` 외 134개 | 19706 |
| `n_h` | 128 | `model.layers.*.self_attn` | 18648 |
| `n_kv` | 8 | `model.layers.*.self_attn` | 11340 |
| `d_ff` | 53248 | `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj`, `model.layers.*.mlp` 외 1개 | 7308 |
| `n_h*d_head` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn` | 6552 |
| `T+1` |  | `model.layers.*.self_attn` | 6048 |
| `n_kv*d_head` |  | `model.layers.*.self_attn.k_proj`, `model.layers.*.self_attn.v_proj`, `model.layers.*.self_attn` | 4536 |
| `d_head/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 3060 |
| `n_h/n_kv` |  | `model.layers.*.self_attn` | 2016 |
| `V` | 128256 | `lm_head`, `model.embed_tokens` | 20 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (143개 모듈 / 370종)

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
- `model.embed_tokens`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.*.input_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
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
- `model.layers.*.self_attn`
  - `[[B, 1, 1, n_h]]`
  - `[[B, 1, T, n_h]]`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, 1, n_h, d_head]]`
  - `[[B, 1, n_kv, d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, T, n_h, d_head]]`
  - `[[B, T, n_kv, d_head]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head/2]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, T+1, d_head]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head/2]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, d_head, T+1]]`
  - `[[B, n_h, d_head, T]]`
  - `[[B, n_kv, 1, T+1, d_head]]`
  - `[[B, n_kv, 1, T, d_head]]`
  - `[[B, n_kv, 1, d_head/2]]`
  - `[[B, n_kv, 1, d_head]]`
  - `[[B, n_kv, T+1, d_head]]`
  - `[[B, n_kv, T, d_head/2]]`
  - `[[B, n_kv, T, d_head]]`
  - `[[B, n_kv, n_h/n_kv, T+1, d_head]]`
  - `[[B, n_kv, n_h/n_kv, T, d_head]]`
  - `[[T, T]]`
  - `[[]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_head]]`
  - `[[n_h, T+1, d_head]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_head]]`
  - `[[n_h, d_head, T+1]]`
  - `[[n_h, d_head, T]]`
- `model.layers.*.self_attn.k_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, n_h*d_head]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, n_h*d_head]]`
- `model.layers.*.self_attn.q_proj`
  - `[[B, 1, n_h*d_head]]`
  - `[[B, T, n_h*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_head]]`
  - `[[n_h*d_head, d_model]]`
- `model.layers.*.self_attn.v_proj`
  - `[[B, 1, n_kv*d_head]]`
  - `[[B, T, n_kv*d_head]]`
  - `[[B, d_model]]`
  - `[[B, n_kv*d_head]]`
  - `[[T, d_model]]`
  - `[[T, n_kv*d_head]]`
  - `[[d_model, n_kv*d_head]]`
- `model.layers.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.1`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.10`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.100`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.101`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.102`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.103`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.104`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.105`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.106`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.107`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.108`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.109`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.11`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.110`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.111`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.112`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.113`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.114`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.115`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.116`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.117`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.118`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.119`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.12`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.120`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.121`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.122`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.123`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.124`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.125`
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
- `model.layers.18`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.19`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.2`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.20`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.21`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.22`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.23`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.24`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.25`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.26`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.27`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.28`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.29`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.3`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.30`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.31`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.32`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.33`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.34`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.35`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.36`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.37`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.38`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.39`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.4`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.40`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.41`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.42`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.43`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.44`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.45`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.46`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.47`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.48`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.49`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.5`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.50`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.51`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.52`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.53`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.54`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.55`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.56`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.57`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.58`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.59`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.6`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.60`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.61`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.62`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.63`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.64`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.65`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.66`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.67`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.68`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.69`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.7`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.70`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.71`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.72`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.73`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.74`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.75`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.76`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.77`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.78`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.79`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.8`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.80`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.81`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.82`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.83`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.84`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.85`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.86`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.87`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.88`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.89`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.9`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.90`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.91`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.92`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.93`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.94`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.95`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.96`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.97`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.98`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.99`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
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
