# 검토 의뢰서 — bzantium/tiny-deepseek-v3

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `deepseek_v3`
- 판단 필요: **8건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_deepseek_v3.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_deepseek_v3.py` — 있음, 이 파일을 열어서 판정한다

- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/deepseek_v3

그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), `structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).

## 판단이 필요한 것

### 6. 값이 겹쳐 **임의로** 고른 축

두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.

**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 `rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). 출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.

- `d_nope vs d_v vs n_h vs n_kv` in `model.layers.*.self_attn` — 값 128 를 두고 후보가 4개, 1128축
- `E vs k` in `model.layers.*.mlp.experts` — 값 8 를 두고 후보가 2개, 436축
- `d_nope vs d_v` in `model.layers.*.self_attn` — 값 128 를 두고 후보가 2개, 282축
- `E vs k` in `model.layers.*.mlp.gate` — 값 8 를 두고 후보가 2개, 200축
- `d_head vs d_rope` in `model.layers.*.self_attn` — 값 64 를 두고 후보가 2개, 192축
- `k_grp vs n_grp` in `model.layers.*.mlp.gate` — 값 2 를 두고 후보가 2개, 126축
- `d_head vs d_rope` in `model.rotary_emb` — 값 64 를 두고 후보가 2개, 26축
- `E vs k` in `model.layers.*.mlp.experts.act_fn` — 값 8 를 두고 후보가 2개, 12축

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

고유 행 70개.

| phase | 모듈 | op | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|
| prefill | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', 'T']]` | `['V', 'd_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.input_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.self_attn.q_a_proj` | matmul | `[['T', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_a_layernorm` | rmsnorm | `[['B', 'T', 'c_q']]` | `['c_q']` | `[['B', 'T', 'c_q']]` |
| prefill | `model.layers.*.self_attn.q_b_proj` | matmul | `[['T', 'c_q'], ['c_q', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'c_q']` | `[['T', 'n_h*(d_nope+d_rope)']]` |
| prefill | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['T', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['T', 'c_kv+d_rope']]` |
| prefill | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', 'T', 'c_kv']]` | `['c_kv']` | `[['B', 'T', 'c_kv']]` |
| prefill | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['T', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['T', 'n_h*(d_nope+d_v)']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'd_nope+d_rope'], ['n_h', 'd_nope+d_rope', 'T']]` | `None` | `[['n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', 'T', 'T']]` | `None` | `[['B', 'n_h', 'T', 'T']]` |
| prefill | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'T', 'T'], ['n_h', 'T', 'd_nope']]` | `None` | `[['n_h', 'T', 'd_nope']]` |
| prefill | `model.layers.*.self_attn.o_proj` | matmul | `[['T', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.act_fn` | silu | `[['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['T', 'd_ff']]` |
| prefill | `model.layers.*.mlp` | elementwise_mul | `[['B', 'T', 'd_ff'], ['B', 'T', 'd_ff']]` | `None` | `[['B', 'T', 'd_ff']]` |
| prefill | `model.layers.*.mlp.down_proj` | matmul | `[['T', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp.gate` | matmul | `[['T', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['T', 'E']]` |
| prefill | `model.layers.*.mlp.gate` | sigmoid | `[['T', 'E']]` | `None` | `[['T', 'E']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['k*T', '2*d_moe']]` |
| prefill | `model.layers.*.mlp.experts.act_fn` | silu | `[['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_moe'], ['k*T', 'd_moe']]` | `None` | `[['k*T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.experts` | grouped_matmul | `[['k*T', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | elementwise_mul | `[['k*T', 'd_model'], ['k*T', 'B']]` | `None` | `[['k*T', 'd_model']]` |
| prefill | `model.layers.*.mlp.experts` | sum | `[['T', 'E', 'd_model']]` | `None` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp.shared_experts.gate_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.act_fn` | silu | `[['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.up_proj` | matmul | `[['T', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts` | elementwise_mul | `[['B', 'T', 'd_moe'], ['B', 'T', 'd_moe']]` | `None` | `[['B', 'T', 'd_moe']]` |
| prefill | `model.layers.*.mlp.shared_experts.down_proj` | matmul | `[['T', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['T', 'd_model']]` |
| prefill | `model.layers.*.mlp` | elementwise_add | `[['B', 'T', 'd_model'], ['B', 'T', 'd_model']]` | `None` | `[['B', 'T', 'd_model']]` |
| prefill | `model.norm` | rmsnorm | `[['B', 'T', 'd_model']]` | `['d_model']` | `[['B', 'T', 'd_model']]` |
| prefill | `lm_head` | matmul | `[['T', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['T', 'V']]` |
| decode | `model.embed_tokens` | embedding | `[['V', 'd_model'], ['B', '1']]` | `['V', 'd_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.input_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.self_attn.q_a_proj` | matmul | `[['B', 'd_model'], ['d_model', 'c_q']]` | `['c_q', 'd_model']` | `[['B', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_a_layernorm` | rmsnorm | `[['B', '1', 'c_q']]` | `['c_q']` | `[['B', '1', 'c_q']]` |
| decode | `model.layers.*.self_attn.q_b_proj` | matmul | `[['B', 'c_q'], ['c_q', 'n_h*(d_nope+d_rope)']]` | `['n_h*(d_nope+d_rope)', 'c_q']` | `[['B', 'n_h*(d_nope+d_rope)']]` |
| decode | `model.layers.*.self_attn.kv_a_proj_with_mqa` | matmul | `[['B', 'd_model'], ['d_model', 'c_kv+d_rope']]` | `['c_kv+d_rope', 'd_model']` | `[['B', 'c_kv+d_rope']]` |
| decode | `model.layers.*.self_attn.kv_a_layernorm` | rmsnorm | `[['B', '1', 'c_kv']]` | `['c_kv']` | `[['B', '1', 'c_kv']]` |
| decode | `model.layers.*.self_attn.kv_b_proj` | matmul | `[['B', 'c_kv'], ['c_kv', 'n_h*(d_nope+d_v)']]` | `['n_h*(d_nope+d_v)', 'c_kv']` | `[['B', 'n_h*(d_nope+d_v)']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'd_nope+d_rope'], ['n_h', 'd_nope+d_rope', 'T+1']]` | `None` | `[['n_h', 'B', 'T+1']]` |
| decode | `model.layers.*.self_attn` | softmax | `[['B', 'n_h', '1', 'T+1']]` | `None` | `[['B', 'n_h', '1', 'T+1']]` |
| decode | `model.layers.*.self_attn` | batched_matmul | `[['n_h', 'B', 'T+1'], ['n_h', 'T+1', 'd_nope']]` | `None` | `[['n_h', 'B', 'd_nope']]` |
| decode | `model.layers.*.self_attn.o_proj` | matmul | `[['B', 'n_h*d_v'], ['n_h*d_v', 'd_model']]` | `['d_model', 'n_h*d_v']` | `[['B', 'd_model']]` |
| decode | `model.layers.*` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.post_attention_layernorm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `model.layers.*.mlp.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp.act_fn` | silu | `[['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_ff']]` | `['d_ff', 'd_model']` | `[['B', 'd_ff']]` |
| decode | `model.layers.*.mlp` | elementwise_mul | `[['B', '1', 'd_ff'], ['B', '1', 'd_ff']]` | `None` | `[['B', '1', 'd_ff']]` |
| decode | `model.layers.*.mlp.down_proj` | matmul | `[['B', 'd_ff'], ['d_ff', 'd_model']]` | `['d_model', 'd_ff']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp.gate` | matmul | `[['B', 'd_model'], ['d_model', 'E']]` | `['E', 'd_model']` | `[['B', 'E']]` |
| decode | `model.layers.*.mlp.gate` | sigmoid | `[['B', 'E']]` | `None` | `[['B', 'E']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['E', 'd_model'], ['E', 'd_model', '2*d_moe'], ['E']]` | `['E', '2*d_moe', 'd_model']` | `[['E', '2*d_moe']]` |
| decode | `model.layers.*.mlp.experts.act_fn` | silu | `[['E', 'd_moe']]` | `None` | `[['E', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['E', 'd_moe'], ['E', 'd_moe']]` | `None` | `[['E', 'd_moe']]` |
| decode | `model.layers.*.mlp.experts` | grouped_matmul | `[['E', 'd_moe'], ['E', 'd_moe', 'd_model'], ['E']]` | `['E', 'd_model', 'd_moe']` | `[['E', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | elementwise_mul | `[['E', 'd_model'], ['E', 'B']]` | `None` | `[['E', 'd_model']]` |
| decode | `model.layers.*.mlp.experts` | sum | `[['B', 'E', 'd_model']]` | `None` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp.shared_experts.gate_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.act_fn` | silu | `[['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.up_proj` | matmul | `[['B', 'd_model'], ['d_model', 'd_moe']]` | `['d_moe', 'd_model']` | `[['B', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts` | elementwise_mul | `[['B', '1', 'd_moe'], ['B', '1', 'd_moe']]` | `None` | `[['B', '1', 'd_moe']]` |
| decode | `model.layers.*.mlp.shared_experts.down_proj` | matmul | `[['B', 'd_moe'], ['d_moe', 'd_model']]` | `['d_model', 'd_moe']` | `[['B', 'd_model']]` |
| decode | `model.layers.*.mlp` | elementwise_add | `[['B', '1', 'd_model'], ['B', '1', 'd_model']]` | `None` | `[['B', '1', 'd_model']]` |
| decode | `model.norm` | rmsnorm | `[['B', '1', 'd_model']]` | `['d_model']` | `[['B', '1', 'd_model']]` |
| decode | `lm_head` | matmul | `[['B', 'd_model'], ['d_model', 'V']]` | `['V', 'd_model']` | `[['B', 'V']]` |

## 전수 점검 — 이 모델이 쓰는 이름 전부

위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 말이 되는지** 보라.

### A. 붙은 이름 전부 (25종)

| 라벨 | 값 | 나타나는 모듈 | 축 수 |
|---|---|---|---|
| `B` |  | `model.layers.*.self_attn`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_a_layernorm`, `model.layers.*.self_attn.kv_a_layernorm` 외 29개 | 3302 |
| `T` |  | `model.layers.*.self_attn`, `model.layers.*.mlp.gate`, `model.layers.*.input_layernorm`, `model.layers.*.self_attn.q_a_layernorm` 외 29개 | 2042 |
| `d_model` | 7168 | `model.layers.*.mlp.experts`, `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm`, `model.layers.*.self_attn.q_a_proj` 외 20개 | 1274 |
| `n_h` | 128 | `model.layers.*.self_attn` | 1128 |
| `d_rope/2` |  | `model.layers.*.self_attn`, `model.rotary_emb` | 636 |
| `E` | 8 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.gate`, `model.layers.*.mlp.experts.act_fn` | 483 |
| `d_nope+d_rope` |  | `model.layers.*.self_attn` | 342 |
| `c_q` | 1536 | `model.layers.*.self_attn.q_a_layernorm`, `model.layers.*.self_attn.q_a_proj`, `model.layers.*.self_attn.q_b_proj` | 336 |
| `d_nope` | 128 | `model.layers.*.self_attn` | 282 |
| `c_kv` | 512 | `model.layers.*.self_attn.kv_a_layernorm`, `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 252 |
| `d_moe` | 2048 | `model.layers.*.mlp.experts`, `model.layers.*.mlp.shared_experts.gate_proj`, `model.layers.*.mlp.shared_experts.up_proj`, `model.layers.*.mlp.shared_experts.down_proj` 외 3개 | 252 |
| `d_head` | 64 | `model.layers.*.self_attn`, `model.rotary_emb` | 218 |
| `T+1` |  | `model.layers.*.self_attn` | 192 |
| `d_ff` | 18432 | `model.layers.*.mlp.gate_proj`, `model.layers.*.mlp.up_proj`, `model.layers.*.mlp.down_proj`, `model.layers.*.mlp` 외 1개 | 174 |
| `k*T` |  | `model.layers.*.mlp.experts`, `model.layers.*.mlp.experts.act_fn` | 165 |
| `n_grp` | 2 | `model.layers.*.mlp.gate` | 126 |
| `n_h*(d_nope+d_rope)` |  | `model.layers.*.self_attn.q_b_proj`, `model.layers.*.self_attn` | 108 |
| `c_kv+d_rope` |  | `model.layers.*.self_attn.kv_a_proj_with_mqa`, `model.layers.*.self_attn` | 108 |
| `n_h*(d_nope+d_v)` |  | `model.layers.*.self_attn.kv_b_proj`, `model.layers.*.self_attn` | 108 |
| `n_h*d_v` |  | `model.layers.*.self_attn.o_proj`, `model.layers.*.self_attn` | 108 |
| `d_nope+d_v` |  | `model.layers.*.self_attn` | 48 |
| `2*d_moe` |  | `model.layers.*.mlp.experts` | 42 |
| `E/n_grp` |  | `model.layers.*.mlp.gate` | 36 |
| `V` | 129280 | `lm_head`, `model.embed_tokens` | 20 |
| `k_grp` | 2 | `model.layers.*.mlp.gate` | 18 |

### B. 이름 없이 남은 정수 전부 (0쌍)

**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, 피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, 마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.

| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |
|---|---|---|---|

### C. 모듈이 내는 출력 shape 전부 (34개 모듈 / 230종)

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
  - `[[B, 1, d_model]]`
  - `[[B, T, d_ff]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[T, d_model]]`
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
- `model.layers.*.mlp.experts`
  - `[[B, E, d_model]]`
  - `[[B, d_model]]`
  - `[[E, 2*d_moe]]`
  - `[[E, B]]`
  - `[[E, d_model, 2*d_moe]]`
  - `[[E, d_model]]`
  - `[[E, d_moe, d_model]]`
  - `[[E, d_moe], [E, d_moe]]`
  - `[[E, d_moe]]`
  - `[[E], [E]]`
  - `[[E]]`
  - `[[T, E, d_model]]`
  - `[[T, d_model]]`
  - `[[k*T, 2*d_moe]]`
  - `[[k*T, B]]`
  - `[[k*T, d_model]]`
  - `[[k*T, d_moe], [k*T, d_moe]]`
  - `[[k*T, d_moe]]`
  - `[[k*T], [k*T]]`
  - `[[k*T]]`
- `model.layers.*.mlp.experts.act_fn`
  - `[[E, d_moe]]`
  - `[[k*T, d_moe]]`
- `model.layers.*.mlp.gate`
  - `[[B, 1]]`
  - `[[B, E], [B, E]]`
  - `[[B, E]]`
  - `[[B, d_model]]`
  - `[[B, n_grp, 1]]`
  - `[[B, n_grp, E/n_grp]]`
  - `[[B, n_grp, k_grp], [B, n_grp, k_grp]]`
  - `[[B, n_grp], [B, n_grp]]`
  - `[[B, n_grp]]`
  - `[[E, d_model]]`
  - `[[T, 1]]`
  - `[[T, E], [T, E]]`
  - `[[T, E]]`
  - `[[T, d_model]]`
  - `[[T, n_grp, 1]]`
  - `[[T, n_grp, E/n_grp]]`
  - `[[T, n_grp, k_grp], [T, n_grp, k_grp]]`
  - `[[T, n_grp], [T, n_grp]]`
  - `[[T, n_grp]]`
  - `[[d_model, E]]`
- `model.layers.*.mlp.gate_proj`
  - `[[B, 1, d_ff]]`
  - `[[B, T, d_ff]]`
  - `[[B, d_ff]]`
  - `[[B, d_model]]`
  - `[[T, d_ff]]`
  - `[[T, d_model]]`
  - `[[d_model, d_ff]]`
- `model.layers.*.mlp.shared_experts`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.mlp.shared_experts.act_fn`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
- `model.layers.*.mlp.shared_experts.down_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_moe, d_model]]`
- `model.layers.*.mlp.shared_experts.gate_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
- `model.layers.*.mlp.shared_experts.up_proj`
  - `[[B, 1, d_moe]]`
  - `[[B, T, d_moe]]`
  - `[[B, d_model]]`
  - `[[B, d_moe]]`
  - `[[T, d_model]]`
  - `[[T, d_moe]]`
  - `[[d_model, d_moe]]`
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
  - `[[B, 1, 1, d_head]]`
  - `[[B, 1, 1, d_rope/2]]`
  - `[[B, 1, T, d_head]]`
  - `[[B, 1, T, d_rope/2]]`
  - `[[B, 1, c_kv], [B, 1, d_head]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, 1, n_h*d_v]]`
  - `[[B, 1, n_h, d_nope+d_rope]]`
  - `[[B, 1, n_h, d_nope+d_v]]`
  - `[[B, 1, n_h, d_nope]]`
  - `[[B, T, c_kv], [B, T, d_head]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, T, n_h*d_v]]`
  - `[[B, T, n_h, d_nope+d_rope]]`
  - `[[B, T, n_h, d_nope+d_v]]`
  - `[[B, T, n_h, d_nope]]`
  - `[[B, n_h, 1, T+1]]`
  - `[[B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_nope+d_rope]]`
  - `[[B, n_h, 1, d_nope+d_v]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_head]]`
  - `[[B, n_h, 1, d_nope], [B, n_h, 1, d_nope]]`
  - `[[B, n_h, 1, d_nope]]`
  - `[[B, n_h, 1, d_rope/2]]`
  - `[[B, n_h, T+1, d_nope+d_rope]]`
  - `[[B, n_h, T+1, d_nope]]`
  - `[[B, n_h, T, T]]`
  - `[[B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_nope+d_rope]]`
  - `[[B, n_h, T, d_nope+d_v]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_head]]`
  - `[[B, n_h, T, d_nope], [B, n_h, T, d_nope]]`
  - `[[B, n_h, T, d_nope]]`
  - `[[B, n_h, T, d_rope/2]]`
  - `[[B, n_h, d_nope+d_rope, T+1]]`
  - `[[B, n_h, d_nope+d_rope, T]]`
  - `[[T, T]]`
  - `[[]]`
  - `[[n_h, B, T+1]]`
  - `[[n_h, B, d_nope+d_rope]]`
  - `[[n_h, B, d_nope]]`
  - `[[n_h, T+1, d_nope]]`
  - `[[n_h, T, T]]`
  - `[[n_h, T, d_nope+d_rope]]`
  - `[[n_h, T, d_nope]]`
  - `[[n_h, d_nope+d_rope, T+1]]`
  - `[[n_h, d_nope+d_rope, T]]`
- `model.layers.*.self_attn.kv_a_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, c_kv]]`
  - `[[B, T, 1]]`
  - `[[B, T, c_kv]]`
- `model.layers.*.self_attn.kv_a_proj_with_mqa`
  - `[[B, 1, c_kv+d_rope]]`
  - `[[B, T, c_kv+d_rope]]`
  - `[[B, c_kv+d_rope]]`
  - `[[B, d_model]]`
  - `[[T, c_kv+d_rope]]`
  - `[[T, d_model]]`
  - `[[d_model, c_kv+d_rope]]`
- `model.layers.*.self_attn.kv_b_proj`
  - `[[B, 1, n_h*(d_nope+d_v)]]`
  - `[[B, T, n_h*(d_nope+d_v)]]`
  - `[[B, c_kv]]`
  - `[[B, n_h*(d_nope+d_v)]]`
  - `[[T, c_kv]]`
  - `[[T, n_h*(d_nope+d_v)]]`
  - `[[c_kv, n_h*(d_nope+d_v)]]`
- `model.layers.*.self_attn.o_proj`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
  - `[[B, d_model]]`
  - `[[B, n_h*d_v]]`
  - `[[T, d_model]]`
  - `[[T, n_h*d_v]]`
  - `[[n_h*d_v, d_model]]`
- `model.layers.*.self_attn.q_a_layernorm`
  - `[[B, 1, 1]]`
  - `[[B, 1, c_q]]`
  - `[[B, T, 1]]`
  - `[[B, T, c_q]]`
- `model.layers.*.self_attn.q_a_proj`
  - `[[B, 1, c_q]]`
  - `[[B, T, c_q]]`
  - `[[B, c_q]]`
  - `[[B, d_model]]`
  - `[[T, c_q]]`
  - `[[T, d_model]]`
  - `[[d_model, c_q]]`
- `model.layers.*.self_attn.q_b_proj`
  - `[[B, 1, n_h*(d_nope+d_rope)]]`
  - `[[B, T, n_h*(d_nope+d_rope)]]`
  - `[[B, c_q]]`
  - `[[B, n_h*(d_nope+d_rope)]]`
  - `[[T, c_q]]`
  - `[[T, n_h*(d_nope+d_rope)]]`
  - `[[c_q, n_h*(d_nope+d_rope)]]`
- `model.layers.0`
  - `[[B, 1, d_model]]`
  - `[[B, T, d_model]]`
- `model.layers.1`
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
- `model.norm`
  - `[[B, 1, 1]]`
  - `[[B, 1, d_model]]`
  - `[[B, T, 1]]`
  - `[[B, T, d_model]]`
- `model.rotary_emb`
  - `[[B, 1, 1]]`
  - `[[B, 1, T]]`
  - `[[B, 1, d_head]]`
  - `[[B, 1, d_rope/2]]`
  - `[[B, T, d_head]]`
  - `[[B, T, d_rope/2]]`
  - `[[B, d_rope/2, 1]]`
  - `[[B, d_rope/2, T]]`
  - `[[B, d_rope/2]]`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
